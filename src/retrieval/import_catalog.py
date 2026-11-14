"""Load reviewed saree records, photo records and embeddings into the local database.

Run from the repo root:
  python -m src.retrieval.import_catalog catalog    products (DRAFT), attributes, photos, embeddings; no prices
  python -m src.retrieval.import_catalog template   write a blank price sheet to fill in
  python -m src.retrieval.import_catalog prices     load the filled price sheet (add --activate to publish priced sarees)

The dataset stays outside the repo (SAREE_DATASET_DIR, default D:\\saree_dataset).
"""
import argparse
import ast
import csv
import glob
import json
import os
import re
from array import array

from sqlalchemy import text

from src.api.db import engine

DATASET = os.environ.get("SAREE_DATASET_DIR", r"D:\saree_dataset")
MODEL_NAME = "google/siglip2-base-patch16-384"
EMB_DIR = os.path.join(DATASET, "embeddings", MODEL_NAME.split("/")[1])
PRICE_SHEET = os.path.join(DATASET, "metadata", "price_stock.csv")
LABEL_REGIONS = {"body_pattern": "body", "border_class": "border", "pallu_class": "pallu"}
STOCK_STATUSES = {"AVAILABLE", "RESERVED", "PAYMENT_PENDING", "SOLD", "DAMAGED", "RETURNED"}


def read_npy(path):
    """Minimal reader for a 2-D little-endian float32 .npy file (no numpy needed)."""
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError("not an npy file")
        major = f.read(2)[0]
        header = ast.literal_eval(f.read(int.from_bytes(f.read(2 if major == 1 else 4), "little")).decode("latin1"))
        if header["descr"] != "<f4" or header["fortran_order"]:
            raise ValueError("unsupported npy layout")
        data = array("f")
        data.frombytes(f.read())
    rows, cols = header["shape"]
    return [data[i * cols:(i + 1) * cols].tolist() for i in range(rows)]


def record_to_rows(rec):
    """Turn one saved review record into product columns and attribute rows."""
    obs = rec["observable"]
    known = lambda v: v if v and v != "not_sure" else None
    labels = {f: (v["human_label"] or None) for f, v in rec["traditional_classification"].items()}
    blouse = obs.get("blouse")
    columns = {"primary_color": known(obs.get("body_colour")), "secondary_color": known(obs.get("border_colour")),
               "border_type": labels.get("border_class"), "blouse_included": blouse in ("plain", "blouse_work")}
    attrs = []
    for key, region in (("body_colour", "body"), ("border_colour", "border"), ("pallu_colour", "pallu")):
        if known(obs.get(key)):
            attrs.append(("colour", obs[key], region, None))
    attrs += [("colour_other", c, "body", None) for c in obs.get("body_other_colours", [])]
    for key, region in (("body_motifs", "body"), ("pallu_motifs", "pallu")):
        for group, values in obs.get(key, {}).items():
            attrs += [("motif", v, region, group) for v in values]
    for field, region in LABEL_REGIONS.items():
        if labels.get(field):
            attrs.append(("label", labels[field], region, None))
    if blouse:
        attrs.append(("blouse", blouse, "blouse", None))
    return columns, list(dict.fromkeys(attrs))


def vector_literal(vec):
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


def display_name(sku):
    """SAM-DATA-0007 -> BunkarHouse00007; other skus get a plain placeholder."""
    m = re.fullmatch(r"SAM-DATA-(\d+)", sku)
    return f"BunkarHouse{int(m.group(1)):05d}" if m else f"Sambalpuri saree {sku} (draft)"


def import_saree(conn, sku, rec, photos, vectors, model_name, meta):
    """Insert or refresh one saree. photos: [{image_id, image_path, image_type}]; vectors: one per photo."""
    columns, attrs = record_to_rows(rec)
    # a name is only replaced while it is still the old placeholder, so hand-edited names survive re-imports
    pid = conn.execute(text("""
        INSERT INTO catalog.products (sku, name, public_slug, primary_color, secondary_color, border_type, blouse_included)
        VALUES (:sku, :name, :slug, :primary_color, :secondary_color, :border_type, :blouse_included)
        ON CONFLICT (sku) DO UPDATE SET primary_color = EXCLUDED.primary_color, secondary_color = EXCLUDED.secondary_color,
            border_type = EXCLUDED.border_type, blouse_included = EXCLUDED.blouse_included,
            name = CASE WHEN catalog.products.name LIKE 'Sambalpuri saree % (draft)' THEN EXCLUDED.name ELSE catalog.products.name END,
            updated_at = now()
        RETURNING id"""), {"sku": sku, "name": display_name(sku), "slug": sku.lower(), **columns}).scalar_one()

    conn.execute(text("DELETE FROM catalog.product_attributes WHERE product_id = :p AND source = 'human_verified'"), {"p": pid})
    for attribute, value, region, detail in attrs:
        conn.execute(text("""INSERT INTO catalog.product_attributes (product_id, attribute, value, region, detail)
                             VALUES (:p, :a, :v, :r, :d)"""), {"p": pid, "a": attribute, "v": value, "r": region, "d": detail})

    conn.execute(text("DELETE FROM catalog.product_images WHERE product_id = :p AND storage_provider = 'local'"), {"p": pid})
    primary_done = False
    for ph in photos:
        primary = ph["image_type"] == "flat_lay" and not primary_done
        primary_done = primary_done or primary
        conn.execute(text("""INSERT INTO catalog.product_images (product_id, image_url, image_type, storage_provider, is_primary, is_private)
                             VALUES (:p, :url, :t, 'local', :prim, TRUE)"""),
                     {"p": pid, "url": "local:" + ph["image_path"], "t": ph["image_type"], "prim": primary})

    conn.execute(text("""DELETE FROM catalog.product_embeddings
                         WHERE product_id = :p AND model_name = :m AND embedding_type = 'photo_whole'"""), {"p": pid, "m": model_name})
    for ph, vec in zip(photos, vectors):
        conn.execute(text("""INSERT INTO catalog.product_embeddings (product_id, embedding_type, model_name, embedding, metadata)
                             VALUES (:p, 'photo_whole', :m, CAST(:v AS vector), CAST(:meta AS jsonb))"""),
                     {"p": pid, "m": model_name, "v": vector_literal(vec),
                      "meta": json.dumps({"image_id": ph["image_id"], "image_type": ph["image_type"], **meta})})
    return pid


def load_dataset():
    records = {}
    for f in sorted(glob.glob(os.path.join(DATASET, "annotations", "human_verified", "saree_level", "SAM-DATA-*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        records[r["saree_id"]] = r
    manifest = {r["image_id"]: r for r in csv.DictReader(open(os.path.join(DATASET, "metadata", "image_annotations.csv"), encoding="utf-8"))}
    vectors = read_npy(os.path.join(EMB_DIR, "image_embeddings.npy"))
    index = list(csv.DictReader(open(os.path.join(EMB_DIR, "index.csv"), encoding="utf-8")))
    meta_in = json.load(open(os.path.join(EMB_DIR, "meta.json"), encoding="utf-8"))
    meta = {"snapshot": meta_in.get("snapshot"), "preprocessing": meta_in.get("preprocessing")}
    return records, manifest, vectors, index, meta


def import_all():
    records, manifest, vectors, index, meta = load_dataset()
    per = {}
    for row in index:
        sid = row["saree_id"]
        fixes = records[sid]["image_type_corrections"]
        photo = {"image_id": row["image_id"], "image_path": manifest[row["image_id"]]["image_path"],
                 "image_type": fixes.get(row["image_id"], row["image_type"])}
        per.setdefault(sid, ([], []))
        per[sid][0].append(photo)
        per[sid][1].append(vectors[int(row["row"])])
    with engine.begin() as conn:
        for sid, rec in records.items():
            import_saree(conn, sid, rec, per[sid][0], per[sid][1], MODEL_NAME, meta)
    print(f"imported {len(records)} sarees, {sum(len(v[0]) for v in per.values())} photos (status stays DRAFT, no prices)")


def write_template():
    if os.path.exists(PRICE_SHEET):
        print("price sheet already exists, not overwritten:", PRICE_SHEET)
        return
    with engine.connect() as conn:
        skus = [r[0] for r in conn.execute(text("SELECT sku FROM catalog.products WHERE sku LIKE 'SAM-DATA-%' ORDER BY sku"))]
    with open(PRICE_SHEET, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sku", "price_inr", "stock_status", "notes"])
        w.writerows([[s, "", "AVAILABLE", ""] for s in skus])
    print("wrote", PRICE_SHEET, "- fill price_inr (and stock_status if not AVAILABLE)")


def parse_price_rows(path):
    """Rows with a price filled in; raises on bad values."""
    out = []
    for n, row in enumerate(csv.DictReader(open(path, encoding="utf-8")), start=2):
        raw = (row.get("price_inr") or "").strip().replace(",", "")
        if not raw:
            continue
        try:
            price = round(float(raw), 2)
        except ValueError:
            raise ValueError(f"line {n}: bad price {raw!r}")
        if price < 0:
            raise ValueError(f"line {n}: negative price")
        status = (row.get("stock_status") or "AVAILABLE").strip().upper()
        if status not in STOCK_STATUSES:
            raise ValueError(f"line {n}: unknown stock_status {status!r}")
        out.append((row["sku"].strip(), price, status))
    return out


def apply_price(conn, sku, price, status, activate=False):
    pid = conn.execute(text("SELECT id FROM catalog.products WHERE sku = :s"), {"s": sku}).scalar_one_or_none()
    if pid is None:
        raise ValueError(f"unknown sku {sku}")
    current = conn.execute(text("SELECT price FROM commerce.product_prices WHERE product_id = :p AND is_current"), {"p": pid}).scalar_one_or_none()
    if current is None or float(current) != price:
        conn.execute(text("UPDATE commerce.product_prices SET is_current = FALSE, effective_to = clock_timestamp() WHERE product_id = :p AND is_current"), {"p": pid})
        conn.execute(text("INSERT INTO commerce.product_prices (product_id, price, currency) VALUES (:p, :price, 'INR')"), {"p": pid, "price": price})
    conn.execute(text("""INSERT INTO commerce.inventory_items (product_id, inventory_code, status)
                         VALUES (:p, :code, CAST(:st AS commerce.inventory_status))
                         ON CONFLICT (inventory_code) DO UPDATE SET status = EXCLUDED.status, updated_at = now()"""),
                 {"p": pid, "code": f"{sku}-INV-01", "st": status})
    if activate:
        conn.execute(text("UPDATE catalog.products SET status = 'ACTIVE', updated_at = now() WHERE id = :p"), {"p": pid})


def load_prices(activate):
    rows = parse_price_rows(PRICE_SHEET)
    with engine.begin() as conn:
        for sku, price, status in rows:
            apply_price(conn, sku, price, status, activate)
    print(f"loaded {len(rows)} prices" + (" and activated those sarees" if activate else " (sarees stay DRAFT)"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["catalog", "template", "prices"])
    p.add_argument("--activate", action="store_true", help="with 'prices': publish the priced sarees")
    args = p.parse_args()
    {"catalog": import_all, "template": write_template, "prices": lambda: load_prices(args.activate)}[args.command]()


if __name__ == "__main__":
    main()
