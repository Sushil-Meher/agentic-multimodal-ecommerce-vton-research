"""Baseline retrieval check against the owner's labels (private data stays outside the repo).

Run:  python src/retrieval/evaluate.py
1. photo -> saree identity: does a query photo find its own saree among other photos?
2. saree -> similar sarees: do the nearest sarees share the owner's labels more than chance?
3. text -> saree: do attribute descriptions (colour, border colour, motif) bring up sarees with that label?
Writes <dataset>/embeddings/<model>/eval_baseline.json.
"""
import csv
import glob
import json
import os
from collections import defaultdict

os.environ.setdefault("HF_HOME", r"D:\Models\huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/siglip2-base-patch16-384"
DATASET = os.environ.get("SAREE_DATASET_DIR", r"D:\saree_dataset")
EMB_DIR = os.path.join(DATASET, "embeddings", MODEL_NAME.split("/")[1])
CATALOG_TYPES = {"flat_lay", "pallu"}

# owner colour words -> base colour (assumption made for this check; None = left out)
BODY_BASE = {"pink": "pink", "light pink": "pink", "yellow": "yellow", "mustard yellow": "yellow", "light green": "green",
             "parrot green": "green", "mehendi green": "green", "deep green": "green", "cement": "grey", "blue": "blue",
             "light orange": "orange", "orange": "orange", "maroon": "maroon", "peach": "peach", "beige": "beige",
             "light chandan": "beige", "multicolour": "multicolour", "matha": None}
BORDER_BASE = {"coffee": "brown", "golden coffee": "brown", "brown": "brown", "red": "red", "black": "black",
               "maroon": "maroon", "green": "green"}


def load_labels():
    labels = {}
    for f in glob.glob(os.path.join(DATASET, "annotations", "human_verified", "saree_level", "SAM-DATA-*.json")):
        r = json.load(open(f, encoding="utf-8"))
        o = r["observable"]
        motifs = {m.lower() for g in ("body_motifs", "pallu_motifs") for ms in o[g].values() for m in ms}
        labels[r["saree_id"]] = {"body": BODY_BASE.get(o["body_colour"]), "border": BORDER_BASE.get(o["border_colour"]),
                                 "blouse": o.get("blouse"), "motifs": motifs, "fixes": r["image_type_corrections"]}
    return labels


def load_index(labels):
    emb = np.load(os.path.join(EMB_DIR, "image_embeddings.npy"))
    rows = list(csv.DictReader(open(os.path.join(EMB_DIR, "index.csv"), encoding="utf-8")))
    for r in rows:
        r["eff_type"] = labels[r["saree_id"]]["fixes"].get(r["image_id"], r["image_type"])
    return emb, rows


def saree_scores(sim_row, rows, mask):
    best = {}
    for j in np.flatnonzero(mask):
        s = rows[j]["saree_id"]
        best[s] = max(best.get(s, -9), sim_row[j])
    return sorted(best, key=best.get, reverse=True)


def identity_eval(emb, rows):
    sims = emb @ emb.T
    types = np.array([r["eff_type"] for r in rows])
    out = defaultdict(lambda: {"n": 0, "hit1": 0, "hit3": 0, "hit5": 0})
    for qtype in ("worn", "flat_lay"):
        for mode in ("any_other_photo", "different_photo_type"):
            for i in np.flatnonzero(types == qtype):
                mask = np.ones(len(rows), bool)
                mask[i] = False
                if mode == "different_photo_type":
                    mask &= types != qtype
                ranked = saree_scores(sims[i], rows, mask)
                truth = rows[i]["saree_id"]
                d = out[f"query={qtype} | gallery={mode}"]
                d["n"] += 1
                for k in (1, 3, 5):
                    d[f"hit{k}"] += truth in ranked[:k]
    return {k: {**v, **{h: round(v[h] / v["n"], 3) for h in ("hit1", "hit3", "hit5")}} for k, v in out.items()}


def saree_vectors(emb, rows):
    vecs = {}
    for s in sorted({r["saree_id"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["saree_id"] == s and r["eff_type"] in CATALOG_TYPES]
        if idx:
            v = emb[idx].mean(0)
            vecs[s] = v / np.linalg.norm(v)
    return vecs


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def similar_eval(vecs, labels, k=3):
    ids = list(vecs)
    mat = np.stack([vecs[s] for s in ids])
    sims = mat @ mat.T
    res = {"sarees_compared": len(ids), "k": k}
    for name, fn in (("body_colour_match", lambda a, b: a["body"] is not None and a["body"] == b["body"]),
                     ("border_colour_match", lambda a, b: a["border"] is not None and a["border"] == b["border"]),
                     ("blouse_match", lambda a, b: a["blouse"] == b["blouse"]),
                     ("motif_jaccard", lambda a, b: jaccard(a["motifs"], b["motifs"]))):
        hits, chance = [], []
        for i, s in enumerate(ids):
            order = [j for j in np.argsort(-sims[i]) if j != i]
            hits += [float(fn(labels[s], labels[ids[j]])) for j in order[:k]]
            chance += [float(fn(labels[s], labels[ids[j]])) for j in order]
        res[name] = {"nearest": round(float(np.mean(hits)), 3), "chance": round(float(np.mean(chance)), 3)}
    return res


def build_queries(labels):
    queries = []
    for field, label_key, template in (("body", "body", "a photo of a {} saree"), ("border", "border", "a photo of a saree with a {} border")):
        for val in sorted({v[label_key] for v in labels.values() if v[label_key]}):
            truth = {s for s, v in labels.items() if v[label_key] == val}
            if len(truth) >= 2:
                queries.append((f"{field}={val}", template.format(val), truth))
    for m in ("peacock", "elephant", "lion", "deer", "duck", "flower", "tree"):
        truth = {s for s, v in labels.items() if m in v["motifs"]}
        if len(truth) >= 3:
            queries.append((f"motif={m}", f"a photo of a saree with {m} motifs", truth))
    return queries


def embed_texts(texts):
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float16).to("cuda").eval()
    with torch.no_grad():
        inp = processor(text=texts, padding="max_length", max_length=64, return_tensors="pt").to("cuda")
        te = model.get_text_features(**inp)
        te = te.pooler_output if hasattr(te, "pooler_output") else te
        return torch.nn.functional.normalize(te.float(), dim=-1).cpu().numpy()


def text_eval(emb, rows, labels):
    queries = build_queries(labels)
    te = embed_texts([q[1] for q in queries])
    sims = te @ emb.T
    mask = np.ones(len(rows), bool)
    out = []
    for qi, (name, text, truth) in enumerate(queries):
        ranked = saree_scores(sims[qi], rows, mask)
        top5 = ranked[:5]
        out.append({"query": name, "text": text, "sarees_with_label": len(truth), "base_rate": round(len(truth) / len(labels), 3),
                    "precision_at_5": round(sum(s in truth for s in top5) / 5, 2), "top1_correct": ranked[0] in truth,
                    "top5": [s[-4:] for s in top5]})
    return out


def main():
    labels = load_labels()
    emb, rows = load_index(labels)
    result = {"model": MODEL_NAME, "photos": len(rows), "sarees": len(labels),
              "identity": identity_eval(emb, rows), "similar_sarees": similar_eval(saree_vectors(emb, rows), labels),
              "text": text_eval(emb, rows, labels),
              "assumptions": {"body_colour_map": BODY_BASE, "border_colour_map": BORDER_BASE, "catalog_photo_types": sorted(CATALOG_TYPES)}}
    json.dump(result, open(os.path.join(EMB_DIR, "eval_baseline.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k != "assumptions"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
