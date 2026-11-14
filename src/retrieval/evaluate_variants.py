"""Compare embedding variants against the owner's labels (run embed_variants.py first).

Run:  python src/retrieval/evaluate_variants.py
Writes <dataset>/embeddings/<model>/variants/eval_variants.json.
"""
import csv
import json
import os
from collections import defaultdict

from evaluate import (EMB_DIR, build_queries, embed_texts, jaccard, load_index, load_labels, saree_scores)

import numpy as np

VAR_DIR = os.path.join(EMB_DIR, "variants")


def load_all():
    labels = load_labels()
    base, rows = load_index(labels)
    whole = np.load(os.path.join(VAR_DIR, "whole_letterbox.npy"))
    tiles = np.load(os.path.join(VAR_DIR, "tiles.npy"))
    tindex = list(csv.DictReader(open(os.path.join(VAR_DIR, "tiles_index.csv"), encoding="utf-8")))
    row_of = {r["image_id"]: i for i, r in enumerate(rows)}
    tile_row = np.array([row_of[t["image_id"]] for t in tindex])
    return labels, base, rows, whole, tiles, tile_row


def photo_scores(q, whole, tiles, tile_row, mode):
    """Score every photo for query vector q."""
    s = whole @ q
    if mode == "whole":
        return s
    tmax = np.full(len(whole), -9.0)
    np.maximum.at(tmax, tile_row, tiles @ q)
    if mode == "tiles":
        return np.where(tmax > -9, tmax, s)
    return np.where(tmax > -9, np.maximum(s, tmax), s)  # whole + tiles


def identity(rows, base, whole, tiles, tile_row):
    types = np.array([r["eff_type"] for r in rows])
    res = {}
    for qtype in ("worn", "flat_lay"):
        for name, mode in (("baseline (squashed)", None), ("letterbox whole", "whole"),
                           ("letterbox whole + tiles", "both"), ("tiles only", "tiles")):
            hits = {1: 0, 3: 0, 5: 0}
            qs = np.flatnonzero(types == qtype)
            for i in qs:
                mask = (types != qtype)
                if mode is None:
                    sc = base @ base[i]
                else:
                    sc = photo_scores(whole[i], whole, tiles, tile_row, mode)
                full = np.full(len(rows), -9.0)
                full[mask] = sc[mask]
                ranked = saree_scores(full, rows, mask)
                for k in hits:
                    hits[k] += rows[i]["saree_id"] in ranked[:k]
            res[f"query={qtype} | {name}"] = {f"hit@{k}": round(v / len(qs), 3) for k, v in hits.items()} | {"n": len(qs)}
    return res


def mean_vec(vs):
    v = np.mean(vs, axis=0)
    return v / np.linalg.norm(v)


def saree_variants(rows, base, whole, tiles, tile_row):
    sarees = sorted({r["saree_id"] for r in rows})
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[(r["saree_id"], r["eff_type"])].append(i)
    out = defaultdict(dict)
    for s in sarees:
        cat = by[(s, "flat_lay")] + by[(s, "pallu")]
        if cat:
            out["S0 baseline: flat_lay+pallu, squashed"][s] = mean_vec(base[cat])
            out["S1 letterbox: flat_lay+pallu"][s] = mean_vec(whole[cat])
        if by[(s, "flat_lay")]:
            out["S2 letterbox: flat_lay only"][s] = mean_vec(whole[by[(s, "flat_lay")]])
            tl = np.flatnonzero(np.isin(tile_row, by[(s, "flat_lay")]))
            if len(tl):
                out["S5 tiles: flat_lay tiles averaged"][s] = mean_vec(tiles[tl])
        if by[(s, "pallu")]:
            out["S3 letterbox: pallu photo only"][s] = mean_vec(whole[by[(s, "pallu")]])
        if by[(s, "border_closeup")]:
            out["S4 letterbox: border close-up only"][s] = mean_vec(whole[by[(s, "border_closeup")]])
    return out


def similar(variants, labels, k=3):
    attrs = {"body_colour": lambda a, b: a["body"] is not None and a["body"] == b["body"],
             "border_colour": lambda a, b: a["border"] is not None and a["border"] == b["border"],
             "blouse": lambda a, b: a["blouse"] == b["blouse"],
             "motif_jaccard": lambda a, b: jaccard(a["motifs"], b["motifs"])}
    res = {}
    for name, vecs in variants.items():
        ids = list(vecs)
        mat = np.stack([vecs[s] for s in ids])
        sims = mat @ mat.T
        entry = {"sarees": len(ids)}
        for an, fn in attrs.items():
            hits, chance = [], []
            for i, s in enumerate(ids):
                order = [j for j in np.argsort(-sims[i]) if j != i]
                vals = [float(fn(labels[s], labels[ids[j]])) for j in order]
                hits += vals[:k]
                chance += vals
            entry[an] = f"{np.mean(hits):.2f} vs chance {np.mean(chance):.2f} (lift {np.mean(hits) - np.mean(chance):+.2f})"
        res[name] = entry
    return res


def text(rows, base, whole, tiles, tile_row, labels):
    queries = build_queries(labels)
    te = embed_texts([q[1] for q in queries])
    types = np.array([r["eff_type"] for r in rows])
    nonworn = types != "worn"
    modes = {"T0 baseline (squashed, all photos)": ("base", np.ones(len(rows), bool)),
             "T1 letterbox whole, catalog photos": ("whole", nonworn),
             "T2 letterbox whole + tiles, catalog photos": ("both", nonworn),
             "T3 tiles only, catalog photos": ("tiles", nonworn)}
    n = len(labels)
    per_query, lift = defaultdict(dict), defaultdict(lambda: defaultdict(list))
    tops = defaultdict(dict)
    for mname, (mode, mask) in modes.items():
        for qi, (qname, _, truth) in enumerate(queries):
            sc = base @ te[qi] if mode == "base" else photo_scores(te[qi], whole, tiles, tile_row, mode)
            full = np.full(len(rows), -9.0)
            full[mask] = sc[mask]
            ranked = saree_scores(full, rows, mask)
            r = min(len(truth), 5)
            p = sum(s in truth for s in ranked[:r]) / r
            per_query[qname][mname] = round(p, 2)
            lift[mname][qname.split("=")[0]].append(p - len(truth) / n)
            tops[mname][qname] = ranked[:5]
    summary = {m: {g: round(float(np.mean(v)), 3) for g, v in groups.items()} for m, groups in lift.items()}
    animals = ["motif=elephant", "motif=lion", "motif=deer", "motif=duck"]
    overlap = {}
    for m in modes:
        pairs = [(a, b) for i, a in enumerate(animals) for b in animals[i + 1:]]
        overlap[m] = round(float(np.mean([len(set(tops[m][a]) & set(tops[m][b])) / 5 for a, b in pairs])), 2)
    return {"precision_at_R (R=min(#true,5))": per_query, "mean_lift_over_base_rate_by_group": summary,
            "animal_queries_top5_overlap (lower = better separation)": overlap,
            "queries": {q[0]: {"text": q[1], "true_sarees": len(q[2])} for q in queries}}


def main():
    labels, base, rows, whole, tiles, tile_row = load_all()
    result = {"identity": identity(rows, base, whole, tiles, tile_row)}
    result["similar_sarees"] = similar(saree_variants(rows, base, whole, tiles, tile_row), labels)
    result["text"] = text(rows, base, whole, tiles, tile_row, labels)
    json.dump(result, open(os.path.join(VAR_DIR, "eval_variants.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
