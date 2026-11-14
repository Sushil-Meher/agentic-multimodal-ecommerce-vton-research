"""Embedding variants for the retrieval experiment (vectors stay outside the repo).

Run:  python src/retrieval/embed_variants.py
- whole_letterbox.npy : every photo padded to a square (aspect ratio kept) instead of squashed
- tiles.npy           : zoomed tiles (4x2 landscape / 2x4 portrait, 15% overlap) of every non-worn photo
"""
import csv
import json
import os
import time

from evaluate import DATASET, EMB_DIR, MODEL_NAME, load_index, load_labels

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

PHOTOS = os.environ.get("SAREE_PHOTO_DIR", r"D:\saree")
OVERLAP = 0.15
BATCH = 16


def letterbox(im):
    s = max(im.size)
    canvas = Image.new("RGB", (s, s), (128, 128, 128))
    canvas.paste(im, ((s - im.width) // 2, (s - im.height) // 2))
    return canvas


def tile_boxes(w, h):
    cols, rows = (4, 2) if w >= h else (2, 4)
    tw, th = w / cols, h / rows
    boxes = []
    for r in range(rows):
        for c in range(cols):
            boxes.append((max(0, int(c * tw - OVERLAP * tw)), max(0, int(r * th - OVERLAP * th)),
                          min(w, int((c + 1) * tw + OVERLAP * tw)), min(h, int((r + 1) * th + OVERLAP * th))))
    return boxes


def main():
    labels = load_labels()
    _, rows = load_index(labels)
    manifest = {r["image_id"]: r for r in csv.DictReader(open(os.path.join(DATASET, "metadata", "image_annotations.csv"), encoding="utf-8"))}
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float16).to("cuda").eval()

    def embed(imgs):
        out = []
        for i in range(0, len(imgs), BATCH):
            with torch.no_grad():
                pix = processor(images=imgs[i:i + BATCH], return_tensors="pt")["pixel_values"].to("cuda").half()
                e = model.get_image_features(pixel_values=pix)
                e = e.pooler_output if hasattr(e, "pooler_output") else e
                out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)

    whole, tiles, tile_index = [], [], []
    t0 = time.time()
    for n, r in enumerate(rows):
        im = ImageOps.exif_transpose(Image.open(os.path.join(PHOTOS, *manifest[r["image_id"]]["image_path"].split("/")))).convert("RGB")
        batch = [letterbox(im)]
        boxes = [] if r["eff_type"] == "worn" else tile_boxes(*im.size)
        batch += [letterbox(im.crop(b)) for b in boxes]
        vecs = embed(batch)
        whole.append(vecs[0])
        for b, v in zip(boxes, vecs[1:]):
            tiles.append(v)
            tile_index.append([r["image_id"], r["saree_id"], r["eff_type"], *b])
        if n % 20 == 0:
            print(f"{n + 1}/{len(rows)} photos, {len(tiles)} tiles, {time.time() - t0:.0f}s", flush=True)

    out_dir = os.path.join(EMB_DIR, "variants")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "whole_letterbox.npy"), np.stack(whole))
    np.save(os.path.join(out_dir, "tiles.npy"), np.stack(tiles))
    with open(os.path.join(out_dir, "tiles_index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "saree_id", "eff_type", "x0", "y0", "x1", "y1"])
        w.writerows(tile_index)
    json.dump({"model": MODEL_NAME, "photos": len(whole), "tiles": len(tiles), "overlap": OVERLAP,
               "letterbox_fill": [128, 128, 128], "seconds": round(time.time() - t0, 1)},
              open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    print("done", len(whole), "photos,", len(tiles), "tiles,", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
