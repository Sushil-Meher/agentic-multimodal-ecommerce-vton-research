"""Embed every catalog photo with SigLIP2 and save the vectors outside the repo.

Run:  python src/retrieval/embed_photos.py
Reads the manifest and photos from the dataset folders, writes
<dataset>/embeddings/<model>/{image_embeddings.npy, index.csv, meta.json}.
The model is loaded offline from the local Hugging Face cache.
"""
import csv
import json
import os
import time
from datetime import datetime, timezone

os.environ.setdefault("HF_HOME", r"D:\Models\huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/siglip2-base-patch16-384"
DATASET = os.environ.get("SAREE_DATASET_DIR", r"D:\saree_dataset")
PHOTOS = os.environ.get("SAREE_PHOTO_DIR", r"D:\saree")
BATCH = 8


def load_manifest():
    with open(os.path.join(DATASET, "metadata", "image_annotations.csv"), encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["saree_id"].startswith("SAM-DATA-")]


def main():
    rows = load_manifest()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float16).to("cuda").eval()
    revision = os.path.basename(os.path.dirname(model.config._name_or_path)) if os.path.isdir(model.config._name_or_path) else None

    vectors = []
    t0 = time.time()
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        imgs = [ImageOps.exif_transpose(Image.open(os.path.join(PHOTOS, *r["image_path"].split("/")))).convert("RGB") for r in batch]
        with torch.no_grad():
            pix = processor(images=imgs, return_tensors="pt")["pixel_values"].to("cuda").half()
            emb = model.get_image_features(pixel_values=pix)
            emb = emb.pooler_output if hasattr(emb, "pooler_output") else emb
            vectors.append(torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy())
        print(f"{min(i + BATCH, len(rows))}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)

    out_dir = os.path.join(DATASET, "embeddings", MODEL_NAME.split("/")[1])
    os.makedirs(out_dir, exist_ok=True)
    arr = np.concatenate(vectors)
    np.save(os.path.join(out_dir, "image_embeddings.npy"), arr)
    with open(os.path.join(out_dir, "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "image_id", "saree_id", "image_type"])
        for n, r in enumerate(rows):
            w.writerow([n, r["image_id"], r["saree_id"], r["image_type"]])
    meta = {"model": MODEL_NAME, "snapshot": revision, "dim": int(arr.shape[1]), "count": int(arr.shape[0]),
            "normalised": True, "dtype": "float16 inference, float32 stored",
            "preprocessing": "model processor default (resize to 384x384, aspect ratio not kept)",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8"), indent=2)
    print("saved", out_dir, meta)


if __name__ == "__main__":
    main()
