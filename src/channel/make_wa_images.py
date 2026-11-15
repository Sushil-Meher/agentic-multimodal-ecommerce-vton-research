"""Make WhatsApp-sized copies of the catalog photos (run once; needs Pillow, so use system Python).

  python src/channel/make_wa_images.py
Reads the manifest and photos from the dataset folders and writes small JPEGs to WA_IMAGE_DIR,
named by the same rule the handler uses (src/channel/media.py: wa_photo_name). Outside the repo.
"""
import csv
import hashlib
import os

from PIL import Image, ImageOps

DATASET = os.environ.get("SAREE_DATASET_DIR", r"D:\saree_dataset")
PHOTOS = os.environ.get("SAREE_PHOTO_DIR", r"D:\saree")
OUT = os.environ.get("WA_IMAGE_DIR", os.path.join(DATASET, "wa_images"))
LONG_SIDE = 1280  # WhatsApp recompresses anyway; this keeps files well under its size limit


def main():
    os.makedirs(OUT, exist_ok=True)
    made = 0
    with open(os.path.join(DATASET, "metadata", "image_annotations.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["saree_id"].startswith("SAM-DATA-"):
                continue
            name = hashlib.sha1(("local:" + row["image_path"]).encode("utf-8")).hexdigest()[:16] + ".jpg"
            target = os.path.join(OUT, name)
            if os.path.exists(target):
                continue
            im = ImageOps.exif_transpose(Image.open(os.path.join(PHOTOS, *row["image_path"].split("/")))).convert("RGB")
            im.thumbnail((LONG_SIDE, LONG_SIDE))
            im.save(target, "JPEG", quality=85)
            made += 1
    print(f"made {made} new copies in {OUT}")


if __name__ == "__main__":
    main()
