"""Customer photos coming in, saree photos going out."""
import hashlib
import os
import time
import uuid
from pathlib import Path

from src.retrieval.search import UPLOAD_DIR

# WhatsApp-sized copies of the catalog photos (made once by make_wa_images.py, outside the repo)
WA_IMAGE_DIR = os.environ.get("WA_IMAGE_DIR", r"D:\saree_dataset\wa_images")
MAX_IMAGE_BYTES = 10 * 1024 * 1024
UPLOAD_RETENTION_DAYS = int(os.environ.get("UPLOAD_RETENTION_DAYS", "30"))


def image_extension(data: bytes) -> str | None:
    """File type from the first bytes, never from a file name the customer controls."""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return None


def save_customer_image(data: bytes) -> str:
    """Store a customer photo in the private uploads folder; returns its (random) file name."""
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("photo is empty or larger than 10 MB")
    ext = image_extension(data)
    if ext is None:
        raise ValueError("only JPEG, PNG or WebP photos are accepted")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    name = f"{uuid.uuid4().hex}{ext}"
    (Path(UPLOAD_DIR) / name).write_bytes(data)
    return name


def delete_old_uploads(days: int = UPLOAD_RETENTION_DAYS) -> int:
    """Customer photos are personal data: remove those older than the retention period."""
    cutoff = time.time() - days * 86400
    removed = 0
    for path in Path(UPLOAD_DIR).glob("*"):
        if path.is_file() and path.stat().st_mtime < cutoff:
            path.unlink()
            removed += 1
    return removed


def wa_photo_name(image_url: str) -> str:
    """File name of the WhatsApp-sized copy of a catalog photo (same rule as make_wa_images.py)."""
    return hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:16] + ".jpg"


def wa_photo_path(image_url: str) -> str | None:
    path = Path(WA_IMAGE_DIR) / wa_photo_name(image_url)
    return str(path) if path.is_file() else None
