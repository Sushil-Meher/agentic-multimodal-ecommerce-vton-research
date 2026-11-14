"""Local embedding service for photo and text search (SigLIP2, offline).

Run:  python src/retrieval/embed_server.py        (system Python with torch; not the project venv)
Listens on 127.0.0.1 only. Photos must sit inside the uploads folder (EMBED_UPLOAD_DIR).
  POST /image {"path": "<file inside uploads>"} -> {"vector": [...], "model": "..."}
  POST /text  {"text": "..."}                    -> {"vector": [...], "model": "..."}
EMBED_DEVICE (default cpu) picks where the model runs; the chat model usually needs the GPU.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("HF_HOME", r"D:\Models\huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/siglip2-base-patch16-384"
PORT = int(os.environ.get("EMBED_PORT", "8770"))
DEVICE = os.environ.get("EMBED_DEVICE", "cpu")
UPLOAD_DIR = os.path.realpath(os.environ.get("EMBED_UPLOAD_DIR", r"D:\saree_dataset\uploads"))
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, dtype=DTYPE).to(DEVICE).eval()


def _normalise(features):
    features = features.pooler_output if hasattr(features, "pooler_output") else features
    return torch.nn.functional.normalize(features.float(), dim=-1)[0].cpu().tolist()


def embed_image(path):
    real = os.path.realpath(path)
    if os.path.commonpath([real, UPLOAD_DIR]) != UPLOAD_DIR or not os.path.isfile(real):
        raise ValueError("image must be a file inside the uploads folder")
    im = ImageOps.exif_transpose(Image.open(real)).convert("RGB")
    with torch.no_grad():
        pix = processor(images=[im], return_tensors="pt")["pixel_values"].to(DEVICE).to(DTYPE)
        return _normalise(model.get_image_features(pixel_values=pix))


def embed_text(text):
    if not text or len(text) > 300:
        raise ValueError("text must be 1-300 characters")
    with torch.no_grad():
        inp = processor(text=[text.lower()], padding="max_length", max_length=64, return_tensors="pt").to(DEVICE)
        return _normalise(model.get_text_features(**inp))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if (self.headers.get("Host") or "").lower() not in (f"127.0.0.1:{PORT}", f"localhost:{PORT}"):
            return self._send(403, {"error": "bad host"})
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            return self._send(415, {"error": "json only"})
        try:
            payload = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 10_000)))
            if self.path == "/image":
                vector = embed_image(payload["path"])
            elif self.path == "/text":
                vector = embed_text(payload["text"])
            else:
                return self._send(404, {"error": "not found"})
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as e:
            return self._send(400, {"error": str(e)})
        self._send(200, {"vector": vector, "model": MODEL_NAME})


class Server(ThreadingHTTPServer):
    allow_reuse_address = False


if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    print(f"embedding service on http://127.0.0.1:{PORT} (device {DEVICE}, uploads {UPLOAD_DIR})", flush=True)
    Server(("127.0.0.1", PORT), Handler).serve_forever()
