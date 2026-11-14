"""Local review tool for saree annotations.

Run:  python tools/review_tool/server.py
Open: http://127.0.0.1:8765

Serves on loopback only. Reads photos and the manifest from the dataset folders
(outside this repo) and writes reviewed records to <dataset>/annotations/.
Standard library only (Pillow is used for thumbnails if it is installed).
"""
import csv
import io
import json
import os
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).parent
PORT = int(os.environ.get("REVIEW_TOOL_PORT", "8765"))

COLOURS = ["white", "off-white", "cream", "beige", "yellow", "mustard", "gold", "orange", "red", "maroon",
           "pink", "magenta", "purple", "violet", "blue", "navy", "teal", "green", "olive", "brown",
           "grey", "black", "silver", "multicolour", "deep green"]
IMAGE_TYPES = ["worn", "flat_lay", "closeup", "pallu", "border_closeup", "blouse_work", "other"]
MOTIF_GROUPS = ["animals_birds", "flowers_plants", "geometric", "other"]
BLOUSE_OPTIONS = ["plain", "blouse_work", "no_blouse"]
CLASS_FIELDS = ["body_pattern", "border_class", "pallu_class"]
SAREE_ID = re.compile(r"^SAM-DATA-\d{4}$")

# starter words only: plain visual names, not Sambalpuri terms. The owner edits the real list.
DEFAULT_VOCAB = {
    "animals_birds": ["elephant", "lion", "peacock", "duck", "deer", "bird"],
    "flowers_plants": ["lotus", "flower", "leaf", "vine", "tree"],
    "geometric": ["star", "diamond", "square or check", "stripe", "zigzag", "circle or medallion", "wave"],
    "other": ["temple", "pot"],
}


def _clean_list(values, allowed=None, limit=40):
    if not isinstance(values, list) or len(values) > limit:
        raise ValueError("expected a short list")
    out = []
    for v in values:
        if not isinstance(v, str) or not v.strip() or len(v) > 60:
            raise ValueError("invalid list item")
        v = v.strip()
        if allowed is not None and v not in allowed:
            raise ValueError(f"value not allowed: {v}")
        if v not in out:
            out.append(v)
    return out


def _colour(value):
    """Free text (the owner's own shade names); 'not sure' is stored as not_sure."""
    value = _text(value, 40).lower()
    if not value:
        return None
    return "not_sure" if value in ("not sure", "not_sure") else value


def _blouse(value):
    if value in ("", None):
        return None
    if value not in BLOUSE_OPTIONS:
        raise ValueError(f"unknown blouse option: {value}")
    return value


def _text(value, limit):
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError("text too long")
    return value.strip()


def _motifs(block):
    block = block or {}
    return {g: _clean_list(block.get(g, [])) for g in MOTIF_GROUPS}


def clean_record(saree_id, payload, image_ids):
    """Validate the form payload and return the record to store."""
    status = payload.get("status")
    if status not in ("verified", "uncertain"):
        raise ValueError("status must be verified or uncertain")
    obs = payload.get("observable") or {}
    corrections = {}
    for image_id, image_type in (payload.get("image_type_corrections") or {}).items():
        if image_id not in image_ids or image_type not in IMAGE_TYPES:
            raise ValueError(f"bad image type correction: {image_id}")
        corrections[image_id] = image_type
    classification = {}
    for field in CLASS_FIELDS:
        label = _text((payload.get("classification") or {}).get(field), 80)
        classification[field] = {"human_label": label or None,
                                 "status": "human_labelled" if label else "not_assessed"}
    return {
        "saree_id": saree_id,
        "schema_version": 1,
        "verification": {"status": status, "reviewed_by": "owner",
                         "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "conventions": {"body_colour": "dominant colour of the saree body", "border_colour": "colour of the border bands"},
        "observable": {
            "body_colour": _colour(obs.get("body_colour")),
            "border_colour": _colour(obs.get("border_colour")),
            "pallu_colour": _colour(obs.get("pallu_colour")),
            "blouse": _blouse(obs.get("blouse")),
            "body_other_colours": [c.lower() for c in _clean_list(obs.get("body_other_colours", []))],
            "body_motifs": _motifs(obs.get("body_motifs")),
            "pallu_motifs": _motifs(obs.get("pallu_motifs")),
        },
        "image_type_corrections": corrections,
        "traditional_classification": classification,
        "notes": _text(payload.get("notes"), 2000),
    }


class Store:
    def __init__(self, dataset, photos):
        self.dataset = Path(dataset)
        self.photos = Path(photos)
        self.out_dir = self.dataset / "annotations" / "human_verified" / "saree_level"
        self.history_dir = self.out_dir / "history"
        self.cand_dir = self.dataset / "annotations" / "candidates"
        self.vocab_path = self.dataset / "annotations" / "vocabulary.json"
        self.images = {}
        self.by_saree = {}
        with open(self.dataset / "metadata" / "image_annotations.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not SAREE_ID.match(row["saree_id"]):
                    continue  # unresolved images are not reviewable yet
                self.images[row["image_id"]] = row
                self.by_saree.setdefault(row["saree_id"], []).append(row)

    def saree_ids(self):
        return sorted(self.by_saree)

    def _read(self, path):
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def record(self, saree_id):
        return self._read(self.out_dir / f"{saree_id}.json")

    def candidate(self, saree_id):
        return self._read(self.cand_dir / f"{saree_id}.json")

    def label_options(self):
        """Labels already used on saved sarees, per field, so they can be picked again."""
        found = {f: set() for f in CLASS_FIELDS}
        for sid in self.saree_ids():
            rec = self.record(sid) or {}
            for field, item in rec.get("traditional_classification", {}).items():
                if field in found and item.get("human_label"):
                    found[field].add(item["human_label"])
        return {f: sorted(v) for f, v in found.items()}

    def summary(self):
        out = []
        for sid in self.saree_ids():
            rec = self.record(sid)
            out.append({"saree_id": sid, "images": len(self.by_saree[sid]),
                        "status": rec["verification"]["status"] if rec else None})
        return out

    def image_list(self, saree_id):
        rec = self.record(saree_id) or {}
        fixed = rec.get("image_type_corrections", {})
        order = {"flat_lay": 0, "pallu": 1, "closeup": 2, "border_closeup": 3, "blouse_work": 4, "worn": 5, "other": 6}
        rows = sorted(self.by_saree[saree_id], key=lambda r: (order.get(r["image_type"], 9), r["image_id"]))
        return [{"image_id": r["image_id"], "audit_type": r["image_type"],
                 "image_type": fixed.get(r["image_id"], r["image_type"]),
                 "width": r["width"], "height": r["height"]} for r in rows]

    def vocab(self):
        v = self._read(self.vocab_path)
        if v is None:
            v = {"_note": "starter list; edit freely. Plain visual words, not Sambalpuri terms.", **DEFAULT_VOCAB}
        if "colours" not in v:
            v["colours"] = list(COLOURS)
            self._write(self.vocab_path, v)
        return v

    def add_term(self, group, term):
        if group not in MOTIF_GROUPS + ["colours"]:
            raise ValueError("unknown vocabulary group")
        term = _text(term, 60)
        if group == "colours":
            term = term.lower()
        if not term:
            raise ValueError("empty term")
        v = self.vocab()
        if term not in v[group]:
            v[group].append(term)
            self._write(self.vocab_path, v)
        return v

    def save(self, saree_id, payload):
        if saree_id not in self.by_saree:
            raise ValueError("unknown saree")
        rec = clean_record(saree_id, payload, {r["image_id"] for r in self.by_saree[saree_id]})
        path = self.out_dir / f"{saree_id}.json"
        if path.exists():  # keep the previous version
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.history_dir.mkdir(parents=True, exist_ok=True)
            (self.history_dir / f"{saree_id}_{stamp}.json").write_bytes(path.read_bytes())
        self._write(path, rec)
        obs = rec["observable"]
        for colour in [obs["body_colour"], obs["border_colour"], obs["pallu_colour"], *obs["body_other_colours"]]:
            if colour and colour != "not_sure":
                self.add_term("colours", colour)  # remember new shade names
        return rec

    def _write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def image_bytes(self, image_id, width):
        row = self.images.get(image_id)
        if row is None:
            return None
        path = self.photos.joinpath(*row["image_path"].split("/"))
        try:
            from PIL import Image, ImageOps
        except ImportError:
            return path.read_bytes()
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        im.thumbnail((width, width))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    store = None
    cache = {}

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _host_ok(self):
        host = (self.headers.get("Host") or "").lower()
        return host in (f"127.0.0.1:{PORT}", f"localhost:{PORT}")

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, {"error": "bad host"})
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        try:
            if not parts:
                return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if parts == ["api", "sarees"]:
                return self._send(200, self.store.summary())
            if parts[:2] == ["api", "saree"] and len(parts) == 3 and parts[2] in self.store.by_saree:
                sid = parts[2]
                return self._send(200, {"saree_id": sid, "images": self.store.image_list(sid),
                                        "record": self.store.record(sid), "candidate": self.store.candidate(sid),
                                        "vocabulary": self.store.vocab(),
                                        "image_types": IMAGE_TYPES, "classification_fields": CLASS_FIELDS,
                                        "label_options": self.store.label_options()})
            if parts[0] == "img" and len(parts) == 2:
                width = min(int(parse_qs(url.query).get("w", ["480"])[0]), 2400)
                key = (parts[1], width)
                if key not in self.cache:
                    data = self.store.image_bytes(parts[1], width)
                    if data is None:
                        return self._send(404, {"error": "not found"})
                    self.cache[key] = data
                return self._send(200, self.cache[key], "image/jpeg")
        except (ValueError, OSError) as e:
            return self._send(500, {"error": str(e)})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._host_ok():
            return self._send(403, {"error": "bad host"})
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            return self._send(415, {"error": "json only"})
        length = int(self.headers.get("Content-Length") or 0)
        if length > 100_000:
            return self._send(413, {"error": "too large"})
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            if parts[:2] == ["api", "saree"] and len(parts) == 3:
                return self._send(200, self.store.save(parts[2], payload))
            if parts == ["api", "vocab"]:
                return self._send(200, self.store.add_term(payload.get("group"), payload.get("term")))
        except (ValueError, json.JSONDecodeError, AttributeError) as e:
            return self._send(400, {"error": str(e)})
        self._send(404, {"error": "not found"})


class Server(ThreadingHTTPServer):
    allow_reuse_address = False  # on Windows this would let a second copy share the port silently


def main():
    dataset = os.environ.get("SAREE_DATASET_DIR", r"D:\saree_dataset")
    photos = os.environ.get("SAREE_PHOTO_DIR", r"D:\saree")
    Handler.store = Store(dataset, photos)
    server = Server(("127.0.0.1", PORT), Handler)
    print(f"review tool on http://127.0.0.1:{PORT}  (dataset: {dataset})")
    server.serve_forever()


if __name__ == "__main__":
    main()
