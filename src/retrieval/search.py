"""Photo search: embed the customer's photo (local embedding service), then rank the
sarees that pass the price/stock/attribute filters by visual similarity."""
import json
import os
import urllib.error
import urllib.request

from sqlalchemy import text
from sqlalchemy.engine import Connection

from src.search.products import search_products

MODEL_NAME = "google/siglip2-base-patch16-384"
SERVICE_URL = os.environ.get("EMBED_SERVICE_URL", "http://127.0.0.1:8770")
UPLOAD_DIR = os.path.realpath(os.environ.get("EMBED_UPLOAD_DIR", r"D:\saree_dataset\uploads"))
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route local calls through a proxy


class EmbeddingServiceError(RuntimeError):
    pass


def resolve_upload(path: str) -> str:
    """Only files inside the uploads folder may be embedded."""
    real = os.path.realpath(path if os.path.isabs(path) else os.path.join(UPLOAD_DIR, path))
    if os.path.commonpath([real, UPLOAD_DIR]) != UPLOAD_DIR or not os.path.isfile(real):
        raise ValueError("photo not found in the uploads folder")
    return real


def _call(endpoint: str, payload: dict) -> list[float]:
    req = urllib.request.Request(SERVICE_URL + endpoint, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with _opener.open(req, timeout=120) as resp:
            return json.loads(resp.read())["vector"]
    except urllib.error.HTTPError as e:
        raise ValueError(json.loads(e.read()).get("error", "embedding request rejected"))
    except (urllib.error.URLError, TimeoutError) as e:
        raise EmbeddingServiceError("the image search service is not running") from e


def embed_image(path: str) -> list[float]:
    return _call("/image", {"path": resolve_upload(path)})


def embed_text(description: str) -> list[float]:
    return _call("/text", {"text": description})


def vector_literal(vec) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vec) + "]"


def rank_by_vector(conn: Connection, vector, product_ids, *, limit: int = 5, model_name: str = MODEL_NAME):
    """[(product_id, similarity)] best first; a saree scores as its most similar photo."""
    if not product_ids:
        return []
    rows = conn.execute(text("""
        SELECT product_id, 1 - min(embedding <=> CAST(:v AS vector)) AS similarity
        FROM catalog.product_embeddings
        WHERE model_name = :m AND embedding_type = 'photo_whole' AND product_id = ANY(CAST(:ids AS uuid[]))
        GROUP BY product_id ORDER BY similarity DESC LIMIT :k"""),
        {"v": vector_literal(vector), "m": model_name, "ids": [str(i) for i in product_ids], "k": limit}).all()
    return [(r.product_id, float(r.similarity)) for r in rows]


def search_by_image(conn: Connection, image_path: str, *, limit: int = 5, **filters) -> list[dict]:
    """Filters are the same keyword arguments as search_products (price, blouse, design...)."""
    if limit <= 0:
        raise ValueError("limit must be > 0")
    vector = embed_image(image_path)  # first, so a bad photo is reported even when no saree passes the filters
    candidates = {r["id"]: dict(r) for r in search_products(conn, **filters)}
    if not candidates:
        return []
    ranked = rank_by_vector(conn, vector, list(candidates), limit=limit)
    return [{**candidates[pid], "similarity": round(sim, 4)} for pid, sim in ranked]
