# Retrieval

Photo search for the agent, plus the scripts that built and measured it. The dataset and vectors stay outside the repo.

## Running photo search
1. Start the embedding service in system Python (needs torch; the project venv does not have it):
   `python src/retrieval/embed_server.py`  (loopback only, CPU by default; `EMBED_DEVICE=cuda` to use the GPU)
2. Put a customer photo in the uploads folder (`EMBED_UPLOAD_DIR`, default `D:\saree_dataset\uploads`).
3. The agent tool `search_by_image(image_name=...)` ranks in-stock sarees by visual similarity; price and blouse filters apply first.

The tool only reads files inside the uploads folder. While the catalog is still DRAFT, set `AGENT_INCLUDE_DRAFT=1` to search it.

## Scripts
- `embed_photos.py` – embed every catalog photo (writes vectors to the dataset folder)
- `import_catalog.py` – load reviewed records, photos and vectors into the database; `template` / `prices` for the price sheet
- `evaluate.py`, `embed_variants.py`, `evaluate_variants.py` – baseline measurements against the owner's labels
