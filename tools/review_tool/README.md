# Review tool

Local page for checking saree annotations by hand. Standard library only; serves on `127.0.0.1:8765`.

```
python tools/review_tool/server.py
```

Then open http://127.0.0.1:8765.

- Reads the manifest (`metadata/image_annotations.csv`) and photos from the dataset folders, which live outside this repo (`SAREE_DATASET_DIR`, `SAREE_PHOTO_DIR`; defaults `D:\saree_dataset`, `D:\saree`).
- Saves one record per saree to `<dataset>/annotations/human_verified/saree_level/`; earlier versions go to a `history/` folder next to it.
- The motif word list is `<dataset>/annotations/vocabulary.json` (created with a starter list on first run; new words added in the page are saved there).
- Optional model proposals: put `<dataset>/annotations/candidates/<saree_id>.json` (same `observable` layout as a saved record) and the form opens pre-filled and flagged.
- Images with an unresolved saree id are not listed.
