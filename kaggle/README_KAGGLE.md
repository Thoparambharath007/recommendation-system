# Phase 4 — Kaggle Run Instructions

This folder contains everything needed to generate CasT5RoBERT embeddings on a Kaggle GPU.

## What's in this folder

| File | Purpose |
|---|---|
| `phase4_embeddings.py` | The main script (loads texts → runs cascade → saves embeddings) |
| `phase4_kaggle.ipynb` | Ready-made Kaggle notebook (import + run) |
| `input/product_texts.parquet` | The Phase 3 output — 371,902 products with full_text (289 MB) |
| `input/product_texts_sample.parquet` | 50-row sample for a quick 2-minute test |
| `requirements.txt` | Python dependencies |
| `output/` | Where embeddings_*.parquet files will be saved |

## IMPORTANT: this is inference, NOT training

The models (sentence-t5-xl, all-distilroberta-v1) are pre-trained by HuggingFace/Google.
We only RUN them on our data to produce fingerprints. No training, no labels, no backprop.

## Step-by-step (Kaggle, ~5 min setup + 3-4 h run)

1. Go to https://www.kaggle.com/code — click "New Notebook"
2. In the right panel: Accelerator = **GPU T4 x2**, Internet = **ON**
3. Files panel → Upload → upload:
   - `phase4_embeddings.py`
   - `input/product_texts.parquet`
   - (optional) `input/product_texts_sample.parquet`
4. Open the notebook editor, delete any default code, then paste:

   Cell 1 (install — ~1 min, only first run):
   ```
   !pip install -q sentence-transformers
   ```

   Cell 2 (quick test, 50 rows — ~1-2 min, verifies everything works):
   ```
   !python phase4_embeddings.py --input input/product_texts_sample.parquet --limit 50 --out output
   ```
   Expected: "50 products to embed" then "DONE: 50 embeddings" and 1 file in output/

   Cell 3 (FULL run — ~3-4 hours):
   ```
   !python phase4_embeddings.py --input input/product_texts.parquet --out output
   ```

5. Watch the progress: it prints a line every chunk (10,000 products) with ETA.
6. When done: Files panel → download `output/embeddings_*.parquet` (all chunks, 38 files)
7. Put them into `scratch/kaggle/output/` on your PC.

## If the session dies mid-run

- Kaggle kills idle sessions (~12 h limit, ~30 h/week GPU quota)
- DON'T PANIC: completed chunks are SKIPPED automatically on re-run
- Just re-run Cell 3 — it resumes from the last saved chunk (loses at most 10K products)

## Expected output

- 38 files: `embeddings_00000.parquet` ... `embeddings_00037.parquet`
- Each: columns `parent_asin` + `embedding` (list of 1792 floats = 1024 T5 + 768 RoBERTa)
- 371,902 rows total, ~2.7 GB on disk
- Phase 5 will merge them and compare against the user query fingerprint

## Local verification after download (on your PC)

```
cd C:\Users\thopa\OneDrive\Desktop\project\scratch
python -c "import pandas as pd, glob; files = glob.glob('kaggle/output/embeddings_*.parquet'); df = pd.concat([pd.read_parquet(f) for f in files]); print(len(df), 'rows')"
```

Expected: 371902 rows.
