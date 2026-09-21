# Metadata & Review-Based Hybrid Apparel Recommendation System

A cascaded LLM-based recommendation engine for men's apparel, implementing the methodology from:

> Roy, Kumar & Kumar — *"Metadata and Review-Based Hybrid Apparel Recommendation System Using Cascaded Large Language Models"* (IEEE Access, Vol. 12, 2024)

Built on the **2023 UCSD Amazon Review Dataset** (371,902 men's shirts/tops).

---

## 7-Phase Pipeline

| Phase | Description | Key Output |
|-------|-------------|------------|
| 1 | Content-based filtering (TF-IDF + cosine similarity) | Top-100 candidates per query |
| 2 | Co-purchase graph expansion (item-based CF from 65M user interactions) | 1.6M co-purchase edges |
| 3 | Review text concatenation (6.62 GB reviews → per-product full text) | `product_texts.parquet` |
| 4 | CasT5RoBERT embeddings (sentence-t5-xl + all-distilroberta-v1, 1536-d) | Embedding index (2.28 GB) |
| 5 | Embedding similarity search (query → cascade fingerprint → dot product) | Top-k semantic matches |
| 6 | Flair sentiment analysis (distilbert, 1.1M reviews scored) | Per-product sentiment scores |
| 7 | Weighted fusion + quality filters (content + embed + sentiment + collab) | Final ranked recommendations |

## Web Demo

FastAPI backend + single-page frontend with product cards, weight sliders, and "People Also Bought" strip.

```powershell
# After setup (see below): double-click start.bat, or:
cd "REAL SYSTEM"
python scripts/bootstrap.py   # only needed on a fresh machine (installs deps + models)
python -m uvicorn src.web_app:app --host 127.0.0.1 --port 8000
```

Open: http://127.0.0.1:8000 — the page loads instantly; the first search waits
~1–2 min while the models load in the background (`/health` says `{"ready":true}` when done).

## Project Structure

```
REAL SYSTEM/
├── src/                    # Core Python modules
│   ├── config.py           # Paths and constants
│   ├── data_loader.py      # Stream + filter 4GB metadata
│   ├── preprocessing.py    # Text cleaning, size/color extraction
│   ├── content_filter.py   # Phase 1: TF-IDF cosine similarity
│   ├── phase2_ibcf.py      # Phase 2: Co-purchase graph builder
│   ├── phase3_text.py      # Phase 3: Review text joiner
│   ├── phase5_similarity.py# Phase 5: Embedding index + query embedder
│   ├── phase6_sentiment.py # Phase 6: Flair sentiment scorer
│   ├── phase7_fusion.py    # Phase 7: Weighted fusion + quality filters
│   └── web_app.py          # FastAPI web server
├── web/                    # Frontend HTML
│   └── index.html          # Single-page recommendation UI
├── tests/                  # Unit tests (48+ tests)
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   ├── test_phase4.py
│   ├── test_phase5.py
│   ├── test_phase6.py
│   └── test_phase7.py
├── scripts/                # Evaluation & diagnostic scripts
│   ├── eval_harness.py     # Offline NDCG@10 evaluation
│   ├── bootstrap.py        # Auto-run by start.bat: deps + models on first run
│   ├── check_install.py    # Verifies deps / models / data (detailed report)
│   └── fetch_models.py     # One-time HuggingFace model download
├── kaggle/                 # Kaggle GPU run scripts (Phase 4 & 6)
├── data/                   # Data files (NOT in repo — see setup below)
│   └── processed/          # Parquet outputs from each phase
├── requirements.txt        # Runtime dependencies
├── start.bat               # THE ONLY FILE TO RUN: auto-setup (once) + server
├── setup.bat               # Optional: explicit setup + full install checklist
├── PROGRESS.md             # Detailed development log
└── README.md               # This file
```

## Setup for Team Members (clone → recommendations in ~15 min)

### 0. Install Python (one time, per machine)
Python **3.11+** from https://www.python.org/downloads/ — tick **"Add python.exe to PATH"** during install.

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/apparel-recommendation-system.git
cd apparel-recommendation-system
```

### 2. Download data files (shared separately)
The `data/processed/` folder (~2.8 GB) contains pre-computed outputs from all phases. Download from the shared drive link and place it inside the project (start.bat will remind you if it's missing):

| File | Size | Description |
|------|------|-------------|
| `embeddings_index.npy` | 2.28 GB | Phase 4: 371,902 × 1536 embedding matrix |
| `product_texts.parquet` | 303 MB | Phase 3: Full text per product |
| `men_shirts_metadata.parquet` | 139 MB | Phase 1: Filtered metadata |
| `reviews_individual.parquet` | 93 MB | Phase 6: Individual reviews |
| `product_details.parquet` | 26 MB | Web: Rating/price/image per product |
| `copurchase_edges.parquet` | 10 MB | Phase 2: Co-purchase graph |
| `product_sentiment.parquet` | 4.6 MB | Phase 6: Sentiment scores |
| `embeddings_asins.parquet` | 3.7 MB | Phase 4: ASIN index for embeddings |

### 3. Double-click `start.bat` — that's it
One file, one click. **First run only** it automatically installs the Python
dependencies and downloads the two AI models (~2.5 GB, needs internet, ~5–15 min);
every later run skips setup in under a second and just starts the server.
If a step fails it says exactly what to do — re-running resumes where it left off.

Then open **http://127.0.0.1:8000** and search — e.g. `black shirt`, or try the
typo test `blak shrit` (it auto-corrects). The first search waits ~1–2 min while
the models load in the background; restarting is much faster (file cache is warm).

Optional: `setup.bat` does the same automatic setup but ends with a detailed
✅/❌ checklist of everything (also runnable any time as
`python scripts/check_install.py`).

### 4. Run tests (optional)
```bash
python -m unittest discover -s tests -p "test_*.py" -v   # 202 tests, ~10 s
```
Stop the server first if RAM is tight (16 GB machines can't run both).

### Troubleshooting
| Symptom | Fix |
|---|---|
| `"ready": false` forever / engine load error in the console | Run `python scripts/check_install.py` — it names the missing piece |
| "Missing data files in data/processed/..." | Step 2 — the shared folder isn't in git |
| Model download fails / hangs | Needs internet the FIRST run only; re-run `start.bat` (downloads resume) |
| `A tensor for a type must be provided to map the type` on model load | transformers 6.x + TF quirk — stay on transformers 4.x from requirements, or `pip install tf-keras` |
| Server killed other apps / `ArrowMemoryError` | Close big apps — the engine needs ~6–8 GB RAM |

## Models Used (all pre-trained, no training)

1. **sentence-t5-xl** — Text → 768-d embeddings (Phase 4)
2. **all-distilroberta-v1** — Cascade second pass → 768-d (Phase 4)
3. **Flair sentiment** — Review sentiment scoring (Phase 6)

## Team

- 5 members (add your names here)

## Dataset

[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) — Clothing, Shoes & Jewelry category (0-core)

## Reference

Roy, Kumar & Kumar. "Metadata and Review-Based Hybrid Apparel Recommendation System Using Cascaded Large Language Models." *IEEE Access*, Vol. 12, 2024.
