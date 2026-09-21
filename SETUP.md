# SETUP & SYSTEM GUIDE — Hybrid Apparel Recommendation System

> **For AI assistants (Antigravity/Claude Code):** Read this entire file to understand the project architecture, how every component works, and how to set it up. This is the single source of truth for this codebase.

---

## What This Project Is

A **7-phase hybrid recommendation engine** for men's apparel (shirts/tops), built on the research paper:

> Roy, Kumar & Kumar — *"Metadata and Review-Based Hybrid Apparel Recommendation System Using Cascaded Large Language Models"* (IEEE Access, Vol. 12, 2024)

It takes a natural-language query like **"white cotton shirt for office"** and returns the best matching products from a catalog of **371,902 men's shirts**, ranked by combining 4 signals:
- **Content similarity** (keyword matching via TF-IDF)
- **Embedding similarity** (semantic meaning via cascaded T5 + RoBERTa models)
- **Sentiment** (how positive the product reviews are)
- **Collaborative filtering** (what other shoppers bought together)

**Dataset:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) — Clothing, Shoes & Jewelry category.

---

## Project Structure

```
REAL SYSTEM/
│
├── src/                         # All Python source code
│   ├── config.py                # Paths and constants (BASE_DIR = parent of src/)
│   ├── data_loader.py           # Phase 1: Streams 4GB metadata, filters men's shirts
│   ├── preprocessing.py         # Text cleaning, size/color extraction from titles
│   ├── content_filter.py        # Phase 1: TF-IDF vectorizer + cosine similarity
│   ├── phase2_ibcf.py           # Phase 2: Co-purchase graph builder from user data
│   ├── phase2_demo.py           # Phase 1+2 end-to-end demo script
│   ├── phase3_text.py           # Phase 3: Joins review texts to product descriptions
│   ├── phase5_similarity.py     # Phase 5: Embedding index builder + query embedder
│   ├── phase6_sentiment.py      # Phase 6: Individual review extraction for Flair scoring
│   ├── phase7_fusion.py         # Phase 7: Weighted fusion + all quality filters + MMR
│   ├── extract_details.py       # Extracts rating/price/image from raw metadata
│   ├── make_gallery.py          # Utility to generate HTML gallery pages
│   ├── main.py                  # CLI entry point for Phase 1 queries
│   ├── web_app.py               # FastAPI web server (the main app)
│   └── __init__.py
│
├── web/                         # Frontend
│   ├── index.html               # Single-page UI: search box, weight sliders, product cards
│   └── white_shirts_gallery.html
│
├── tests/                       # Unit tests (48+ tests across all phases)
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   ├── test_phase4.py
│   ├── test_phase5.py
│   ├── test_phase6.py
│   ├── test_phase7.py           # 35+ tests including quality filter tests
│   ├── qa_api.py
│   └── __init__.py
│
├── scripts/                     # Evaluation and diagnostic scripts
│   ├── eval_harness.py          # Offline NDCG@10 evaluation with grid search
│   └── (other diagnostic scripts)
│
├── kaggle/                      # Kaggle GPU scripts (used to generate embeddings)
│   ├── phase4_embeddings.py     # Cascade embedding script (ran on Kaggle T4 GPU)
│   ├── phase6_sentiment.py      # Flair sentiment scorer (ran on Kaggle T4 GPU)
│   ├── phase4_kaggle.ipynb      # Ready-to-run Kaggle notebook
│   └── requirements.txt
│
├── data/
│   ├── processed/               # Pre-computed outputs (ALL INCLUDED — ~2.8 GB)
│   │   ├── embeddings_index.npy          # 371,902 × 1536 float32 embedding matrix (2.28 GB)
│   │   ├── embeddings_asins.parquet      # ASIN index for the embedding matrix
│   │   ├── product_texts.parquet         # Full text per product (description + reviews)
│   │   ├── men_shirts_metadata.parquet   # Filtered men's shirt metadata
│   │   ├── reviews_individual.parquet    # Individual reviews for sentiment scoring
│   │   ├── product_details.parquet       # Rating, price, image URL per product
│   │   ├── copurchase_edges.parquet      # Co-purchase graph (1.6M edges)
│   │   └── product_sentiment.parquet     # Sentiment scores per product
│   ├── sample_metadata_records.json
│   ├── sample_reviews_records.json
│   └── sample_zerocore_records.csv
│
├── results/                     # Evaluation outputs
├── PROGRESS.md                  # Detailed development log (full history)
├── README.md                    # Project overview
└── SETUP.md                    # THIS FILE — setup + system guide
```

---

## How to Set Up (Step by Step)

### Step 1: Install Python + clone

Python **3.11+** from python.org (tick "Add to PATH"), then `git clone` the repo.

### Step 2: Place the shared data folder

Copy the shared `data/processed/` folder (~2.8 GB — NOT in git) into the project.
`start.bat` reminds you on every launch if it's missing, and the server refuses
to serve recommendations with a clear "Missing data files in data/processed/..." error.

### Step 3: Double-click `start.bat` — it self-installs on first run

**One file does everything.** On the FIRST run on a machine, `start.bat`
automatically (via `scripts/bootstrap.py`):

1. `pip install -r requirements.txt` (deps from Step 1 missing?)
2. Downloads the two HuggingFace models (~2.5 GB — the only step needing internet)
3. Warns (without blocking) if the data folder isn't in place yet

On success it writes the `.setup_done` marker; EVERY LATER click on start.bat
sees the marker and skips all checking entirely (0.15 s) - straight to the
server. Switching Python versions re-triggers the check automatically; delete
`.setup_done` (or run setup.bat, which uses --force) to force a full re-check.
A failed step prints the fix and re-running resumes where it left off.

`setup.bat` is the optional explicit version (same bootstrap + a full ✅/❌
checklist via `scripts/check_install.py`).

### Step 4: Use it

- The server binds in ~3 s and the page is usable immediately (search + autocomplete);
  the engine loads in a background thread (~1–2 min cold, faster warm).
- Check readiness: http://127.0.0.1:8000/health → `{"ready": true}` when loaded
- Open the UI: **http://127.0.0.1:8000** — search `black shirt` to get recommendations
- `HF_HUB_OFFLINE` is decided automatically: offline when both models are cached,
  online on a fresh machine so the first download can happen.

### Step 5: Run Tests

```bash
cd "REAL SYSTEM"
python -m unittest discover -s tests -p "test_*.py" -v   # 202 tests
```

---

## How the System Works (End-to-End Flow)

When a user types a query like **"white cotton shirt for office"** and hits search, here's exactly what happens inside:

### 1. Query Received
`web_app.py` receives a POST request at `/api/recommend` with the query text and weight sliders.

### 2. Phase 1 — Content Matching (TF-IDF)
**File:** `content_filter.py`

- The query is vectorized using **TF-IDF** (term frequency–inverse document frequency)
- This vector is compared against the TF-IDF vectors of all 371,902 products' metadata (brand + category + size + color)
- Returns the **top 200 candidates** by cosine similarity
- This is pure keyword matching — fast but shallow ("cotton" must literally appear)

### 3. Phase 5 — Embedding Matching (Semantic Search)
**File:** `phase5_similarity.py`

- The query passes through **two AI models** (cascaded):
  - **sentence-t5-xl** → produces 768-dimensional embedding
  - **all-distilroberta-v1** → produces 768-dimensional embedding
  - These are **concatenated** → 1536-dimensional query vector
- This vector is compared against ALL 371,902 pre-computed product embeddings via dot product
- Returns the **top 200 candidates** by semantic similarity
- This understands MEANING — "breathable office shirt" finds "wrinkle-free dress shirt" even without exact keyword match

### 4. Phase 2 — Collaborative Signal (Co-purchase)
**File:** `phase7_fusion.py` → `build_seed_scores()` + `build_collab_scores()`

- The top 25 hits from content + embedding become "seed" products
- The co-purchase graph (1.6M edges, built from 65M user purchase records) is walked:
  - "People who bought seed products also bought X" → X gets a collab score
- This surfaces products that statistically go together with the user's taste

### 5. Phase 6 — Sentiment Signal (Pre-computed)
**File:** `phase6_sentiment.py` (built offline on Kaggle)

- Each product has a pre-computed sentiment score (0 to 1) based on its reviews
- Scored by **Flair sentiment model** (distilbert-based, trained on Amazon reviews)
- Products with few reviews are "shrunk toward neutral" (0.5) to prevent one 5-star review from dominating

### 6. Phase 7 — Fusion (Combining All 4 Signals)
**File:** `phase7_fusion.py` → `fuse()`

```
final_score = w_content × content_score      (default 0.30)
            + w_embed   × embedding_score     (default 0.40)
            + w_sentiment × sentiment_score   (default 0.30)
            + w_collab  × collab_score        (default 0.15)
```

Users can adjust these weights via sliders in the web UI.

### 7. Quality Filters (5 Layers)
**File:** `phase7_fusion.py`

After fusion, results pass through these filters in order:

| Filter | What it does | Example |
|--------|-------------|---------|
| `apply_color_preference` | Only keeps products matching the query's color | "white shirt" → only white products |
| `apply_type_preference` | Separates shirts from t-shirts from tanks | "shirt" → no tees; "tshirt" → only tees |
| `apply_occasion_preference` | For office/formal queries, keeps dress shirts only | "office shirt" → dress/button-down only |
| `apply_novelty_filter` | Bans joke/graphic/gift shirts (100+ banned words) | Removes "Kevin's Famous Chili" shirt |
| `apply_garment_filter` | Removes non-clothing items (pins, hats, socks) | Removes lapel pins, watches |

### 8. Quality Rerank (Bayesian Rating)
**File:** `web_app.py` → `apply_quality_rerank()`

Uses Bayesian shrinkage so a 1000-review 4.5★ product outranks a 1-review 5★ product.

### 9. MMR Diversity Rerank
**File:** `phase7_fusion.py` → `apply_mmr()`

Maximal Marginal Relevance ensures the top results aren't all identical (e.g., 5 Hanes packs).

### 10. Response Sent
The top 12 products are returned with: title, brand, price, rating, image URL, and all individual scores. The web UI renders them as product cards.

---

## The 3 AI Models Used (All Pre-trained, No Training Happens)

| Model | Used In | Purpose | Size |
|-------|---------|---------|------|
| `sentence-t5-xl` | Phase 4 & 5 | Text → 768-dim semantic embedding | 2.2 GB |
| `all-distilroberta-v1` | Phase 4 & 5 | Text → 768-dim semantic embedding (cascade) | 330 MB |
| Flair `sentiment` | Phase 6 | Review text → positive/negative score | 253 MB |

**Important:** The Flair model is only used during the Kaggle scoring step. It is NOT needed to run the web server — sentiment scores are pre-computed in `product_sentiment.parquet`.

Only the **T5 and RoBERTa** models are needed at runtime (to embed the user's query).

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the web UI (`web/index.html`) |
| `/health` | GET | Returns `{"ready": true/false}` — check if models are loaded |
| `/api/recommend` | POST | Main recommendation endpoint (query, weights, top_k, brand, size, min_rating, price in INR) |
| `/api/suggest?q=` | GET | Autocomplete — works even before the engine is warm |
| `/api/facets` | GET | Brand/type/size/price-bucket filter choices (blocks until warm) |
| `/api/brands?q=` | GET | Search all ~35k brands (prefix first, contains fallback) |
| `/api/collab/{asin}` | GET | "People also bought" data for a specific product |

### Example API Call
```bash
curl -X POST http://127.0.0.1:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "white cotton shirt", "top_k": 8}'
```

---

## Key Config (src/config.py)

- `BASE_DIR` = parent of `src/` (auto-detected, no hardcoded path)
- `DATA_PROCESSED_DIR` = `BASE_DIR/data/processed/` (where all parquet files live)
- `RAW_METADATA_PATH`, `RAW_REVIEWS_PATH`, `RAW_ZERO_CORE_PATH` = paths to the original Amazon dataset files (~11 GB total, NOT included — only needed to rebuild processed data from scratch)
- `TOP_K = 100`, `MAX_PRODUCTS_PER_USER = 50`

---

## Hardware Requirements

- **RAM:** 16 GB minimum (the embedding index is 2.28 GB, models ~3 GB)
- **CPU:** Any modern CPU works; GPU not required (inference is CPU-based)
- **Disk:** ~6 GB free (2.8 GB data + 3 GB models in cache)
- **First query:** ~30-60 seconds (embedding the query through two models on CPU)
- **Subsequent queries:** ~5-10 seconds

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Server crashes on startup (OOM) | Close other apps. The engine needs ~8 GB RAM. |
| `HF_HUB_OFFLINE` error / model not found | Run Step 2 above to download models first |
| Server dies when terminal closes | Normal — restart with the uvicorn command |
| "embeddings_index.npy not found" | Make sure `data/processed/` has all 8 parquet/npy files |
| Tests fail on test_phase4 | Known — it needs `kaggle/output/` chunks which aren't required for running |

---

## Reference Paper

Roy, Kumar & Kumar. "Metadata and Review-Based Hybrid Apparel Recommendation System Using Cascaded Large Language Models." *IEEE Access*, Vol. 12, 2024.

**Key methodology:**
- Phase 1: Content-based filtering with TF-IDF
- Phase 2: Item-based collaborative filtering via co-purchase graph
- Phase 3: Concatenate product description + user reviews
- Phase 4: Generate CasT5RoBERT embeddings (cascaded sentence-t5 + RoBERTa)
- Phase 5: Compute embedding similarity for natural-language queries
- Phase 6: Flair sentiment analysis on reviews (paper reports 82.87% accuracy)
- Phase 7: Weighted hybrid scoring: `score = w1·content + w2·embedding + w3·sentiment`
