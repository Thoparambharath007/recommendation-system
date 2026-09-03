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
# Start the server (models load in ~2 min on CPU)
cd "REAL SYSTEM"
$env:HF_HUB_OFFLINE='1'; python -m uvicorn src.web_app:app --host 127.0.0.1 --port 8000
```

Open: http://127.0.0.1:8000

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
│   └── eval_harness.py     # Offline NDCG@10 evaluation
├── kaggle/                 # Kaggle GPU run scripts (Phase 4 & 6)
├── data/                   # Data files (NOT in repo — see setup below)
│   └── processed/          # Parquet outputs from each phase
├── PROGRESS.md             # Detailed development log
└── README.md               # This file
```

## Setup for Team Members

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/apparel-recommendation-system.git
cd apparel-recommendation-system
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn scipy pyarrow fastapi uvicorn sentence-transformers flair torch
```

### 3. Download data files (shared separately)
The `data/processed/` folder (~2.8 GB) contains pre-computed outputs from all phases. Download from the shared drive link and place inside the project:

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

### 4. Download HuggingFace models (first time only)
The server needs two cached models (~3 GB total):
- `sentence-transformers/sentence-t5-xl` (2.2 GB)
- `sentence-transformers/all-distilroberta-v1` (330 MB)

Run once with internet to download:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/sentence-t5-xl'); SentenceTransformer('sentence-transformers/all-distilroberta-v1')"
```

### 5. Run tests
```bash
python -m unittest discover -s tests -p "test_phase*.py" -v
```

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
