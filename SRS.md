# Software Requirements Specification (SRS)
## Metadata & Review-Based Hybrid Apparel Recommendation System

**Version:** 1.1 — 2026-09-23 (verified against codebase: constants, API routes, data files, test suite)
**Based on:** Roy, Kumar & Kumar, “Metadata and Review-Based Hybrid Apparel Recommendation System Using Cascaded Large Language Models”, *IEEE Access*, Vol. 12, 2024. DOI 10.1109/ACCESS.2024.3462793
**Implementation folder:** `REAL SYSTEM/`
**Dataset in use:** Amazon Reviews 2023 — Clothing, Shoes & Jewelry (0-core), 371,902 men's shirts/tops (adaptation of paper's 2018 dataset)
**Team:** 5 members

---

## 1. Introduction

### 1.1 Purpose
This SRS defines functional and non-functional requirements for a 7-phase hybrid recommendation engine that takes a natural-language apparel query (e.g. “white cotton shirt for office”, “blak shrit”) and returns ranked men's shirt products from a 371K catalog, combining content, semantic, sentiment, and collaborative signals via a web demo.

Intended audience: developers, testers, faculty reviewers, team members cloning the repo.

### 1.2 Scope
- In scope: offline data pipeline Phases 1–7, CasT5RoBERT embeddings (1536-d), Flair sentiment, co-purchase graph, FastAPI backend + single-page frontend, typo correction, facets/filters, weight sliders, People-Also-Bought, setup/test scripts.
- Out of scope: online training/fine-tuning (all 3 models pre-trained, inference only), checkout/payments, women's/other categories (requires Kaggle re-embedding ~6h), product detail pages/Amazon deep links, dark theme, post-filter facet counts.
- Product benefits: mitigates content overspecialization, CF sparsity/cold-start via hybrid + sentiment + co-purchase; semantic match beyond keywords.

### 1.3 Definitions
- CasT5RoBERT: cascade `sentence-t5-xl` (768-d) + `all-distilroberta-v1` (768-d) → 1536-d concatenated fingerprint.
- IBCF: item-based collaborative filtering via co-purchase graph.
- MMR: Maximal Marginal Relevance diversity re-rank (lam 0.72, pool 50).
- Rufus layer (`src/rufus.py`): typo correction, intent gate, suggest, facets, INR conversion.
- ASIN/parent_asin: Amazon product identifier (parent variant level used throughout).

### 1.4 References
- Roy et al. IEEE Access 2024 (paper text + Fig.1 two-level hybrid, Fig.4 cascade, Alg.1/2, Eq.1–17)
- Amazon Reviews 2023: https://amazon-reviews-2023.github.io/
- Repo docs: `README.md`, `SETUP.md`, `PROGRESS.md`, `requirements.txt`
- Code: `src/config.py`, `content_filter.py`, `phase2_ibcf.py`, `phase3_text.py`, `phase5_similarity.py`, `phase6_sentiment.py`, `phase7_fusion.py`, `web_app.py`, `rufus.py`, `preprocessing.py`

### 1.5 Overview
Section 2: overall description. Section 3: external interfaces. Section 4: functional requirements by phase/feature. Section 5: non-functional. Section 6: data. Section 7: verification.

---

## 2. Overall Description

### 2.1 Product Perspective
Standalone system: offline-built indexes (`data/processed/`) + runtime engine (`src/web_app.py:Engine`) serving HTTP. No external recommender dependency at runtime except HuggingFace model cache on first run. Replaces single-signal baselines from paper (TF-IDF only, Word2Vec/Doc2Vec/FastText, single transformers).

### 2.2 Product Functions (7-phase pipeline)
| Phase | Function | Key output |
|---|---|---|
| 1 | TF-IDF content filter on brand+category+size+color | top-200 candidates/query |
| 2 | Co-purchase graph from 65M 0-core interactions | 1.6M edges parquet |
| 3 | Join description+features+reviews (max 20 revs, 2000 chars each) | `product_texts.parquet` 303MB |
| 4 | CasT5RoBERT embeddings 371,902×1536 | `embeddings_index.npy` 2.28GB |
| 5 | Query embed + dot-product search | top-200 semantic candidates |
| 6 | Flair distilbert sentiment per review → per-product mean | `product_sentiment.parquet` 4.6MB |
| 7 | Weighted fusion 4 signals + 7 quality filters + Bayesian rerank + MMR | final top-12 |

Fusion formula: `fused = w_content*c + w_embed*e + w_sentiment*s_eff + w_collab*co`, defaults 0.30/0.40/0.30/0.15, user-tunable. Sentiment shrinkage: `s_eff = 0.5+(raw-0.5)*min(n,10)/10`.

### 2.3 User Classes
- Shopper: natural-language search, filters, sliders; needs <10s response, never-blank page.
- Developer/teammate: clone → `start.bat` → search in ~15 min; runs tests, eval harness.
- Reviewer/faculty: checks paper fidelity, runs gauntlets, inspects scores.

### 2.4 Operating Environment
- OS: Windows 10/11 (PowerShell 5.1, `start.bat`/`setup.bat`), also runs via `uvicorn` on any Python 3.11+ host.
- Runtime: Python 3.11+ (tested 3.13), torch 2.4+ CPU, sentence-transformers 3.3+, FastAPI+uvicorn, pandas/numpy/sklearn/scipy/pyarrow. Flair NOT needed at serve time.
- HW: 16GB RAM min (engine 6–8GB: 2.28GB mmap index + T5 2.2GB + RoBERTa 330MB), ~6GB disk (2.8GB data + ~3GB HF cache). No GPU required at serve; Kaggle T4 used offline for Phase 4 (5.75h) and Phase 6 (36 min).
- Network: internet only on first run (~2.5GB model download); thereafter `HF_HUB_OFFLINE=1` auto when cached.

### 2.5 Design Constraints
- Inference only; no model training.
- Parent_asin granularity; edges undirected, weight normalized by global max.
- Embeddings unit-normalized at index build; query vector unit-normalized per request; negative embed clamped to 0.
- Static mount LAST in FastAPI (Starlette order); garment class map built from RAW `categories` before `preprocess_metadata` strips `|`.
- Server binds fast, engine warms in background thread; concurrent warm-up must not double-load (lock-guarded `get_engine()`).

### 2.6 Assumptions & Dependencies
- Shared `data/processed/` (~2.8GB, 8 files) placed manually — NOT in git; server fails fast with actionable message if missing.
- HF cache contains `models--sentence-transformers--sentence-t5-xl` + `all-distilroberta-v1` after `scripts/fetch_models.py`.
- Catalog men's shirts/tops only; 30% have price, 31% titles carry size, ratings sparse — filters must relax, not empty.
- INR rate fixed 83 INR/USD.

---

## 3. External Interface Requirements

### 3.1 User Interface
Single-page app `web/index.html` + `style.css` + `app.js` (no build, offline-friendly):
- Search box with 200ms debounced suggest dropdown (keyboard nav), typo-fix banner (“Fixed blak→black”), fallback banner for no-intent.
- Filter bar: garment chips (Shirts/T-Shirts/Polos/Henleys/Hoodies/Tanks), searchable brand input (top-40 + `/api/brands` 35k search), size XS–4XL+Big&Tall, 5 INR buckets (0-415/415-1245/1245-2490/2490-4980/4980+), 4★&up chip, clear-all + active-filter chips with ✕.
- Product cards: image, title, brand, ★rating+count, price USD+INR or “price not listed”, 4 score bars + fused score, in_budget/in_brand/in_size/in_rating tags, People-Also-Bought strip for top hit.
- Advanced: 4 weight sliders (persisted localStorage), sort (best/price/rating), recent (5) + popular searches, skeleton shimmer + cold-start progress strip, error + Retry, `/` focuses search.

### 3.2 Hardware Interfaces
None beyond standard CPU/RAM/disk. No GPU, printer, sensor interfaces.

### 3.3 Software Interfaces
- Amazon 2023 raw: `meta_Clothing_Shoes_and_Jewelry.jsonl.gz` 3.77GB, `Clothing_Shoes_and_Jewelry.jsonl.gz` 6.62GB/22.6M reviews, `Clothing_Shoes_and_Jewelry.csv.gz` 1.29GB/65M rows (rebuild only).
- HuggingFace: `sentence-transformers/sentence-t5-xl`, `all-distilroberta-v1`, Flair `sentiment` (HU Berlin mirror, Kaggle only).
- Parquet/npy via pyarrow/numpy mmap; HF cache `~/.cache/huggingface/hub`.

### 3.4 Communications / API
Base `http://127.0.0.1:8000`. All JSON.

- `GET /` → index.html. `GET /health` → `{"ready":bool,"error"?:str}`.
- `GET /api/suggest?q=` → `{"suggestions":[]}` works pre-warm.
- `GET /api/facets` → `{"usd_inr":83,"brands":[{name,count}],"types":[],"sizes":[],"price_buckets":[]}` blocks till ready (UI retries).
- `GET /api/brands?q=` → prefix-first + contains fallback, case-insensitive, never raises.
- `GET /api/collab/{asin}?k=6` → `{"asin":"","items":[{parent_asin,title,rating,n_ratings,price,image_url,copurchase_weight}]}`.
- `POST /api/recommend` request: `{query* (1-200 chars), top_k 1-50 def 12, w_content/w_embed/w_sentiment 0-1 def .3/.4/.3, w_collab 0-1 def .15, min_reviews 0-20 def 1, min_price/max_price USD?, brand?, size? (SIZE_OPTIONS), min_rating 0-5?, min_price_inr/max_price_inr?}`. INR overrides parsed budget; explicit USD markers force USD, else ≥100 treated as INR/83.
- Response: `{query, corrected_query?, corrections?[{from,to}], elapsed_s, weights{}, price?{min,max,relaxed,min_inr,max_inr}, note?{fallback,message?,brand_relaxed?,size_relaxed?,rating_relaxed?}, items[]:{parent_asin,title,brand,rating,n_ratings,price,image_url,content_score,embed_score,sentiment_score,collab_score,n_reviews,fused_score,in_budget?,in_brand?,in_size?,in_rating?,price_inr?}}`.
- Errors: 422 empty/unknown size, 503 engine not ready/failed, 500 friendly without stack dump. Unknown path 404.

---

## 4. Functional Requirements

### FR-1 Data Ingestion & Preprocessing
- FR-1.1 Stream 4GB metadata line-by-line, filter men's shirts via exact category tokens (`shirt/shirts/tee/tops`, exclude women's leakage), cache `men_shirts_metadata.parquet` 139MB/371,902 rows.
- FR-1.2 `clean_text`: strip HTML, keep `[A-Za-z0-9 &'-]`, lowercase; `extract_size` with apostrophe guard (men's ≠ S); `extract_color` via COLOR_WORDS; `classify_garment` from raw pipe-joined categories → novelty/costume/underwear/swim/hoodie/tank/tee/polo/henley/shirt/unknown (exact “novelty” segment).
- FR-1.3 Build `metadata_string = brand+categories+size+color`; drop empty brand/empty string rows.

### FR-2 Content Filtering (Phase 1)
- FR-2.1 `ContentFilter`: `TfidfVectorizer.fit` on metadata_string; `get_top_k(query,k)` cosine → DataFrame with `score`.
- FR-2.2 Default pool CONTENT_K 200 (widen ×25 when price/color filter or retry active); zero-score queries warn, not dump 100 junk rows.

### FR-3 Co-purchase Graph (Phase 2)
- FR-3.1 Stream 0-core CSV 65M rows in ~86s, keep catalog parent_asins, cap 50 products/user, count unordered pairs → 1,621,543 edges (max wt 320), 211K products with edges → `copurchase_edges.parquet` 10MB.
- FR-3.2 `build_neighbor_map`: bidirectional adj, weight/max normalization 0–1. `build_seed_scores`: top-25 content+embed normalized union. `build_collab_scores`: `seed*weight*0.5`, max-take, renormalize 0–1 + via map.

### FR-4 Review Text Join (Phase 3)
- FR-4.1 Stream 6.62GB reviews, match catalog, keep ≤20/product ≤2000 chars → 2.84M matched, 1.098M kept for 371,834 products; `full_text = description+features+reviews` avg 948 chars → `product_texts.parquet`.

### FR-5 Cascade Embeddings (Phase 4)
- FR-5.1 Kaggle: `sentence-t5-xl` + `all-distilroberta-v1` fp16 CUDA batch 8, checkpoint every 10K, resume-safe, 38 chunks 1536-d float32 (~1GB parquet, 2.28GB npy), 371,902 rows in 345 min; order == product_texts order; norm ~1.41.
- FR-5.2 `EmbeddingIndex.build`: vstack, zero-norm guard, unit-norm, save npy + asins parquet in ~11s; `load` via mmap.

### FR-6 Semantic Search (Phase 5)
- FR-6.1 `QueryEmbedder.embed`: same cascade CPU, concat 768+768, float32; `top_k`: unit-norm query, `matrix@vec`, argpartition.
- FR-6.2 Engine: EMBED_K 200 (widen ×25), LRU cache 64 queries; query ~3–5s CPU + search ~1.9s over 371K.

### FR-7 Sentiment (Phase 6)
- FR-7.1 Extract `reviews_individual.parquet` 93MB/1,098,446 rows in ~298s; Kaggle Flair distilbert scores each review (504 rev/s, 36 min) → `product_sentiment.parquet` 4.6MB/371,643 rows (mean 0.728, median 0.99, n 1–20).
- FR-7.2 Mapping: POSITIVE `0.5+0.5*conf`, NEGATIVE `0.5-0.5*conf`; aggregate mean+n; `sentiment_to_dict` precomputed once; missing → neutral 0.5.

### FR-8 Hybrid Fusion + Quality (Phase 7)
- FR-8.1 `fuse(candidates=union(content,embed), normalize=True scale-by-max)`: clamp embed/collab ≥0, shrinkage `s_eff`, `min_reviews` gate, weighted sum, sort fused desc.
- FR-8.2 Order: fuse → color → type → occasion → novelty → garment → gender → plain-shirt → brand/size/rating/price (relax-and-flag if < thresholds: brand ≥1, size ≥4, rating ≥4, budget ≥3) → `renormalize_signals` → secondary-color +0.10 → `quality_rerank` Bayesian (prior 3.9, c=25, wq 0.20) → `apply_mmr`.
- FR-8.3 Filters: color primary-first + brand-strip + collision ban + Newport-Blue prefix demote; type authoritative class + title safety nets + USPA brand-strip + polo/henley word split; occasion strict→relaxed; novelty 100+ words plural-tolerant + class ban unless query wants merch; garment accessory ban; gender women's-title drop; plain-shirt spec-word guard for bare color-shirt; price INR parse with size/neck guard, relax if <3 in budget.
- FR-8.4 Widen-and-retry once (pool ×25) if empty after strict filters and garment type detected; `parse_price_range` handles under/below/over/around/between/not-more/up-to + ranges.

### FR-9 Rufus Robustness
- FR-9.1 `correct_query`: OSA distance-1 only vs ~250-word vocab freq tie-break; numbers/apostrophes/no-candidate untouched; returns corrected + corrections list (e.g. “blak shrit”→“black shirt”).
- FR-9.2 `has_apparel_intent`: garment/color/style/brand token or multi-word brand run (under armour) required; else top-rated fallback (sort rating,raring_number) with fused 1.0-0.01*i and message.
- FR-9.3 `suggest` tiers curated>vocab>brands; `build_facets` top-40 brands (junk filtered), 6 types, 9 sizes, 5 contiguous INR buckets; `filter_by_rating/brand/size`, `search_brands`, INR `usd_to_inr/inr_to_usd` roundtrip.

### FR-10 Web Engine
- FR-10.1 `Engine.__init__` loads content filter+garment map, index, embedder, sentiment dict, adj, titles/brands/details/ratings/sizes/facets with `del+gc` between loads; fail-fast missing-data message; `recommend()` implements §FR-8 + widen logic.
- FR-10.2 Startup: bind instantly, warm in daemon thread; `get_engine` lock-blocks; `/health` reflects ready/error.

### FR-11 Setup & Ops
- FR-11.1 `start.bat` → `scripts/bootstrap.py`: pip install requirements, fetch 2 models resume-safe, warn if data missing, write `.setup_done`; later runs skip (0.15s); `setup.bat`/`check_install.py` full ✅/❌ checklist.
- FR-11.2 `scripts/eval_harness.py`: cached stage-A + grid search NDCG@10/P@10 over 10 landmark queries → `results/eval_results.json`.

---

## 5. Non-Functional Requirements

### 5.1 Performance
- NFR-1 Server bind ~3s; warm 60s cold disk / 10–30s warm cache; `/suggest` instant pre-warm.
- NFR-2 First query 30–60s CPU (T5+RoBERTa), subsequent 5–10s (cache hit ~0s embed + ~2s search + fusion).
- NFR-3 Index build 11s; Phase 2 86s; Phase 3 392s; tests 215 in ~17s (server stopped).

### 5.2 Reliability / Robustness
- NFR-4 Never blank/500 on bad input: typo fix, fallback top-rated, relax-and-flag, strict-pass retry, friendly 500 catch-all.
- NFR-5 Verified: 6-query type gauntlet 12/12 zero leaks; 10-color audit 0 tee-leaks; budget-relax keeps full page.

### 5.3 Usability
- NFR-6 Single-click run, instant page, banners/chips explain corrections/relaxes, INR formatting, skeleton loading, persisted sliders/searches.

### 5.4 Maintainability & Portability
- NFR-7 Pure functions for filters/Rufus (unit-testable), <500-line web assets, `BASE_DIR` auto-detect, Windows + generic Python host.

### 5.5 Security / Privacy
- NFR-8 No auth/PII; localStorage only for prefs; `esc()` XSS escaping for titles/brands; no stack dumps to client; size/query length caps (200 chars, top_k 50).

### 5.6 Quality Targets (paper fidelity)
- NFR-9 Paper reports CasT5RoBERT 94.76% vs FastText 66.9%/Word2Vec 47.7%/Doc2Vec 46.6%; Flair 82.87% vs VADER 77.11%/TextBlob 59.94% — implementation uses same choices (Flair distilbert Amazon-domain) + TF-IDF+Flair combo.

---

## 6. Data Requirements

`data/processed/` (~2.8GB, NOT in git):
- `embeddings_index.npy` 2.28GB 371,902×1536 float32 unit-norm; `embeddings_asins.parquet` 3.7MB
- `product_texts.parquet` 303MB (parent_asin,title,reviews_text,full_text)
- `men_shirts_metadata.parquet` 139MB; `reviews_individual.parquet` 93MB/1.09M rows
- `product_details.parquet` 26MB (title,brand,average_rating,rating_number,price,image_url)
- `copurchase_edges.parquet` 10MB (product_a,product_b,weight); `product_sentiment.parquet` 4.6MB (parent_asin,sentiment_score,n_reviews)
- Catalog stats: 35,004 brands, 30% priced median $19.99, 31% sized, garment novelty 43%/shirt 18%/tee 15%/hoodie 6%/polo 5%/henley 2%.

---

## 7. Verification & Acceptance

- `python -m unittest discover -s tests -p "test_*.py"` → 215/215 PASS (~17s, server stopped). Covers Phase 1–7 + Rufus (fuse math, filters, MMR, collab, typo, facets, INR). Test breakdown: phase1 4, phase2 6, phase3 5, phase4 5, phase5 8, phase6 6, phase7 137, rufus 44.
- Live checks: `blak shrit`→fix banner 12 items; `asdfghjkl`→fallback 12 top-rated; `black shirt` 12/12 genuine; combo brand+size+price+4★ correct; `/api/brands essentials`→amazon essentials; assets 200, unknown 404.
- Acceptance: clone → data in place → `start.bat` → :8000 ready → 12 relevant cards with scores + PAB strip.

---

## 8. Future Enhancements (out of scope)
Post-filter facet counts, detail page/deep links, dark toggle, full-catalog expansion, head-to-head sentiment eval on 2K star-rated reviews.
