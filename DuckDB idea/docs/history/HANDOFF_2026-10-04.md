# SESSION HANDOFF — 2026-10-04 (price-estimation era)

> Purpose: ship the entire conversation's state into a new Qoder session.
> Read this + long-term memory → full continuity. Zero re-asking needed.

## Where we are (one paragraph)
50,224 vision-verified men's shirts (CLIP, Kaggle run) now carry a **100%
price-coverage table** (`DuckDB idea/data/shirts_priced.parquet` + .csv):
8,251 exact Amazon-2023 metadata prices + 41,973 brand-median estimates
(72.3% within ±$10 on blind holdout, MAE $9.28). The Kaggle bestseller
price dataset was tested, found truthful (82-85%) but only ~1.7% coverage
— and is now **BANNED from production tables by user decision**. All work
happens ONLY inside `DuckDB idea\`; scratch files go to `JUNK created by qoder\`.

## Decisions locked (do not relitigate)
1. Backbone = the 50,224 verified shirts. NOT the 5.8k Kaggle shirts. No strip-to-8k.
2. Old dataset (Amazon Reviews 2023 lineage) ONLY. Kaggle bestseller file excluded.
3. Price = tiered with `price_source` labels; exacts never overwritten; fabrication forbidden.
4. Queries ALWAYS search all 50k; price splits results into tiers (exact → estimated), never hides unpriced items.
5. Currency: INR_PER_USD = 83.0 config constant (₹415 query ≈ $5 → honest empty + nearest-match).
6. Base paper (Roy et al., IEEE Access 2024) facts verified from PDF:
   - nulls REMOVED (their corpus 793,872 → 221,167 reviews; product count never disclosed)
   - the words price/cost/budget appear ZERO times in 19 pages → their system cannot answer
     price queries AT ALL → our judgment-experiment ammunition
   - full text grep-able: JUNK created by qoder/paper_roy2024_fulltext.txt
7. Reviews-v2 (verified_purchase filter) still pending — paper does the same (cite p.11).

## Files that matter
| File | What |
|---|---|
| DuckDB idea/data/shirts_priced.parquet | THE price table, 50,224 rows, 100% priced, labeled |
| DuckDB idea/data/shirts_final.parquet | CLIP-verified catalog (canonical: segregation/verified/) |
| DuckDB idea/data/asin_bridge.parquet | 100,902 child→parent pairs (built for Kaggle join; now spare) |
| DuckDB idea/01_first_query.py | founding DuckDB demo |
| DuckDB idea/02_build_asin_bridge.py | bridge builder (66M-line scan, 7 min) |
| DuckDB idea/03_price_estimator.py | current estimator (brand-median + holdout card) |
| datasets/amazon_products_2023/ | banned-from-use bestseller file (leave on disk or JUNK) |
| C:\Users\thopa\.kaggle\kaggle.json | API token (installed kaggle CLI 2.2.4) |

## NEXT STEP — approved in principle, not yet built: V2 price model
Train `sklearn.ensemble.HistGradientBoostingRegressor` on the 8,251 exact prices
(features: title TF-IDF, brand target-encoding, sleeve, average_rating,
log(rating_number)) → blind 2-fold exam → split-conformal 80% bands →
re-estimate the 41,973 as `price_source='estimated-ml'` → scoreboard:
brand-median 72.3% vs ML model X%. Script name: `03b_price_estimator_v2.py`.
Known weaknesses to fix in v2: bands too tight (41% coverage), brand-only
blindness to style words (silk vs cotton), 1-sibling "med" tier.

## Then the queue (agreed roadmap)
04_shirts_master.py (bolt sentiment+reviews+copurchase onto priced table) →
reviews_v2 (verified_purchase gates, cite paper) → DuckDB sql/ library →
LLM→SQL natural-language layer (mentor's proposal) → judgment experiment
(20 price queries: our engine vs paper's similarity-only method) → eval
metrics (Recall@10, NDCG@10, MRR) → PROGRESS.md + SRS updates.

## House rules (memorized, IDs in memory system)
- Work ONLY in "DuckDB idea"; scratch/verification → "JUNK created by qoder", referenced from HQ.
- Every dataset artifact ships with a CSV copy for human verification.
- Plain-language explanations with real project examples.
- Never claim without measurement; if asked for "my word" → run the test instead.
