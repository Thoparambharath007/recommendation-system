# Project Progress — Metadata & Review-Based Hybrid Apparel Recommendation System

**Based on:** Roy, Kumar & Kumar — "Metadata and Review-Based Hybrid Apparel Recommendation System Using Cascaded Large Language Models" (IEEE Access, Vol. 12, 2024)

**Working folder:** `scratch/` (all project files live here)

---

## Project Decision (IMPORTANT)
We are implementing this project using the **2023-version UCSD Amazon dataset** (richer data), NOT the 2018 version the paper used. The methodology is being **adapted** accordingly (see adaptations below).

---

## 7-Phase Implementation Plan

- [x] **Phase 1 — Content-Based Filtering (COMPLETE, verified)**
  - Load metadata → filter men's shirts → preprocess → concatenate metadata → TF-IDF → cosine similarity → top 100 products
- [x] **Phase 2 — Expand top 100 via co-purchase inference (COMPLETE, verified)**
  - Rebuilt the missing also_bought/also_viewed graph from 0-core reviews (user_id + parent_asin)
- [x] **Phase 3 — Concatenate description + reviews (COMPLETE, verified)**
  - Streamed 6.62 GB reviews file, joined review texts to catalog products
- [x] **Phase 4 — Generate CasT5RoBERT embeddings** (cascaded sentence-t5-xl + all-distilroberta-v1) ✅ COMPLETE, verified
- [x] **Phase 5 — Compute embedding similarity** (custom phrase vs product info) ✅ COMPLETE, verified
- [x] **Phase 6 — Sentiment analysis** (Flair chosen in paper: 82.87% vs VADER 77.11%, TextBlob 59.94%) ✅ COMPLETE, verified
- [x] **Phase 7 — Weighted scoring + final recommendations** (weighted score = w1·content + w2·embedding + w3·sentiment) ✅ COMPLETE, verified — see Phase 7 section

---

## Dataset — 2023 Version (WHAT WE HAVE)

**Files on disk** (`C:\Users\thopa\OneDrive\Desktop\project\documents\`):
- `meta_Clothing_Shoes_and_Jewelry.jsonl.gz` — **3.77 GB** ✅ (2023 schema)
- `Clothing_Shoes_and_Jewelry.jsonl.gz` — **6.62 GB** ✅ (22.6M full reviews, verified decompress)
- `Clothing_Shoes_and_Jewelry.csv.gz` — **1.29 GB** ✅ (0-core IDs, 65.18M rows, verified decompress)

**2023 metadata schema** (confirmed): main_category, title, average_rating, rating_number, features, description, price, images, videos, store, categories, details, parent_asin, bought_together

**Schema differences from 2018 version:**
- NO top-level `asin` — only `parent_asin`
- NO `brand` field — brand comes from `store`
- NO size/color fields — extracted from titles (done in Phase 1)
- NO `also_buy` / `also_view` fields — **Phase 2 blocker**
- `bought_together` exists but is EMPTY (0/500K records)

**Incomplete downloads found** (`Unconfirmed *.crdownload` files — Chrome downloads that never finished):
- Unconfirmed 998326.crdownload — 4.72 GB
- Unconfirmed 607888.crdownload — 2.22 GB
- Unconfirmed 860123.crdownload — 1.09 GB
- Unconfirmed 922515.crdownload — 0.01 GB
- (One of these may have been the reviews file download)

---

## Phase 4 — Kaggle run folder (READY — user runs today)

**Folder:** `scratch/kaggle/` — everything needed for the Kaggle T4 GPU run:
- `phase4_embeddings.py` — cascade script (sentence-t5-xl + all-distilroberta-v1), fp16 on CUDA, checkpoint-resume every 10K products, batch size 8
- `phase4_kaggle.ipynb` — ready notebook (pip install → sample test → full run)
- `input/product_texts.parquet` — Phase 3 output copy (289 MB, 371,902 rows)
- `input/product_texts_sample.parquet` — 50-row quick test (1-2 min)
- `requirements.txt`, `README_KAGGLE.md` — full step-by-step

**Key facts:**
- Output: 38 chunks `embeddings_XXXXX.parquet`, cols parent_asin + embedding (1536 dims = 768 T5 + 768 RoBERTa), ~2.3 GB total
- 3-4 h on T4; resume-safe (completed chunks skipped on re-run)
- ETA printed every chunk; user downloads chunks back to `scratch/kaggle/output/` and verifies 371,902 rows
- Kaggle: Accelerator GPU T4 x2, Internet ON, 12h session / ~30h weekly quota
- CONCEPT: inference only — models are pre-trained, no training happens anywhere

**Phase 4 — STATUS (COMPLETE — verified 10 Aug 2026)**

**Run log:**
- 50-row sample test passed first (0.1 min; confirmed T5 -> 768 dims, RoBERTa -> 768 dims, cascade -> 1536 dims)
- ⚠️ BUG FOUND & FIXED: the sample test wrote `chunk 0` into the SAME default output dir as the full run → full run reported "chunk 0 already done, skipping" and would have silently MISSED products 1–10,000. Fixed by stopping, `!rm -rf output`, re-running clean. (Lesson: never run the sample with the same --out dir — use `--out output_test` for tests.)
- ⚠️ ZIP BUG: first `!zip -r output.zip output` on Kaggle died mid-zipping (758 MB truncated zip, downloaded twice before noticing). Fixed: `!rm output.zip`, re-zip, verify with `!unzip -t output.zip`.
- Full run: **371,902 embeddings in 345.3 min (5.75h) on T4**; ETA dropped 9.6h → 0 as GPU warmed
- Downloaded via `FileLink('output.zip')` → 1,069,996,006 bytes → extracted to `scratch/kaggle/output/` (38 chunks, 1.01 GB)
- **Verified: `tests/test_phase4.py` ALL PASS** — 38 chunks, 371,902 rows, 1536 dims, float32, finite, consistent norm (~1.41, cosine unaffected), ASIN order == product_texts.parquet order
- Note: `encode(normalize_embeddings=True)` is NOT default in sentence-transformers → embeddings are non-unit norm; cosine_similarity handles it (Phase 5)

**Models used in whole project (3 total, all pre-trained, no training anywhere):**
1. sentence-t5-xl (or base) — Phase 4, text → fingerprint
2. all-distilroberta-v1 — Phase 4, cascade second pass
3. Flair sentiment — Phase 6, review sentiment
(Phases 1-2 are math, not models: TF-IDF/cosine + co-purchase counting)

**IMPORTANT conceptual points (user clarified on 9 Aug, keep in mind):**
- Phase 2's expanded list does NOT feed Phase 3. Phase 2 output feeds the final ranking (Phase 5/7) at query time (milliseconds, graph lookups). Phase 3 input = ENTIRE catalog (all 371,902 shirts, PROVEN: product_texts.parquet row count == metadata row count).
- Two parallel tracks merge at Phase 5: (a) query → top-100 → expanded list; (b) catalog → full_text → fingerprints.
- Query time ≈ 3 seconds total (user's query only passes through the model, ~1 sentence). All heavy work is ONE-TIME build: TF-IDF index (Phase 1), co-purchase graph (Phase 2, 86s), full_text (Phase 3, 6.5 min), embeddings (Phase 4, Kaggle 2-4h).

---

## Phase 5 — Implementation Details (done)

**Files:**
- `src/phase5_similarity.py` — `QueryEmbedder` (query → cascade fingerprint on CPU) + `EmbeddingIndex` (unit-norm matrix, top-k dot product)
- `tests/test_phase5.py` — 8 unit tests, ALL PASS (28 total across phases)

**Index build (real data):**
- 38 chunks → `data/processed/embeddings_index.npy` (2.28 GB, float32, 371,902 × 1536, unit-normalized) + `embeddings_asins.parquet`; built in 11.1s
- Loaded via `np.load(mmap_mode="r")` — memory-safe

**Query time (local CPU):**
- Model load (T5-xl + RoBERTa, cached in ~/.cache/huggingface): ~30s first time
- Query embed: 3.3s | Search 371,902 products: 1.87s | total ≈ 5s
- Demo: "breathable cotton shirt for office or business meetings" → DKNY cotton dress shirt, Stafford 100% cotton 3-pack, wrinkle-free polo in top-5 (semantic sense; "The Office" shirts rank high on 'office' keyword — fusion in Phase 7 will refine)

**Setup notes:**
- Installed locally: sentence-transformers 5.7.0, h5py, tf-keras (transformers 6.x TF-import quirk)
- embeddings are NOT unit-norm from encode() (default normalize_embeddings=False) — index normalizes at build
- CLI: `python src\phase5_similarity.py --build-index` / `--query "..." --top-k 10`

---

## Phase 6 — Implementation Details (COMPLETE — verified 11 Aug 2026)

**Task:** sentiment score (0-1) per product, averaged from its reviews, using the paper's Flair sentiment (82.87%).

**Done so far (10 Aug 2026):**
- `src/phase6_sentiment.py` — local build: re-streams 6.62 GB reviews, saves ONE ROW PER REVIEW → `data/processed/reviews_individual.parquet` (**1,098,446 reviews for 371,834 products, built in 298s**; Phase 3 had joined reviews into a string, so individual reviews had to be re-extracted)
- `scratch/kaggle/phase6_sentiment.py` — Kaggle GPU scorer: reads reviews parquet, scores each review, aggregates per product (mean + n_reviews) → `product_sentiment.parquet` (small, ~10MB)
- `tests/test_phase6.py` — 6 tests (extraction caps/cleaning, sentiment mapping, aggregation) ALL PASS — **34/34 total**
- Sentiment mapping: Flair POSITIVE/NEGATIVE + confidence → positive: 0.5+0.5·conf, negative: 0.5−0.5·conf, else 0.5

**FULL RUN COMPLETE (11 Aug, overnight Kaggle):**
- One-cell set-and-forget script (install → sample test → full run → auto-push to dataset)
- Sample test: 5,000 reviews in 17s (303 rev/s) ✅
- Full run: **1,098,446 reviews in 2,181s (36.6 min)** on T4, 504 rev/s
- Saved **371,643 product sentiment scores → product_sentiment.parquet** (4.6 MB)
- Downloaded via Kaggle dataset → `data/processed/product_sentiment.parquet`
- **Verified**: 371,643 rows, sentiment_score 0-1 (mean 0.728, median 0.9925), n_reviews 1-20 (cap), parent_asin str matches index

**⚠️ THE BLOCKER — SOLVED (11 Aug):**
- Flair model server is NOT the problem — the HU Berlin university server is public: `https://nlp.informatik.hu-berlin.de/resources/models/sentiment-curated-distilbert/sentiment-en-mix-distillbert_4.pt` (253 MB, HTTP 200). The 401 was only on the HF **mirror** repos (flair/sentiment* are gated). flair 0.15.1's `TextClassifier.load("sentiment")` downloads from HU Berlin directly — works.
- Earlier "hangs" were just slow downloads of the big RNN model (1.18 GB).

**DECISION MADE (11 Aug): use Flair's standard `sentiment` model** (distilbert-based, trained on a mix incl. Amazon review corpus — domain-matched; better than the paper's old RNN which scored 82.87%). distilbert SST-2 (91.3% SST-2) rejected: movie-review domain, no paper fidelity, no guarantee on Amazon data. Optional future: head-to-head eval on 2K reviews with star ratings as ground truth.

**Local proof COMPLETE (11 Aug) — script validated:**
- Model downloaded (253 MB) + loaded; 1,000 reviews scored in 91s on CPU (11 rev/s)
- 867 products scored; mean 0.781 (realistic — Amazon reviews skew positive); 178 products < 0.3, 670 > 0.7
- Judgment spot-check PASSED: "Very thin and cheap fabric" → 0.00 ✅ | "come out of the dryer looking like brand new... strong yet soft" → 1.00 ✅
- Proof file deleted after verification

**Next steps:**
1. Upload `reviews_individual.parquet` (88.8 MB) + `kaggle/phase6_sentiment.py` to Kaggle dataset (upload-folder)
2. Notebook: `!pip install flair` → sample test `--limit 5000` → full run (~5-10 min T4, model downloads there too)
3. Download `product_sentiment.parquet` → `data/processed/`, spot-check, Phase 6 done
4. Phase 7 = weighted fusion: score = w1·content + w2·embedding + w3·sentiment (defaults 0.2/0.5/0.3)

---

## Phase 7 — Implementation Details (COMPLETE — verified 11 Aug 2026)

**Task (paper finale):** weighted hybrid score = w1·content + w2·embedding + w3·sentiment, one ranking from all three signals.

**Files:**
- `src/phase7_fusion.py` — CLI: `python src/phase7_fusion.py --query "..." --top-k 10`
- `tests/test_phase7.py` — 11 unit tests (weights math, ranking order, missing-signal defaults, embed clamping, min-review gating, custom weights) — **45/45 total PASS**

**Design:**
- Candidates = union of content top-200 (TF-IDF, Phase 1) + embedding top-200 (Phase 5)
- Sentiment shrunk toward neutral by review count: `sent_eff = 0.5 + (raw − 0.5)·min(n_reviews, 10)/10`; products with < min_reviews (default 1) → neutral 0.5
- Default weights w=(0.3, 0.4, 0.3), tunable via CLI; embed scores clamped ≥ 0
- BUG FOUND & FIXED: query vector was not unit-normalized before `index @ vec` → embed scores > 1.0 (cosine impossible); fixed in phase7_fusion.py (normalize query like phase5's top_k does)

**Live run (local CPU, ~2 min incl. model load):**
- "breathable cotton shirt for office" → top: Fruit of the Loom 4-pack white crew tees, Haggar microfiber woven shirt, WOLVERINE plaid shirt — with sentiment ON, well-reviewed shirts (n≥2) outrank one-review novelty tees; with w_sentiment=0, novelty tees (Kevin's Chili, Schrute Farms) dominate → sentiment signal demonstrably improves ranking

**PROJECT COMPLETE — all 7 phases implemented, 61/61 tests passing.**

---

## Website (web demo — 11 Aug 2026, READY)

**URL:** http://127.0.0.1:8000 (local)

**Run it (12 Aug — HF_HUB_OFFLINE wrapper REQUIRED):**
```powershell
cd C:\Users\thopa\OneDrive\Desktop\project\scratch
# PowerShell 5.1 has no -Environment param; wrap like this (or set env in System Properties):
$cmd = "`$env:HF_HUB_OFFLINE='1'; python -m uvicorn src.web_app:app --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList "-NoProfile","-Command",$cmd -WorkingDirectory "C:\Users\thopa\OneDrive\Desktop\project\scratch"
```
- First startup loads models in background (~2 min CPU; T5 2.2GB + RoBERTa — **memory-tight on 16GB RAM**, see crash note above; /health says ready when done)
- After load, each query ~30-60s on CPU (embedding 1536-d search over 371,902 products)
- ⚠️ SERVER DIES if terminal closed or machine sleeps — restart with the command above
- ⚠️ Every code change = restart = model reload (~2 min); logs: `C:\Users\thopa\AppData\Local\Temp\opencode\webapp_out.log` / `webapp_err.log`

**Files:**
- `src/web_app.py` — FastAPI backend (POST /api/recommend, GET /, GET /health); loads Engine once: content filter + embedding index + T5/RoBERTa embedder + sentiment + titles + details
- `web/index.html` — single-file UI: search box, weight sliders (content/embedding/sentiment), product cards with image/price/rating + 3 score bars + hybrid score
- `src/extract_details.py` — extracts rating/price/image_url per product from raw metadata → `data/processed/product_details.parquet` (26 MB, 371,902 rows; built in 171s)

**Quality filters (added after user testing — 5 layers, in phase7_fusion.py, applied in CLI + web):**
1. `apply_color_preference` — query color word → ONLY products whose title contains that color (case-insensitive); PRIMARY color = first color word in query ("white shirt with black buttons" → only white); brand name is STRIPPED from title before matching (fixes "White Water" brand matching "white"); COLOR_NAME_COLLISIONS ban proper-noun color phrases (walter white, white house, black friday, black widow, white water, red bull, grey goose, black rifle, us navy, usn, charlie brown, brown brothers, red shirt friday, allis chalmers, garth brooks, peanuts, duke)
2. `apply_type_preference` — query "shirt" → exclude tees (dress/button-down/oxford only); query "tshirt/tee/t shirt" → only tees; TEE_MARKERS incl " t " standalone; TANK_MARKERS: tank top, " tank ", sleeveless, muscle, a-shirt
3. `apply_occasion_preference` — office/formal/business queries: strict = office words (dress/oxford/button/slim fit/business/formal/wrinkle/professional/performance/collar) + no novelty; relaxed fallback = just ban novelty
4. `apply_novelty_filter` — ALWAYS-ON ban: funny, graphic, gift, grandma, birthday, st patrick, shamrock, parody, lantern, gypsy, sneaker, spoof, comical, grunt style, shirt for jordan, logo, doobie, scooby, shaggy, political, proud, soldier...; matching via `_word_matches()` — REGEX with plural/possessive tolerance ("shamrocks" matches "shamrock", "patricks" matches "patrick") — this was a real bug (exact-match missed plurals)
5. `apply_garment_filter` — drop accessories/non-garments (pin, badge, brooch, watch, hat, socks, gloves, belt, scarf, shoes, jewelry, necklace, tie, cufflink...)

**User-observed issues fixed (11 Aug):**
- "white shirt" returned black/blue tees → color filter fixed
- "white shirt" returned t-shirts → type filter fixed (shirt ≠ tee)
- "white shirt" returned a LAPEL PIN ("A N KINGPiiN White Fighter Jet...Pin Badge...Brooch") → garment filter fixed
- Verify command: `$body = '{"query":"white shirt","top_k":8}'; Invoke-WebRequest -Uri http://127.0.0.1:8000/api/recommend -Method POST -Body $body -ContentType application/json`
- Tests: `tests/test_phase7.py` grew 11 → 27 tests (color/type/garment preference) — **61/61 total PASS**

**User-observed issues fixed (12 Aug) — color boundary + tee/shirt audit:**
- ⚠️ SERVER CRASH (12 Aug): PyTorch c10.dll access violation (0xc0000005) while loading sentence-t5-xl — RAM pressure (7.4 GB free at crash: 2.28GB index mmap + TF-IDF + 2.2GB T5 + RoBERTa + TF). FIXED in web_app.py Engine.__init__: `del df / del titles_df / del brands_df` + `gc.collect()` between loads. Also `import re` added to phase7_fusion.py. Server requires `HF_HUB_OFFLINE=1` (HuggingFace HEAD requests hang offline → use wrapper: `$env:HF_HUB_OFFLINE='1'; python -m uvicorn src.web_app:app --host 127.0.0.1 --port 8000` via Start-Process powershell).
- **10-color audit experiment** (30k sampled pool through full filter chain): green/yellow/blue/red/grey/navy/black/white/brown/orange — **0 tee-leaks in ALL colors** (proves shirt-vs-tee differentiation is airtight; earlier t-shirt results were the pre-fix server)
- "green shirt" → St. Patrick's Day/shamrock/parody/Green Lantern/Scooby-Doo leaks → fixed via NOVELTY_WORDS expansion + plural-tolerant regex matcher
- "navy shirt" → "US Navy USN Retired Polo" (military navy ≠ color navy) → COLOR_NAME_COLLISIONS
- "brown shirt" → "Charlie Brown" (cartoon character) → COLOR_NAME_COLLISIONS
- "XDIAN Men's A-Shirt" (undershirt/vest) leaked → added "a-shirt" to TANK_MARKERS
- FINAL VERIFIED (server live): "green shirt" → 6 genuine green dress shirts, zero novelty; "white shirt" → 12 genuine white shirts; **69/69 tests PASS** (test_phase7 = 35, all phases via `python -m unittest discover -s tests -p "test_phase*.py"`)
- test_phase4 setUpClass error is pre-existing/unrelated (missing kaggle/output chunks dir when run standalone)

**Notes:**
- Catalog = 371,902 men's shirts/tops ONLY (Phase 1 filter) — no pants/shoes/women's; expanding needs re-embedding all 22M products on Kaggle (~6h), not yet done
- Queries like "i want black tshirt with white layered shirt" work but are vague; specific colors/types/brands work best

---

## Phase 3 — Implementation Details (done)

**Files:**
- `src/phase3_text.py` — streams 6.62 GB reviews gz, filters to catalog parent_asins, joins review texts to products → `data/processed/product_texts.parquet`
- `tests/test_phase3.py` — 5 unit tests, ALL PASS (15 total across phases)

**Results (real data):**
- 2.84M matched reviews (of 22.6M total) for our catalog in 392s (~6.5 min)
- 1.098M review texts kept for 371,834 products (371,643 with non-empty reviews)
- Saved: `data/processed/product_texts.parquet` (371,902 rows: parent_asin, title, reviews_text, full_text)
- Avg full_text = 948 chars, max = 30,444 chars

**Design notes:**
- MAX_REVIEWS_PER_PRODUCT = 20 (top reviews per product, keeps embeddings tractable)
- MAX_REVIEW_CHARS = 2000 per review
- full_text = description + features + reviews (paper's Phase 3 formula)
- Verified sample: Nike Golf polo full_text contains description + real review ("...very sharp and fits beautifully")
- CLI: `python src\phase3_text.py --sample <ASIN>`

---

## Phase 2 — Implementation Details (done)

**Data source:** 0-core CSV (`documents\Clothing_Shoes_and_Jewelry.csv.gz`, **1,389,718,806 bytes verified**, 65,179,587 rows) — downloaded via download manager to `C:\AmazonData` then moved to documents (OneDrive interference killed the first 2 attempts).

**Files:**
- `src/phase2_ibcf.py` — streams CSV, filters rows to catalog parent_asins, groups by user_id, counts item→item pairs → edges parquet
- `tests/test_phase2.py` — 6 unit tests, ALL PASS (10 total with Phase 1)

**Results (real data):**
- 65.18M rows read in 86s; 2.8M matched our 371K-product catalog; 2.08M shoppers with ≥1 catalog product
- **1,621,543 co-purchase edges** (weight = # shoppers who bought both products, max weight 320)
- 211,000 products have ≥1 edge
- Saved: `data/processed/copurchase_edges.parquet`

**Demo verified:**
- Query "nike shirts large blue" → top = B00H8Y09CA (Nike Golf polo, score 0.87)
- Expansion of highly-connected B00KBZOB40: Hanes t-shirt packs (weight 132), Gildan cotton tees (102), Hanes Sport (72), Ecosmart hoodie (70), Gildan long sleeve (64), Dickies crew tee (64) — real co-purchase patterns
- CLI: `python src\phase2_ibcf.py --expand B00KBZOB40 --top-k 10`
- `src/phase2_demo.py` = Phase1+Phase2 end-to-end demo

**Design notes:**
- MAX_PRODUCTS_PER_USER = 50 cap (stops heavy users flooding the graph)
- Edges at parent_asin level (asin would fragment variants and dilute weights)
- 0-core keeps ALL users (5-core would delete quiet buyers = fewer links); noise handled by ≥2-products-per-user rule

---

## Phase 1 — Implementation Details (done)

**Files:**
- `src/config.py` — paths, TOP_K=100, category tokens
- `src/data_loader.py` — streams 4GB gz line-by-line, filters men's shirts → cached parquet (chunked, low-memory)
- `src/preprocessing.py` — clean text, extract size/color from title, build metadata_string
- `src/content_filter.py` — TF-IDF + cosine similarity → top K
- `src/main.py` — CLI: `python src/main.py --brand nike --category shirts --size large --color blue --top-k 10`
- `tests/test_phase1.py` — 4 unit tests, ALL PASS (`python -m unittest discover -s tests -v` from `scratch/`)

**Results:**
- 371,902 men's shirt products extracted (women's leakage bug fixed)
- Cached: `data/processed/men_shirts_metadata.parquet` (371,219 after preprocess)
- Query "nike shirts large blue" → 0.87 similarity; "hanes xl black" → 0.92
- Pipeline: load ~12-20s + fit ~4-7s + query ~0.14s

**Bugs found & fixed:**
1. "women" contains "men" → exact category token match
2. `\bs\b` matched "s" in "men's" → apostrophe-guard tokenization
3. Zero-score queries now show WARNING instead of printing 100 worthless products

---

## Run Commands
```powershell
cd C:\Users\thopa\OneDrive\Desktop\project\scratch
python src\main.py --brand nike --category shirts --size large --color blue --top-k 10
python -m unittest discover -s tests -v
```

---

## Environment
- Python with pandas 3.0.3, sklearn 1.8.0, numpy 2.5.0, scipy 1.17.1, pyarrow 23.0.1 (no pytest)
- 16GB RAM (build code is chunked to avoid OOM — full-subset load crashed before)
- Windows, PowerShell
- torch 2.10.0+cpu (NO CUDA/GPU), sentence-transformers 5.7.0 INSTALLED (models cached in ~/.cache/huggingface: sentence-t5-xl + all-distilroberta-v1), flair 0.15.1 present, h5py 3.14 + tf-keras (transformers 6.x TF import quirk)
- Dataset site: https://amazon-reviews-2023.github.io/ (0-core page: benchmark/0core/rating_only/)

## Resuming (tomorrow)
1. Session continues automatically via this opencode session; if closed, use `/sessions` (session under "global" project since Desktop\project is not a git repo) or `opencode -c`.
2. Read this PROGRESS.md — it's the full text backup.
3. **All 7 phases + website COMPLETE (69/69 tests).** To restart the demo: use the HF_HUB_OFFLINE wrapper command in the Website section → http://127.0.0.1:8000. Optional next steps: tune fusion weights, Flair vs star-rating eval, Amazon links on cards, "people also bought" (Phase 2 edges), dark theme, expand catalog beyond men's shirts (~6h Kaggle re-embed). PPT: user preparing slides — intro/requirement/design/development/status/lessons/deployment one-liners delivered (12 Aug).

## Today's session log (9 Aug 2026)
- Downloaded + verified 0-core CSV (1.29GB, 65.18M rows) — user's download manager to C:\AmazonData (OneDrive broke 2 earlier attempts), moved to documents
- Phase 2 built: 1,621,543 co-purchase edges in 86s → copurchase_edges.parquet; demo verified (Hanes/Gildan packs appear as top neighbors)
- Phase 3 built: full_text for all 371,902 products in 6.5 min → product_texts.parquet; sample verified (Nike Golf polo)
- 15/15 unit tests pass (`python -m unittest discover -s tests -v` from scratch/)
- Samples kept: data/sample_metadata_records.json, data/sample_reviews_records.json, data/sample_zerocore_records.csv
- Extra files: src/phase2_demo.py, src/phase3_proof.py (phase3_proof can be deleted; was a one-off proof)

## Today's session log (10 Aug 2026)
- Kaggle Phase 4 full run started (371,902 products, T4 GPU): models loaded, encoding chunk 0 — awaiting first chunk line + ETA
- Sample-test bug found & fixed: sample wrote chunk 0 into the default output dir → full run skipped products 1-10,000; fixed by deleting output/ and re-running
- Added `tests/test_phase4.py` — validates downloaded chunks (38 files, 371,902 rows, 1536 dims, float32, ASIN order vs product_texts.parquet)
- PROGRESS.md corrected: cascade = 1536 dims (768 T5 + 768 RoBERTa), ~2.3 GB output (was wrongly logged as 1792 / 2.7 GB)
- **Phase 4 DONE**: Kaggle run finished (345 min), zip download saga (2 broken zips — re-zipped on Kaggle + FileLink), extracted 38 chunks to scratch/kaggle/output/, tests 20/20 PASS (test_float32 fixed: embeddings are non-unit norm ~1.41 by design)
- **Phase 5 DONE**: sentence-transformers 5.7.0 installed locally (+h5py, tf-keras for transformers 6 import quirk); index built (371,902×1536, 2.28GB, 11.1s); first live query in ~5s total — top hits make semantic sense; tests 28/28 PASS
- **Phase 6 STARTED**: reviews_individual.parquet built (1,098,446 reviews, 298s); tests 34/34 PASS; Kaggle scorer script written
- **Phase 6 UNBLOCKED (11 Aug)**: found flair's real download server = nlp.informatik.hu-berlin.de (public, 200 OK) — the 401 was only the HF mirror. DECISION: Flair `sentiment` (distilbert, Amazon-domain trained). Local proof PASSED: 1,000 reviews in 91s CPU, spot-checks accurate (cheap fabric→0.0, brand new soft→1.0). Ready for Kaggle full run (88.8 MB upload).
- **Phase 6 DONE + Phase 7 DONE (11 Aug)**: overnight Kaggle run scored all 1,098,446 reviews in 36.6 min (504 rev/s); 371,643 products scored; file downloaded & verified. Phase 7 fusion engine built (11 tests) + real query run: sentiment visibly improves ranking (well-reviewed shirts beat 1-review novelty tees). **45/45 tests PASS. PROJECT COMPLETE.**
- **WEBSITE DONE (11 Aug)**: FastAPI + web UI at 127.0.0.1:8000; product_details.parquet extracted (rating/price/image); 3 quality filters added after user testing (color, shirt-vs-tee type, accessory removal); **61/61 tests PASS** — see "Website" section
- **COLOR BOUNDARY + QUALITY AUDIT (12 Aug)**: brand-strip color fix ("White Water" brand no longer matches "white"); COLOR_NAME_COLLISIONS (US Navy/Charlie Brown/Walter White/White House...); 10-color audit → **0 tee-leaks all colors**; novelty word list expanded + `_word_matches()` plural-tolerant regex (shamrocks/patricks fix); "a-shirt" added to tank markers; server crash (c10.dll OOM) fixed via `del df + gc.collect()` in engine init; **69/69 tests PASS** — final verified queries: "green shirt" = 6 genuine green dress shirts, "white shirt" = 12 genuine white shirts
## Session 2026-09-01 - Phase 7 hybrid upgrade (collab signal + MMR + eval)

### src/phase7_fusion.py
- Fixed corrupted fuse() definition (unmatched-paren syntax error) - file parses again.
- Added W_COLLAB_DEFAULT = 0.15; fuse() takes w_collab + collab_scores, emits a 4th signal column.
- New build_seed_scores(): top-25 content/embedding hits -> normalized seeds for the graph walk.
- New apply_mmr(): MMR diversity re-rank (lam 0.72, pool 50) with Jaccard title similarity.
- apply_type_preference(fallback=): strict mode (web first pass) returns EMPTY on zero matches so the web layer widens + retries; default falls back unchanged (last resort).
- build_seed_scores() guarded for empty inputs (unit-tested).

### src/web_app.py
- Engine loads the co-purchase graph at startup; recommend() computes collab scores per query.
- Empty strict-pass retry: if color/type filters empty the pool, widen candidates once + retry.
- apply_quality_rerank now uses Bayesian shrinkage (prior 3.9 stars, c=25) instead of raw avg blend.
- New GET /api/collab/{asin} endpoint - People-also-bought strip data.
- /api/recommend accepts and echoes w_collab.

### web/index.html
- 4th weight slider (collab, default 0.15) + updated footer formula.
- People also bought strip under the grid (fetches /api/collab for the top hit).

### tests/test_phase7.py
- +13 tests (collab fuse, seed scores, MMR, type fallback) -> 48/48 pass.

### scripts/eval_harness.py (new)
- Offline eval: cached stage-A (content+embedding per query), grid-search fusion weights, NDCG@10 / P@10 with title-graded relevance over 10 landmark queries.
- Run: set HF_HUB_OFFLINE=1 && python scripts/eval_harness.py --grid quick
- Outputs results/eval_results.json (best weights) + results/eval_cache.json.
- 3-query quick grid launched in background (results/eval_run.log).
## Session 2026-09-01 (accuracy + crash fixes)
- Fixed crash: concurrent request during warm-up built a 2nd Engine (double model load, OOM, server death). Now lock-guarded get_engine(); requests wait for warm-up instead.
- Accuracy: fuse(normalize=True) scales content/embed scores by their max so TF-IDF is no longer drowned by cosine scale; web_app opts in. UI score bars are now comparable.
- apply_type_preference: plain "shirt" queries now exclude polos/henleys (POLO_MARKERS); explicit "polo shirt" queries keep them.
- NOVELTY_WORDS += band/biker merch brands (fender, gibson, telecaster, rockabilly, harley, guitar, ...).
- Verified: 48/48 unittest OK; scripts/sanity_fix_check.py all pass.
