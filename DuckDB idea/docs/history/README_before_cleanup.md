# DuckDB Idea — the query layer HQ

## Current application (2026-10-13)

The master now contains the completed Kaggle v3.2 color/pattern predictions for all
50,224 product IDs, with uncertainty and original-attribute provenance preserved.
49,658 products have nonempty colors; 41,752 have known patterns. The original
pre-merge Parquet is preserved in `data/backups`.

Double-click `Start_Query_App.bat` and open http://127.0.0.1:8010.
The app implements natural language -> LLM-generated SQL -> validation -> DuckDB
on `data/shirts_master.parquet` -> ranked product cards. `.env` is prepared for
NVIDIA's hosted Nemotron 3.5 Lightning model. Live SQL generation has been tested
against the merged catalog; the API key is stored in the ignored local `.env`.
See [QUERY_APP.md](QUERY_APP.md) for configuration, restrictions, ranking and tests.

The historical notes below describe the earlier dataset-building stage.

**Purpose:** build the proposed solution: SQL directly on the vision-verified
shirt catalog (parquet, no server), later wrapped by an LLM text-to-SQL layer.

## House rules (agreed 2026-10-04)
1. ALL new work happens inside this folder.
2. Verification/scratch outputs go to `..\JUNK created by qoder\` and are
   referenced from here — never litter this folder.
3. `data\` copies are working replicas; the canonical archive stays in
   `..\segregation\verified\`.

## Layout
```
DuckDB idea\
├── data\        shirts_final.parquet (50,224 verified) + rejects + CV ledger
├── sql\         saved .sql queries (reusable, citable in the SRS)
├── notebook\    exploration notebooks (later)
└── 01_first_query.py   smoke test: DuckDB on the parquet
```

## The catalog being queried
50,224 men's shirts, each confirmed by CLIP ViT-B/32 (shirt_prob >= 0.5)
from 63,190 rule-segregated candidates out of 7.2M Amazon metadata records.
Columns: shirt_id, parent_asin, title, store, price, average_rating,
rating_number, sleeve, confidence, signals, categories_path, image_url,
+ vision verdict columns (shirt_prob, vision_class, vision_sleeve, ...).

## Next milestones
- [x] DuckDB running directly on shirts_final.parquet
- [ ] reviews_v2 (verified-purchase filter) -> shirts_master.parquet
- [ ] sentiment join: ORDER BY rating*0.5 + sentiment*0.5
- [ ] LLM -> SQL natural-language layer
- [ ] eval metrics (Recall@10, NDCG@10, MRR)
