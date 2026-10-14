# SmartShirts

Search a 50,224-product shirt catalog using natural language and optional filters.
The app translates required constraints into validated SQL, runs DuckDB over
Parquet, and ranks matches. Office and comfort wording are ranking preferences.
Results show one card per image URL, with recorded size/listing choices in the
details dialog. The 50,224 underlying listings are retained across 31,793 image
groups so size-specific evidence and prices are not lost.

## Start the app

Double-click **start.bat**, then use http://127.0.0.1:8010/.
Keep the server window open. An already-running server is reused.
The private `.env` holds API settings; `.env.example` is the public template.
See [QUERY_APP.md](QUERY_APP.md) for configuration, query rules and limitations.

## Folder guide

| Location | Contents |
|---|---|
| Root Python files | Live API, SQL generation/validation, ranking, startup and current maintenance modules |
| `web/` | Website HTML, JavaScript and CSS |
| `data/shirts_master.parquet` | Current enriched catalog |
| `data/product_details.parquet` | Original listing details and images |
| `data/models/` | Saved price-estimation model |
| `data/reports/` | Color/pattern, size/brand and sleeve enrichment summaries |
| `data/intermediate/` | Earlier build datasets, historical CSV exports and screening evidence |
| `data/exports/` | Fresh CSV exports, created when current maintenance scripts run |
| `data/backups/` | Recovery copies from before catalog enrichment |
| `scripts/check_live.py` | Optional real-provider smoke check |
| `scripts/legacy/` | Original numbered dataset-building scripts |
| `scripts/garment_verification/` | Dima806 Kaggle notebook, upload ZIP, varied sample and resume tools |
| `tests/` | Automated API, SQL, enrichment and ranking checks |
| `docs/srs/` | SRS Word/PDF, diagrams and generation sources |
| `docs/templates/` | Original college document template |
| `docs/verification/` | Verification notes and recorded live searches |
| `docs/history/` | Earlier plans, handoff and folder guide |
| `docs/Shirt_Recommendation_Project_Team_Guide.docx` | Simple guide for the five members |
| `ppt/` | Current presentation, original template and recovery backups |
| `.vendor/` | Local runtime dependency, retained for the app |

## Development and maintenance

Run commands from this folder:

```powershell
python launch.py
python -m unittest discover -s tests -v
python scripts/check_live.py
```

Current maintenance modules are `merge_vision.py`, `enrich_sleeves.py`, and
`product_enrichment.py`. They preserve backups before applying catalog changes.
Sleeve/size enrichment reads original evidence from the parent project's `data/`.
Kaggle scripts and source archives are also outside this app folder. Copying this
folder alone does not include all rebuild inputs.

Legacy jobs write older build outputs into `data/intermediate/`. Do not replace
the enriched live master with these outputs without the later enrichment stages.
`data/intermediate/shirts_master.csv` is historical and may differ from the current
Parquet. The app always reads the live Parquet files listed above.

## Limits

Color and pattern predictions can be wrong. Sizes are recorded listing mentions,
not live stock. Prices are historical or labeled estimates. Missing attributes
remain unknown; requested filters are not silently relaxed.

The 2026-10-14 cleanup removed generated caches and empty placeholders and grouped
files by purpose. Data, API settings, SRS and presentation contents were preserved.
[Cleanup manifest](docs/cleanup_manifest.json) records moves and removed folders.
# Active compact shirt catalog

The app now uses `data/shirts_catalog.parquet`: 19,834 shirt-candidate records,
one per image URL. `data/shirt_size_options.parquet` separately retains recorded
size/listing IDs, prices and other listing-specific fields. Filtering operates
on these options before one card per photo is selected, so size and price cannot
be matched from different options.

**`data/shirts_master.parquet` is preserved unchanged** with all 50,224 original
listings. Uncertain and non-shirt classifier results remain in the source result
archive; they are not part of the active compact catalog. Rebuild the compact
files with `python scripts/build_compact_catalog.py`, which checks source ID/data
alignment and the original master's hash before and after writing new files.

Click a product photo to open `/product/<parent_asin>` in a new tab. Its recorded
size buttons load the associated listing's information and price while retaining
the same primary photo. Missing sizes are shown as unavailable, never invented.
These image groups and size mentions are historical evidence, not confirmed
live Amazon variant families or stock. The classification remains model output.

