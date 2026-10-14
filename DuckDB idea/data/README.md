# Project data

The app reads `shirts_master.parquet` and `product_details.parquet` here.

- `intermediate/`: earlier pipeline stages and historical CSV exports.
- `models/`: saved price model.
- `reports/`: enrichment and merge summaries.
- `exports/`: current CSV exports, created when maintenance scripts run.
- `backups/`: recovery snapshots, intentionally retained.

`intermediate/shirts_master.csv` predates later enrichment and is not the current
database. Use the live Parquet when analyzing the final catalog.
