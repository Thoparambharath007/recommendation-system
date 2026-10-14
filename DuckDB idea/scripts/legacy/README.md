# Historical catalog-building scripts

The numbered scripts document the original pipeline: screening queries, ASIN
bridge, price estimation, attributes, reviews, master assembly and early vision fills.
They are not used when starting the app. Paths were adjusted after moving here.
Build inputs/outputs use `data/intermediate/`; the price model uses `data/models/`.
Source metadata and reviews remain in the parent project's folders.

From the project folder, run a job explicitly, for example:
`python scripts/legacy/01_first_query.py`.

Several jobs perform large scans or training. Master builders produce an older
schema without current color, pattern, size and sleeve enrichment. Keep the
current `data/shirts_master.parquet` as the live catalog.
