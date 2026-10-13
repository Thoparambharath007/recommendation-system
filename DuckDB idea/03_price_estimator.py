"""03_price_estimator.py - price for ALL 50,224 verified shirts, no lies.

Sources ONLY from our own project data (Amazon Reviews 2023 lineage):
  1. exact-metadata : price already in shirts_final.parquet (Amazon 2023)
  2. estimated-brand: brand (store) median from that brand's own priced siblings
                      + p25-p75 band + confidence tier

Honesty gate: 2-fold holdout - hide half the exact prices, estimate them from
the other half, report error. The estimator ships with a report card.

Output: data/shirts_priced.parquet (+ .csv human copy, house convention)
"""
import random
from pathlib import Path

import duckdb
import pandas as pd

HERE = Path(__file__).parent
SHIRTS = HERE / "data" / "shirts_final.parquet"
OUT_PARQUET = HERE / "data" / "shirts_priced.parquet"
OUT_CSV = HERE / "data" / "shirts_priced.csv"

INR_PER_USD = 83.0  # documented snapshot rate for query conversion

con = duckdb.connect()
con.execute(f"CREATE TABLE shirts AS SELECT * FROM read_parquet('{SHIRTS.as_posix()}')")

# ---- exact layer: Amazon-2023 metadata prices ONLY ----
con.execute("""
    CREATE TABLE exact AS
    SELECT parent_asin,
           NULLIF(price, 0) AS price_exact,
           CASE WHEN price > 0 THEN 'exact-metadata' END AS price_source
    FROM shirts
""")

# ---- 2. holdout report card (2-fold by brand-median, hide half the prices) ----
rows = con.execute("SELECT parent_asin, s.store, price_exact FROM exact e JOIN shirts s USING (parent_asin) WHERE price_exact IS NOT NULL").fetchdf()
random.seed(42)
rows["fold"] = [i % 2 for i in random.sample(range(len(rows)), len(rows))]
con.register("holdout", rows)
card = con.execute("""
    WITH
         -- brand median from the OTHER fold only:
         other AS (SELECT h.store, h.fold, MEDIAN(h.price_exact) AS m_pred,
                  QUANTILE_CONT(h.price_exact, 0.25) AS lo, QUANTILE_CONT(h.price_exact, 0.75) AS hi
                  FROM holdout h GROUP BY h.store, h.fold),
         pred AS (SELECT ho.parent_asin, ho.price_exact AS truth, op.m_pred, op.lo, op.hi
                  FROM holdout ho JOIN other op ON ho.store = op.store AND ho.fold != op.fold)
    SELECT COUNT(*) AS tested,
           ROUND(AVG(ABS(m_pred - truth)), 2)                     AS mae_usd,
           ROUND(100.0 * COUNT(*) FILTER (WHERE ABS(m_pred - truth) <= 10) / COUNT(*), 1) AS pct_within_10usd,
           ROUND(100.0 * COUNT(*) FILTER (WHERE truth BETWEEN lo AND hi) / COUNT(*), 1)   AS pct_inside_band
    FROM pred
""").fetchdf()
print("\n=== HOLDOUT REPORT CARD (brand-median estimator, prices hidden) ===")
print(card.to_string(index=False))

# ---- 3. final estimator on ALL exact prices ----
con.execute("""
    CREATE TABLE brand_stats AS
    SELECT s.store, MEDIAN(e.price_exact) AS m,
           QUANTILE_CONT(e.price_exact, 0.25) AS lo,
           QUANTILE_CONT(e.price_exact, 0.75) AS hi,
           COUNT(*) AS n
    FROM exact e JOIN shirts s USING (parent_asin)
    WHERE e.price_exact IS NOT NULL
    GROUP BY s.store
""")
out = con.execute(f"""
    COPY (
        SELECT s.parent_asin, s.title, s.store, s.sleeve, s.average_rating,
               s.rating_number, s.image_url, s.vision_class, s.shirt_prob,
               e.price_exact,
               COALESCE(e.price_exact, b.m, (SELECT MEDIAN(price_exact) FROM exact)) AS price_usd,
               CASE WHEN e.price_exact IS NOT NULL THEN NULL
                    WHEN b.n >= 20 THEN b.lo WHEN b.n >= 1 THEN b.lo
                    ELSE (SELECT QUANTILE_CONT(price_exact, 0.25) FROM exact) END AS band_lo,
               CASE WHEN e.price_exact IS NOT NULL THEN NULL
                    WHEN b.n >= 20 THEN b.hi WHEN b.n >= 1 THEN b.hi
                    ELSE (SELECT QUANTILE_CONT(price_exact, 0.75) FROM exact) END AS band_hi,
               COALESCE(e.price_source,
                        CASE WHEN b.n >= 20 THEN 'estimated-brand-high'
                             WHEN b.n >= 1  THEN 'estimated-brand-med'
                             ELSE 'estimated-category-low' END) AS price_source,
               ROUND(COALESCE(e.price_exact, b.m, (SELECT MEDIAN(price_exact) FROM exact)) * {INR_PER_USD}) AS price_inr_est
        FROM shirts s
        LEFT JOIN exact e USING (parent_asin)
        LEFT JOIN brand_stats b ON s.store = b.store
    ) TO '{OUT_PARQUET.as_posix()}' (FORMAT PARQUET)
""")
con.execute(f"COPY (SELECT * FROM read_parquet('{OUT_PARQUET.as_posix()}')) TO '{OUT_CSV.as_posix()}' (FORMAT CSV)")

df = pd.read_parquet(OUT_PARQUET)
print(f"\n=== FINAL CATALOG: {len(df):,} shirts, price coverage: {df.price_usd.notna().mean()*100:.1f}% ===")
print(df.price_source.value_counts().to_string())
print(f"\nsample rows:")
print(df[df.price_source.str.startswith('estimated')][['store', 'price_usd', 'band_lo', 'band_hi', 'price_source']].head(5).to_string(index=False))
print(f"\nsaved: {OUT_PARQUET.name} + {OUT_CSV.name}")
