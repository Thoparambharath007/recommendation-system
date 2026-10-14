"""01_first_query.py - the founding smoke test of the DuckDB idea.

SQL runs DIRECTLY on the parquet file: no import, no server, no database.
This is the architecture as proposed: DuckDB + shirts_final.parquet.
"""
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parents[2]
SHIRTS = HERE / "data" / "intermediate" / "shirts_final.parquet"

con = duckdb.connect()  # in-memory, zero setup

print("=" * 62)
print("FOUNDING QUERIES ON THE VISION-VERIFIED SHIRT CATALOG")
print("=" * 62)

q1 = f"""
    SELECT COUNT(*)                        AS total_shirts,
           COUNT(DISTINCT store)           AS brands,
           ROUND(AVG(average_rating), 2)   AS avg_rating,
           ROUND(MEDIAN(price), 2)         AS median_price
    FROM read_parquet('{SHIRTS.as_posix()}')
"""
print("\n[1] catalog overview:\n")
print(con.execute(q1).fetchdf().to_string(index=False))

q2 = f"""
    SELECT sleeve, COUNT(*) AS n,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
    FROM read_parquet('{SHIRTS.as_posix()}')
    GROUP BY sleeve ORDER BY n DESC
"""
print("\n[2] sleeve census (rules, cross-checked by CLIP):")
print(con.execute(q2).fetchdf().to_string(index=False))

q3 = f"""
    SELECT title, store, price, average_rating, rating_number, sleeve, shirt_prob
    FROM read_parquet('{SHIRTS.as_posix()}')
    WHERE sleeve = 'half' AND price > 0 AND price < 30
      AND average_rating >= 4.4 AND rating_number > 100
    ORDER BY rating_number DESC
    LIMIT 5
"""
print("\n[3] THE DEMO QUERY - 'half sleeve, under $30, beloved, proven':")
out = con.execute(q3).fetchdf()
for _, r in out.iterrows():
    print(f"  {str(r['title'])[:58]:60s} {r['store'][:18]:18s} "
          f"${r['price']:<7} {r['average_rating']}x{r['rating_number']}")

q4 = f"""
    SELECT vision_class, COUNT(*) AS n
    FROM read_parquet('{SHIRTS.as_posix()}')
    GROUP BY vision_class ORDER BY n DESC LIMIT 3
"""
print("\n[4] every row above carries a CLIP vision verdict:")
print(con.execute(q4).fetchdf().to_string(index=False))
print("\nDuckDB is live on the verified catalog. Welcome to the HQ.")
