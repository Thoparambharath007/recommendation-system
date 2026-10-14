"""02_build_asin_bridge.py - the child<->parent ASIN bridge for price joining.

The raw Amazon Reviews 2023 file carries BOTH 'asin' (child variant) and
'parent_asin' (product family) on every review record. We exploit that to
build the mapping table that lets Kaggle child-ASIN prices reach our
parent-ASIN-verified 50k shirts.

Output (DuckDB idea/data/):
  asin_bridge.parquet  - distinct child -> parent pairs (verified parents only)
  asin_bridge.csv      - human-verifiable copy (house convention)
"""
import gzip
import json
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
RAW = HERE.parent / "datasets" / "Clothing_Shoes_and_Jewelry.jsonl.gz"
OUT_DIR = HERE / "data" / "intermediate"

t0 = time.time()
verified = set(pd.read_parquet(OUT_DIR / "shirts_final.parquet")["parent_asin"])
print(f"verified parents to map: {len(verified):,}")

pairs = {}          # child asin -> parent asin (dict = free dedupe)
seen_parents = set()
lines = 0

with gzip.open(RAW, "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        lines += 1
        if lines % 2_000_000 == 0:
            el = time.time() - t0
            print(f"  {lines/1e6:.0f}M lines | parents hit {len(seen_parents):,} "
                  f"| pairs {len(pairs):,} | {el/60:.1f} min", flush=True)
        # cheap pre-filter before json parsing
        if '"parent_asin"' not in line or '"asin"' not in line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = r.get("parent_asin")
        if p in verified:
            c = r.get("asin")
            if c and c != p:
                pairs[c] = p
            seen_parents.add(p)

df = pd.DataFrame(sorted(pairs.items()), columns=["child_asin", "parent_asin"])
df.to_parquet(OUT_DIR / "asin_bridge.parquet", index=False)
df.to_csv(OUT_DIR / "asin_bridge.csv", index=False)

el = time.time() - t0
print("=" * 60)
print(f"scanned lines          : {lines:,}")
print(f"verified parents with >=1 child mapped: {len(seen_parents):,} / {len(verified):,}")
print(f"distinct child->parent pairs          : {len(df):,}")
print(f"children per parent (mean)            : {len(df)/max(len(seen_parents),1):.1f}")
print(f"elapsed                               : {el/60:.1f} min")
print("saved: data/asin_bridge.parquet + asin_bridge.csv")
