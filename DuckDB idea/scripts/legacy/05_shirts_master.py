"""05_shirts_master.py - build the FIRST shirts_master.parquet.

Stitches (read-only) the frozen base table with Job A's mined attributes and
reserves empty slots for the attributes still to come (color, pattern):

    shirts_master = shirts_priced (frozen base, 10 cols)
                  + fabric, fit_type, pocket          (Job A, local text)
                  + color, pattern                     (reserved = NA, filled by Job B)

Base table is NEVER modified. Output: data/shirts_master.parquet (+ .csv).
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
BASE = HERE / "data" / "intermediate" / "shirts_priced.parquet"
ATTR = HERE / "data" / "intermediate" / "shirt_attributes.parquet"
OUT_PQ = HERE / "data" / "intermediate" / "shirts_master.parquet"
OUT_CSV = HERE / "data" / "intermediate" / "shirts_master.csv"

COLUMNS = ["parent_asin", "title", "store", "sleeve", "average_rating",
           "rating_number", "image_url", "price_usd", "price_source",
           "price_inr_est", "color", "pattern", "fabric", "fit_type", "pocket"]


def main():
    base = pd.read_parquet(BASE)
    attr = pd.read_parquet(ATTR)[["parent_asin", "fabric", "fit_type", "pocket"]]

    m = base.merge(attr, on="parent_asin", how="left")
    m["color"] = pd.NA      # reserved for Job B (Kaggle CLIP)
    m["pattern"] = pd.NA    # reserved for Job B (Kaggle CLIP)
    m = m[COLUMNS]

    m.to_parquet(OUT_PQ, index=False)
    m.to_csv(OUT_CSV, index=False)

    print("===== SHIRTS MASTER (first build) =====")
    print(f"rows: {len(m):,}   cols: {len(m.columns)}  ->  {OUT_PQ.name} (+ .csv)\n")
    print("coverage per column:")
    for c in COLUMNS:
        filled = m[c].notna().sum()
        pct = filled / len(m) * 100
        tag = "" if filled else "  <- reserved (filled later)"
        print(f"  {c:15s} {filled:>7,}  {pct:5.1f}%{tag}")
    print("\npocket breakdown:")
    print(m["pocket"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
