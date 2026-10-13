"""
07_finalize_master.py - FINAL master builder (supersedes 05 first-draft).

Stitches, with FULL provenance so trust levels are never hidden:
  base (frozen shirts_priced)  -> identity + price columns
  Job A shirt_attributes       -> fabric / fit_type / pocket  (+ *_source = text)
  Job C review_attributes      -> fills fabric/fit_type ONLY where text was silent,
                                  tagged source='review'; adds fit_feedback (+n)
  reserved                     -> color / pattern (NA, for Kaggle Job B)

Trust rule: a text value always beats a review value. Review-derived values are
explicitly labelled so the UI can show an "inferred" badge.

Output: data/shirts_master.parquet (+ .csv)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(r"c:\Users\thopa\OneDrive\Desktop\project\REAL SYSTEM")
DATA = ROOT / "DuckDB idea" / "data"
BASE = DATA / "shirts_priced.parquet"          # frozen
ATTR = DATA / "shirt_attributes.parquet"       # Job A
REV = DATA / "review_attributes.parquet"       # Job C
STYLE = DATA / "style_attributes.parquet"      # Job D
OUT_PQ = DATA / "shirts_master.parquet"
OUT_CSV = OUT_PQ.with_suffix(".csv")

FIT_MAP = {"slim": "slim fit", "regular": "regular fit", "relaxed": "relaxed fit"}

COLUMNS = [
    "parent_asin", "title", "store", "sleeve", "style", "average_rating", "rating_number",
    "image_url", "price_usd", "price_source", "price_inr_est",
    "color", "pattern",
    "fabric", "fabric_source",
    "fit_type", "fit_source", "fit_review_n",
    "pocket", "pocket_source",
    "fit_feedback", "fit_feedback_n", "review_count",
]


def main():
    base = pd.read_parquet(BASE)
    attr = pd.read_parquet(ATTR)[["parent_asin", "fabric", "fabric_source",
                                  "fit_type", "fit_source", "pocket", "pocket_source"]]
    rev = pd.read_parquet(REV)
    sty = pd.read_parquet(STYLE)[["parent_asin", "style"]]

    m = (base.merge(attr, on="parent_asin", how="left")
             .merge(rev, on="parent_asin", how="left")
             .merge(sty, on="parent_asin", how="left"))

    # review fill ONLY where text was silent
    need_fab = m["fabric"].isna() & m["fabric_review"].notna()
    m.loc[need_fab, "fabric"] = m.loc[need_fab, "fabric_review"]
    m.loc[need_fab, "fabric_source"] = "review"

    need_fit = m["fit_type"].isna() & m["fit_review"].notna()
    m.loc[need_fit, "fit_type"] = m.loc[need_fit, "fit_review"].map(FIT_MAP)
    m.loc[need_fit, "fit_source"] = "review"

    m["color"] = pd.NA
    m["pattern"] = pd.NA
    for c in ["fit_review_n", "fit_feedback_n", "review_count"]:
        m[c] = m[c].fillna(0).astype("int32")

    m = m[COLUMNS]
    m.to_parquet(OUT_PQ, index=False)
    m.to_csv(OUT_CSV, index=False)

    n = len(m)
    print("===== FINAL SHIRTS MASTER =====")
    print(f"rows: {n:,}   cols: {len(COLUMNS)}  ->  shirts_master.parquet (+ .csv)\n")
    fab_review = (m["fabric_source"] == "review").sum()
    print(f"fabric       filled: {m['fabric'].notna().sum():,} ({m['fabric'].notna().mean()*100:.1f}%)  "
          f"[review-added {fab_review:,}]")
    print(f"fit_type     filled: {m['fit_type'].notna().sum():,} ({m['fit_type'].notna().mean()*100:.1f}%)  "
          f"[review-added {(m['fit_source']=='review').sum():,}]")
    print(f"fit_feedback filled: {m['fit_feedback'].notna().sum():,} ({m['fit_feedback'].notna().mean()*100:.1f}%)")
    print(f"style        filled: {(m['style']!='other').sum():,} ({(m['style']!='other').mean()*100:.1f}%)  "
          f"other/unknown: {(m['style']=='other').sum():,}")
    print(f"pocket       filled: {m['pocket'].notna().sum():,} (100% - single/double/no/unknown)")
    print(m["pocket"].value_counts().to_string())
    print("\nfit_source breakdown:")
    print(m["fit_source"].value_counts(dropna=False).to_string())
    print("\nfabric_source breakdown:")
    print(m["fabric_source"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
