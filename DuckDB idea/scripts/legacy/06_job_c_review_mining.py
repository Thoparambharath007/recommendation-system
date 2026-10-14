"""
Job C (LOCAL, free) - mine the 1.1M reviews for signals the seller text lacks.

Produces review_attributes.parquet keyed on parent_asin:
  * fit_review      : slim / regular / relaxed  (consensus of review wording)
  * fit_feedback    : runs_small / true_to_size / runs_large  (NEW, user-valuable)
  * fabric_review   : majority fabric word mentioned in reviews
  * *_n             : number of agreeing reviews (confidence)
  * review_count

These are only USED to fill master rows where the trusted seller text was silent;
they are tagged source='review' so we never hide that they are inferred.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REV = ROOT / "data" / "processed" / "reviews_individual.parquet"
MASTER = ROOT / "DuckDB idea" / "data" / "intermediate" / "shirts_master.parquet"
OUT_PQ = ROOT / "DuckDB idea" / "data" / "intermediate" / "review_attributes.parquet"
OUT_CSV = OUT_PQ.with_suffix(".csv")

FABRIC_WORDS = ['linen', 'silk', 'flannel', 'denim', 'chambray', 'oxford', 'poplin',
                'twill', 'canvas', 'jersey', 'satin', 'velvet', 'wool', 'nylon',
                'polyester', 'rayon', 'cotton']

MIN_VOTES = 2      # need at least 2 agreeing reviews
MIN_SHARE = 0.60   # winner must be >=60% of votes and strictly ahead


def decide(counts: dict):
    """Return (label, n) if there is a clear majority, else (None, total)."""
    total = int(sum(counts.values()))
    if total < MIN_VOTES:
        return None, total
    top = max(counts, key=counts.get)
    others = total - counts[top]
    if counts[top] >= MIN_VOTES and counts[top] >= others and (counts[top] / total) >= MIN_SHARE:
        return top, counts[top]
    return None, total


def main():
    base = set(pd.read_parquet(MASTER)["parent_asin"])
    r = pd.read_parquet(REV)[["parent_asin", "review_text"]]
    r = r[r["parent_asin"].isin(base)].copy()
    txt = r["review_text"].fillna("").str.lower()

    cols = {
        "slim": txt.str.contains(r"\bslim\b|\bskinny\b|\bform[- ]fitting\b", regex=True),
        "regular": txt.str.contains(r"\bregular fit\b|\btrue to size\b|\bstandard fit\b|\bclassic fit\b", regex=True),
        "relaxed": txt.str.contains(r"\brelaxed\b|\bloose\b|\broomy\b|\boversized\b|\bbaggy\b", regex=True),
        "fb_small": txt.str.contains(r"runs small|too small|order.{0,8}size up|size up", regex=True),
        "fb_true": txt.str.contains(r"true to size|fits (?:perfect|well|great|nice)|as expected|fits good", regex=True),
        "fb_large": txt.str.contains(r"runs large|too big|too large|order.{0,8}size down", regex=True),
    }
    for w in FABRIC_WORDS:
        cols["fab_" + w] = txt.str.contains(r"\b" + w + r"\b", regex=True)

    flags = pd.DataFrame(cols).astype("int8")
    flags["parent_asin"] = r["parent_asin"].values
    agg = flags.groupby("parent_asin", as_index=False).sum()
    rc = r.groupby("parent_asin").size().rename("review_count").reset_index()
    agg = agg.merge(rc, on="parent_asin", how="left")

    fit_rows, fb_rows, fab_rows = [], [], []
    for _, row in agg.iterrows():
        fl, fn = decide({"slim": row["slim"], "regular": row["regular"], "relaxed": row["relaxed"]})
        fit_rows.append((fl, fn))
        fbmap = {"runs_small": row["fb_small"], "true_to_size": row["fb_true"], "runs_large": row["fb_large"]}
        fbl, fbn = decide(fbmap)
        fb_rows.append((fbl, fbn))
        fabcounts = {w: row["fab_" + w] for w in FABRIC_WORDS}
        fbl2, fn2 = decide(fabcounts)
        fab_rows.append((fbl2, fn2))

    out = pd.DataFrame({
        "parent_asin": agg["parent_asin"],
        "review_count": agg["review_count"],
        "fit_review": [x[0] for x in fit_rows],
        "fit_review_n": [x[1] for x in fit_rows],
        "fit_feedback": [x[0] for x in fb_rows],
        "fit_feedback_n": [x[1] for x in fb_rows],
        "fabric_review": [x[0] for x in fab_rows],
        "fabric_review_n": [x[1] for x in fab_rows],
    })
    out.to_parquet(OUT_PQ, index=False)
    out.to_csv(OUT_CSV, index=False)

    n = len(out)
    print("===== JOB C REVIEW MINING =====")
    print(f"products with >=1 review: {n:,}\n")
    print(f"fit_review   resolved : {out['fit_review'].notna().sum():,}")
    print(f"fit_feedback resolved : {out['fit_feedback'].notna().sum():,}")
    print(f"fabric_review resolved: {out['fabric_review'].notna().sum():,}")
    print("\nfit_review dist:")
    print(out["fit_review"].value_counts(dropna=False).to_string())
    print("\nfit_feedback dist:")
    print(out["fit_feedback"].value_counts(dropna=False).to_string())
    print("\nsample:")
    print(out[out["fit_review"].notna() | out["fit_feedback"].notna()].head(8).to_string(index=False))
    print(f"\nwrote -> {OUT_PQ.name} (+ .csv)")


if __name__ == "__main__":
    main()
