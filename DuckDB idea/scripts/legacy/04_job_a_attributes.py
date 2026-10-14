"""04_job_a_attributes.py - Job A: mine fabric, fit_type, pocket from LOCAL text.

Sources (all on disk - no GPU, no Kaggle):
  - base catalog : DuckDB idea/data/shirts_priced.parquet  (the 50,224 universe + title)
  - text witness : data/processed/men_shirts_metadata.parquet (features + description)

Honest rules:
  * fabric / fit_type : keyword match -> a value, else NULL (unknown, never guessed)
  * pocket            : 3-class from text -> 'no' ("no pockets"), 'double'
                        ("two/dual/2 ... pockets"), 'single' (singular "pocket"),
                        else 'unknown' (silent or bare plural). The calibrated CLIP
                        photo voter later resolves only the 'unknown' rows.
  * every attribute carries a *_source label for provenance.

Output: data/shirt_attributes.parquet (+ .csv)  keyed on parent_asin
        -> merged into shirts_master.parquet later (base table stays frozen).
"""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[2]
BASE = HERE / "data" / "intermediate" / "shirts_priced.parquet"
META = HERE.parent / "data" / "processed" / "men_shirts_metadata.parquet"
OUT_PQ = HERE / "data" / "intermediate" / "shirt_attributes.parquet"
OUT_CSV = HERE / "data" / "intermediate" / "shirt_attributes.csv"

# priority order = specific -> generic (first hit wins)
FABRIC = ['linen', 'silk', 'flannel', 'denim', 'chambray', 'seersucker', 'poplin',
          'oxford', 'twill', 'canvas', 'wool', 'cashmere', 'velvet', 'satin',
          'georgette', 'rayon', 'viscose', 'modal', 'lyocell', 'bamboo', 'jersey',
          'fleece', 'terry', 'corduroy', 'chiffon', 'lace', 'polyester', 'nylon',
          'spandex', 'elastane', 'lycra', 'cotton']
FIT = ['extra slim fit', 'super slim fit', 'skinny fit', 'slim fit', 'tailored fit',
       'athletic fit', 'classic fit', 'regular fit', 'relaxed fit', 'loose fit',
       'straight fit', 'modern fit', 'muscle fit', 'standard fit', 'heritage fit',
       'oversized', 'tapered', 'fitted']
# NOTE: bare words like 'slim'/'classic' were REMOVED after an audit showed they
# over-matched ("classic collar", "Standard-Fit") for ~5,900 rows. Only explicit
# "<x> fit" phrases (hyphen or space) are trusted.

FABRIC_RX = {w: re.compile(r'\b' + re.escape(w) + r'\b') for w in FABRIC}
FIT_RX = {w: re.compile(r'\b' + w.replace(' ', r'[-\s]') + r'\b') for w in FIT}
POCKET_RX = re.compile(r'\bpockets?\b')
SINGLE_RX = re.compile(r'\bpocket\b')   # singular noun only; 'pockets' will NOT match
DOUBLE_RX = re.compile(r'\b(?:two|2|three|3|dual|twin|double)\b[^.]{0,25}\bpockets?\b')
NEG_POCKET_RX = re.compile(r'\bno (?:chest )?pockets?\b|\bwithout (?:a |any )?pockets?\b')


def _first(text, rxs):
    for w, r in rxs.items():
        if r.search(text):
            return w
    return None


def _all(text, rxs):
    return ";".join(w for w, r in rxs.items() if r.search(text))


def _source(word, fields):
    if not word:
        return ""
    r = re.compile(r'\b' + re.escape(word) + r'\b')
    for name, txt in fields:
        if r.search(txt):
            return name
    return ""


def _pocket(text):
    """3-class from text: 'no' (negation), 'double' (two/2/dual/double...), 'single'
    (singular 'pocket'), else 'unknown' (silent or bare plural -> CLIP resolves)."""
    if NEG_POCKET_RX.search(text):
        return "no"
    if DOUBLE_RX.search(text):
        return "double"
    if SINGLE_RX.search(text):
        return "single"
    return "unknown"


def _pocket_src(p, fields):
    rx = {"no": NEG_POCKET_RX, "double": DOUBLE_RX, "single": SINGLE_RX}.get(p)
    if rx is None:
        return ""
    for name, txt in fields:
        if rx.search(txt):
            return name
    return ""


def main():
    base = pd.read_parquet(BASE)[["parent_asin", "title"]]
    meta = pd.read_parquet(META)[["parent_asin", "features", "description"]]
    df = base.merge(meta, on="parent_asin", how="left")

    title = df["title"].fillna("").astype(str).str.lower()
    feat = df["features"].fillna("").astype(str).str.lower()
    desc = df["description"].fillna("").astype(str).str.lower()
    blob = title + " " + feat + " " + desc
    fields_iter = list(zip(title, feat, desc))

    fab_word = [_first(b, FABRIC_RX) for b in blob]
    fit_word = [_first(b, FIT_RX) for b in blob]
    pocket_flag = [_pocket(b) for b in blob]

    out = pd.DataFrame({
        "parent_asin": df["parent_asin"],
        "fabric": [w if w else pd.NA for w in fab_word],
        "fabric_all": [_all(b, FABRIC_RX) for b in blob],
        "fabric_source": [_source(w, [("title", t), ("features", f), ("description", d)])
                          for w, (t, f, d) in zip(fab_word, fields_iter)],
        "fit_type": [w if w else pd.NA for w in fit_word],
        "fit_source": [_source(w, [("title", t), ("features", f), ("description", d)])
                       for w, (t, f, d) in zip(fit_word, fields_iter)],
        "pocket": pocket_flag,
        "pocket_source": [_pocket_src(p, [("title", t), ("features", f), ("description", d)])
                          for p, (t, f, d) in zip(pocket_flag, fields_iter)],
    })

    out.to_parquet(OUT_PQ, index=False)
    out.to_csv(OUT_CSV, index=False)

    n = len(out)
    print("===== JOB A ATTRIBUTES (local text mining) =====")
    print(f"rows: {n:,}  ->  {OUT_PQ.name} (+ .csv)\n")
    print(f"fabric    filled : {(out['fabric'].notna()).sum():,} ({out['fabric'].notna().mean()*100:.1f}%)   unknown: {(out['fabric'].isna()).sum():,}")
    print(f"fit_type  filled : {(out['fit_type'].notna()).sum():,} ({out['fit_type'].notna().mean()*100:.1f}%)   unknown: {(out['fit_type'].isna()).sum():,}")
    print("pocket    counts :")
    print(out["pocket"].value_counts().to_string())
    print("\ntop fabric values:")
    print(out["fabric"].value_counts().head(8).to_string())
    print("\ntop fit_type values:")
    print(out["fit_type"].value_counts().head(8).to_string())
    print("\nsample rows:")
    print(out[out["fabric"].notna() & (out["pocket"] == "yes")].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
