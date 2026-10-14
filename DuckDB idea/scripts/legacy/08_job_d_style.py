"""
Job D (LOCAL, free) - derive `style` from the category breadcrumb.

The categories field never contains fit, but its deepest non-generic segment
tells us the SHIRT STYLE (casual / formal / work / polo / flannel ...), which is
a real, ~100%-coverable recommendation signal.

Parse rule: split on '|', drop generic tokens, keep the deepest specific segment,
then map keywords -> controlled style vocabulary. Raw segment is kept for audit.

Output: data/style_attributes.parquet (+ .csv)  keyed on parent_asin
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MASTER = ROOT / "DuckDB idea" / "data" / "intermediate" / "shirts_master.parquet"
META = ROOT / "data" / "processed" / "men_shirts_metadata.parquet"
OUT_PQ = ROOT / "DuckDB idea" / "data" / "intermediate" / "style_attributes.parquet"
OUT_CSV = OUT_PQ.with_suffix(".csv")

GENERIC = {"clothing, shoes & jewelry", "clothing", "shoes & jewelry", "men",
           "apparel", "tops", "shirts", "men's clothing", "fashion"}

# ordered: specific styles first, generic catch-alls last
STYLE_RULES = [
    ("polo", r"\bpolo\b|\bknit shirt\b"),
    ("henley", r"\bhenley\b"),
    ("flannel", r"\bflannel\b|\bplaid\b"),
    ("denim", r"\bdenim\b|\bjean\b|\bchambray\b"),
    ("formal", r"\bdress\b|\bformal\b|\btuxedo\b|\bsuit\b|\bdress shirt"),
    ("work", r"\bwork\b|\butility\b|\bsafety\b|\buniform\b|\bindustrial\b|\bservice\b|\bchef\b|\bmilitary\b|\btactical\b"),
    ("performance", r"\bathletic\b|\bperformance\b|\bgolf\b|\bfishing\b|\boutdoor\b|\bsport\b|\bmoisture\b|\bhiking\b|\bcycling\b|\brunning\b"),
    ("sweater", r"\bsweater\b|\bcardigan\b|\bpullover\b|\bknitwear\b"),
    ("outerwear", r"\bjacket\b|\bcoat\b|\bouterwear\b|\bvest\b|\bblazer\b|\bwindbreaker\b"),
    ("tee_knit", r"\bt-shirt\b|\btee\b|\btshirt\b|\bjersey\b|\bknit top\b"),
    ("hawaiian", r"\bhawaiian\b|\bcamp\b|\bguayabera\b|\bcuban\b|\baloha\b|\bvacation\b"),
    ("casual", r"\bcasual\b|\bbutton[- ]down\b|\boxford\b|\blinen\b|\bprint\b|\bgraphic\b|\bpattern\b|\bshort sleeve\b|\blong sleeve\b|\bspread\b|\bclub\b"),
]
STYLE_RX = [(name, re.compile(rx, re.I)) for name, rx in STYLE_RULES]


def deepest_segment(cat: str) -> str:
    parts = [p.strip() for p in str(cat).split("|") if p.strip()]
    specific = [p for p in parts if p.lower() not in GENERIC]
    return specific[-1] if specific else (parts[-1] if parts else "")


def map_style(seg: str) -> str:
    s = seg.lower()
    for name, rx in STYLE_RX:
        if rx.search(s):
            return name
    return "other"


def main():
    base = pd.read_parquet(MASTER)[["parent_asin"]]
    meta = pd.read_parquet(META)[["parent_asin", "categories"]]
    df = base.merge(meta, on="parent_asin", how="left")
    df["category_segment"] = df["categories"].fillna("").map(deepest_segment)
    df["style"] = df["category_segment"].map(map_style)

    out = df[["parent_asin", "category_segment", "style"]]
    out.to_parquet(OUT_PQ, index=False)
    out.to_csv(OUT_CSV, index=False)

    n = len(out)
    print("===== JOB D STYLE EXTRACTION =====")
    print(f"rows: {n:,}  ->  {OUT_PQ.name} (+ .csv)\n")
    print(f"mapped (style != other): {(out['style']!='other').sum():,} "
          f"({(out['style']!='other').mean()*100:.1f}%)   other: {(out['style']=='other').sum():,}\n")
    print("style distribution:")
    print(out["style"].value_counts().to_string())
    print("\ntop 25 raw deepest segments (to verify mapping):")
    print(out["category_segment"].value_counts().head(25).to_string())
    print("\n'other' segments (need a rule?) top 15:")
    print(out.loc[out["style"] == "other", "category_segment"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
