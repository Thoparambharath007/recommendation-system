"""08_fill_pocket_color.py  (run LOCALLY, in DuckDB idea)

Fills the two remaining master fields from the Kaggle job outputs, incrementally
(safe to re-run; only touches still-empty cells):

  pocket  <- kaggle/pocket verification/pocket_yolo26k.csv   (YOLOv8-cls vision)
             only where pocket == 'unknown'; pocket_source = 'yolo_vision'
  color   <- kaggle/color verification/color_text.csv        (Layer-1 text)
             then  kaggle/color verification/color_vision.csv (Layer-2 pixel, optional)
             only where color is empty; color_source = 'text' / 'vision'

Rewrites shirts_master.parquet AND its shirts_master.csv copy. Never invents a
value: rows with no prediction stay 'unknown' / empty.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
MASTER = DATA / "shirts_master.parquet"
MASTER_CSV = DATA / "shirts_master.csv"
POCKET = HERE.parent / "kaggle" / "pocket verification" / "pocket_yolo26k.csv"
COLOR_TEXT = HERE.parent / "kaggle" / "color verification" / "color_text.csv"
COLOR_VIS = HERE.parent / "kaggle" / "color verification" / "color_vision.csv"


def main():
    m = pd.read_parquet(MASTER)
    n0 = len(m)
    print(f"master rows: {n0:,}")

    # ---- pocket (vision YOLO) ----
    if "pocket_source" not in m.columns:
        m["pocket_source"] = "text"
    if POCKET.exists():
        p = pd.read_csv(POCKET)[["parent_asin", "pocket_cls"]]
        mask = m["pocket"].eq("unknown")
        lut = p.drop_duplicates("parent_asin").set_index("parent_asin")["pocket_cls"]
        fill = m.loc[mask, "parent_asin"].map(lut)
        got = fill.notna()
        m.loc[mask, "pocket"] = fill.where(got, m.loc[mask, "pocket"])
        m.loc[mask & got.fillna(False).reindex(m.index, fill_value=False), "pocket_source"] = "yolo_vision"
        print(f"pocket filled from YOLO: {int(got.sum()):,}  "
              f"(still unknown: {int(m['pocket'].eq('unknown').sum()):,})")
    else:
        print(f"!! missing {POCKET.name}")

    # ---- color (text first, then vision) ----
    if "color_source" not in m.columns:
        m["color_source"] = None
    for path, tag in [(COLOR_TEXT, "text"), (COLOR_VIS, "vision")]:
        if not path.exists():
            print(f"-- {path.name} not present yet, skipping layer '{tag}'")
            continue
        c = pd.read_csv(path)[["parent_asin", "color"]]
        lut = c.drop_duplicates("parent_asin").set_index("parent_asin")["color"]
        empty = m["color"].isna() | m["color"].eq("")
        if tag == "text":
            # text layer is deterministic: re-mining replaces its own earlier
            # rows (e.g. after a vocabulary upgrade) without touching vision
            mask = m["color_source"].eq("text") | empty
        else:
            mask = empty
        fill = m.loc[mask, "parent_asin"].map(lut)
        got = fill.notna()
        m.loc[mask, "color"] = fill.where(got, m.loc[mask, "color"])
        m.loc[mask & got.fillna(False).reindex(m.index, fill_value=False), "color_source"] = tag
        print(f"color filled from {tag}: {int(got.sum()):,}  "
              f"(still empty: {int(m['color'].isna().sum()):,})")

    assert len(m) == n0
    m.to_parquet(MASTER, index=False)
    m.to_csv(MASTER_CSV, index=False)
    print(f"\nwrote {MASTER.name} + {MASTER_CSV.name}")
    print("pocket:", m["pocket"].value_counts(dropna=False).to_dict())
    print("color_source:", m["color_source"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
