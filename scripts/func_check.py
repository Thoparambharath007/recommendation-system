# -*- coding: utf-8 -*-
"""Quick functional check of the new collab + MMR path (synthetic data)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from phase7_fusion import (
    apply_mmr,
    build_collab_scores,
    build_neighbor_map,
    build_seed_scores,
    fuse,
)

seeds = build_seed_scores({"A": 1.0, "B": 0.6, "C": 0.2}, {"A": 0.9, "D": 0.5})
print("seeds:", {k: round(v, 3) for k, v in seeds.items()})

edges = pd.DataFrame({"product_a": ["A", "A", "B"], "product_b": ["X", "Y", "X"], "weight": [10, 4, 2]})
adj = build_neighbor_map(edges)
collab, via = build_collab_scores(seeds, adj)
print("collab:", {k: round(v, 3) for k, v in collab.items()}, "via:", via)

r = fuse(["X", "A", "Z"], {"A": 1.0, "X": 0.2}, {"A": 0.5, "X": 0.1}, None, collab_scores=collab)
print(r.to_string(index=False))
assert float(r.loc[r.parent_asin == "X", "fused"].iloc[0]) > float(
    r.loc[r.parent_asin == "Z", "fused"].iloc[0]
), "collab must lift X above Z"
assert "collab" in r.columns

titles = {
    "A": "white shirt",
    "B": "white tee",
    "C": "blue denim shirt",
    "D": "blue shirt",
    "X": "white t-shirt",
}
df = pd.DataFrame({"parent_asin": ["A", "B", "C", "D", "X"], "fused": [0.9, 0.85, 0.8, 0.75, 0.7]})
mmr = apply_mmr(df, titles)
print("mmr order:", list(mmr.parent_asin))
assert set(mmr.parent_asin) == set(df.parent_asin) and mmr.parent_asin.iloc[0] == "A"
print("FUNCTIONAL OK")
