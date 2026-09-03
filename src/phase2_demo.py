import sys

sys.path.insert(0, r"C:\Users\thopa\OneDrive\Desktop\project\REAL SYSTEM\src")

import pandas as pd

from content_filter import ContentFilter
from data_loader import load_men_shirts_metadata
from phase2_ibcf import get_neighbors, load_edges
from preprocessing import preprocess_metadata

df = preprocess_metadata(load_men_shirts_metadata())
cf = ContentFilter(df)
top = cf.get_top_k("nike shirts large blue", k=3)

row = top.iloc[0]
print("Phase 1 top product:")
print(" ", row["parent_asin"], "|", row["title"][:70], "| score", round(row["score"], 3))

edges = load_edges()
neigh = get_neighbors(edges, row["parent_asin"], top_k=8)
lookup = df.set_index("parent_asin")
asin = row["parent_asin"]
print(f"\nPhase 2 expansion of {asin} (people also bought):")
print(f"{'neighbor':<14}{'weight':<8}title")
for n in neigh.itertuples():
    t = lookup.loc[n.neighbor, "title"][:60] if n.neighbor in lookup.index else "?"
    print(f"{n.neighbor:<14}{n.weight:<8}{t}")

print("\n---")
print("Expansion of a highly-connected product (B00KBZOB40):")
neigh2 = get_neighbors(edges, "B00KBZOB40", top_k=8)
for n in neigh2.itertuples():
    t = lookup.loc[n.neighbor, "title"][:60] if n.neighbor in lookup.index else "?"
    print(f"{n.neighbor:<14}{n.weight:<8}{t}")
