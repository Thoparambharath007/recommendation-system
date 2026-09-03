"""Quick sanity check for the normalization + novelty fixes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from phase7_fusion import fuse, apply_novelty_filter

# 1: normalization makes a strong lexical match win over a slightly
# higher cosine on a weak match (the real "black shirt" failure mode)
res = fuse(
    ["A_SHIRT", "B_TEE"],
    {"A_SHIRT": 0.32, "B_TEE": 0.10},   # A is a true lexical match
    {"A_SHIRT": 0.700, "B_TEE": 0.715},  # B's cosine only marginally higher
    None,
    normalize=True,
)
top = res.loc[0, "parent_asin"]
print("normalized top:", top, "(want A_SHIRT)")
assert top == "A_SHIRT", "normalization failed to favor lexical match"

# also prove raw mode got it wrong (the bug we are fixing)
res_raw = fuse(
    ["A_SHIRT", "B_TEE"],
    {"A_SHIRT": 0.32, "B_TEE": 0.10},
    {"A_SHIRT": 0.700, "B_TEE": 0.715},
    None,
    normalize=False,
)
print("raw-mode top (before fix):", res_raw.loc[0, "parent_asin"])

# 4: real-data replay — cached scores for an actual black-shirt query
import json

cache = json.load(open(Path(__file__).resolve().parent.parent / "results" / "eval_cache.json"))
key = next((k for k in cache if k.startswith("black shirt")), None)
if key:
    entry = cache[key]
    cands = list(dict.fromkeys(list(entry["content_scores"]) + list(entry["embed_scores"])))[:2000]
    before = fuse(cands, entry["content_scores"], entry["embed_scores"], None, normalize=False)
    after = fuse(cands, entry["content_scores"], entry["embed_scores"], None, normalize=True)
    texts = pd.read_parquet(
        Path(__file__).resolve().parent.parent / "data" / "processed" / "product_texts.parquet",
        columns=["parent_asin", "title"],
    )
    tmap = dict(zip(texts["parent_asin"].astype(str), texts["title"].astype(str)))
    print(f"\nREAL QUERY: {key!r} ({len(cands)} candidates)")
    print("--- BEFORE (raw scales) top 5 ---")
    for a in before["parent_asin"].head(5):
        print("   ", tmap.get(a, a)[:80])
    print("--- AFTER (normalized) top 5 ---")
    for a in after["parent_asin"].head(5):
        print("   ", tmap.get(a, a)[:80])

    def black_in_top(df, n=20):
        titles_top = [tmap.get(a, "").lower() for a in df["parent_asin"].head(n)]
        return sum(1 for t in titles_top if " black " in f" {t} ")

    print("black-title items in top 20  BEFORE:", black_in_top(before), " AFTER:", black_in_top(after))
    best_c = max(cands, key=lambda a: entry["content_scores"].get(a, -1))
    rb = int(before.index[before["parent_asin"] == best_c][0])
    ra = int(after.index[after["parent_asin"] == best_c][0])
    print(f"rank of best content match  BEFORE: {rb+1}  AFTER: {ra+1}")

print("ALL SANITY CHECKS PASSED")



# 2: novelty ban catches the Fender tee from the user's screenshot
titles = {
    "F1": "Fender Telecaster Guitars, Angels, and Bikes Rockabilly Work Shirt (XL) Black",
    "S1": "Van Heusen Black Dress Shirt",
}
r = pd.DataFrame({"parent_asin": ["F1", "S1"], "fused": [0.9, 0.8]})
out = apply_novelty_filter(r, titles)
kept = list(out["parent_asin"])
print("novelty kept:", kept, "(want ['S1'])")
assert kept == ["S1"], "novelty ban missed the Fender tee"

# 3: default (raw) fuse unchanged — tests rely on raw mode
r2 = fuse(["A", "B"], {"A": 1.0, "B": 0.5}, {"A": 0.5, "B": 0.9}, None)
expected = 0.3 * 1.0 + 0.4 * 0.5 + 0.3 * 0.5
got = float(r2.loc[r2["parent_asin"] == "A", "fused"].iloc[0])
print("raw-mode fused A:", got, "(want", round(expected, 4), ")")
assert abs(got - expected) < 1e-9

# 3b: plain "shirt" query drops polos; explicit "polo" query keeps them
from phase7_fusion import apply_type_preference

titles3 = {
    "P1": "Croker Scottish Performance Shirt Medium Polyester Short Sleeve Polo Jersey Black",
    "D1": "Van Heusen Black Dress Shirt",
}
r3 = pd.DataFrame({"parent_asin": ["P1", "D1"], "fused": [0.9, 0.8]})
out3 = apply_type_preference(r3, titles3, "black shirt")
print("plain 'black shirt' kept:", list(out3["parent_asin"]), "(want ['D1'])")
assert list(out3["parent_asin"]) == ["D1"]
out4 = apply_type_preference(r3, titles3, "black polo shirt")
print("'black polo shirt' kept:", list(out4["parent_asin"]), "(want ['P1'])")
assert list(out4["parent_asin"]) == ["P1"]

print("ALL SANITY CHECKS PASSED")
