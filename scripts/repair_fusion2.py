# -*- coding: utf-8 -*-
"""Rebuild broken lines in build_neighbor_map by index (safe style, single-close parens only.\\
"""
from pathlib import Path

P = Path(r"C:\Users\thopa\OneDrive\Desktop\project\REAL SYSTEM\src\phase7_fusion.py")
lines = P.read_text(encoding="utf-8").splitlines()

# Locate the 'wn = w / w_max' anchor line
anchor = next(i for i, l in enumerate(lines) if "wn = w / w_max" in l)
# The broken block starts four lines above the anchor
start = anchor - 4
assert "itertuples" in lines[start - 1], lines[start - 1]

new_block = [
    "        a = a",
    "        b = b",
    "        w = w",
    lines[anchor],
    "        lst = adj.setdefault(a, [])",
    "        t = tuple([b, wn])",
    "        lst.append(t)",
    "        lst2 = adj.setdefault(b, [])",
    "        t2 = tuple([a, wn])",
    "        lst2.append(t2)",
]
lines[start : anchor + 1] = new_block

P.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("rebuilt lines", start + 1, "to", start + len(new_block))