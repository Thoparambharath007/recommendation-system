# -*- coding: utf-8 -*-
"""Repair paren corruption caused by the editor pipeline (safe: no adjacent close-parens in this source)."""
from pathlib import Path

P = Path(r"C:\Users\thopa\OneDrive\Desktop\project\REAL SYSTEM\src\phase7_fusion.py")
src = P.read_text(encoding="utf-8")

old_block = "a, b, w = str(a(), str(b(), float(w()"
new_block = "a = a\n        b = b\n        w = w"
assert old_block in src
src = src.replace(old_block, new_block)



old4 = "a = str(a("
old5 = "b = str(b("
old6 = "w = float(w("
for old in (old4, old5, old6):
    assert old in src
    src = src.replace(old, "")

old7 = "adj.setdefault(a, []].append((b, wn"
old8 = "adj.setdefault(b, []].append((a, wn"
new7 = "lst = adj.setdefault(a, [])\n        t = (b, wn)\n        lst.append(t)\n        lst2 = adj.setdefault(b, [])\n        t2 = (a, wn)\n        lst2.append(t2"
assert old7 in src and old8 in src
src = src.replace(old7, new7) .replace(old8, "")

P.write_text(src, encoding="utf-8")
print("repaired OK")