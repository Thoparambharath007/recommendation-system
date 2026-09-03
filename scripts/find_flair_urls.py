import flair
from pathlib import Path
import re

base = Path(flair.__file__).parent / "models" / "__init__.py"
txt = base.read_text(encoding="utf-8", errors="ignore")

names = ["sentiment-fast", "sentiment", "en-sentiment"]
for name in names:
    idx = txt.find('"' + name + '"')
    if idx >= 0:
        seg = txt[idx : idx + 600]
        urls = re.findall(r"https?://[^\s\"']+", seg)
        print(name, "=>", urls[:2])
    else:
        print(name, "=> NOT FOUND in registry")
