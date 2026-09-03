"""Build a browsable HTML gallery of all shirts whose title contains a color.

The dataset has no image files - only Amazon CDN URLs - so the gallery loads
images directly from those URLs (needs internet).

Usage: python src/make_gallery.py white
Saves: web/<color>_shirts_gallery.html
"""
import sys
from pathlib import Path

import pandas as pd

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
TEXTS = Path(__file__).resolve().parent.parent / "data" / "processed" / "product_texts.parquet"
DETAILS = Path(__file__).resolve().parent.parent / "data" / "processed" / "product_details.parquet"


def main() -> None:
    color = sys.argv[1] if len(sys.argv) > 1 else "white"
    texts = pd.read_parquet(TEXTS, columns=["parent_asin", "title"])
    det = pd.read_parquet(DETAILS, columns=["parent_asin", "price", "image_url", "average_rating", "rating_number"])
    df = det.merge(texts, on="parent_asin", how="left")
    df = df[df["title"].str.lower().str.contains(f" {color.lower()} ", na=False)]
    df = df[df["image_url"].notna() & (df["image_url"] != "")]
    df = df.sort_values("rating_number", ascending=False)
    print(f"{len(df)} {color} shirts with images")

    cards = []
    for row in df.itertuples():
        price = f"${row.price:.2f}" if pd.notna(row.price) else "price N/A"
        rating = f"{row.average_rating:.1f} ({int(row.rating_number)})" if pd.notna(row.rating_number) else ""
        t = str(row.title).replace("&", "&amp;").replace("<", "&lt;")
        cards.append(
            f'<div class="card"><img loading="lazy" src="{row.image_url}" alt="">'
            f"<div class='info'><b>{t[:80]}</b><span>{price} &middot; {rating}</span>"
            f"<code>{row.parent_asin}</code></div></div>"
        )

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{color} shirts - full catalog ({len(df)})</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#f5f5f7;margin:20px}}
h1{{font-size:20px}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px}}
.card{{background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.12)}}
.card img{{width:100%;height:210px;object-fit:contain;background:#fff}}
.info{{padding:8px;font-size:12px;display:flex;flex-direction:column;gap:4px}}
code{{color:#666;font-size:10px}}
</style></head><body>
<h1>All "{color}" shirts in catalog: {len(df):,} (sorted by review count)</h1>
<div class="grid">{''.join(cards)}</div></body></html>"""

    out = WEB_DIR / f"{color.lower()}_shirts_gallery.html"
    out.write_text(html, encoding="utf-8")
    print(f"saved -> {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
