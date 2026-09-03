"""Diagnose the 'shirt query returns tees' bug using the real catalog (no models).

Loads the Phase 1 metadata parquet, builds the TF-IDF content filter, and
simulates the web_app pipeline's candidate pool + filter chain for a query,
printing how many tees/tanks survive and what the filter returns at each step.

Usage:
    python scripts/diag_pipeline.py "black shirt with white buttons and price range between 1000 to 2000"
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import re

import pandas as pd

from content_filter import ContentFilter
from preprocessing import preprocess_metadata
from config import PROCESSED_METADATA_PATH, PRODUCT_TEXTS_PATH

from phase7_fusion import (
    TEE_MARKERS,
    TANK_MARKERS,
    apply_color_preference,
    apply_garment_filter,
    apply_novelty_filter,
    apply_occasion_preference,
    apply_type_preference,
    detect_garment_type,
    parse_price_range,
)
from preprocessing import COLOR_PATTERN


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "black shirt with white buttons and price range between 1000 to 2000"
    q = query.lower()
    print(f"QUERY: {q!r}\n")

    t0 = time.time()
    df = pd.read_parquet(PROCESSED_METADATA_PATH)
    df = preprocess_metadata(df)
    print(f"loaded + preprocessed {len(df):,} products in {time.time() - t0:.1f}s")

    t0 = time.time()
    cf = ContentFilter(df)
    print(f"content filter fit {cf.matrix.shape[0]:,} x {cf.matrix.shape[1]:,} in {time.time() - t0:.1f}s")

    t0 = time.time()
    titles_df = pd.read_parquet(PRODUCT_TEXTS_PATH, columns=["parent_asin", "title"])
    titles = dict(zip(titles_df["parent_asin"].astype(str), titles_df["title"].astype(str)))
    del titles_df
    print(f"titles loaded in {time.time() - t0:.1f}s")

    # ---- mirror web_app pooling
    min_price, max_price = parse_price_range(q)
    print(f"price intent: min={min_price} max={max_price}")
    color_filter = bool(COLOR_PATTERN.search(q))
    kind = detect_garment_type(q)
    print(f"color intent: {color_filter} | garment type intent: {kind!r}")

    pool = 200 * 25 if (color_filter or min_price is not None or max_price is not None) else 200
    content_top = cf.get_top_k(q, k=pool)
    print(f"\ncontent pool: {len(content_top):,} candidates (scores>0: {(content_top['score'] > 0).sum():,})\n")

    # emulate: content + (dummy) candidates. Real system also has embed; here content-only.
    candidate = content_top.copy()
    result = pd.DataFrame(
        {
            "parent_asin": candidate["parent_asin"].astype(str),
            "content": candidate["score"].round(4),
            "embed": 0.5,
            "sentiment_eff": 0.5,
            "n_reviews": 0,
            "fused": candidate["score"].round(4),
        }
    )

    def stats(label: str, r: pd.DataFrame) -> None:
        tees = sum(1 for a in r["parent_asin"] if any(m in f" {titles.get(a, '').lower()} " for m in TEE_MARKERS))
        tanks = sum(1 for a in r["parent_asin"] if any(m in f" {titles.get(a, '').lower()} " for m in TANK_MARKERS))
        no_shirt_word = sum(1 for a in r["parent_asin"] if "shirt" not in titles.get(a, "").lower())
        print(f"{label:28s} n={len(r):5d} | tees={tees:4d} tanks={tanks:4d} no_shirt_word={no_shirt_word:5d}")

    stats("after fuse (pool)", result)
    r = apply_color_preference(result, titles, q, 12)
    stats("after color", r)
    r = apply_type_preference(r, titles, q)
    stats("after type", r)
    r = apply_occasion_preference(r, titles, q)
    stats("after occasion", r)
    r = apply_novelty_filter(r, titles)
    stats("after novelty", r)
    r = apply_garment_filter(r, titles)
    stats("after garment", r)
    if min_price is not None or max_price is not None:
        before = len(r)
        prices = None
        # need real prices for the real check; try details parquet
        try:
            from config import DATA_PROCESSED_DIR
            det = pd.read_parquet(DATA_PROCESSED_DIR / "product_details.parquet", columns=["parent_asin", "price"])
            det = det.set_index("parent_asin")["price"]
            p = det.reindex(r["parent_asin"].astype(str)).to_numpy()
            mask = ~pd.isna(p)
            if min_price is not None:
                mask &= p >= min_price
            if max_price is not None:
                mask &= p <= max_price
            r = r[mask]
            stats("after price", r)
            dropped = before - len(r)
            print(f"  (price filter dropped {dropped} — this is the silent-fallback danger zone)")
        except Exception as e:
            print(f"  (could not price-check: {e})")

    print(f"\n=== TOP 20 after full chain ===")
    for i, row in r.head(20).iterrows():
        a = str(row["parent_asin"])
        t = titles.get(a, "")
        tl = t.lower()
        tag = "TEE" if any(m in f" {tl} " for m in TEE_MARKERS) else ("TANK" if any(m in f" {tl} " for m in TANK_MARKERS) else "OK  ")
        print(f"{tag} {row['fused']:.4f} {a}  {t[:85]}")


if __name__ == "__main__":
    main()