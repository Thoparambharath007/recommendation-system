"""Extract product details (rating, price, image URL) from raw metadata -> parquet.

The Phase 1 processed parquet kept only text fields; the website needs
rating/price/image per product. Streams the 3.77 GB raw metadata once.

Usage:
    python src/extract_details.py
"""

import gzip
import json
import time

import pandas as pd

from config import (
    DATA_PROCESSED_DIR,
    PRODUCT_TEXTS_PATH,
    RAW_METADATA_PATH,
)

DETAILS_PATH = DATA_PROCESSED_DIR / "product_details.parquet"


def extract_details(
    metadata_path=RAW_METADATA_PATH,
    catalog: set[str] | None = None,
) -> list[dict]:
    if catalog is None:
        df = pd.read_parquet(PRODUCT_TEXTS_PATH, columns=["parent_asin"])
        catalog = {str(v) for v in df["parent_asin"].dropna()}

    rows: list[dict] = []
    total = 0
    t0 = time.time()
    with gzip.open(metadata_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            parent = rec.get("parent_asin")
            if not parent or parent not in catalog:
                continue
            images = rec.get("images") or []
            image_url = ""
            if images:
                image_url = (
                    images[0].get("large")
                    or images[0].get("hi_res")
                    or images[0].get("thumb")
                    or ""
                )
            rows.append(
                {
                    "parent_asin": parent,
                    "title": rec.get("title") or "",
                    "brand": rec.get("store") or "",
                    "average_rating": rec.get("average_rating"),
                    "rating_number": rec.get("rating_number"),
                    "price": rec.get("price"),
                    "image_url": image_url,
                }
            )
            if total % 500_000 == 0:
                print(f"  {total:,} records, {len(rows):,} kept, {time.time() - t0:.0f}s")

    print(f"extracted {len(rows):,} products in {time.time() - t0:.0f}s")
    return rows


def main() -> None:
    t0 = time.time()
    rows = extract_details()
    df = pd.DataFrame(rows)
    DETAILS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DETAILS_PATH, index=False)
    print(f"saved {len(df):,} rows -> {DETAILS_PATH} ({DETAILS_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
