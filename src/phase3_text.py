import argparse
import gzip
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from config import (
    PROCESSED_METADATA_PATH,
    PRODUCT_TEXTS_PATH,
    RAW_REVIEWS_PATH,
)

PROGRESS_EVERY = 2_000_000
MAX_REVIEWS_PER_PRODUCT = 20
MAX_REVIEW_CHARS = 2000


def load_catalog(path: Path = PROCESSED_METADATA_PATH) -> set[str]:
    df = pd.read_parquet(path, columns=["parent_asin"])
    return {str(v) for v in df["parent_asin"].dropna()}


def load_metadata(path: Path = PROCESSED_METADATA_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def clean_review_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def collect_reviews(
    reviews_path: Path = RAW_REVIEWS_PATH,
    catalog: set[str] | None = None,
    max_per_product: int = MAX_REVIEWS_PER_PRODUCT,
    max_chars: int = MAX_REVIEW_CHARS,
) -> dict[str, list[str]]:
    if catalog is None:
        catalog = load_catalog()

    product_reviews: dict[str, list[str]] = defaultdict(list)
    total = 0
    matched = 0
    t0 = time.time()

    with gzip.open(reviews_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            parent = rec.get("parent_asin")
            if not parent or parent not in catalog:
                continue
            matched += 1
            texts = product_reviews[parent]
            if len(texts) >= max_per_product:
                continue
            text = clean_review_text(rec.get("text") or "")[:max_chars]
            if text:
                texts.append(text)
            if total % PROGRESS_EVERY == 0:
                print(
                    f"  {total:,} reviews, {len(product_reviews):,} products with reviews, "
                    f"{time.time() - t0:.0f}s"
                )

    print(
        f"collected {sum(len(v) for v in product_reviews.values()):,} review texts "
        f"for {len(product_reviews):,} products from {matched:,} matched reviews in "
        f"{time.time() - t0:.0f}s"
    )
    return dict(product_reviews)


def assemble_texts(
    metadata: pd.DataFrame,
    product_reviews: dict[str, list[str]],
) -> pd.DataFrame:
    df = metadata[["parent_asin", "description", "features", "title"]].copy()
    df["parent_asin"] = df["parent_asin"].astype(str)

    df["reviews_text"] = df["parent_asin"].map(
        lambda p: " ".join(product_reviews.get(p, []))
    )
    df["full_text"] = (
        df["description"].fillna("").str.strip()
        + " "
        + df["features"].fillna("").str.strip()
        + " "
        + df["reviews_text"]
    )
    df["full_text"] = df["full_text"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[["parent_asin", "title", "reviews_text", "full_text"]]
    return df.reset_index(drop=True)


def save_texts(df: pd.DataFrame, path: Path = PRODUCT_TEXTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"saved {len(df):,} product texts -> {path}")


def load_texts(path: Path = PRODUCT_TEXTS_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: concatenate description + reviews per product"
    )
    parser.add_argument(
        "--sample", metavar="ASIN", help="show the concatenated text for one product"
    )
    args = parser.parse_args()

    if args.sample:
        texts = load_texts()
        row = texts[texts["parent_asin"] == args.sample]
        if row.empty:
            print(f"product {args.sample} not found")
            return
        row = row.iloc[0]
        print(f"title: {row['title']}\n")
        print(f"full_text ({len(row['full_text']):,} chars):\n")
        print(row["full_text"][:3000])
        return

    t0 = time.time()
    print("loading catalog ...")
    catalog = load_catalog()
    print(f"  {len(catalog):,} products in {time.time() - t0:.1f}s")

    product_reviews = collect_reviews(catalog=catalog)

    print("loading metadata ...")
    metadata = load_metadata()
    df = assemble_texts(metadata, product_reviews)
    save_texts(df)


if __name__ == "__main__":
    main()
