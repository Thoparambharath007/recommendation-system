"""Phase 6: sentiment analysis — extract individual reviews + aggregate Flair scores.

Build step (local): re-stream the reviews file and save ONE ROW PER REVIEW
(the Phase 3 parquet joined reviews into a string, losing the per-review split).

Scoring step (Kaggle GPU): `scratch/kaggle/phase6_sentiment.py` reads
reviews_individual.parquet, runs Flair sentiment on every review, aggregates per
product -> product_sentiment.parquet.

Sentiment mapping (Flair POSITIVE/NEGATIVE + confidence -> [0,1]):
    positive: 0.5 + 0.5*conf      negative: 0.5 - 0.5*conf

Usage:
    python src/phase6_sentiment.py --build          # stream reviews -> parquet
    python src/phase6_sentiment.py --sample ASIN    # show that product's reviews
"""

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
    PRODUCT_SENTIMENT_PATH,
    RAW_REVIEWS_PATH,
    REVIEWS_INDIVIDUAL_PATH,
)

PROGRESS_EVERY = 2_000_000
MAX_REVIEWS_PER_PRODUCT = 20
MAX_REVIEW_CHARS = 2000

NEUTRAL_SCORE = 0.5


def load_catalog(path: Path = PROCESSED_METADATA_PATH) -> set[str]:
    df = pd.read_parquet(path, columns=["parent_asin"])
    return {str(v) for v in df["parent_asin"].dropna()}


def clean_review_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def map_sentiment(label: str, confidence: float) -> float:
    """Flair label+confidence -> [0,1] score (0.5 neutral)."""
    if label == "POSITIVE":
        return 0.5 + 0.5 * confidence
    if label == "NEGATIVE":
        return 0.5 - 0.5 * confidence
    return NEUTRAL_SCORE


def aggregate_sentiment(
    scores: pd.DataFrame, default: float = NEUTRAL_SCORE
) -> pd.DataFrame:
    """Average per-review scores per product -> (parent_asin, sentiment_score, n_reviews)."""
    grouped = (
        scores.groupby("parent_asin", sort=False)["sentiment"]
        .agg(sentiment_score="mean", n_reviews="size")
        .reset_index()
    )
    grouped["sentiment_score"] = grouped["sentiment_score"].round(4)
    return grouped


def extract_reviews(
    reviews_path: Path = RAW_REVIEWS_PATH,
    catalog: set[str] | None = None,
    max_per_product: int = MAX_REVIEWS_PER_PRODUCT,
    max_chars: int = MAX_REVIEW_CHARS,
) -> list[dict]:
    if catalog is None:
        catalog = load_catalog()

    rows: list[dict] = []
    product_counts: defaultdict[str, int] = defaultdict(int)
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
            if product_counts[parent] >= max_per_product:
                continue
            text = clean_review_text(rec.get("text") or "")[:max_chars]
            if text:
                product_counts[parent] += 1
                rows.append({"parent_asin": parent, "review_text": text})
            if total % PROGRESS_EVERY == 0:
                print(
                    f"  {total:,} reviews, {len(rows):,} kept, "
                    f"{len(product_counts):,} products, {time.time() - t0:.0f}s"
                )

    print(
        f"extracted {len(rows):,} reviews for {len(product_counts):,} products "
        f"from {matched:,} matched reviews in {time.time() - t0:.0f}s"
    )
    return rows


def save_reviews(rows: list[dict], path: Path = REVIEWS_INDIVIDUAL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["parent_asin", "review_text"])
    df.to_parquet(path, index=False)
    print(f"saved {len(df):,} reviews -> {path}")


def load_reviews(path: Path = REVIEWS_INDIVIDUAL_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: sentiment analysis")
    parser.add_argument("--build", action="store_true", help="extract reviews -> parquet")
    parser.add_argument("--sample", metavar="ASIN", help="show a product's reviews")
    args = parser.parse_args()

    if args.sample:
        df = load_reviews()
        rows = df[df["parent_asin"] == args.sample]
        print(f"{len(rows)} reviews for {args.sample}:")
        for r in rows["review_text"].tolist():
            print(f"  - {r[:200]}")
        return

    if args.build:
        t0 = time.time()
        print("loading catalog ...")
        catalog = load_catalog()
        print(f"  {len(catalog):,} products in {time.time() - t0:.1f}s")
        rows = extract_reviews(catalog=catalog)
        save_reviews(rows)
        return

    parser.error("provide --build or --sample ASIN")


if __name__ == "__main__":
    main()
