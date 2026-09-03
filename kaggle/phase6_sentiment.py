"""Phase 6 (Kaggle GPU): Flair sentiment over all reviews, aggregated per product.

The paper's sentiment framework (Flair). Scans reviews_individual.parquet (one
row per review), scores each review with Flair's standard sentiment model
(distilBERT-based, trained on a mix including the Amazon review corpus —
downloaded from HU Berlin's public server, NOT the gated HF mirror),
aggregates per product (mean), saves product_sentiment.parquet (small).

Score mapping (Flair POSITIVE/NEGATIVE + confidence -> [0,1]):
    positive: 0.5 + 0.5*conf      negative: 0.5 - 0.5*conf      other: 0.5

Usage:
    python phase6_sentiment.py --limit 1000      # sample test (fast)
    python phase6_sentiment.py                   # full run
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

MODEL_NAME = "sentiment"
MINI_BATCH = 2048
PROGRESS_EVERY = 100_000


def find_input() -> str:
    for pattern in [
        "input/reviews_individual.parquet",
        "reviews_individual.parquet",
        "../input/reviews_individual.parquet",
    ]:
        hits = list(Path(".").glob(pattern)) + list(Path("..").glob(pattern))
        if hits:
            return str(hits[0])
    recursive = sorted(Path("/kaggle/input").rglob("reviews_individual.parquet"))
    if recursive:
        return str(recursive[0])
    raise FileNotFoundError(
        "reviews_individual.parquet not found. Upload it or mount a dataset containing it."
    )


def map_sentiment(label: str, confidence: float) -> float:
    if label == "POSITIVE":
        return 0.5 + 0.5 * confidence
    if label == "NEGATIVE":
        return 0.5 - 0.5 * confidence
    return 0.5


def score_reviews(classifier, texts: list[str]) -> list[float]:
    from flair.data import Sentence

    scores: list[float] = []
    t0 = time.time()
    for start in range(0, len(texts), MINI_BATCH):
        chunk = texts[start : start + MINI_BATCH]
        sentences = [Sentence(t) for t in chunk]
        classifier.predict(sentences, mini_batch_size=64, verbose=False)
        for s in sentences:
            lab = s.labels[0]
            scores.append(map_sentiment(lab.value, float(lab.score)))
        if (start + len(chunk)) % PROGRESS_EVERY < MINI_BATCH or start + len(chunk) >= len(texts):
            rate = (start + len(chunk)) / (time.time() - t0)
            print(f"  scored {start + len(chunk):,}/{len(texts):,} ({rate:,.0f} rev/s)")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: Flair sentiment on GPU")
    parser.add_argument("--input", default=None, help="reviews parquet (auto-found if omitted)")
    parser.add_argument("--out", default="product_sentiment.parquet")
    parser.add_argument("--limit", type=int, default=0, help="only score first N reviews (test)")
    args = parser.parse_args()

    input_path = args.input or find_input()
    print(f"using input: {input_path}")
    df = pd.read_parquet(input_path)
    if args.limit:
        df = df.head(args.limit)
    print(f"{len(df):,} reviews to score")

    print(f"loading Flair model '{MODEL_NAME}' (downloads from nlp.informatik.hu-berlin.de, 253 MB) ...")
    from flair.models import TextClassifier

    classifier = TextClassifier.load(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    print(f"model on {device}")

    t0 = time.time()
    texts = df["review_text"].fillna("").astype(str).tolist()
    scores = score_reviews(classifier, texts)
    print(f"scored {len(scores):,} reviews in {time.time() - t0:.0f}s")

    scored = df[["parent_asin"]].copy()
    scored["sentiment"] = scores
    grouped = (
        scored.groupby("parent_asin", sort=False)["sentiment"]
        .agg(sentiment_score="mean", n_reviews="size")
        .reset_index()
    )
    grouped["sentiment_score"] = grouped["sentiment_score"].round(4)
    grouped.to_parquet(args.out, index=False)
    print(f"saved {len(grouped):,} product sentiment scores -> {args.out}")


if __name__ == "__main__":
    main()
