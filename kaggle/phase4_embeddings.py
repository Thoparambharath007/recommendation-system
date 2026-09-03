"""Phase 4: CasT5RoBERT embeddings — cascade of sentence-t5-xl + all-distilroberta-v1.

Runs on Kaggle (T4 GPU). For each product's full_text, produces a fingerprint
embedding. Checkpointed: every chunk is saved immediately, so a killed session
can resume without re-running completed chunks.

Usage:
    python phase4_embeddings.py --input input/product_texts_sample.parquet --limit 50
    python phase4_embeddings.py --input input/product_texts.parquet

If --input is omitted, the script auto-searches common locations
(working dir, ./input, ../input, and Kaggle's /kaggle/input mounted datasets).
"""

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODEL_T5 = "sentence-transformers/sentence-t5-xl"
MODEL_ROBERTA = "sentence-transformers/all-distilroberta-v1"

DEFAULT_CHUNK_SIZE = 10_000
DEFAULT_BATCH_SIZE = 8

CANDIDATE_INPUTS = [
    "input/product_texts.parquet",
    "product_texts.parquet",
    "../input/product_texts.parquet",
]


def find_input(limit: int | None = None) -> str:
    for pattern in CANDIDATE_INPUTS:
        hits = glob.glob(pattern)
        if hits:
            return hits[0]
    dataset_hits = sorted(glob.glob("/kaggle/input/*/product_texts.parquet"))
    if dataset_hits:
        return dataset_hits[0]
    raise FileNotFoundError(
        "product_texts.parquet not found. Upload it next to the script, or mount a dataset containing it."
    )


def build_model(name: str) -> SentenceTransformer:
    print(f"loading {name} ...")
    model = SentenceTransformer(name, device="cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model = model.half()
    model.max_seq_length = 512
    return model


def encode_cascade(
    model_t5: SentenceTransformer,
    model_roberta: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    emb_t5 = model_t5.encode(
        texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
    )
    emb_roberta = model_roberta.encode(
        texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
    )
    return np.concatenate([emb_t5, emb_roberta], axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: generate CasT5RoBERT embeddings")
    parser.add_argument("--input", default=None, help="product_texts parquet path (auto-searched if omitted)")
    parser.add_argument("--out", default="output", help="output directory")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=0, help="only process first N rows (test)")
    parser.add_argument("--t5-model", default=MODEL_T5)
    parser.add_argument("--roberta-model", default=MODEL_ROBERTA)
    args = parser.parse_args()

    input_path = args.input or find_input()
    print(f"using input: {input_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    if args.limit:
        df = df.head(args.limit)
    texts = df["full_text"].fillna("").astype(str).tolist()
    asins = df["parent_asin"].astype(str).tolist()
    n = len(texts)
    print(f"{n:,} products to embed")

    t5 = build_model(args.t5_model)
    roberta = build_model(args.roberta_model)
    print(f"models ready: T5 -> {t5.get_sentence_embedding_dimension()} dims, "
          f"RoBERTa -> {roberta.get_sentence_embedding_dimension()} dims, "
          f"cascade -> {t5.get_sentence_embedding_dimension() + roberta.get_sentence_embedding_dimension()} dims")

    t0 = time.time()
    completed = 0
    for start in range(0, n, args.chunk_size):
        chunk_id = start // args.chunk_size
        chunk_path = out_dir / f"embeddings_{chunk_id:05d}.parquet"
        if chunk_path.exists():
            print(f"chunk {chunk_id} already done, skipping")
            completed += len(pd.read_parquet(chunk_path))
            continue

        chunk_texts = texts[start : start + args.chunk_size]
        chunk_asins = asins[start : start + args.chunk_size]
        embeddings = encode_cascade(t5, roberta, chunk_texts, args.batch_size)

        chunk_df = pd.DataFrame(
            {"parent_asin": chunk_asins, "embedding": list(embeddings)}
        )
        chunk_df.to_parquet(chunk_path, index=False)
        completed += len(chunk_df)
        elapsed = time.time() - t0
        rate = completed / elapsed
        eta = (n - completed) / rate / 3600 if rate else 0
        print(f"chunk {chunk_id}: saved {len(chunk_df):,} (total {completed:,}/{n:,}) "
              f"| {elapsed/60:.1f} min elapsed | ETA {eta:.1f} h")

    print(f"DONE: {completed:,} embeddings in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
