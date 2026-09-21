"""Phase 5: embedding similarity — query fingerprinting + top-k retrieval.

Builds a cosine-similarity index over the Phase 4 embeddings (1536 dims,
sentence-t5-xl + all-distilroberta-v1 cascade) and answers natural-language
queries against it. The query sentence passes through the same cascade models
on CPU (one sentence, a few seconds).

Usage:
    python src/phase5_similarity.py --build-index
    python src/phase5_similarity.py --query "breathable cotton shirt for office"
    python src/phase5_similarity.py --query "soft nike polo for golf" --top-k 20
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_PROCESSED_DIR,
    PRODUCT_TEXTS_PATH,
)

EMBEDDING_CHUNKS_DIR = Path(__file__).resolve().parent.parent / "kaggle" / "output"
INDEX_PATH = DATA_PROCESSED_DIR / "embeddings_index.npy"
INDEX_ASINS_PATH = DATA_PROCESSED_DIR / "embeddings_asins.parquet"

MODEL_T5 = "sentence-transformers/sentence-t5-xl"
MODEL_ROBERTA = "sentence-transformers/all-distilroberta-v1"

EXPECTED_DIMS = 1536


def build_model(name: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device="cpu")
    model.max_seq_length = 512
    return model


class QueryEmbedder:
    """Fingerprints a query sentence with the same cascade used in Phase 4."""

    def __init__(self, t5_model: str = MODEL_T5, roberta_model: str = MODEL_ROBERTA):
        print(f"loading {t5_model} ...")
        self.t5 = build_model(t5_model)
        print(f"loading {roberta_model} ...")
        self.roberta = build_model(roberta_model)
        print(
            f"models ready: T5 -> {self.t5.get_embedding_dimension()} dims, "
            f"RoBERTa -> {self.roberta.get_embedding_dimension()} dims"
        )

    def embed(self, text: str) -> np.ndarray:
        emb_t5 = self.t5.encode(
            [text], batch_size=1, convert_to_numpy=True, normalize_embeddings=False
        )
        emb_roberta = self.roberta.encode(
            [text], batch_size=1, convert_to_numpy=True, normalize_embeddings=False
        )
        return np.concatenate([emb_t5, emb_roberta], axis=1).astype(np.float32)[0]


class EmbeddingIndex:
    """Unit-norm embedding matrix with fast top-k dot-product retrieval."""

    def __init__(self, asins: np.ndarray, matrix: np.ndarray):
        if matrix.ndim != 2 or matrix.dtype != np.float32:
            raise ValueError(f"matrix must be 2D float32, got {matrix.shape} {matrix.dtype}")
        self.asins = asins
        self.matrix = matrix

    @classmethod
    def build(
        cls,
        chunk_dir: Path = EMBEDDING_CHUNKS_DIR,
        index_path: Path = INDEX_PATH,
        asins_path: Path = INDEX_ASINS_PATH,
    ) -> "EmbeddingIndex":
        t0 = time.time()
        paths = sorted(chunk_dir.glob("embeddings_*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no embeddings_*.parquet in {chunk_dir}")
        frames = [pd.read_parquet(p) for p in paths]
        df = pd.concat(frames, ignore_index=True)
        asins = df["parent_asin"].astype(str).to_numpy()
        matrix = np.vstack(df["embedding"].to_numpy()).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # guard: a zero-norm row would divide to NaN and poison every
        # dot-product search against that product
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        np.save(index_path, matrix)
        pd.DataFrame({"parent_asin": asins}).to_parquet(asins_path, index=False)
        print(
            f"index built: {len(asins):,} x {matrix.shape[1]} in {time.time() - t0:.1f}s "
            f"-> {index_path} ({index_path.stat().st_size / 1e9:.2f} GB)"
        )
        return cls(asins, matrix)

    @classmethod
    def load(cls, index_path: Path = INDEX_PATH, asins_path: Path = INDEX_ASINS_PATH):
        matrix = np.load(index_path, mmap_mode="r")
        asins = pd.read_parquet(asins_path, columns=["parent_asin"])[
            "parent_asin"
        ].astype(str).to_numpy()
        return cls(asins, matrix)

    def top_k(self, vec: np.ndarray, k: int = 10):
        vec = vec.astype(np.float32)
        if vec.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                f"query dims {vec.shape[0]} != index dims {self.matrix.shape[1]}"
            )
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        scores = self.matrix @ vec
        idx = np.argpartition(scores, -k)[-k:]
        order = np.argsort(-scores[idx])
        return self.asins[idx[order]], scores[idx[order]]


def load_titles() -> dict[str, str]:
    df = pd.read_parquet(PRODUCT_TEXTS_PATH, columns=["parent_asin", "title"])
    return dict(zip(df["parent_asin"].astype(str), df["title"].astype(str)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: embedding similarity")
    parser.add_argument("--build-index", action="store_true", help="build index from chunks")
    parser.add_argument("--query", default="", help="natural-language query sentence")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.build_index:
        EmbeddingIndex.build()
        return

    if not args.query:
        parser.error("provide --build-index or --query")

    index = EmbeddingIndex.load()
    embedder = QueryEmbedder()
    t0 = time.time()
    vec = embedder.embed(args.query.strip().lower())
    print(f"query embedded ({vec.shape[0]} dims) in {time.time() - t0:.1f}s")

    t0 = time.time()
    asins, scores = index.top_k(vec, args.top_k)
    print(f"searched {len(index.asins):,} products in {time.time() - t0:.2f}s\n")

    titles = load_titles()
    print("=== TOP RECOMMENDATIONS ===")
    for rank, (asin, score) in enumerate(zip(asins, scores), 1):
        print(f"{rank:2d}. {score:.4f}  {asin}  {titles.get(asin, '')[:80]}")


if __name__ == "__main__":
    main()
