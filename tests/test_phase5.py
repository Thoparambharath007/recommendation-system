"""Phase 5 tests: EmbeddingIndex build/load/top-k (no models needed)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from phase5_similarity import EMBEDDING_CHUNKS_DIR, EmbeddingIndex


def make_chunks(tmp: Path, n_chunks: int = 2, rows: int = 4, dims: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    asins = []
    for c in range(n_chunks):
        chunk_asins = [f"B{c:02d}{i:04d}" for i in range(rows)]
        if c == 0:
            emb = np.array(
                [
                    [1.0, 0, 0, 0, 0, 0, 0, 0],
                    [0.9, 0.1, 0, 0, 0, 0, 0, 0],
                    [0.8, 0.2, 0, 0, 0, 0, 0, 0],
                    [0.0, 0.0, 1.0, 0, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            )
        else:
            emb = rng.normal(size=(rows, dims)).astype(np.float32)
            emb[:, :3] = 0.0
        pd.DataFrame({"parent_asin": chunk_asins, "embedding": list(emb)}).to_parquet(
            tmp / f"embeddings_{c:05d}.parquet", index=False
        )
        asins.extend(chunk_asins)
    return asins


class TestEmbeddingIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.chunk_dir = Path(self.tmp.name) / "chunks"
        self.chunk_dir.mkdir()
        self.asins = make_chunks(self.chunk_dir)
        self.index_path = Path(self.tmp.name) / "index.npy"
        self.asins_path = Path(self.tmp.name) / "asins.parquet"

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_orders_asins_like_chunks(self):
        index = EmbeddingIndex.build(self.chunk_dir, self.index_path, self.asins_path)
        self.assertEqual(list(index.asins), self.asins)
        self.assertEqual(index.matrix.shape, (len(self.asins), 8))
        self.assertEqual(index.matrix.dtype, np.float32)

    def test_rows_unit_norm(self):
        index = EmbeddingIndex.build(self.chunk_dir, self.index_path, self.asins_path)
        norms = np.linalg.norm(index.matrix, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))

    def test_top_k_ranks_by_cosine(self):
        index = EmbeddingIndex.build(self.chunk_dir, self.index_path, self.asins_path)
        vec = index.matrix[0].copy()
        top_asins, scores = index.top_k(vec, k=2)
        self.assertEqual(top_asins[0], self.asins[0])  # exact match wins
        self.assertEqual(top_asins[1], self.asins[1])  # nearest neighbor next
        self.assertGreaterEqual(scores[0], scores[1])

    def test_scale_invariance(self):
        index = EmbeddingIndex.build(self.chunk_dir, self.index_path, self.asins_path)
        a, sa = index.top_k(index.matrix[0] * 3.0, k=2)
        b, sb = index.top_k(index.matrix[0], k=2)
        self.assertEqual(list(a), list(b))

    def test_load_roundtrip(self):
        EmbeddingIndex.build(self.chunk_dir, self.index_path, self.asins_path)
        loaded = EmbeddingIndex.load(self.index_path, self.asins_path)
        self.assertEqual(list(loaded.asins), self.asins)
        np.testing.assert_array_equal(loaded.matrix, np.load(self.index_path))

    def test_dims_mismatch_raises(self):
        index = EmbeddingIndex.build(self.chunk_dir, self.index_path, self.asins_path)
        with self.assertRaises(ValueError):
            index.top_k(np.zeros(5, dtype=np.float32))

    def test_missing_chunks_raise(self):
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            EmbeddingIndex.build(empty, self.index_path, self.asins_path)

    def test_chunk_dir_default_exists(self):
        self.assertTrue(EMBEDDING_CHUNKS_DIR.exists())


if __name__ == "__main__":
    unittest.main()
