"""Phase 4 verification: validates the embedding chunks downloaded from Kaggle.

Everything skips gracefully until the chunks exist locally at
scratch/kaggle/output/ (the script's default --out dir).

Run:
    python -m unittest discover -s tests -v   (from scratch/)
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
CHUNK_DIR = PROJECT / "kaggle" / "output"
INPUT_FILE = PROJECT / "kaggle" / "input" / "product_texts.parquet"

EXPECTED_TOTAL = 371_902
EXPECTED_DIMS = 1536
EXPECTED_CHUNKS = 38


def load_all(chunk_dir: Path):
    paths = sorted(chunk_dir.glob("embeddings_*.parquet"))
    if not paths:
        return paths, None
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    return paths, df


class TestPhase4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunk_paths, cls.df = load_all(CHUNK_DIR)
        cls.ready = len(cls.chunk_paths) > 0

    def test_chunks_downloaded(self):
        if not self.ready:
            self.skipTest("no chunks in scratch/kaggle/output yet")
        self.assertEqual(len(self.chunk_paths), EXPECTED_CHUNKS)

    def test_total_rows(self):
        if not self.ready:
            self.skipTest("no chunks yet")
        self.assertEqual(len(self.df), EXPECTED_TOTAL)
        self.assertEqual(self.df["parent_asin"].nunique(), EXPECTED_TOTAL)

    def test_dimensions(self):
        if not self.ready:
            self.skipTest("no chunks yet")
        first = self.df["embedding"].iloc[0]
        self.assertEqual(len(first), EXPECTED_DIMS)
        self.assertEqual(self.df["embedding"].apply(len).nunique(), 1)

    def test_float32_finite_consistent_norm(self):
        if not self.ready:
            self.skipTest("no chunks yet")
        arr = np.vstack(self.df["embedding"].to_numpy())
        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(np.isfinite(arr).all())
        norms = np.linalg.norm(arr, axis=1)
        self.assertGreater(norms.min(), 0)
        self.assertLess(norms.std() / norms.mean(), 0.01)

    def test_asin_order_matches_input(self):
        if not self.ready or not INPUT_FILE.exists():
            self.skipTest("chunks or product_texts.parquet missing")
        src = pd.read_parquet(INPUT_FILE, columns=["parent_asin"])
        self.assertTrue(
            (self.df["parent_asin"].to_numpy() == src["parent_asin"].to_numpy()).all()
        )


if __name__ == "__main__":
    unittest.main()
