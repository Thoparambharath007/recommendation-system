"""Phase 6 tests: review extraction, sentiment mapping, aggregation (no Flair needed)."""

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from phase6_sentiment import (
    aggregate_sentiment,
    clean_review_text,
    extract_reviews,
    map_sentiment,
)


def make_reviews(records: list[dict], path: Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class TestPhase6(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reviews_path = Path(self.tmp.name) / "reviews.jsonl.gz"
        self.catalog = {"A", "B"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_clean_review_text(self):
        self.assertEqual(clean_review_text("  love   it!!\n"), "love it!!")
        self.assertEqual(clean_review_text(None), "")

    def test_map_sentiment(self):
        self.assertAlmostEqual(map_sentiment("POSITIVE", 1.0), 1.0)
        self.assertAlmostEqual(map_sentiment("POSITIVE", 0.6), 0.8)
        self.assertAlmostEqual(map_sentiment("NEGATIVE", 0.8), 0.1)
        self.assertAlmostEqual(map_sentiment("NEGATIVE", 0.0), 0.5)
        self.assertEqual(map_sentiment("NEUTRAL", 0.9), 0.5)

    def test_extract_reviews_one_row_per_review(self):
        make_reviews(
            [
                {"parent_asin": "A", "text": "Great fit."},
                {"parent_asin": "A", "text": "Love it."},
                {"parent_asin": "B", "text": "Too small."},
                {"parent_asin": "ZZZ", "text": "Ignored."},
            ],
            self.reviews_path,
        )
        rows = extract_reviews(self.reviews_path, catalog=self.catalog)
        self.assertEqual(len(rows), 3)
        self.assertNotIn("ZZZ", [r["parent_asin"] for r in rows])

    def test_extract_reviews_caps_per_product(self):
        make_reviews(
            [{"parent_asin": "A", "text": f"review {i}"} for i in range(25)],
            self.reviews_path,
        )
        rows = extract_reviews(self.reviews_path, catalog=self.catalog, max_per_product=3)
        self.assertEqual(len(rows), 3)

    def test_extract_skips_empty_texts(self):
        make_reviews(
            [{"parent_asin": "A", "text": ""}, {"parent_asin": "A", "text": "ok"}],
            self.reviews_path,
        )
        rows = extract_reviews(self.reviews_path, catalog=self.catalog)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["review_text"], "ok")

    def test_aggregate_sentiment(self):
        scores = pd.DataFrame(
            {
                "parent_asin": ["A", "A", "B"],
                "sentiment": [1.0, 0.0, 0.5],
            }
        )
        agg = aggregate_sentiment(scores)
        self.assertEqual(len(agg), 2)
        self.assertAlmostEqual(agg.loc[0, "sentiment_score"], 0.5)
        self.assertEqual(agg.loc[0, "n_reviews"], 2)
        self.assertEqual(agg.loc[1, "n_reviews"], 1)


if __name__ == "__main__":
    unittest.main()
