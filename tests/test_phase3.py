import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from phase3_text import assemble_texts, clean_review_text, collect_reviews


def make_reviews(records: list[dict], path: Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class TestPhase3(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reviews_path = Path(self.tmp.name) / "reviews.jsonl.gz"
        self.catalog = {"A", "B", "C"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_clean_review_text(self):
        self.assertEqual(clean_review_text("  great   shirt!\n"), "great shirt!")

    def test_collect_reviews_groups_by_product(self):
        make_reviews(
            [
                {"parent_asin": "A", "text": "Great fit."},
                {"parent_asin": "A", "text": "Love it."},
                {"parent_asin": "B", "text": "Too small."},
                {"parent_asin": "ZZZ", "text": "Ignored."},
            ],
            self.reviews_path,
        )
        reviews = collect_reviews(self.reviews_path, catalog=self.catalog)
        self.assertEqual(len(reviews["A"]), 2)
        self.assertEqual(len(reviews["B"]), 1)
        self.assertNotIn("ZZZ", reviews)

    def test_max_reviews_per_product(self):
        make_reviews(
            [{"parent_asin": "A", "text": f"review {i}"} for i in range(25)],
            self.reviews_path,
        )
        reviews = collect_reviews(self.reviews_path, catalog=self.catalog, max_per_product=3)
        self.assertEqual(len(reviews["A"]), 3)

    def test_assemble_texts_joins_metadata_and_reviews(self):
        make_reviews(
            [{"parent_asin": "A", "text": "soft cotton shirt"}],
            self.reviews_path,
        )
        reviews = collect_reviews(self.reviews_path, catalog=self.catalog)
        metadata = pd.DataFrame(
            {
                "parent_asin": ["A", "B"],
                "description": ["nike dri-fit polo", "hanes tee"],
                "features": ["moisture wicking", "100% cotton"],
                "title": ["Nike Polo", "Hanes Tee"],
            }
        )
        df = assemble_texts(metadata, reviews)
        row = df[df["parent_asin"] == "A"].iloc[0]
        self.assertIn("nike dri-fit polo", row["full_text"])
        self.assertIn("moisture wicking", row["full_text"])
        self.assertIn("soft cotton shirt", row["full_text"])
        row_b = df[df["parent_asin"] == "B"].iloc[0]
        self.assertNotIn("soft cotton shirt", row_b["full_text"])

    def test_empty_reviews_ok(self):
        make_reviews([{"parent_asin": "A", "text": "ok"}], self.reviews_path)
        reviews = collect_reviews(self.reviews_path, catalog=self.catalog)
        metadata = pd.DataFrame(
            {
                "parent_asin": ["B"],
                "description": [""],
                "features": [""],
                "title": ["B Tee"],
            }
        )
        df = assemble_texts(metadata, reviews)
        row = df.iloc[0]
        self.assertEqual(row["reviews_text"], "")
        self.assertEqual(row["full_text"], "")


if __name__ == "__main__":
    unittest.main()
