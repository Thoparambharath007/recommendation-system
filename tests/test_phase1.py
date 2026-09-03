import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from content_filter import ContentFilter
from preprocessing import preprocess_metadata

SAMPLE = pd.DataFrame(
    {
        "parent_asin": ["A1", "A2", "A3", "A4"],
        "brand": ["Hanes", "Hanes", "Nike", "Calvin Klein"],
        "title": [
            "Hanes Men's Long Sleeve T-Shirt Black XL",
            "Hanes Men's V-Neck T-Shirt Black XL",
            "Nike Men's Dri-FIT Polo Blue Large",
            "Calvin Klein Men's Dress Shirt White Medium",
        ],
        "categories": [
            "Clothing Shoes Jewelry Men Clothing Shirts T-Shirts",
            "Clothing Shoes Jewelry Men Clothing Shirts T-Shirts",
            "Clothing Shoes Jewelry Men Clothing Active Active Shirts Tees Polo",
            "Clothing Shoes Jewelry Men Clothing Shirts Dress Shirts",
        ],
        "description": ["", "", "", ""],
        "features": ["", "", "", ""],
        "bought_together": ["[]", "[]", "[]", "[]"],
    }
)


class TestPhase1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = preprocess_metadata(SAMPLE)
        cls.cf = ContentFilter(cls.df)

    def test_preprocessing_string(self):
        row = self.df[self.df["parent_asin"] == "A1"].iloc[0]
        self.assertIn("hanes", row["metadata_string"])
        self.assertIn("xl", row["metadata_string"])
        self.assertIn("black", row["metadata_string"])

    def test_exact_match_is_top(self):
        top = self.cf.get_top_k("hanes black xl", k=4)
        self.assertEqual(top.iloc[0]["parent_asin"], "A1")
        self.assertGreater(top.iloc[0]["score"], 0.5)

    def test_top_k_count(self):
        top = self.cf.get_top_k("nike blue large", k=2)
        self.assertEqual(len(top), 2)

    def test_wrong_brand_scores_lower(self):
        top = self.cf.get_top_k("hanes black xl", k=4)
        nike = top[top["parent_asin"] == "A3"]
        hanes = top[top["parent_asin"] == "A1"]
        self.assertLess(nike.iloc[0]["score"], hanes.iloc[0]["score"])


if __name__ == "__main__":
    unittest.main()
