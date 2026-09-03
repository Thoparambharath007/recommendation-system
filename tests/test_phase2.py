import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from phase2_ibcf import build_copurchase_edges, get_neighbors


def make_csv(rows: list[list[str]], path: Path) -> None:
    lines = ["user_id,parent_asin,rating,timestamp"]
    lines += [",".join(r) for r in rows]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


class TestPhase2(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.tmp.name) / "reviews.csv.gz"
        self.catalog = {"A", "B", "C", "D"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_basic_pairs(self):
        make_csv(
            [
                ["u1", "A", "5.0", "1"],
                ["u1", "B", "4.0", "2"],
                ["u2", "A", "5.0", "3"],
                ["u2", "C", "3.0", "4"],
                ["u3", "B", "2.0", "5"],
            ],
            self.csv_path,
        )
        edges = build_copurchase_edges(self.csv_path, catalog=self.catalog)
        weights = {(a, b): w for a, b, w in edges.itertuples(index=False)}
        self.assertEqual(weights[("A", "B")], 1)
        self.assertEqual(weights[("A", "C")], 1)
        self.assertNotIn(("B", "C"), weights)

    def test_single_product_users_produce_no_edges(self):
        make_csv(
            [
                ["u1", "A", "5.0", "1"],
                ["u2", "B", "4.0", "2"],
            ],
            self.csv_path,
        )
        edges = build_copurchase_edges(self.csv_path, catalog=self.catalog)
        self.assertEqual(len(edges), 0)

    def test_products_outside_catalog_ignored(self):
        make_csv(
            [
                ["u1", "A", "5.0", "1"],
                ["u1", "ZZZ", "4.0", "2"],
                ["u1", "B", "4.0", "3"],
                ["u2", "ZZZ", "4.0", "4"],
            ],
            self.csv_path,
        )
        edges = build_copurchase_edges(self.csv_path, catalog=self.catalog)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges.iloc[0]["product_a"], "A")
        self.assertEqual(edges.iloc[0]["product_b"], "B")

    def test_weight_counts_multiple_users(self):
        make_csv(
            [
                ["u1", "A", "5.0", "1"],
                ["u1", "B", "4.0", "2"],
                ["u2", "A", "5.0", "3"],
                ["u2", "B", "4.0", "4"],
                ["u3", "A", "5.0", "5"],
                ["u3", "B", "4.0", "6"],
            ],
            self.csv_path,
        )
        edges = build_copurchase_edges(self.csv_path, catalog=self.catalog)
        self.assertEqual(edges.iloc[0]["weight"], 3)

    def test_max_products_per_user_cap(self):
        catalog = {f"P{i:03d}" for i in range(60)}
        rows = [[f"u1", f"P{i:03d}", "5.0", str(i)] for i in range(60)]
        make_csv(rows, self.csv_path)
        edges = build_copurchase_edges(self.csv_path, catalog=catalog, max_per_user=50)
        self.assertEqual(len(edges), 50 * 49 // 2)
        self.assertNotIn("P050", set(edges["product_a"]) | set(edges["product_b"]))

    def test_get_neighbors_ranks_by_weight(self):
        edges = pd.DataFrame(
            {
                "product_a": ["A", "A", "B", "A"],
                "product_b": ["B", "C", "D", "D"],
                "weight": [5, 1, 2, 3],
            }
        )
        neighbors = get_neighbors(edges, "A")
        self.assertEqual(neighbors.iloc[0]["neighbor"], "B")
        self.assertEqual(neighbors.iloc[0]["weight"], 5)
        self.assertNotIn("D", neighbors.iloc[0]["neighbor"])


if __name__ == "__main__":
    unittest.main()
