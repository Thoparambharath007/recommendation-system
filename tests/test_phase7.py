"""Phase 7 tests: hybrid fusion math (no models, no data files needed)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from phase7_fusion import (
    apply_color_preference,
    apply_garment_filter,
    apply_mmr,
    apply_occasion_preference,
    apply_type_preference,
    build_seed_scores,
    detect_garment_type,
    fuse,
    sentiment_effective,
)


def make_sentiment(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["parent_asin", "sentiment_score", "n_reviews"])


class TestSentimentEffective(unittest.TestCase):
    def test_full_reliability_keeps_score(self):
        self.assertAlmostEqual(sentiment_effective(0.8, 10), 0.8)
        self.assertAlmostEqual(sentiment_effective(0.8, 999), 0.8)

    def test_few_reviews_shrinks_toward_neutral(self):
        self.assertAlmostEqual(sentiment_effective(0.8, 1), 0.53)
        self.assertAlmostEqual(sentiment_effective(0.2, 1), 0.47)

    def test_zero_reviews_is_neutral(self):
        self.assertAlmostEqual(sentiment_effective(0.9, 0), 0.5)

    def test_negative_raw_works(self):
        self.assertAlmostEqual(sentiment_effective(0.0, 10), 0.0)


class TestFuse(unittest.TestCase):
    def test_weights_and_ranking(self):
        content = {"A": 1.0, "B": 0.5}
        embed = {"A": 0.5, "B": 0.9}
        sent = make_sentiment([("A", 0.8, 10), ("B", 0.2, 10)])
        out = fuse(["A", "B"], content, embed, sent)
        self.assertEqual(list(out["parent_asin"]), ["A", "B"])
        fused_a = 0.3 * 1.0 + 0.4 * 0.5 + 0.3 * 0.8
        fused_b = 0.3 * 0.5 + 0.4 * 0.9 + 0.3 * 0.2
        self.assertAlmostEqual(out.loc[0, "fused"], round(fused_a, 4))
        self.assertAlmostEqual(out.loc[1, "fused"], round(fused_b, 4))

    def test_missing_sentiment_defaults_neutral(self):
        out = fuse(["A"], {"A": 1.0}, {"A": 0.5}, None)
        self.assertAlmostEqual(out.loc[0, "sentiment_eff"], 0.5)
        self.assertAlmostEqual(out.loc[0, "n_reviews"], 0)

    def test_missing_content_and_embed_default_zero(self):
        out = fuse(["A"], None, None, None)
        self.assertAlmostEqual(out.loc[0, "fused"], 0.3 * 0.0 + 0.4 * 0.0 + 0.3 * 0.5)

    def test_embed_clamped_non_negative(self):
        out = fuse(["A"], None, {"A": -0.9}, None)
        self.assertAlmostEqual(out.loc[0, "embed"], 0.0)

    def test_min_reviews_gates_sentiment(self):
        sent = make_sentiment([("A", 0.9, 1)])
        out = fuse(["A"], {"A": 1.0}, {"A": 0.5}, sent, min_reviews=5)
        self.assertAlmostEqual(out.loc[0, "sentiment_eff"], 0.5)
        out2 = fuse(["A"], {"A": 1.0}, {"A": 0.5}, sent, min_reviews=1)
        self.assertAlmostEqual(out2.loc[0, "sentiment_eff"], 0.54)

    def test_custom_weights(self):
        content = {"A": 1.0, "B": 1.0}
        embed = {"A": 1.0, "B": 0.0}
        sent = make_sentiment([("A", 0.5, 10), ("B", 0.5, 10)])
        out = fuse(["A", "B"], content, embed, sent, w_embed=0.8, w_content=0.1, w_sentiment=0.1)
        self.assertEqual(list(out["parent_asin"]), ["A", "B"])
        self.assertAlmostEqual(out.loc[0, "fused"], 0.1 + 0.8 + 0.05)

    def test_candidate_order_does_not_matter(self):
        content = {"A": 1.0, "B": 0.2}
        embed = {"A": 0.1, "B": 0.9}
        sent = make_sentiment([("A", 0.5, 10), ("B", 0.5, 10)])
        a = fuse(["B", "A"], content, embed, sent)
        b = fuse(["A", "B"], content, embed, sent)
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


class TestColorPreference(unittest.TestCase):
    TITLES = {
        "WHITE1": "Men's White Cotton T Shirt",
        "WHITE2": "white v-neck tee",
        "BLACK1": "Black Slim Fit Polo Shirt",
        "BLUE1": "Blue Oxford Dress Shirt",
        "PLAIN": "Plain Cotton Tee",
    }

    def _result(self, asins, scores):
        return pd.DataFrame(
            {
                "parent_asin": asins,
                "content": [0.5] * len(asins),
                "embed": [0.5] * len(asins),
                "sentiment_eff": [0.5] * len(asins),
                "n_reviews": [1] * len(asins),
                "fused": scores,
            }
        )

    def test_filters_to_matching_color_when_enough(self):
        res = self._result(list(self.TITLES), [0.1, 0.2, 0.3, 0.4, 0.5])
        out = apply_color_preference(res, self.TITLES, "white shirt", top_k=3)
        self.assertEqual(set(out["parent_asin"]), {"WHITE1", "WHITE2"})
        self.assertEqual(len(out), 2)

    def test_filters_to_single_match(self):
        res = self._result(["BLACK1", "BLUE1", "PLAIN"], [0.2, 0.4, 0.6])
        out = apply_color_preference(res, self.TITLES, "black shirt", top_k=10)
        self.assertEqual(list(out["parent_asin"]), ["BLACK1"])

    def test_no_match_returns_unchanged(self):
        res = self._result(["BLACK1", "BLUE1", "PLAIN"], [0.2, 0.4, 0.6])
        out = apply_color_preference(res, self.TITLES, "green shirt", top_k=10)
        self.assertEqual(len(out), 3)
        self.assertEqual(list(out["parent_asin"]), ["BLACK1", "BLUE1", "PLAIN"])

    def test_no_color_word_returns_unchanged(self):
        res = self._result(list(self.TITLES), [0.5] * 5)
        out = apply_color_preference(res, self.TITLES, "slim fit shirt", top_k=3)
        self.assertEqual(len(out), 5)
        pd.testing.assert_frame_equal(out, res)

    def test_color_match_is_case_insensitive(self):
        res = self._result(["WHITE1", "PLAIN"], [0.5, 0.9])
        out = apply_color_preference(res, self.TITLES, "WHITE shirt", top_k=2)
        self.assertEqual(set(out["parent_asin"]), {"WHITE1"})

    def test_plurals_do_not_false_match(self):
        res = self._result(["BLUE1", "PLAIN"], [0.5, 0.9])
        out = apply_color_preference(res, self.TITLES, "blue shirts", top_k=10)
        self.assertEqual(list(out["parent_asin"]), ["BLUE1"])

    def test_primary_color_wins_in_multi_color_query(self):
        res = self._result(["WHITE1", "BLACK1"], [0.5, 0.9])
        out = apply_color_preference(res, self.TITLES, "white shirt with black buttons", top_k=10)
        self.assertEqual(list(out["parent_asin"]), ["WHITE1"])

    def test_first_color_in_query_is_primary(self):
        res = self._result(["WHITE1", "BLACK1"], [0.5, 0.9])
        out = apply_color_preference(res, self.TITLES, "black shirt with white stripes", top_k=10)
        self.assertEqual(list(out["parent_asin"]), ["BLACK1"])

    def test_brand_color_name_does_not_match(self):
        titles = {
            "WW": "White Water Men's Technoflex Long Sleeve Performance Shirt with UV50",
            "W": "Men's White Slim Fit Dress Shirt",
        }
        brands = {"WW": "White Water", "W": ""}
        res = self._result(list(titles), [0.9, 0.1])
        out = apply_color_preference(res, titles, "white shirt", top_k=10, brands=brands)
        self.assertEqual(list(out["parent_asin"]), ["W"])

    def test_real_color_outside_brand_still_matches(self):
        titles = {"BR": "Black Rifle Coffee Company T Shirt Black"}
        brands = {"BR": "Black Rifle Coffee Company"}
        res = self._result(list(titles), [0.5])
        out = apply_color_preference(res, titles, "black shirt", top_k=10, brands=brands)
        self.assertEqual(list(out["parent_asin"]), ["BR"])


class TestTypePreference(unittest.TestCase):
    TITLES = {
        "DRESS": "Men's White Slim Fit Dress Shirt",
        "OXFORD": "Blue Oxford Button Down Shirt",
        "TEE1": "White Cotton T-Shirt",
        "TEE2": "Black Graphic T Shirt",
        "TEE3": "Red Tee Shirt",
        "HOODIE": "Grey Pullover Hoodie",
    }

    def _result(self, asins, scores):
        return pd.DataFrame(
            {
                "parent_asin": asins,
                "content": [0.5] * len(asins),
                "embed": [0.5] * len(asins),
                "sentiment_eff": [0.5] * len(asins),
                "n_reviews": [1] * len(asins),
                "fused": scores,
            }
        )

    def test_detect_shirt(self):
        self.assertEqual(detect_garment_type("white shirt"), "shirt")
        self.assertEqual(detect_garment_type("dress shirt for office"), "shirt")

    def test_detect_tee(self):
        self.assertEqual(detect_garment_type("black tshirt"), "tee")
        self.assertEqual(detect_garment_type("white t-shirt"), "tee")
        self.assertEqual(detect_garment_type("cool tee"), "tee")
        self.assertEqual(detect_garment_type("blue t shirt"), "tee")

    def test_detect_none(self):
        self.assertEqual(detect_garment_type("flannel"), "")
        self.assertEqual(detect_garment_type("hoodie"), "")

    def test_shirt_excludes_tees(self):
        res = self._result(list(self.TITLES), [0.5] * 6)
        out = apply_type_preference(res, self.TITLES, "white shirt")
        self.assertEqual(set(out["parent_asin"]), {"DRESS", "OXFORD"})

    def test_tee_excludes_dress_shirts(self):
        res = self._result(list(self.TITLES), [0.5] * 6)
        out = apply_type_preference(res, self.TITLES, "black tshirt")
        self.assertEqual(set(out["parent_asin"]), {"TEE1", "TEE2", "TEE3"})

    def test_no_type_word_unchanged(self):
        res = self._result(list(self.TITLES), [0.5] * 6)
        out = apply_type_preference(res, self.TITLES, "flannel")
        self.assertEqual(len(out), 6)

    def test_no_match_falls_back_unchanged(self):
        res = self._result(["HOODIE"], [0.5])
        out = apply_type_preference(res, self.TITLES, "white shirt")
        self.assertEqual(list(out["parent_asin"]), ["HOODIE"])


class TestGarmentFilter(unittest.TestCase):
    TITLES = {
        "SHIRT1": "Men's White Slim Fit Dress Shirt",
        "TEE1": "White Cotton T-Shirt",
        "PIN1": "A N KINGPiiN White Fighter Jet Airplane Lapel Pin Badge Gift Party Shirt Collar Costume Pin Accessories For Men Brooch",
        "PIN2": "Cool White Lapel Pin for Men",
        "SOCK": "Men's White Sports Socks 6 Pack",
        "WATCH": "White Digital Sports Watch for Men",
        "TIEDYE": "White Tie Dye T Shirt",
    }

    def _result(self, asins, scores):
        return pd.DataFrame(
            {
                "parent_asin": asins,
                "content": [0.5] * len(asins),
                "embed": [0.5] * len(asins),
                "sentiment_eff": [0.5] * len(asins),
                "n_reviews": [1] * len(asins),
                "fused": scores,
            }
        )

    def test_drops_accessories(self):
        res = self._result(list(self.TITLES), [0.5] * 7)
        out = apply_garment_filter(res, self.TITLES)
        self.assertEqual(set(out["parent_asin"]), {"SHIRT1", "TEE1", "TIEDYE"})

    def test_keeps_garments_when_no_accessories(self):
        res = self._result(["SHIRT1", "TEE1"], [0.5, 0.6])
        out = apply_garment_filter(res, self.TITLES)
        self.assertEqual(len(out), 2)

    def test_all_accessories_falls_back_unchanged(self):
        res = self._result(["PIN2", "WATCH"], [0.5, 0.5])
        out = apply_garment_filter(res, self.TITLES)
        self.assertEqual(len(out), 2)


class TestOccasionPreference(unittest.TestCase):
    TITLES = {
        "DRESS": "Men's White Slim Fit Dress Shirt",
        "OXFORD": "Blue Oxford Button Down Shirt",
        "CASUAL": "White Casual Poplin Shirt",
        "TANK": "Men's Tank Top Mama Tried Country Music Rebel Shirt White",
        "NOVELTY": "Personalized Grammy Grandma Sunflower Shirt Gifts Birthday White",
        "TEE": "White Cotton T-Shirt",
    }

    def _result(self, asins, scores):
        return pd.DataFrame(
            {
                "parent_asin": asins,
                "content": [0.5] * len(asins),
                "embed": [0.5] * len(asins),
                "sentiment_eff": [0.5] * len(asins),
                "n_reviews": [1] * len(asins),
                "fused": scores,
            }
        )

    def test_office_query_keeps_office_shirts(self):
        res = self._result(list(self.TITLES), [0.5] * 6)
        out = apply_occasion_preference(res, self.TITLES, "white shirt for office")
        self.assertEqual(set(out["parent_asin"]), {"DRESS", "OXFORD"})

    def test_office_query_bans_tank_and_novelty(self):
        res = self._result(["TANK", "NOVELTY", "DRESS"], [0.9, 0.9, 0.1])
        out = apply_occasion_preference(res, self.TITLES, "white shirt for office")
        self.assertEqual(list(out["parent_asin"]), ["DRESS"])

    def test_non_office_query_unchanged(self):
        res = self._result(list(self.TITLES), [0.5] * 6)
        out = apply_occasion_preference(res, self.TITLES, "white shirt")
        self.assertEqual(len(out), 6)

    def test_no_office_titles_relaxes_to_ban_novelty(self):
        titles = dict(self.TITLES)
        titles["TANK"] = "Men's White Sleeveless Tank Top"
        res = self._result(["NOVELTY", "TANK", "TEE"], [0.9, 0.8, 0.1])
        out = apply_occasion_preference(res, titles, "work shirt")
        self.assertEqual(set(out["parent_asin"]), {"TANK", "TEE"})


class TestCollabFusion(unittest.TestCase):
    def test_collab_score_lifts_ranking(self):
        # X has weak content/embed but a strong co-purchase signal
        out = fuse(
            ["A", "X", "Z"],
            {"A": 1.0, "X": 0.2},
            {"A": 0.5, "X": 0.1},
            None,
            collab_scores={"X": 1.0},
        )
        fused = dict(zip(out["parent_asin"], out["fused"]))
        self.assertGreater(fused["X"], fused["Z"])
        self.assertAlmostEqual(fused["X"], round(0.3 * 0.2 + 0.4 * 0.1 + 0.3 * 0.5 + 0.15 * 1.0, 4))

    def test_no_collab_scores_defaults_zero(self):
        out = fuse(["A"], {"A": 1.0}, {"A": 0.5}, None)
        self.assertAlmostEqual(out.loc[0, "collab"], 0.0)

    def test_collab_clamped_non_negative(self):
        out = fuse(["A"], {"A": 1.0}, {"A": 0.5}, None, collab_scores={"A": -0.7})
        self.assertAlmostEqual(out.loc[0, "collab"], 0.0)


class TestBuildSeedScores(unittest.TestCase):
    def test_union_normalized_to_one(self):
        seeds = build_seed_scores({"A": 1.0, "B": 0.5}, {"C": 0.25}, k=2)
        self.assertAlmostEqual(seeds["A"], 1.0)
        self.assertAlmostEqual(max(seeds.values()), 1.0)
        self.assertIn("C", seeds)

    def test_empty_inputs(self):
        self.assertEqual(build_seed_scores(None, None), {})

    def test_k_limits_each_source(self):
        seeds = build_seed_scores({"A": 1.0, "B": 0.9, "C": 0.1}, None, k=2)
        self.assertEqual(len(seeds), 2)
        self.assertIn("A", seeds)


class TestApplyMMR(unittest.TestCase):
    TITLES = {
        "A": "white formal dress shirt",
        "B": "white formal dress shirt slim",
        "C": "blue denim jeans jacket",
        "D": "black casual tee",
        "E": "red polo shirt",
    }

    def _df(self):
        return pd.DataFrame(
            {"parent_asin": list(self.TITLES), "fused": [0.9, 0.88, 0.5, 0.4, 0.3]}
        )

    def test_keeps_all_items_and_top_rank(self):
        out = apply_mmr(self._df(), self.TITLES)
        self.assertEqual(set(out["parent_asin"]), set(self.TITLES))
        self.assertEqual(out["parent_asin"].iloc[0], "A")

    def test_diversifies_near_duplicates(self):
        out = apply_mmr(self._df(), self.TITLES, lam=0.0)  # pure diversity
        order = list(out["parent_asin"])
        # B is a near-duplicate of A — it must not be picked right after A
        self.assertNotEqual(order[1], "B")

    def test_small_input_unchanged(self):
        df = pd.DataFrame({"parent_asin": ["A", "B"], "fused": [0.9, 0.8]})
        out = apply_mmr(df, self.TITLES)
        self.assertEqual(list(out["parent_asin"]), ["A", "B"])

    def test_missing_titles_safe(self):
        out = apply_mmr(self._df(), {})
        self.assertEqual(len(out), 5)


class TestTypeFallbackModes(unittest.TestCase):
    TITLES = {"HOODIE": "grey hoodie", "TEE": "white tee"}

    def _df(self, asins):
        return pd.DataFrame({"parent_asin": asins, "fused": [0.5] * len(asins)})

    def test_default_fallback_returns_pool(self):
        out = apply_type_preference(self._df(["HOODIE"]), self.TITLES, "white shirt")
        self.assertEqual(list(out["parent_asin"]), ["HOODIE"])

    def test_strict_mode_returns_empty(self):
        out = apply_type_preference(
            self._df(["HOODIE"]), self.TITLES, "white shirt", fallback=False
        )
        self.assertEqual(len(out), 0)

    def test_strict_mode_with_match(self):
        out = apply_type_preference(
            self._df(["HOODIE", "TEE"]), self.TITLES, "white tee", fallback=False
        )
        self.assertEqual(list(out["parent_asin"]), ["TEE"])


if __name__ == "__main__":
    unittest.main()
