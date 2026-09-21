"""Phase 7 tests: hybrid fusion math (no models, no data files needed)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from phase7_fusion import (
    apply_color_preference,
    apply_gender_filter,
    apply_garment_filter,
    apply_mmr,
    apply_novelty_filter,
    apply_occasion_preference,
    apply_plain_shirt_preference,
    apply_secondary_color_boost,
    apply_type_preference,
    build_garment_class_map,
    build_neighbor_map,
    build_seed_scores,
    detect_garment_type,
    fuse,
    parse_price_range,
    quality_rerank,
    renormalize_signals,
    sentiment_effective,
    sentiment_to_dict,
)
from preprocessing import classify_garment
from web_app import clean_display_text


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

    def test_brand_color_prefix_not_matched_when_real_color_differs(self):
        # 'Newport Blue' is a brand prefix (Amazon records brand 'Generic');
        # the real garment color is the trailing word. 'Ocean Blue' with no
        # other color in the title is a legit shade and must still match.
        titles = {
            "NPB_Y": "Newport Blue Men's T-Shirt, Tropical Therapy, Yellow",
            "NPB_W": "Newport Blue Men's T-Shirt, Palm Trees Insta Snaps, White",
            "NPB_T": "Newport Blue Men's T-Shirt, 70s Classic Rides, Camaro, American Made Cars,Teal",
            "NPB_B": "Newport Blue Men's T-Shirt, Big Kahuna, Ocean Blue",
            "REAL_B": "Nike Mens Training Short Sleeve T-Shirt Blue",
        }
        res = self._result(list(titles), [0.1, 0.2, 0.3, 0.4, 0.5])
        out = apply_color_preference(res, titles, "blue tshirt", top_k=10)
        # NPB_T is dropped too: Amazon glues the variant color with no space
        # ('Cars,Teal'), and a comma counts as a token boundary.
        self.assertEqual(set(out["parent_asin"]), {"NPB_B", "REAL_B"})

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

    def test_detect_polo(self):
        self.assertEqual(detect_garment_type("black polo"), "polo")
        self.assertEqual(detect_garment_type("black polo shirt"), "polo")
        self.assertEqual(detect_garment_type("black henley"), "polo")
        self.assertEqual(detect_garment_type("flannel shirt"), "shirt")

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


class TestPlainShirtPreference(unittest.TestCase):
    """User-reported: 'black shirt' always returned black T-shirts. Root
    cause: merch sellers stuff 'Black Shirt' into T-shirt titles for Amazon
    SEO; such titles have the word 'shirt' (so the type filter keeps them)
    but no garment spec word and no tee marker. The plain-shirt guard keeps
    only real-shirt titles for bare color+type queries."""

    TITLES = {
        "MERCH1": "Vital_Apparel Yhlqmdlg Bad Bunny Black Shirt",
        "MERCH2": "Gatti Boxing Club Organized Crime Mafia Mens Shirt X-Large Black",
        "MERCH3": "Chemical Guys SHE601M I Love Detailing Shirt Black Medium",
        "MERCH4": "by The Sword of The Samurai Shirt Black",
        "MERCH5": "ALFASHIRT Kurt Knispel Panzerkommandant Panzer Replacement Shirt",
        "REAL1": "Stylish Smart Dress Shirts for Men Pockets Cotton Black S",
        "REAL2": "Men's Wrinkle Free Dress Shirt Slim Fit Black XL",
        "REAL3": "UNTUCKit Black Stone Wrinkle Free Untucked Shirt for Men",
        "REAL4": "NUTEXROL Men's Leaves Prints Casual Short Sleeve Button Down Cotton",
    }
    MERCH = list(k for k in TITLES if k.startswith("MERCH"))
    REAL = list(k for k in TITLES if k.startswith("REAL"))

    def _df(self, asins):
        return pd.DataFrame({"parent_asin": asins, "fused": [0.5] * len(asins)})

    def test_bare_black_shirt_drops_merch(self):
        out = apply_plain_shirt_preference(self._df(list(self.TITLES)), self.TITLES, "black shirt")
        self.assertEqual(set(out["parent_asin"]), set(self.REAL))

    def test_bare_white_shirt_for_men_is_bare(self):
        out = apply_plain_shirt_preference(self._df(list(self.TITLES)), self.TITLES, "white shirt for men")
        self.assertEqual(set(out["parent_asin"]), set(self.REAL))

    def test_modified_query_untouched(self):
        # 'flannel' is a modifier -> guard off, result unchanged
        res = self._df(list(self.TITLES))
        out = apply_plain_shirt_preference(res, self.TITLES, "black flannel shirt")
        pd.testing.assert_frame_equal(out, res)

    def test_occasion_query_untouched(self):
        res = self._df(list(self.TITLES))
        out = apply_plain_shirt_preference(res, self.TITLES, "black shirt for office")
        pd.testing.assert_frame_equal(out, res)

    def test_price_query_untouched(self):
        res = self._df(list(self.TITLES))
        out = apply_plain_shirt_preference(res, self.TITLES, "black shirt under 2000 rupees")
        pd.testing.assert_frame_equal(out, res)

    def test_all_merch_falls_back_unchanged(self):
        out = apply_plain_shirt_preference(self._df(self.MERCH), self.TITLES, "black shirt")
        self.assertEqual(set(out["parent_asin"]), set(self.MERCH))

    def test_tee_query_untouched(self):
        # 'black tshirt' -> garment type 'tee', guard off
        res = self._df(list(self.TITLES))
        out = apply_plain_shirt_preference(res, self.TITLES, "black tshirt")
        pd.testing.assert_frame_equal(out, res)


class TestShirtQueryExcludesHoodies(unittest.TestCase):
    """Regression: 'shirt' appearing inside merch phrases ('Shirt Warehouse
    ... on a Black Hoodie') fooled the type filter — hoodies/sweatshirts are
    not shirts and must not answer a 'shirt' query."""

    TITLES = {
        "HOOD": "Shirt Warehouse UK Kentucky Love KY on a Black Hoodie",
        "SWEAT": "Men's Black Graphic Sweatshirt",
        "DRESS": "Men's Black Slim Fit Dress Shirt",
    }

    def _df(self, asins):
        return pd.DataFrame({"parent_asin": asins, "fused": [0.5] * len(asins)})

    def test_shirt_query_excludes_hoodie_and_sweatshirt(self):
        out = apply_type_preference(self._df(list(self.TITLES)), self.TITLES, "black shirt")
        self.assertEqual(list(out["parent_asin"]), ["DRESS"])

    def test_hoodie_query_has_no_type_filter(self):
        # 'black hoodie' carries no shirt/tee marker -> type filter is a no-op
        out = apply_type_preference(self._df(["HOOD"]), self.TITLES, "black hoodie")
        self.assertEqual(list(out["parent_asin"]), ["HOOD"])


class TestBrandStripCaseSensitivity(unittest.TestCase):
    """Regression: the brands dict is lowercased (clean_text) while titles
    keep raw case. Brand stripping must happen on the lowercased title, and
    must work for brands containing color words that are NOT in
    COLOR_NAME_COLLISIONS (the collision list alone would mask the bug)."""

    TITLES = {
        "BS_GREEN": "Blue Sky Men's Green Cotton T Shirt",
        "BS_BLUE": "Blue Sky Men's Blue Cotton T Shirt",
        "B": "Men's Blue Slim Fit Polo Shirt",
    }
    BRANDS = {"BS_GREEN": "blue sky", "BS_BLUE": "blue sky", "B": ""}

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

    def test_lowercase_brand_color_word_does_not_false_match(self):
        res = self._result(list(self.TITLES), [0.9, 0.8, 0.1])
        out = apply_color_preference(
            res, self.TITLES, "blue shirt", top_k=10, brands=self.BRANDS
        )
        # BS_GREEN must NOT match (only the brand says "blue"); BS_BLUE and B do.
        self.assertEqual(set(out["parent_asin"]), {"BS_BLUE", "B"})

    def test_real_color_survives_brand_strip(self):
        res = self._result(["BS_BLUE"], [0.5])
        out = apply_color_preference(
            res, self.TITLES, "blue shirt", top_k=10, brands=self.BRANDS
        )
        self.assertEqual(list(out["parent_asin"]), ["BS_BLUE"])


class TestGarmentFilterPlurals(unittest.TestCase):
    """Regression: 'Hats'/'Boots' (plural forms absent from ACCESSORY_WORDS
    verbatim) used to leak past the exact-word check."""

    TITLES = {
        "HATS": "Men's Baseball Caps Hats Collection",
        "BOOTS": "Men's Leather Boots Brown",
        "SHIRT1": "Men's White Slim Fit Dress Shirt",
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

    def test_plural_accessory_words_are_dropped(self):
        res = self._result(list(self.TITLES), [0.5] * 3)
        out = apply_garment_filter(res, self.TITLES)
        self.assertEqual(set(out["parent_asin"]), {"SHIRT1"})

    def test_singular_still_dropped(self):
        res = self._result(["BOOTS", "SHIRT1"], [0.5, 0.6])
        out = apply_garment_filter(res, {"BOOTS": "Men's Leather Boot", "SHIRT1": self.TITLES["SHIRT1"]})
        self.assertEqual(list(out["parent_asin"]), ["SHIRT1"])

    def test_tuxedo_studs_are_accessories(self):
        # regression from a live 'black shirts' run: a tuxedo-studs listing
        # (no 'shirt' in the title, high fused score) leaked into results
        titles = {
            "STUDS": "AO OA Set of Four Rhodium Plated Black Onyx Tuxedo Studs",
            "SHIRT1": self.TITLES["SHIRT1"],
        }
        res = self._result(["STUDS", "SHIRT1"], [0.9, 0.5])
        out = apply_garment_filter(res, titles)
        self.assertEqual(list(out["parent_asin"]), ["SHIRT1"])


class TestParsePriceRange(unittest.TestCase):
    def test_range_without_price_word(self):
        # "1000 to 2000" is a price signal even without "under"/"price";
        # bare numbers >= 100 are treated as INR (catalog is USD).
        self.assertEqual(parse_price_range("shirt 1000 to 2000"), (12.05, 24.1))

    def test_range_with_hyphen_small_values_stay_usd(self):
        self.assertEqual(parse_price_range("tshirt 15-30"), (15.0, 30.0))

    def test_usd_marker_forces_usd(self):
        self.assertEqual(parse_price_range("under $50"), (None, 50.0))

    def test_inr_marker_converts(self):
        self.assertEqual(parse_price_range("under 2000 rupees"), (None, 24.1))

    def test_size_number_is_not_a_price(self):
        self.assertEqual(parse_price_range("black shirt size 42"), (None, None))

    def test_no_digits_no_price(self):
        self.assertEqual(parse_price_range("black shirt"), (None, None))


class TestFuseSentimentDict(unittest.TestCase):
    def test_prebuilt_dict_matches_dataframe(self):
        sent = pd.DataFrame(
            [("A", 0.8, 10), ("B", 0.2, 1)],
            columns=["parent_asin", "sentiment_score", "n_reviews"],
        )
        out_df = fuse(["A", "B"], {"A": 1.0, "B": 0.5}, {"A": 0.5, "B": 0.9}, sent)
        out_dict = fuse(
            ["A", "B"], {"A": 1.0, "B": 0.5}, {"A": 0.5, "B": 0.9}, sentiment_to_dict(sent)
        )
        pd.testing.assert_frame_equal(out_df, out_dict)

    def test_sentiment_to_dict_values(self):
        sent = pd.DataFrame(
            [("A", 0.8, 10)], columns=["parent_asin", "sentiment_score", "n_reviews"]
        )
        d = sentiment_to_dict(sent)
        self.assertEqual(d["A"], (0.8, 10))


class TestBuildNeighborMap(unittest.TestCase):
    def test_symmetric_and_normalized_by_global_max(self):
        edges = pd.DataFrame(
            {"product_a": ["A", "B"], "product_b": ["B", "C"], "weight": [10, 20]}
        )
        adj = build_neighbor_map(edges)
        self.assertIn(("B", 0.5), adj["A"])
        self.assertIn(("A", 0.5), adj["B"])
        self.assertIn(("C", 1.0), adj["B"])
        self.assertIn(("B", 1.0), adj["C"])

    def test_numeric_asins_coerced_to_str(self):
        edges = pd.DataFrame({"product_a": [1], "product_b": [2], "weight": [5.0]})
        adj = build_neighbor_map(edges)
        self.assertIn("1", adj)
        self.assertEqual(adj["1"], [("2", 1.0)])

    def test_none_or_empty(self):
        self.assertEqual(build_neighbor_map(None), {})
        self.assertEqual(build_neighbor_map(pd.DataFrame()), {})


class TestQualityRerank(unittest.TestCase):
    def test_reliable_good_rating_beats_unreliable_perfect(self):
        result = pd.DataFrame({"parent_asin": ["A", "B"], "fused": [1.0, 1.0]})
        details = pd.DataFrame(
            {"average_rating": [5.0, 4.8], "rating_number": [1, 1000]},
            index=["A", "B"],
        )
        out = quality_rerank(result, details)
        self.assertEqual(out["parent_asin"].iloc[0], "B")

    def test_multipliers_within_expected_bounds(self):
        result = pd.DataFrame({"parent_asin": ["A"], "fused": [1.0]})
        details = pd.DataFrame(
            {"average_rating": [5.0], "rating_number": [1000]}, index=["A"]
        )
        out = quality_rerank(result, details)
        # quality ~0.96 -> fused ~0.992; must not inflate above 1.0
        self.assertLess(out["fused"].iloc[0], 1.0)
        self.assertGreater(out["fused"].iloc[0], 0.98)

    def test_no_details_is_noop(self):
        result = pd.DataFrame({"parent_asin": ["A"], "fused": [1.0]})
        self.assertIs(quality_rerank(result, None), result)

    def test_empty_result_is_noop(self):
        result = pd.DataFrame({"parent_asin": [], "fused": []})
        details = pd.DataFrame({"average_rating": [], "rating_number": []})
        self.assertEqual(len(quality_rerank(result, details)), 0)


class TestRenormalizeSignals(unittest.TestCase):
    """Regression: post-filters can strip one signal's top hits (e.g. the
    type filter kills TF-IDF's t-shirt top hits for 'black shirt'), leaving
    that signal weakened against the others regardless of the user's weights."""

    def _res(self):
        return pd.DataFrame(
            {
                "parent_asin": ["C", "E"],
                "content": [0.5, 0.0],
                "embed": [0.0, 1.0],
                "sentiment_eff": [0.5, 0.5],
                "collab": [0.0, 0.0],
                "n_reviews": [10, 10],
                "fused": [0.4, 0.4],
            }
        )

    def test_weakened_signal_restored_to_weighted_strength(self):
        out = renormalize_signals(
            self._res(), {"content": 0.5, "embed": 0.5, "sentiment": 0.0, "collab": 0.0}
        )
        fused = dict(zip(out["parent_asin"], out["fused"]))
        self.assertAlmostEqual(fused["C"], 0.5)
        self.assertAlmostEqual(fused["E"], 0.5)

    def test_user_weights_still_decide(self):
        out = renormalize_signals(
            self._res(), {"content": 0.9, "embed": 0.1, "sentiment": 0.0, "collab": 0.0}
        )
        self.assertEqual(out["parent_asin"].iloc[0], "C")

    def test_untouched_signals_unchanged(self):
        res = self._res()
        res.loc[0, "embed"] = 0.4
        out = renormalize_signals(
            res, {"content": 0.5, "embed": 0.5, "sentiment": 0.0, "collab": 0.0}
        )
        self.assertAlmostEqual(out["fused"].iloc[0], 0.5 * 1.0 + 0.5 * 0.4)

    def test_all_zero_signal_no_nan(self):
        res = self._res()
        res["content"] = 0.0
        out = renormalize_signals(
            res, {"content": 0.5, "embed": 0.5, "sentiment": 0.0, "collab": 0.0}
        )
        self.assertFalse(out["fused"].isna().any())
        self.assertEqual(out["parent_asin"].iloc[0], "E")


class TestClassifyGarment(unittest.TestCase):
    """The category path is the structural signal that title keywords can't
    be trusted for: merch stuffs 'Black Shirt' into tee titles for SEO."""

    def test_exact_novelty_segment(self):
        self.assertEqual(
            classify_garment("Clothing, Shoes & Jewelry | Clothing | Men | Tops | Novelty"),
            "novelty",
        )

    def test_novelty_department_is_not_novelty(self):
        # 'Novelty & More' is a DEPARTMENT (2nd segment); a plain shirt filed
        # under it must classify by its leaf segment, not the department.
        self.assertEqual(
            classify_garment("Clothing, Shoes & Jewelry | Novelty & More | Men | Tops | Shirts"),
            "shirt",
        )

    def test_tee_segment(self):
        self.assertEqual(
            classify_garment("Clothing, Shoes & Jewelry | Men | Tops | T-Shirts"), "tee"
        )

    def test_shirts_and_tees_is_unknown(self):
        # mixed category — the title heuristics decide instead
        self.assertEqual(
            classify_garment("Men | Tops | Shirts & Tees"), "unknown"
        )

    def test_dress_shirts_leaf(self):
        self.assertEqual(
            classify_garment("Men | Tops | Shirts | Dress Shirts"), "shirt"
        )

    def test_polo_henley_hoodie_costume_tank(self):
        self.assertEqual(classify_garment("Men | Tops | Polos"), "polo")
        self.assertEqual(classify_garment("Men | Tops | Henleys"), "henley")
        self.assertEqual(classify_garment("Men | Tops | Hoodies & Sweatshirts"), "hoodie")
        self.assertEqual(classify_garment("Men | Tops | Costumes"), "costume")
        self.assertEqual(classify_garment("Men | Tops | Tank Tops"), "tank")

    def test_empty_and_none_safe(self):
        self.assertEqual(classify_garment(""), "unknown")
        self.assertEqual(classify_garment(None), "unknown")


class TestBuildGarmentClassMap(unittest.TestCase):
    def test_maps_each_asin(self):
        df = pd.DataFrame(
            {
                "parent_asin": ["A", "B", 3],
                "categories": [
                    "Men | Tops | T-Shirts",
                    "Men | Tops | Novelty",
                    "",
                ],
            }
        )
        self.assertEqual(
            build_garment_class_map(df),
            {"A": "tee", "B": "novelty", "3": "unknown"},
        )

    def test_missing_columns_empty(self):
        self.assertEqual(build_garment_class_map(pd.DataFrame({"parent_asin": ["A"]})), {})


class TestTypePreferenceWithGarmentClass(unittest.TestCase):
    """User acceptance criterion: 'black shirt' must return ONLY black
    dress/casual shirts — no merch tees, no polos, no tanks, no v-shirts.
    The garment class (from the Amazon category path) is authoritative."""

    TITLES = {
        "MERCH": "Bad Bunny Black Shirt",
        "TEEMIS": "Life is Good Funny T-Shirt",
        "BARE": "Black Shirt",
        "DRESS": "Men's Black Slim Fit Dress Shirt",
        "POL": "Black Polo Shirt",
        "NOV": "X-Men Black Graphic Novelty T Shirt",
        "TANK": "Black Tank Top",
        "VSHIRT": "Sueded V Shirt Black",
    }
    CLASSES = {
        "MERCH": "novelty",  # merch lives in the Novelty category
        "TEEMIS": "shirt",   # tee title misfiled under a shirts category
        "BARE": "shirt",     # real shirt with a bare title
        "DRESS": "shirt",
        "POL": "polo",
        "NOV": "novelty",
        "TANK": "tank",
        "VSHIRT": "tee",
    }

    def _df(self, asins):
        return pd.DataFrame({"parent_asin": asins, "fused": [0.5] * len(asins)})

    def test_shirt_query_keeps_only_real_shirt_class(self):
        out = apply_type_preference(
            self._df(list(self.TITLES)), self.TITLES, "black shirt",
            garment_class=self.CLASSES,
        )
        self.assertEqual(set(out["parent_asin"]), {"BARE", "DRESS"})

    def test_tee_title_kills_even_confirmed_shirt_class(self):
        # misfiled tee: category says 'shirt' but the title is a tee —
        # the title safety net must still drop it for a shirt query
        out = apply_type_preference(
            self._df(["TEEMIS", "DRESS"]), self.TITLES, "black shirt",
            garment_class=self.CLASSES,
        )
        self.assertEqual(list(out["parent_asin"]), ["DRESS"])

    def test_tee_query_keeps_tee_class_and_ree_titles(self):
        out = apply_type_preference(
            self._df(list(self.TITLES)), self.TITLES, "black tshirt",
            garment_class=self.CLASSES,
        )
        # TEEMIS (tee title in shirt class), NOV (novelty class, tee title),
        # VSHIRT (tee class) — not MERCH (no tee title), not TANK/POL/DRESS.
        self.assertEqual(set(out["parent_asin"]), {"TEEMIS", "NOV", "VSHIRT"})

    def test_polo_query_keeps_polo_class_and_polo_titles(self):
        out = apply_type_preference(
            self._df(list(self.TITLES)), self.TITLES, "black polo shirt",
            garment_class=self.CLASSES,
        )
        self.assertEqual(set(out["parent_asin"]), {"POL"})

    def test_without_class_legacy_heuristics_still_apply(self):
        # without the class map, the bare-titled merch leaks — this is why
        # the map matters (documents the regression this fix addresses)
        out = apply_type_preference(
            self._df(["MERCH", "BARE", "DRESS"]), self.TITLES, "black shirt"
        )
        self.assertEqual(set(out["parent_asin"]), {"MERCH", "BARE", "DRESS"})

    def test_misfiled_jacket_and_polo_in_shirt_class_dropped(self):
        # regression from a live 'black shirt' run: a Metal Mulisha 'Flannel
        # Jacket' and an Oakley 'Warren Polo' filed under Shirts categories
        # leaked through — the title veto must apply even for class 'shirt'
        titles = {
            "JACKET": "Metal Mulisha Men's Ripper Flannel Jacket Black",
            "POLO": "Oakley Warren Polo 2.0 Jet Black Mens Shirt",
            "DRESS": "Men's Black Slim Fit Dress Shirt",
        }
        classes = {"JACKET": "shirt", "POLO": "shirt", "DRESS": "shirt"}
        out = apply_type_preference(
            self._df(list(titles)), titles, "black shirt", garment_class=classes
        )
        self.assertEqual(list(out["parent_asin"]), ["DRESS"])


class TestPoloHenleyQuerySeparation(unittest.TestCase):
    """Boundary regression: 'black henley' used to be an UNFILTERED query —
    detect_garment_type returned '' for queries without 'shirt', so plain
    crew tees (Hanes 5-packs, Nike 23/7) flooded the results. Polo/henley
    queries now get their own type family."""

    TITLES = {
        "HEN1": "Hanes Men's X-Temp Long Sleeve Henley T-Shirt Black",
        "HEN2": "Jockey Men's Henley Shirt Black",
        "TEE": "Hanes Men's 5 Pack Crew Neck Tee Black",
        "NTP": "Nike Men's 23/7 T-Shirt Black",
        "POLO": "U.S. Polo Assn. Black Mallet Polo Shirt",
        "TANK": "Babioboa Mens Henleys V Neck Sleeveless 2 Pack Workout Tank",
    }
    CLASSES = {
        "HEN1": "tee", "HEN2": "shirt", "TEE": "tee", "NTP": "tee",
        "POLO": "polo", "TANK": "tank",
    }

    def _df(self, asins):
        return pd.DataFrame({"parent_asin": asins, "fused": [0.5] * len(asins)})

    def test_henley_query_excludes_crew_tees_and_polos(self):
        out = apply_type_preference(
            self._df(list(self.TITLES)), self.TITLES, "black henley",
            garment_class=self.CLASSES,
        )
        # henleys kept (henley-tee AND henley-shirt); crew tees, the plain
        # polo and the sleeveless henley-tank all dropped
        self.assertEqual(set(out["parent_asin"]), {"HEN1", "HEN2"})

    def test_polo_query_excludes_crew_tees(self):
        out = apply_type_preference(
            self._df(list(self.TITLES)), self.TITLES, "black polo",
            garment_class=self.CLASSES,
        )
        self.assertIn("POLO", set(out["parent_asin"]))
        self.assertNotIn("TEE", set(out["parent_asin"]))
        self.assertNotIn("NTP", set(out["parent_asin"]))

    def test_polo_henley_query_keeps_both_families(self):
        out = apply_type_preference(
            self._df(list(self.TITLES)), self.TITLES, "black polo henley",
            garment_class=self.CLASSES,
        )
        self.assertEqual(set(out["parent_asin"]), {"HEN1", "HEN2", "POLO"})

    def test_brand_name_polo_does_not_match_hoodies_or_tees(self):
        # 'U.S. Polo Assn.' is a BRAND: the brand name alone must not make a
        # hoodie or a tee match a polo query (live 'black polo' leak). Brands
        # arrive as display text (dots + case intact, e.g. 'U.S. Polo Assn.').
        titles = {
            "HOOD": "U.S. Polo Assn. Tricolor Full Zip Hoodie Black XL",
            "TEET": "U.S. Polo Assn. Black Mallet Mens Burnout Camo Print Short Sleeve T-Shirt",
            "POL": "U.S. Polo Assn. Black Mallet Mens Slim Fit Long Sleeve Solid Stretch",
            "REAL": "Mexican Baseball Teams Polo Shirt Black",
        }
        classes = {"HOOD": "shirt", "TEET": "shirt", "POL": "polo", "REAL": "unknown"}
        brands = {
            "HOOD": "U.S. Polo Assn.", "TEET": "U.S. Polo Assn.",
            "POL": "U.S. Polo Assn.", "REAL": "",
        }
        out = apply_type_preference(
            self._df(list(titles)), titles, "black polo",
            garment_class=classes, brands=brands,
        )
        self.assertEqual(set(out["parent_asin"]), {"POL", "REAL"})

    def test_unknown_class_polo_title_not_answered_by_henley_query(self):
        # Helly Hansen polo filed under 'Active Shirts & Tees' (class
        # unknown) leaked into 'black henley': the misfiled branch matched
        # 'polo' via POLO_MARKERS (which contains 'henley' too). A bare
        # query must only be answered by its own word.
        titles = {
            "HPLO": "Helly Hansen Mens Hydro Power Lazer Polo - Black - L",
            "HSHIRT": "Jockey Men's Henley Shirt Black",
        }
        classes = {"HPLO": "unknown", "HSHIRT": "unknown"}
        out = apply_type_preference(
            self._df(list(titles)), titles, "black henley",
            garment_class=classes,
        )
        self.assertEqual(set(out["parent_asin"]), {"HSHIRT"})
        out2 = apply_type_preference(
            self._df(list(titles)), titles, "black polo",
            garment_class=classes,
        )
        self.assertEqual(set(out2["parent_asin"]), {"HPLO"})


class TestNoveltyFilterWithGarmentClass(unittest.TestCase):
    TITLES = {
        "NOV1": "X-Men Black Shirt",  # no NOVELTY_WORDS keyword in the title
        "KEY": "Funny Grandma Gift Shirt Black",
        "REAL": "Men's Black Slim Fit Dress Shirt",
    }

    def _df(self, asins):
        return pd.DataFrame({"parent_asin": asins, "fused": [0.5] * len(asins)})

    def test_class_based_novelty_dropped_without_keyword(self):
        classes = {"NOV1": "novelty", "KEY": "unknown", "REAL": "shirt"}
        out = apply_novelty_filter(
            self._df(["NOV1", "REAL"]), self.TITLES, garment_class=classes
        )
        self.assertEqual(list(out["parent_asin"]), ["REAL"])

    def test_keyword_novelty_still_dropped_without_class(self):
        out = apply_novelty_filter(self._df(["KEY", "REAL"]), self.TITLES)
        self.assertEqual(list(out["parent_asin"]), ["REAL"])

    def test_costume_class_dropped(self):
        out = apply_novelty_filter(
            self._df(["NOV1", "REAL"]), self.TITLES,
            garment_class={"NOV1": "costume", "REAL": "shirt"},
        )
        self.assertEqual(list(out["parent_asin"]), ["REAL"])

    def test_query_wants_novelty_bypasses_filter(self):
        out = apply_novelty_filter(
            self._df(["NOV1", "KEY"]), self.TITLES,
            garment_class={"NOV1": "novelty", "KEY": "novelty"},
            query="funny gift shirt",
        )
        self.assertEqual(len(out), 2)

    def test_all_novelty_falls_back_unchanged(self):
        classes = {"NOV1": "novelty", "KEY": "novelty"}
        out = apply_novelty_filter(
            self._df(["NOV1", "KEY"]), self.TITLES, garment_class=classes
        )
        self.assertEqual(len(out), 2)


class TestGenderFilter(unittest.TestCase):
    TITLES = {
        "M1": "Men's Black Slim Fit Dress Shirt",
        "W1": "Women's White Button Down Shirt",
        "W2": "Womens Casual Linen Shirt",
        "U1": "Unisex Black Graphic T Shirt",
    }

    def _df(self, asins):
        return pd.DataFrame({"parent_asin": asins, "fused": [0.5] * len(asins)})

    def test_drops_womens_titles(self):
        out = apply_gender_filter(self._df(list(self.TITLES)), self.TITLES)
        self.assertEqual(set(out["parent_asin"]), {"M1", "U1"})

    def test_all_womens_falls_back_unchanged(self):
        out = apply_gender_filter(self._df(["W1", "W2"]), self.TITLES)
        self.assertEqual(len(out), 2)


class TestSecondaryColorBoost(unittest.TestCase):
    TITLES = {
        "A": "Black Button Down Shirt with White Buttons",
        "B": "Black Solid Dress Shirt",
        "C": "Black and White Stripe Shirt",
    }

    def _df(self):
        return pd.DataFrame(
            {"parent_asin": list(self.TITLES), "fused": [0.9, 0.85, 0.8]}
        )

    def test_secondary_color_items_ranked_first(self):
        out = apply_secondary_color_boost(
            self._df(), self.TITLES, "black shirt with white buttons"
        )
        # A: 0.9+0.1, C: 0.8+0.1, B: 0.85
        self.assertEqual(list(out["parent_asin"]), ["A", "C", "B"])
        self.assertAlmostEqual(out.loc[0, "fused"], 1.0)
        self.assertAlmostEqual(out.loc[2, "fused"], 0.85)

    def test_single_color_query_unchanged(self):
        res = self._df()
        out = apply_secondary_color_boost(res, self.TITLES, "black shirt")
        pd.testing.assert_frame_equal(out, res)

    def test_empty_pool_safe(self):
        res = pd.DataFrame({"parent_asin": [], "fused": []})
        out = apply_secondary_color_boost(res, self.TITLES, "black shirt with white buttons")
        self.assertEqual(len(out), 0)


class TestParsePriceRangeGuards(unittest.TestCase):
    def test_size_range_is_not_a_price(self):
        # regression: "size 17-18" (a neck/waist measurement) was parsed as
        # a sub-dollar price budget and emptied the results
        self.assertEqual(parse_price_range("black shirt size 17-18"), (None, None))

    def test_neck_range_is_not_a_price(self):
        self.assertEqual(parse_price_range("collar shirt neck 15-16"), (None, None))

    def test_not_more_than_is_a_max(self):
        self.assertEqual(parse_price_range("black shirt not more than 2000"), (None, 24.1))

    def test_not_less_than_is_a_min(self):
        self.assertEqual(
            parse_price_range("black shirt not less than 500 rupees"), (6.02, None)
        )

    def test_up_to_is_a_max(self):
        self.assertEqual(parse_price_range("shirt up to 2000"), (None, 24.1))


class TestPlainShirtWithGarmentClass(unittest.TestCase):
    TITLES = {
        "BARE": "Black Shirt",
        "MERCH": "Bad Bunny Black Shirt",
        "REAL": "Men's Black Slim Fit Dress Shirt",
    }
    CLASSES = {"BARE": "shirt", "MERCH": "novelty", "REAL": "shirt"}

    def _df(self, asins):
        return pd.DataFrame({"parent_asin": asins, "fused": [0.5] * len(asins)})

    def test_bare_title_kept_when_class_confirmed_shirt(self):
        out = apply_plain_shirt_preference(
            self._df(list(self.TITLES)), self.TITLES, "black shirt",
            garment_class=self.CLASSES,
        )
        self.assertEqual(set(out["parent_asin"]), {"BARE", "REAL"})

    def test_without_class_bare_title_dropped(self):
        # no class map: the bare title has no spec word -> dropped (old behavior)
        out = apply_plain_shirt_preference(self._df(list(self.TITLES)), self.TITLES, "black shirt")
        self.assertEqual(set(out["parent_asin"]), {"REAL"})


class TestCleanDisplayText(unittest.TestCase):
    def test_strips_html_tags(self):
        self.assertEqual(
            clean_display_text("Nike T-Shirt CJ0921-011 <br />"),
            "Nike T-Shirt CJ0921-011",
        )

    def test_repairs_double_encoded_utf8(self):
        # Double-encoded UTF-8 (cp1252 roundtrip): '’' (U+2019) reads
        # "â€™" and '—' (U+2014) reads "â€"\"" — written as escapes so the
        # lookalike quote chars cannot be silently swapped by an editor.
        self.assertEqual(
            clean_display_text("Menâ€™s Slim Shirt â€” Black"),
            "Men’s Slim Shirt — Black",
        )

    def test_clean_text_unchanged(self):
        self.assertEqual(clean_display_text("Nike Men's T-Shirt"), "Nike Men's T-Shirt")

    def test_none_and_empty_safe(self):
        self.assertEqual(clean_display_text(None), "")
        self.assertEqual(clean_display_text(""), "")


if __name__ == "__main__":
    unittest.main()
