"""Rufus-layer tests: typo correction, intent gate, autocomplete, facets,
brand/size filters, INR conversion. Pure functions — no models or data files.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from rufus import (
    PRICE_BUCKETS,
    QUERY_VOCAB,
    SIZE_OPTIONS,
    TYPE_ORDER,
    TYPE_QUERY_WORD,
    build_facets,
    correct_query,
    filter_by_brand,
    filter_by_rating,
    filter_by_size,
    has_apparel_intent,
    inr_to_usd,
    search_brands,
    suggest,
    usd_to_inr,
)


class TestCorrectQuery(unittest.TestCase):
    def test_single_edit_typo_each_token(self):
        q, fixes = correct_query("blak shrit")
        self.assertEqual(q, "black shirt")
        self.assertEqual(fixes, [("blak", "black"), ("shrit", "shirt")])

    def test_transposition_fixed(self):
        q, _ = correct_query("polo shert")
        self.assertEqual(q, "polo shirt")
        q, _ = correct_query("whtie tshirt")
        self.assertEqual(q, "white tshirt")

    def test_clean_query_untouched(self):
        q, fixes = correct_query("black shirt")
        self.assertEqual(q, "black shirt")
        self.assertEqual(fixes, [])

    def test_gibberish_untouched(self):
        q, fixes = correct_query("asdfgh 123")
        self.assertEqual(q, "asdfgh 123")
        self.assertEqual(fixes, [])

    def test_case_insensitive(self):
        q, fixes = correct_query("BLAK SHIRT")
        self.assertEqual(q, "black shirt")
        self.assertEqual(len(fixes), 1)

    def test_apostrophe_word_never_corrected(self):
        q, fixes = correct_query("men's polo")
        self.assertEqual(q, "men's polo")
        self.assertEqual(fixes, [])

    def test_numbers_never_corrected(self):
        q, fixes = correct_query("shirt under 2000")
        self.assertEqual(q, "shirt under 2000")
        self.assertEqual(fixes, [])

    def test_unknown_brand_not_mangled(self):
        # 'nike' is not distance-1 from any catalog word — a real brand must
        # survive correction, otherwise a brand query becomes a garment.
        q, fixes = correct_query("nike shirt")
        self.assertEqual(q, "nike shirt")
        self.assertEqual(fixes, [])

    def test_frequency_tiebreak(self):
        # 'tea' is distance-1 from 'tee' (the catalog garment word).
        q, fixes = correct_query("tea shirt")
        self.assertEqual(q, "tee shirt")
        self.assertEqual(fixes, [("tea", "tee")])

    def test_whitespace_collapsed(self):
        q, _ = correct_query("  blak   shrit ")
        self.assertEqual(q, "black shirt")


class TestApparelIntent(unittest.TestCase):
    def test_garment_and_color(self):
        self.assertTrue(has_apparel_intent("black shirt"))
        self.assertTrue(has_apparel_intent("navy"))

    def test_style_words(self):
        self.assertTrue(has_apparel_intent("slim fit"))
        self.assertTrue(has_apparel_intent("cotton"))
        self.assertTrue(has_apparel_intent("party"))

    def test_gibberish_is_no_intent(self):
        self.assertFalse(has_apparel_intent("asdfgh 123"))
        self.assertFalse(has_apparel_intent(""))
        self.assertFalse(has_apparel_intent("quantum flux"))

    def test_known_brand_counts_as_intent(self):
        brands = frozenset({"under armour"})
        self.assertTrue(has_apparel_intent("under armour", brands))
        self.assertFalse(has_apparel_intent("under armour"))


class TestSuggest(unittest.TestCase):
    def test_empty_prefix_returns_curated_phrases(self):
        out = suggest("")
        self.assertEqual(len(out), 8)
        self.assertIn("black shirt", out)

    def test_prefix_matches_phrases_first(self):
        out = suggest("bl")
        self.assertEqual(out[0], "black shirt")
        self.assertTrue(all(o.startswith("bl") or "bl" in o for o in out))

    def test_polo_suggestion(self):
        out = suggest("po")
        self.assertEqual(out[0], "polo shirt")
        # the bare vocab word lands once phrase slots are exhausted
        self.assertIn("polo", suggest("po", limit=12))

    def test_brand_tier(self):
        out = suggest("han", brands=["hanes", "hollister"])
        self.assertIn("hanes", out)

    def test_gibberish_returns_empty(self):
        self.assertEqual(suggest("zzzzzz"), [])

    def test_limit_respected(self):
        self.assertEqual(len(suggest("b", limit=3)), 3)

    def test_never_raises(self):
        self.assertIsInstance(suggest(None), list)
        self.assertIsInstance(suggest("   "), list)


class TestInr(unittest.TestCase):
    def test_usd_to_inr(self):
        self.assertEqual(usd_to_inr(19.99), 1659)
        self.assertEqual(usd_to_inr(25.0), 2075)

    def test_inr_to_usd(self):
        # 83 * 20 = 1660, so 1659 INR is 19.99 USD — never overstate it.
        self.assertEqual(inr_to_usd(1659), 19.99)
        self.assertEqual(inr_to_usd(1660), 20.0)
        self.assertEqual(inr_to_usd(415), 5.0)

    def test_roundtrip(self):
        for inr in (2000, 500, 4980):
            self.assertEqual(usd_to_inr(inr_to_usd(inr)), inr)


class TestBuildFacets(unittest.TestCase):
    def test_brand_junk_filtered_and_counted(self):
        brands = {
            "a1": "Hanes", "a2": "Hanes", "a3": "Nike",
            "a4": "Generic", "a5": "", "a6": "no brand", "a7": "N/A",
        }
        f = build_facets(brands, {"a1": "tee", "a2": "shirt"})
        self.assertEqual(f["brands"], [{"name": "hanes", "count": 2},
                                       {"name": "nike", "count": 1}])

    def test_types_only_known_garment_classes(self):
        f = build_facets({}, {"a1": "shirt", "a2": "shirt", "a3": "tee",
                              "a4": "novelty", "a5": "costume"})
        self.assertEqual([(t["key"], t["count"]) for t in f["types"]],
                         [("shirt", 2), ("tee", 1)])

    def test_brand_cap_40(self):
        brands = {f"a{i}": f"brand{i}" for i in range(45)}
        f = build_facets(brands, {})
        self.assertEqual(len(f["brands"]), 40)

    def test_sizes_and_price_buckets(self):
        f = build_facets({}, {})
        self.assertEqual([s["key"] for s in f["sizes"]], list(SIZE_OPTIONS))
        self.assertEqual(len(f["price_buckets"]), 5)
        self.assertEqual(f["price_buckets"][0]["label"], "Under ₹415")

    def test_price_buckets_contiguous(self):
        self.assertEqual(PRICE_BUCKETS[0]["min_inr"], 0)
        for lo, hi in zip(PRICE_BUCKETS, PRICE_BUCKETS[1:]):
            self.assertEqual(lo["max_inr"], hi["min_inr"])
        self.assertIsNone(PRICE_BUCKETS[-1]["max_inr"])

    def test_type_query_words_are_engine_vocabulary(self):
        # A chip's query word must be a word the engine's vocabulary knows —
        # otherwise clicking the chip could silently search nothing.
        for k in TYPE_ORDER:
            self.assertIn(k, TYPE_QUERY_WORD)
            self.assertIn(TYPE_QUERY_WORD[k], QUERY_VOCAB)


class TestPostFusionFilters(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"parent_asin": ["A", "B", "C"]})

    def test_brand_case_insensitive(self):
        brands = {"A": "Hanes", "B": "NIKE", "C": "Hanes"}
        out = filter_by_brand(self.df, brands, "hanes")
        self.assertEqual(list(out["parent_asin"]), ["A", "C"])

    def test_brand_empty_returns_all(self):
        out = filter_by_brand(self.df, {"A": "Hanes"}, "")
        self.assertEqual(len(out), 3)

    def test_brand_no_match_empty(self):
        out = filter_by_brand(self.df, {"A": "Hanes"}, "zara")
        self.assertEqual(len(out), 0)

    def test_size_match(self):
        sizes = {"A": "l", "B": "m", "C": ""}
        out = filter_by_size(self.df, sizes, "L")
        self.assertEqual(list(out["parent_asin"]), ["A"])

    def test_size_empty_returns_all(self):
        out = filter_by_size(self.df, {"A": "l"}, "")
        self.assertEqual(len(out), 3)

    def test_size_no_match_empty(self):
        out = filter_by_size(self.df, {"A": "l"}, "xs")
        self.assertEqual(len(out), 0)


class TestFilterByRating(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"parent_asin": [f"A{i}" for i in range(8)]})
        # A0-A6 rated, A7 has no rating (common: ~30% of the catalog does)
        self.ratings = {"A0": 4.8, "A1": 4.5, "A2": 4.2, "A3": 4.0,
                        "A4": 3.5, "A5": 3.0, "A6": 2.5}

    def test_strict_keeps_qualifying_rows(self):
        out, in_r, relaxed = filter_by_rating(self.df, self.ratings, 4.0)
        self.assertFalse(relaxed)
        self.assertEqual(list(out["parent_asin"]), ["A0", "A1", "A2", "A3"])
        self.assertTrue(bool(in_r.all()))

    def test_relaxes_when_fewer_than_four_match(self):
        out, in_r, relaxed = filter_by_rating(self.df, self.ratings, 4.7)
        self.assertTrue(relaxed)
        self.assertEqual(len(out), 8)          # full pool kept, page not empty
        self.assertEqual(int(in_r.sum()), 1)   # only A0 (4.8) clears 4.7
        self.assertEqual(in_r.tolist(), [True, False, False, False,
                                         False, False, False, False])

    def test_unrated_rows_never_pass(self):
        out, _, relaxed = filter_by_rating(self.df, self.ratings, 0.0)
        self.assertFalse(relaxed)
        self.assertNotIn("A7", list(out["parent_asin"]))


class TestSearchBrands(unittest.TestCase):
    BRANDS = ["amazon essentials", "hanes", "hollister", "nike",
              "under armour", "u.s. polo assn."]

    def test_prefix_matches(self):
        self.assertEqual(search_brands(self.BRANDS, "h"),
                         ["hanes", "hollister"])

    def test_contains_fallback(self):
        # 'essentials' is not a prefix of anything — the fallback finds it
        self.assertIn("amazon essentials", search_brands(self.BRANDS, "essentials"))

    def test_empty_query_returns_empty(self):
        self.assertEqual(search_brands(self.BRANDS, ""), [])
        self.assertEqual(search_brands(self.BRANDS, None), [])

    def test_case_insensitive(self):
        self.assertEqual(search_brands(self.BRANDS, "NIKE"), ["nike"])

    def test_limit_respected(self):
        self.assertEqual(len(search_brands(["aa"] * 50, "a", limit=12)), 12)


if __name__ == "__main__":
    unittest.main()
