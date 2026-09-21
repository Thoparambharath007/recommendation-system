"""Phase 7: hybrid fusion — content + embedding + sentiment into one ranking.

Final stage of the cascaded-LLM pipeline. A natural-language query is scored
by three signals and fused into a single recommendation score:

    content   — TF-IDF cosine over product metadata (Phase 1)
    embed     — cosine similarity of cascade embeddings (Phase 5)
    sentiment — mean Flair sentiment, review-count reliability (Phase 6)

    fused = w_content*content + w_embed*embed + w_sentiment*sentiment_eff

Sentiment is shrunk toward neutral (0.5) for products with few reviews:
    rel = min(n_reviews, cap) / cap
    sentiment_eff = 0.5 + (sentiment - 0.5) * rel

Usage:
    python src/phase7_fusion.py --query "breathable cotton shirt for office" --top-k 10
    python src/phase7_fusion.py --query "soft nike polo" --top-k 20 \\
        --w-content 0.2 --w-embed 0.4 --w-sentiment 0.4
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_PROCESSED_DIR,
    PRODUCT_SENTIMENT_PATH,
    PRODUCT_TEXTS_PATH,
    PROCESSED_METADATA_PATH,
)
from content_filter import ContentFilter
from phase5_similarity import INDEX_ASINS_PATH, INDEX_PATH, EmbeddingIndex, QueryEmbedder
from preprocessing import COLOR_PATTERN, COLOR_WORDS, classify_garment, preprocess_metadata

W_CONTENT_DEFAULT = 0.3
W_EMBED_DEFAULT = 0.4
W_SENTIMENT_DEFAULT = 0.3
W_COLLAB_DEFAULT = 0.15
REVIEW_CAP = 10
NEUTRAL = 0.5
COLOR_BOOST = 0.10
COLLAB_DECAY = 0.5

# NB: padded forms (" t ") require the marker to appear with word boundaries;
# bare "t" is too dangerous (e.g. "Cotton", "Outlet"). The hyphen-space variant
# "T- Shirt" was leaking through before (now covered by "t- shirt"). "v shirt"
# (Amazon's spelling of V-neck tees, e.g. "Next Level Sueded V shirt") leaked
# into "black shirt" results before.
TEE_MARKERS = (
    "t shirt", "t-shirts", "t-shirt", "tshirt", "t- shirt", "t- shirts",
    " tee", " tees", " tee shirt", " tee shirts", " t ", "undershirt",
    "v shirt",
)
TANK_MARKERS = ("tank top", " tank ", "sleeveless", "muscle", "a-shirt")
# A plain "shirt" query means button-up/dress-style shirts; polos and henleys
# are a different category and were the top leaks in real "black shirt" runs.
POLO_MARKERS = ("polo", "henley")
# A 'shirt' query wants button-up/dress shirts; hoodie/sweatshirt/jacket
# titles contain the word 'shirt' and were leaking through the type filter
# (e.g. 'Shirt Warehouse UK Kentucky Love KY on a Black Hoodie' ranked #1
# for 'black shirt').
SHIRT_EXCLUDE_MARKERS = ("hoodie", "sweatshirt", "jacket", "puffer")


OCCASION_WORDS = ("office", "formal", "business", "interview", "professional", "work")
OFFICE_WORDS = (
    "dress", "oxford", "button", "slim fit", "business", "formal",
    "wrinkle", "professional", "performance", "collar",
)
NOVELTY_WORDS = (
    "funny", "graphic", "novelty", "gift", "grandma", "grandpa", "grandson",
    "granddaughter", "mom", "dad", "mother", "father", "sister", "brother",
    "aunt", "uncle", "birthday", "christmas", "halloween", "costume", "rebel",
    "music", "mama", "movie", "band", "tv", "show", "meme", "joke", "gaming",
    "star", "customized", "personalized", "tried", "lucky", "black friday",
    "political", "proud", "soldier", "veteran", "retired", "army", "marine",
    "military", "police", "firefighter", "first responder", "dog tag", "flag",
    "humor", "walter white", "jesse pinkman", "breaking bad", "cartoon",
    "white house", "st patrick", "shamrock", "parody", "lantern", "gypsy",
    "sneaker", "spoof", "comical", "grunt style", "shirt for jordan", "logo",
    "doobie", "scooby", "shaggy", "hellboy", "bprd", "shop4ever",
    "cool shirt", "cool water", "cooling", "circulatory",
    "funky", "inspirational", "quote",
    # band / biker / brand-merch graphic tees that pass a plain "shirt" title
    "fender", "gibson", "telecaster", "stratocaster", "rockabilly",
    "biker", "bikes", "harley", "motorcycle", "guitar", "rock band",
    "grateful dead", "led zeppelin", "pink floyd", "metallica", "nirvana",
    # political / patriotic / slogan shirts (leaked in real-catalog audit)
    "gadsden", "tread on me", "tread", "don't tread", "dont tread",
    "appeal to heaven", "molon labe", "3 percent", "1776", "constitution",
    "liberty", "patriotic", "american flag", "eagle", "gunsmoke",
    "anniversary", "thankful", "thanksgiving", "turkey", "resist",
    "science is power", "uncle steve", "believe", "awesome shirt",
    "ya gotta", "gotcha", "gangster", "thug", "skull", "crossbones",
    "minion", "paw patrol", "disney", "star wars", "superman", "batman",
    "spider man", "pokemon", "mickey", "snoopy", "peanuts",
    # merch flooded real 'black shirt' runs (band/crime/military/car slogans)
    "bad bunny", "yhlqmdlg", "mafia", "organized crime", "boxing club",
    "samurai", "sword", "panzer", "detailing", "i love", "screen printed",
    "replica", "vital_apparel",
)

COLOR_NAME_COLLISIONS = (
    "walter white", "white house", "black friday", "black widow",
    "white water", "red bull", "grey goose", "black rifle",
    "us navy", "usn", "charlie brown", "brown brothers", "red shirt friday",
    "allis chalmers", "garth brooks", "peanuts", "duke",
)
# Brand-color prefix: 'Newport Blue Men's T-Shirt, ..., Yellow'. Amazon
# records these with brand 'Generic', so 'Newport Blue' is just a title
# prefix and the REAL garment color is the trailing word. Shape guard
# (word + color + gender at the START of the title) keeps ordinary titles
# like 'Black White Striped Shirt' untouched.
_BRAND_COLOR_PREFIX_RE = re.compile(
    r"^\s*(\w+) ((?:" + "|".join(COLOR_WORDS) + r")) (men'?s|women'?s|unisex)\b"
)


def _word_matches(title: str, words: tuple[str, ...]) -> bool:
    """True if any word appears in title, tolerant of plural/possessive suffixes."""
    return any(
        re.search(rf"(?<![\w-]){re.escape(w)}(s|es)?(?:'s)?(?![\w-])", title)
        for w in words
    )


def build_garment_class_map(df: pd.DataFrame) -> dict[str, str]:
    """asin -> garment class from the RAW (pipe-joined) Amazon category path.

    Must be called on the raw parquet df BEFORE preprocess_metadata (which
    strips the '|' separators that classify_garment splits on). A dict is
    cheap to keep and survives the row drops preprocess_metadata does.
    """
    if "categories" not in df.columns or "parent_asin" not in df.columns:
        return {}
    return {
        str(a): classify_garment(c)
        for a, c in zip(df["parent_asin"], df["categories"])
    }


def apply_novelty_filter(
    result: pd.DataFrame,
    titles: dict[str, str],
    *,
    garment_class: dict[str, str] | None = None,
    query: str = "",
) -> pd.DataFrame:
    """Ban novelty/gift/funny items from results — unless the query itself
    asks for one ("funny gift shirt" -> the user WANTS merch).

    A product is novelty when its Amazon category is the exact 'Novelty' or
    'Costume' segment (structural — the keyword blocklist alone cannot catch
    all ~160K merch titles) or when its title carries a NOVELTY_WORDS keyword.
    """
    q = f" {query.lower().strip()} "
    if any(
        re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", q) for w in NOVELTY_WORDS
    ):
        return result  # query itself wants novelty merch

    def has_novelty(asin: str) -> bool:
        if garment_class and garment_class.get(asin) in ("novelty", "costume"):
            return True
        title = titles.get(asin, "").lower()
        return _word_matches(title, NOVELTY_WORDS)

    kept = result[result["parent_asin"].map(lambda a: not has_novelty(a))]
    if len(kept) == 0:
        return result
    return kept.reset_index(drop=True)


def apply_occasion_preference(
    result: pd.DataFrame,
    titles: dict[str, str],
    query: str,
) -> pd.DataFrame:
    """For office/formal intents: keep office-appropriate shirts only.

    A title must contain an office word (dress/oxford/button-down/...) and no
    novelty word (funny/gift/grandma/...). Falls back to just banning novelty
    words when nothing survives the strict pass.
    """
    q = f" {query.lower().strip()} "
    if not any(w in q for w in OCCASION_WORDS):
        return result

    def words(asin: str) -> str:
        return f" {titles.get(asin, '').lower()} "

    has_novelty = lambda t: _word_matches(t, NOVELTY_WORDS)
    has_office = lambda t: any(f" {w} " in t for w in OFFICE_WORDS)

    strict = result[result["parent_asin"].map(
        lambda a: has_office(words(a)) and not has_novelty(words(a))
    )]
    if len(strict) > 0:
        return strict.reset_index(drop=True)

    relaxed = result[result["parent_asin"].map(lambda a: not has_novelty(words(a)))]
    if len(relaxed) == 0:
        return result
    return relaxed.reset_index(drop=True)

ACCESSORY_WORDS = (
    "pin", "brooch", "badge", "cufflink", "stud", "necktie", "bow tie", "armband",
    "necklace", "bracelet", "ring", "watch", "hat", "glove", "gloves",
    "sock", "socks", "belt", "bag", "wallet", "shoe", "shoes", "boot",
    "boots", "sandal", "sandals", "scarf", "glasses", "sunglasses",
    "jewelry", "jewellery", "costume pin", "brooch",
)


def apply_garment_filter(
    result: pd.DataFrame,
    titles: dict[str, str],
) -> pd.DataFrame:
    """Drop accessories/non-garments (pins, hats, socks, ...) from results.

    Uses the plural/possessive-tolerant matcher so "Hats" / "Boots" / "Belt
    set" don't leak past the exact-word check.
    """
    def is_garment(asin: str) -> bool:
        title = titles.get(asin, "").lower()
        return not _word_matches(title, ACCESSORY_WORDS)

    kept = result[result["parent_asin"].map(is_garment)]
    if len(kept) == 0:
        return result
    return kept.reset_index(drop=True)


def detect_garment_type(query: str) -> str:
    """'tee' for t-shirt queries, 'polo' for polo/henley queries, 'shirt'
    for (button-up) shirt queries.

    'polo' is its own family: before this, 'black henley' / 'black polo'
    (no 'shirt' word, no tee marker) returned '' and the type filter was a
    no-op — plain crew tees flooded the results.
    """
    q = f" {query.lower().strip()} "
    if any(m in q for m in TEE_MARKERS):
        return "tee"
    if "polo" in q or "henley" in q:
        return "polo"
    if "shirt" in q:
        return "shirt"
    return ""


# A real shirt title carries at least one garment spec word. Merch titles
# ('Bad Bunny Black Shirt', 'Mafia Mens Shirt', 'I Love Detailing Shirt')
# stuff 'shirt' in for Amazon SEO and have none of these — they were the top
# embed/TF-IDF hits for bare 'black shirt' queries and looked like T-shirts
# in the results.
# NB: no "crewneck"/"v-neck" — those read as T-shirts and must not qualify
# a title for a bare 'shirt' (button-up) query.
PLAIN_SHIRT_SPEC = (
    "button", "collar", "sleeve", "dress", "oxford", "poplin", "flannel",
    "linen", "chambray", "twill", "cotton", "polyester", "wrinkle",
    "slim fit", "regular fit", "classic fit", "suit fit", "untucked",
    "tuxedo", "pocket", "casual", "formal", "plain", "solid", "non iron",
)
_BARE_FILLER = ("", "a", "an", "the", "for", "of", "men", "men's", "mens")


def _is_bare_color_type_query(query: str) -> bool:
    """True when the query is just color(s) + shirt + filler, e.g.
    'black shirt', 'white shirt for men'. Any other modifier (material,
    pattern, occasion, brand) means the query is specific and the plain-shirt
    guard is not applied."""
    q = query.lower().strip()
    if "shirt" not in q:
        return False
    for tok in q.split():
        if tok in _BARE_FILLER or tok in ("shirt", "shirts") or tok in COLOR_WORDS:
            continue
        return False
    return True


def apply_plain_shirt_preference(
    result: pd.DataFrame,
    titles: dict[str, str],
    query: str,
    *,
    garment_class: dict[str, str] | None = None,
) -> pd.DataFrame:
    """For bare 'color shirt' queries keep only titles that read like a real
    dress/casual shirt (carry a spec word from PLAIN_SHIRT_SPEC).

    Merch T-shirts titled '… Black Shirt' flood both the TF-IDF and embedding
    pools for such queries and are T-shirts in the images — the user asked
    for a shirt and gets a graphic tee. Falls back to the unfiltered result
    when nothing survives, so the user never gets an empty page.

    A product whose category is a confirmed 'shirt' (button-up/dress) segment
    is exempt from the spec-word requirement — a bare-titled real shirt
    ('Black Shirt', Dress Shirts category) is a legitimate match, not merch.
    """
    if detect_garment_type(query) != "shirt" or not _is_bare_color_type_query(query):
        return result

    def is_real_shirt(asin: str) -> bool:
        cls = garment_class.get(asin, "unknown") if garment_class else "unknown"
        if cls == "shirt":
            return True
        t = f" {titles.get(asin, '').lower()} "
        return any(m in t for m in PLAIN_SHIRT_SPEC)

    kept = result[result["parent_asin"].map(is_real_shirt)]
    if len(kept) == 0:
        return result
    return kept.reset_index(drop=True)


WOMEN_PATTERN = re.compile(r"\bwomen'?s?\b")


def apply_gender_filter(
    result: pd.DataFrame,
    titles: dict[str, str],
) -> pd.DataFrame:
    """Drop women's-titled products.

    The catalog is men's-only by category, but ~3K unisex/misfiled listings
    carry 'Women's'/'Womens' in the title and leak into results.
    """
    def is_womens(asin: str) -> bool:
        return bool(WOMEN_PATTERN.search(titles.get(asin, "").lower()))

    kept = result[~result["parent_asin"].map(is_womens)]
    if len(kept) == 0:
        return result
    return kept.reset_index(drop=True)


def apply_secondary_color_boost(
    result: pd.DataFrame,
    titles: dict[str, str],
    query: str,
    boost: float = COLOR_BOOST,
) -> pd.DataFrame:
    """Rank products carrying the query's SECONDARY color(s) first.

    "black shirt with white buttons": apply_color_preference already keeps
    only black-titled items; this adds +boost to items whose title ALSO
    mentions the accent color (white buttons/trim/pocket) and re-sorts, so
    the garment the user described floats to the top. No-op with fewer than
    two query colors or an empty pool. Must run AFTER renormalize_signals
    (fused is on its final scale) and before quality_rerank.
    """
    if len(result) == 0:
        return result
    colors = [m.group(0).lower() for m in COLOR_PATTERN.finditer(query)]
    if len(colors) < 2:
        return result
    secondary = colors[1:]

    def has_secondary(asin: str) -> bool:
        title = f" {titles.get(asin, '').lower()} "
        return any(f" {c} " in title for c in secondary)

    result = result.copy()
    mask = result["parent_asin"].map(has_secondary)
    result.loc[mask, "fused"] += boost
    return result.sort_values("fused", ascending=False).reset_index(drop=True)


# Garment classes that can NEVER answer a 'tee' query (polos, henleys,
# hoodies, underwear, swimwear, tank tops are not t-shirts no matter what the
# title says).
TEE_HARD_EXCLUDE = ("polo", "henley", "hoodie", "underwear", "swim", "tank")
# Classes that can NEVER answer a plain 'shirt' (button-up) query.
SHIRT_CLASS_EXCLUDE = ("novelty", "costume", "underwear", "swim", "tee", "tank", "hoodie")


def apply_type_preference(
    result: pd.DataFrame,
    titles: dict[str, str],
    query: str,
    fallback: bool = True,
    *,
    garment_class: dict[str, str] | None = None,
    brands: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Restrict to the garment type named in the query ('shirt' vs 'tee'
    vs 'polo'/'henley').

    'shirt' -> only non-tee shirts (dress/button-down etc.); 'tshirt/tee' ->
    only t-shirts. No type word -> unchanged. With fallback=False (strict mode,
    used by the web layer's first pass) a zero-match pool returns EMPTY so the
    caller can widen the pool and retry; with fallback=True (default) a
    zero-match pool is returned unchanged as a last resort so the user still
    sees results instead of a blank page.

    garment_class (asin -> class from the Amazon category path; see
    preprocessing.classify_garment) is the authoritative signal when present:
    a 'shirt' category product is a real shirt even with a bare title, and a
    'novelty'/'tee'/'tank'/'polo' category product is never a dress shirt no
    matter what the title claims (merch stuffs 'Black Shirt' into tee titles
    for SEO — that was the top leak for 'black shirt' queries). When a class
    is unknown/missing the legacy title heuristics apply.
    """
    kind = detect_garment_type(query)
    if not kind:
        return result

    def matches(asin: str) -> bool:
        title = f" {titles.get(asin, '').lower()} "
        cls = garment_class.get(asin, "unknown") if garment_class else "unknown"
        has_tee = any(m in title for m in TEE_MARKERS)
        has_tank = any(m in title for m in TANK_MARKERS)
        is_heavy = any(m in title for m in SHIRT_EXCLUDE_MARKERS)
        if kind == "tee":
            if cls in TEE_HARD_EXCLUDE:
                return False
            if cls == "tee":
                return not has_tank
            # 'shirt' / 'novelty' / 'unknown' categories: title decides
            # (catches misfiled tees and merch-tees alike).
            return has_tee and not has_tank
        if kind == "polo":
            has_henley_title = "henley" in title
            # A 'Henley T-Shirt' is a henley, not a crew tee — the tee veto
            # must not fire on henley titles or henley queries lose them.
            if has_tee and not has_henley_title:
                return False  # a tee title is never a polo (USPA 'T-Shirt' leak)
            # Strip the brand before the polo check: 'U.S. Polo Assn.' is a
            # brand name — without stripping, every USPA hoodie/jacket would
            # match a polo query. Punctuation is normalized on both sides
            # because clean_text strips dots from the brand but not the title.
            norm_t = re.sub(r"[^a-z0-9]+", " ", title).strip()
            brand = (brands or {}).get(asin, "").strip().lower()
            norm_b = re.sub(r"[^a-z0-9]+", " ", brand).strip()
            if norm_b and norm_b in norm_t:
                norm_t = norm_t.replace(norm_b, " ", 1)
            # POLO_MARKERS mixes 'polo' and 'henley'; a bare 'polo' query must
            # not be answered by a title that only says 'henley' (and vice
            # versa) — so the two words are tracked separately.
            has_polo_title = "polo" in f" {norm_t} "
            polo_only = "polo" in query and "henley" not in query
            henley_only = "henley" in query and "polo" not in query
            if polo_only:
                need_word = has_polo_title
            elif henley_only:
                need_word = has_henley_title
            else:
                need_word = has_polo_title or has_henley_title
            if cls in ("polo", "henley"):
                if henley_only:
                    # a bare 'henley' query should not surface plain polos
                    return has_henley_title and not has_tank and not is_heavy
                if polo_only:
                    # a bare 'polo' query should not surface henley-category
                    # items unless the title itself says 'polo'
                    return (cls == "polo" or need_word) and not has_tank and not is_heavy
                return not has_tank and not is_heavy
            # misfiled polos and henley-tees: the title must say so; tanks,
            # hoodies and novelty never qualify
            return (need_word and not has_tank and not is_heavy
                    and cls not in ("novelty", "costume", "hoodie", "tank"))
        # kind == "shirt"
        if has_tee:
            return False  # safety net: a tee title is never a dress shirt
        if cls in SHIRT_CLASS_EXCLUDE:
            return False
        has_polo_title = any(m in title for m in POLO_MARKERS)
        wants_polo = "polo" in query or "henley" in query
        if cls in ("polo", "henley"):
            return wants_polo and not is_heavy
        if cls == "shirt":
            # Misfiles happen even in the shirt category (a 'Flannel Jacket'
            # or 'Warren Polo' filed under Shirts) — the title still vetoes.
            if wants_polo:
                return has_polo_title and not has_tank and not is_heavy
            return not has_tank and not has_polo_title and not is_heavy
        # cls == "unknown" (mixed category / missing) -> legacy title heuristics
        is_polo = any(m in title for m in POLO_MARKERS)
        polo_ok = is_polo == wants_polo
        return ("shirt" in title) and not has_tank and polo_ok and not is_heavy

    matched = result[result["parent_asin"].map(matches)]
    if len(matched) == 0 and fallback:
        return result
    return matched.reset_index(drop=True)


def parse_price_range(query: str) -> tuple[float | None, float | None]:
    """Extract a price intent ("under 2000", "from 1000 to 2000") from the query.

    Returns (min_price, max_price) in USD, or (None, None) when the query
    carries no price intent. Bare numbers >= 100 are treated as rupees
    (the catalog is priced in USD, median ~$20) and converted at ~83 INR/USD;
    explicit "$/usd/dollar" forces USD, "rs/inr/rupee" forces INR.
    """
    q = query.lower()
    if not re.search(r"\d", q):
        return (None, None)

    nums = r"(\d+(?:,\d{3})*(?:\.\d+)?)"
    # "1000 to 2000" / "1000-2000" is itself a price signal even without a
    # price word ("shirt 1000 to 2000").
    has_range = bool(re.search(nums + r"\s*(?:to|and|-|–)\s*" + nums, q))
    has_price_word = bool(
        re.search(
            r"price|budget|cost|rs\.?|rupee|inr|₹|\$|usd|dollar|under|below|above|over|around|between"
            r"|less than|more than|not more|not less|up to|exceed",
            q,
        )
    )
    if not has_price_word and not has_range:
        return (None, None)

    # "size 17-18" / "neck 15-16" is a garment measurement, not a price —
    # the range pattern would otherwise read it as a sub-dollar budget.
    if re.search(r"\b(?:size|neck)\b\s*" + nums + r"\s*(?:to|and|-|–)\s*" + nums, q):
        return (None, None)

    usd_marker = bool(re.search(r"\$|usd|dollar", q))
    inr_marker = bool(re.search(r"rs\.?|rupee|inr|₹", q))
    q = q.replace("$", " ").replace("₹", " ")

    def to_usd(value: float) -> float:
        if usd_marker:
            return value
        if inr_marker or value >= 100:
            return round(value / 83.0, 2)
        return value

    def num(s: str) -> float:
        return float(s.replace(",", ""))

    m = re.search(nums + r"\s*(?:to|and|-|–)\s*" + nums, q)
    if m:
        lo, hi = to_usd(num(m.group(1))), to_usd(num(m.group(2)))
        if lo < 1.0 and hi < 1.0:
            return (None, None)  # sub-dollar "range" is not a price
        return (min(lo, hi), max(lo, hi))

    # "not more than 500" must be checked BEFORE the bare "more than"
    # (over) pattern can grab its tail and invert the direction.
    m = re.search(r"(?:not more than|not higher than|not exceed(?:ing)?|up to)\s*" + nums, q)
    if m:
        return (None, to_usd(num(m.group(1))))

    m = re.search(r"(?:not less than|not lower than)\s*" + nums, q)
    if m:
        return (to_usd(num(m.group(1))), None)

    m = re.search(r"(?:under|below|less than|max(?:imum)?|cheaper than)\s*" + nums, q)
    if m:
        return (None, to_usd(num(m.group(1))))

    m = re.search(r"(?:over|above|more than|min(?:imum)?|at least)\s*" + nums, q)
    if m:
        return (to_usd(num(m.group(1))), None)

    m = re.search(r"(?:around|about|approx\.?)\s*" + nums, q)
    if m:
        v = to_usd(num(m.group(1)))
        return (round(v * 0.8, 2), round(v * 1.2, 2))

    m = re.search(r"(?:budget|price|cost)\s*(?:of|is|range)?\s*" + nums, q)
    if m:
        return (None, to_usd(num(m.group(1))))

    return (None, None)


def apply_color_preference(
    result: pd.DataFrame,
    titles: dict[str, str],
    query: str,
    top_k: int,
    brands: dict[str, str] | None = None,
    boost: float = COLOR_BOOST,
) -> pd.DataFrame:
    """Restrict results to products whose title contains the query's PRIMARY color.

    If the query names several colors ("white shirt with black buttons"), the
    FIRST color in the query is treated as the garment color. The brand name is
    stripped from the title before matching so "White Water" (a brand) never
    matches "white". If any candidate matches, ONLY matches are returned;
    otherwise results are unchanged.
    """
    matches = list(COLOR_PATTERN.finditer(query))
    if not matches:
        return result
    primary = matches[0].group(0)
    brands = brands or {}

    def has_color(asin: str) -> bool:
        title = f" {titles.get(asin, '').lower()} "
        brand = brands.get(asin, "").strip().lower()
        if brand:
            # Strip on the LOWERCASED title: the brands dict is lowercased
            # (clean_text) while titles keep their raw case, so a case-sensitive
            # replace was a silent no-op for e.g. brand "nike" vs title "Nike...".
            title = title.replace(brand, " ", 1)
        if any(f" {c} " in title for c in COLOR_NAME_COLLISIONS):
            return False
        # Amazon glues variant colors to the previous word
        # ('...American Made Cars,Teal'); a comma is a token boundary.
        title = title.replace(",", " ")
        # 'Newport Blue Men's T-Shirt, ..., Yellow': if the query's color
        # sits in a brand-color prefix AND the title names a DIFFERENT
        # color, the prefix is a brand, not the garment color. (A prefix
        # color with no other color in the title — 'Ocean Blue', 'True
        # Blue' — is a legitimate shade name and still matches.)
        m = _BRAND_COLOR_PREFIX_RE.match(title)
        if m and m.group(2) == primary.lower() and any(
            f" {c} " in title for c in COLOR_WORDS if c != m.group(2)
        ):
            return False
        return f" {primary.lower()} " in title

    matched = result[result["parent_asin"].map(has_color)]
    if len(matched) == 0:
        return result
    return matched.reset_index(drop=True)


def sentiment_effective(
    raw: float, n_reviews: int, cap: int = REVIEW_CAP, neutral: float = NEUTRAL
) -> float:
    rel = min(n_reviews, cap) / cap
    return neutral + (raw - neutral) * rel


def load_copurchase_edges() -> pd.DataFrame | None:
    """Load the Phase 2 co-purchase graph (product_a, product_b, weight.\\
    Returns None when the parquet is missing (the system still works, just
    without the collaborative signal).
    """
    try:
        from config import CO_PURCHASE_EDGES_PATH

        if not CO_PURCHASE_EDGES_PATH.exists():
            return None
        return pd.read_parquet(CO_PURCHASE_EDGES_PATH)
    except Exception as e:
        print(f"  WARNING: could not load co-purchase edges: {e}")
        return None


def build_neighbor_map(edges: pd.DataFrame | None) -> dict[str, list[tuple[str, float]]]:
    """Adjacency list: asin -> [(neighbor, weight), ...] (both directions).

    The Phase 2 builder stores each pair once (sorted, counted in a dict), so
    every neighbor appears at most once per node. Weights are normalized to
    0..1 by the global max so the collab signal mixes with the other signals.
    """
    if edges is None or len(edges) == 0:
        return {}
    w_max = float(edges["weight"].max()) or 1.0
    adj: dict[str, list[tuple[str, float]]] = {}
    for a, b, w in edges.itertuples(index=False):
        a, b = str(a), str(b)
        wn = float(w) / w_max  # normalize weights to 0..1 so they mix with the other signals
        adj.setdefault(a, []).append((b, wn))
        adj.setdefault(b, []).append((a, wn))
    return adj


def build_collab_scores(
    seeds: dict[str, float],
    adj: dict[str, list[tuple[str, float]]],
    decay: float = COLLAB_DECAY,
) -> tuple:
    """Score candidates by co-purchase affinity to the query's seed products.

    For every seed (content/embed top-K), each co-purchased neighbor contributes
    `seed_score * neighbor_weight * decay`; scores are normalized by the max so the
    resulting collab signal is in 0..1. Also returns a map asin -> [seed asins]
    (used for the web's \"people also bought\" explanation).
    """
    scores: dict[str, float] = {}
    via: dict[str, list[str]] = {}
    for seed_asn, seed_score in seeds.items():
        for nbr, wn in adj.get(seed_asn, []):
            add = seed_score * wn * decay
            prev = scores.get(nbr)
            if prev is None or add > prev:
                scores[nbr] = add
                via.setdefault(nbr, []).append(seed_asn)
    if not scores:
        return {}, {}
    mx = max(scores.values()) or 1.0
    return {k: v / mx for k, v in scores.items()}, via


def build_seed_scores(
    content_scores: dict[str, float] | None,
    embed_scores: dict[str, float] | None,
    k: int = 25,
) -> dict[str, float]:
    """Top-K content/embedding hits become the seeds for the collab signal.

    Each list is normalized by its own max so both signals contribute equally,
    then the union is re-normalized to 0..1.
    """
    seeds: dict[str, float] = {}
    for source in (content_scores or {}, embed_scores or {}):
        top = sorted(source.items(), key=lambda kv: kv[1], reverse=True)[:k]
        if not top:
            continue
        mx = max(v for _, v in top) or 1.0
        for asin, v in top:
            seeds[asin] = seeds.get(asin, 0.0) + v / mx
    if not seeds:
        return {}
    mx = max(seeds.values()) or 1.0
    return {a: v / mx for a, v in seeds.items()}

def apply_mmr(
    result: pd.DataFrame, titles: dict[str, str], lam: float = 0.72, pool: int = 50
) -> pd.DataFrame:
    """Maximal Marginal Relevance re-rank of the top-`pool` rows.

    Greedily picks the row maximizing lam*fused_norm - (1-lam)*max_title_similarity
    to the already-picked set, then appends the untouched tail unchanged.
    """
    if len(result) < 3:
        return result
    head = result.head(pool)
    tail = result.iloc[len(head):]
    raw = head["fused"].to_numpy()
    mx = float(raw.max()) or 1.0
    scores = raw / mx
    toks = [
        set(str(titles.get(a, "")).lower().split())
        for a in head["parent_asin"].astype(str).tolist()
    ]
    picked: list[int] = [0]
    rest = list(range(1, len(head)))
    while rest:
        best_i, best_v = rest[0], -1e9
        for i in rest:
            div = 0.0
            for j in picked:
                a, b = toks[i], toks[j]
                if not a or not b:
                    continue
                sim = len(a & b) / (len(a | b) or 1)
                if sim > div:
                    div = sim
            v = lam * scores[i] - (1 - lam) * div
            if v > best_v:
                best_v, best_i = v, i
        picked.append(best_i)
        rest.remove(best_i)
    out = head.iloc[picked].reset_index(drop=True)
    return pd.concat([out, tail]).reset_index(drop=True)


def _scale_by_max(values: dict[str, float]) -> dict[str, float]:
    """Scale a score dict so its maximum maps to 1.0 (ratios preserved).

    Puts signals with different natural ranges (TF-IDF cosine ~0.1-0.4 vs
    embedding cosine ~0.5-0.9) on a comparable 0..1 scale so the user's
    weights mean what they say. Unlike min-max, this does NOT amplify tiny
    differences at the bottom of the range into full-strength swings.
    """
    if not values:
        return {}
    hi = max(values.values())
    if hi < 1e-12:
        return dict.fromkeys(values, 0.0)
    return {a: v / hi for a, v in values.items()}


def sentiment_to_dict(sentiment: pd.DataFrame) -> dict[str, tuple[float, int]]:
    """Pre-compute the per-product sentiment lookup once (asin -> (score, n)).

    The web engine and the CLI call fuse() per request; iterating the full
    371K-row parquet with itertuples on every call wasted ~1s each. Build the
    dict once at load time and pass it to fuse().
    """
    out: dict[str, tuple[float, int]] = {}
    for row in sentiment.itertuples(index=False):
        out[str(getattr(row, "parent_asin"))] = (
            float(getattr(row, "sentiment_score")),
            int(getattr(row, "n_reviews")),
        )
    return out


def fuse(
    candidates: list[str],
    content_scores: dict[str, float] | None,
    embed_scores: dict[str, float] | None,
    sentiment: pd.DataFrame | dict[str, tuple[float, int]] | None,
    w_content: float = W_CONTENT_DEFAULT,
    w_embed: float = W_EMBED_DEFAULT,
    w_sentiment: float = W_SENTIMENT_DEFAULT,
    w_collab: float = W_COLLAB_DEFAULT,
    min_reviews: int = 1,
    review_cap: int = REVIEW_CAP,
    neutral: float = NEUTRAL,
    collab_scores: dict[str, float] | None = None,
    normalize: bool = False,
) -> pd.DataFrame:
    """Fuse per-candidate scores into a ranked DataFrame (pure, testable).

    With normalize=True (used by the web app) content and embedding scores
    are min-max scaled across the candidate pool before weighting, so a
    strong lexical match is never outranked by a slightly-higher cosine on
    an unrelated item.

    `sentiment` accepts either the raw parquet DataFrame (built into a lookup
    here — fine for one-off CLI runs) or a pre-computed dict from
    sentiment_to_dict() (used by the web engine to avoid rebuilding the
    371K-row lookup on every request).
    """
    content_scores = content_scores or {}
    embed_scores = embed_scores or {}
    collab = collab_scores or {}
    if normalize:
        content_scores = _scale_by_max(content_scores)
        embed_scores = _scale_by_max(embed_scores)

    if isinstance(sentiment, dict):
        sent = sentiment
    elif sentiment is not None and len(sentiment):
        sent = sentiment_to_dict(sentiment)
    else:
        sent = {}

    rows = []
    for asin in candidates:
        c = float(content_scores.get(asin, 0.0))
        e = max(float(embed_scores.get(asin, 0.0)), 0.0)
        co = max(float(collab.get(asin, 0.0)), 0.0)
        raw, n = sent.get(asin, (neutral, 0))
        if n < min_reviews:
            s_eff = neutral
        else:
            s_eff = sentiment_effective(raw, n, review_cap, neutral)
        fused = w_content * c + w_embed * e + w_sentiment * s_eff + w_collab * co
        rows.append(
            {
                "parent_asin": asin,
                "content": round(c, 4),
                "embed": round(e, 4),
                "sentiment_eff": round(s_eff, 4),
                "collab": round(co, 4),
                "n_reviews": n,
                "fused": round(fused, 4),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values("fused", ascending=False).reset_index(drop=True)


def quality_rerank(
    result: pd.DataFrame,
    details: pd.DataFrame | None,
    wq: float = 0.20,
) -> pd.DataFrame:
    """Blend rating reliability into the fused score using Bayesian shrinkage:
    quality = (n·avg + c·prior) / (n + c) with prior 3.9 stars and c = 25,
    scaled to 0..1. A 1-review 5-star item no longer outranks a
    1000-review 4.5-star item. final = fused * (1 - wq + wq*quality).

    `details` must be indexed by parent_asin (see web_app.Engine). No-op when
    details is None or empty (CLI runs without the details file still work).
    """
    if len(result) == 0 or details is None or "average_rating" not in details.columns:
        return result
    det = details.reindex(result["parent_asin"].astype(str))
    avg = pd.to_numeric(det["average_rating"], errors="coerce").fillna(3.9) / 5.0
    n = pd.to_numeric(det["rating_number"], errors="coerce").fillna(0)
    prior = 3.9 / 5.0
    c = 25.0
    quality = ((n * avg + c * prior) / (n + c)).clip(0, 1)
    result = result.copy()
    multiplier = 1 - wq + wq * quality.to_numpy()
    result["fused"] = result["fused"].to_numpy() * multiplier
    return result.sort_values("fused", ascending=False).reset_index(drop=True)


def renormalize_signals(result: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Re-scale content/embed to their post-filter maxima and recompute fused.

    The post-filters (color/type/occasion/novelty/garment/price) run AFTER
    fusion, and fuse(normalize=True) scaled each signal by its PRE-filter
    max. A filter can strip out one signal's top hits — e.g. TF-IDF's top
    "black shirt" hits are t-shirts that the type filter kills — silently
    weakening that signal against the others regardless of the user's
    weights (observed: content score 0.000 in the web top-20 for
    "black shirt" even with w_content=0.30). Re-scaling each signal to its
    surviving pool's max restores the weights' meaning. Signals whose top
    hits survived are unchanged (their max is already 1.0).
    """
    if len(result) == 0:
        return result
    c = result["content"].to_numpy(dtype=float)
    e = result["embed"].to_numpy(dtype=float)
    s = result["sentiment_eff"].to_numpy(dtype=float)
    co = result["collab"].to_numpy(dtype=float)

    def _scale(v: np.ndarray) -> np.ndarray:
        m = v.max()
        return v / m if m > 1e-12 else v

    fused = (
        weights.get("content", W_CONTENT_DEFAULT) * _scale(c)
        + weights.get("embed", W_EMBED_DEFAULT) * _scale(e)
        + weights.get("sentiment", W_SENTIMENT_DEFAULT) * s
        + weights.get("collab", W_COLLAB_DEFAULT) * co
    )
    out = result.copy()
    out["fused"] = fused
    return out.sort_values("fused", ascending=False).reset_index(drop=True)


def load_titles() -> dict[str, str]:
    df = pd.read_parquet(PRODUCT_TEXTS_PATH, columns=["parent_asin", "title"])
    return dict(zip(df["parent_asin"].astype(str), df["title"].astype(str)))


def load_brands() -> dict[str, str]:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "product_details.parquet", columns=["parent_asin", "brand"])
    return dict(zip(df["parent_asin"].astype(str), df["brand"].fillna("").astype(str)))


def load_prices() -> pd.Series:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "product_details.parquet", columns=["parent_asin", "price"])
    return pd.Series(df["price"].values, index=df["parent_asin"].astype(str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7: hybrid fusion ranking")
    parser.add_argument("--query", default="", help="natural-language query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--content-k", type=int, default=200, help="content candidates")
    parser.add_argument("--embed-k", type=int, default=200, help="embedding candidates")
    parser.add_argument("--w-content", type=float, default=W_CONTENT_DEFAULT)
    parser.add_argument("--w-embed", type=float, default=W_EMBED_DEFAULT)
    parser.add_argument("--w-sentiment", type=float, default=W_SENTIMENT_DEFAULT)
    parser.add_argument("--w-collab", type=float, default=W_COLLAB_DEFAULT)
    parser.add_argument("--min-reviews", type=int, default=1)
    args = parser.parse_args()

    if not args.query.strip():
        parser.error("provide --query")
    query = args.query.strip().lower()

    t0 = time.time()
    print(f"loading content filter ({PROCESSED_METADATA_PATH.name}) ...")
    df = pd.read_parquet(PROCESSED_METADATA_PATH)
    garment_class = build_garment_class_map(df)  # needs the raw '|' separators
    df = preprocess_metadata(df)
    cf = ContentFilter(df)
    print(f"  {len(cf.df):,} content products in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("loading embedding index ...")
    index = EmbeddingIndex.load(INDEX_PATH, INDEX_ASINS_PATH)
    print(f"  {len(index.asins):,} products indexed")

    print("loading query embedder (T5 + RoBERTa, first load is slow) ...")
    embedder = QueryEmbedder()
    vec = embedder.embed(query)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    sims = index.matrix @ vec
    print(f"  query embedded and scored in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("loading sentiment ...")
    if PRODUCT_SENTIMENT_PATH.exists():
        sent_dict = sentiment_to_dict(pd.read_parquet(PRODUCT_SENTIMENT_PATH))
        print(f"  {len(sent_dict):,} products with sentiment")
    else:
        sent_dict = {}
        print(f"  WARNING: {PRODUCT_SENTIMENT_PATH.name} missing — using neutral 0.5")

    print("loading co-purchase graph ...")
    adj = build_neighbor_map(load_copurchase_edges())
    if adj:
        print(f"  {len(adj):,} products with co-purchase neighbors")
    else:
        print("  no co-purchase edges found (collab signal off)")

    print("loading titles / brands / details ...")
    titles = load_titles()
    brands = load_brands()
    details_path = DATA_PROCESSED_DIR / "product_details.parquet"
    if details_path.exists():
        details = pd.read_parquet(
            details_path,
            columns=["parent_asin", "average_rating", "rating_number", "price"],
        ).set_index("parent_asin")
    else:
        details = None
        print(f"  WARNING: {details_path.name} missing — quality re-rank and price filter off")
    print(f"  loaded in {time.time() - t0:.1f}s")

    min_price, max_price = parse_price_range(query)
    if min_price is not None or max_price is not None:
        print(f"price intent: min={min_price} max={max_price}")

    weights = (args.w_content, args.w_embed, args.w_sentiment, args.w_collab)

    def run(pool_mult: int) -> pd.DataFrame:
        """Full pipeline for one candidate-pool size (retry widens the pool)."""
        t0 = time.time()
        content_top = cf.get_top_k(query, k=args.content_k * pool_mult)
        content_scores = dict(
            zip(content_top["parent_asin"].astype(str), content_top["score"])
        )
        embed_k = min(args.embed_k * pool_mult, len(index.asins))
        top_embed_idx = np.argpartition(sims, -embed_k)[-embed_k:]
        embed_scores = {str(index.asins[i]): float(sims[i]) for i in top_embed_idx}

        seeds = build_seed_scores(content_scores, embed_scores)
        collab_scores, _via = build_collab_scores(seeds, adj)

        candidates = list(dict.fromkeys(list(content_scores) + list(embed_scores)))
        print(f"  fusing {len(candidates):,} candidates with w=({weights[0]}, {weights[1]}, {weights[2]}, {weights[3]})")

        result = fuse(
            candidates,
            content_scores,
            embed_scores,
            sent_dict,
            w_content=args.w_content,
            w_embed=args.w_embed,
            w_sentiment=args.w_sentiment,
            w_collab=args.w_collab,
            min_reviews=args.min_reviews,
            collab_scores=collab_scores,
            normalize=True,
        )
        result = apply_color_preference(result, titles, query, args.top_k, brands=brands)
        result = apply_type_preference(
            result, titles, query, garment_class=garment_class, brands=brands
        )
        result = apply_occasion_preference(result, titles, query)
        result = apply_novelty_filter(result, titles, garment_class=garment_class, query=query)
        result = apply_garment_filter(result, titles)
        result = apply_gender_filter(result, titles)
        result = apply_plain_shirt_preference(result, titles, query, garment_class=garment_class)
        if min_price is not None or max_price is not None:
            if details is not None:
                p = details["price"].reindex(result["parent_asin"].astype(str)).to_numpy()
                in_budget = ~pd.isna(p)
                if min_price is not None:
                    in_budget &= p >= min_price
                if max_price is not None:
                    in_budget &= p <= max_price
                if in_budget.sum() >= 3:
                    result = result[in_budget].reset_index(drop=True)
                else:
                    # 70% of the catalog has no scraped price; a hard cut
                    # would leave a near-empty page. Keep the pool and tell
                    # the user the budget was relaxed.
                    print("  few/no items priced in budget — showing closest (budget relaxed)")
            else:
                print("  (price filter skipped — no details file)")
        result = renormalize_signals(
            result,
            {"content": args.w_content, "embed": args.w_embed,
             "sentiment": args.w_sentiment, "collab": args.w_collab},
        )
        result = apply_secondary_color_boost(result, titles, query)
        result = quality_rerank(result, details)
        result = apply_mmr(result, titles)
        print(f"  pipeline done in {time.time() - t0:.1f}s\n")
        return result

    result = run(1)
    if len(result) == 0 and detect_garment_type(query):
        print("  empty after strict filters — widening pool and retrying once")
        result = run(25)

    print("=== HYBRID RECOMMENDATIONS (content + embedding + sentiment + collab) ===")
    print(f"{'rank':<5}{'fused':<8}{'content':<9}{'embed':<8}{'sent':<8}{'collab':<8}{'n':<5}{'title'}")
    print("-" * 130)
    for i, row in result.head(args.top_k).iterrows():
        title = titles.get(row["parent_asin"], "")
        print(
            f"{i + 1:<5}{row['fused']:<8.4f}{row['content']:<9.4f}{row['embed']:<8.4f}"
            f"{row['sentiment_eff']:<8.4f}{row['collab']:<8.4f}{row['n_reviews']:<5}{title[:80]}"
        )

    if not PRODUCT_SENTIMENT_PATH.exists():
        print("\nTip: run this again once product_sentiment.parquet is downloaded "
              "(Phase 6 finished) to include the sentiment signal.")


if __name__ == "__main__":
    main()
