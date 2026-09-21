"""Rufus-style layer: typo-tolerant queries, autocomplete, facets, INR.

Pure functions + static data so everything is unit-testable without loading
the 2.3 GB engine. web_app wires these into /api/recommend, /api/suggest
and /api/facets.

Design rule: nothing here may ever raise on a bad query — a gibberish
search falls back to top-rated picks (see has_apparel_intent), a misspelt
word is either corrected (distance 1) or passed through untouched.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from preprocessing import COLOR_WORDS

USD_INR = 83.0  # same rate parse_price_range uses for bare rupee numbers


def usd_to_inr(usd: float) -> int:
    return int(round(usd * USD_INR))


def inr_to_usd(inr: float) -> float:
    return round(inr / USD_INR, 2)


# ---------------------------------------------------------------------------
# Query vocabulary (from the 371,902-title catalog scan)
# ---------------------------------------------------------------------------

GARMENT_WORDS = (
    "shirt", "shirts", "tshirt", "tshirts", "tee", "tees", "polo", "polos",
    "henley", "hoodie", "hoodies", "sweatshirt", "sweatshirts", "tank",
    "vest", "sweat",
)
COLORS = tuple(COLOR_WORDS)
FIT_WORDS = ("slim", "regular", "relaxed", "loose", "fitted", "plus", "tall")
MATERIAL_WORDS = (
    "cotton", "polyester", "linen", "flannel", "oxford", "denim", "wool",
    "silk", "chiffon", "spandex", "jersey", "twill",
)
PATTERN_WORDS = (
    "solid", "plain", "printed", "print", "striped", "stripes", "plaid",
    "check", "checkered", "floral", "graphic", "logo", "camouflage",
    "camo", "tiedye", "tie", "colorblock",
)
SLEEVE_WORDS = ("short", "long", "sleeve", "sleeves", "sleeveless", "half")
OCCASION_WORDS = (
    "casual", "formal", "office", "work", "party", "wedding", "golf",
    "sport", "sports", "fitness", "running", "summer", "winter", "beach",
    "travel", "interview",
)
STYLE_WORDS = (
    "round", "neck", "button", "collar", "vneck", "crew", "mock",
    "turtleneck", "pocket", "pockets",
)
GENDER_WORDS = ("men", "mens", "womens", "women", "unisex", "boys", "kids")
SIZE_HINT_WORDS = ("xs", "s", "m", "l", "xl", "2xl", "3xl", "4xl")

QUERY_VOCAB: frozenset[str] = frozenset(
    GARMENT_WORDS + COLORS + FIT_WORDS + MATERIAL_WORDS + PATTERN_WORDS
    + SLEEVE_WORDS + OCCASION_WORDS + STYLE_WORDS + GENDER_WORDS + SIZE_HINT_WORDS
)

# A query with NONE of these words (and no known brand) has no apparel
# intent — "asdfgh 123" lands on the top-rated fallback instead of
# nonsense results.
INTENT_WORDS: frozenset[str] = frozenset(
    GARMENT_WORDS + COLORS + FIT_WORDS + MATERIAL_WORDS + PATTERN_WORDS
    + OCCASION_WORDS
)

# Frequencies from the catalog scan; used only to tie-break a word that is
# distance-1 from two vocab words ("tea" -> "tee" wins over "ted"-ish).
_VOCAB_FREQ = {
    "shirt": 138669, "sleeve": 61394, "short": 32990, "long": 29985,
    "tee": 23948, "fit": 22919, "black": 21560, "cotton": 15958,
    "polo": 13547, "button": 12105, "neck": 11142, "white": 8490,
    "blue": 7323, "sweatshirt": 7489, "tank": 6838, "hoodie": 6443,
    "navy": 4641, "red": 5050, "green": 3615, "grey": 4182, "gray": 1943,
    "henley": 4025, "slim": 10163, "casual": 14598, "printed": 3416,
    "solid": 5459, "graphic": 9628, "plain": 846, "tshirt": 3207,
    "maroon": 461, "brown": 679, "olive": 303, "khaki": 366,
    "beige": 155, "linen": 2946, "flannel": 2304, "oxford": 1521,
    "summer": 3523, "winter": 831, "golf": 4514, "sport": 3013,
    "fitness": 1301, "running": 2576, "party": 1239, "formal": 308,
    "office": 306, "wedding": 379, "striped": 1570, "muscle": 2979,
    "vest": 943, "sleeveless": 3270,
}

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _osa1(a: str, b: str) -> int:
    """Optimal string alignment distance (Levenshtein + adjacent
    transposition), capped at 2 — correction only accepts distance 1,
    which covers single substitution / insertion / deletion / the classic
    "shrit"->"shirt" transposition, and almost nothing that would mangle a
    real word (no brand like 'nike' is 1 edit from any apparel word)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return 2
    prev2: list[int] | None = None
    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
        cur = [0] * (lb + 1)
    return prev[lb] if prev[lb] < 2 else 2


def correct_query(query: str) -> tuple[str, list[tuple[str, str]]]:
    """Fix single-edit typos in a query token-by-token.

    Returns (corrected_query, [(original, fixed), ...]). Words already in
    the catalog vocabulary, pure numbers, apostrophe words ("men's"), and
    anything with no distance-1 vocab candidate pass through untouched, so
    a weird query can never be mangled into the wrong garment.
    """
    fixes: list[tuple[str, str]] = []
    q = query.lower()

    def repl(m: re.Match) -> str:
        w = m.group(0)
        if w in QUERY_VOCAB or len(w) < 3 or "'" in w or not re.search(r"[a-z]", w):
            return w
        best: str | None = None
        best_freq = -1
        for cand in QUERY_VOCAB:
            if abs(len(cand) - len(w)) > 1:
                continue
            if _osa1(w, cand) == 1:
                f = _VOCAB_FREQ.get(cand, 0)
                if f > best_freq:
                    best, best_freq = cand, f
        if best is not None:
            fixes.append((w, best))
            return best
        return w

    corrected = re.sub(r"\s+", " ", _TOKEN_RE.sub(repl, q)).strip()
    return corrected, fixes


def has_apparel_intent(query: str, brands: frozenset[str] = frozenset()) -> bool:
    """True when the (already corrected) query names a garment, color,
    style or known brand — the gate that keeps gibberish from producing
    nonsense results. Brands are checked per-token AND as contiguous
    token runs, so multi-word brands ("under armour", "the north face")
    count as intent too."""
    tokens = _TOKEN_RE.findall(query.lower())
    for w in tokens:
        if w in INTENT_WORDS or (brands and w in brands):
            return True
    if brands:
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens) + 1):
                if " ".join(tokens[i:j]) in brands:
                    return True
    return False


# ---------------------------------------------------------------------------
# Autocomplete
# ---------------------------------------------------------------------------

SUGGEST_PHRASES = (
    "black shirt", "black tshirt", "black polo", "black henley",
    "polo shirt", "henley shirt", "tshirt", "dress shirt", "button down",
    "white shirt", "white tshirt", "white polo", "blue tshirt",
    "blue shirt", "navy shirt", "navy polo", "red tshirt",
    "green tshirt", "grey shirt", "gray tshirt", "maroon shirt",
    "brown shirt", "olive shirt", "khaki shirt", "beige shirt",
    "black hoodie", "black sweatshirt", "black tank top", "white hoodie",
    "slim fit shirt", "slim fit tshirt", "slim fit polo",
    "cotton tshirt", "cotton shirt", "linen shirt", "flannel shirt",
    "oxford shirt", "casual shirt", "casual tshirt", "formal shirt",
    "office shirt", "party shirt", "wedding shirt", "summer shirt",
    "summer tshirt", "winter shirt", "golf polo", "sport tshirt",
    "fitness shirt", "running tshirt", "short sleeve shirt",
    "short sleeve tshirt", "long sleeve shirt", "sleeveless shirt",
    "round neck tshirt", "v neck tshirt", "button down shirt",
    "printed shirt", "graphic tee", "graphic tshirt", "striped shirt",
    "plain tshirt", "solid polo", "black dress shirt", "white dress shirt",
)


def suggest(prefix: str, brands: list[str] | None = None, limit: int = 8) -> list[str]:
    """Autocomplete while typing. Curated phrases first, then single vocab
    words, then brands (top-40 from the facets). Never raises."""
    p = (prefix or "").lower().strip()
    scored: list[tuple[int, int, str]] = []
    for i, ph in enumerate(SUGGEST_PHRASES):
        if not p or ph.startswith(p) or (
            len(p) >= 2 and any(w.startswith(p) for w in ph.split())
        ):
            scored.append((0 if (p and ph.startswith(p)) else 1, i, ph))
    for i, w in enumerate(sorted(QUERY_VOCAB)):
        if p and len(w) >= len(p) and w.startswith(p):
            scored.append((2, i, w))
    for i, b in enumerate(brands or []):
        if p and b.startswith(p):
            scored.append((3, i, b))
    scored.sort(key=lambda x: (x[0], x[1]))
    seen: set[str] = set()
    out: list[str] = []
    for _, _, s in scored:
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) == limit:
            break
    return out


# ---------------------------------------------------------------------------
# Facets (brand / type / size / price choices for the clickable filters)
# ---------------------------------------------------------------------------

_BRAND_JUNK = {"", "generic", "no brand", "nan", "unknown", "n/a", "other"}

TYPE_LABELS = {
    "shirt": "Shirts", "tee": "T-Shirts", "polo": "Polos",
    "henley": "Henleys", "hoodie": "Hoodies", "tank": "Tanks",
}
TYPE_ORDER = ("shirt", "tee", "polo", "henley", "hoodie", "tank")
# The word a UI type-chip appends to the query — it must trigger
# detect_garment_type's matching family.
TYPE_QUERY_WORD = {
    "shirt": "shirt", "tee": "tshirt", "polo": "polo", "henley": "henley",
    "hoodie": "hoodie", "tank": "tank",
}

# extract_size() values that appear in >= a few hundred titles (catalog
# scan: L 31k, S 13.5k, M 9.9k, XL 9.6k, ... "big" = Big & Tall, 13.6k).
SIZE_OPTIONS = {
    "xs": "XS", "s": "S", "m": "M", "l": "L", "xl": "XL",
    "2xl": "2XL", "3xl": "3XL", "4xl": "4XL", "big": "Big & Tall",
}

# Buckets from the priced-catalog scan (30% of items are priced; 68% of
# those sit in Rs 1,245-2,490).
PRICE_BUCKETS = (
    {"key": "under415", "label": "Under ₹415", "min_inr": 0, "max_inr": 415},
    {"key": "415-1245", "label": "₹415 – ₹1,245", "min_inr": 415, "max_inr": 1245},
    {"key": "1245-2490", "label": "₹1,245 – ₹2,490", "min_inr": 1245, "max_inr": 2490},
    {"key": "2490-4980", "label": "₹2,490 – ₹4,980", "min_inr": 2490, "max_inr": 4980},
    {"key": "4980plus", "label": "₹4,980 +", "min_inr": 4980, "max_inr": None},
)


def build_facets(brands: dict[str, str], garment_class: dict[str, str]) -> dict:
    """One-shot facet build for /api/facets (web_app computes it at engine
    load, so the endpoint is instant)."""
    bc: Counter = Counter()
    for v in brands.values():
        b = (v or "").strip().lower()
        if b not in _BRAND_JUNK:
            bc[b] += 1
    cc = Counter(garment_class.values())
    return {
        "brands": [{"name": k, "count": v} for k, v in bc.most_common(40)],
        "types": [
            {"key": k, "label": TYPE_LABELS[k],
             "query_word": TYPE_QUERY_WORD[k], "count": cc.get(k, 0)}
            for k in TYPE_ORDER if cc.get(k, 0)
        ],
        "sizes": [{"key": k, "label": v} for k, v in SIZE_OPTIONS.items()],
        "price_buckets": list(PRICE_BUCKETS),
    }


# ---------------------------------------------------------------------------
# Post-fusion hard filters (brand / size) — pure, unit-testable
# ---------------------------------------------------------------------------

def filter_by_brand(
    df: pd.DataFrame, brands: dict[str, str], brand: str
) -> pd.DataFrame:
    """Keep only rows whose brand equals `brand` (case-insensitive)."""
    if not brand:
        return df
    b = brand.strip().lower()
    mask = df["parent_asin"].astype(str).map(
        lambda a: brands.get(a, "").strip().lower() == b
    )
    return df[mask].reset_index(drop=True)


def filter_by_size(
    df: pd.DataFrame, sizes: dict[str, str], size: str
) -> pd.DataFrame:
    """Keep only rows whose title-extracted size equals `size`."""
    if not size:
        return df
    s = size.strip().lower()
    mask = df["parent_asin"].astype(str).map(lambda a: sizes.get(a, "") == s)
    return df[mask].reset_index(drop=True)


def filter_by_rating(
    df: pd.DataFrame, ratings: dict[str, float], min_rating: float
) -> tuple[pd.DataFrame, pd.Series, bool]:
    """Keep rows rated >= min_rating. Only ~30% of the catalog has a price
    and many items have no rating at all, so (like the size filter) a cut
    that would leave fewer than 4 items RELAXES: the full pool is kept and
    `in_rating` flags which items actually clear the bar. Returns
    (df, in_rating, relaxed)."""
    import numpy as np

    arr = np.array(
        [ratings.get(str(a), np.nan) for a in df["parent_asin"]], dtype=float
    )
    ok = ~np.isnan(arr) & (arr >= min_rating)
    if int(ok.sum()) >= 4:
        keep = df[ok].reset_index(drop=True)
        return keep, pd.Series(True, index=keep.index), False
    return df, pd.Series(ok, index=df.index), True


def search_brands(brands: list[str], q: str, limit: int = 12) -> list[str]:
    """Brand search over the FULL (lowercased, sorted) catalog brand list.
    Prefix matches first; when fewer than 3 land, a contains fallback runs
    so 'essentials' finds 'amazon essentials'. Never raises."""
    q = (q or "").strip().lower()
    if not q:
        return []
    out = [b for b in brands if b.startswith(q)][:limit]
    if len(out) < 3:
        out = [b for b in brands if q in b][:limit]
    return out
