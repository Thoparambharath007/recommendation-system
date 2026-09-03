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
from preprocessing import COLOR_PATTERN, preprocess_metadata

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
# "T- Shirt" was leaking through before (now covered by "t- shirt").
TEE_MARKERS = (
    "t shirt", "t-shirts", "t-shirt", "tshirt", "t- shirt", "t- shirts",
    " tee", " tees", " tee shirt", " tee shirts", " t ", "undershirt",
)
TANK_MARKERS = ("tank top", " tank ", "sleeveless", "muscle", "a-shirt")
# A plain "shirt" query means button-up/dress-style shirts; polos and henleys
# are a different category and were the top leaks in real "black shirt" runs.
POLO_MARKERS = ("polo", "henley")


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
)

COLOR_NAME_COLLISIONS = (
    "walter white", "white house", "black friday", "black widow",
    "white water", "red bull", "grey goose", "black rifle",
    "us navy", "usn", "charlie brown", "brown brothers", "red shirt friday",
    "allis chalmers", "garth brooks", "peanuts", "duke",
)


def _word_matches(title: str, words: tuple[str, ...]) -> bool:
    """True if any word appears in title, tolerant of plural/possessive suffixes."""
    return any(
        re.search(rf"(?<![\w-]){re.escape(w)}(s|es)?(?:'s)?(?![\w-])", title)
        for w in words
    )


def apply_novelty_filter(
    result: pd.DataFrame,
    titles: dict[str, str],
) -> pd.DataFrame:
    """Always ban novelty/gift/funny shirts from results."""
    def has_novelty(asin: str) -> bool:
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
    "pin", "brooch", "badge", "cufflink", "necktie", "bow tie",
    "necklace", "bracelet", "ring", "watch", "hat", "glove", "gloves",
    "sock", "socks", "belt", "bag", "wallet", "shoe", "shoes", "boot",
    "boots", "sandal", "sandals", "scarf", "glasses", "sunglasses",
    "jewelry", "jewellery", "costume pin", "brooch",
)


def apply_garment_filter(
    result: pd.DataFrame,
    titles: dict[str, str],
) -> pd.DataFrame:
    """Drop accessories/non-garments (pins, hats, socks, ...) from results."""
    def is_garment(asin: str) -> bool:
        title = f" {titles.get(asin, '').lower()} "
        return not any(f" {w} " in title for w in ACCESSORY_WORDS)

    kept = result[result["parent_asin"].map(is_garment)]
    if len(kept) == 0:
        return result
    return kept.reset_index(drop=True)


def detect_garment_type(query: str) -> str:
    """'tee' if the query asks for t-shirts, 'shirt' if it asks for shirts."""
    q = f" {query.lower().strip()} "
    if any(m in q for m in TEE_MARKERS):
        return "tee"
    if "shirt" in q:
        return "shirt"
    return ""


def apply_type_preference(
    result: pd.DataFrame,
    titles: dict[str, str],
    query: str,
    fallback: bool = True,
) -> pd.DataFrame:
    """Restrict to the garment type named in the query ('shirt' vs 'tee').

    'shirt' -> only non-tee shirts (dress/button-down etc.); 'tshirt/tee' ->
    only t-shirts. No type word -> unchanged. With fallback=False (strict mode,
    used by the web layer's first pass) a zero-match pool returns EMPTY so the
    caller can widen the pool and retry; with fallback=True (default) a
    zero-match pool is returned unchanged as a last resort so the user still
    sees results instead of a blank page.
    """
    kind = detect_garment_type(query)
    if not kind:
        return result

    def matches(asin: str) -> bool:
        title = f" {titles.get(asin, '').lower()} "
        has_tee = any(m in title for m in TEE_MARKERS)
        has_tank = any(m in title for m in TANK_MARKERS)
        if kind == "tee":
            return has_tee and not has_tank
        is_polo = any(m in title for m in POLO_MARKERS)
        # "polo shirt" queries explicitly ask for polos — don't exclude them
        wants_polo = "polo" in query or "henley" in query
        polo_ok = is_polo == wants_polo
        return ("shirt" in title) and not has_tee and not has_tank and polo_ok

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
    has_price_word = bool(
        re.search(r"price|budget|cost|rs\.?|rupee|inr|₹|\$|usd|dollar|under|below|above|over|around|between", q)
    )
    if not has_price_word:
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

    nums = r"(\d+(?:,\d{3})*(?:\.\d+)?)"

    def num(s: str) -> float:
        return float(s.replace(",", ""))

    m = re.search(nums + r"\s*(?:to|and|-|–)\s*" + nums, q)
    if m:
        lo, hi = to_usd(num(m.group(1))), to_usd(num(m.group(2)))
        return (min(lo, hi), max(lo, hi))

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
        title = titles.get(asin, "")
        brand = brands.get(asin, "")
        if brand:
            title = title.replace(brand, "", 1)
        title = f" {title.lower()} "
        if any(f" {c} " in title for c in COLOR_NAME_COLLISIONS):
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
    Returns None when the parquet is missing (system still works, just ohne
    the collaborative signal).
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
    """Adjacency list: asin -> [(neighbor, weight), ...] (both directions, deduped.\\
    """
    if edges is None or len(edges) == 0:
        return {}
    w_max = float(edges["weight"].max()) or 1.0
    adj: dict[str, list[tuple[str, float]]] = {}
    for a, b, w in edges.itertuples(index=False):
        a = a
        b = b
        w = w
        wn = w / w_max  # normalize weights to 0..1 so they mix with the other signals
        lst = adj.setdefault(a, [])
        t = tuple([b, wn])
        lst.append(t)
        lst2 = adj.setdefault(b, [])
        t2 = tuple([a, wn])
        lst2.append(t2)
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


def fuse(
    candidates: list[str],
    content_scores: dict[str, float] | None,
    embed_scores: dict[str, float] | None,
    sentiment: pd.DataFrame | None,
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
    """
    content_scores = content_scores or {}
    embed_scores = embed_scores or {}
    collab = collab_scores or {}
    if normalize:
        content_scores = _scale_by_max(content_scores)
        embed_scores = _scale_by_max(embed_scores)

    sent = {}
    if sentiment is not None and len(sentiment):
        for row in sentiment.itertuples():
            sent[str(getattr(row, "parent_asin"))] = (
                float(getattr(row, "sentiment_score")),
                int(getattr(row, "n_reviews")),
            )

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
    parser.add_argument("--min-reviews", type=int, default=1)
    args = parser.parse_args()

    if not args.query.strip():
        parser.error("provide --query")
    query = args.query.strip().lower()

    t0 = time.time()
    print(f"loading content filter ({PROCESSED_METADATA_PATH.name}) ...")
    df = pd.read_parquet(PROCESSED_METADATA_PATH)
    df = preprocess_metadata(df)
    cf = ContentFilter(df)
    content_top = cf.get_top_k(query, k=args.content_k)
    content_scores = dict(zip(content_top["parent_asin"].astype(str), content_top["score"]))
    print(f"  {len(content_top):,} content candidates in {time.time() - t0:.1f}s")

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

    embed_k = min(args.embed_k, len(index.asins))
    top_embed_idx = np.argpartition(sims, -embed_k)[-embed_k:]
    embed_scores = {
        str(index.asins[i]): float(sims[i]) for i in top_embed_idx
    }

    t0 = time.time()
    print("loading sentiment ...")
    if PRODUCT_SENTIMENT_PATH.exists():
        sentiment = pd.read_parquet(PRODUCT_SENTIMENT_PATH)
        print(f"  {len(sentiment):,} products with sentiment")
    else:
        sentiment = None
        print(f"  WARNING: {PRODUCT_SENTIMENT_PATH.name} missing — using neutral 0.5")

    candidates = list(dict.fromkeys(list(content_scores) + list(embed_scores)))
    print(f"fusing {len(candidates):,} candidates with w=({args.w_content}, {args.w_embed}, {args.w_sentiment})")

    result = fuse(
        candidates,
        content_scores,
        embed_scores,
        sentiment,
        w_content=args.w_content,
        w_embed=args.w_embed,
        w_sentiment=args.w_sentiment,
        min_reviews=args.min_reviews,
    )
    titles = load_titles()
    brands = load_brands()
    min_price, max_price = parse_price_range(query)
    if min_price is not None or max_price is not None:
        print(f"price intent: min={min_price} max={max_price}")
        prices = load_prices()
        p = prices.reindex(result["parent_asin"].astype(str)).to_numpy()
        mask = ~pd.isna(p)
        if min_price is not None:
            mask &= p >= min_price
        if max_price is not None:
            mask &= p <= max_price
        result = result[mask]
    result = apply_color_preference(result, titles, query, args.top_k, brands=brands)
    result = apply_type_preference(result, titles, query)
    result = apply_occasion_preference(result, titles, query)
    result = apply_novelty_filter(result, titles)
    result = apply_garment_filter(result, titles)
    print(f"fused in {time.time() - t0:.1f}s\n")

    print("=== HYBRID RECOMMENDATIONS (weighted content + embedding + sentiment) ===")
    print(f"{'rank':<5}{'fused':<8}{'content':<9}{'embed':<8}{'sent':<8}{'n':<5}{'title'}")
    print("-" * 120)
    for i, row in result.head(args.top_k).iterrows():
        title = titles.get(row["parent_asin"], "")
        print(
            f"{i + 1:<5}{row['fused']:<8.4f}{row['content']:<9.4f}{row['embed']:<8.4f}"
            f"{row['sentiment_eff']:<8.4f}{row['n_reviews']:<5}{title[:80]}"
        )

    if not PRODUCT_SENTIMENT_PATH.exists():
        print("\nTip: run this again once product_sentiment.parquet is downloaded "
              "(Phase 6 finished) to include the sentiment signal.")


if __name__ == "__main__":
    main()
