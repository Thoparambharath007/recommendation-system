"""Hybrid apparel recommendation website — FastAPI backend.

Loads the Phase 7 engine once (content filter + embedding index +
query embedder + sentiment), then serves recommendations over HTTP.

Run:
    uvicorn src.web_app:app --host 127.0.0.1 --port 8000
Open: http://127.0.0.1:8000
"""

import os
import re
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import threading

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import (
    DATA_PROCESSED_DIR,
    PROCESSED_METADATA_PATH,
)
from content_filter import ContentFilter
from phase5_similarity import INDEX_ASINS_PATH, INDEX_PATH, EmbeddingIndex, QueryEmbedder
from phase7_fusion import (
    W_COLLAB_DEFAULT,
    apply_color_preference,
    apply_gender_filter,
    apply_garment_filter,
    apply_mmr,
    apply_novelty_filter,
    apply_occasion_preference,
    apply_plain_shirt_preference,
    apply_secondary_color_boost,
    apply_type_preference,
    build_collab_scores,
    build_garment_class_map,
    build_neighbor_map,
    build_seed_scores,
    detect_garment_type,
    fuse,
    load_copurchase_edges,
    quality_rerank,
    renormalize_signals,
    sentiment_to_dict,
)
from preprocessing import preprocess_metadata
from rufus import (
    SIZE_OPTIONS,
    USD_INR,
    build_facets,
    correct_query,
    filter_by_rating,
    has_apparel_intent,
    inr_to_usd,
    search_brands,
    suggest,
    usd_to_inr,
)

DETAILS_PATH = DATA_PROCESSED_DIR / "product_details.parquet"
SENTIMENT_PATH = DATA_PROCESSED_DIR / "product_sentiment.parquet"
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

CONTENT_K = 200
EMBED_K = 200
EMBED_CACHE_SIZE = 64

_TAG_RE = re.compile(r"<[^>]+>")


def clean_display_text(s: str) -> str:
    """Strip stray HTML tags and repair mojibake in titles/brands.

    The paper's preprocessing removes HTML tags from scraped metadata, but a
    few raw titles slipped through (e.g. '... CJ0921-011 <br />'); ~46 titles
    are doubly-encoded UTF-8 ('Menâ€™s', 'â€“'). Fix once at load so every
    consumer (quality filters, /api/collab, 'people also bought') sees clean
    text. The cp1252->utf-8 roundtrip only succeeds on genuine mojibake;
    clean text (incl. legit accented chars) fails one leg and is kept as-is.
    """
    if not isinstance(s, str) or not s:
        return "" if s is None else s
    s = _TAG_RE.sub(" ", s)
    try:
        repaired = s.encode("cp1252").decode("utf-8")
        if repaired != s:
            s = repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return re.sub(r"\s+", " ", s).strip()

app = FastAPI(title="Hybrid Apparel Recommender")


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(12, ge=1, le=50)
    w_content: float = Field(0.3, ge=0, le=1)
    w_embed: float = Field(0.4, ge=0, le=1)
    w_sentiment: float = Field(0.3, ge=0, le=1)
    w_collab: float = Field(0.15, ge=0, le=1)
    min_reviews: int = Field(1, ge=0, le=20)
    min_price: float | None = Field(None, ge=0)
    max_price: float | None = Field(None, ge=0)
    brand: str | None = Field(None, max_length=100)
    size: str | None = Field(None, max_length=20)
    # "4★ & up" chip — star floor, applied with the same relax-if-too-few
    # pattern as brand/size (see Engine.recommend).
    min_rating: float | None = Field(None, ge=0, le=5)
    # Price from the UI's rupee buckets; overrides the text-parsed budget.
    min_price_inr: float | None = Field(None, ge=0)
    max_price_inr: float | None = Field(None, ge=0)


class Engine:
    def __init__(self):
        import gc

        t0 = time.time()
        print("loading content filter ...")
        df = pd.read_parquet(PROCESSED_METADATA_PATH)
        # Garment class from the RAW category path — preprocess_metadata
        # strips the '|' separators classify_garment splits on, so this must
        # happen first. This is what keeps merch tees out of "black shirt".
        self.garment_class = build_garment_class_map(df)
        df = preprocess_metadata(df)
        self.content_filter = ContentFilter(df)
        del df
        gc.collect()
        print(f"  {len(self.content_filter.df):,} content products "
              f"({len(self.garment_class):,} garment classes) in {time.time() - t0:.1f}s")

        t0 = time.time()
        print("loading embedding index ...")
        self.index = EmbeddingIndex.load(INDEX_PATH, INDEX_ASINS_PATH)
        print(f"  {len(self.index.asins):,} indexed products")

        print("loading query embedder (T5 + RoBERTa, ~1 min first time) ...")
        self.embedder = QueryEmbedder()
        # Repeat queries (and the web's widen+retry path) skip the 3-5s T5
        # encode when the vector is already cached.
        self._embed_cache: dict[str, np.ndarray] = {}
        self._embed_cache_lock = threading.Lock()

        t0 = time.time()
        print("loading sentiment ...")
        if SENTIMENT_PATH.exists():
            # Pre-compute the asin -> (score, n) dict ONCE; fuse() used to
            # itertuples over all ~371K rows on every request.
            self.sentiment = sentiment_to_dict(pd.read_parquet(SENTIMENT_PATH))
            print(f"  {len(self.sentiment):,} products with sentiment")
        else:
            self.sentiment = {}
            print("  no sentiment file — using neutral 0.5")

        print("loading co-purchase graph ...")
        edges = load_copurchase_edges()
        self.adj = build_neighbor_map(edges)
        if self.adj:
            print(f"  {len(self.adj):,} products with co-purchase neighbors")
        else:
            print("  no co-purchase edges found (collab signal off)")

        print("loading titles ...")
        from config import PRODUCT_TEXTS_PATH

        titles_df = pd.read_parquet(PRODUCT_TEXTS_PATH, columns=["parent_asin", "title"])
        self.titles = {
            str(a): clean_display_text(t)
            for a, t in zip(titles_df["parent_asin"], titles_df["title"])
        }
        del titles_df
        print("loading brands ...")
        brands_df = pd.read_parquet(DETAILS_PATH, columns=["parent_asin", "brand"])
        self.brands = {
            str(a): clean_display_text(b)
            for a, b in zip(brands_df["parent_asin"], brands_df["brand"].fillna(""))
        }
        del brands_df
        print("loading details ...")
        details = pd.read_parquet(
            DETAILS_PATH,
            columns=["parent_asin", "title", "brand", "average_rating",
                     "rating_number", "price", "image_url"],
        )
        details["title"] = details["title"].map(clean_display_text)
        details["brand"] = details["brand"].fillna("").map(clean_display_text)
        # Star ratings as a plain dict for the '4★ & up' filter (vectorized
        # reindex in recommend(), same pattern as the price filter).
        self.ratings = {
            str(a): float(r)
            for a, r in zip(details["parent_asin"], details["average_rating"])
            if pd.notna(r)
        }
        self.details = details.set_index("parent_asin")

        print("building size map / facets ...")
        cdf = self.content_filter.df
        self.sizes = {
            str(a): s
            for a, s in zip(cdf["parent_asin"], cdf["size"])
            if isinstance(s, str) and s
        }
        self.brand_set = frozenset(
            v.strip().lower() for v in self.brands.values() if v and v.strip()
        )
        self.brand_sorted = sorted(self.brand_set)  # for /api/brands search
        self.facets = build_facets(self.brands, self.garment_class)
        print(f"  {len(self.sizes):,} sized items, {len(self.brand_set):,} brands")
        gc.collect()
        print("engine ready")

    def embed_query(self, query: str) -> np.ndarray:
        """Unit-normalized cascade embedding for a query (LRU-cached)."""
        with self._embed_cache_lock:
            vec = self._embed_cache.get(query)
        if vec is None:
            t0 = time.time()
            vec = self.embedder.embed(query)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vec = vec.astype(np.float32)
            with self._embed_cache_lock:
                if len(self._embed_cache) >= EMBED_CACHE_SIZE:
                    self._embed_cache.pop(next(iter(self._embed_cache)))
                self._embed_cache[query] = vec
            print(f"  query embedded in {time.time() - t0:.1f}s (cache miss)")
        else:
            print("  query embedded from cache (0.000s)")
        return vec

    def recommend(self, query: str, top_k: int, weights: dict, min_reviews: int,
                  min_price: float | None = None, max_price: float | None = None,
                  brand: str | None = None, size: str | None = None,
                  min_rating: float | None = None, _retry: bool = False):
        """Returns (result, price_info, note).

        note carries UI flags: a fallback message for no-intent queries,
        and/or brand_relaxed / size_relaxed when a hard filter matched too
        little (the pool is kept and items flagged instead of emptying).
        """
        # No apparel intent ("asdfgh 123") -> top-rated fallback so the page
        # never shows nonsense or empties out. The query must already be
        # typo-corrected by the caller.
        if not has_apparel_intent(query, self.brand_set):
            print("  no apparel intent — serving top-rated fallback")
            det = self.details.sort_values(
                ["average_rating", "rating_number"], ascending=[False, False]
            )
            n = min(top_k, len(det))
            out = pd.DataFrame({"parent_asin": det.index.astype(str).tolist()[:n]})
            out["content"] = 0.5
            out["embed"] = 0.5
            out["sentiment_eff"] = 0.5
            out["collab"] = 0.0
            out["n_reviews"] = 0
            out["fused"] = [round(1.0 - 0.01 * i, 4) for i in range(n)]
            return out, None, {
                "fallback": True,
                "message": "We couldn't find an apparel intent in that search — "
                           "here are the top-rated picks from the catalog instead.",
            }

        price_filter = min_price is not None or max_price is not None
        from preprocessing import COLOR_PATTERN

        color_filter = bool(COLOR_PATTERN.search(query))
        # Strict post-filters (color/price) keep only a fraction of candidates,
        # so widen the candidate pool when they are active or when a previous
        # strict pass came back empty (type-intent retry).
        widen = price_filter or color_filter or _retry
        pool = CONTENT_K * 25 if widen else CONTENT_K
        content_top = self.content_filter.get_top_k(query, k=pool)
        content_scores = dict(
            zip(content_top["parent_asin"].astype(str), content_top["score"])
        )

        t0 = time.time()
        vec = self.embed_query(query)
        sims = self.index.matrix @ vec
        embed_k = min(EMBED_K * 25 if widen else EMBED_K, len(self.index.asins))
        top_idx = np.argpartition(sims, -embed_k)[-embed_k:]
        embed_scores = {str(self.index.asins[i]): float(sims[i]) for i in top_idx}
        print(f"  embed + search done in {time.time() - t0:.1f}s")

        # Collaborative signal: top content/embedding hits seed the co-purchase graph
        seeds = build_seed_scores(content_scores, embed_scores)
        collab_scores, _via = build_collab_scores(seeds, self.adj)

        candidates = list(dict.fromkeys(list(content_scores) + list(embed_scores)))
        result = fuse(
            candidates,
            content_scores,
            embed_scores,
            self.sentiment,
            w_content=weights["content"],
            w_embed=weights["embed"],
            w_sentiment=weights["sentiment"],
            w_collab=weights.get("collab", W_COLLAB_DEFAULT),
            min_reviews=min_reviews,
            collab_scores=collab_scores,
            normalize=True,
        )
        result = apply_color_preference(result, self.titles, query, top_k, brands=self.brands)
        result = apply_type_preference(
            result, self.titles, query, fallback=False,
            garment_class=self.garment_class, brands=self.brands,
        )
        result = apply_occasion_preference(result, self.titles, query)
        result = apply_novelty_filter(
            result, self.titles, garment_class=self.garment_class, query=query
        )
        result = apply_garment_filter(result, self.titles)
        result = apply_gender_filter(result, self.titles)
        result = apply_plain_shirt_preference(
            result, self.titles, query, garment_class=self.garment_class
        )

        # Clickable-filter hard cuts (UI brand / size). Only ~31% of titles
        # carry a size, so a size cut that matches <4 items relaxes to the
        # full pool with per-item in_size flags (same pattern as price).
        note: dict = {}
        if brand:
            b = brand.strip().lower()
            in_brand = np.array([
                self.brands.get(str(a), "").strip().lower() == b
                for a in result["parent_asin"]
            ])
            if int(in_brand.sum()) >= 1:
                result = result[in_brand].reset_index(drop=True)
                in_brand = np.ones(len(result), dtype=bool)
            else:
                note["brand_relaxed"] = (
                    f"No '{brand}' in the top matches — showing closest picks instead."
                )
            result["in_brand"] = in_brand
        if size:
            s = size.strip().lower()
            in_size = np.array([
                self.sizes.get(str(a), "") == s for a in result["parent_asin"]
            ])
            if int(in_size.sum()) >= 4:
                result = result[in_size].reset_index(drop=True)
                in_size = np.ones(len(result), dtype=bool)
            else:
                note["size_relaxed"] = (
                    f"Few items list size {SIZE_OPTIONS.get(s, s)} in the title — "
                    "showing closest picks."
                )
            result["in_size"] = in_size
        if min_rating is not None:
            result, in_rating, relaxed = filter_by_rating(result, self.ratings, min_rating)
            if relaxed:
                note["rating_relaxed"] = (
                    f"Few items rated {min_rating:.0f}★ or better matched — "
                    "showing closest picks."
                )
            result["in_rating"] = in_rating

        price_info = None
        if min_price is not None or max_price is not None:
            price_info = {"min": min_price, "max": max_price, "relaxed": False}
            prices = self.details["price"].reindex(result["parent_asin"].astype(str)).to_numpy()
            in_budget = ~pd.isna(prices)
            if min_price is not None:
                in_budget &= prices >= min_price
            if max_price is not None:
                in_budget &= prices <= max_price
            if int(in_budget.sum()) >= 3:
                result = result[in_budget].reset_index(drop=True)
                in_budget = np.ones(len(result), dtype=bool)
            else:
                # ~70% of the catalog has no scraped price and most priced
                # black shirts exceed a $24 budget, so a hard cut left the
                # page with 0-2 items ("i want black shirt ... under 2000"
                # returned a single card). Keep the full pool and let the UI
                # flag which items are actually in budget.
                price_info["relaxed"] = True
                print("  few/no items priced in budget — showing closest (budget relaxed)")
            result["in_budget"] = in_budget
        if len(result) == 0 and not _retry and detect_garment_type(query):
            print("  empty after strict filters — widening pool and retrying once")
            return self.recommend(
                query, top_k, weights, min_reviews,
                min_price=min_price, max_price=max_price,
                brand=brand, size=size, min_rating=min_rating, _retry=True,
            )
        # Filters above may have stripped one signal's top hits (e.g. the
        # type filter kills TF-IDF's t-shirt top hits for "black shirt");
        # re-balance the signals to the surviving pool before re-ranking.
        result = renormalize_signals(
            result,
            {"content": weights["content"], "embed": weights["embed"],
             "sentiment": weights["sentiment"],
             "collab": weights.get("collab", W_COLLAB_DEFAULT)},
        )
        result = apply_secondary_color_boost(result, self.titles, query)
        result = self.apply_quality_rerank(result)
        result = apply_mmr(result, self.titles)
        return result, price_info, note or None

    def apply_quality_rerank(self, result: pd.DataFrame, wq: float = 0.20) -> pd.DataFrame:
        """Blend rating reliability into the fused score (Bayesian shrinkage;
        see phase7_fusion.quality_rerank for the formula)."""
        return quality_rerank(result, self.details, wq)


engine: Engine | None = None
_engine_lock = threading.Lock()
_engine_error: str | None = None


@app.on_event("startup")
def warm_up() -> None:
    """Load the engine in a background thread so the server binds fast."""

    def load() -> None:
        global engine, _engine_error
        try:
            with _engine_lock:
                if engine is None:
                    engine = Engine()
        except Exception as e:  # surface load failures to /health and requests
            _engine_error = f"{type(e).__name__}: {e}"
            print(f"ENGINE LOAD FAILED: {_engine_error}")

    threading.Thread(target=load, daemon=True).start()


def get_engine() -> Engine:
    """Return the shared engine.

    If the background warm-up is still running, block on the lock until it
    finishes instead of constructing a second Engine (which would load all
    models twice and can exhaust RAM / crash the server).
    """
    global engine, _engine_error
    if engine is None:
        with _engine_lock:  # blocks while warm-up thread holds the lock
            if engine is None:  # re-check: warm-up may have completed
                if _engine_error is not None:
                    raise HTTPException(
                        status_code=503,
                        detail=f"engine failed to load: {_engine_error}",
                    )
                try:
                    engine = Engine()  # only hit when no warm-up thread exists
                except Exception as e:
                    _engine_error = f"{type(e).__name__}: {e}"
                    raise HTTPException(
                        status_code=503, detail=f"engine failed to load: {_engine_error}"
                    )
    return engine


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    if engine is not None:
        return {"ready": True}
    return {"ready": False, "error": _engine_error}


@app.get("/api/suggest")
def suggest_endpoint(q: str = Query("", max_length=30)) -> dict:
    """Autocomplete while typing. Works even before the engine warms up
    (curated phrases + vocab); brands join in once the facets are loaded."""
    brands = [b["name"] for b in engine.facets["brands"]] if engine is not None else None
    return {"suggestions": suggest(q, brands=brands)}


@app.get("/api/brands")
def brand_search(q: str = Query("", max_length=30)) -> dict:
    """Search the FULL ~35k-brand catalog (the filter dropdown only lists
    the top 40). Prefix matches first, contains fallback for the rest."""
    eng = get_engine()
    return {"brands": search_brands(eng.brand_sorted, q)}


@app.get("/api/facets")
def facets() -> dict:
    """Brand / garment-type / size / price-bucket choices for the
    clickable filters (computed once at engine load)."""
    eng = get_engine()
    return {"usd_inr": USD_INR, **eng.facets}


@app.post("/api/recommend")
def recommend(req: RecommendRequest) -> dict:
    eng = get_engine()
    raw = req.query.strip()[:200]
    if not raw:
        raise HTTPException(status_code=422, detail="query is empty")
    size = req.size.strip().lower() if req.size else None
    if size and size not in SIZE_OPTIONS:
        raise HTTPException(status_code=422, detail=f"unknown size '{req.size}'")
    brand = req.brand.strip() if req.brand and req.brand.strip() else None

    t0 = time.time()
    # Rufus behavior: fix single-edit typos ("blak shrit" -> "black shirt")
    # and tell the client what was corrected so it can show a chip.
    query, corrections = correct_query(raw)

    from phase7_fusion import parse_price_range

    min_price, max_price = parse_price_range(query)
    if min_price is not None or max_price is not None:
        print(f"  price intent: min={min_price} max={max_price}")
    if req.min_price is not None:
        min_price = req.min_price
    if req.max_price is not None:
        max_price = req.max_price
    if req.min_price_inr:
        min_price = inr_to_usd(req.min_price_inr)
    if req.max_price_inr:
        max_price = inr_to_usd(req.max_price_inr)

    try:
        result, price_info, note = eng.recommend(
            query,
            req.top_k,
            {"content": req.w_content, "embed": req.w_embed, "sentiment": req.w_sentiment, "collab": req.w_collab},
            req.min_reviews,
            min_price=min_price,
            max_price=max_price,
            brand=brand,
            size=size,
            min_rating=req.min_rating,
        )
    except Exception as e:  # never let an engine hiccup 500 with a stack dump
        print(f"RECOMMEND FAILED: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong on our side — please try again.",
        )
    top = result.head(req.top_k)
    has_budget = "in_budget" in top.columns
    has_brand = "in_brand" in top.columns
    has_size = "in_size" in top.columns
    has_rating = "in_rating" in top.columns

    # Titles/brands are cleaned (tags + mojibake) once at Engine load.
    details = eng.details

    items = []
    for row in top.itertuples():
        asin = str(row.parent_asin)
        det = details.loc[asin] if asin in details.index else None
        rating = None
        n_ratings = None
        price = None
        image_url = ""
        title = ""
        if det is not None:
            rating = det["average_rating"]
            n_ratings = int(det["rating_number"]) if pd.notna(det["rating_number"]) else None
            price = float(det["price"]) if pd.notna(det["price"]) else None
            image_url = det["image_url"] or ""
            title = det["title"] or ""
        items.append(
            {
                "parent_asin": asin,
                "title": title,
                "brand": str(det["brand"]) if det is not None and pd.notna(det["brand"]) else "",
                "rating": rating,
                "n_ratings": n_ratings,
                "price": price,
                "image_url": image_url,
                "content_score": float(row.content),
                "embed_score": float(row.embed),
                "sentiment_score": float(row.sentiment_eff),
                "collab_score": float(row.collab),
                "n_reviews": int(row.n_reviews),
                "fused_score": float(row.fused),
                "in_budget": bool(row.in_budget) if has_budget else None,
                "in_brand": bool(row.in_brand) if has_brand else None,
                "in_size": bool(row.in_size) if has_size else None,
                "in_rating": bool(row.in_rating) if has_rating else None,
                "price_inr": usd_to_inr(price) if price is not None else None,
            }
        )

    price_out = None
    if price_info is not None:
        price_out = dict(price_info)
        if price_info.get("min") is not None:
            price_out["min_inr"] = usd_to_inr(price_info["min"])
        if price_info.get("max") is not None:
            price_out["max_inr"] = usd_to_inr(price_info["max"])

    return {
        "query": req.query,
        "corrected_query": query if corrections else None,
        "corrections": (
            [{"from": a, "to": b} for a, b in corrections] if corrections else None
        ),
        "elapsed_s": round(time.time() - t0, 1),
        "weights": {
            "content": req.w_content,
            "embed": req.w_embed,
            "sentiment": req.w_sentiment,
            "collab": req.w_collab,
        },
        "price": price_out,
        "note": note,
        "items": items,
    }


@app.get("/api/collab/{asin}")
def collab_for(asin: str, k: int = Query(6, ge=1, le=50)) -> dict:
    """Co-purchased products for the 'People also bought' strip."""
    eng = get_engine()
    nbrs = eng.adj.get(asin, [])
    if not nbrs:
        return {"asin": asin, "items": []}
    nbrs = sorted(nbrs, key=lambda t: t[1], reverse=True)[:k]
    items = []
    for nb, w in nbrs:
        a = str(nb)
        if a not in eng.details.index:
            continue
        det = eng.details.loc[a]
        items.append(
            {
                "parent_asin": a,
                "title": eng.titles.get(a, ""),
                "rating": det["average_rating"],
                "n_ratings": int(det["rating_number"]) if pd.notna(det["rating_number"]) else None,
                "price": float(det["price"]) if pd.notna(det["price"]) else None,
                "image_url": det["image_url"] or "",
                "copurchase_weight": round(float(w), 4),
            }
        )
    return {"asin": asin, "items": items}


# Static web assets (style.css / app.js). Mounted LAST so every API route
# and the / index above keep winning; unmatched paths fall through to web/.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
