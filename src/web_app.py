"""Hybrid apparel recommendation website — FastAPI backend.

Loads the Phase 7 engine once (content filter + embedding index +
query embedder + sentiment), then serves recommendations over HTTP.

Run:
    uvicorn src.web_app:app --host 127.0.0.1 --port 8000
Open: http://127.0.0.1:8000
"""

import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import threading
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
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
    apply_garment_filter,
    apply_mmr,
    apply_novelty_filter,
    apply_occasion_preference,
    apply_type_preference,
    build_collab_scores,
    build_neighbor_map,
    build_seed_scores,
    detect_garment_type,
    fuse,
    load_copurchase_edges,
)
from preprocessing import preprocess_metadata

DETAILS_PATH = DATA_PROCESSED_DIR / "product_details.parquet"
SENTIMENT_PATH = DATA_PROCESSED_DIR / "product_sentiment.parquet"
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

CONTENT_K = 200
EMBED_K = 200

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


class Engine:
    def __init__(self):
        import gc

        t0 = time.time()
        print("loading content filter ...")
        df = pd.read_parquet(PROCESSED_METADATA_PATH)
        df = preprocess_metadata(df)
        self.content_filter = ContentFilter(df)
        del df
        gc.collect()
        print(f"  {len(self.content_filter.df):,} content products in {time.time() - t0:.1f}s")

        t0 = time.time()
        print("loading embedding index ...")
        self.index = EmbeddingIndex.load(INDEX_PATH, INDEX_ASINS_PATH)
        print(f"  {len(self.index.asins):,} indexed products")

        print("loading query embedder (T5 + RoBERTa, ~1 min first time) ...")
        self.embedder = QueryEmbedder()

        t0 = time.time()
        print("loading sentiment ...")
        self.sentiment = (
            pd.read_parquet(SENTIMENT_PATH) if SENTIMENT_PATH.exists() else None
        )
        if self.sentiment is not None:
            print(f"  {len(self.sentiment):,} products with sentiment")

        print("loading co-purchase graph ...")
        edges = load_copurchase_edges()
        self.adj = build_neighbor_map(edges)
        if self.adj:
            print(f"  {len(self.adj):,} products with co-purchase neighbors")
        else:
            print("  no co-purchase edges found (collab signal off)")

        print("loading titles ...")
        titles_df = pd.read_parquet(
            __import__("config").PRODUCT_TEXTS_PATH, columns=["parent_asin", "title"]
        )
        self.titles = dict(zip(titles_df["parent_asin"].astype(str), titles_df["title"].astype(str)))
        del titles_df
        print("loading brands ...")
        brands_df = pd.read_parquet(DETAILS_PATH, columns=["parent_asin", "brand"])
        self.brands = dict(zip(brands_df["parent_asin"].astype(str), brands_df["brand"].fillna("").astype(str)))
        del brands_df
        print("loading details ...")
        self.details = pd.read_parquet(
            DETAILS_PATH,
            columns=["parent_asin", "title", "brand", "average_rating",
                     "rating_number", "price", "image_url"],
        ).set_index("parent_asin")
        gc.collect()
        print("engine ready")

    def recommend(self, query: str, top_k: int, weights: dict, min_reviews: int,
                  min_price: float | None = None, max_price: float | None = None,
                  _retry: bool = False) -> pd.DataFrame:
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
        vec = self.embedder.embed(query)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        sims = self.index.matrix @ vec
        embed_k = min(EMBED_K * 25 if widen else EMBED_K, len(self.index.asins))
        top_idx = np.argpartition(sims, -embed_k)[-embed_k:]
        embed_scores = {str(self.index.asins[i]): float(sims[i]) for i in top_idx}
        print(f"  query embedded + searched in {time.time() - t0:.1f}s")

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
        result = apply_type_preference(result, self.titles, query, fallback=False)
        result = apply_occasion_preference(result, self.titles, query)
        result = apply_novelty_filter(result, self.titles)
        result = apply_garment_filter(result, self.titles)
        if min_price is not None or max_price is not None:
            prices = self.details["price"].reindex(result["parent_asin"].astype(str)).to_numpy()
            mask = ~pd.isna(prices)
            if min_price is not None:
                mask &= prices >= min_price
            if max_price is not None:
                mask &= prices <= max_price
            result = result[mask]
        if len(result) == 0 and not _retry and detect_garment_type(query):
            print("  empty after strict filters — widening pool and retrying once")
            return self.recommend(
                query, top_k, weights, min_reviews,
                min_price=min_price, max_price=max_price, _retry=True,
            )
        result = self.apply_quality_rerank(result)
        result = apply_mmr(result, self.titles)
        return result

    def apply_quality_rerank(self, result: pd.DataFrame, wq: float = 0.20) -> pd.DataFrame:
        """Blend rating reliability into the fused score using Bayesian
        shrinkage: quality = (n·avg + c·prior) / (n + c) with prior 3.9★ and
        c = 25, scaled to 0..1. A 1-review 5★ item no longer outranks a
        1000-review 4.5★ item. final = fused * (1 - wq + wq*quality).
        """
        if len(result) == 0 or "average_rating" not in self.details.columns:
            return result
        det = self.details.reindex(result["parent_asin"].astype(str))
        avg = pd.to_numeric(det["average_rating"], errors="coerce").fillna(3.9) / 5.0
        n = pd.to_numeric(det["rating_number"], errors="coerce").fillna(0)
        prior = 3.9 / 5.0
        c = 25.0
        quality = ((n * avg + c * prior) / (n + c)).clip(0, 1)
        result = result.copy()
        multiplier = 1 - wq + wq * quality.to_numpy()
        result["fused"] = result["fused"].to_numpy() * multiplier
        return result.sort_values("fused", ascending=False).reset_index(drop=True)


engine: Engine | None = None
_engine_lock = threading.Lock()


@app.on_event("startup")
def warm_up() -> None:
    """Load the engine in a background thread so the server binds fast."""

    def load() -> None:
        global engine
        with _engine_lock:
            if engine is None:
                engine = Engine()

    threading.Thread(target=load, daemon=True).start()


def get_engine() -> Engine:
    """Return the shared engine.

    If the background warm-up is still running, block on the lock until it
    finishes instead of constructing a second Engine (which would load all
    models twice and can exhaust RAM / crash the server).
    """
    global engine
    if engine is None:
        with _engine_lock:  # blocks while warm-up thread holds the lock
            if engine is None:  # re-check: warm-up may have completed
                engine = Engine()  # only hit when no warm-up thread exists
    return engine


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"ready": engine is not None}


@app.post("/api/recommend")
def recommend(req: RecommendRequest) -> dict:
    eng = get_engine()
    query = req.query.strip().lower()[:200]
    if not query:
        raise HTTPException(status_code=422, detail="query is empty")

    t0 = time.time()
    from phase7_fusion import parse_price_range

    min_price, max_price = parse_price_range(query)
    if min_price is not None or max_price is not None:
        print(f"  price intent: min={min_price} max={max_price}")
    if req.min_price is not None:
        min_price = req.min_price
    if req.max_price is not None:
        max_price = req.max_price
    result = eng.recommend(
        query,
        req.top_k,
        {"content": req.w_content, "embed": req.w_embed, "sentiment": req.w_sentiment, "collab": req.w_collab},
        req.min_reviews,
        min_price=min_price,
        max_price=max_price,
    )
    top = result.head(req.top_k)

    details = eng.details

    def fix_mojibake(s: str) -> str:
        if not s:
            return s
        try:
            repaired = s.encode("cp1252").decode("utf-8")
            return repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s

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
            title = fix_mojibake(det["title"] or "")
        items.append(
            {
                "parent_asin": asin,
                "title": title,
                "brand": fix_mojibake(str(det["brand"]) if det is not None and pd.notna(det["brand"]) else ""),
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
            }
        )

    return {
        "query": req.query,
        "elapsed_s": round(time.time() - t0, 1),
        "weights": {
            "content": req.w_content,
            "embed": req.w_embed,
            "sentiment": req.w_sentiment,
            "collab": req.w_collab,
        },
        "items": items,
    }


@app.get("/api/collab/{asin}")
def collab_for(asin: str, k: int = 6) -> dict:
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
