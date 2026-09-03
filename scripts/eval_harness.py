# -*- coding: utf-8 -*-
"""Offline evaluation harness for the hybrid recommender.

Stage A (expensive, cached): for each eval query compute content scores and
embedding scores once. Stage B (cheap): grid-search fusion weights and score
each ranking with NDCG@10 / P@10 using title-based relevance judgments.

Usage (from REAL SYSTEM/):
    set HF_HUB_OFFLINE=1
    python scripts/eval_harness.py --queries 10 --grid quick

Outputs: results/eval_results.json (+ results/eval_cache.json)
"""
import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "results"
CACHE_PATH = RESULTS_DIR / "eval_cache.json"
OUT_PATH = RESULTS_DIR / "eval_results.json"

# Landmarked eval queries: relevance is graded from title keywords.
# must: all required; any: at least one boosts relevance; ban: hard zero.
EVAL_QUERIES = [
    {"query": "white shirt for office", "must": ["shirt"], "any": ["dress", "formal", "oxford", "button"], "ban": ["t-shirt", "t shirt", "tank"]},
    {"query": "black shirt with white buttons", "must": ["shirt"], "any": ["button"], "ban": ["t-shirt", "t shirt", "tank"]},
    {"query": "black tshirt", "must": [], "any": ["t-shirt", "t shirt"], "ban": []},
    {"query": "blue denim shirt", "must": ["shirt"], "any": ["denim"], "ban": ["t-shirt", "t shirt"]},
    {"query": "flannel shirt", "must": ["shirt"], "any": ["flannel", "checked", "plaid"], "ban": ["t-shirt", "t shirt"]},
    {"query": "linen shirt for beach", "must": ["shirt"], "any": ["linen"], "ban": ["t-shirt", "t shirt"]},
    {"query": "slim fit formal shirt", "must": ["shirt"], "any": ["slim", "formal"], "ban": ["t-shirt", "t shirt"]},
    {"query": "casual check shirt", "must": ["shirt"], "any": ["check", "checked", "casual"], "ban": ["t-shirt", "t shirt"]},
    {"query": "white t shirt pack", "must": [], "any": ["t-shirt", "t shirt", "pack"], "ban": []},
    {"query": "maroon polo shirt", "must": ["shirt"], "any": ["polo", "maroon"], "ban": ["t-shirt", "t shirt"]},
]

CONTENT_POOL = 500
EMBED_POOL = 500


def load_engine_bits():
    import pandas as pd
    from config import PRODUCT_SENTIMENT_PATH, PRODUCT_TEXTS_PATH, PROCESSED_METADATA_PATH
    from content_filter import ContentFilter
    from phase5_similarity import INDEX_ASINS_PATH, INDEX_PATH, EmbeddingIndex, QueryEmbedder
    from preprocessing import preprocess_metadata

    t0 = time.time()
    df = pd.read_parquet(PROCESSED_METADATA_PATH)
    df = preprocess_metadata(df)
    cf = ContentFilter(df)
    del df
    print(f"content filter ready in {time.time() - t0:.1f}s")

    index = EmbeddingIndex.load(INDEX_PATH, INDEX_ASINS_PATH)
    print("embedding index ready")
    embedder = QueryEmbedder()
    print("query embedder ready")

    sent = None
    if PRODUCT_SENTIMENT_PATH.exists():
        sent = pd.read_parquet(PRODUCT_SENTIMENT_PATH)
    tdf = pd.read_parquet(PRODUCT_TEXTS_PATH, columns=["parent_asin", "title"])
    titles = dict(zip(tdf["parent_asin"].astype(str), tdf["title"].astype(str)))
    return cf, index, embedder, sent, titles


def stage_a(query: str, cf, index, embedder):
    import numpy as np

    content_top = cf.get_top_k(query, k=CONTENT_POOL)
    content_scores = dict(zip(content_top["parent_asin"].astype(str), content_top["score"]))
    vec = embedder.embed(query)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    sims = index.matrix @ vec
    k = min(EMBED_POOL, len(index.asins))
    top_idx = np.argpartition(sims, -k)[-k:]
    embed_scores = {str(index.asins[i]): float(sims[i]) for i in top_idx}
    return {"content_scores": content_scores, "embed_scores": embed_scores}


def grade(title: str, spec: dict) -> int:
    t = f" {str(title).lower()} "
    for b in spec["ban"]:
        if b in t:
            return 0
    must_ok = all(m in t for m in spec["must"])
    any_ok = any(a in t for a in spec["any"]) if spec["any"] else True
    if spec["must"] and must_ok and any_ok:
        return 2
    if must_ok or (not spec["must"] and any_ok):
        return 1
    return 0


def ndcg_at_k(rels: list, k: int = 10) -> float:
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    ideal = sorted(rels, reverse=True)
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def rank_for(query: str, spec: dict, cached: dict, sent, titles, weights):
    from phase7_fusion import (
        apply_color_preference,
        apply_garment_filter,
        apply_mmr,
        apply_novelty_filter,
        apply_occasion_preference,
        apply_type_preference,
        fuse,
    )

    candidates = list(dict.fromkeys(
        list(cached["content_scores"]) + list(cached["embed_scores"])
    ))
    result = fuse(
        candidates,
        cached["content_scores"],
        cached["embed_scores"],
        sent,
        w_content=weights[0],
        w_embed=weights[1],
        w_sentiment=weights[2],
        w_collab=weights[3],
        collab_scores=cached.get("collab_scores"),
    )
    result = apply_color_preference(result, titles, query, 10)
    result = apply_type_preference(result, titles, query, fallback=True)
    result = apply_occasion_preference(result, titles, query)
    result = apply_novelty_filter(result, titles)
    result = apply_garment_filter(result, titles)
    result = apply_mmr(result, titles)
    top = result.head(10)["parent_asin"].astype(str).tolist()
    rels = [grade(titles.get(a, ""), spec) for a in top]
    p10 = sum(1 for r in rels if r > 0) / 10.0
    return ndcg_at_k(rels), p10, top


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline eval + fusion weight grid search")
    parser.add_argument("--queries", type=int, default=len(EVAL_QUERIES))
    parser.add_argument("--grid", choices=["quick", "full"], default="quick")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    specs = EVAL_QUERIES[: args.queries]

    cache = {}
    if CACHE_PATH.exists() and not args.no_cache:
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        print(f"loaded cache for {len(cache)} queries")

    print("loading engine components ...")
    cf, index, embedder, sent, titles = load_engine_bits()

    for spec in specs:
        q = spec["query"]
        if q in cache:
            continue
        t0 = time.time()
        print(f"embedding + content: {q!r} ...")
        cache[q] = stage_a(q, cf, index, embedder)
        cache[q]["elapsed_s"] = round(time.time() - t0, 1)
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"  done in {cache[q]['elapsed_s']}s (cached)")

    if args.grid == "quick":
        g1 = [0.2, 0.3, 0.4]
        g2 = [0.3, 0.4, 0.5]
        g3 = [0.1, 0.2, 0.3]
        g4 = [0.0, 0.1, 0.15, 0.2]
    else:
        steps = [round(0.05 * i, 2) for i in range(13)]
        g1, g2, g3 = steps, steps, steps
        g4 = [0.0, 0.05, 0.1, 0.15, 0.2]

    combos = [
        (a, b, c, d)
        for a, b, c, d in itertools.product(g1, g2, g3, g4)
        if 0.7 <= a + b + c + d <= 1.3
    ]
    print(f"grid search over {len(combos)} weight combos x {len(specs)} queries ...")

    rows = []
    for i, w in enumerate(combos):
        ndcgs, p10s = [], []
        for spec in specs:
            nd, p10, _top = rank_for(spec["query"], spec, cache[spec["query"]], sent, titles, w)
            ndcgs.append(nd)
            p10s.append(p10)
        rows.append({
            "w_content": w[0], "w_embed": w[1], "w_sentiment": w[2], "w_collab": w[3],
            "ndcg10": round(sum(ndcgs) / len(ndcgs), 4),
            "p10": round(sum(p10s) / len(p10s), 4),
        })
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(combos)} combos scored")

    rows.sort(key=lambda r: (r["ndcg10"], r["p10"]), reverse=True)
    best = rows[0]

    per_query = []
    bw = (best["w_content"], best["w_embed"], best["w_sentiment"], best["w_collab"])
    for spec in specs:
        nd, p10, top = rank_for(spec["query"], spec, cache[spec["query"]], sent, titles, bw)
        per_query.append({"query": spec["query"], "ndcg10": round(nd, 4), "p10": round(p10, 4), "top10": top})

    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_queries": len(specs),
        "n_combos": len(combos),
        "best_weights": best,
        "top5_combos": rows[:5],
        "per_query_best": per_query,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nBEST WEIGHTS:", best)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()