import argparse
import gzip
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from config import (
    CO_PURCHASE_EDGES_PATH,
    MAX_PRODUCTS_PER_USER,
    PROCESSED_METADATA_PATH,
    RAW_ZERO_CORE_PATH,
)

PROGRESS_EVERY = 5_000_000


def load_catalog(path: Path = PROCESSED_METADATA_PATH) -> set[str]:
    df = pd.read_parquet(path, columns=["parent_asin"])
    return {str(v) for v in df["parent_asin"].dropna()}


def iter_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            yield line.rstrip("\n").split(",")


def build_copurchase_edges(
    csv_path: Path = RAW_ZERO_CORE_PATH,
    catalog: set[str] | None = None,
    max_per_user: int = MAX_PRODUCTS_PER_USER,
) -> pd.DataFrame:
    if catalog is None:
        catalog = load_catalog()

    user_products: dict[str, set[str]] = defaultdict(set)
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    total = 0
    matched = 0
    t0 = time.time()

    for parts in iter_rows(csv_path):
        if parts[0] == "user_id":
            continue
        total += 1
        user_id, parent_asin = parts[0], parts[1]
        if parent_asin not in catalog:
            continue
        products = user_products[user_id]
        if len(products) < max_per_user:
            products.add(parent_asin)
            matched += 1
        if total % PROGRESS_EVERY == 0:
            print(
                f"  {total:,} rows, {len(user_products):,} users, "
                f"{len(pair_counts):,} edges, {time.time() - t0:.0f}s"
            )

    for products in user_products.values():
        items = sorted(products)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pair_counts[(items[i], items[j])] += 1

    edges = pd.DataFrame(
        {
            "product_a": [k[0] for k in pair_counts],
            "product_b": [k[1] for k in pair_counts],
            "weight": list(pair_counts.values()),
        }
    )
    print(
        f"built {len(edges):,} co-purchase edges from {len(user_products):,} "
        f"users / {matched:,} matched rows in {time.time() - t0:.0f}s"
    )
    return edges


def save_edges(df: pd.DataFrame, path: Path = CO_PURCHASE_EDGES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"saved {len(df):,} edges -> {path}")


def load_edges(path: Path = CO_PURCHASE_EDGES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def get_neighbors(edges: pd.DataFrame, asin: str, top_k: int = 20) -> pd.DataFrame:
    mask = (edges["product_a"] == asin) | (edges["product_b"] == asin)
    rel = edges[mask]
    other = rel["product_b"].where(rel["product_a"] == asin, rel["product_a"])
    out = rel.assign(neighbor=other)
    out = out[["neighbor", "weight"]]
    out = out.groupby("neighbor", as_index=False)["weight"].sum()
    out = out.sort_values("weight", ascending=False).head(top_k)
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: co-purchase inference (item-based collaborative filtering)"
    )
    parser.add_argument("--expand", metavar="ASIN", help="show related products for a product")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    if args.expand:
        edges = load_edges()
        print(f"loaded {len(edges):,} edges")
        neighbors = get_neighbors(edges, args.expand, top_k=args.top_k)
        print(f"\nneighbors of {args.expand}:")
        print(neighbors.to_string(index=False))
        return

    t0 = time.time()
    print("loading catalog ...")
    catalog = load_catalog()
    print(f"  {len(catalog):,} products in {time.time() - t0:.1f}s")

    edges = build_copurchase_edges(catalog=catalog)
    save_edges(edges)


if __name__ == "__main__":
    main()
