import argparse
import time

from content_filter import ContentFilter
from data_loader import load_men_shirts_metadata
from preprocessing import clean_text, preprocess_metadata


def build_input(brand: str, category: str, size: str, color: str) -> str:
    parts = [clean_text(p) for p in (brand, category, size, color)]
    return " ".join(p for p in parts if p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: content-based filtering (TF-IDF + cosine)")
    parser.add_argument("--brand", default="")
    parser.add_argument("--category", default="")
    parser.add_argument("--size", default="")
    parser.add_argument("--color", default="")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    user_input = build_input(args.brand, args.category, args.size, args.color)
    if not user_input:
        parser.error("at least one of --brand/--category/--size/--color is required")

    t0 = time.time()
    print(f"Loading men's shirts metadata ...")
    df = load_men_shirts_metadata()
    df = preprocess_metadata(df)
    print(f"  {len(df):,} products in {time.time() - t0:.1f}s")

    t1 = time.time()
    cf = ContentFilter(df)
    print(f"  TF-IDF fit: {cf.matrix.shape[0]:,} docs x {cf.matrix.shape[1]:,} terms in {time.time() - t1:.1f}s")

    t2 = time.time()
    results = cf.get_top_k(user_input, k=args.top_k)
    print(f"  query '{user_input}' matched in {time.time() - t2:.2f}s\n")

    if results["score"].max() == 0:
        print(f"WARNING: no products matched '{user_input}' (all scores 0). Check your spelling.\n")

    print(f"{'rank':<5}{'score':<10}{'brand':<20}{'size':<8}{'color':<12}{'title'}")
    print("-" * 110)
    for i, row in results.iterrows():
        print(
            f"{i + 1:<5}{row['score']:<10.4f}{row['brand'][:19]:<20}"
            f"{row['size'][:7]:<8}{row['color'][:11]:<12}{row['title'][:70]}"
        )


if __name__ == "__main__":
    main()
