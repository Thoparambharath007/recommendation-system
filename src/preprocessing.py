import re

import pandas as pd

SIZE_WORDS = {
    "xs", "s", "m", "l", "xl", "xxl", "xxxl", "2xl", "3xl", "4xl", "5xl", "6xl",
    "xsmall", "small", "medium", "large", "xlarge", "xxlarge",
    "big", "tall", "petite", "plus",
}
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")

COLOR_WORDS = [
    "black", "white", "navy", "blue", "grey", "gray", "green", "red",
    "maroon", "brown", "beige", "tan", "olive", "khaki", "charcoal",
    "purple", "pink", "yellow", "orange", "teal", "burgundy", "ivory",
    "cream", "coral", "indigo", "lilac", "mint", "peach", "silver",
    "gold", "copper", "heather", "multicolor", "camouflage", "denim",
]
COLOR_PATTERN = re.compile(r"\b(?:" + "|".join(COLOR_WORDS) + r")\b", re.IGNORECASE)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s&'\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_size(title: str) -> str:
    if not isinstance(title, str):
        return ""
    for m in TOKEN_PATTERN.finditer(title):
        if title[m.start() - 1 : m.start()] == "'":
            continue
        tok = m.group(0).lower()
        if tok in SIZE_WORDS:
            return tok
    return ""


def extract_color(title: str) -> str:
    if not isinstance(title, str):
        return ""
    match = COLOR_PATTERN.search(title)
    return match.group(0).lower() if match else ""


def _join(parts: list[str]) -> str:
    return " ".join(p for p in parts if p).strip()


def classify_garment(categories: str) -> str:
    """Classify a product from its RAW Amazon category path (structural signal).

    Returns one of: novelty, costume, underwear, swim, hoodie, tank, tee,
    polo, henley, shirt, unknown. Title keywords can be gamed (merch sellers
    stuff 'Black Shirt' into T-shirt titles for SEO), but the category path
    is assigned by Amazon's taxonomy and cannot be faked that way — a
    'T-Shirts' node product is a tee, a 'Novelty' node product is merch,
    regardless of what the title claims.

    'Novelty' must match the EXACT segment: 'Novelty & More' is a department
    (2nd path segment) under which legit solid shirts are filed, so a
    substring match would misclassify ~40K real shirts as merch.

    Expects the raw pipe-joined string from data_loader (e.g.
    "Clothing, Shoes & Jewelry | Men | Tops | T-Shirts"); preprocess_metadata
    strips the '|' separators, so call this on the raw parquet df.
    """
    segs = [s.strip().lower() for s in (categories or "").split("|") if s.strip()]
    if not segs:
        return "unknown"
    if "novelty" in segs:
        return "novelty"
    if any("costume" in s or "cosplay" in s for s in segs):
        return "costume"
    if any("underwear" in s or "undershirt" in s for s in segs):
        return "underwear"
    if any("swim" in s or "rash guard" in s for s in segs):
        return "swim"
    if any("hoodie" in s or "sweatshirt" in s or "sweat" in s for s in segs):
        return "hoodie"
    if any("tank" in s for s in segs):
        return "tank"
    if any("t-shirt" in s or "tshirts" in s or "t shirt" in s for s in segs):
        return "tee"
    if any("shirts & tees" in s for s in segs):
        # mixed category — the title heuristics decide instead
        return "unknown"
    if any("polo" in s for s in segs):
        return "polo"
    if any("henley" in s for s in segs):
        return "henley"
    if any("shirt" in s for s in segs):
        return "shirt"
    return "unknown"


def preprocess_metadata(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["parent_asin", "brand", "title", "categories", "bought_together"]
    df = df[[c for c in needed if c in df.columns]].copy()

    for col in df.columns:
        df[col] = df[col].astype("object")

    df["brand"] = df["brand"].map(clean_text)
    df["title"] = df["title"].map(clean_text)
    df["categories"] = df["categories"].map(clean_text)
    df["size"] = df["title"].map(extract_size)
    df["color"] = df["title"].map(extract_color)

    keep = df["brand"].str.len() > 0
    df = df[keep]
    df["metadata_string"] = [
        _join([b, c, s, col])
        for b, c, s, col in zip(df["brand"], df["categories"], df["size"], df["color"])
    ]
    keep = df["metadata_string"].str.len() > 0
    df = df[keep]
    return df.reset_index(drop=True)
