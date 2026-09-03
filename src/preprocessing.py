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
