import gzip
import json
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import MEN_CATEGORY_TOKENS, PROCESSED_METADATA_PATH, RAW_METADATA_PATH

COLUMNS = [
    "parent_asin",
    "brand",
    "title",
    "categories",
    "description",
    "features",
    "bought_together",
]
CHUNK_SIZE = 100_000


def iter_jsonl_gz(path: str) -> Iterable[dict[str, Any]]:
    """Stream records from a JSONL.gz file, skipping unparseable lines.

    The 4GB Amazon metadata file can contain a truncated final line or
    occasional encoding artifacts; a single bad line should not kill the
    whole rebuild. ``errors="replace"`` turns undecodable bytes into U+FFFD
    instead of a UnicodeDecodeError.
    """
    skipped = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        print(f"  skipped {skipped} unparseable line(s) in {path}")


def is_men_shirt(categories: list[str]) -> bool:
    if not categories:
        return False
    lowered = [c.strip().lower() for c in categories]
    is_men = any(c in ("men", "men's", "mens") for c in lowered)
    is_top = any(any(tok in c for tok in MEN_CATEGORY_TOKENS) for c in lowered)
    return is_men and is_top


def extract_record(d: dict[str, Any]) -> dict[str, Any]:
    categories = d.get("categories") or []
    return {
        "parent_asin": d.get("parent_asin"),
        "brand": (d.get("store") or "").strip(),
        "title": (d.get("title") or "").strip(),
        "categories": " | ".join(categories),
        "description": " ".join(d.get("description") or []).strip(),
        "features": " ".join(d.get("features") or []).strip(),
        "bought_together": json.dumps(d.get("bought_together") or []),
    }


def _write_parquet_chunk(records: list[dict[str, Any]], writer: pq.ParquetWriter) -> None:
    table = pa.Table.from_pylist(records)
    table = table.select([c for c in COLUMNS if c in table.column_names])
    writer.write_table(table)


def build_men_shirts_df(path: str = str(RAW_METADATA_PATH)) -> None:
    PROCESSED_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("parent_asin", pa.string()),
            pa.field("brand", pa.string()),
            pa.field("title", pa.string()),
            pa.field("categories", pa.string()),
            pa.field("description", pa.string()),
            pa.field("features", pa.string()),
            pa.field("bought_together", pa.string()),
        ]
    )
    records = []
    total = 0
    with pq.ParquetWriter(PROCESSED_METADATA_PATH, schema, compression="snappy") as writer:
        for d in iter_jsonl_gz(path):
            if is_men_shirt(d.get("categories") or []):
                records.append(extract_record(d))
                total += 1
                if len(records) >= CHUNK_SIZE:
                    _write_parquet_chunk(records, writer)
                    records.clear()
        if records:
            _write_parquet_chunk(records, writer)
    print(f"wrote {total:,} men's shirt products -> {PROCESSED_METADATA_PATH}")


def load_men_shirts_metadata(force_rebuild: bool = False) -> pd.DataFrame:
    if not force_rebuild and PROCESSED_METADATA_PATH.exists():
        return pd.read_parquet(PROCESSED_METADATA_PATH)
    build_men_shirts_df()
    return pd.read_parquet(PROCESSED_METADATA_PATH)
