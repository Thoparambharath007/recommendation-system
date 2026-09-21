from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

DATASETS_DIR = BASE_DIR / "datasets"
# Original dev-machine location; the repo's datasets/ folder is preferred so
# the project is self-contained (the raw files are shared separately anyway).
_DOCUMENTS_DIR = Path(r"C:\Users\thopa\OneDrive\Desktop\project\documents")


def _raw_path(filename: str) -> Path:
    """Resolve a raw data file: repo datasets/ first, then documents/."""
    repo = DATASETS_DIR / filename
    return repo if repo.exists() else _DOCUMENTS_DIR / filename


RAW_METADATA_PATH = _raw_path("meta_Clothing_Shoes_and_Jewelry.jsonl.gz")
RAW_ZERO_CORE_PATH = _raw_path("Clothing_Shoes_and_Jewelry.csv.gz")
RAW_REVIEWS_PATH = _raw_path("Clothing_Shoes_and_Jewelry.jsonl.gz")
PROCESSED_METADATA_PATH = DATA_PROCESSED_DIR / "men_shirts_metadata.parquet"
CO_PURCHASE_EDGES_PATH = DATA_PROCESSED_DIR / "copurchase_edges.parquet"
PRODUCT_TEXTS_PATH = DATA_PROCESSED_DIR / "product_texts.parquet"
REVIEWS_INDIVIDUAL_PATH = DATA_PROCESSED_DIR / "reviews_individual.parquet"
PRODUCT_SENTIMENT_PATH = DATA_PROCESSED_DIR / "product_sentiment.parquet"

TOP_K = 100
MAX_PRODUCTS_PER_USER = 50

MEN_CATEGORY_TOKENS = ("shirt", "shirts", "tee", "tees", "top", "tops")
