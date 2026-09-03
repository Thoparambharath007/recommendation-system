from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_METADATA_PATH = Path(
    r"C:\Users\thopa\OneDrive\Desktop\project\documents\meta_Clothing_Shoes_and_Jewelry.jsonl.gz"
)
RAW_ZERO_CORE_PATH = Path(
    r"C:\Users\thopa\OneDrive\Desktop\project\documents\Clothing_Shoes_and_Jewelry.csv.gz"
)
RAW_REVIEWS_PATH = Path(
    r"C:\Users\thopa\OneDrive\Desktop\project\documents\Clothing_Shoes_and_Jewelry.jsonl.gz"
)
PROCESSED_METADATA_PATH = DATA_PROCESSED_DIR / "men_shirts_metadata.parquet"
CO_PURCHASE_EDGES_PATH = DATA_PROCESSED_DIR / "copurchase_edges.parquet"
PRODUCT_TEXTS_PATH = DATA_PROCESSED_DIR / "product_texts.parquet"
REVIEWS_INDIVIDUAL_PATH = DATA_PROCESSED_DIR / "reviews_individual.parquet"
PRODUCT_SENTIMENT_PATH = DATA_PROCESSED_DIR / "product_sentiment.parquet"

TOP_K = 100
MAX_PRODUCTS_PER_USER = 50

MEN_CATEGORY_TOKENS = ("shirt", "shirts", "tee", "tees", "top", "tops")
