"""Install checker: verifies everything needed to run the recommendation
server, so failures show up HERE with plain messages instead of as a dead
web page later.

Usage:
    python scripts/check_install.py
Exit code 0 = ready to run start.bat; 1 = something must be fixed first.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

ok_all = True


def report(ok: bool, label: str, fix: str = "") -> None:
    global ok_all
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}")
    if not ok:
        ok_all = False
        if fix:
            print(f"         -> fix: {fix}")


def check_python() -> None:
    v = sys.version_info
    report(v >= (3, 11), f"Python {v.major}.{v.minor}.{v.micro} (3.11+ required)",
           "install Python 3.11+ from python.org and re-run setup.bat")


def check_deps() -> None:
    for mod in ("pandas", "numpy", "sklearn", "scipy", "pyarrow", "fastapi",
                "uvicorn", "sentence_transformers", "torch"):
        try:
            __import__(mod)
            report(True, f"dependency: {mod}")
        except Exception as e:
            report(False, f"dependency: {mod}", f"pip install -r requirements.txt ({e})")


def check_models() -> None:
    hub = Path(os.environ.get("HF_HUB_CACHE")
               or Path.home() / ".cache" / "huggingface" / "hub")
    for slug in ("models--sentence-transformers--sentence-t5-xl",
                 "models--sentence-transformers--all-distilroberta-v1"):
        report((hub / slug).exists(), f"HF model cached: {slug}",
               "run: python scripts/fetch_models.py  (needs internet, one time)")


def check_data() -> None:
    try:
        from config import (
            DATA_PROCESSED_DIR,
            PROCESSED_METADATA_PATH,
            PRODUCT_TEXTS_PATH,
        )
    except Exception as e:
        report(False, "src/config.py importable", str(e))
        return
    required = {
        "men_shirts_metadata.parquet": PROCESSED_METADATA_PATH,
        "embeddings_index.npy": DATA_PROCESSED_DIR / "embeddings_index.npy",
        "embeddings_asins.parquet": DATA_PROCESSED_DIR / "embeddings_asins.parquet",
        "product_texts.parquet": PRODUCT_TEXTS_PATH,
        "product_details.parquet": DATA_PROCESSED_DIR / "product_details.parquet",
        "product_sentiment.parquet": DATA_PROCESSED_DIR / "product_sentiment.parquet",
        "copurchase_edges.parquet": DATA_PROCESSED_DIR / "copurchase_edges.parquet",
    }
    for name, path in required.items():
        # collab edges + sentiment are graceful-degrade (neutral scores), but
        # the shared folder contains them, so still flag their absence
        report(path.exists(), f"data file: {name}",
               "download the shared data/processed folder (~2.8 GB) into the "
               "project - see README step 3")


def main() -> int:
    print("StyleSense install check")
    print("=" * 40)
    check_python()
    check_deps()
    check_models()
    check_data()
    print("=" * 40)
    if ok_all:
        print("READY - double-click start.bat, then open http://127.0.0.1:8000")
        return 0
    print("NOT READY - fix the FAIL lines above, then re-run this check.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
