"""One-shot bootstrap, run automatically by start.bat on every launch.

Idempotent and cheap when nothing needs doing (finishes in <1 s via
importlib.metadata probes - no heavy imports). Three stages:

  [1] deps    - if any package from requirements.txt is missing, run
                pip install (first run only)
  [2] models  - if the two HuggingFace models are not cached, download them
                via scripts/fetch_models.py (~2.5 GB, first run only,
                needs internet)
  [3] data    - WARN (do not fail) when the shared data/processed folder is
                absent; the server still starts and shows the same guidance
                on /health, so this must never block launch.

Exit codes: 0 = ready, start the server. 1 = setup failed (message printed;
start.bat then stops and tells the user to re-run).
"""
import importlib.metadata as md
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

PKGS = (
    "pandas", "numpy", "scikit-learn", "scipy", "pyarrow",
    "fastapi", "uvicorn", "sentence-transformers", "torch",
)
MODEL_SLUGS = (
    "models--sentence-transformers--sentence-t5-xl",
    "models--sentence-transformers--all-distilroberta-v1",
)
DATA_MARKERS = (  # enough to say "the shared folder is in place"
    "men_shirts_metadata.parquet",
    "embeddings_index.npy",
    "embeddings_asins.parquet",
    "product_texts.parquet",
)


def missing_deps() -> list[str]:
    out = []
    for pkg in PKGS:
        try:
            md.version(pkg)
        except md.PackageNotFoundError:
            out.append(pkg)
    return out


def models_cached() -> bool:
    hub = pathlib.Path(
        os.environ.get("HF_HUB_CACHE")
        or pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    )
    return all((hub / slug).exists() for slug in MODEL_SLUGS)


def data_missing() -> list[str]:
    folder = ROOT / "data" / "processed"
    return [name for name in DATA_MARKERS if not (folder / name).exists()]


def ensure_python() -> bool:
    if sys.version_info < (3, 11):
        print(
            f"[ERROR] This project needs Python 3.11+ "
            f"(found {sys.version_info.major}.{sys.version_info.minor})."
        )
        print("        Install a newer Python from python.org and re-run start.bat.")
        return False
    return True


def ensure_deps() -> bool:
    pkgs = missing_deps()
    if not pkgs:
        return True
    print(f"[setup] installing dependencies (first run only): {', '.join(pkgs)}")
    print("[setup] running pip install -r requirements.txt ... this takes a few minutes")
    if subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                       str(ROOT / "requirements.txt")]).returncode != 0:
        print("[ERROR] pip install failed - check your internet connection,")
        print("        then run start.bat again (it resumes where it left off).")
        return False
    still = missing_deps()
    if still:
        print(f"[ERROR] still missing after install: {', '.join(still)}")
        print("        Manual fix:  python -m pip install -r requirements.txt")
        return False
    return True


def ensure_models() -> bool:
    if models_cached():
        return True
    print("[setup] downloading the two AI models (~2.5 GB, one time, needs internet)")
    print("[setup] this can take 5-15 minutes depending on your connection ...")
    if subprocess.run([sys.executable,
                       str(ROOT / "scripts" / "fetch_models.py")]).returncode != 0:
        print("[ERROR] model download failed - re-run start.bat with internet")
        print("        (downloads resume where they left off).")
        return False
    return True


def check_data() -> None:
    missing = data_missing()
    if not missing:
        return
    # warn only: the server starts anyway and shows this on /health
    print("[warn] the shared data/processed folder is not in place yet:")
    for name in missing:
        print(f"[warn]   missing: {name}")
    print("[warn] The server WILL start (UI + autocomplete work), but search")
    print("[warn] results need the folder: see README 'Setup for Team Members'")
    print("[warn] step 3 - download the shared data/processed folder (~2.8 GB)")
    print("[warn] and place it in the project root, then restart start.bat.")
    print()


def main() -> int:
    if not ensure_python():
        return 1
    if not ensure_deps():
        return 1
    if not ensure_models():
        return 1
    check_data()
    print("[setup] all good - starting server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
