"""One-time download of the two HuggingFace models the server needs (~2.5 GB).

Run with internet access (setup.bat calls this). After the models are cached
in ~/.cache/huggingface/hub, the server works fully offline.

Usage:
    python scripts/fetch_models.py
"""
import os
import sys
import time
from pathlib import Path

# force ONLINE regardless of inherited env (web_app sets HF_HUB_OFFLINE when
# the models are already cached — here they are not yet)
for var in ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_NETWORK", "TRANSFORMERS_OFFLINE"):
    os.environ.pop(var, None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phase5_similarity import MODEL_ROBERTA, MODEL_T5  # noqa: E402


def main() -> int:
    from sentence_transformers import SentenceTransformer

    total_t0 = time.time()
    for name in (MODEL_T5, MODEL_ROBERTA):
        t0 = time.time()
        print(f"downloading/loading {name} ... (first time downloads ~GB, be patient)")
        model = SentenceTransformer(name)
        dims = model.get_embedding_dimension()
        print(f"  OK: {name} -> {dims}-dim embeddings ({time.time() - t0:.0f}s)")

    # sanity: the cascade must produce 1536-d vectors to match the shared index
    vec = SentenceTransformer(MODEL_T5).encode(
        ["test shirt"], convert_to_numpy=True, normalize_embeddings=False
    )
    assert vec.shape[1] == 768, f"expected 768 dims from T5, got {vec.shape[1]}"
    print(f"\nall models cached and verified in {time.time() - total_t0:.0f}s")
    print("you can now start the server offline: double-click start.bat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
