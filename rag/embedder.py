"""Local embedding model — sentence-transformers all-MiniLM-L6-v2.

Why this model
--------------
* Fully local: no API key, no network call after first download.
* Small (80 MB) and fast: CPU-inference is acceptable for this project.
* 384-dimensional dense vectors: good quality/speed trade-off.
* Widely used benchmark baseline for semantic similarity.

The model is loaded lazily on first call and cached for the process lifetime.
"""
from __future__ import annotations

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None   # lazily initialised SentenceTransformer


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed.\n"
                "Run:  pip install sentence-transformers"
            )
        print(f"[embedder] loading {MODEL_NAME} (first call only)…")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Return a list of float vectors, one per input text."""
    model  = _get_model()
    vecs   = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vecs.tolist()


def embed_one(text: str) -> list[float]:
    """Convenience wrapper for a single text."""
    return embed([text])[0]
