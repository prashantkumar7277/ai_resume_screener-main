"""
embedder.py -- Phase 2: Embedding Engine

Wraps sentence-transformers `all-MiniLM-L6-v2` to convert text into
384-dimensional float32 vectors for cosine similarity comparison.

Features:
  - Singleton model loading (cached via @st.cache_resource in app.py later)
  - embed_text(text) -> np.ndarray  (L2-normalised, ready for FAISS / cosine)
  - embed_batch(texts) -> np.ndarray  (efficient batch encoding)
  - Hash-based in-memory cache to avoid re-embedding identical text
  - cosine_similarity(a, b) helper for quick ad-hoc comparisons
"""

from __future__ import annotations

import hashlib
try:
    import numpy as np
except Exception as e:  # pragma: no cover - environment/dependency issue
    raise ImportError(
        "numpy is required for embedder.py. Install with: pip install numpy"
    ) from e

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover - environment/dependency issue
    raise ImportError(
        "sentence-transformers is required for embedder.py. Install with: pip install sentence-transformers"
    ) from e

# ─────────────────────────────────────────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384          # fixed output dimension for this model

_model: SentenceTransformer | None = None  # lazy-loaded singleton


def _get_model() -> SentenceTransformer:
    """
    Lazy-load and cache the SentenceTransformer model.
    First call downloads ~90 MB; subsequent calls return instantly.
    """
    global _model
    if _model is None:
        print(f"[INFO] Loading embedding model: {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[INFO] Model loaded. Embedding dimension: {EMBEDDING_DIM}")
    assert _model is not None  # narrow type for static analysis
    return _model


# ─────────────────────────────────────────────────────────────────────────────
# In-memory hash cache
# ─────────────────────────────────────────────────────────────────────────────

_embedding_cache: dict[str, np.ndarray] = {}


def _text_hash(text: str) -> str:
    """Return a short SHA-256 hex digest of the input text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def embed_text(text: str) -> np.ndarray:
    """
    Embed a single string into a normalised 384-d float32 vector.

    Caches results by SHA-256 hash so identical text is never re-encoded.

    Args:
        text: Input string (JD or resume text).

    Returns:
        np.ndarray of shape (384,), dtype=float32, L2-normalised.
    """
    if not text or not text.strip():
        raise ValueError("embed_text() received empty or whitespace-only string.")

    key = _text_hash(text)
    if key in _embedding_cache:
        return _embedding_cache[key]

    model = _get_model()
    # encode() returns (1, 384) when given a list; squeeze to (384,)
    embedding = model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalise so dot-product == cosine
        show_progress_bar=False,
    )[0].astype(np.float32)

    _embedding_cache[key] = embedding
    return embedding


def embed_batch(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings efficiently in a single model call.

    Returns cached results where available; encodes only uncached texts
    in a single batch call for maximum throughput.

    Args:
        texts: List of input strings.

    Returns:
        np.ndarray of shape (len(texts), 384), dtype=float32, L2-normalised.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    results = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    uncached_indices = []
    uncached_texts = []

    for i, text in enumerate(texts):
        key = _text_hash(text)
        if key in _embedding_cache:
            results[i] = _embedding_cache[key]
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        model = _get_model()
        batch_embeddings = model.encode(
            uncached_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(uncached_texts) > 5,
            batch_size=32,
        ).astype(np.float32)

        for idx, embedding in zip(uncached_indices, batch_embeddings):
            key = _text_hash(texts[idx])
            _embedding_cache[key] = embedding
            results[idx] = embedding

    return results


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalised vectors.

    Since embed_text() already normalises, this is just the dot product.
    Result is in [-1, 1]; for semantic text: typically [0, 1].

    Args:
        a: 1-D float32 numpy array (normalised).
        b: 1-D float32 numpy array (normalised).

    Returns:
        float in [-1, 1].
    """
    return float(np.dot(a, b))


def cache_size() -> int:
    """Return the number of cached embeddings."""
    return len(_embedding_cache)


def clear_cache() -> None:
    """Clear the in-memory embedding cache."""
    _embedding_cache.clear()
