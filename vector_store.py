"""
vector_store.py -- Phase 3: FAISS Vector Store

Builds a FAISS inner-product index from resume embeddings and searches it
to find the top-K most semantically similar resumes for a given JD.

Design notes:
  - Uses IndexFlatIP (exact inner product search). Since embedder.py
    returns L2-normalised vectors, inner product == cosine similarity.
  - Stores a parallel metadata list so we can map FAISS result indices
    back to resume names and original texts.
  - No disk persistence needed for MVP (stored in session_state in Phase 6).

Public API:
  build_index(resumes)               -> (VectorStore instance)
  store.search(jd_text, top_k=10)    -> list[SearchResult]
  store.size                         -> int
"""

from __future__ import annotations

import numpy as np
import faiss
from dataclasses import dataclass, field

from utils.embedder import embed_text, embed_batch, EMBEDDING_DIM


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResumeEntry:
    """Metadata stored alongside each vector in the FAISS index."""
    name: str          # candidate / filename identifier
    text: str          # cleaned resume text (used by LLM scorer later)
    faiss_idx: int     # row index inside the FAISS index


@dataclass
class SearchResult:
    """One result returned from VectorStore.search()."""
    rank: int
    name: str
    text: str
    similarity: float  # cosine similarity in [0, 1] (higher = better match)

    def __repr__(self) -> str:
        return (f"SearchResult(rank={self.rank}, name={self.name!r}, "
                f"similarity={self.similarity:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# VectorStore class
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    In-memory FAISS vector store for resume embeddings.

    Usage:
        store = build_index([
            {"name": "Alice.pdf", "text": "..."},
            {"name": "Bob.pdf",   "text": "..."},
        ])
        results = store.search(jd_text, top_k=5)
        for r in results:
            print(r.rank, r.name, r.similarity)
    """

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._entries: list[ResumeEntry] = []

    # ── Build ────────────────────────────────────────────────────────────────

    def add_resumes(self, resumes: list[dict]) -> None:
        """
        Embed and add a list of resumes to the index.

        Args:
            resumes: List of dicts with keys:
                       "name" (str) -- display name / filename
                       "text" (str) -- cleaned resume text
        """
        if not resumes:
            raise ValueError("add_resumes() received an empty list.")

        texts = [r["text"] for r in resumes]
        names = [r["name"] for r in resumes]

        print(f"[INFO] Embedding {len(texts)} resume(s)...")
        embeddings = embed_batch(texts)  # shape (N, 384), already normalised

        # FAISS expects float32 C-contiguous arrays
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        start_idx = self._index.ntotal
        self._index.add(embeddings)

        for i, (name, text) in enumerate(zip(names, texts)):
            self._entries.append(ResumeEntry(
                name=name,
                text=text,
                faiss_idx=start_idx + i,
            ))

        print(f"[INFO] Index now contains {self._index.ntotal} vector(s).")

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, jd_text: str, top_k: int = 10) -> list[SearchResult]:
        """
        Find the top-K most similar resumes for a given JD.

        Args:
            jd_text: Job description text (will be embedded internally).
            top_k:   Number of results to return. Clamped to index size.

        Returns:
            List of SearchResult sorted by similarity descending (rank 1 = best).
        """
        if self._index.ntotal == 0:
            raise RuntimeError("Vector store is empty. Call add_resumes() first.")

        k = min(top_k, self._index.ntotal)
        jd_vec = embed_text(jd_text).reshape(1, -1)           # (1, 384)
        jd_vec = np.ascontiguousarray(jd_vec, dtype=np.float32)

        # Returns (1, k) arrays of scores and index positions
        distances, indices = self._index.search(jd_vec, k)

        results: list[SearchResult] = []
        for rank, (faiss_idx, sim) in enumerate(
            zip(indices[0], distances[0]), start=1
        ):
            if faiss_idx == -1:          # FAISS pads with -1 if fewer results
                continue
            entry = self._entries[faiss_idx]
            results.append(SearchResult(
                rank=rank,
                name=entry.name,
                text=entry.text,
                similarity=float(sim),
            ))

        return results

    # ── Utilities ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of resumes currently in the index."""
        return self._index.ntotal

    def reset(self) -> None:
        """Clear the index and metadata (useful for Streamlit reruns)."""
        self._index.reset()
        self._entries.clear()

    def get_all_entries(self) -> list[ResumeEntry]:
        """Return all stored resume metadata (for debugging)."""
        return list(self._entries)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def build_index(resumes: list[dict]) -> VectorStore:
    """
    Build and return a populated VectorStore in one call.

    Args:
        resumes: List of {"name": str, "text": str} dicts.

    Returns:
        VectorStore instance ready for .search() calls.
    """
    store = VectorStore()
    store.add_resumes(resumes)
    return store
