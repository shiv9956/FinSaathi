"""
src/retriever.py

Loads the vector index and performs cosine similarity search.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine_sim(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return cosine similarity of query_vec against each row of matrix."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    normed = matrix / norms
    return normed @ q


# ---------------------------------------------------------------------------
# Index loader (singleton)
# ---------------------------------------------------------------------------

_chunks: Optional[list[dict]] = None
_embeddings: Optional[np.ndarray] = None


def _load_index():
    global _chunks, _embeddings
    if _chunks is not None:
        return

    if not config.INDEX_EMBEDDINGS_FILE.exists() or not config.INDEX_CHUNKS_FILE.exists():
        logger.info("Index not found — building automatically …")
        from src.ingest import run_ingestion
        _chunks, _embeddings = run_ingestion()
        return

    logger.info("Loading index from %s", config.INDEX_DIR)
    _embeddings = np.load(str(config.INDEX_EMBEDDINGS_FILE)).astype(np.float32)
    with open(config.INDEX_CHUNKS_FILE, "r", encoding="utf-8") as f:
        _chunks = json.load(f)
    logger.info("Index loaded: %d chunks, embedding dim=%d", len(_chunks), _embeddings.shape[1] if _embeddings.ndim > 1 else 0)


def reload_index():
    """Force reload (call after rebuilding the index)."""
    global _chunks, _embeddings
    _chunks = None
    _embeddings = None
    _load_index()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(query: str, k: int = config.RETRIEVAL_TOP_K) -> list[dict]:
    """
    Search the index for chunks relevant to `query`.

    Returns a list of dicts:
      {text, source_name, source_url, file, score}

    Chunks with score < RETRIEVAL_MIN_SCORE (in real mode) are excluded.
    In mock mode, a lower threshold (0.0) is used so tests always get results.
    """
    from src import watsonx_client  # local import to respect mock flag

    _load_index()

    if not _chunks or _embeddings is None or len(_chunks) == 0:
        return []

    query_vec = watsonx_client.embed_query(query)
    scores = _cosine_sim(query_vec, _embeddings)

    top_k = min(k, len(_chunks))
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Score threshold — relax in mock mode
    threshold = 0.0 if config.MOCK_MODE else config.RETRIEVAL_MIN_SCORE

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score < threshold:
            continue
        chunk = dict(_chunks[idx])
        chunk["score"] = round(score, 4)
        results.append(chunk)

    return results
