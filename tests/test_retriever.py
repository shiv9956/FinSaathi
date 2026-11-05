"""
tests/test_retriever.py — Tests for retriever using mock embeddings.
"""

import os
os.environ.setdefault("MOCK_MODE", "true")
os.environ["MOCK_MODE"] = "true"

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


class TestRetriever:

    def _build_temp_index(self, tmp_path: Path):
        """Build a small in-memory mock index for testing."""
        # Patch config to use tmp_path
        import config
        config.INDEX_EMBEDDINGS_FILE = tmp_path / "embeddings.npy"
        config.INDEX_CHUNKS_FILE = tmp_path / "chunks.json"
        config.EMBED_CACHE_FILE = tmp_path / "embed_cache.json"
        config.MOCK_MODE = True

        # Reload the retriever module state
        import src.retriever as retriever
        retriever._chunks = None
        retriever._embeddings = None

        # Create small mock index
        from src.watsonx_client import embed_texts
        chunks = [
            {"text": "UPI is a payment system. Use it to send money.", "source_name": "UPI Guide", "source_url": "https://www.npci.org.in", "file": "upi.md"},
            {"text": "Never share your OTP with anyone, even bank staff.", "source_name": "Safety Guide", "source_url": "https://rbikehtahai.rbi.org.in", "file": "safety.md"},
            {"text": "EMI is the monthly instalment for a loan repayment.", "source_name": "Loans", "source_url": "https://www.rbi.org.in", "file": "loans.md"},
            {"text": "Fixed deposits give safe returns at a fixed rate.", "source_name": "Savings", "source_url": "https://www.rbi.org.in", "file": "savings.md"},
        ]
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

        tmp_path.mkdir(parents=True, exist_ok=True)
        np.save(str(config.INDEX_EMBEDDINGS_FILE), embeddings)
        with open(config.INDEX_CHUNKS_FILE, "w") as f:
            json.dump(chunks, f)

        return retriever, chunks

    def test_search_returns_results(self, tmp_path):
        retriever, chunks = self._build_temp_index(tmp_path)
        results = retriever.search("How do I pay with UPI?")
        assert len(results) >= 1

    def test_search_results_have_required_keys(self, tmp_path):
        retriever, chunks = self._build_temp_index(tmp_path)
        results = retriever.search("What is OTP?")
        assert len(results) >= 1
        for r in results:
            assert "text" in r
            assert "source_name" in r
            assert "score" in r

    def test_search_respects_k_limit(self, tmp_path):
        retriever, chunks = self._build_temp_index(tmp_path)
        results = retriever.search("finance", k=2)
        assert len(results) <= 2

    def test_scores_between_0_and_1(self, tmp_path):
        retriever, chunks = self._build_temp_index(tmp_path)
        results = retriever.search("UPI payment")
        for r in results:
            assert 0 <= r["score"] <= 1.0
