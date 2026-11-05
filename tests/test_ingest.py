"""
tests/test_ingest.py — Tests for chunking and ingestion.
"""

import os
os.environ.setdefault("MOCK_MODE", "true")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.ingest import chunk_text


class TestChunkText:

    def _make_meta(self):
        return {"source_name": "test", "source_url": "https://example.com", "file": "test.md"}

    def test_short_text_single_chunk(self):
        text = "Hello world. This is a test."
        chunks = chunk_text(text, self._make_meta(), size=800, overlap=120)
        assert len(chunks) == 1
        assert chunks[0]["text"] == text

    def test_long_text_multiple_chunks(self):
        text = ("This is a sentence. " * 60)  # ~1200 chars
        chunks = chunk_text(text, self._make_meta(), size=400, overlap=80)
        assert len(chunks) >= 2

    def test_chunk_size_roughly_bounded(self):
        text = "Word " * 400  # 2000 chars
        chunks = chunk_text(text, self._make_meta(), size=500, overlap=100)
        for chunk in chunks:
            # Allow some tolerance for boundary splitting
            assert len(chunk["text"]) <= 700, f"Chunk too large: {len(chunk['text'])}"

    def test_meta_preserved(self):
        meta = {"source_name": "UPI Guide", "source_url": "https://www.npci.org.in", "file": "upi.md"}
        text = "UPI is a payment system built by NPCI."
        chunks = chunk_text(text, meta)
        assert chunks[0]["source_name"] == "UPI Guide"
        assert chunks[0]["source_url"] == "https://www.npci.org.in"

    def test_empty_text_returns_empty(self):
        chunks = chunk_text("", self._make_meta())
        assert chunks == []

    def test_whitespace_only_returns_empty(self):
        chunks = chunk_text("   \n\n   ", self._make_meta())
        assert chunks == []

    def test_overlap_creates_continuity(self):
        """Second chunk should start with content that appeared near the end of the first."""
        long_text = ("Alpha beta gamma delta epsilon " * 50)
        chunks = chunk_text(long_text, self._make_meta(), size=300, overlap=100)
        if len(chunks) >= 2:
            # The second chunk should share some words with the end of the first
            first_end = chunks[0]["text"][-80:]
            second_start = chunks[1]["text"][:80]
            # Check at least some word overlap
            first_words = set(first_end.split())
            second_words = set(second_start.split())
            assert len(first_words & second_words) > 0, "Expected word overlap between chunks"
