"""
tests/test_agent.py — Tests for the agent router.
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


def _build_temp_index(tmp_path: Path):
    """Shared helper to set up a mock index for agent tests."""
    import config
    config.MOCK_MODE = True
    config.INDEX_EMBEDDINGS_FILE = tmp_path / "embeddings.npy"
    config.INDEX_CHUNKS_FILE = tmp_path / "chunks.json"
    config.EMBED_CACHE_FILE = tmp_path / "embed_cache.json"

    import src.retriever as retriever
    retriever._chunks = None
    retriever._embeddings = None

    from src.watsonx_client import embed_texts
    chunks = [
        {"text": "UPI is a real-time digital payment system built by NPCI.", "source_name": "UPI Guide", "source_url": "https://www.npci.org.in", "file": "upi.md"},
        {"text": "KYC fraud: fraudsters send fake messages about expired KYC to steal your data.", "source_name": "Scam Guide", "source_url": "https://rbikehtahai.rbi.org.in", "file": "scams.md"},
    ]
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    tmp_path.mkdir(parents=True, exist_ok=True)
    np.save(str(config.INDEX_EMBEDDINGS_FILE), embeddings)
    with open(config.INDEX_CHUNKS_FILE, "w") as f:
        json.dump(chunks, f)


class TestAgentRouting:

    def test_emi_route_with_numbers(self, tmp_path):
        _build_temp_index(tmp_path)
        from src.agent import answer
        result = answer("Calculate EMI for loan 100000 at 10 percent for 24 months")
        assert result["tool_used"] == "emi_calculator"
        assert "EMI" in result["answer"] or "emi" in result["answer"].lower()

    def test_emi_route_asks_for_numbers(self, tmp_path):
        _build_temp_index(tmp_path)
        from src.agent import answer
        result = answer("I need a loan EMI calculation")
        # Without numbers, should ask for them
        assert result["tool_used"] in ("emi_calculator", "emi_prompt", "rag")

    def test_budget_route(self, tmp_path):
        _build_temp_index(tmp_path)
        from src.agent import answer
        result = answer("Help me budget my monthly salary of 30000")
        assert result["tool_used"] == "budget_planner"
        assert "Needs" in result["answer"] or "needs" in result["answer"].lower() or "50" in result["answer"]

    def test_scam_route(self, tmp_path):
        _build_temp_index(tmp_path)
        from src.agent import answer
        result = answer("Is this a scam: Your KYC expired click here to update")
        assert result["tool_used"] == "scam_checker"
        assert "risk" in result

    def test_rag_route_returns_answer(self, tmp_path):
        _build_temp_index(tmp_path)
        from src.agent import answer
        result = answer("What is UPI?")
        assert result["answer"]
        assert len(result["answer"]) > 10

    def test_investment_refusal(self, tmp_path):
        _build_temp_index(tmp_path)
        from src.agent import answer
        result = answer("Which stock should I buy?")
        assert result["tool_used"] == "refusal"
        assert "SEBI" in result["answer"]

    def test_always_returns_dict(self, tmp_path):
        _build_temp_index(tmp_path)
        from src.agent import answer
        result = answer("")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "tool_used" in result

    def test_hindi_language_accepted(self, tmp_path):
        _build_temp_index(tmp_path)
        from src.agent import answer
        result = answer("UPI kya hai?", language="Hindi")
        assert result["answer"]
