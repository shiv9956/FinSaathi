"""
tests/test_guardrails.py — Tests for PII masking and safety checks.
"""

import os
os.environ.setdefault("MOCK_MODE", "true")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.guardrails import mask_pii, check_input, process_output


class TestMaskPii:

    def test_aadhaar_masked(self):
        text = "My Aadhaar is 1234 5678 9012"
        masked, warnings = mask_pii(text)
        assert "1234 5678 9012" not in masked
        assert "[AADHAAR MASKED]" in masked
        assert len(warnings) > 0

    def test_pan_masked(self):
        text = "PAN card: ABCDE1234F"
        masked, warnings = mask_pii(text)
        assert "ABCDE1234F" not in masked
        assert "[PAN MASKED]" in masked
        assert len(warnings) > 0

    def test_card_number_masked(self):
        text = "My card is 4111 1111 1111 1111"
        masked, warnings = mask_pii(text)
        assert "4111 1111 1111 1111" not in masked
        assert "[CARD MASKED]" in masked

    def test_16_digit_card_masked(self):
        text = "Card number 4532015112830366"
        masked, warnings = mask_pii(text)
        assert "4532015112830366" not in masked

    def test_clean_text_unchanged(self):
        text = "How does UPI work?"
        masked, warnings = mask_pii(text)
        assert masked == text
        assert warnings == []

    def test_multiple_pii_in_one_message(self):
        text = "My Aadhaar 1234-5678-9012 and PAN ABCDE1234F"
        masked, warnings = mask_pii(text)
        assert "1234-5678-9012" not in masked
        assert "ABCDE1234F" not in masked
        assert len(warnings) > 0


class TestCheckInput:

    def test_investment_advice_refused(self):
        result = check_input("Which stock should I buy today?")
        assert result is not None
        assert "SEBI" in result

    def test_crypto_refused(self):
        result = check_input("Should I invest in bitcoin?")
        assert result is not None

    def test_normal_query_allowed(self):
        result = check_input("How does UPI work?")
        assert result is None

    def test_emi_query_allowed(self):
        result = check_input("Calculate EMI for a 5 lakh loan")
        assert result is None

    def test_legal_advice_refused(self):
        result = check_input("Can I sue my bank for this fraud?")
        assert result is not None


class TestProcessOutput:

    def test_disclaimer_appended(self):
        out = process_output("Some answer", query="test")
        assert "Educational information only" in out

    def test_sources_included(self):
        sources = [{"source_name": "Test Source", "source_url": "https://example.com", "score": 0.9}]
        out = process_output("Answer", query="test", sources=sources)
        assert "Test Source" in out

    def test_fraud_helpline_for_fraud_query(self):
        out = process_output("Here is what to do", query="I was defrauded", sources=[])
        assert "1930" in out
        assert "cybercrime.gov.in" in out

    def test_no_helpline_for_normal_query(self):
        out = process_output("UPI is a payment system", query="What is UPI", sources=[])
        assert "1930" not in out
