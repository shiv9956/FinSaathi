"""
tests/test_tools.py — Unit tests for pure-Python tools.
Run with: pytest tests/ -v (with MOCK_MODE=true)
"""

import os
os.environ.setdefault("MOCK_MODE", "true")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.tools import emi_calculator, budget_planner, scam_checker


# ---------------------------------------------------------------------------
# EMI Calculator tests
# ---------------------------------------------------------------------------

class TestEmiCalculator:

    def test_standard_loan(self):
        """₹1L at 12% for 12 months — known result ≈ 8884.88"""
        result = emi_calculator(100000, 12.0, 12)
        assert result["emi"] == pytest.approx(8884.88, abs=1.0)
        assert result["total_payment"] == pytest.approx(result["emi"] * 12, abs=1.0)
        assert result["total_interest"] == pytest.approx(
            result["total_payment"] - 100000, abs=1.0
        )

    def test_zero_interest(self):
        """Zero interest — EMI should equal principal / tenure."""
        result = emi_calculator(60000, 0.0, 12)
        assert result["emi"] == pytest.approx(5000.0, abs=0.01)
        assert result["total_interest"] == 0.0
        assert result["total_payment"] == pytest.approx(60000.0, abs=0.01)

    def test_single_month(self):
        """1-month loan — EMI should equal principal + one month's interest."""
        result = emi_calculator(10000, 12.0, 1)
        expected_emi = 10000 * (0.12 / 12) * (1 + 0.12 / 12) / ((1 + 0.12 / 12) - 1)
        assert result["emi"] == pytest.approx(expected_emi, abs=0.01)

    def test_total_payment_consistency(self):
        """total_payment == emi * tenure (within rounding)."""
        r = emi_calculator(500000, 10.5, 60)
        assert abs(r["total_payment"] - r["emi"] * 60) < 1.0

    def test_interest_is_positive(self):
        """Non-zero rate should produce positive interest."""
        r = emi_calculator(100000, 8.0, 24)
        assert r["total_interest"] > 0

    def test_negative_principal_raises(self):
        with pytest.raises(ValueError):
            emi_calculator(-1000, 10.0, 12)

    def test_zero_principal_raises(self):
        with pytest.raises(ValueError):
            emi_calculator(0, 10.0, 12)

    def test_zero_tenure_raises(self):
        with pytest.raises(ValueError):
            emi_calculator(100000, 10.0, 0)

    def test_negative_rate_raises(self):
        with pytest.raises(ValueError):
            emi_calculator(100000, -5.0, 12)


# ---------------------------------------------------------------------------
# Budget Planner tests
# ---------------------------------------------------------------------------

class TestBudgetPlanner:

    def test_sum_equals_income(self):
        """needs + wants + savings should equal monthly_income (within rounding)."""
        result = budget_planner(30000)
        total = result["needs"] + result["wants"] + result["savings"]
        assert abs(total - 30000) < 0.05

    def test_50_30_20_split(self):
        result = budget_planner(50000)
        assert result["needs"] == pytest.approx(25000.0, abs=0.01)
        assert result["wants"] == pytest.approx(15000.0, abs=0.01)
        assert result["savings"] == pytest.approx(10000.0, abs=0.01)

    def test_small_income(self):
        result = budget_planner(1000)
        total = result["needs"] + result["wants"] + result["savings"]
        assert abs(total - 1000) < 0.05

    def test_fractional_income(self):
        result = budget_planner(12345.67)
        total = result["needs"] + result["wants"] + result["savings"]
        assert abs(total - 12345.67) < 0.10

    def test_zero_income_raises(self):
        with pytest.raises(ValueError):
            budget_planner(0)

    def test_negative_income_raises(self):
        with pytest.raises(ValueError):
            budget_planner(-5000)


# ---------------------------------------------------------------------------
# Scam Checker tests
# ---------------------------------------------------------------------------

class TestScamChecker:

    def test_obvious_fake_kyc_is_high(self):
        msg = (
            "URGENT: Your KYC is expired. Click bit.ly/fakelink to update immediately "
            "or your account will be blocked. Share OTP to verify."
        )
        result = scam_checker(msg)
        assert result["level"] == "High", f"Expected High, got {result['level']} (score {result['score']})"
        assert result["score"] >= 50
        assert len(result["reasons"]) > 0

    def test_normal_message_is_low(self):
        msg = "Hi, I wanted to know the balance on my savings account."
        result = scam_checker(msg)
        assert result["level"] == "Low"
        assert result["score"] < 20

    def test_otp_request_detected(self):
        msg = "Please share your OTP to complete the transaction."
        result = scam_checker(msg)
        assert any("OTP" in r for r in result["reasons"])

    def test_qr_scan_trick_detected(self):
        msg = "Scan this QR code to receive your refund money."
        result = scam_checker(msg)
        assert any("QR" in r or "collect" in r.lower() for r in result["reasons"])

    def test_empty_message(self):
        result = scam_checker("")
        assert result["score"] == 0
        assert result["level"] == "Low"
        assert result["reasons"] == []

    def test_score_capped_at_100(self):
        # A message triggering many indicators
        msg = (
            "URGENT! Your KYC expired! Share OTP now! Enter UPI PIN! "
            "Scan QR to receive prize! Call back 9876543210! "
            "Click bit.ly/scam to claim lottery winner refund cashback."
        )
        result = scam_checker(msg)
        assert result["score"] <= 100
        assert result["level"] == "High"

    def test_medium_risk(self):
        msg = "You have won a cashback reward. Claim now before it expires."
        result = scam_checker(msg)
        assert result["level"] in ("Medium", "High")
