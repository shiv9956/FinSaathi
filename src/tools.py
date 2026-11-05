"""
src/tools.py

Pure Python financial tools — no LLM needed.
Fully unit-tested and deterministic.
"""

from __future__ import annotations

import math
import re
from typing import Optional


# ---------------------------------------------------------------------------
# 1. EMI Calculator (reducing balance)
# ---------------------------------------------------------------------------

def emi_calculator(
    principal: float,
    annual_rate_percent: float,
    tenure_months: int,
) -> dict:
    """
    Calculate EMI using the standard reducing-balance (flat rate not used) formula.

    Formula: EMI = P * r * (1+r)^n / ((1+r)^n - 1)
    where r = monthly interest rate, n = tenure in months.

    Special case: rate = 0 -> EMI = principal / tenure_months

    Returns:
        {emi, total_payment, total_interest}

    Raises:
        ValueError: on invalid inputs with a friendly message.
    """
    if principal <= 0:
        raise ValueError("Principal must be a positive number (e.g., 100000 for ₹1 lakh).")
    if annual_rate_percent < 0:
        raise ValueError("Annual interest rate cannot be negative.")
    if tenure_months <= 0 or not isinstance(tenure_months, (int, float)) or int(tenure_months) < 1:
        raise ValueError("Tenure must be at least 1 month.")

    tenure_months = int(tenure_months)

    if annual_rate_percent == 0:
        emi = principal / tenure_months
        return {
            "emi": round(emi, 2),
            "total_payment": round(principal, 2),
            "total_interest": 0.0,
        }

    r = annual_rate_percent / 100 / 12  # monthly rate
    power = (1 + r) ** tenure_months
    emi = principal * r * power / (power - 1)
    total_payment = emi * tenure_months
    total_interest = total_payment - principal

    return {
        "emi": round(emi, 2),
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_interest, 2),
    }


# ---------------------------------------------------------------------------
# 2. Budget Planner (50/30/20 rule)
# ---------------------------------------------------------------------------

def budget_planner(monthly_income: float) -> dict:
    """
    Apply the 50/30/20 budgeting rule.

    Returns:
        {needs, wants, savings, monthly_income}

    Raises:
        ValueError: if income is not positive.
    """
    if monthly_income <= 0:
        raise ValueError("Monthly income must be a positive number.")

    needs = round(monthly_income * 0.50, 2)
    wants = round(monthly_income * 0.30, 2)
    savings = round(monthly_income * 0.20, 2)

    return {
        "monthly_income": round(monthly_income, 2),
        "needs": needs,
        "wants": wants,
        "savings": savings,
    }


# ---------------------------------------------------------------------------
# 3. Scam Checker
# ---------------------------------------------------------------------------

# Each entry: (regex_pattern, weight, reason_text)
_SCAM_INDICATORS: list[tuple[re.Pattern, int, str]] = [
    # Urgency words
    (re.compile(r"(?i)\b(urgent|immediately|right\s+now|last\s+chance|expire[sd]?|hurry|act\s+now|do\s+not\s+delay)\b"), 10, "Uses urgency language"),
    # Requests for sensitive data
    (re.compile(r"(?i)\b(share\s+(your\s+)?otp|send\s+otp|enter\s+otp|otp\s+is|otp\s+pe|give\s+(your\s+)?otp|otp\s+to\s+verify|otp\s+share)\b"), 25, "Requests OTP"),
    (re.compile(r"(?i)\b(share\s+your\s+pin|your\s+upi\s+pin|enter\s+upi\s+pin|pin\s+enter|atm\s+pin)\b"), 25, "Requests UPI/ATM PIN"),
    (re.compile(r"(?i)\b(cvv|card\s+number|card\s+details|card\s+no)\b"), 20, "Requests card details"),
    # KYC / account threats
    (re.compile(r"(?i)\b(kyc.{0,10}(expired|update|pending|verify)|account.{0,10}(block|suspend|deactivat)|verify.{0,10}kyc|your\s+kyc)\b"), 20, "Fake KYC or account-block threat"),
    # Prize / lottery / job fee
    (re.compile(r"(?i)\b(won|winner|lottery|prize|reward|cash\s+prize|job\s+offer|work\s+from\s+home|advance\s+fee|registration\s+fee)\b"), 15, "Prize, lottery, or job-fee claim"),
    # Refund / cashback trick
    (re.compile(r"(?i)\b(refund|cashback|money\s+back|claim\s+your)\b"), 10, "Refund or cashback claim"),
    # QR / collect request trick
    (re.compile(r"(?i)\b(scan\s+(this\s+)?qr|approve\s+(this\s+)?collect|accept\s+(the\s+)?request|receive\s+money\s+by\s+scanning)\b"), 25, "QR-code or collect-request trick (scanning to receive money is a scam)"),
    # Suspicious links
    (re.compile(r"(?i)(bit\.ly|tinyurl|t\.co|short\.url|rb\.gy|ow\.ly|is\.gd|cutt\.ly|clck\.ru)"), 20, "Contains shortened/suspicious URL"),
    # Unofficial domains claiming to be banks/NPCI
    (re.compile(r"(?i)(sbi[^\.]*\.(?!co\.in)\w{2,}|npci[^\.]*\.(?!org\.in)\w{2,}|rbi[^\.]*\.(?!org\.in)\w{2,})"), 15, "Suspicious unofficial bank/NPCI domain"),
    # Asking to call back on unknown numbers
    (re.compile(r"(?i)\b(call\s+(back\s+on\s+)?(\+91\s*)?\d{10}|contact\s+helpline\s+(\+91\s*)?\d{10})\b"), 15, "Asks you to call an unverified phone number"),
    # Vishing / fake customer care
    (re.compile(r"(?i)\b(customer\s*(care|support|service)\s*(\+91\s*)?\d{10}|helpline\s*\d{10})\b"), 15, "Fake customer-care number"),
]


def scam_checker(text: str) -> dict:
    """
    Rule-based scam risk scoring.

    Returns:
        {
            score: int (0-100),
            level: "Low" | "Medium" | "High",
            reasons: list[str],
        }
    """
    if not text or not text.strip():
        return {"score": 0, "level": "Low", "reasons": []}

    raw_score = 0
    reasons: list[str] = []

    for pattern, weight, reason in _SCAM_INDICATORS:
        if pattern.search(text):
            raw_score += weight
            reasons.append(reason)

    # Clamp to 100
    score = min(raw_score, 100)

    if score >= 50:
        level = "High"
    elif score >= 20:
        level = "Medium"
    else:
        level = "Low"

    return {"score": score, "level": level, "reasons": reasons}
