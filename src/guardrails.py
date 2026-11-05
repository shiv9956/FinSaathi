"""
src/guardrails.py

Input PII masking and output safety for FinSaathi.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------

# Aadhaar: exactly 12 digits in 4-4-4 groups separated by space or hyphen
_AADHAAR_RE = re.compile(r"(?<!\d)(\d{4}[\s\-]\d{4}[\s\-]\d{4})(?!\d)")
_PAN_RE = re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b")
# Card: 13-19 consecutive digits (possibly separated by spaces/hyphens), NOT preceded/followed by more digits
_CARD_RE = re.compile(r"(?<!\d)((?:\d[\s\-]?){13,19}\d)(?!\d)")
# OTP / PIN: 4-8 digit standalone number preceded by OTP/PIN/CVV keyword
_OTP_RE = re.compile(r"(?i)\b(otp|pin|cvv|password)\b[\s:is]*\b(\d{4,8})\b")

_PII_WARNING = (
    "⚠️ **Privacy Notice:** Never share your OTP, PIN, CVV or card number with anyone, "
    "including this assistant. Your message has been sanitised before processing."
)


def mask_pii(text: str) -> tuple[str, list[str]]:
    """
    Detect and mask PII in `text`.

    Returns (masked_text, list_of_warnings).
    Order: card numbers first (longest match), then Aadhaar (subset digits),
    then PAN, then OTP.
    """
    warnings: list[str] = []
    original = text

    # Card numbers first (they can overlap with Aadhaar digit groups)
    def _mask_card(m: re.Match) -> str:
        digits = re.sub(r"[\s\-]", "", m.group(0))
        if 13 <= len(digits) <= 20:
            return "[CARD MASKED]"
        return m.group(0)

    text = _CARD_RE.sub(_mask_card, text)
    text = _AADHAAR_RE.sub("[AADHAAR MASKED]", text)
    text = _PAN_RE.sub("[PAN MASKED]", text)
    text = _OTP_RE.sub(r"\1 [MASKED]", text)

    if text != original:
        warnings.append(_PII_WARNING)

    return text, warnings


# ---------------------------------------------------------------------------
# Input refusal checks
# ---------------------------------------------------------------------------

_INVESTMENT_PATTERNS = re.compile(
    r"(?i)\b(which\s+stock|buy\s+stock|sell\s+stock|invest\s+in|crypto|bitcoin|"
    r"mutual\s+fund\s+to\s+buy|share\s+market\s+tip|trading\s+advice|"
    r"best\s+stock\s+to|should\s+i\s+invest)\b"
)
_LEGAL_PATTERNS = re.compile(
    r"(?i)\b(legal\s+advice|sue|lawsuit|court\s+case|file\s+a\s+case\s+against|"
    r"is\s+it\s+legal|legal\s+action\s+against)\b"
)

_INVESTMENT_REFUSAL = (
    "I can share general educational information about saving and investing, "
    "but I cannot recommend specific stocks, mutual funds, cryptocurrencies, or trading strategies. "
    "For personalised investment advice, please consult a SEBI-registered investment advisor. "
    "You can find one at https://www.sebi.gov.in."
)

_LEGAL_REFUSAL = (
    "I can share general financial literacy information, but I cannot provide legal advice. "
    "For legal matters related to financial fraud, contact a qualified legal professional "
    "or reach out to your nearest consumer court."
)


def check_input(text: str) -> Optional[str]:
    """
    Returns a refusal message if the query should not be answered by the model,
    else None.
    """
    if _INVESTMENT_PATTERNS.search(text):
        return _INVESTMENT_REFUSAL
    if _LEGAL_PATTERNS.search(text):
        return _LEGAL_REFUSAL
    return None


# ---------------------------------------------------------------------------
# Output post-processing
# ---------------------------------------------------------------------------

_DISCLAIMER = (
    "\n\n---\n*📚 Educational information only — not financial advice. "
    "Always verify with official sources.*"
)

_FRAUD_HELPLINE = (
    "\n\n🆘 **If you have been defrauded:** Call **1930** (National Cybercrime Helpline) "
    "immediately and report at **https://cybercrime.gov.in**."
)

_FRAUD_KEYWORDS = re.compile(
    r"(?i)\b(scam|fraud|cheated|defrauded|stolen|hacked|fake\s+call|vishing|"
    r"phishing|otp\s+stole|lost\s+money|money\s+gone|account\s+hacked)\b"
)


def process_output(answer: str, query: str = "", sources: Optional[list[dict]] = None) -> str:
    """
    Append disclaimer, source citations, and fraud helpline if relevant.
    """
    result = answer.rstrip()

    # Append sources
    if sources:
        seen_urls: set[str] = set()
        source_lines: list[str] = []
        for s in sources:
            url = s.get("source_url", "")
            name = s.get("source_name", s.get("file", ""))
            if url and url not in seen_urls:
                source_lines.append(f"- [{name}]({url})" if url else f"- {name}")
                seen_urls.add(url)
            elif not url and name and name not in seen_urls:
                source_lines.append(f"- {name}")
                seen_urls.add(name)
        if source_lines:
            result += "\n\n**Sources:**\n" + "\n".join(source_lines)

    # Fraud helpline for relevant queries
    if _FRAUD_KEYWORDS.search(query) or _FRAUD_KEYWORDS.search(answer):
        result += _FRAUD_HELPLINE

    result += _DISCLAIMER
    return result
