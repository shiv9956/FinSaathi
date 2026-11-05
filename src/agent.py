"""
src/agent.py

FinSaathi agent — rule-based router + RAG + tools.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src import guardrails, tools
from src.tools import emi_calculator, budget_planner, scam_checker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for RAG answers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_EN = """You are FinSaathi, a friendly financial literacy assistant for first-time digital users in India.

Rules:
- Answer ONLY using the information provided in the <context> block below.
- Keep your answer under 150 words.
- Use short, numbered steps when explaining a process.
- Use simple language. Avoid jargon.
- Never ask the user for personal information (Aadhaar, bank account, card number, OTP, PIN).
- Never recommend specific stocks, mutual funds, or crypto investments.
- If the context does not contain enough information, say: "I don't have a reliable source for that. Please check the official websites listed below."
- Always be encouraging and non-judgmental.
"""

_SYSTEM_PROMPT_HI = """आप FinSaathi हैं — भारत के पहली बार डिजिटल सेवाएँ इस्तेमाल करने वाले उपयोगकर्ताओं के लिए एक मित्रवत वित्तीय साक्षरता सहायक।

नियम:
- नीचे दिए गए <context> ब्लॉक में मौजूद जानकारी से ही उत्तर दें।
- उत्तर 150 शब्दों से कम रखें।
- किसी प्रक्रिया को समझाने के लिए छोटे, क्रमांकित चरणों का उपयोग करें।
- सरल हिंदी में बोलें। UPI, OTP, EMI जैसे तकनीकी शब्द अंग्रेज़ी में रखें।
- उपयोगकर्ता से कोई व्यक्तिगत जानकारी (आधार, बैंक खाता, कार्ड नंबर, OTP, PIN) न माँगें।
- किसी विशेष शेयर, म्यूचुअल फंड या क्रिप्टो में निवेश की सलाह न दें।
- यदि context में पर्याप्त जानकारी नहीं है, तो कहें: "मेरे पास इसके लिए कोई विश्वसनीय स्रोत नहीं है। कृपया नीचे दी गई आधिकारिक वेबसाइट देखें।"
"""

# ---------------------------------------------------------------------------
# Intent detection helpers
# ---------------------------------------------------------------------------

_SCAM_INTENT_RE = re.compile(
    r"(?i)\b(scam|fraud|fake|phishing|is\s+this\s+real|is\s+this\s+safe|"
    r"check\s+(this\s+)?(message|sms|link|number)|legitimate|suspicious|"
    r"otp\s+maang|paisa\s+gaya|ठगी|धोखा|फ्रॉड)\b"
)

_NUMBER_RE = re.compile(r"\b(\d[\d,\.]*)\b")


def _extract_numbers(text: str) -> list[float]:
    nums = []
    for m in _NUMBER_RE.finditer(text):
        try:
            n = float(m.group(1).replace(",", ""))
            nums.append(n)
        except ValueError:
            pass
    return nums


_EMI_INTENT_RE = re.compile(
    r"(?i)\b(emi|loan|equated|monthly\s+install|calculate|borrow|interest\s+on)\b"
)

_BUDGET_INTENT_RE = re.compile(
    r"(?i)\b(budget|salary|income|50.30.20|fifty|thirty|twenty|plan\s+my|"
    r"monthly\s+plan|kitna\s+kharch|खर्च|बजट|तनख्वाह)\b"
)


# ---------------------------------------------------------------------------
# RAG answer builder
# ---------------------------------------------------------------------------

def _rag_answer(query: str, language: str, history: list) -> dict:
    from src import retriever, watsonx_client

    results = retriever.search(query)

    if not results:
        no_source_msg = (
            "I don't have a reliable source for that in my knowledge base. "
            "Please check official sites:\n"
            "- https://www.npci.org.in\n"
            "- https://www.rbi.org.in\n"
            "- https://cybercrime.gov.in"
        )
        return {"answer": no_source_msg, "sources": [], "tool_used": "rag"}

    context_parts = []
    for i, chunk in enumerate(results, 1):
        context_parts.append(f"[{i}] {chunk['text']}")
    context_block = "\n\n".join(context_parts)

    system_prompt = _SYSTEM_PROMPT_HI if language == "Hindi" else _SYSTEM_PROMPT_EN
    user_prompt = f"<context>\n{context_block}\n</context>\n\nQuestion: {query}"

    try:
        answer = watsonx_client.generate(system_prompt, user_prompt)
    except Exception as exc:
        logger.warning("Generation failed: %s — returning context summary", exc)
        # Fallback: summarise context without LLM
        answer = (
            "Here is what I found from my knowledge base:\n\n"
            + "\n\n".join(f"• {chunk['text'][:300]}…" for chunk in results[:2])
        )

    return {"answer": answer, "sources": results, "tool_used": "rag"}


# ---------------------------------------------------------------------------
# Main agent entry point
# ---------------------------------------------------------------------------

def answer(
    user_message: str,
    language: str = "English",
    history: Optional[list] = None,
) -> dict:
    """
    Route the user message and return a response dict:
        {answer, sources, tool_used, risk (optional)}

    Never raises — always returns a helpful message.
    """
    if history is None:
        history = []

    try:
        # --- Step 1: PII masking ---
        clean_msg, pii_warnings = guardrails.mask_pii(user_message)

        # --- Step 2: Refusal check ---
        refusal = guardrails.check_input(clean_msg)
        if refusal:
            out = guardrails.process_output(refusal, query=clean_msg, sources=[])
            return {"answer": out, "sources": [], "tool_used": "refusal"}

        # --- Step 3: Route ---
        if _SCAM_INTENT_RE.search(clean_msg):
            return _handle_scam(clean_msg, language, pii_warnings)

        if _EMI_INTENT_RE.search(clean_msg):
            nums = _extract_numbers(clean_msg)
            if len(nums) >= 3:
                return _handle_emi(nums, clean_msg, language, pii_warnings)
            # Not enough numbers — ask the user
            return {
                "answer": (
                    "To calculate your EMI, I need three things:\n"
                    "1. **Loan amount** (principal in ₹)\n"
                    "2. **Annual interest rate** (e.g., 10.5 for 10.5%)\n"
                    "3. **Loan tenure** (in months, e.g., 24 for 2 years)\n\n"
                    "Please provide these three numbers and I'll calculate right away!"
                ),
                "sources": [],
                "tool_used": "emi_prompt",
            }

        if _BUDGET_INTENT_RE.search(clean_msg):
            nums = _extract_numbers(clean_msg)
            if nums:
                return _handle_budget(nums[0], language, pii_warnings)

        # --- Default: RAG ---
        result = _rag_answer(clean_msg, language, history)

        # Post-process output
        result["answer"] = guardrails.process_output(
            result["answer"],
            query=clean_msg,
            sources=result.get("sources", []),
        )

        if pii_warnings:
            result["answer"] = "\n\n".join(pii_warnings) + "\n\n" + result["answer"]

        return result

    except Exception as exc:
        logger.exception("Unhandled error in agent.answer: %s", exc)
        return {
            "answer": (
                "😕 Something went wrong while processing your question. "
                "Please try again. If the problem persists, check that your IBM credentials are correct "
                "or set MOCK_MODE=true in your .env file."
            ),
            "sources": [],
            "tool_used": "error",
        }


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _handle_scam(text: str, language: str, pii_warnings: list) -> dict:
    from src import retriever, watsonx_client

    result = scam_checker(text)
    score = result["score"]
    level = result["level"]
    reasons = result["reasons"]

    # Build a human-readable explanation
    if level == "High":
        base_msg = (
            f"🚨 **High Risk (Score: {score}/100)** — This looks like a scam!\n\n"
            "**Red flags detected:**\n" + "\n".join(f"• {r}" for r in reasons)
        )
    elif level == "Medium":
        base_msg = (
            f"⚠️ **Medium Risk (Score: {score}/100)** — Be careful, this could be a scam.\n\n"
            "**Suspicious signs:**\n" + "\n".join(f"• {r}" for r in reasons)
        )
    else:
        base_msg = (
            f"✅ **Low Risk (Score: {score}/100)** — This message looks relatively safe.\n\n"
            + ("No obvious scam indicators found." if not reasons else
               "**Minor concerns:**\n" + "\n".join(f"• {r}" for r in reasons))
        )

    # Try to get LLM explanation with retrieved context
    try:
        rag_results = retriever.search(text + " scam fraud safety")
        if rag_results:
            context = "\n\n".join(c["text"] for c in rag_results[:3])
            sys_prompt = (
                "You are FinSaathi. Based on the scam risk analysis and context below, "
                "explain in 2-3 simple sentences what this risk level means and what the user should do next. "
                "Be direct and clear. Do not repeat the risk score — that is already shown."
            )
            if language == "Hindi":
                sys_prompt += " Respond in simple Hindi."
            user_prompt = (
                f"Risk level: {level} (score {score}/100)\n"
                f"Red flags: {', '.join(reasons) if reasons else 'None'}\n\n"
                f"<context>\n{context}\n</context>\n\n"
                f"Original message snippet: {text[:300]}"
            )
            llm_advice = watsonx_client.generate(sys_prompt, user_prompt)
            advice_text = f"\n\n**What you should do:**\n{llm_advice}"
        else:
            advice_text = _default_scam_advice(level)
    except Exception as exc:
        logger.warning("LLM scam explanation failed: %s", exc)
        advice_text = _default_scam_advice(level)

    full_answer = base_msg + advice_text
    full_answer = guardrails.process_output(full_answer, query=text, sources=[])

    if pii_warnings:
        full_answer = "\n\n".join(pii_warnings) + "\n\n" + full_answer

    return {
        "answer": full_answer,
        "sources": [],
        "tool_used": "scam_checker",
        "risk": {"score": score, "level": level, "reasons": reasons},
    }


def _default_scam_advice(level: str) -> str:
    if level == "High":
        return (
            "\n\n**Immediately:**\n"
            "1. Do NOT share any OTP, PIN, or personal details.\n"
            "2. Block the sender or caller.\n"
            "3. Call **1930** if you have already shared any data or lost money.\n"
            "4. Report at **https://cybercrime.gov.in**."
        )
    elif level == "Medium":
        return (
            "\n\n**Be cautious:**\n"
            "1. Do not click any links.\n"
            "2. Do not share OTP or PIN.\n"
            "3. Verify by calling the official bank helpline (find it on the bank's official website)."
        )
    else:
        return (
            "\n\n**Stay safe:** Never share OTP, PIN, or card details with anyone, "
            "even if they claim to be from your bank or a government body."
        )


def _handle_emi(nums: list, original_text: str, language: str, pii_warnings: list) -> dict:
    from src import watsonx_client

    principal, rate, tenure = nums[0], nums[1], nums[2]

    try:
        result = emi_calculator(principal, rate, int(tenure))
    except ValueError as exc:
        return {
            "answer": f"❌ {exc}",
            "sources": [],
            "tool_used": "emi_calculator",
        }

    emi_msg = (
        f"**EMI Calculation Result:**\n\n"
        f"| | |\n|---|---|\n"
        f"| Loan Amount | ₹{principal:,.0f} |\n"
        f"| Annual Rate | {rate}% |\n"
        f"| Tenure | {int(tenure)} months |\n"
        f"| **Monthly EMI** | **₹{result['emi']:,.2f}** |\n"
        f"| Total Interest | ₹{result['total_interest']:,.2f} |\n"
        f"| Total Payment | ₹{result['total_payment']:,.2f} |\n"
    )

    # Optional LLM explanation
    try:
        sys_prompt = (
            "You are FinSaathi. Explain this EMI result in 2-3 simple sentences "
            "for a first-time loan taker. Mention what EMI means and a tip about comparing rates."
        )
        if language == "Hindi":
            sys_prompt += " Respond in simple Hindi."
        explanation = watsonx_client.generate(
            sys_prompt,
            f"Principal ₹{principal}, Rate {rate}%, Tenure {int(tenure)} months, EMI ₹{result['emi']}",
        )
        emi_msg += f"\n\n**What this means:** {explanation}"
    except Exception:
        pass

    emi_msg = guardrails.process_output(emi_msg, query=original_text, sources=[])
    if pii_warnings:
        emi_msg = "\n\n".join(pii_warnings) + "\n\n" + emi_msg

    return {
        "answer": emi_msg,
        "sources": [],
        "tool_used": "emi_calculator",
    }


def _handle_budget(income: float, language: str, pii_warnings: list) -> dict:
    from src import watsonx_client

    try:
        result = budget_planner(income)
    except ValueError as exc:
        return {"answer": f"❌ {exc}", "sources": [], "tool_used": "budget_planner"}

    budget_msg = (
        f"**Budget Plan (50/30/20 Rule) for ₹{income:,.0f}/month:**\n\n"
        f"| Category | Amount | Purpose |\n|---|---|---|\n"
        f"| 🏠 Needs (50%) | ₹{result['needs']:,.2f} | Rent, food, utilities, transport |\n"
        f"| 🎉 Wants (30%) | ₹{result['wants']:,.2f} | Entertainment, shopping, dining |\n"
        f"| 💰 Savings (20%) | ₹{result['savings']:,.2f} | Emergency fund, investments |\n"
    )

    try:
        sys_prompt = (
            "You are FinSaathi. Add 2-3 simple tips for someone just starting to budget "
            "on this income. Keep it practical and encouraging."
        )
        if language == "Hindi":
            sys_prompt += " Respond in simple Hindi."
        tips = watsonx_client.generate(sys_prompt, f"Monthly income: ₹{income}")
        budget_msg += f"\n\n**Quick Tips:** {tips}"
    except Exception:
        pass

    budget_msg = guardrails.process_output(budget_msg, query="budget", sources=[])
    if pii_warnings:
        budget_msg = "\n\n".join(pii_warnings) + "\n\n" + budget_msg

    return {
        "answer": budget_msg,
        "sources": [],
        "tool_used": "budget_planner",
    }
