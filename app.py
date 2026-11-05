"""
app.py — FinSaathi Streamlit UI
"""

import sys
import os
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

import config

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FinSaathi — Financial Literacy AI",
    page_icon="💰",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Cached resource: retriever index
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge index…")
def get_retriever():
    from src import retriever
    retriever._load_index()
    return retriever


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def _mock_banner():
    if config.MOCK_MODE:
        st.warning(
            "⚠️ **MOCK MODE ACTIVE** — Running without IBM watsonx credentials. "
            "Answers are simulated. Set your credentials in `.env` and restart for real responses.",
            icon="🤖",
        )


def _sidebar():
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=80)
        st.title("FinSaathi")
        st.caption("AI Agent for Digital Financial Literacy")
        st.divider()

        language = st.selectbox("🌐 Language", ["English", "Hindi"], key="language")

        st.divider()

        if st.button("🔄 Rebuild Knowledge Index", use_container_width=True):
            with st.spinner("Rebuilding index…"):
                try:
                    from src.ingest import run_ingestion
                    from src import retriever
                    run_ingestion()
                    retriever.reload_index()
                    st.success("Index rebuilt successfully!")
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")

        st.divider()
        st.markdown(
            "🔒 **Privacy:** Your messages are processed by IBM watsonx.ai. "
            "Never enter real Aadhaar, PAN, card numbers, or OTPs."
        )

        if config.MOCK_MODE:
            st.error("🤖 MOCK MODE")

    return language


# ---------------------------------------------------------------------------
# Tab 1: Chat
# ---------------------------------------------------------------------------
def _render_chat(language: str):
    st.subheader("💬 Ask FinSaathi")
    st.caption("Ask anything about UPI, loans, budgeting, scams, or digital banking.")

    # Example question buttons
    col1, col2, col3 = st.columns(3)
    example_q = None
    with col1:
        if st.button("How do I send money via UPI?", use_container_width=True):
            example_q = "How do I send money via UPI?"
    with col2:
        if st.button("What is a safe interest rate for a loan?", use_container_width=True):
            example_q = "What is a safe interest rate for a loan?"
    with col3:
        if st.button("How do I avoid UPI scams?", use_container_width=True):
            example_q = "How do I avoid UPI scams?"

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for s in msg["sources"]:
                        url = s.get("source_url", "")
                        name = s.get("source_name", s.get("file", ""))
                        score = s.get("score", "")
                        if url:
                            st.markdown(f"- [{name}]({url}) (score: {score:.3f})" if score else f"- [{name}]({url})")
                        else:
                            st.markdown(f"- {name}")

    # Input
    user_input = st.chat_input("Type your question…") or example_q
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                from src.agent import answer
                result = answer(
                    user_input,
                    language=language,
                    history=st.session_state.chat_history[:-1],
                )
            st.markdown(result["answer"])
            if result.get("sources"):
                with st.expander("📚 Sources"):
                    for s in result["sources"]:
                        url = s.get("source_url", "")
                        name = s.get("source_name", s.get("file", ""))
                        score = s.get("score", "")
                        if url:
                            st.markdown(f"- [{name}]({url}) (score: {score:.3f})" if score else f"- [{name}]({url})")
                        else:
                            st.markdown(f"- {name}")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
        })


# ---------------------------------------------------------------------------
# Tab 2: Scam Checker
# ---------------------------------------------------------------------------
def _render_scam_checker(language: str):
    st.subheader("🔍 Scam Checker")
    st.caption(
        "Paste a suspicious SMS, WhatsApp message, email or phone call description below. "
        "FinSaathi will analyse it for scam indicators."
    )

    sms_text = st.text_area(
        "Paste message here",
        height=150,
        placeholder='E.g. "Your KYC is expired. Click here to update: bit.ly/abc123 or your account will be blocked."',
    )

    if st.button("🛡️ Check for Scam", use_container_width=True, type="primary"):
        if not sms_text.strip():
            st.warning("Please paste a message to check.")
        else:
            with st.spinner("Analysing…"):
                from src.agent import answer
                result = answer(
                    f"Check if this is a scam: {sms_text}",
                    language=language,
                )

            risk = result.get("risk", {})
            level = risk.get("level", "")
            score = risk.get("score", 0)
            reasons = risk.get("reasons", [])

            color_map = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            icon = color_map.get(level, "ℹ️")

            st.markdown(f"### {icon} Risk Level: **{level}** (Score: {score}/100)")

            if reasons:
                st.markdown("**Red flags found:**")
                for r in reasons:
                    st.markdown(f"- {r}")

            st.divider()
            st.markdown(result["answer"])


# ---------------------------------------------------------------------------
# Tab 3: Calculators
# ---------------------------------------------------------------------------
def _render_calculators(language: str):
    st.subheader("🧮 Financial Calculators")

    calc_tab1, calc_tab2 = st.tabs(["📊 EMI Calculator", "📋 Budget Planner"])

    with calc_tab1:
        st.markdown("Calculate your monthly loan EMI using the reducing-balance formula.")
        c1, c2, c3 = st.columns(3)
        with c1:
            principal = st.number_input("Loan Amount (₹)", min_value=1000.0, value=100000.0, step=1000.0)
        with c2:
            rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, max_value=50.0, value=10.5, step=0.1)
        with c3:
            tenure = st.number_input("Tenure (months)", min_value=1, max_value=360, value=24, step=1)

        if st.button("Calculate EMI", use_container_width=True, type="primary"):
            with st.spinner("Calculating…"):
                from src.agent import answer
                result = answer(
                    f"Calculate EMI for loan {principal} at {rate}% for {tenure} months",
                    language=language,
                )
            st.markdown(result["answer"])

    with calc_tab2:
        st.markdown("Apply the **50/30/20 rule** to plan your monthly budget.")
        income = st.number_input("Monthly Income (₹)", min_value=1000.0, value=30000.0, step=1000.0)

        if st.button("Plan My Budget", use_container_width=True, type="primary"):
            with st.spinner("Planning…"):
                from src.agent import answer
                result = answer(
                    f"Help me budget my monthly income of {income}",
                    language=language,
                )
            st.markdown(result["answer"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("💰 FinSaathi")
    st.markdown("**Your AI Friend for Digital Financial Literacy** — Powered by IBM Granite on watsonx.ai")
    st.divider()

    _mock_banner()
    language = _sidebar()

    # Pre-load retriever
    try:
        get_retriever()
    except Exception as e:
        st.warning(f"Knowledge index not ready: {e}. Run `python -m src.ingest` first.")

    tab1, tab2, tab3 = st.tabs(["💬 Ask FinSaathi", "🔍 Scam Checker", "🧮 Calculators"])

    with tab1:
        _render_chat(language)
    with tab2:
        _render_scam_checker(language)
    with tab3:
        _render_calculators(language)


if __name__ == "__main__":
    main()
