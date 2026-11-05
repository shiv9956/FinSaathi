# FinSaathi — AI Agent for Digital Financial Literacy

**Powered by IBM Granite on watsonx.ai | Built with Streamlit | RAG + Tools**

FinSaathi is a beginner-friendly, responsible AI assistant that helps Indian users understand UPI payments, avoid online scams, plan budgets, and calculate EMIs — grounded in official RBI, NPCI, SEBI, and cybercrime.gov.in resources.

---

## Quick Start (6 commands)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy .env.example and add your IBM credentials
copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux
# Edit .env with your WATSONX_API_KEY and WATSONX_PROJECT_ID

# 4. Verify setup (will run in MOCK_MODE if credentials are missing)
python scripts/check_setup.py

# 5. Build the knowledge index
python -m src.ingest

# 6. Start the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## Getting IBM watsonx Credentials (Free Lite Plan)

1. Sign up at https://cloud.ibm.com (free, no credit card for Lite plan).
2. Create a **watsonx.ai** project.
3. Go to **Manage → Access (IAM)** → Create an **API Key** → copy it into `WATSONX_API_KEY`.
4. Copy your **Project ID** from the project settings → paste into `WATSONX_PROJECT_ID`.
5. Leave `WATSONX_URL=https://us-south.ml.cloud.ibm.com` (or change region if needed).

---

## Configuration (.env)

| Variable | Description | Default |
|---|---|---|
| `WATSONX_API_KEY` | IBM Cloud API Key | *(required for real mode)* |
| `WATSONX_PROJECT_ID` | watsonx.ai Project ID | *(required for real mode)* |
| `WATSONX_URL` | watsonx.ai endpoint | `https://us-south.ml.cloud.ibm.com` |
| `GEN_MODEL_IDS` | Comma-separated fallback model list | `ibm/granite-3-3-8b-instruct,...` |
| `EMBED_MODEL_ID` | Embedding model | `ibm/granite-embedding-278m-multilingual` |
| `MOCK_MODE` | `true` = run without credentials | `false` |

Set `MOCK_MODE=true` to run the full app offline for testing and demos without any IBM credentials.

---

## Project Structure

```
finsaathi/
├── app.py                      # Streamlit UI
├── config.py                   # All settings from .env
├── requirements.txt
├── requirements-optional.txt   # Docling (optional, better PDF parsing)
├── .env.example
├── src/
│   ├── watsonx_client.py       # IBM SDK wrapper (generate + embed + mock)
│   ├── ingest.py               # Document loading → chunking → embedding → index
│   ├── retriever.py            # Cosine similarity search
│   ├── guardrails.py           # PII masking + refusal + disclaimers
│   ├── tools.py                # EMI calculator, budget planner, scam checker
│   └── agent.py               # Router → RAG + tools
├── data/
│   ├── knowledge_base/         # Source .md files (add PDFs here for production)
│   └── index/                  # Generated embeddings.npy + chunks.json
├── scripts/
│   └── check_setup.py          # End-to-end setup verification
├── tests/                      # pytest tests (run with MOCK_MODE=true)
└── docs/
    ├── architecture.md         # Mermaid diagrams + Responsible AI
    └── demo_script.md          # 60-90 second demo walkthrough
```

---

## Running Tests

```bash
# Tests run entirely offline using MOCK_MODE
MOCK_MODE=true pytest tests/ -v          # macOS/Linux
$env:MOCK_MODE="true"; pytest tests/ -v  # Windows PowerShell
```

---

## Optional: Better PDF/DOCX Parsing (Docling)

Install Docling for richer document parsing (requires additional system libraries):

```bash
pip install -r requirements-optional.txt
```

If Docling is not installed or fails, the system automatically falls back to `pypdf`. The app works correctly without Docling.

---

## Knowledge Base

The `data/knowledge_base/` folder contains 10 Markdown files covering:
- How UPI works and safe payment practices
- Common online scams (fake KYC, OTP theft, QR tricks, phishing)
- What to do if defrauded (call 1930, report at cybercrime.gov.in)
- Interest rates, EMI, and credit scores
- Budgeting (50/30/20 rule) and emergency fund
- Savings basics (FD, RD, PPF)
- Digital banking safety
- Hindi finance glossary
- Responsible investing basics
- Understanding loans

**For production use:** Replace or extend these files with official PDFs from RBI, NPCI, SEBI, and cybercrime.gov.in. Drop them in `data/knowledge_base/` and re-run `python -m src.ingest`.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Auth error: 401 Unauthorized` | Check `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` in `.env`. Ensure no leading/trailing spaces. |
| `Model not found` or `Model ID invalid` | The model may have been deprecated. Run `python scripts/check_setup.py` to list available models, then update `GEN_MODEL_IDS` in `.env`. |
| `Quota exceeded` on Lite plan | Wait for quota reset, or reduce `EMBED_BATCH_SIZE` in `config.py`. The embedding cache prevents re-embedding the same chunks. |
| Docling install fails | Skip it — run `pip install -r requirements.txt` only. The app uses `pypdf` as a fallback automatically. |
| `Index not found` error in UI | Run `python -m src.ingest` first to build the index. |
| App starts in MOCK_MODE unexpectedly | Check that `.env` exists and has non-empty `WATSONX_API_KEY` and `WATSONX_PROJECT_ID`. |
| `ibm-watsonx-ai` version conflict | Try `pip install ibm-watsonx-ai --upgrade`. The SDK API changed between versions 0.x and 1.x — version 1.0+ is required. |

---

## Responsible AI

- Answers are grounded in retrieved documents — the model will say it doesn't know rather than guess.
- PII (Aadhaar, PAN, card numbers, OTPs) is masked before reaching the model.
- No investment or legal advice is given. Users are directed to SEBI-registered advisors.
- Every response includes a disclaimer: "Educational information only — not financial advice."
- Fraud-related answers always include the national helpline 1930 and cybercrime.gov.in.
- Mock mode is clearly labeled — simulated responses are never presented as real.

---

## License

MIT — See LICENSE file. Knowledge base content is for educational purposes only. Not financial advice.
