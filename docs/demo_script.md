# FinSaathi Demo Script (60–90 seconds)

This script walks through the key capabilities of FinSaathi in a live demo.

---

## Step 1 — UPI Question (Ask FinSaathi tab) — ~20 seconds

1. Open the app at `http://localhost:8501`.
2. In the **Ask FinSaathi** tab, click the example button **"How do I send money via UPI?"**
3. Show the response: step-by-step answer, sources expander with NPCI link.
4. **Key point:** "The answer comes from official knowledge-base content grounded in NPCI documentation."

---

## Step 2 — Hindi Language Toggle — ~10 seconds

1. In the **sidebar**, change the language selector to **Hindi**.
2. Click the example button **"How do I avoid UPI scams?"**
3. Show the response in simple Hindi with UPI, OTP, EMI kept in English.
4. **Key point:** "FinSaathi responds in Hindi for users who are more comfortable with it."

---

## Step 3 — Scam Checker — Fake KYC SMS — ~20 seconds

1. Click the **Scam Checker** tab.
2. Paste the following message:
   ```
   URGENT: Your KYC is expired. Click bit.ly/update-kyc to verify immediately or your SBI account will be blocked. Share OTP to confirm.
   ```
3. Click **Check for Scam**.
4. Show: 🔴 **High Risk** badge, red flags listed (urgency language, OTP request, shortened URL, fake KYC threat), and the fraud helpline (1930) automatically included.
5. **Key point:** "No LLM needed for the initial scoring — it's deterministic rule-based, then Granite explains it in plain language."

---

## Step 4 — EMI Calculation — ~15 seconds

1. Click the **Calculators** tab → **EMI Calculator** sub-tab.
2. Enter: Loan Amount **₹5,00,000**, Rate **10.5%**, Tenure **60 months**.
3. Click **Calculate EMI**.
4. Show the formatted table: Monthly EMI, Total Interest, Total Payment.
5. **Key point:** "Transparent calculation — users can verify the formula themselves. No black box."

---

## Wrap-Up Points (if time allows)

- **Mock Mode:** "If credentials aren't available, the app still runs with simulated responses and a clear banner — it never crashes."
- **Privacy:** "OTP, Aadhaar, PAN are masked before reaching the model."
- **Open Knowledge Base:** "The 10 markdown files can be replaced with official RBI/SEBI PDFs for production quality."
- **IBM Granite:** "All generation and embeddings run on IBM watsonx.ai Lite plan — zero cost to get started."
