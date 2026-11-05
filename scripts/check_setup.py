"""
scripts/check_setup.py

Verifies the FinSaathi setup end-to-end.
Run: python scripts/check_setup.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force .env load
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import config


def _pass(msg):
    print(f"  [PASS] {msg}")


def _fail(msg):
    print(f"  [FAIL] {msg}")


def _info(msg):
    print(f"  [INFO] {msg}")


def check_env():
    print("\n[1] Checking .env / environment variables")
    ok = True
    if config.WATSONX_API_KEY:
        _pass("WATSONX_API_KEY is set")
    else:
        _fail("WATSONX_API_KEY is missing")
        ok = False
    if config.WATSONX_PROJECT_ID:
        _pass("WATSONX_PROJECT_ID is set")
    else:
        _fail("WATSONX_PROJECT_ID is missing")
        ok = False
    _info(f"WATSONX_URL = {config.WATSONX_URL}")
    _info(f"GEN_MODEL_IDS = {config.GEN_MODEL_IDS}")
    _info(f"EMBED_MODEL_ID = {config.EMBED_MODEL_ID}")
    _info(f"MOCK_MODE = {config.MOCK_MODE}")
    return ok


def check_ibm_auth():
    print("\n[2] Checking IBM watsonx.ai authentication")
    if config.MOCK_MODE:
        _info("MOCK_MODE active — skipping real IBM auth check")
        return True
    try:
        from ibm_watsonx_ai import Credentials, APIClient
        creds = Credentials(url=config.WATSONX_URL, api_key=config.WATSONX_API_KEY)
        client = APIClient(creds)
        client.set.default_project(config.WATSONX_PROJECT_ID)
        _pass("IBM authentication successful")
        return True
    except Exception as exc:
        _fail(f"IBM authentication failed: {exc}")
        return False


def check_gen_models():
    print("\n[3] Checking generation models")
    if config.MOCK_MODE:
        _info("MOCK_MODE active — skipping model check")
        return True

    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference

    creds = Credentials(url=config.WATSONX_URL, api_key=config.WATSONX_API_KEY)
    usable = []
    failed = []

    for model_id in config.GEN_MODEL_IDS:
        try:
            model = ModelInference(
                model_id=model_id,
                credentials=creds,
                project_id=config.WATSONX_PROJECT_ID,
                params={"max_new_tokens": 10},
            )
            if hasattr(model, "chat"):
                resp = model.chat(messages=[{"role": "user", "content": "hi"}])
            else:
                resp = model.generate_text(prompt="hi")
            usable.append(model_id)
            _pass(f"Model usable: {model_id}")
        except Exception as exc:
            failed.append((model_id, str(exc)))
            _fail(f"Model failed: {model_id} — {exc}")

    if not usable:
        _fail("No generation models are usable. Check model IDs.")
        # Try to list available models
        try:
            from ibm_watsonx_ai import Credentials, APIClient
            client = APIClient(Credentials(url=config.WATSONX_URL, api_key=config.WATSONX_API_KEY))
            client.set.default_project(config.WATSONX_PROJECT_ID)
            models = client.foundation_models.TextModels
            _info("Available foundation models (text):")
            for m in dir(models):
                if not m.startswith("_"):
                    val = getattr(models, m, None)
                    if isinstance(val, str) and "/" in val:
                        print(f"     {val}")
        except Exception as e:
            _info(f"Could not list available models: {e}")
        return False
    return True


def check_embeddings():
    print("\n[4] Checking embeddings")
    if config.MOCK_MODE:
        from src.watsonx_client import embed_query
        vec = embed_query("test")
        _pass(f"Mock embeddings working — dim={len(vec)}")
        return True
    try:
        from src.watsonx_client import embed_query
        vec = embed_query("What is UPI?")
        _pass(f"Embeddings working — dim={len(vec)}, first value={vec[0]:.4f}")
        return True
    except Exception as exc:
        _fail(f"Embedding failed: {exc}")
        return False


def check_generation():
    print("\n[5] Checking text generation")
    if config.MOCK_MODE:
        from src.watsonx_client import generate
        resp = generate("You are a helpful assistant.", "What is UPI in one sentence?")
        _pass(f"Mock generation working: {resp[:80]}…")
        return True
    try:
        from src.watsonx_client import generate, get_active_gen_model_id
        resp = generate(
            "You are FinSaathi, a financial literacy assistant.",
            "What is UPI? Answer in one sentence.",
        )
        model_used = get_active_gen_model_id()
        _pass(f"Generation working (model: {model_used}): {resp[:100]}…")
        return True
    except Exception as exc:
        _fail(f"Generation failed: {exc}")
        return False


def main():
    print("=" * 60)
    print("FinSaathi Setup Check")
    print("=" * 60)

    results = {
        "env": check_env(),
        "auth": check_ibm_auth(),
        "models": check_gen_models(),
        "embeddings": check_embeddings(),
        "generation": check_generation(),
    }

    print("\n" + "=" * 60)
    all_pass = all(results.values())
    if all_pass:
        print("All checks passed!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"Some checks failed: {', '.join(failed)}")

    if config.MOCK_MODE:
        print("\nRunning in MOCK MODE — set real credentials in .env for production use.")

    print("\nNext steps:")
    if not config.WATSONX_API_KEY or not config.WATSONX_PROJECT_ID:
        print("  1. Copy .env.example to .env and fill in your IBM watsonx credentials")
        print("  2. Re-run: python scripts/check_setup.py")
    else:
        print("  1. python -m src.ingest        (build the knowledge index)")
        print("  2. streamlit run app.py         (start the app)")
    print("=" * 60)


if __name__ == "__main__":
    main()
