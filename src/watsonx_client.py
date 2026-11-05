"""
src/watsonx_client.py

Wraps IBM watsonx.ai SDK for text generation and embeddings.
Supports mock mode (no credentials needed) for testing and demos.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class WatsonxError(Exception):
    """Raised when any watsonx.ai SDK call fails."""


# ---------------------------------------------------------------------------
# Disk embedding cache
# ---------------------------------------------------------------------------

class _EmbedCache:
    """Simple JSON cache: sha256(text) -> list[float]"""

    def __init__(self, path: Path):
        self._path = path
        self._data: dict[str, list[float]] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f)

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[list[float]]:
        return self._data.get(self._key(text))

    def set(self, text: str, vector: list[float]):
        self._data[self._key(text)] = vector
        self._save()

    def get_many(self, texts: list[str]) -> dict[str, list[float]]:
        return {t: self._data[self._key(t)] for t in texts if self._key(t) in self._data}


_embed_cache = _EmbedCache(config.EMBED_CACHE_FILE)

# ---------------------------------------------------------------------------
# Mock implementation
# ---------------------------------------------------------------------------

_MOCK_DIM = 128


def _mock_embed(text: str) -> list[float]:
    """Deterministic fake embedding based on text hash."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Expand to _MOCK_DIM floats in [-1, 1]
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    vec = rng.uniform(-1.0, 1.0, _MOCK_DIM).astype(np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 0 else vec).tolist()


def _mock_generate(system_prompt: str, user_prompt: str) -> str:
    return (
        "[MOCK MODE] This is a simulated response. "
        "Set MOCK_MODE=false and provide valid IBM watsonx credentials to get real answers.\n\n"
        f"Your question was: {user_prompt[:200]}"
    )


# ---------------------------------------------------------------------------
# Real watsonx helpers
# ---------------------------------------------------------------------------

_gen_model = None       # ModelInference instance (lazily initialised)
_active_gen_model_id: Optional[str] = None


def _init_gen_model():
    """Try each model in GEN_MODEL_IDS until one works. Cache the result."""
    global _gen_model, _active_gen_model_id
    if _gen_model is not None:
        return _gen_model

    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference

    creds = Credentials(url=config.WATSONX_URL, api_key=config.WATSONX_API_KEY)
    last_err: Optional[Exception] = None

    for model_id in config.GEN_MODEL_IDS:
        try:
            logger.info("Trying generation model: %s", model_id)
            model = ModelInference(
                model_id=model_id,
                credentials=creds,
                project_id=config.WATSONX_PROJECT_ID,
                params=config.GEN_PARAMS,
            )
            # Quick smoke-test (very short call)
            _test_generate(model)
            _gen_model = model
            _active_gen_model_id = model_id
            logger.info("Using generation model: %s", model_id)
            return _gen_model
        except Exception as exc:
            logger.warning("Model %s failed: %s", model_id, exc)
            last_err = exc

    raise WatsonxError(
        f"None of the generation models in GEN_MODEL_IDS could be used. "
        f"Last error: {last_err}"
    )


def _test_generate(model) -> None:
    """Send a minimal prompt to confirm the model is accessible."""
    try:
        if hasattr(model, "chat"):
            model.chat(messages=[{"role": "user", "content": "hi"}])
        else:
            model.generate_text(prompt="hi")
    except Exception as exc:
        raise exc


def _call_generate(model, system_prompt: str, user_prompt: str) -> str:
    """Call generate on a ModelInference object, supporting both APIs."""
    if hasattr(model, "chat"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        response = model.chat(messages=messages)
        # Parse response – structure varies by SDK version
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                return msg.get("content", str(response))
            return response.get("result", str(response))
        return str(response)
    else:
        full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        result = model.generate_text(prompt=full_prompt)
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return result.get("results", [{}])[0].get("generated_text", str(result))
        return str(result)


def _init_embed_model():
    """Return an Embeddings instance."""
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import Embeddings

    creds = Credentials(url=config.WATSONX_URL, api_key=config.WATSONX_API_KEY)
    return Embeddings(
        model_id=config.EMBED_MODEL_ID,
        credentials=creds,
        project_id=config.WATSONX_PROJECT_ID,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_mock() -> bool:
    return config.MOCK_MODE


def generate(system_prompt: str, user_prompt: str) -> str:
    """Generate text using Granite. Falls back to mock on any failure."""
    if config.MOCK_MODE:
        return _mock_generate(system_prompt, user_prompt)

    retries = 0
    delay = 1.0
    last_err: Optional[Exception] = None

    while retries < config.EMBED_MAX_RETRIES:
        try:
            model = _init_gen_model()
            return _call_generate(model, system_prompt, user_prompt)
        except WatsonxError:
            raise
        except Exception as exc:
            last_err = exc
            logger.warning("Generation attempt %d failed: %s", retries + 1, exc)
            time.sleep(delay)
            delay *= 2
            retries += 1

    raise WatsonxError(f"Generation failed after {retries} retries: {last_err}")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts. Returns float32 ndarray of shape (N, dim)."""
    if not texts:
        return np.empty((0,), dtype=np.float32)

    if config.MOCK_MODE:
        return np.array([_mock_embed(t) for t in texts], dtype=np.float32)

    # Check cache
    cached = _embed_cache.get_many(texts)
    results: list[Optional[list[float]]] = [cached.get(t) for t in texts]
    missing_indices = [i for i, v in enumerate(results) if v is None]
    missing_texts = [texts[i] for i in missing_indices]

    if missing_texts:
        embed_model = _init_embed_model()
        # Batch with retries
        for batch_start in range(0, len(missing_texts), config.EMBED_BATCH_SIZE):
            batch = missing_texts[batch_start: batch_start + config.EMBED_BATCH_SIZE]
            delay = 1.0
            for attempt in range(config.EMBED_MAX_RETRIES):
                try:
                    response = embed_model.embed_documents(texts=batch)
                    # Response may be list of lists or list of dicts
                    vectors = _parse_embed_response(response)
                    for j, vec in enumerate(vectors):
                        idx = missing_indices[batch_start + j]
                        results[idx] = vec
                        _embed_cache.set(texts[idx], vec)
                    break
                except Exception as exc:
                    if attempt == config.EMBED_MAX_RETRIES - 1:
                        raise WatsonxError(f"Embedding failed: {exc}") from exc
                    logger.warning("Embed attempt %d failed: %s", attempt + 1, exc)
                    time.sleep(delay)
                    delay *= 2

    return np.array(results, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string. Returns 1-D float32 array."""
    if config.MOCK_MODE:
        return np.array(_mock_embed(text), dtype=np.float32)

    cached = _embed_cache.get(text)
    if cached is not None:
        return np.array(cached, dtype=np.float32)

    embed_model = _init_embed_model()
    delay = 1.0
    last_err: Optional[Exception] = None
    for attempt in range(config.EMBED_MAX_RETRIES):
        try:
            # Try embed_query first, fall back to embed_documents
            if hasattr(embed_model, "embed_query"):
                response = embed_model.embed_query(text=text)
                vec = _parse_single_embed(response)
            else:
                response = embed_model.embed_documents(texts=[text])
                vec = _parse_embed_response(response)[0]
            _embed_cache.set(text, vec)
            return np.array(vec, dtype=np.float32)
        except Exception as exc:
            last_err = exc
            logger.warning("Query embed attempt %d failed: %s", attempt + 1, exc)
            time.sleep(delay)
            delay *= 2

    raise WatsonxError(f"Query embedding failed: {last_err}")


# ---------------------------------------------------------------------------
# Response parsing helpers (SDK returns different shapes across versions)
# ---------------------------------------------------------------------------

def _parse_embed_response(response) -> list[list[float]]:
    """Extract list of float-vectors from embed_documents response."""
    if isinstance(response, np.ndarray):
        return response.tolist()
    if isinstance(response, list):
        if not response:
            return []
        first = response[0]
        if isinstance(first, (list, np.ndarray)):
            return [v if isinstance(v, list) else v.tolist() for v in response]
        if isinstance(first, dict):
            key = next((k for k in ("embedding", "vector", "values") if k in first), None)
            if key:
                return [item[key] for item in response]
        # flat list of floats for single item
        if isinstance(first, float):
            return [response]
    if isinstance(response, dict):
        for key in ("results", "embeddings", "data"):
            if key in response:
                return _parse_embed_response(response[key])
    raise WatsonxError(f"Unrecognised embedding response format: {type(response)}")


def _parse_single_embed(response) -> list[float]:
    """Parse a single embed_query response."""
    if isinstance(response, (list, np.ndarray)):
        arr = response if isinstance(response, list) else response.tolist()
        if arr and isinstance(arr[0], float):
            return arr
        if arr and isinstance(arr[0], list):
            return arr[0]
    if isinstance(response, dict):
        for key in ("embedding", "vector", "values"):
            if key in response:
                return response[key]
    raise WatsonxError(f"Unrecognised single embedding response: {type(response)}")


def get_active_gen_model_id() -> Optional[str]:
    return _active_gen_model_id
