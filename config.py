# config.py - All configuration loaded from environment variables

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_ROOT = Path(__file__).parent
load_dotenv(_ROOT / ".env")

# IBM watsonx credentials
WATSONX_API_KEY: str = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID: str = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL: str = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# Model settings
_raw_gen_models = os.getenv(
    "GEN_MODEL_IDS",
    "ibm/granite-3-3-8b-instruct,ibm/granite-3-8b-instruct,ibm/granite-3-2-8b-instruct",
)
GEN_MODEL_IDS: list[str] = [m.strip() for m in _raw_gen_models.split(",") if m.strip()]
EMBED_MODEL_ID: str = os.getenv(
    "EMBED_MODEL_ID", "ibm/granite-embedding-278m-multilingual"
)

# Mock mode - active when MOCK_MODE=true OR credentials are missing
_mock_env = os.getenv("MOCK_MODE", "false").lower() in ("true", "1", "yes")
MOCK_MODE: bool = _mock_env or not WATSONX_API_KEY or not WATSONX_PROJECT_ID

# Generation parameters
GEN_PARAMS: dict = {
    "temperature": 0.2,
    "max_new_tokens": 500,
    "repetition_penalty": 1.05,
}

# Paths
DATA_DIR = _ROOT / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
INDEX_DIR = DATA_DIR / "index"
INDEX_EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
INDEX_CHUNKS_FILE = INDEX_DIR / "chunks.json"
EMBED_CACHE_FILE = INDEX_DIR / "embed_cache.json"

# Chunking
CHUNK_SIZE: int = 800
CHUNK_OVERLAP: int = 120

# Retrieval
RETRIEVAL_TOP_K: int = 4
RETRIEVAL_MIN_SCORE: float = 0.25  # ignored in mock mode (threshold lowered)

# Embedding batch size
EMBED_BATCH_SIZE: int = 16
EMBED_MAX_RETRIES: int = 3
