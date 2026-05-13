"""Central configuration for the Doc-to-Code AI Assistant.

Loads environment variables from `.env`, resolves project paths, and exposes
all tunable constants (chunking, retrieval, generation, logging) as module-level
attributes. Import as:

    from config import settings
    print(settings.OLLAMA_MODEL)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DOCS_DIR: Path = DATA_DIR / "raw_docs"
PROCESSED_DIR: Path = DATA_DIR / "processed"
CHROMA_DIR: Path = DATA_DIR / "chroma_storage"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
TUTORIALS_DIR: Path = PROJECT_ROOT / "tutorials"

for _dir in (DATA_DIR, RAW_DOCS_DIR, PROCESSED_DIR, CHROMA_DIR, LOGS_DIR, TUTORIALS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Environment (.env)
# --------------------------------------------------------------------------- #

load_dotenv(PROJECT_ROOT / ".env")

# --------------------------------------------------------------------------- #
# LLM — primary (Ollama, local)
# --------------------------------------------------------------------------- #

OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# Day 2 testing showed qwen2.5-coder:1.5b silently dropped the explanation
# and Sources-line parts of the multi-part prompt. Upgraded to 3b — same
# family, ~50% more params, reliably emits code + explanation + Sources.
# Latency trade: ~17s -> ~30s/query on CPU. See REPORT_DELTAS.md §14.
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")
OLLAMA_TIMEOUT_SECONDS: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

# --------------------------------------------------------------------------- #
# LLM — fallback (Google Gemini, cloud)
# --------------------------------------------------------------------------- #

GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or None
# gemini-1.5-pro-latest was retired by Google. Default to 2.5-flash for the
# fallback path: fast, generous free-tier quota, strong instruction-following.
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIMENSIONS: int = 768
EMBEDDING_BATCH_SIZE: int = 32

# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 100

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

CHROMA_COLLECTION: str = "fastapi_docs_embeddings"
K_RESULTS: int = 5
# cosine_distance upper bound (lower = stricter). Bumped from spec value 0.3 to
# 0.5 to match nomic-embed-text's distance calibration — see REPORT_DELTAS.md
# for the rationale. Will be re-tuned during Day 2 query testing.
SIMILARITY_THRESHOLD: float = 0.5

# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

LLM_TEMPERATURE: float = 0.3
MAX_TOKENS: int = 1000
MAX_REGENERATION_ATTEMPTS: int = 2

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_ROTATION: str = "10 MB"
LOG_RETENTION: str = "7 days"
