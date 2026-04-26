"""Central configuration for the RAG pipeline."""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMADB_DIR = PROCESSED_DIR / "chromadb"

# Embedding model — runs locally, free, 384 dimensions
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking settings
CHUNK_SIZE = 800           # characters per chunk
CHUNK_OVERLAP = 100        # overlap between consecutive chunks

# Retrieval settings
TOP_K = 5                  # number of chunks to retrieve per query

# LLM settings — LLM_MODEL can be overridden via env var for A/B testing
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = 1024
TEMPERATURE = 0.1          # low = more deterministic, better for grounded Q&A

# Collection name in ChromaDB
COLLECTION_NAME = "claims_documents"

if not ANTHROPIC_API_KEY:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY is not set. Add it to your .env file or export it as an environment variable."
    )
