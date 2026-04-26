"""Ingest PDFs from data/raw/, chunk them, embed them, and store in ChromaDB."""

import logging

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import (
    CHROMADB_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME,
    EMBEDDING_MODEL, RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)


def build_index() -> VectorStoreIndex:
    """Run the full ingestion pipeline and return the built VectorStoreIndex."""
    try:
        logger.info("Loading documents from %s", RAW_DATA_DIR)
        documents = SimpleDirectoryReader(
            input_dir=str(RAW_DATA_DIR),
            recursive=True,
        ).load_data()
        logger.info("Loaded %d documents", len(documents))
    except Exception as exc:
        raise RuntimeError(f"Failed to load documents from {RAW_DATA_DIR}: {exc}") from exc

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    logger.info("Chunking: size=%d, overlap=%d", CHUNK_SIZE, CHUNK_OVERLAP)

    try:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    except Exception as exc:
        raise RuntimeError(f"Failed to load embedding model '{EMBEDDING_MODEL}': {exc}") from exc

    try:
        logger.info("Initialising ChromaDB at %s", CHROMADB_DIR)
        CHROMADB_DIR.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
        chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise ChromaDB at {CHROMADB_DIR}: {exc}") from exc

    try:
        logger.info("Building index (embedding every chunk — takes 1-3 minutes)")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
            transformations=[splitter],
            show_progress=True,
        )
        count = chroma_collection.count()
        logger.info("Done. %d chunks stored in ChromaDB.", count)
        return index
    except Exception as exc:
        raise RuntimeError(f"Failed to build vector index: {exc}") from exc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    build_index()
