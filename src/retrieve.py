"""Retrieve relevant chunks from ChromaDB given a query."""

import logging

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import CHROMADB_DIR, COLLECTION_NAME, EMBEDDING_MODEL, MODELS_DIR, TOP_K

logger = logging.getLogger(__name__)

_index: VectorStoreIndex | None = None


def _load_embed_model() -> HuggingFaceEmbedding:
    """Load the embedding model, preferring the committed local cache."""
    if MODELS_DIR.exists():
        try:
            return HuggingFaceEmbedding(
                model_name=EMBEDDING_MODEL,
                cache_folder=str(MODELS_DIR),
            )
        except Exception:
            logger.warning("Local model cache at %s failed; falling back to HuggingFace download.", MODELS_DIR)
    return HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)


def load_index() -> VectorStoreIndex:
    """Load the ChromaDB index (cached for the process lifetime — loads once, reused every query)."""
    global _index
    if _index is not None:
        return _index
    try:
        embed_model = _load_embed_model()
        chroma_client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
        chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        _index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model,
        )
        return _index
    except Exception as exc:
        raise RuntimeError(f"Failed to load ChromaDB index from {CHROMADB_DIR}: {exc}") from exc


def retrieve(query: str, top_k: int = TOP_K) -> list[NodeWithScore]:
    """Return the top-k most relevant chunks for a query.

    Args:
        query: Natural language question to search for.
        top_k: Number of chunks to retrieve.

    Returns:
        List of NodeWithScore objects ordered by similarity descending.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    logger.debug("Retrieving top-%d chunks for query: %.80s", top_k, query)
    try:
        index = load_index()
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        logger.debug("Retrieved %d chunks", len(nodes))
        return nodes
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Retrieval failed for query '%.80s': {exc}" % query) from exc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    _query = "What are the maximum weekly payments for an injured worker?"
    _results = retrieve(_query)
    print(f"\nQuery: {_query}\n")
    for i, node in enumerate(_results, 1):
        print(f"--- Result {i} (score={node.score:.3f}) ---")
        print(f"Source: {node.metadata.get('file_name', 'unknown')}")
        print(f"Text: {node.text[:200]}...\n")
