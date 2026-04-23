"""Retrieve relevant chunks from ChromaDB given a query."""

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import CHROMADB_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K


def load_index():
    """Load the ChromaDB index built during ingestion."""
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return index


def retrieve(query: str, top_k: int = TOP_K):
    """Return the top-k most relevant chunks for a query."""
    index = load_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return nodes


if __name__ == "__main__":
    # Quick sanity test
    query = "What are the maximum weekly payments for an injured worker?"
    results = retrieve(query)
    print(f"\n🔍 Query: {query}\n")
    for i, node in enumerate(results, 1):
        print(f"--- Result {i} (score={node.score:.3f}) ---")
        print(f"Source: {node.metadata.get('file_name', 'unknown')}")
        print(f"Text: {node.text[:200]}...\n")