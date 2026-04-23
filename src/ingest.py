"""Ingest PDFs from data/raw/, chunk them, embed them, and store in ChromaDB."""

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import (
    CHROMADB_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME,
    EMBEDDING_MODEL, RAW_DATA_DIR,
)


def build_index():
    """Run the full ingestion pipeline."""
    print(f"📚 Loading documents from {RAW_DATA_DIR}")
    documents = SimpleDirectoryReader(
        input_dir=str(RAW_DATA_DIR),
        recursive=True,
    ).load_data()
    print(f"   Loaded {len(documents)} documents")

    print(f"✂️  Chunking: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print(f"🧠 Loading embedding model: {EMBEDDING_MODEL}")
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

    print(f"💾 Initialising ChromaDB at {CHROMADB_DIR}")
    CHROMADB_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"🚀 Building index (this embeds every chunk — takes 1-3 minutes)")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )

    count = chroma_collection.count()
    print(f"✅ Done. {count} chunks stored in ChromaDB.")
    return index


if __name__ == "__main__":
    build_index()