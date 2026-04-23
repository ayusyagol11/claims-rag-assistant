"""End-to-end RAG pipeline: query in, grounded answer + sources out."""

from src.generate import generate_answer
from src.retrieve import retrieve


def ask(query: str) -> dict:
    """Full RAG flow: retrieve relevant chunks, then generate grounded answer."""
    nodes = retrieve(query)
    result = generate_answer(query, nodes)
    return result


if __name__ == "__main__":
    query = "What is the maximum weekly compensation for a worker with total incapacity?"
    result = ask(query)
    print(f"\n❓ Question: {query}\n")
    print(f"💬 Answer:\n{result['answer']}\n")
    print("📚 Sources used:")
    for s in result["sources"]:
        print(f"  [{s['index']}] {s['file']} (score: {s['score']:.3f})")