"""End-to-end RAG pipeline: query in, grounded answer + sources out."""

import logging

from src.generate import generate_answer
from src.retrieve import retrieve

logger = logging.getLogger(__name__)


def ask(query: str) -> dict:
    """Full RAG flow: retrieve relevant chunks, then generate a grounded answer.

    Args:
        query: Natural language question from the user.

    Returns:
        Dict with keys 'answer' (str) and 'sources' (list of source dicts).

    Raises:
        ValueError: If query is empty.
        RuntimeError: If retrieval or generation fails.
    """
    logger.info("Processing query: %.80s", query)
    nodes = retrieve(query)
    result = generate_answer(query, nodes)
    logger.info("Query answered with %d sources", len(result.get("sources", [])))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    _query = "What is the maximum weekly compensation for a worker with total incapacity?"
    _result = ask(_query)
    print(f"\nQuestion: {_query}\n")
    print(f"Answer:\n{_result['answer']}\n")
    print("Sources used:")
    for s in _result["sources"]:
        print(f"  [{s['index']}] {s['file']} (score: {s['score']:.3f})")
