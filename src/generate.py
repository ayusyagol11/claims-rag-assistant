"""Generate answers using retrieved context and Claude."""

import logging

import anthropic

from src.config import ANTHROPIC_API_KEY, LLM_MODEL, MAX_TOKENS, TEMPERATURE

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a knowledgeable assistant specialising in Australian \
workers compensation, insurance claims, and related regulatory frameworks.

Your job is to answer questions using ONLY the provided source documents. Follow \
these rules strictly:

1. Ground every factual claim in the provided sources. Do not use outside knowledge.
2. Cite sources inline using the format [Source N] where N is the source number.
3. If the sources do not contain enough information to answer, say so clearly. \
   Do not guess or fabricate.
4. For regulatory or legal questions, quote the relevant clause when possible and \
   note the source document.
5. Be concise. Claims Advisors are busy — lead with the answer, then supporting detail.
6. If the question involves a specific claim calculation, walk through the logic \
   step by step.

You are a decision-support tool, not an autonomous authority. Users must verify \
critical decisions against the source documents themselves."""


def build_prompt(query: str, retrieved_nodes: list) -> str:
    """Format retrieved chunks + user query into a single prompt.

    Args:
        query: The user's question.
        retrieved_nodes: List of NodeWithScore objects from the retriever.

    Returns:
        Formatted prompt string ready to send to the LLM.
    """
    context_parts = []
    for i, node in enumerate(retrieved_nodes, 1):
        source = node.metadata.get("file_name", "unknown")
        context_parts.append(f"[Source {i}: {source}]\n{node.text}\n")
    context = "\n---\n".join(context_parts)

    return f"""Here are the relevant source documents:

{context}

---

Question: {query}

Answer the question using only the sources above. Cite sources inline as [Source N]."""


def generate_answer(query: str, retrieved_nodes: list) -> dict:
    """Call Claude with the assembled prompt and return the answer with source metadata.

    Args:
        query: The user's question.
        retrieved_nodes: List of NodeWithScore objects from the retriever.

    Returns:
        Dict with keys 'answer' (str) and 'sources' (list of source dicts).

    Raises:
        RuntimeError: If the Claude API call fails after retries.
    """
    # max_retries=3 enables SDK-native exponential backoff on 429/5xx errors
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, max_retries=3)
    prompt = build_prompt(query, retrieved_nodes)

    logger.debug("Calling %s (max_tokens=%d, temperature=%s)", LLM_MODEL, MAX_TOKENS, TEMPERATURE)
    try:
        message = client.messages.create(
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError as exc:
        raise RuntimeError("Invalid Anthropic API key — check ANTHROPIC_API_KEY in your .env file.") from exc
    except anthropic.APIError as exc:
        raise RuntimeError(f"Claude API error after retries: {exc}") from exc

    answer = message.content[0].text
    sources = [
        {
            "index": i,
            "file": node.metadata.get("file_name", "unknown"),
            "score": node.score,
            "preview": node.text[:200],
        }
        for i, node in enumerate(retrieved_nodes, 1)
    ]
    logger.debug("Answer generated (%d chars, %d sources)", len(answer), len(sources))
    return {"answer": answer, "sources": sources}
