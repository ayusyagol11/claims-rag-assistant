"""Generate answers using retrieved context and Claude."""

import anthropic

from src.config import ANTHROPIC_API_KEY, LLM_MODEL, MAX_TOKENS, TEMPERATURE

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


def build_prompt(query: str, retrieved_nodes) -> str:
    """Format retrieved chunks + user query into a single prompt."""
    context_parts = []
    for i, node in enumerate(retrieved_nodes, 1):
        source = node.metadata.get("file_name", "unknown")
        context_parts.append(
            f"[Source {i}: {source}]\n{node.text}\n"
        )
    context = "\n---\n".join(context_parts)

    return f"""Here are the relevant source documents:

{context}

---

Question: {query}

Answer the question using only the sources above. Cite sources inline as [Source N]."""


def generate_answer(query: str, retrieved_nodes) -> dict:
    """Call Claude with the assembled prompt. Return answer + source list."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = build_prompt(query, retrieved_nodes)

    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

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
    return {"answer": answer, "sources": sources}