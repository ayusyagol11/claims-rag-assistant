# Claims Document Intelligence

> RAG-powered decision-support assistant for Australian Workers Compensation queries.
> Grounded answers with inline source citations, built by a practising Claims Advisor.

![Architecture](docs/architecture.png)

## Why this project

As a Claims Advisor at Suncorp managing Workers Compensation claims under the ACT
Workers Compensation Act, I spend significant time navigating legislation, regulatory
guidelines, and internal policy documents to answer procedural questions. This project
prototypes a Retrieval-Augmented Generation (RAG) system that grounds LLM responses in
a curated corpus of authoritative sources — reducing hallucination risk while keeping
citations visible to the user.

**The target user** is a Claims Advisor triaging policy questions. The system is
designed as a decision-support tool, not an autonomous agent.

## What it does

- Ingests PDF documents (legislation, SIRA guidelines, Comcare policy, synthetic claim narratives)
- Chunks, embeds, and stores them in a local vector database (ChromaDB)
- Retrieves the top 5 most relevant chunks for any user question
- Generates grounded answers via Claude with inline `[Source N]` citations
- Displays source passages alongside the answer for human verification

## Demo

![Streamlit UI screenshot](docs/screenshots/chat_ui.png)

## Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Orchestration | LlamaIndex | Mature RAG framework, clean abstractions |
| Embeddings | Sentence-Transformers `all-MiniLM-L6-v2` | Free, local, 384-dim, sufficient for this scale |
| Vector DB | ChromaDB | Zero-setup persistent local store |
| LLM | Anthropic Claude Sonnet 4.5 | Strong reasoning, good at instruction-following for grounded Q&A |
| UI | Streamlit | Fast to build, suitable for internal tools |
| Evaluation | Custom harness + Pandas | Lexical coverage + source match on 20 Q&A pairs |

## Architecture decisions

**Chunk size 800 chars / 100 overlap.** Tested 500/1000/1500. 800 balanced retrieval
precision against semantic completeness for legislative text, where clauses often span
a few hundred characters.

**Top-k = 5.** Fewer than 5 missed relevant context on multi-clause questions. More
than 5 diluted the context window with marginal retrievals.

**Temperature 0.1.** Low temperature keeps the LLM close to source material and reduces
paraphrasing drift — important when users may quote the answer to a stakeholder.

**Local embeddings over API embeddings.** Privacy-adjacent — in a real deployment,
claimant-related queries shouldn't leave the organisation. Local embeddings make this
architecturally cleaner.

## Evaluation

20 test questions across 4 categories (legislative, procedural, conceptual, scenario-based)
plus 2 intentionally out-of-scope questions to test grounding discipline.

| Metric | Result |
|--------|--------|
| Mean term coverage | [fill in] |
| Source match rate | [fill in] |
| Out-of-scope refusal | [fill in] |

Full results: [`eval/results/eval_results.csv`](eval/results/eval_results.csv)

**Limitations of this evaluation.** Term coverage is lexical, not semantic — a correct
answer phrased differently will score lower than it should. 20 questions is a small
sample. A production system would need a larger, stratified eval set with human-rated
faithfulness and relevance scores (e.g. via RAGAS).

## Responsible AI considerations

- **Grounding enforced in the system prompt.** The LLM is instructed to refuse when
  sources don't cover the question.
- **Citations visible in the UI.** Users can verify every factual claim.
- **Synthetic claim narratives clearly marked.** No real claimant data was used.
- **Not a legal, medical, or actuarial authority.** The Streamlit UI carries an explicit
  disclaimer.

## Running locally

```bash
git clone https://github.com/ayusyagol11/claims-rag-assistant.git
cd claims-rag-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Add your Anthropic API key
cp .env.example .env
# Edit .env and paste your key

# Place PDFs in data/raw/
python -m src.ingest           # Build the index
python -m eval.run_eval        # Run evaluation
streamlit run app/streamlit_app.py  # Launch UI
```

## What I'd build next

- **Semantic evaluation** with RAGAS (faithfulness, answer relevance, context precision)
- **Agent layer** — a tool-calling pattern with calculators for entitlement math and
  date arithmetic for weeks-since-injury
- **Hybrid retrieval** — combine dense vector search with BM25 keyword matching for
  specific clause numbers and defined terms
- **Citation-level evaluation** — measure how well the LLM's `[Source N]` tags align
  with the chunk that actually supports the claim

## About

Built by [Aayush Yagol](https://aayushyagol.com) — Insurance Data Analyst and practising
Claims Advisor at Suncorp Group, Canberra.