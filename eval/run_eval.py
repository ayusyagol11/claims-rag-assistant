"""Run evaluation on the test question set."""

import json
from pathlib import Path

import pandas as pd

from src.rag_pipeline import ask

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_questions():
    with open(EVAL_DIR / "test_questions.json") as f:
        return json.load(f)


def check_contains(answer: str, expected_terms: list[str]) -> float:
    """Simple lexical check: what fraction of expected terms appear in answer?"""
    if not expected_terms:
        return 1.0
    hits = sum(1 for term in expected_terms if term.lower() in answer.lower())
    return hits / len(expected_terms)


def check_source(sources: list, hint: str) -> bool:
    """Did the system retrieve from the expected source document?"""
    if not hint:
        return True
    return any(hint.lower() in s["file"].lower() for s in sources)


def run_eval():
    questions = load_questions()
    rows = []

    for q in questions:
        print(f"Running {q['id']}: {q['question'][:60]}...")
        result = ask(q["question"])

        term_coverage = check_contains(
            result["answer"], q.get("expected_answer_contains", [])
        )
        source_match = check_source(
            result["sources"], q.get("expected_source_hint", "")
        )

        rows.append({
            "id": q["id"],
            "question": q["question"],
            "term_coverage": term_coverage,
            "source_match": source_match,
            "answer": result["answer"],
            "top_source": result["sources"][0]["file"] if result["sources"] else None,
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "eval_results.csv", index=False)

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Questions evaluated:       {len(df)}")
    print(f"Mean term coverage:        {df['term_coverage'].mean():.2%}")
    print(f"Source match rate:         {df['source_match'].mean():.2%}")
    print(f"{'='*60}\n")
    print(f"Full results saved to: {RESULTS_DIR / 'eval_results.csv'}")


if __name__ == "__main__":
    run_eval()