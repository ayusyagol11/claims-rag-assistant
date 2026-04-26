"""Run evaluation on the test question set and generate a term coverage chart."""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from src.rag_pipeline import ask

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Brand palette ─────────────────────────────────────────────────────────────
BG         = "#09090B"
BG_CARD    = "#18181B"
BG_RAISED  = "#27272A"
BORDER     = "#3F3F46"
TEXT       = "#FAFAFA"
TEXT_MUTED = "#A1A1AA"
AMBER      = "#BA7517"
AMBER_MID  = "#EF9F27"
AMBER_DARK = "#633806"


def load_questions() -> list[dict]:
    """Load the test question set from test_questions.json."""
    with open(EVAL_DIR / "test_questions.json") as f:
        return json.load(f)


def check_contains(answer: str, expected_terms: list[str]) -> float:
    """Simple lexical check: what fraction of expected terms appear in answer?"""
    if not expected_terms:
        return 1.0
    hits = sum(1 for term in expected_terms if term.lower() in answer.lower())
    return hits / len(expected_terms)


def check_source(sources: list[dict], hint: str) -> bool:
    """Did the system retrieve from the expected source document?"""
    if not hint:
        return True
    return any(hint.lower() in s["file"].lower() for s in sources)


def generate_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Generate a branded bar chart of term coverage per question and save as PNG.

    Args:
        df: Evaluation results DataFrame with columns id, term_coverage, source_match.
        output_path: Destination path for the PNG file.
    """
    ids       = df["id"].tolist()
    coverages = df["term_coverage"].tolist()
    sources   = df["source_match"].tolist()

    # Bar colour: amber for full coverage, muted for partial, raised for zero
    def bar_colour(cov: float) -> str:
        if cov == 1.0:
            return AMBER
        elif cov == 0.0:
            return "#EF4444"   # red — hard miss
        else:
            return AMBER_MID   # partial

    colours = [bar_colour(c) for c in coverages]

    fig, ax = plt.subplots(figsize=(13, 5.5))

    # Background
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG_CARD)

    # Bars
    bars = ax.bar(ids, coverages, color=colours, width=0.62, zorder=3)

    # Mean line
    mean_cov = sum(coverages) / len(coverages)
    ax.axhline(
        mean_cov,
        color=TEXT_MUTED,
        linewidth=1.1,
        linestyle="--",
        zorder=4,
        label=f"Mean  {mean_cov:.0%}",
    )

    # Source-match indicator: small white dot on top of each bar when source matched
    for bar, src_ok, cov in zip(bars, sources, coverages):
        if src_ok:
            ax.plot(
                bar.get_x() + bar.get_width() / 2,
                cov + 0.025,
                marker="o",
                markersize=4,
                color=TEXT,
                zorder=5,
                clip_on=False,
            )

    # Value labels on bars
    for bar, cov in zip(bars, coverages):
        label = f"{cov:.0%}"
        y_pos = cov + 0.04 if cov < 0.92 else cov - 0.09
        colour = TEXT if cov < 0.92 else AMBER_DARK
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            label,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=colour,
            fontweight="500",
            zorder=6,
        )

    # Axes styling
    ax.set_ylim(0, 1.18)
    ax.set_xlim(-0.6, len(ids) - 0.4)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, color=TEXT_MUTED, fontsize=8.5)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], color=TEXT_MUTED, fontsize=8.5)
    ax.tick_params(axis="both", which="both", length=0)

    # Grid
    ax.yaxis.grid(True, color=BORDER, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Spines
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)

    # Labels
    ax.set_xlabel("Question ID", color=TEXT_MUTED, fontsize=9, labelpad=8)
    ax.set_ylabel("Term Coverage", color=TEXT_MUTED, fontsize=9, labelpad=8)

    # Eyebrow label (top-left, mono style)
    fig.text(
        0.013, 0.97,
        "RETRIEVAL EVALUATION  ·  TERM COVERAGE BY QUESTION",
        color=AMBER,
        fontsize=7,
        fontfamily="monospace",
        va="top",
        transform=fig.transFigure,
    )

    # Title
    ax.set_title(
        "Term Coverage per Question",
        color=TEXT,
        fontsize=13,
        fontweight="bold",
        pad=18,
        loc="left",
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color=AMBER,    label="Full coverage (1.0)"),
        mpatches.Patch(color=AMBER_MID, label="Partial coverage"),
        mpatches.Patch(color="#EF4444", label="Zero coverage"),
        plt.Line2D([0], [0], color=TEXT_MUTED, linewidth=1.1, linestyle="--",
                   label=f"Mean {mean_cov:.0%}"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=TEXT,
                   markersize=5, linestyle="None", label="Source matched ✓"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=1,
        facecolor=BG_RAISED,
        edgecolor=BORDER,
        labelcolor=TEXT_MUTED,
        fontsize=8,
        handlelength=1.4,
    )

    # Summary annotation (bottom-right)
    n_perfect  = sum(1 for c in coverages if c == 1.0)
    n_zero     = sum(1 for c in coverages if c == 0.0)
    n_src_ok   = sum(1 for s in sources if s)
    summary = (
        f"n = {len(df)}  ·  "
        f"Perfect: {n_perfect}  ·  "
        f"Zero: {n_zero}  ·  "
        f"Source match: {n_src_ok}/{len(df)}"
    )
    fig.text(
        0.987, 0.013,
        summary,
        color=TEXT_MUTED,
        fontsize=7.5,
        fontfamily="monospace",
        ha="right",
        va="bottom",
        transform=fig.transFigure,
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    logger.info("Chart saved to: %s", output_path)


def run_eval() -> None:
    """Run all test questions through the RAG pipeline and write results."""
    questions = load_questions()
    rows = []

    for q in questions:
        logger.info("Running %s: %.60s", q["id"], q["question"])
        try:
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
        except Exception as exc:
            logger.error("Question %s failed: %s", q["id"], exc, exc_info=True)
            rows.append({
                "id": q["id"],
                "question": q["question"],
                "term_coverage": 0.0,
                "source_match": False,
                "answer": f"ERROR: {exc}",
                "top_source": None,
            })

    df = pd.DataFrame(rows)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "eval_results.csv"
    df.to_csv(csv_path, index=False)

    # ── Generate chart ─────────────────────────────────────────────────────────
    chart_path = RESULTS_DIR / "term_coverage_chart.png"
    generate_chart(df, chart_path)

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Questions evaluated:    {len(df)}")
    print(f"Mean term coverage:     {df['term_coverage'].mean():.2%}")
    print(f"Source match rate:      {df['source_match'].mean():.2%}")
    print(f"Perfect scores (1.0):   {(df['term_coverage'] == 1.0).sum()}")
    print(f"Zero scores (0.0):      {(df['term_coverage'] == 0.0).sum()}")
    print(f"{'='*60}\n")
    print(f"CSV:   {csv_path}")
    print(f"Chart: {chart_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    run_eval()
