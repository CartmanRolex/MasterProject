"""Generate the orange outcome distribution figure for the report.

The result file paths are intentionally kept here so the figure can be
regenerated from updated evaluation logs later.
"""

from __future__ import annotations

from pathlib import Path

from plot_lib import ResultFile, draw_figure, parse_result


REPORT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = REPORT_DIR.parent
OUTPUT_PDF = REPORT_DIR / "figures" / "orange_outcome_distribution.pdf"


RESULT_FILES = [
    ResultFile(
        label="ACT\nHF full-task\nsingle-task",
        description="ACT on official HuggingFace monolithic task dataset",
        path=ROOT_DIR / "isaac-inference" / "results" / "ACT-pick-orange" / "eval_ACT-pick-orange_2026-03-23_19-18-01.txt",
    ),
    ResultFile(
        label="SmolVLA\nHF full-task\nsingle-task",
        description="SmolVLA on official HuggingFace monolithic task dataset",
        path=ROOT_DIR / "isaac-inference" / "results" / "pretrained_model" / "eval_pretrained_model_2026-04-07_13-18-38.txt",
    ),
    ResultFile(
        label="SmolVLA\nHandmade subtasks\norchestrated subtasks",
        description="SmolVLA subtask model trained on the handmade dataset",
        path=ROOT_DIR / "isaac-inference" / "results" / "Gal-pick-orange-tailedCH20" / "latest.txt",
    ),
    ResultFile(
        label="SmolVLA\nHand+auto\norchestrated subtasks\nprelim.",
        description="SmolVLA subtask model trained on handmade data merged with automated data generation",
        path=ROOT_DIR / "isaac-inference" / "results" / "Gal-merged-tailed-auto" / "latest.txt",
    ),
    ResultFile(
        label="SmolVLA\nHand+auto\nsingle-task no language\nprelim.",
        description=(
            "SmolVLA monolithic fair-comparison model trained from the handmade + automated "
            "dataset after removing language labels, tail frames, and unused home tasks"
        ),
        path=ROOT_DIR / "isaac-inference" / "results" / "Gal-merged-tailed-auto-no-lang-no-home" / "flat_latest.txt",
    ),
]


def main() -> None:
    results = [(result_file, parse_result(result_file)) for result_file in RESULT_FILES]
    draw_figure(results, OUTPUT_PDF)

    print(f"Wrote {OUTPUT_PDF}")
    for result_file, parsed in results:
        values = ", ".join(f"{oranges}/3={parsed.outcomes[oranges][2]:.1f}%" for oranges in [0, 1, 2, 3])
        print(f"{result_file.description}: N={parsed.total}, mean={parsed.mean:.2f}/3, {values}")
        print(f"  source: {result_file.path}")


if __name__ == "__main__":
    main()
