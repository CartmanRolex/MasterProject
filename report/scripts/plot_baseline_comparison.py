"""Generate the two-bar baseline comparison figure (ACT vs SmolVLA) for the report."""

from __future__ import annotations

from pathlib import Path

from plot_lib import ResultFile, draw_figure, parse_result


REPORT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = REPORT_DIR.parent
OUTPUT_PDF = REPORT_DIR / "figures" / "baseline_comparison.pdf"


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
]


def main() -> None:
    results = [(result_file, parse_result(result_file)) for result_file in RESULT_FILES]
    draw_figure(results, OUTPUT_PDF, bar_w=90)

    print(f"Wrote {OUTPUT_PDF}")
    for result_file, parsed in results:
        values = ", ".join(f"{oranges}/3={parsed.outcomes[oranges][2]:.1f}%" for oranges in [0, 1, 2, 3])
        print(f"{result_file.description}: N={parsed.total}, mean={parsed.mean:.2f}/3, {values}")


if __name__ == "__main__":
    main()
