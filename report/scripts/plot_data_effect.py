"""Generate the two-bar data-effect figure (Teleop vs Teleop+Auto subtasks) for the report."""

from __future__ import annotations

from pathlib import Path

from plot_lib import ResultFile, draw_figure, parse_result


REPORT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = REPORT_DIR.parent
OUTPUT_PDF = REPORT_DIR / "figures" / "data_effect.pdf"


RESULT_FILES = [
    ResultFile(
        label="SmolVLA\nTeleop\nsubtasks",
        description="SmolVLA subtask model trained on the cleaned Teleop dataset (846 episodes)",
        path=ROOT_DIR / "isaac-inference" / "results" / "Gal-pick-orange-tailedCH20" / "latest.txt",
        dataset="Teleop",
        policy="SmolVLA",
        mode="subtasks",
        tag="T",
    ),
    ResultFile(
        label="SmolVLA\nTeleop+Auto\nsubtasks",
        description="SmolVLA subtask model trained on merged dataset (846 Teleop + 414 Auto = 1260 episodes)",
        path=ROOT_DIR / "isaac-inference" / "results" / "Gal-merged-tailed-auto" / "latest.txt",
        dataset="Teleop+Auto",
        policy="SmolVLA",
        mode="subtasks",
        tag="T+A",
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
