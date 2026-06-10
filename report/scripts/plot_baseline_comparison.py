"""Generate the two-bar baseline comparison figure (ACT vs SmolVLA) for the report."""

from __future__ import annotations

from pathlib import Path

from plot_lib import ResultFile, draw_grouped_figure, parse_result


REPORT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = REPORT_DIR.parent
OUTPUT_PDF = REPORT_DIR / "figures" / "baseline_comparison.pdf"


RESULT_FILES = [
    ResultFile(
        label="ACT\nLightwheelAI\nBaseline\nmonotask",
        description="ACT on the LightwheelAI Baseline dataset (execute chunk size 20)",
        path=ROOT_DIR / "isaac-inference" / "results" / "ACT-pick-orange-chunk20" / "act_latest.txt",
        dataset="LightwheelAI Baseline",
        policy="ACT",
        mode="monotask",
        tag="LW",
    ),
    ResultFile(
        label="SmolVLA\nLightwheelAI\nBaseline\nmonotask",
        description="SmolVLA on the LightwheelAI Baseline dataset",
        path=ROOT_DIR / "isaac-inference" / "results" / "pick-orange-mimic" / "flat_latest.txt",
        dataset="LightwheelAI Baseline",
        policy="SmolVLA",
        mode="monotask",
        tag="LW",
    ),
]


def main() -> None:
    results = [(result_file, parse_result(result_file)) for result_file in RESULT_FILES]
    draw_grouped_figure(results, OUTPUT_PDF, bar_w=60)

    print(f"Wrote {OUTPUT_PDF}")
    for result_file, parsed in results:
        values = ", ".join(f"{oranges}/3={parsed.outcomes[oranges][2]:.1f}%" for oranges in [0, 1, 2, 3])
        print(f"{result_file.description}: N={parsed.total}, mean={parsed.mean:.2f}/3, {values}")
        print(f"  source: {result_file.path}")


if __name__ == "__main__":
    main()
