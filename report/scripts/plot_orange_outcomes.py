"""Generate the orange outcome distribution figure for the report.

The result file paths are intentionally kept here so the figure can be
regenerated from updated evaluation logs later.
"""

from __future__ import annotations

from pathlib import Path

from plot_lib import ResultFile, draw_grouped_figure, parse_result


REPORT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = REPORT_DIR.parent
OUTPUT_PDF = REPORT_DIR / "figures" / "orange_outcome_distribution.pdf"


RESULT_FILES = [
    ResultFile(
        label="ACT\nHF Full-Task\nmonotask",
        description="ACT on the HF Full-Task dataset (execute chunk size 20)",
        path=ROOT_DIR / "isaac-inference" / "results" / "ACT-pick-orange-chunk20" / "act_latest.txt",
        dataset="HF Full-Task",
        policy="ACT",
        mode="monotask",
        tag="HF",
    ),
    ResultFile(
        label="SmolVLA\nHF Full-Task\nmonotask",
        description="SmolVLA on the HF Full-Task dataset",
        path=ROOT_DIR / "isaac-inference" / "results" / "pick-orange-mimic" / "flat_latest.txt",
        dataset="HF Full-Task",
        policy="SmolVLA",
        mode="monotask",
        tag="HF",
    ),
    ResultFile(
        label="SmolVLA\nTeleop\nmonotask",
        description="SmolVLA monotask model trained on the teleoperated dataset",
        path=ROOT_DIR / "isaac-inference" / "results" / "Gal_split_nolang" / "flat_latest.txt",
        dataset="Teleop",
        policy="SmolVLA",
        mode="monotask",
        tag="T",
    ),
    ResultFile(
        label="SmolVLA\nTeleop\nsubtasks",
        description="SmolVLA subtask model trained on the teleoperated dataset",
        path=ROOT_DIR / "isaac-inference" / "results" / "Gal-pick-orange-tailedCH20" / "latest.txt",
        dataset="Teleop",
        policy="SmolVLA",
        mode="subtasks",
        tag="T",
    ),
    ResultFile(
        label="SmolVLA\nTeleop+Auto\nmonotask",
        description=(
            "SmolVLA monotask fair-comparison model trained from the Teleop+Auto "
            "dataset with a fixed full-task prompt"
        ),
        path=ROOT_DIR / "isaac-inference" / "results" / "Gal-merged-tailed-auto-no-lang-no-home" / "flat_latest.txt",
        dataset="Teleop+Auto",
        policy="SmolVLA",
        mode="monotask",
        tag="T+A",
    ),
    ResultFile(
        label="SmolVLA\nTeleop+Auto\nsubtasks",
        description="SmolVLA subtask model trained on teleoperated data merged with automated data generation",
        path=ROOT_DIR / "isaac-inference" / "results" / "Gal-merged-tailed-auto" / "latest.txt",
        dataset="Teleop+Auto",
        policy="SmolVLA",
        mode="subtasks",
        tag="T+A",
    ),
]


def main() -> None:
    results = [
        (result_file, None if result_file.placeholder else parse_result(result_file))
        for result_file in RESULT_FILES
    ]
    draw_grouped_figure(results, OUTPUT_PDF)

    print(f"Wrote {OUTPUT_PDF}")
    for result_file, parsed in results:
        if parsed is None:
            print(f"{result_file.description}: placeholder")
            continue
        values = ", ".join(f"{oranges}/3={parsed.outcomes[oranges][2]:.1f}%" for oranges in [0, 1, 2, 3])
        print(f"{result_file.description}: N={parsed.total}, mean={parsed.mean:.2f}/3, {values}")
        print(f"  source: {result_file.path}")


if __name__ == "__main__":
    main()
