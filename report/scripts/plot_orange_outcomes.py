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


def _r(subdir, fname):
    return ROOT_DIR / "isaac-inference" / "results" / subdir / fname


# Grouped by model family (source x formulation); within a family the bars are the
# training-recipe variants (frozen / unfrozen-VLM / tail-free). Keep same-group entries
# consecutive — the figure groups consecutive rows that share `group`.
RESULT_FILES = [
    # --- LightwheelAI baseline (two policy backbones, frozen) ---
    ResultFile(label="ACT", description="ACT on the LightwheelAI Baseline (chunk 20)",
               path=_r("ACT-pick-orange-chunk20", "act_latest.txt"),
               policy="ACT", mode="monotask", group="LightwheelAI\nbaseline", variant="ACT"),
    ResultFile(label="SmolVLA", description="SmolVLA on the LightwheelAI Baseline",
               path=_r("pick-orange-mimic", "flat_latest.txt"),
               policy="SmolVLA", mode="monotask", group="LightwheelAI\nbaseline", variant="SmolVLA"),

    # --- Teleop monotask: frozen vs unfrozen ---
    ResultFile(label="Teleop monotask frozen", description="SmolVLA Teleop monotask (frozen)",
               path=_r("Gal_split_nolang", "flat_latest.txt"),
               policy="SmolVLA", mode="monotask", group="Teleop\nmonotask", variant="frozen"),
    ResultFile(label="Teleop monotask unfrozen", description="SmolVLA Teleop monotask (unfrozen VLM)",
               path=_r("Gal_split_nolang-unfrozen-vlm", "flat_latest.txt"),
               policy="SmolVLA", mode="monotask", group="Teleop\nmonotask", variant="unfrozen"),

    # --- Teleop subtask: frozen vs unfrozen vs tail-free ---
    ResultFile(label="Teleop subtask frozen", description="SmolVLA Teleop subtask (frozen)",
               path=_r("Gal-pick-orange-tailedCH20", "latest.txt"),
               policy="SmolVLA", mode="subtasks", group="Teleop\nsubtask", variant="frozen"),
    ResultFile(label="Teleop subtask unfrozen", description="SmolVLA Teleop subtask (unfrozen VLM)",
               path=_r("Gal-pick-orange-tailedCH20-unfrozen-vlm", "latest.txt"),
               policy="SmolVLA", mode="subtasks", group="Teleop\nsubtask", variant="unfrozen"),
    ResultFile(label="Teleop subtask no-tail", description="SmolVLA Teleop subtask (tail-free, frozen)",
               path=_r("Gal-pick-orange-notailCH20", "latest.txt"),
               policy="SmolVLA", mode="subtasks", group="Teleop\nsubtask", variant="no-tail"),

    # --- Teleop+Auto monotask: frozen vs unfrozen ---
    ResultFile(label="Teleop+Auto monotask frozen", description="SmolVLA Teleop+Auto monotask (frozen)",
               path=_r("Gal-merged-tailed-auto-no-lang-no-home", "flat_latest.txt"),
               policy="SmolVLA", mode="monotask", group="Teleop+Auto\nmonotask", variant="frozen"),
    ResultFile(label="Teleop+Auto monotask unfrozen", description="SmolVLA Teleop+Auto monotask (unfrozen VLM)",
               path=_r("Gal-merged-tailed-auto-no-lang-no-home-unfrozen-vlm", "flat_latest.txt"),
               policy="SmolVLA", mode="monotask", group="Teleop+Auto\nmonotask", variant="unfrozen"),

    # --- Teleop+Auto subtask: frozen vs unfrozen ---
    ResultFile(label="Teleop+Auto subtask frozen", description="SmolVLA Teleop+Auto subtask (frozen)",
               path=_r("Gal-merged-tailed-auto", "latest.txt"),
               policy="SmolVLA", mode="subtasks", group="Teleop+Auto\nsubtask", variant="frozen"),
    ResultFile(label="Teleop+Auto subtask unfrozen", description="SmolVLA Teleop+Auto subtask (unfrozen VLM)",
               path=_r("Gal-merged-tailed-auto-unfrozen-vlm", "latest.txt"),
               policy="SmolVLA", mode="subtasks", group="Teleop+Auto\nsubtask", variant="unfrozen"),
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
