"""Generate the orange outcome distribution figure for the report.

The result file paths are intentionally kept here so the figure can be
regenerated from updated evaluation logs later.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from plot_lib import ResultFile, draw_grouped_figure, parse_result


REPORT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = REPORT_DIR.parent
CLASSICAL_PDF = REPORT_DIR / "figures" / "orange_outcome_classical.pdf"
RECIPES_PDF = REPORT_DIR / "figures" / "orange_outcome_recipes.pdf"


def _r(subdir, fname):
    return ROOT_DIR / "isaac-inference" / "results" / subdir / fname


# Each model is defined once; two figures reuse these specs.
#   §4.1 Figure 1 (classical): the default recipe (backbone frozen; only the action
#     expert + state projection are trained) for every model, grouped by source so the
#     policy-backbone and monotask-vs-subtask choices compare on equal footing.
#   §4.1 Figure 2 (recipe variations): the effect of unfreezing the language backbone and
#     removing the terminal freeze frames (tail-free), shown only for the custom families
#     that actually have those variants (frozen shown alongside as the reference).

# --- LightwheelAI baseline (two policy backbones, frozen) ---
ACT = ResultFile(label="ACT", description="ACT on the LightwheelAI Baseline (chunk 20)",
                 path=_r("ACT-pick-orange-chunk20", "act_latest.txt"),
                 policy="ACT", mode="monotask", group="LightwheelAI\nbaseline", variant="ACT")
SMOLVLA = ResultFile(label="SmolVLA", description="SmolVLA on the LightwheelAI Baseline",
                     path=_r("pick-orange-mimic", "flat_latest.txt"),
                     policy="SmolVLA", mode="monotask", group="LightwheelAI\nbaseline", variant="SmolVLA")

# --- Teleop monotask: frozen vs unfrozen ---
TELEOP_MONO_FROZEN = ResultFile(label="Teleop monotask frozen", description="SmolVLA Teleop monotask (frozen)",
                                path=_r("Gal_split_nolang", "flat_latest.txt"),
                                policy="SmolVLA", mode="monotask", group="Teleop\nmonotask", variant="frozen")
TELEOP_MONO_UNFROZEN = ResultFile(label="Teleop monotask unfrozen", description="SmolVLA Teleop monotask (unfrozen VLM)",
                                  path=_r("Gal_split_nolang-unfrozen-vlm", "flat_latest.txt"),
                                  policy="SmolVLA", mode="monotask", group="Teleop\nmonotask", variant="unfrozen")

# --- Teleop subtask: frozen vs unfrozen vs tail-free ---
TELEOP_SUB_FROZEN = ResultFile(label="Teleop subtask frozen", description="SmolVLA Teleop subtask (frozen)",
                               path=_r("Gal-pick-orange-tailedCH20", "latest.txt"),
                               policy="SmolVLA", mode="subtasks", group="Teleop\nsubtask", variant="frozen")
TELEOP_SUB_UNFROZEN = ResultFile(label="Teleop subtask unfrozen", description="SmolVLA Teleop subtask (unfrozen VLM)",
                                 path=_r("Gal-pick-orange-tailedCH20-unfrozen-vlm", "latest.txt"),
                                 policy="SmolVLA", mode="subtasks", group="Teleop\nsubtask", variant="unfrozen")
TELEOP_SUB_NOTAIL = ResultFile(label="Teleop subtask no-tail", description="SmolVLA Teleop subtask (tail-free, frozen)",
                               path=_r("Gal-pick-orange-notailCH20", "latest.txt"),
                               policy="SmolVLA", mode="subtasks", group="Teleop\nsubtask", variant="no-tail")

# --- Teleop+Auto monotask: frozen vs unfrozen ---
AUTO_MONO_FROZEN = ResultFile(label="Teleop+Auto monotask frozen", description="SmolVLA Teleop+Auto monotask (frozen)",
                              path=_r("Gal-merged-tailed-auto-no-lang-no-home", "flat_latest.txt"),
                              policy="SmolVLA", mode="monotask", group="Teleop+Auto\nmonotask", variant="frozen")
AUTO_MONO_UNFROZEN = ResultFile(label="Teleop+Auto monotask unfrozen", description="SmolVLA Teleop+Auto monotask (unfrozen VLM)",
                                path=_r("Gal-merged-tailed-auto-no-lang-no-home-unfrozen-vlm", "flat_latest.txt"),
                                policy="SmolVLA", mode="monotask", group="Teleop+Auto\nmonotask", variant="unfrozen")

# --- Teleop+Auto subtask: frozen vs unfrozen ---
AUTO_SUB_FROZEN = ResultFile(label="Teleop+Auto subtask frozen", description="SmolVLA Teleop+Auto subtask (frozen)",
                             path=_r("Gal-merged-tailed-auto", "latest.txt"),
                             policy="SmolVLA", mode="subtasks", group="Teleop+Auto\nsubtask", variant="frozen")
AUTO_SUB_UNFROZEN = ResultFile(label="Teleop+Auto subtask unfrozen", description="SmolVLA Teleop+Auto subtask (unfrozen VLM)",
                               path=_r("Gal-merged-tailed-auto-unfrozen-vlm", "latest.txt"),
                               policy="SmolVLA", mode="subtasks", group="Teleop+Auto\nsubtask", variant="unfrozen")


# Figure 1 — classical fine-tuning (frozen only): regroup by source, label bars by
# formulation so the source and monotask-vs-subtask comparisons read directly.
CLASSICAL = [
    ACT,
    SMOLVLA,
    replace(TELEOP_MONO_FROZEN, group="Teleop", variant="monotask"),
    replace(TELEOP_SUB_FROZEN, group="Teleop", variant="subtask"),
    replace(AUTO_MONO_FROZEN, group="Teleop+Auto", variant="monotask"),
    replace(AUTO_SUB_FROZEN, group="Teleop+Auto", variant="subtask"),
]

# Figure 2 — training-recipe variations: frozen vs unfrozen-VLM vs tail-free, per family.
RECIPES = [
    TELEOP_MONO_FROZEN,
    TELEOP_MONO_UNFROZEN,
    TELEOP_SUB_FROZEN,
    TELEOP_SUB_UNFROZEN,
    TELEOP_SUB_NOTAIL,
    AUTO_MONO_FROZEN,
    AUTO_MONO_UNFROZEN,
    AUTO_SUB_FROZEN,
    AUTO_SUB_UNFROZEN,
]


def _parse_all(result_files):
    return [
        (result_file, None if result_file.placeholder else parse_result(result_file))
        for result_file in result_files
    ]


def _report(output_pdf, results):
    print(f"Wrote {output_pdf}")
    for result_file, parsed in results:
        if parsed is None:
            print(f"{result_file.description}: placeholder")
            continue
        values = ", ".join(f"{oranges}/3={parsed.outcomes[oranges][2]:.1f}%" for oranges in [0, 1, 2, 3])
        print(f"{result_file.description}: N={parsed.total}, mean={parsed.mean:.2f}/3, {values}")
        print(f"  source: {result_file.path}")


def main() -> None:
    classical = _parse_all(CLASSICAL)
    draw_grouped_figure(
        classical, CLASSICAL_PDF,
        title="Final orange count - classical fine-tuning",
        recipe_legend=None,
    )
    _report(CLASSICAL_PDF, classical)

    recipes = _parse_all(RECIPES)
    draw_grouped_figure(
        recipes, RECIPES_PDF,
        title="Final orange count - training-recipe variations",
        recipe_legend=("frozen", "unfrozen", "no-tail"),
    )
    _report(RECIPES_PDF, recipes)


if __name__ == "__main__":
    main()
