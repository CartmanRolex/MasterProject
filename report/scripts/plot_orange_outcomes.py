"""Generate the orange outcome distribution figures for the report and slides.

The result file paths are intentionally kept here so the figures can be
regenerated from updated evaluation logs later.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from plot_lib import RecipePanel, ResultFile, draw_grouped_figure, draw_recipe_panels, parse_result


REPORT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = REPORT_DIR.parent
STANDARD_PDF = REPORT_DIR / "figures" / "orange_outcome_standard.pdf"   # slides-only
RECIPES_PDF = REPORT_DIR / "figures" / "orange_outcome_recipes.pdf"    # report §4.1


def _r(subdir, fname):
    return ROOT_DIR / "isaac-inference" / "results" / subdir / fname


# Each model is defined once; the figures reuse these specs.
#   §4.1 (report): orange_outcome_recipes.pdf — three stacked panels, one per
#     fine-tuning recipe (standard / partial / full), the five model families in
#     fixed columns so the same family reads vertically across recipes. Batch
#     size and gradient steps are annotated per panel (exceptions per bar).
#   Slides: orange_outcome_standard.pdf — the standard-recipe comparison grouped
#     by source (kept because slides/slides.tex includes it).

# --- LightwheelAI baseline (two policy backbones) ---
ACT = ResultFile(label="ACT", description="ACT on the LightwheelAI Baseline (chunk 20)",
                 path=_r("ACT-pick-orange-chunk20", "act_latest.txt"),
                 policy="ACT", mode="monotask", group="LightwheelAI\nbaseline", variant="ACT")
SMOLVLA = ResultFile(label="SmolVLA", description="SmolVLA on the LightwheelAI Baseline",
                     path=_r("pick-orange-mimic", "flat_latest.txt"),
                     policy="SmolVLA", mode="monotask", group="LightwheelAI\nbaseline", variant="SmolVLA",
                     note="batch 32, 40k")
BASELINE_FULL = ResultFile(label="Baseline fully-tuned", description="SmolVLA Baseline (fully-tuned)",
                           path=_r("pick-orange-mimic-unfrozen-all", "flat_latest.txt"),
                           policy="SmolVLA", mode="monotask", group="LightwheelAI\nbaseline", variant="")

# --- Teleop monotask ---
TELEOP_MONO_FROZEN = ResultFile(label="Teleop monotask frozen", description="SmolVLA Teleop monotask (frozen)",
                                path=_r("Gal_split_nolang", "flat_latest.txt"),
                                policy="SmolVLA", mode="monotask", group="Teleop\nmonotask", variant="standard")
TELEOP_MONO_UNFROZEN = ResultFile(label="Teleop monotask unfrozen", description="SmolVLA Teleop monotask (unfrozen VLM)",
                                  path=_r("Gal_split_nolang-unfrozen-vlm", "flat_latest.txt"),
                                  policy="SmolVLA", mode="monotask", group="Teleop\nmonotask", variant="LM-tuned")
TELEOP_MONO_FULL = ResultFile(label="Teleop monotask fully-tuned", description="SmolVLA Teleop monotask (fully-tuned)",
                              path=_r("Gal_split_nolang-unfrozen-all", "flat_latest.txt"),
                              policy="SmolVLA", mode="monotask", group="Teleop\nmonotask", variant="")

# --- Teleop subtask ---
TELEOP_SUB_FROZEN = ResultFile(label="Teleop subtask frozen", description="SmolVLA Teleop subtask (frozen)",
                               path=_r("Gal-pick-orange-tailedCH20", "latest.txt"),
                               policy="SmolVLA", mode="subtasks", group="Teleop\nsubtask", variant="standard")
TELEOP_SUB_UNFROZEN = ResultFile(label="Teleop subtask unfrozen", description="SmolVLA Teleop subtask (unfrozen VLM)",
                                 path=_r("Gal-pick-orange-tailedCH20-unfrozen-vlm", "latest.txt"),
                                 policy="SmolVLA", mode="subtasks", group="Teleop\nsubtask", variant="LM-tuned")
TELEOP_SUB_PARTIAL_CTRL = ResultFile(
    label="Teleop subtask regime control",
    description="SmolVLA Teleop subtask (unfrozen VLM, batch 32 / 40k regime control)",
    path=_r("Gal-pick-orange-tailedCH20-unfrozen-vlm-b32s40k", "latest.txt"),
    policy="SmolVLA", mode="subtasks", group="Teleop\nsubtask", variant="regime\ncontrol",
    note="batch 32, 40k")
TELEOP_SUB_FULL = ResultFile(label="Teleop subtask fully-tuned",
                             description="SmolVLA Teleop subtask (fully-tuned: all weights incl. vision encoder)",
                             path=_r("Gal-pick-orange-tailedCH20-unfrozen-all", "latest.txt"),
                             policy="SmolVLA", mode="subtasks", group="Teleop\nsubtask", variant="")

# --- Teleop+Auto monotask ---
AUTO_MONO_FROZEN = ResultFile(label="Teleop+Auto monotask frozen", description="SmolVLA Teleop+Auto monotask (frozen)",
                              path=_r("Gal-merged-tailed-auto-no-lang-no-home", "flat_latest.txt"),
                              policy="SmolVLA", mode="monotask", group="Teleop+Auto\nmonotask", variant="standard")
AUTO_MONO_UNFROZEN = ResultFile(label="Teleop+Auto monotask unfrozen", description="SmolVLA Teleop+Auto monotask (unfrozen VLM)",
                                path=_r("Gal-merged-tailed-auto-no-lang-no-home-unfrozen-vlm", "flat_latest.txt"),
                                policy="SmolVLA", mode="monotask", group="Teleop+Auto\nmonotask", variant="LM-tuned")
AUTO_MONO_FULL = ResultFile(label="Teleop+Auto monotask fully-tuned", description="SmolVLA Teleop+Auto monotask (fully-tuned)",
                            path=_r("Gal-merged-tailed-auto-no-lang-no-home-unfrozen-all", "flat_latest.txt"),
                            policy="SmolVLA", mode="monotask", group="Teleop+Auto\nmonotask", variant="")

# --- Teleop+Auto subtask ---
AUTO_SUB_FROZEN = ResultFile(label="Teleop+Auto subtask frozen", description="SmolVLA Teleop+Auto subtask (frozen)",
                             path=_r("Gal-merged-tailed-auto", "latest.txt"),
                             policy="SmolVLA", mode="subtasks", group="Teleop+Auto\nsubtask", variant="standard")
AUTO_SUB_UNFROZEN = ResultFile(label="Teleop+Auto subtask unfrozen", description="SmolVLA Teleop+Auto subtask (unfrozen VLM)",
                               path=_r("Gal-merged-tailed-auto-unfrozen-vlm", "latest.txt"),
                               policy="SmolVLA", mode="subtasks", group="Teleop+Auto\nsubtask", variant="LM-tuned")
AUTO_SUB_FULL = ResultFile(label="Teleop+Auto subtask fully-tuned", description="SmolVLA Teleop+Auto subtask (fully-tuned)",
                           path=_r("Gal-merged-tailed-auto-unfrozen-all", "latest.txt"),
                           policy="SmolVLA", mode="subtasks", group="Teleop+Auto\nsubtask", variant="")


# Slides figure — standard fine-tuning only, regrouped by source, bars labelled
# by formulation. Kept in sync with slides/slides.tex (not used by the report).
STANDARD = [
    ACT,
    SMOLVLA,
    replace(TELEOP_MONO_FROZEN, group="Teleop", variant="monotask"),
    replace(TELEOP_SUB_FROZEN, group="Teleop", variant="subtask"),
    replace(AUTO_MONO_FROZEN, group="Teleop+Auto", variant="monotask"),
    replace(AUTO_SUB_FROZEN, group="Teleop+Auto", variant="subtask"),
]

# Report figure — three panels, one per fine-tuning recipe. Fixed family column
# order; the frozen SmolVLA bars carry no variant label (the column headers and
# panel titles already say everything).
FAMILIES = [
    "LightwheelAI\nbaseline",
    "Teleop\nmonotask",
    "Teleop\nsubtask",
    "Teleop+Auto\nmonotask",
    "Teleop+Auto\nsubtask",
]

PANEL_STANDARD = [
    ACT,
    replace(SMOLVLA, variant=""),   # every bar but ACT is SmolVLA; only ACT is labelled
    replace(TELEOP_MONO_FROZEN, variant=""),
    replace(TELEOP_SUB_FROZEN, variant=""),
    replace(AUTO_MONO_FROZEN, variant=""),
    replace(AUTO_SUB_FROZEN, variant=""),
]
PANEL_PARTIAL = [
    replace(TELEOP_MONO_UNFROZEN, variant=""),
    replace(TELEOP_SUB_UNFROZEN, variant=""),
    TELEOP_SUB_PARTIAL_CTRL,
    replace(AUTO_MONO_UNFROZEN, variant=""),
    replace(AUTO_SUB_UNFROZEN, variant=""),
]
PANEL_FULL = [
    BASELINE_FULL,
    TELEOP_MONO_FULL,
    TELEOP_SUB_FULL,
    AUTO_MONO_FULL,
    AUTO_SUB_FULL,
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
    standard = _parse_all(STANDARD)
    draw_grouped_figure(
        standard, STANDARD_PDF,
        title="Final orange count - standard fine-tuning",
    )
    _report(STANDARD_PDF, standard)

    panels = [
        RecipePanel(
            name="Standard fine-tuning",
            detail="only the action expert and state projection trained; language model and vision encoder frozen",
            regime_note="SmolVLA: batch 64, 20k steps unless noted",
            results=_parse_all(PANEL_STANDARD),
        ),
        RecipePanel(
            name="Partial fine-tuning",
            detail="language model also trained; vision encoder still frozen",
            regime_note="batch 64, 20k steps unless noted",
            results=_parse_all(PANEL_PARTIAL),
        ),
        RecipePanel(
            name="Full fine-tuning",
            detail="all weights trained, vision encoder included",
            regime_note="all models: batch 32, 40k steps",
            results=_parse_all(PANEL_FULL),
        ),
    ]
    draw_recipe_panels(panels, RECIPES_PDF, families=FAMILIES)
    for panel in panels:
        _report(RECIPES_PDF, panel.results)


if __name__ == "__main__":
    main()
