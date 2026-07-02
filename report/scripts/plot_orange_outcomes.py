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
STANDARD_PDF = REPORT_DIR / "figures" / "orange_outcome_standard.pdf"
VARIANTS_PDF = REPORT_DIR / "figures" / "orange_outcome_variants.pdf"


def _r(subdir, fname):
    return ROOT_DIR / "isaac-inference" / "results" / subdir / fname


# Each model is defined once; two figures reuse these specs.
#   §4.1 Figure 1 (standard): the default fine-tuning (only the action expert and state
#     projection are trained; the vision-language backbone stays frozen) for every model,
#     grouped by source so the policy and monotask-vs-subtask choices compare directly.
#   §4.1 Figure 2 (variants): the effect of also training the language model ("LM-tuned")
#     and of removing the terminal freeze frames from the data ("no-tail"), shown only for
#     the custom families that have those variants (standard shown alongside as reference).

# --- LightwheelAI baseline (two policy backbones, frozen) ---
ACT = ResultFile(label="ACT", description="ACT on the LightwheelAI Baseline (chunk 20)",
                 path=_r("ACT-pick-orange-chunk20", "act_latest.txt"),
                 policy="ACT", mode="monotask", group="LightwheelAI\nbaseline", variant="ACT")
SMOLVLA = ResultFile(label="SmolVLA", description="SmolVLA on the LightwheelAI Baseline",
                     path=_r("pick-orange-mimic", "flat_latest.txt"),
                     policy="SmolVLA", mode="monotask", group="LightwheelAI\nbaseline", variant="SmolVLA")

# --- Teleop monotask: standard vs LM-tuned ---
TELEOP_MONO_FROZEN = ResultFile(label="Teleop monotask frozen", description="SmolVLA Teleop monotask (frozen)",
                                path=_r("Gal_split_nolang", "flat_latest.txt"),
                                policy="SmolVLA", mode="monotask", group="Teleop\nmonotask", variant="standard")
TELEOP_MONO_UNFROZEN = ResultFile(label="Teleop monotask unfrozen", description="SmolVLA Teleop monotask (unfrozen VLM)",
                                  path=_r("Gal_split_nolang-unfrozen-vlm", "flat_latest.txt"),
                                  policy="SmolVLA", mode="monotask", group="Teleop\nmonotask", variant="LM-tuned")

# --- Teleop subtask: standard vs LM-tuned vs no-tail ---
TELEOP_SUB_FROZEN = ResultFile(label="Teleop subtask frozen", description="SmolVLA Teleop subtask (frozen)",
                               path=_r("Gal-pick-orange-tailedCH20", "latest.txt"),
                               policy="SmolVLA", mode="subtasks", group="Teleop\nsubtask", variant="standard")
TELEOP_SUB_UNFROZEN = ResultFile(label="Teleop subtask unfrozen", description="SmolVLA Teleop subtask (unfrozen VLM)",
                                 path=_r("Gal-pick-orange-tailedCH20-unfrozen-vlm", "latest.txt"),
                                 policy="SmolVLA", mode="subtasks", group="Teleop\nsubtask", variant="LM-tuned")
TELEOP_SUB_NOTAIL = ResultFile(label="Teleop subtask no-tail", description="SmolVLA Teleop subtask (no-tail, standard)",
                               path=_r("Gal-pick-orange-notailCH20", "latest.txt"),
                               policy="SmolVLA", mode="subtasks", group="Teleop\nsubtask", variant="Standard\n+ No-tail")

# --- Teleop+Auto monotask: standard vs LM-tuned ---
AUTO_MONO_FROZEN = ResultFile(label="Teleop+Auto monotask frozen", description="SmolVLA Teleop+Auto monotask (frozen)",
                              path=_r("Gal-merged-tailed-auto-no-lang-no-home", "flat_latest.txt"),
                              policy="SmolVLA", mode="monotask", group="Teleop+Auto\nmonotask", variant="standard")
AUTO_MONO_UNFROZEN = ResultFile(label="Teleop+Auto monotask unfrozen", description="SmolVLA Teleop+Auto monotask (unfrozen VLM)",
                                path=_r("Gal-merged-tailed-auto-no-lang-no-home-unfrozen-vlm", "flat_latest.txt"),
                                policy="SmolVLA", mode="monotask", group="Teleop+Auto\nmonotask", variant="LM-tuned")

# --- Teleop+Auto subtask: standard vs LM-tuned ---
AUTO_SUB_FROZEN = ResultFile(label="Teleop+Auto subtask frozen", description="SmolVLA Teleop+Auto subtask (frozen)",
                             path=_r("Gal-merged-tailed-auto", "latest.txt"),
                             policy="SmolVLA", mode="subtasks", group="Teleop+Auto\nsubtask", variant="standard")
AUTO_SUB_UNFROZEN = ResultFile(label="Teleop+Auto subtask unfrozen", description="SmolVLA Teleop+Auto subtask (unfrozen VLM)",
                               path=_r("Gal-merged-tailed-auto-unfrozen-vlm", "latest.txt"),
                               policy="SmolVLA", mode="subtasks", group="Teleop+Auto\nsubtask", variant="LM-tuned")


# Figure 1 — standard fine-tuning only: regroup by source, label bars by formulation
# so the source and monotask-vs-subtask comparisons read directly.
STANDARD = [
    ACT,
    SMOLVLA,
    replace(TELEOP_MONO_FROZEN, group="Teleop", variant="monotask"),
    replace(TELEOP_SUB_FROZEN, group="Teleop", variant="subtask"),
    replace(AUTO_MONO_FROZEN, group="Teleop+Auto", variant="monotask"),
    replace(AUTO_SUB_FROZEN, group="Teleop+Auto", variant="subtask"),
]

# Figure 2 — fine-tuning variants: standard vs LM-tuned vs no-tail, per family.
VARIANTS = [
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
    standard = _parse_all(STANDARD)
    draw_grouped_figure(
        standard, STANDARD_PDF,
        title="Final orange count - standard fine-tuning",
    )
    _report(STANDARD_PDF, standard)

    variants = _parse_all(VARIANTS)
    draw_grouped_figure(
        variants, VARIANTS_PDF,
        title="Final orange count - fine-tuning variants",
        variant_row_label="Fine-tuning:",
    )
    _report(VARIANTS_PDF, variants)


if __name__ == "__main__":
    main()
