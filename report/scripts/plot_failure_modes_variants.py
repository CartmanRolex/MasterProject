"""Per-subtask outcome composition across the three fine-tuning variants.

Two figures, one per formulation:
  * ``failure_modes_variants_subtask.pdf``  -- GRASP / LIFT / PLACE
  * ``failure_modes_variants_monotask.pdf`` -- LIFT / PLACE only

Each figure has one panel per dataset (Teleop, Teleop+Auto); within a panel the
subtasks are grouped and each carries three horizontal stacked bars -- one per
variant (standard / partial / full), top-to-bottom -- split into success /
drop-or-slip / timeout as a share of genuine attempts. A percentage x-axis gives
the scale.

GRASP is subtask-only: the monotask per-attempt grasp rate is not inferable (its
offline reconstruction over-segments the grasp search, see
``compute_failure_modes.py``). The monotask LIFT/PLACE are that same
reconstruction; note it expresses failures as drop/slip only (no timeout).

Pure-Python PDF via ``plot_lib`` (no matplotlib). Colours match
``plot_failure_modes.py`` so the figures read as one system.

    python plot_failure_modes_variants.py
"""

from __future__ import annotations

import json
from pathlib import Path

from compute_failure_modes import RESULTS, composition
from plot_failure_modes import grasp_strict, GREEN, ORANGE, GRAY
from plot_lib import PdfFigure

FIGDIR = Path(__file__).resolve().parents[1] / "figures"
SUBTASK_PDF = FIGDIR / "failure_modes_variants_subtask.pdf"
MONOTASK_PDF = FIGDIR / "failure_modes_variants_monotask.pdf"

VARIANTS = ("standard", "partial", "full")
LEGEND = [("Success", GREEN), ("Drop / slip", ORANGE), ("Timeout", GRAY)]
# darker ink for numbers printed on the white margin (right of narrow segments)
OUTSIDE_INK = {GREEN: (0.18, 0.42, 0.24), ORANGE: (0.82, 0.45, 0.10), GRAY: (0.40, 0.42, 0.45)}

# (dataset, {formulation: {variant: results-subdir}})
DATA = [
    ("Teleop", {
        "subtask": {"standard": "Gal-pick-orange-tailedCH20",
                    "partial":  "Gal-pick-orange-tailedCH20-unfrozen-vlm",
                    "full":     "Gal-pick-orange-tailedCH20-unfrozen-all"},
        "monotask": {"standard": "Gal_split_nolang",
                     "partial":  "Gal_split_nolang-unfrozen-vlm",
                     "full":     "Gal_split_nolang-unfrozen-all"},
    }),
    ("Teleop+Auto", {
        "subtask": {"standard": "Gal-merged-tailed-auto",
                    "partial":  "Gal-merged-tailed-auto-unfrozen-vlm",
                    "full":     "Gal-merged-tailed-auto-unfrozen-all"},
        "monotask": {"standard": "Gal-merged-tailed-auto-no-lang-no-home",
                     "partial":  "Gal-merged-tailed-auto-no-lang-no-home-unfrozen-vlm",
                     "full":     "Gal-merged-tailed-auto-no-lang-no-home-unfrozen-all"},
    }),
]

# ---- geometry ----
BAR_H, BAR_GAP, GROUP_GAP = 14.0, 3.5, 16.0
VLAB_W, VARLAB_W, BAR_W, OUT_W, N_W = 16.0, 44.0, 160.0, 32.0, 38.0
PANEL_W = VLAB_W + VARLAB_W + BAR_W + OUT_W + N_W
PANEL_GAP, LEFT, RIGHT = 48.0, 30.0, 18.0
SEG_GAP = 1.5
GROUP_H = 3 * BAR_H + 2 * BAR_GAP
WIDTH = LEFT + 2 * PANEL_W + PANEL_GAP + RIGHT


def load(subdir, formulation):
    fname = "checkpoint.json" if formulation == "subtask" else "flat_checkpoint.json"
    return json.load(open(RESULTS / subdir / fname))


def rows_for(data, formulation):
    """{subtask: (n, [(pct, color), ...])}. GRASP only for the subtask formulation."""
    out = {}
    if formulation == "subtask":
        sg, tg = grasp_strict(data)
        ng = sg + tg
        out["GRASP"] = (ng, [(100 * sg / ng, GREEN), (100 * tg / ng, GRAY)])
    for sub in ("LIFT", "PLACE"):
        succ, drop, tmo, n, _, _ = composition(data, sub)
        out[sub] = (n, [(100 * succ / n, GREEN), (100 * drop / n, ORANGE), (100 * tmo / n, GRAY)])
    return out


def draw_bars(fig, bx, g_top, variant_rows):
    """variant_rows: list of (variant, n, segs) drawn top-to-bottom."""
    for vi, (variant, n, segs) in enumerate(variant_rows):
        by = g_top - BAR_H - vi * (BAR_H + BAR_GAP)
        fig.text(bx - 6, by + BAR_H / 2 - 3, variant, 7.5, "right", rgb=(0.30, 0.30, 0.30), bold=True)
        cx = bx
        outside = []
        for si, (pct, color) in enumerate(segs):
            w = BAR_W * pct / 100.0
            fig.set_fill(color)
            fig.rect(cx, by, w, BAR_H)
            if w >= 13:
                rgb = (0.12, 0.12, 0.12) if color == ORANGE else (1, 1, 1)
                fig.text(cx + w / 2, by + BAR_H / 2 - 3, f"{pct:.0f}", 7.2, "center", rgb=rgb, bold=True)
            elif pct >= 0.5:
                outside.append((pct, color))
            cx += w
            if si < len(segs) - 1 and 0 < w < BAR_W:
                fig.set_fill((1, 1, 1))
                fig.rect(cx - SEG_GAP / 2, by, SEG_GAP, BAR_H)
        fig.set_stroke((0.22, 0.22, 0.22), 0.6)
        fig.rect(bx, by, BAR_W, BAR_H, fill=False)
        ox = bx + BAR_W + 5
        for pct, color in outside:
            fig.text(ox, by + BAR_H / 2 - 3, f"{pct:.0f}", 7.0, "left",
                     rgb=OUTSIDE_INK.get(color, (0.3, 0.3, 0.3)), bold=True)
            ox += len(f"{pct:.0f}") * 5.0 + 5
        fig.text(bx + BAR_W + OUT_W + 4, by + BAR_H / 2 - 3, f"n={n}", 7.0, "left", rgb=(0.34, 0.34, 0.34))


def percent_axis(fig, bx, y):
    fig.set_stroke((0.55, 0.55, 0.55), 0.6)
    fig.line(bx, y, bx + BAR_W, y)
    for pct, lab in [(0, "0"), (50, "50"), (100, "100%")]:
        x = bx + BAR_W * pct / 100.0
        fig.line(x, y, x, y - 3)
        fig.text(x, y - 12, lab, 7.0, "center", rgb=(0.35, 0.35, 0.35))


def draw_figure(formulation, subtasks, title, output_path, legend):
    bars_block = len(subtasks) * GROUP_H + (len(subtasks) - 1) * GROUP_GAP
    height = 60 + bars_block + 66
    fig = PdfFigure(width=WIDTH, height=height)
    fig.text(WIDTH / 2, height - 24, title, 13.0, "center", bold=True)

    bars_top = height - 58
    for di, (dataset, forms) in enumerate(DATA):
        rbv = {v: rows_for(load(sub, formulation), formulation) for v, sub in forms[formulation].items()}
        px = LEFT + di * (PANEL_W + PANEL_GAP)
        bx = px + VLAB_W + VARLAB_W
        fig.text(bx + BAR_W / 2, bars_top + 14, dataset, 10.5, "center", bold=True)
        for gi, sub in enumerate(subtasks):
            g_top = bars_top - gi * (GROUP_H + GROUP_GAP)
            g_bottom = g_top - GROUP_H
            fig.vtext(px + VLAB_W - 3, (g_top + g_bottom) / 2, sub, 8.6, "center",
                      rgb=(0.20, 0.20, 0.20), bold=True)
            variant_rows = [(v, rbv[v][sub][0], rbv[v][sub][1]) for v in VARIANTS]
            draw_bars(fig, bx, g_top, variant_rows)

    axis_y = bars_top - bars_block - 8
    for di in range(len(DATA)):
        px = LEFT + di * (PANEL_W + PANEL_GAP)
        percent_axis(fig, px + VLAB_W + VARLAB_W, axis_y)
    fig.text(WIDTH / 2, axis_y - 24, "Share of genuine attempts (%)", 8.4, "center", rgb=(0.35, 0.35, 0.35))

    lx = WIDTH / 2 - 54 * len(legend)
    ly = axis_y - 42
    for seg_label, color in legend:
        fig.set_fill(color)
        fig.rect(lx, ly, 11, 11)
        fig.text(lx + 15, ly + 2, seg_label, 8.6)
        lx += 108

    fig.save(output_path)


def main() -> None:
    draw_figure("subtask", ("GRASP", "LIFT", "PLACE"),
                "Per-subtask outcome composition -- orchestrated (subtask) models",
                SUBTASK_PDF, LEGEND)
    draw_figure("monotask", ("LIFT", "PLACE"),
                "Per-subtask outcome composition -- monotask models",
                MONOTASK_PDF, LEGEND[:2])  # monotask reconstruction has no timeout
    print(f"Wrote {SUBTASK_PDF}")
    print(f"Wrote {MONOTASK_PDF}")
    for dataset, forms in DATA:
        for formulation in ("subtask", "monotask"):
            for v, sub in forms[formulation].items():
                for st, (n, segs) in rows_for(load(sub, formulation), formulation).items():
                    labels = ["succ", "timeout"] if st == "GRASP" else ["succ", "drop/slip", "timeout"]
                    cells = "  ".join(f"{l} {p:4.1f}%" for l, (p, _) in zip(labels, segs))
                    print(f"  {dataset:12} {formulation:8} {v:8} {st:6} n={n:4}  {cells}")


if __name__ == "__main__":
    main()
