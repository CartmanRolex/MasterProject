"""Per-subtask outcome-composition figure for the orchestrated (subtask) models.

One panel per subtask model; horizontal stacked bars for GRASP / LIFT / PLACE, each
split into success / drop-or-slip / timeout (share of genuine attempts). GRASP has no
drop/slip -- an attempt either secures an orange or times out.

Monotask models are intentionally excluded: their per-subtask outcomes are reconstructed
offline and the grasp search over-segments (median ~9 segments/episode), so there is no
trustworthy per-attempt grasp rate (see ``compute_failure_modes.py``). Their LIFT/PLACE
slip signatures are reported in ``tab:failure_geometry`` instead.

Pure-Python PDF via ``plot_lib`` (no matplotlib).

    python plot_failure_modes.py
"""

from __future__ import annotations

import json
from pathlib import Path

from compute_failure_modes import RESULTS, composition, grasp_composition
from plot_lib import PdfFigure

OUTPUT_PDF = Path(__file__).resolve().parents[1] / "figures" / "failure_modes.pdf"

MODELS = [
    ("Teleop subtask",      "Gal-pick-orange-tailedCH20", "checkpoint.json"),
    ("Teleop+Auto subtask", "Gal-merged-tailed-auto",     "checkpoint.json"),
]

GREEN = (0.24, 0.55, 0.32)   # success
ORANGE = (0.93, 0.56, 0.21)  # drop / slip
GRAY = (0.50, 0.53, 0.57)    # timeout
LEGEND = [("success", GREEN), ("drop / slip", ORANGE), ("timeout", GRAY)]


def rows_for(data):
    """[(subtask, n, [(pct, color), ...]), ...] in GRASP, LIFT, PLACE order."""
    sg, tg, ng, _, _ = grasp_composition(data)
    out = [("GRASP", ng, [(100 * sg / ng, GREEN), (100 * tg / ng, GRAY)])]
    for sub in ("LIFT", "PLACE"):
        succ, drop, tmo, n, _, _ = composition(data, sub)
        out.append((sub, n, [
            (100 * succ / n, GREEN),
            (100 * drop / n, ORANGE),
            (100 * tmo / n, GRAY),
        ]))
    return out


def main() -> None:
    fig = PdfFigure(width=720, height=205)
    fig.text(fig.width / 2, fig.height - 22,
             "Per-subtask outcome composition (orchestrated models)", 13.5, "center", bold=True)
    fig.text(fig.width / 2, fig.height - 37,
             "Share of genuine attempts; GRASP fails only by timeout (no drop/slip)",
             8.8, "center", rgb=(0.30, 0.30, 0.30))

    panel_top = fig.height - 70
    bar_h, row_gap = 20, 13
    label_w, bar_w = 48, 196
    panels_left, panel_gap = 54, 48
    panel_w = label_w + bar_w + 38

    for pi, (name, subdir, fname) in enumerate(MODELS):
        data = json.load(open(RESULTS / subdir / fname))
        px = panels_left + pi * (panel_w + panel_gap)
        bx = px + label_w
        fig.text(bx + bar_w / 2, panel_top + 16, name, 10.5, "center", bold=True)
        for ri, (sub, n, segs) in enumerate(rows_for(data)):
            y = panel_top - 16 - ri * (bar_h + row_gap)
            fig.text(px + label_w - 6, y + bar_h / 2 - 3, sub, 8.4, "right", bold=True)
            cx = bx
            for pct, color in segs:
                w = bar_w * pct / 100.0
                fig.set_fill(color)
                fig.rect(cx, y, w, bar_h)
                if w >= 13:
                    rgb = (0.1, 0.1, 0.1) if color == ORANGE else (1, 1, 1)
                    fig.text(cx + w / 2, y + bar_h / 2 - 3, f"{pct:.0f}", 7.4, "center", rgb=rgb, bold=True)
                cx += w
            fig.set_stroke((0.20, 0.20, 0.20), 0.6)
            fig.rect(bx, y, bar_w, bar_h, fill=False)
            fig.text(bx + bar_w + 6, y + bar_h / 2 - 3, f"n={n}", 7.3, "left", rgb=(0.32, 0.32, 0.32))

    lx, ly = 214, 16
    for seg_label, color in LEGEND:
        fig.set_fill(color)
        fig.rect(lx, ly, 11, 11)
        fig.text(lx + 15, ly + 2, seg_label, 8.6)
        lx += 110
    fig.save(OUTPUT_PDF)

    print(f"Wrote {OUTPUT_PDF}")
    for name, subdir, fname in MODELS:
        data = json.load(open(RESULTS / subdir / fname))
        for sub, n, segs in rows_for(data):
            labels = ["success", "drop/slip", "timeout"] if sub != "GRASP" else ["success", "timeout"]
            print(f"  {name:20} {sub:6} n={n:4} " + "  ".join(f"{l} {p:.1f}%" for l, (p, _) in zip(labels, segs)))


if __name__ == "__main__":
    main()
