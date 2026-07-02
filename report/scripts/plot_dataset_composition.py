"""Episode-length distribution by subtask, split into Teleop and Auto sources.

Self-contained: the per-source x per-subtask statistics below were measured from
the merged subtask training set ``Gal-merged-tailed-auto`` with
``extract_dataset_composition.py`` (run on the desktop). They are baked in here
so the figure regenerates on the laptop without the datasets present. Lengths
include the 20-frame terminal hold tail appended to every subtask episode.

Re-run ``extract_dataset_composition.py`` and paste its output into ``STATS`` if
the training datasets change.
"""

from __future__ import annotations

from pathlib import Path

from plot_lib import PdfFigure


REPORT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PDF = REPORT_DIR / "figures" / "dataset_composition.pdf"

# Measured by extract_dataset_composition.py (Gal-merged-tailed-auto, 1260 episodes).
STATS = {
    "Teleop/GRASP": {"n": 281, "mean": 260.2, "p5": 82.0, "p25": 166.0, "p50": 240.0, "p75": 320.0, "p95": 551.0},
    "Teleop/LIFT":  {"n": 274, "mean": 93.8,  "p5": 58.6, "p25": 71.0,  "p50": 84.5,  "p75": 107.8, "p95": 158.7},
    "Teleop/PLACE": {"n": 282, "mean": 122.3, "p5": 65.0, "p25": 83.0,  "p50": 105.0, "p75": 136.8, "p95": 267.3},
    "Auto/GRASP":   {"n": 143, "mean": 200.7, "p5": 71.1, "p25": 149.5, "p50": 207.0, "p75": 251.0, "p95": 316.4},
    "Auto/LIFT":    {"n": 136, "mean": 86.8,  "p5": 63.8, "p25": 71.8,  "p50": 81.0,  "p75": 94.5,  "p95": 136.2},
    "Auto/PLACE":   {"n": 135, "mean": 85.9,  "p5": 61.0, "p25": 71.0,  "p50": 81.0,  "p75": 101.0, "p95": 114.0},
}

SUBTASKS = ("GRASP", "LIFT", "PLACE")
SOURCES = ("Teleop", "Auto")

INK = (0.16, 0.16, 0.16)
MUTED = (0.42, 0.42, 0.42)
GRID = (0.90, 0.90, 0.90)
BG = (1.0, 1.0, 1.0)
SOURCE_FILL = {"Teleop": (0.93, 0.56, 0.21), "Auto": (0.34, 0.63, 0.76)}
SOURCE_STROKE = {"Teleop": (0.58, 0.32, 0.10), "Auto": (0.16, 0.36, 0.46)}

# Canvas / plot-area geometry (points, origin bottom-left).
W, H = 470, 300
PLOT_L, PLOT_R, PLOT_B, PLOT_T = 54, 454, 52, 250
Y_MAX = 600.0
BOX_W = 30.0
PAIR_GAP = 8.0  # gap between the Teleop and Auto box within a subtask group


def y_of(value: float) -> float:
    return PLOT_B + (min(value, Y_MAX) / Y_MAX) * (PLOT_T - PLOT_B)


def draw_box(fig: PdfFigure, cx: float, stat: dict, source: str) -> None:
    fill = SOURCE_FILL[source]
    stroke = SOURCE_STROKE[source]
    x = cx - BOX_W / 2

    # Whisker: p5 -> p95 with end caps.
    fig.set_stroke(stroke, 0.9)
    fig.line(cx, y_of(stat["p5"]), cx, y_of(stat["p95"]))
    fig.line(cx - 6, y_of(stat["p5"]), cx + 6, y_of(stat["p5"]))
    fig.line(cx - 6, y_of(stat["p95"]), cx + 6, y_of(stat["p95"]))

    # Interquartile box.
    y25, y75 = y_of(stat["p25"]), y_of(stat["p75"])
    fig.set_fill(fill)
    fig.rect(x, y25, BOX_W, y75 - y25)
    fig.set_stroke(stroke, 1.1)
    fig.rect(x, y25, BOX_W, y75 - y25, fill=False)

    # Median line (bold) and mean marker (dashed).
    ymed = y_of(stat["p50"])
    fig.set_stroke((0.12, 0.12, 0.12), 1.6)
    fig.line(x, ymed, x + BOX_W, ymed)
    ymean = y_of(stat["mean"])
    fig.commands.append("[3 2] 0 d")
    fig.set_stroke((0.12, 0.12, 0.12), 0.9)
    fig.line(x, ymean, x + BOX_W, ymean)
    fig.commands.append("[] 0 d")

    # Per-box source tag + episode count.
    fig.text(cx, PLOT_B - 12, source, 7.5, "center", rgb=MUTED, bold=True)
    fig.text(cx, PLOT_B - 21, f"n={stat['n']}", 6.8, "center", rgb=MUTED)


def main() -> None:
    fig = PdfFigure(width=W, height=H)
    fig.set_fill(BG)
    fig.rect(0, 0, W, H)

    # Title.
    fig.text((PLOT_L + PLOT_R) / 2, H - 22, "Subtask episode-length distribution by data source",
             12.5, "center", rgb=INK, bold=True)

    # Y gridlines + labels.
    for tick in range(0, int(Y_MAX) + 1, 100):
        yt = y_of(tick)
        fig.set_stroke(GRID, 0.7)
        fig.line(PLOT_L, yt, PLOT_R, yt)
        fig.text(PLOT_L - 8, yt - 3, str(tick), 8.0, "right", rgb=MUTED)
    fig.text(PLOT_L - 30, PLOT_T + 6, "Frames", 8.0, "left", rgb=MUTED)

    # Plot border.
    fig.set_stroke((0.30, 0.30, 0.30), 1.0)
    fig.rect(PLOT_L, PLOT_B, PLOT_R - PLOT_L, PLOT_T - PLOT_B, fill=False)

    # Subtask groups.
    group_w = (PLOT_R - PLOT_L) / len(SUBTASKS)
    for gi, subtask in enumerate(SUBTASKS):
        group_cx = PLOT_L + (gi + 0.5) * group_w
        offset = (BOX_W + PAIR_GAP) / 2
        centers = {"Teleop": group_cx - offset, "Auto": group_cx + offset}
        for source in SOURCES:
            draw_box(fig, centers[source], STATS[f"{source}/{subtask}"], source)
        fig.text(group_cx, PLOT_B - 34, subtask, 10.0, "center", rgb=INK, bold=True)
        if gi:
            sep = PLOT_L + gi * group_w
            fig.set_stroke(GRID, 0.6)
            fig.line(sep, PLOT_B, sep, PLOT_T)

    # Legend.
    lx, ly = PLOT_L + 6, PLOT_T - 16
    for source in SOURCES:
        fig.set_fill(SOURCE_FILL[source])
        fig.rect(lx, ly, 12, 9)
        fig.set_stroke(SOURCE_STROKE[source], 1.0)
        fig.rect(lx, ly, 12, 9, fill=False)
        fig.text(lx + 17, ly + 1, source, 8.5, "left", rgb=INK)
        lx += 78

    fig.save(OUTPUT_PDF)
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
