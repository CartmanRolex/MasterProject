"""Generate the orchestrator decision-flow diagram for the report.

Rendered with matplotlib (FancyBboxPatch boxes + FancyArrowPatch connectors)
for rounded boxes, soft shadows, and filled arrowheads.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPORT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PDF = REPORT_DIR / "figures" / "orchestrator_decision_flow.pdf"

plt.rcParams["font.family"] = "DejaVu Sans"

INK = "#1f2330"
MUTED = "#5a606e"

# (fill, edge/accent) per category.
CONTROLLER = ("#e8f0fc", "#2f6fd0")
VLA = ("#e6f6ec", "#2f9e57")
MOTION = ("#fdf3d6", "#d39b1b")
ROUTING = ("#fce8e4", "#cf5340")
TERMINAL = ("#eceef5", "#5b6276")

ROUTE_COLOR = "#cf5340"  # dashed retry/redirection routing
FLOW_COLOR = "#3a3f4c"   # solid main flow


def add_box(ax, x, y, w, h, title, lines, fill, accent, *, title_size=11):
    """Draw a rounded box with a soft drop shadow, accent border and divider."""
    cx, cy = x + w / 2, y + h / 2
    # Soft shadow.
    shadow = FancyBboxPatch(
        (x + 1.6, y - 1.6), w, h,
        boxstyle="round,pad=0,rounding_size=8",
        linewidth=0, facecolor="#11141c", alpha=0.10, zorder=1,
        mutation_aspect=1,
    )
    ax.add_patch(shadow)
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=8",
        linewidth=1.6, edgecolor=accent, facecolor=fill, zorder=2,
        mutation_aspect=1,
    )
    ax.add_patch(box)
    # Title.
    title_y = y + h - 16
    ax.text(cx, title_y, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=INK, zorder=4)
    # Accent divider under title.
    div_y = title_y - 11
    ax.plot([x + 14, x + w - 14], [div_y, div_y], color=accent,
            lw=1.1, alpha=0.55, zorder=3, solid_capstyle="round")
    # Body lines.
    line_y = div_y - 13
    for ln in lines:
        ax.text(cx, line_y, ln, ha="center", va="center",
                fontsize=8.0, color=MUTED, zorder=4)
        line_y -= 12.5


def add_arrow(ax, posA, posB, *, color, dashed=False, rad=0.0, lw=1.7, z=2):
    style = "arc3,rad=%.3f" % rad
    arrow = FancyArrowPatch(
        posA, posB,
        connectionstyle=style,
        arrowstyle="-|>", mutation_scale=15,
        linewidth=lw, color=color,
        linestyle=(0, (5, 4)) if dashed else "solid",
        shrinkA=2, shrinkB=2, zorder=z,
        capstyle="round", joinstyle="round",
    )
    ax.add_patch(arrow)


def add_label(ax, x, y, text, *, color=MUTED):
    ax.text(
        x, y, text, ha="center", va="center", fontsize=7.6, color=color,
        zorder=5,
        bbox=dict(boxstyle="round,pad=0.32", fc="#ffffff", ec="#d7dae2", lw=0.7),
    )


def main() -> None:
    W, H = 1000.0, 700.0
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=200)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_aspect("equal")

    # Title block.
    ax.text(W / 2, 678, "Orchestrator decision logic", ha="center", va="center",
            fontsize=17, fontweight="bold", color=INK)
    ax.text(
        W / 2, 656,
        "The VLA executes only GRASP(target), LIFT and PLACE; the controller sequences prompts from privileged simulator state.",
        ha="center", va="center", fontsize=9.5, color=MUTED,
    )

    # Legend.
    legend = [
        (CONTROLLER, "controller decision"),
        (VLA, "prompted VLA task"),
        (MOTION, "controller motion"),
        (ROUTING, "retry routing"),
        (TERMINAL, "terminal state"),
    ]
    lx = 96
    for (fill, accent), text in legend:
        chip = FancyBboxPatch(
            (lx, 626), 14, 14,
            boxstyle="round,pad=0,rounding_size=3",
            linewidth=1.1, edgecolor=accent, facecolor=fill, zorder=3,
        )
        ax.add_patch(chip)
        ax.text(lx + 20, 633, text, ha="left", va="center", fontsize=8.2, color=MUTED)
        lx += 22 + len(text) * 5.6 + 26

    # Box geometry: (x, y, w, h).
    select = (58, 432, 168, 96)
    grasp = (286, 432, 168, 96)
    lift = (514, 432, 168, 96)
    place = (742, 432, 168, 96)
    finish = (926, 446, 70, 70)
    reset = (272, 250, 210, 96)
    retry = (596, 250, 232, 96)

    add_box(ax, *select, "SELECT_TARGET",
            ["choose unplaced orange", "assign left / middle / right",
             "avoid last timeout if possible"], *CONTROLLER)
    add_box(ax, *grasp, "GRASP(target)",
            ["prompt: “Grasp <label>”", "confirm contact",
             "timeout: 700 steps"], *VLA)
    add_box(ax, *lift, "LIFT",
            ["prompt: “Pick it up”", "confirm held + height",
             "timeout: 400 steps"], *VLA)
    add_box(ax, *place, "PLACE",
            ["prompt: “Place in plate”", "confirm plate release",
             "timeout: 500 steps"], *VLA)
    add_box(ax, *finish, "END", ["3 oranges", "in plate"], *TERMINAL, title_size=10)
    add_box(ax, *reset, "HOME-POSE RESET",
            ["bypass VLA", "40 steps shoulder-lift",
             "60 steps all joints", "clear policy queue"], *MOTION)
    add_box(ax, *retry, "LOCAL RETRY",
            ["same target preserved", "return directly to GRASP",
             "no home reset"], *ROUTING)

    def right(b):
        return (b[0] + b[2], b[1] + b[3] / 2)

    def left(b):
        return (b[0], b[1] + b[3] / 2)

    def top(b, fx=0.5):
        return (b[0] + b[2] * fx, b[1] + b[3])

    def bottom(b, fx=0.5):
        return (b[0] + b[2] * fx, b[1])

    # Main left-to-right flow.
    add_arrow(ax, right(select), left(grasp), color=FLOW_COLOR)
    add_label(ax, 256, 480, "target")
    add_arrow(ax, right(grasp), left(lift), color=FLOW_COLOR)
    add_label(ax, 484, 480, "ok")
    add_arrow(ax, right(lift), left(place), color=FLOW_COLOR)
    add_label(ax, 712, 480, "ok")
    add_arrow(ax, right(place), left(finish), color=FLOW_COLOR)
    add_label(ax, 918, 500, "all 3")

    # Loop back from PLACE to SELECT_TARGET (more oranges remain).
    arrow = FancyArrowPatch(
        top(place), top(select),
        connectionstyle="arc3,rad=0.16",
        arrowstyle="-|>", mutation_scale=15, linewidth=1.7,
        color=FLOW_COLOR, shrinkA=2, shrinkB=2, zorder=2,
        capstyle="round", joinstyle="round",
    )
    ax.add_patch(arrow)
    # Label sits on the arc; its white bbox cleanly masks the line behind it.
    add_label(ax, 484, 600, "place confirmed; more oranges remain", color=MUTED)

    # GRASP timeout -> home-pose reset.
    add_arrow(ax, bottom(grasp, 0.45), top(reset, 0.7),
              color=ROUTE_COLOR, dashed=True, rad=-0.12)
    add_label(ax, 318, 392, "GRASP timeout")
    # Home reset -> reselect target.
    add_arrow(ax, left(reset), bottom(select, 0.5),
              color=ROUTE_COLOR, dashed=True, rad=-0.30)
    add_label(ax, 175, 320, "reselect target")
    add_label(ax, 175, 300, "avoid failed orange if possible")

    # LIFT / PLACE drop or timeout -> local retry.
    add_arrow(ax, bottom(lift, 0.5), top(retry, 0.25),
              color=ROUTE_COLOR, dashed=True, rad=0.12)
    add_arrow(ax, bottom(place, 0.5), top(retry, 0.75),
              color=ROUTE_COLOR, dashed=True, rad=-0.12)
    add_label(ax, 700, 392, "drop or timeout in LIFT / PLACE")
    # Local retry -> back to GRASP, same target. Routed up-left through the
    # open corridor between HOME-POSE RESET and LOCAL RETRY (avoids overlap).
    add_arrow(ax, top(retry, 0.0), bottom(grasp, 0.95),
              color=ROUTE_COLOR, dashed=True, rad=0.0)
    add_label(ax, 476, 360, "same target, direct retry")

    # Footer panel: physical success checks.
    footer = FancyBboxPatch(
        (58, 60), 882, 132,
        boxstyle="round,pad=0,rounding_size=8",
        linewidth=1.0, edgecolor="#c9ccd4", facecolor="#f6f7f9", zorder=1,
    )
    ax.add_patch(footer)
    ax.text(78, 172, "Physical success checks", ha="left", va="center",
            fontsize=10, fontweight="bold", color=INK)
    footer_lines = [
        "GRASP: centred closed grip plus projected contact force on both fingertips for 10 consecutive frames.",
        "LIFT: active orange remains held and rises at least 0.06 m above its initial height for 10 consecutive frames.",
        "PLACE: active orange is inside the oriented plate bounds, stable for 10 frames, and released with the gripper open.",
        "A GRASP timeout triggers a home-pose reset before reselection; LIFT / PLACE failures preserve the same target.",
    ]
    fy = 150
    for ln in footer_lines:
        ax.text(78, fy, ln, ha="left", va="center", fontsize=8.4, color=MUTED)
        fy -= 23

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
