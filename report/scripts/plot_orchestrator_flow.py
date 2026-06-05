"""Generate the orchestrator decision-flow diagram for the report."""

from __future__ import annotations

from pathlib import Path

from plot_lib import PdfFigure


REPORT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PDF = REPORT_DIR / "figures" / "orchestrator_decision_flow.pdf"


INK = (0.13, 0.13, 0.13)
MUTED = (0.35, 0.35, 0.35)
CONTROLLER = (0.88, 0.93, 0.98)
VLA = (0.88, 0.96, 0.89)
MOTION = (0.99, 0.95, 0.78)
ROUTING = (0.98, 0.88, 0.86)
TERMINAL = (0.91, 0.92, 0.95)
GRAY = (0.95, 0.95, 0.93)
LABEL_BG = (0.985, 0.985, 0.97)


def box(
    fig: PdfFigure,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    *,
    fill: tuple[float, float, float],
    stroke: tuple[float, float, float] = INK,
) -> None:
    fig.set_fill(fill)
    fig.rect(x, y, w, h)
    fig.set_stroke(stroke, 1.0)
    fig.rect(x, y, w, h, fill=False)
    fig.text(x + w / 2, y + h - 19, title, 9.2, "center", rgb=INK, bold=True)
    line_y = y + h - 37
    for line in lines:
        fig.text(x + w / 2, line_y, line, 7.0, "center", rgb=MUTED)
        line_y -= 12


def label_box(
    fig: PdfFigure,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 7.2,
    fill: tuple[float, float, float] = LABEL_BG,
) -> None:
    width = len(text) * size * 0.58 + 12
    height = size + 7
    fig.set_fill(fill)
    fig.rect(x - width / 2, y - 3, width, height)
    fig.set_stroke((0.78, 0.78, 0.74), 0.35)
    fig.rect(x - width / 2, y - 3, width, height, fill=False)
    fig.text(x, y + 1.5, text, size, "center", rgb=MUTED)


def legend_item(
    fig: PdfFigure,
    x: float,
    y: float,
    fill: tuple[float, float, float],
    text: str,
) -> None:
    fig.set_fill(fill)
    fig.rect(x, y, 12, 12)
    fig.set_stroke((0.28, 0.28, 0.28), 0.55)
    fig.rect(x, y, 12, 12, fill=False)
    fig.text(x + 18, y + 2, text, 7.6, "left", rgb=MUTED)


def arrowhead(fig: PdfFigure, x: float, y: float, direction: str) -> None:
    if direction == "right":
        fig.line(x, y, x - 8, y + 4)
        fig.line(x, y, x - 8, y - 4)
    elif direction == "left":
        fig.line(x, y, x + 8, y + 4)
        fig.line(x, y, x + 8, y - 4)
    elif direction == "up":
        fig.line(x, y, x - 4, y - 8)
        fig.line(x, y, x + 4, y - 8)
    elif direction == "down":
        fig.line(x, y, x - 4, y + 8)
        fig.line(x, y, x + 4, y + 8)


def arrow(
    fig: PdfFigure,
    points: list[tuple[float, float]],
    *,
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
    dashed: bool = False,
    stroke: tuple[float, float, float] = INK,
) -> None:
    fig.set_stroke(stroke, 1.0)
    if dashed:
        fig.commands.append("[5 4] 0 d")
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        fig.line(x1, y1, x2, y2)
    (x1, y1), (x2, y2) = points[-2], points[-1]
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = "right" if x2 > x1 else "left"
    else:
        direction = "up" if y2 > y1 else "down"
    arrowhead(fig, x2, y2, direction)
    if dashed:
        fig.commands.append("[] 0 d")
    if label and label_xy:
        label_box(fig, label_xy[0], label_xy[1], label)


def main() -> None:
    fig = PdfFigure(width=1000, height=660)

    fig.text(500, 632, "Orchestrator decision logic", 16, "center", bold=True)
    fig.text(
        500,
        612,
        "The VLA executes only GRASP(target), LIFT, and PLACE; the controller sequences prompts from privileged simulator state.",
        9.2,
        "center",
        rgb=MUTED,
    )

    legend_item(fig, 90, 574, CONTROLLER, "controller decision")
    legend_item(fig, 250, 574, VLA, "prompted VLA task")
    legend_item(fig, 410, 574, MOTION, "controller motion")
    legend_item(fig, 565, 574, ROUTING, "retry routing")
    legend_item(fig, 710, 574, TERMINAL, "terminal state")

    select = (58, 432, 160, 92)
    grasp = (278, 432, 160, 92)
    lift = (494, 432, 160, 92)
    place = (710, 432, 160, 92)
    finish = (906, 432, 88, 92)
    reset = (270, 252, 205, 92)
    retry = (590, 252, 230, 92)

    box(
        fig,
        *select,
        "SELECT_TARGET",
        ["choose unplaced orange", "assign left/middle/right", "avoid last timeout if possible"],
        fill=CONTROLLER,
    )
    box(
        fig,
        *grasp,
        "GRASP(target)",
        ["prompt: Grasp label", "confirm contact", "timeout: 700 steps"],
        fill=VLA,
    )
    box(
        fig,
        *lift,
        "LIFT",
        ["prompt: Pick it up", "confirm held + height", "timeout: 400 steps"],
        fill=VLA,
    )
    box(
        fig,
        *place,
        "PLACE",
        ["prompt: Place in plate", "confirm plate release", "timeout: 500 steps"],
        fill=VLA,
    )
    box(
        fig,
        *finish,
        "END",
        ["3 oranges", "in plate", "log count"],
        fill=TERMINAL,
    )
    box(
        fig,
        *reset,
        "HOME-POSE RESET",
        ["bypass VLA", "40 steps shoulder_lift", "60 steps all joints", "clear policy queue"],
        fill=MOTION,
    )
    box(
        fig,
        *retry,
        "LOCAL RETRY",
        ["same target preserved", "return directly to GRASP", "no home reset"],
        fill=ROUTING,
    )

    arrow(fig, [(218, 478), (278, 478)], label="target", label_xy=(249, 493))
    arrow(fig, [(438, 478), (494, 478)], label="ok", label_xy=(467, 493))
    arrow(fig, [(654, 478), (710, 478)], label="ok", label_xy=(682, 493))
    arrow(fig, [(870, 478), (906, 478)], label="all 3", label_xy=(888, 493))

    arrow(
        fig,
        [(790, 524), (790, 550), (138, 550), (138, 524)],
        label="place confirmed; more oranges remain",
        label_xy=(462, 558),
    )

    arrow(
        fig,
        [(358, 432), (358, 344)],
        label="GRASP timeout",
        label_xy=(358, 391),
        dashed=True,
    )
    arrow(
        fig,
        [(270, 298), (138, 298), (138, 432)],
        dashed=True,
    )
    label_box(fig, 170, 320, "reselect target")
    label_box(fig, 170, 303, "avoid failed orange if possible")

    arrow(
        fig,
        [(574, 432), (574, 382), (642, 382), (642, 344)],
        dashed=True,
    )
    arrow(
        fig,
        [(790, 432), (790, 382), (770, 382), (770, 344)],
        dashed=True,
    )
    label_box(fig, 700, 390, "drop or timeout in LIFT/PLACE")
    arrow(
        fig,
        [(590, 298), (500, 298), (500, 372), (398, 372), (398, 432)],
        label="same target, direct retry",
        label_xy=(502, 384),
        dashed=True,
    )

    fig.set_fill(GRAY)
    fig.rect(58, 68, 882, 120)
    fig.set_stroke((0.64, 0.64, 0.61), 0.8)
    fig.rect(58, 68, 882, 120, fill=False)
    fig.text(76, 166, "Physical success checks", 9.0, "left", rgb=INK, bold=True)
    footer_lines = [
        "GRASP: centred closed grip plus projected contact force on both fingertips for 10 consecutive frames.",
        "LIFT: active orange remains held and rises at least 0.06 m above its initial height for 10 consecutive frames.",
        "PLACE: active orange is inside the oriented plate bounds, stable for 10 frames, and released with the gripper open.",
        "A GRASP timeout triggers a home-pose reset before reselection; LIFT/PLACE failures preserve the same target.",
    ]
    y = 148
    for line in footer_lines:
        fig.text(76, y, line, 7.8, "left", rgb=MUTED)
        y -= 22

    fig.save(OUTPUT_PDF)
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
