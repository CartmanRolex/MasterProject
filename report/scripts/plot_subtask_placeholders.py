"""Generate placeholder subtask illustrations for the report."""

from __future__ import annotations

from pathlib import Path

from plot_lib import PdfFigure


REPORT_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPORT_DIR / "figures"

INK = (0.16, 0.16, 0.16)
MUTED = (0.42, 0.42, 0.42)
BG = (0.96, 0.96, 0.94)
ACCENT = (0.88, 0.93, 0.98)


def draw_placeholder(filename: str, title: str, prompt: str, state: str) -> None:
    fig = PdfFigure(width=360, height=235)
    fig.set_fill(BG)
    fig.rect(0, 0, 360, 235)

    fig.set_fill(ACCENT)
    fig.rect(16, 16, 328, 203)
    fig.set_stroke((0.20, 0.20, 0.20), 1.1)
    fig.rect(16, 16, 328, 203, fill=False)

    fig.commands.append("[6 5] 0 d")
    fig.set_stroke((0.55, 0.55, 0.52), 0.9)
    fig.rect(32, 38, 296, 122, fill=False)
    fig.commands.append("[] 0 d")

    fig.text(180, 190, title, 17.0, "center", rgb=INK, bold=True)
    fig.text(180, 144, "IMAGE PLACEHOLDER", 10.0, "center", rgb=MUTED, bold=True)
    fig.text(180, 116, prompt, 8.8, "center", rgb=MUTED)
    fig.text(180, 94, state, 8.8, "center", rgb=MUTED)
    fig.text(180, 58, "Replace this file with a representative frame", 8.2, "center", rgb=MUTED)

    fig.save(FIGURE_DIR / filename)
    print(f"Wrote {FIGURE_DIR / filename}")


def main() -> None:
    draw_placeholder(
        "subtask_grasp_placeholder.pdf",
        "GRASP",
        "Prompt example: Grasp top right orange",
        "Expected result: gripper closes on target",
    )
    draw_placeholder(
        "subtask_lift_placeholder.pdf",
        "LIFT",
        "Prompt: Pick it up",
        "Expected result: held orange rises",
    )
    draw_placeholder(
        "subtask_place_placeholder.pdf",
        "PLACE",
        "Prompt: Place it into plate",
        "Expected result: orange released in plate",
    )


if __name__ == "__main__":
    main()
