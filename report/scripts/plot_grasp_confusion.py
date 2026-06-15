"""Grasp-obedience confusion heatmaps (figures/grasp_obedience_confusion.pdf).

Reads the two orchestrated subtask eval checkpoints under
``isaac-inference/results`` (git-tracked, so this runs on the laptop) and draws a
2x2 grid of confusion matrices: rows = the requested grasp position, columns =
the position actually grasped (by orange identity), for both models and for the
two informative scene states (0 and 1 oranges already placed). The 2-placed
state is omitted (one orange remains, trivially obeyed).

Two subtleties are handled so the matrices are faithful:

* Correctness is anchored on orange identity (``target_match``), not on the
  position label, so the diagonal equals the obeyed rate exactly.
* The position labels are recomputed for the scene *at grasp time*: the two
  oranges still on the table sit at their start positions while the placed
  orange is moved to the plate, then ``classify`` is re-run. Using the
  start-of-episode labels would be wrong once an orange has been placed.

Pure stdlib + the shared ``plot_lib`` PDF writer (no matplotlib).

    python plot_grasp_confusion.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from plot_lib import PdfFigure

RESULTS = Path(__file__).resolve().parents[2] / "isaac-inference" / "results"
OUT = Path(__file__).resolve().parents[1] / "figures" / "grasp_obedience_confusion.pdf"

MODELS = [
    ("Teleop", "Gal-pick-orange-tailedCH20", "checkpoint.json"),
    ("Teleop+Auto", "Gal-merged-tailed-auto", "checkpoint.json"),
]
# Fixed column order shared by every panel so they read consistently.
COLS = ["left", "middle", "right", "top right", "bottom right"]
SHORT = {"left": "left", "middle": "mid", "right": "right", "top right": "t.R", "bottom right": "b.R"}

# classify_orange_positions constants (see eval_utils.py): x primary, both inverts on, 3 cm tol.
TOL = 0.03


def classify(pos: dict) -> dict:
    if len(pos) != 3:
        return {n: n for n in pos}
    vals = {n: p[0] for n, p in pos.items()}
    sec = {n: p[1] for n, p in pos.items()}
    n0, n1, n2 = sorted(vals, key=lambda n: vals[n])[::-1]
    c01 = abs(vals[n0] - vals[n1]) < TOL
    c12 = abs(vals[n1] - vals[n2]) < TOL

    def split(a, b):
        lo, hi = (a, b) if sec[a] <= sec[b] else (b, a)
        return hi, lo  # SECONDARY_INVERT

    if c01 and c12:
        by = sorted([n0, n1, n2], key=lambda n: sec[n])[::-1]
        sl = {by[0]: "bottom", by[1]: "middle", by[2]: "top"}
        pl = {n0: "left", n1: "middle", n2: "right"}
        out = {}
        for n in (n0, n1, n2):
            s, p = sl[n], pl[n]
            out[n] = "middle" if (s == "middle" and p == "middle") else p if s == "middle" else s if p == "middle" else f"{s} {p}"
        return out
    if c01:
        b, t = split(n0, n1)
        return {b: "bottom left", t: "top left", n2: "right"}
    if c12:
        b, t = split(n1, n2)
        return {n0: "left", b: "bottom right", t: "top right"}
    return {n0: "left", n1: "middle", n2: "right"}


def grasp_time_labels(episode: dict, attempt: dict) -> dict:
    """Position labels for the scene as it stood when this grasp was attempted."""
    oranges = episode["initial_scene"]["oranges"]
    scene = {n: list(o["position"]) for n, o in oranges.items()}
    plate = episode["initial_scene"]["plate_position"]
    placed: list[str] = []
    for ev in episode.get("timeline", []):
        if ev.get("event_type") == "place_success" and ev["step"] <= attempt["start_step"]:
            o = ev.get("actual_orange") or ev.get("requested_orange")
            if o and o not in placed:
                placed.append(o)
    for po in placed[: attempt.get("n_placed_start", 0)]:
        scene[po] = plate
    return classify(scene)


def confusion(subdir: str, fname: str, n_placed: int):
    data = json.load(open(RESULTS / subdir / fname))
    conf: dict[str, Counter] = defaultdict(Counter)
    for e in data["episodes"]:
        for a in e["subtask_attempts"]:
            if a.get("subtask") != "GRASP" or a.get("result") != "success":
                continue
            if a.get("n_placed_start") != n_placed:
                continue
            req = a["requested_label"]
            if a.get("target_match"):
                conf[req][req] += 1            # obeyed: anchored on identity
            else:
                conf[req][grasp_time_labels(e, a).get(a["actual_orange"], "?")] += 1
    return conf


def cell_color(frac: float):
    return (1 - 0.80 * frac, 1 - 0.55 * frac, 1 - 0.28 * frac)  # white -> blue


def draw_panel(fig: PdfFigure, x0: float, y0: float, title: str, conf: dict) -> None:
    cw, ch = 42.0, 22.0
    grid_left = x0 + 104
    header_y = y0 - 26
    grid_top = header_y - 8
    rows = [r for r in COLS if r in conf]  # requested positions present, in canonical order

    fig.text(x0, y0 - 2, title, 10.5, "left", bold=True)
    fig.text(grid_left + len(COLS) * cw / 2, header_y + 12, "grasped", 7.6, "center", rgb=(0.30, 0.30, 0.30))
    for j, c in enumerate(COLS):
        fig.text(grid_left + j * cw + cw / 2, header_y, SHORT[c], 7.8, "center", bold=True)

    for i, r in enumerate(rows):
        total = sum(conf[r].values())
        correct = conf[r][r]
        cy = grid_top - i * ch - ch
        fig.text(grid_left - 8, cy + ch / 2 - 3, f"{r}", 8.0, "right", bold=True)
        fig.text(grid_left - 8, cy + ch / 2 - 11, f"{100 * correct / total:.0f}%  n={total}", 6.4, "right", rgb=(0.35, 0.35, 0.35))
        for j, c in enumerate(COLS):
            v = conf[r].get(c, 0)
            frac = v / total if total else 0
            cx = grid_left + j * cw
            fig.set_fill(cell_color(frac))
            fig.rect(cx, cy, cw, ch)
            if c == r:  # diagonal = obeyed: green outline
                fig.set_stroke((0.16, 0.52, 0.24), 1.6)
                fig.rect(cx, cy, cw, ch, fill=False)
            if v:
                txt_rgb = (1, 1, 1) if frac > 0.55 else (0.10, 0.10, 0.10)
                fig.text(cx + cw / 2, cy + ch / 2 - 3, str(v), 8.2, "center", rgb=txt_rgb, bold=(c == r))
        fig.set_stroke((0.80, 0.80, 0.80), 0.4)
        for j in range(len(COLS) + 1):
            fig.line(grid_left + j * cw, grid_top, grid_left + j * cw, grid_top - len(rows) * ch)
        fig.line(grid_left, cy, grid_left + len(COLS) * cw, cy)
    fig.line(grid_left, grid_top, grid_left + len(COLS) * cw, grid_top)
    fig.text(grid_left - 104 + 8, grid_top - len(rows) * ch - 14, "rows: requested", 6.8, "left", rgb=(0.30, 0.30, 0.30))


def main() -> None:
    fig = PdfFigure(width=760, height=470)
    fig.text(fig.width / 2, fig.height - 24, "Grasp obedience: requested vs. actually grasped position", 14, "center", bold=True)
    fig.text(fig.width / 2, fig.height - 40, "Cell = grasps; green box = obeyed (diagonal); shade = row fraction. Left of each row: obeyed %, n.", 8.4, "center", rgb=(0.28, 0.28, 0.28))

    col_x = [40, 412]
    row_y = [fig.height - 70, fig.height - 285]
    states = [(0, "0 placed (all three on table)"), (1, "1 placed (two remaining)")]
    for ri, (n_placed, state_label) in enumerate(states):
        for ci, (disp, subdir, fname) in enumerate(MODELS):
            conf = confusion(subdir, fname, n_placed)
            draw_panel(fig, col_x[ci], row_y[ri], f"{disp} - {state_label}", conf)

    fig.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
