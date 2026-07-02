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

# Four subtask models in a 2x2 grid: rows = training source, columns = standard
# fine-tuning vs LM-tuned. Scene states are pooled (0+1), so each model is one panel.
MODELS = [
    ("Teleop standard",      "Gal-pick-orange-tailedCH20",              "checkpoint.json"),
    ("Teleop LM-tuned",      "Gal-pick-orange-tailedCH20-unfrozen-vlm", "checkpoint.json"),
    ("Teleop+Auto standard", "Gal-merged-tailed-auto",                  "checkpoint.json"),
    ("Teleop+Auto LM-tuned", "Gal-merged-tailed-auto-unfrozen-vlm",     "checkpoint.json"),
]
# Fixed column order shared by every panel so they read consistently.
COLS = ["left", "middle", "right", "top right", "bottom right"]
SHORT = {"left": "Left", "middle": "Mid", "right": "Right", "top right": "T.R", "bottom right": "B.R"}

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


def confusion(subdir: str, fname: str, n_placed):
    """Requested-vs-grasped counts. n_placed: an int for one scene state, or an
    iterable of states to pool (e.g. (0, 1))."""
    states = (n_placed,) if isinstance(n_placed, int) else tuple(n_placed)
    data = json.load(open(RESULTS / subdir / fname))
    conf: dict[str, Counter] = defaultdict(Counter)
    for e in data["episodes"]:
        for a in e["subtask_attempts"]:
            if a.get("subtask") != "GRASP" or a.get("result") != "success":
                continue
            if a.get("n_placed_start") not in states:
                continue
            req = a["requested_label"]
            if a.get("target_match"):
                conf[req][req] += 1            # obeyed: anchored on identity
            else:
                conf[req][grasp_time_labels(e, a).get(a["actual_orange"], "?")] += 1
    return conf


CW, CH = 48.0, 30.0          # cell size
LABEL_W = 92.0               # row-label gutter
INK = (0.16, 0.17, 0.20)
MUTE = (0.42, 0.44, 0.48)
GRID = (0.88, 0.88, 0.90)
OBEY_TARGET = (0.20, 0.59, 0.36)   # green, for the diagonal (obeyed)
MISS_TARGET = (0.86, 0.42, 0.30)   # warm red, for off-diagonal (misgrab)


def ramp(target, frac):
    frac = max(0.0, min(1.0, frac))
    return tuple(1 - (1 - t) * frac for t in target)


def draw_panel(fig: PdfFigure, x0: float, top: float, state_label: str, conf: dict) -> None:
    grid_left = x0 + LABEL_W
    rows = [r for r in COLS if r in conf]
    gh = len(rows) * CH
    grid_top = top - 16

    # state tag, with overall obeyed rate for the panel
    tot = sum(sum(conf[r].values()) for r in rows)
    cor = sum(conf[r][r] for r in rows)
    fig.set_fill((0.95, 0.95, 0.97))
    fig.rect(x0, top + 2, LABEL_W + len(COLS) * CW, 15)
    fig.text(x0 + 6, top + 6, state_label, 8.6, "left", bold=True, rgb=INK)
    fig.text(grid_left + len(COLS) * CW, top + 6, f"{100 * cor / tot:.0f}% obeyed overall", 8.2, "right", bold=True, rgb=OBEY_TARGET)

    # column headers
    for j, c in enumerate(COLS):
        fig.text(grid_left + j * CW + CW / 2, grid_top + 5, SHORT[c], 8.2, "center", bold=True, rgb=MUTE)

    # cells
    for i, r in enumerate(rows):
        total = sum(conf[r].values())
        cy = grid_top - (i + 1) * CH
        for j, c in enumerate(COLS):
            v = conf[r].get(c, 0)
            frac = v / total if total else 0.0
            cx = grid_left + j * CW
            on_diag = (c == r)
            fig.set_fill(ramp(OBEY_TARGET if on_diag else MISS_TARGET, frac) if v else (0.975, 0.975, 0.978))
            fig.rect(cx, cy, CW, CH)
            if v:
                light = frac > 0.5
                fig.text(cx + CW / 2, cy + CH / 2 - 3.5, str(v), 9.5, "center",
                         rgb=(1, 1, 1) if light else INK, bold=on_diag)
        # row label + obeyed rate
        fig.text(grid_left - 7, cy + CH / 2 - 1, r.capitalize(), 8.6, "right", bold=True, rgb=INK)
        fig.text(grid_left - 7, cy + CH / 2 - 10, f"{100 * conf[r][r] / total:.0f}% obeyed  (n={total})", 6.3, "right", rgb=MUTE)

    # thin gridlines + frame
    fig.set_stroke(GRID, 0.5)
    for j in range(len(COLS) + 1):
        fig.line(grid_left + j * CW, grid_top, grid_left + j * CW, grid_top - gh)
    for i in range(len(rows) + 1):
        fig.line(grid_left, grid_top - i * CH, grid_left + len(COLS) * CW, grid_top - i * CH)
    fig.set_stroke(INK, 1.0)
    fig.rect(grid_left, grid_top - gh, len(COLS) * CW, gh, fill=False)


def swatch(fig, x, y, target, label):
    fig.set_fill(ramp(target, 0.85))
    fig.rect(x, y, 13, 13)
    fig.set_stroke(INK, 0.6)
    fig.rect(x, y, 13, 13, fill=False)
    fig.text(x + 19, y + 3, label, 8.6, "left", rgb=INK)


def main() -> None:
    fig = PdfFigure(width=792, height=508)
    grid_w = LABEL_W + len(COLS) * CW
    col_x = [44, 44 + grid_w + 60]
    center = [col_x[0] + LABEL_W + len(COLS) * CW / 2, col_x[1] + LABEL_W + len(COLS) * CW / 2]
    content_cx = (col_x[0] + col_x[1] + grid_w) / 2  # centre titles on the panel grid

    fig.text(content_cx, fig.height - 28, "Which orange is grasped vs. which was requested", 15, "center", bold=True, rgb=INK)
    fig.text(content_cx, fig.height - 45,
             "Rows: requested position.   Columns: position grasped.   Cell shade = share of that row.   Scene states 0--1 pooled.",
             8.8, "center", rgb=MUTE)

    # column super-headers: standard fine-tuning vs LM-tuned
    fig.text(center[0], fig.height - 67, "Standard fine-tuning", 11.5, "center", bold=True, rgb=INK)
    fig.text(center[1], fig.height - 67, "LM-tuned", 11.5, "center", bold=True, rgb=INK)

    row_top = [fig.height - 92, fig.height - 300]
    for i, (disp, subdir, fname) in enumerate(MODELS):
        r, c = i // 2, i % 2
        source = "Teleop+Auto" if disp.startswith("Teleop+Auto") else "Teleop"
        draw_panel(fig, col_x[c], row_top[r], source, confusion(subdir, fname, (0, 1)))

    swatch(fig, col_x[0], 30, OBEY_TARGET, "Obeyed (grasped the requested orange)")
    swatch(fig, col_x[0] + 250, 30, MISS_TARGET, "Misgrab (grasped a different orange)")
    fig.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
