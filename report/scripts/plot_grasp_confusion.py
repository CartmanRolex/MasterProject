"""Grasp-obedience confusion heatmaps (figures/grasp_obedience_confusion.pdf).

Reads the orchestrated subtask eval checkpoints under ``isaac-inference/results``
(git-tracked, so this runs on the laptop) and draws a 2x3 grid of confusion
matrices: rows = the requested grasp position, columns = the position actually
grasped (by orange identity). Grid rows = training source (Teleop, Teleop+Auto);
grid columns = fine-tuning recipe (standard, LM-tuned, fully-tuned). The
fully-tuned recipe exists for Teleop only, so the sixth cell holds the legend.
Scene states 0 and 1 are pooled; the 2-placed state is omitted (one orange
remains, trivially obeyed).

Two subtleties are handled so the matrices are faithful:

* Correctness is anchored on orange identity (``target_match``), not on the
  position label, so the diagonal equals the obeyed rate exactly.
* The position labels are recomputed for the scene *at grasp time*: the two
  oranges still on the table sit at their start positions while the placed
  orange is moved to the plate, then ``classify`` is re-run. Using the
  start-of-episode labels would be wrong once an orange has been placed.

Displayed rows are the five canonical labels; the handful of grasps whose
requested label comes from a rare near-diagonal scene layout (at most 4 per
model) are counted in each panel's overall obeyed rate but have no row.

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

# Grid: rows = training source, columns = fine-tuning recipe. The fully-tuned
# recipe (all weights trained, incl. the vision encoder) was run on Teleop only,
# so the Teleop+Auto row has two panels and the legend takes the last cell.
RECIPES = [
    ("Standard fine-tuning", "action expert + state projection"),
    ("LM-tuned", "+ language model"),
    ("Fully-tuned", "+ language model + vision encoder"),
]
GRID = [
    ("Teleop", ["Gal-pick-orange-tailedCH20",
                "Gal-pick-orange-tailedCH20-unfrozen-vlm",
                "Gal-pick-orange-tailedCH20-unfrozen-all"]),
    ("Teleop+Auto", ["Gal-merged-tailed-auto",
                     "Gal-merged-tailed-auto-unfrozen-vlm",
                     None]),
]
# Fixed row/column order shared by every panel so they read consistently.
COLS = ["left", "middle", "right", "top right", "bottom right"]
HEAD = {"left": ("Left",), "middle": ("Middle",), "right": ("Right",),
        "top right": ("Top", "right"), "bottom right": ("Bottom", "right")}

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


CW, CH = 44.0, 30.0          # cell size
LABEL_W = 78.0               # row-label gutter (leftmost panel only; rows are shared)
GRID_W = len(COLS) * CW
PANEL_GAP = 18.0
INK = (0.16, 0.17, 0.20)
MUTE = (0.42, 0.44, 0.48)
GRID_RGB = (0.88, 0.88, 0.90)
OBEY_TARGET = (0.20, 0.59, 0.36)   # green, for the diagonal (obeyed)
MISS_TARGET = (0.86, 0.42, 0.30)   # warm red, for off-diagonal (misgrab)


def ramp(target, frac):
    frac = max(0.0, min(1.0, frac))
    return tuple(1 - (1 - t) * frac for t in target)


def draw_col_headers(fig: PdfFigure, grid_left: float, grid_top: float) -> None:
    for j, c in enumerate(COLS):
        lines = HEAD[c]
        cx = grid_left + j * CW + CW / 2
        if len(lines) == 1:
            fig.text(cx, grid_top + 5, lines[0], 7.9, "center", bold=True, rgb=MUTE)
        else:
            fig.text(cx, grid_top + 13, lines[0], 7.9, "center", bold=True, rgb=MUTE)
            fig.text(cx, grid_top + 5, lines[1], 7.9, "center", bold=True, rgb=MUTE)


def draw_panel(fig: PdfFigure, grid_left: float, top: float, conf: dict, row_labels: bool) -> None:
    """One confusion matrix. ``top`` is the top edge of the tag bar; the grid
    starts 38 pt below it (tag bar + two-line column headers)."""
    grid_top = top - 38
    gh = len(COLS) * CH

    # tag bar: overall obeyed rate (over ALL grasps, matching compute_obedience)
    tot = sum(sum(c.values()) for c in conf.values())
    cor = sum(conf[r][r] for r in conf)
    fig.set_fill((0.94, 0.94, 0.96))
    fig.rect(grid_left, top - 14, GRID_W, 14)
    fig.text(grid_left + 5, top - 10, f"{100 * cor / tot:.0f}% obeyed", 8.6, "left", bold=True, rgb=OBEY_TARGET)
    fig.text(grid_left + GRID_W - 5, top - 10, f"n = {tot} grasps", 7.6, "right", rgb=MUTE)

    draw_col_headers(fig, grid_left, grid_top)

    for i, r in enumerate(COLS):
        row = conf.get(r, Counter())
        total = sum(row.values())
        cy = grid_top - (i + 1) * CH
        for j, c in enumerate(COLS):
            v = row.get(c, 0)
            frac = v / total if total else 0.0
            cx = grid_left + j * CW
            on_diag = (c == r)
            fig.set_fill(ramp(OBEY_TARGET if on_diag else MISS_TARGET, frac) if v else (0.975, 0.975, 0.978))
            fig.rect(cx, cy, CW, CH)
            if v:
                txt_rgb = (1, 1, 1) if frac > 0.55 else INK
                if on_diag:
                    fig.text(cx + CW / 2, cy + CH / 2 + 1, f"{100 * frac:.0f}", 9.5, "center", rgb=txt_rgb, bold=True)
                    fig.text(cx + CW / 2, cy + 4, f"n={total}", 5.6, "center", rgb=txt_rgb)
                else:
                    fig.text(cx + CW / 2, cy + CH / 2 - 3.5, f"{100 * frac:.0f}", 9.0, "center", rgb=txt_rgb)
        if row_labels:
            lines = HEAD[r]
            if len(lines) == 1:
                fig.text(grid_left - 7, cy + CH / 2 - 3, lines[0], 8.4, "right", bold=True, rgb=INK)
            else:
                fig.text(grid_left - 7, cy + CH / 2 + 1, lines[0], 8.4, "right", bold=True, rgb=INK)
                fig.text(grid_left - 7, cy + CH / 2 - 8, lines[1], 8.4, "right", bold=True, rgb=INK)

    # thin gridlines + frame
    fig.set_stroke(GRID_RGB, 0.5)
    for j in range(len(COLS) + 1):
        fig.line(grid_left + j * CW, grid_top, grid_left + j * CW, grid_top - gh)
    for i in range(len(COLS) + 1):
        fig.line(grid_left, grid_top - i * CH, grid_left + GRID_W, grid_top - i * CH)
    fig.set_stroke(INK, 1.0)
    fig.rect(grid_left, grid_top - gh, GRID_W, gh, fill=False)


def draw_legend(fig: PdfFigure, x0: float, top: float) -> None:
    """Legend in the unused sixth grid cell."""
    y = top - 52

    def swatch(target, label, sub):
        nonlocal y
        fig.set_fill(ramp(target, 0.85))
        fig.rect(x0, y, 13, 13)
        fig.set_stroke(INK, 0.6)
        fig.rect(x0, y, 13, 13, fill=False)
        fig.text(x0 + 19, y + 7, label, 8.4, "left", bold=True, rgb=INK)
        fig.text(x0 + 19, y - 2, sub, 7.6, "left", rgb=MUTE)
        y -= 32

    swatch(OBEY_TARGET, "Obeyed", "grasped the requested orange")
    swatch(MISS_TARGET, "Misgrab", "grasped a different orange")
    fig.text(x0, y + 4, "Cell value: % of the row's grasps;", 7.6, "left", rgb=MUTE)
    fig.text(x0, y - 5, "darker shade = larger share.", 7.6, "left", rgb=MUTE)
    y -= 28
    fig.text(x0, y + 4, "No fully-tuned run exists for", 7.6, "left", rgb=MUTE)
    fig.text(x0, y - 5, "Teleop+Auto.", 7.6, "left", rgb=MUTE)


def main() -> None:
    fig = PdfFigure(width=846, height=512)
    grid_x = [LABEL_W + 44 + k * (GRID_W + PANEL_GAP) for k in range(3)]
    content_cx = (grid_x[0] + grid_x[2] + GRID_W) / 2

    fig.text(content_cx, fig.height - 26, "Which orange is grasped vs. which was requested", 15, "center", bold=True, rgb=INK)
    fig.text(content_cx, fig.height - 43,
             "Rows: requested position.   Columns: position grasped.   Cells: % of the row's grasps.   Scene states 0-1 pooled.",
             8.8, "center", rgb=MUTE)

    # column super-headers: the three fine-tuning recipes
    for k, (name, sub) in enumerate(RECIPES):
        cx = grid_x[k] + GRID_W / 2
        fig.text(cx, fig.height - 64, name, 11.5, "center", bold=True, rgb=INK)
        fig.text(cx, fig.height - 76, sub, 7.6, "center", rgb=MUTE)

    row_top = [fig.height - 90, fig.height - 308]
    for r, (source, subdirs) in enumerate(GRID):
        # vertical source label, centred on the matrix
        grid_mid = row_top[r] - 38 - len(COLS) * CH / 2
        fig.vtext(30, grid_mid, source, 10.5, "center", bold=True, rgb=INK)
        for k, subdir in enumerate(subdirs):
            if subdir is None:
                draw_legend(fig, grid_x[k] + 12, row_top[r])
            else:
                draw_panel(fig, grid_x[k], row_top[r], confusion(subdir, "checkpoint.json", (0, 1)),
                           row_labels=(k == 0))

    fig.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
