"""Outcome-level recovery: after the first orange fails, does the system still place a
DIFFERENT one? (replaces the misleading "oranges tried" metric)

Definition (observational, ground truth — no attempt-counting, which is not comparable
across formulations):
  * "first orange" = the first orange the system commits to.
      - subtask : the first GRASP prompt (orchestrator log, `subtask_attempts`).
      - monotask: the first force-confirmed grasp (gripper closed + >=10 N on both tips,
                  reaching an orange) read from the v4 `geometry_trace`.
  * the first orange "failed" = it is not in the plate at the end.
  * "recovered" = a DIFFERENT orange ends in the plate.
  * recovery rate = recovered / (episodes whose first orange failed).

This is an OUTCOME comparison only: it shows the orchestrated formulation recovers far
more often, without claiming the recovery *logic* (vs better per-attempt grasping) is the
isolated cause.

  python plot_recovery_outcomes.py
"""

from __future__ import annotations

import json
from pathlib import Path

from plot_lib import PdfFigure

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "isaac-inference" / "results"
OUTPUT_PDF = ROOT / "report" / "figures" / "recovery_outcomes.pdf"

# (label, source, formulation, results-subdir, checkpoint file)
MODELS = [
    ("Teleop",      "monotask", "Gal_split_nolang",                       "flat_checkpoint.json"),
    ("Teleop",      "subtask",  "Gal-pick-orange-tailedCH20",             "checkpoint.json"),
    ("Teleop+Auto", "monotask", "Gal-merged-tailed-auto-no-lang-no-home", "flat_checkpoint.json"),
    ("Teleop+Auto", "subtask",  "Gal-merged-tailed-auto",                 "checkpoint.json"),
]

CLOSED, REACH, FORCE_MIN = 0.60, 0.03, 10.0  # monotask force-confirmed-grasp thresholds


def episodes(d):
    eps = d["episodes"]
    return list(eps.values()) if isinstance(eps, dict) else eps


def placed_set(e):
    return {n for n, o in (e.get("final_scene", {}).get("oranges", {}) or {}).items() if o.get("in_plate")}


def org(a):
    return a.get("requested_orange") or a.get("actual_orange") or a.get("inferred_target_orange")


def mono_first_engaged(e):
    gt = e.get("geometry_trace") or {}
    cols, rows = gt.get("columns", []), gt.get("rows", [])
    ax = {n[:-10]: i for i, n in enumerate(cols) if n.endswith(":axis_dist")}
    if not ax or not rows:
        return None
    gp, gf, jf = cols.index("gripper_pos"), cols.index("grip_force_n"), cols.index("jaw_force_n")
    oranges = list(ax)
    cur, run, forced, first = None, 0, False, None

    def flush():
        nonlocal cur, run, forced, first
        if cur is not None and run >= 2 and forced and first is None:
            first = cur
        cur, run, forced = None, 0, False

    for r in rows:
        best = min(((n, r[ax[n]]) for n in oranges), key=lambda t: t[1])
        if r[gp] < CLOSED and best[1] < REACH:
            if best[0] == cur:
                run += 1
            else:
                flush(); cur, run, forced = best[0], 1, False
            if min(r[gf], r[jf]) >= FORCE_MIN:
                forced = True
        else:
            flush()
        if first is not None:
            break
    flush()
    return first


def sub_first_target(e):
    g = sorted([a for a in e.get("subtask_attempts", []) if a.get("subtask") == "GRASP"],
               key=lambda a: a.get("start_step", 0))
    return org(g[0]) if g else None


def recovery_rate(subdir, fname, formulation):
    data = json.load(open(RESULTS / subdir / fname))
    first_failed = recovered = 0
    for e in episodes(data):
        placed = placed_set(e)
        first = mono_first_engaged(e) if formulation == "monotask" else sub_first_target(e)
        if first is None or first in placed:
            continue
        first_failed += 1
        if len(placed) >= 1:
            recovered += 1
    return recovered, first_failed


# ---- compute -----------------------------------------------------------------
FILL = {"monotask": (0.80, 0.45, 0.28), "subtask": (0.27, 0.55, 0.72)}
STROKE = {"monotask": (0.52, 0.27, 0.13), "subtask": (0.13, 0.33, 0.47)}
bars = []
for src, form, subdir, fname in MODELS:
    rec, ff = recovery_rate(subdir, fname, form)
    rate = 100 * rec / ff if ff else 0.0
    bars.append((src, form, rate, rec, ff))
    print(f"{src:11s} {form:8s}: recovered {rec}/{ff} = {rate:.0f}%")

# ---- draw --------------------------------------------------------------------
INK, MUTED, GRID = (0.16, 0.16, 0.16), (0.42, 0.42, 0.42), (0.90, 0.90, 0.90)
W, H = 460, 300
PL, PR, PB, PT = 54, 446, 56, 248
YMAX = 100.0


def y_of(v):
    return PB + (v / YMAX) * (PT - PB)


def main():
    fig = PdfFigure(width=W, height=H)
    fig.set_fill((1, 1, 1)); fig.rect(0, 0, W, H)
    fig.text(W / 2, H - 22, "Recovery after the first orange fails", 12.5, "center", rgb=INK, bold=True)

    for tick in range(0, 101, 25):
        yt = y_of(tick)
        fig.set_stroke(GRID, 0.7); fig.line(PL, yt, PR, yt)
        fig.text(PL - 8, yt - 3, f"{tick}", 8.0, "right", rgb=MUTED)
    fig.text(PL - 10, PT + 12, "episodes that placed a different orange (%)", 8.0, "left", rgb=MUTED)
    fig.set_stroke((0.30, 0.30, 0.30), 1.0); fig.rect(PL, PB, PR - PL, PT - PB, fill=False)

    # two source groups, two bars (monotask, subtask) each
    sources = ["Teleop", "Teleop+Auto"]
    group_w = (PR - PL) / len(sources)
    BW, GAP = 52, 14
    for gi, src in enumerate(sources):
        gx = PL + (gi + 0.5) * group_w
        members = [b for b in bars if b[0] == src]
        offset = (BW + GAP) / 2
        centers = {"monotask": gx - offset, "subtask": gx + offset}
        for src2, form, rate, rec, ff in members:
            cx = centers[form]; x = cx - BW / 2; top = y_of(rate)
            fig.set_fill(FILL[form]); fig.rect(x, PB, BW, top - PB)
            fig.set_stroke(STROKE[form], 1.1); fig.rect(x, PB, BW, top - PB, fill=False)
            fig.text(cx, top + 4, f"{rate:.0f}%", 10.0, "center", rgb=INK, bold=True)
            fig.text(cx, PB - 13, form, 8.5, "center", rgb=MUTED, bold=True)
            fig.text(cx, PB - 23, f"{rec}/{ff}", 7.0, "center", rgb=MUTED)
        fig.text(gx, PB - 38, src, 10.0, "center", rgb=INK, bold=True)
        if gi:
            sep = PL + gi * group_w
            fig.set_stroke(GRID, 0.6); fig.line(sep, PB, sep, PT)

    fig.save(OUTPUT_PDF)
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
