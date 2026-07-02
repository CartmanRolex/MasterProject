"""Outcome of the FIRST orange each episode commits to, split three ways:
the first orange itself is placed / a DIFFERENT orange is placed / nothing is placed.

Why outcomes, not attempts: what the monotask policy is *trying* to grasp is not
observable (a bad approach may not even touch its orange), so attempts cannot be
counted fairly across formulations. Definitions (ground truth):
  * "first orange" = the first orange the system commits to.
      - subtask : the first GRASP prompt (orchestrator log, `subtask_attempts`).
      - monotask: the first force-confirmed grasp (gripper closed + >=10 N on both tips,
                  reaching an orange) read from the v4 `geometry_trace`.
  * outcome classes (exclusive): first orange in plate at the end; else >=1 different
    orange in plate ("recovered"); else nothing placed.
  * monotask episodes with no confirmed grasp at all are excluded (they all end empty,
    so the exclusion flatters the monotask if anything); n is printed per bar.
  * conditional recovery rate = recovered / (first-orange-failed) is still printed for
    the prose.

This is an OUTCOME comparison only: it shows the orchestrated formulation recovers far
more often, without claiming the recovery *logic* (vs better per-attempt grasping) is the
isolated cause. The first-orange-placed segment also shows the monotask's low recovery
is not hidden same-orange substitution.

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


def first_orange_outcomes(subdir, fname, formulation):
    """Return (n_identified, n_first_placed, n_diff_placed, n_nothing)."""
    data = json.load(open(RESULTS / subdir / fname))
    n = first_placed = diff_placed = nothing = 0
    for e in episodes(data):
        placed = placed_set(e)
        first = mono_first_engaged(e) if formulation == "monotask" else sub_first_target(e)
        if first is None:
            continue
        n += 1
        if first in placed:
            first_placed += 1
        elif len(placed) >= 1:
            diff_placed += 1
        else:
            nothing += 1
    return n, first_placed, diff_placed, nothing


# ---- compute -----------------------------------------------------------------
# stacking order bottom->top and segment colours
SEGMENTS = [
    ("first",   "First orange placed",     (0.23, 0.55, 0.31)),
    ("diff",    "Different orange placed", (0.34, 0.63, 0.76)),
    ("nothing", "Nothing placed",          (0.42, 0.45, 0.50)),
]
bars = []
for src, form, subdir, fname in MODELS:
    n, fp, dp, no = first_orange_outcomes(subdir, fname, form)
    bars.append((src, form, n, {"first": fp, "diff": dp, "nothing": no}))
    rate = 100 * dp / (dp + no) if (dp + no) else 0.0
    print(f"{src:11s} {form:8s}: n={n}  first={fp} ({100*fp/n:.0f}%)  diff={dp} ({100*dp/n:.0f}%)  "
          f"nothing={no} ({100*no/n:.0f}%)  | recovery after first failed: {dp}/{dp+no} = {rate:.0f}%")

# ---- draw --------------------------------------------------------------------
INK, MUTED, GRID = (0.16, 0.16, 0.16), (0.42, 0.42, 0.42), (0.90, 0.90, 0.90)
W, H = 600, 300
PL, PR, PB, PT = 54, 440, 56, 248
YMAX = 100.0


def y_of(v):
    return PB + (v / YMAX) * (PT - PB)


def main():
    fig = PdfFigure(width=W, height=H)
    fig.set_fill((1, 1, 1)); fig.rect(0, 0, W, H)
    fig.text((PL + PR) / 2, H - 22, "Outcome of the first orange each episode commits to", 12.5, "center", rgb=INK, bold=True)

    for tick in range(0, 101, 25):
        yt = y_of(tick)
        fig.set_stroke(GRID, 0.7); fig.line(PL, yt, PR, yt)
        fig.text(PL - 8, yt - 3, f"{tick}", 8.0, "right", rgb=MUTED)
    fig.vtext(18, (PB + PT) / 2, "Share of episodes (%)", 8.5, "center", rgb=MUTED)
    fig.set_stroke((0.30, 0.30, 0.30), 1.0); fig.rect(PL, PB, PR - PL, PT - PB, fill=False)

    # two source groups, two stacked bars (monotask, subtask) each
    sources = ["Teleop", "Teleop+Auto"]
    group_w = (PR - PL) / len(sources)
    BW, GAP = 52, 14
    for gi, src in enumerate(sources):
        gx = PL + (gi + 0.5) * group_w
        members = [b for b in bars if b[0] == src]
        offset = (BW + GAP) / 2
        centers = {"monotask": gx - offset, "subtask": gx + offset}
        for src2, form, n, counts in members:
            cx = centers[form]; x = cx - BW / 2
            y = PB
            for key, _, rgb in SEGMENTS:
                pct = 100 * counts[key] / n
                seg_h = (pct / 100) * (PT - PB)
                fig.set_fill(rgb); fig.rect(x, y, BW, seg_h)
                if seg_h >= 13:
                    label_rgb = (1, 1, 1) if key in ("first", "nothing") else (0.05, 0.05, 0.05)
                    fig.text(cx, y + seg_h / 2 - 3, f"{pct:.0f}%", 7.6, "center", rgb=label_rgb, bold=True)
                y += seg_h
            fig.set_stroke((0.25, 0.25, 0.25), 1.0); fig.rect(x, PB, BW, PT - PB, fill=False)
            fig.text(cx, PB - 13, form.capitalize(), 8.5, "center", rgb=MUTED, bold=True)
            fig.text(cx, PB - 23, f"n={n}", 7.0, "center", rgb=MUTED)
        fig.text(gx, PB - 38, src, 10.0, "center", rgb=INK, bold=True)
        if gi:
            sep = PL + gi * group_w
            fig.set_stroke(GRID, 0.6); fig.line(sep, PB, sep, PT)

    # vertical legend on the right, top-to-bottom matching the stacking order
    lx = PR + 18
    ly = (PB + PT) / 2 + 24
    for key, label, rgb in reversed(SEGMENTS):
        fig.set_fill(rgb); fig.rect(lx, ly, 10, 10)
        fig.set_stroke((0.25, 0.25, 0.25), 0.7); fig.rect(lx, ly, 10, 10, fill=False)
        fig.text(lx + 15, ly + 2, label, 8.0, rgb=INK)
        ly -= 20

    fig.save(OUTPUT_PDF)
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
