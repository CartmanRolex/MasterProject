"""Monotask grasp intent from the per-step geometry_trace (eval schema v4).

The end-to-end (monotask) policy never names a target, so we cannot ask which orange
it tried from the checkpoint's attempt trace (that target is re-guessed every few
frames). Instead this reads the raw per-step gripper->orange geometry logged by
PhaseMonitor (`geometry_trace`, schema v4) and derives, with tunable thresholds:

  * Distinct oranges TRIED / episode -- fills the `pending` cells in experiments.tex.
  * Fixation -- distribution of distinct oranges engaged per episode (1 = fixates on a
    single orange and burns the episode; >1 = it does move on), and the share of
    centering time spent on the single most-engaged orange.
  * Closest grip-axis approach per orange.

"Tried" = the policy *sustained-centers* the grip axis on an orange: the orange is the
nearest unplaced one AND within CENTER_M of the grip axis for >= MIN_CENTER_FRAMES
*consecutive* recorded frames. Plain proximity is NOT enough -- the oranges sit ~10 cm
apart, so the gripper passes within ~1 cm of all three while working one; only a
sustained centred run marks a genuine attempt. Thresholds live here, so refining the
definition needs no eval rerun.

  python compute_grasp_intent.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import statistics as st

RESULTS = Path(__file__).resolve().parents[2] / "isaac-inference" / "results"

# Monotask (flat) models only -- the subtask runs already name their target.
MODELS = [
    ("Teleop",      "Gal_split_nolang",                       "flat_checkpoint.json"),
    ("Teleop+Auto", "Gal-merged-tailed-auto-no-lang-no-home", "flat_checkpoint.json"),
]

# --- Tunable definitions (no eval rerun needed to change these) --------------
CENTER_M = 0.02           # grip-axis distance (m) under which the policy is "centred" on an orange
MIN_CENTER_FRAMES = 5     # consecutive recorded frames centred -> a genuine attempt
                          # (with TELEMETRY_EVERY_STEPS=10, 5 frames ~= 50 sim steps ~= 0.8 s)


def episodes(data):
    eps = data["episodes"]
    return list(eps.values()) if isinstance(eps, dict) else eps


def orange_cols(columns):
    """Map orange name -> (axis_dist_idx, in_plate_idx) from the columnar header."""
    out = {}
    for i, c in enumerate(columns):
        if c.endswith(":axis_dist"):
            out.setdefault(c[:-len(":axis_dist")], {})["axis"] = i
        elif c.endswith(":in_plate"):
            out.setdefault(c[:-len(":in_plate")], {})["plate"] = i
    return out


def nearest_unplaced(row, ocols):
    """(name, axis_dist) of the nearest not-in-plate orange in this row, or None."""
    best = None
    for name, idx in ocols.items():
        if row[idx["plate"]]:
            continue
        d = row[idx["axis"]]
        if best is None or d < best[1]:
            best = (name, d)
    return best


def centering(rows, ocols):
    """Per-orange: total centred frames, max consecutive centred run."""
    total = defaultdict(int)
    maxrun = defaultdict(int)
    cur, run = None, 0
    for row in rows:
        nb = nearest_unplaced(row, ocols)
        if nb and nb[1] < CENTER_M:
            total[nb[0]] += 1
            run = run + 1 if nb[0] == cur else 1
            cur = nb[0]
            maxrun[cur] = max(maxrun[cur], run)
        else:
            cur, run = None, 0
    return total, maxrun


def analyse(ds, subdir, fname):
    path = RESULTS / subdir / fname
    print(f"\n##### {ds} monotask  ({subdir}) #####")
    if not path.exists():
        print(f"  checkpoint missing: {path}")
        return
    data = json.load(open(path))
    eps = episodes(data)

    tried_per_ep = []
    distinct_hist = defaultdict(int)
    dominant_shares = []
    closest_all = []
    no_trace = 0

    for e in eps:
        gt = e.get("geometry_trace") or {}
        rows = gt.get("rows")
        if not rows:
            no_trace += 1
            continue
        ocols = orange_cols(gt.get("columns", []))
        total, maxrun = centering(rows, ocols)
        tried = [n for n, r in maxrun.items() if r >= MIN_CENTER_FRAMES]
        tried_per_ep.append(len(tried))
        distinct_hist[len(tried)] += 1
        tot = sum(total[n] for n in tried)
        if tot:
            dominant_shares.append(max(total[n] for n in tried) / tot)
        closest = {}
        for row in rows:
            for n, idx in ocols.items():
                closest[n] = min(closest.get(n, 9.0), row[idx["axis"]])
        closest_all.extend(closest.values())

    n_used = len(tried_per_ep)
    if not n_used:
        print("  no geometry_trace in any episode (rerun eval with schema v4 to populate)")
        return

    print(f"  episodes with trace:                {n_used}/{len(eps)}"
          + (f"   ({no_trace} without trace)" if no_trace else ""))
    print(f"  distinct oranges TRIED / episode:   {st.mean(tried_per_ep):.2f}"
          f"   (median {st.median(tried_per_ep)})")
    dist = "  ".join(f"{k}:{distinct_hist[k]}" for k in sorted(distinct_hist))
    print(f"  fixation (distinct tried -> #eps):  {dist}")
    if dominant_shares:
        print(f"  share of centred time on top orange: {100*st.mean(dominant_shares):.1f}%")
    if closest_all:
        print(f"  median closest grip-axis approach:  {st.median(closest_all)*100:.1f} cm")
    print(f"  [CENTER_M={CENTER_M*100:.0f} cm, MIN_CENTER_FRAMES={MIN_CENTER_FRAMES}]")


if __name__ == "__main__":
    for m in MODELS:
        analyse(*m)
