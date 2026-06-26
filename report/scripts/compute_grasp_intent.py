"""Monotask grasp intent from the per-step geometry_trace (eval schema v4).

The end-to-end (monotask) policy never names a target, so we cannot ask which orange
it tried from the checkpoint's attempt trace (that target is re-guessed every few
frames). Instead this reads the raw per-step gripper->orange geometry logged by
PhaseMonitor (`geometry_trace`, schema v4) and derives, with tunable thresholds:

  * Distinct oranges TRIED / episode -- fills the `pending` cells in experiments.tex.
  * Fixation -- distribution of distinct oranges engaged per episode (1 = fixates on a
    single orange and burns the episode; >1 = it does move on), and the share of
    engaged frames spent on the single most-engaged orange.
  * Closest approach per orange -- min grip-axis distance reached, with step + closure.

"Engaged" = the orange is the nearest unplaced one to the grip axis AND within
ENGAGE_M of it; an orange counts as "tried" once it accrues >= MIN_ENGAGED_FRAMES
such frames. Thresholds live here, so refining the definition needs no eval rerun.

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

# --- Tunable definitions (no rerun needed to change these) -------------------
ENGAGE_M = 0.06            # grip-axis distance (m) under which the policy is "aiming" at an orange
MIN_ENGAGED_FRAMES = 3     # recorded frames an orange must accrue to count as "tried"


def episodes(data):
    eps = data["episodes"]
    return list(eps.values()) if isinstance(eps, dict) else eps


def per_frame_engaged(frame):
    """Return (name, axis_dist) of the nearest unplaced orange within ENGAGE_M, else None."""
    best = None
    for name, o in frame.get("oranges", {}).items():
        if o.get("in_plate"):
            continue
        d = o.get("axis_dist")
        if d is None:
            continue
        if best is None or d < best[1]:
            best = (name, d)
    if best is not None and best[1] < ENGAGE_M:
        return best
    return None


def analyse(ds, subdir, fname):
    path = RESULTS / subdir / fname
    print(f"\n##### {ds} monotask  ({subdir}) #####")
    if not path.exists():
        print(f"  checkpoint missing: {path}")
        return
    data = json.load(open(path))
    eps = episodes(data)

    tried_per_ep = []
    distinct_hist = defaultdict(int)     # distinct oranges tried -> #episodes
    dominant_shares = []
    closest_by_orange = defaultdict(list)
    no_trace = 0

    for e in eps:
        trace = (e.get("geometry_trace") or {})
        frames = trace.get("frames")
        if not frames:
            no_trace += 1
            continue

        engaged_frames = defaultdict(int)
        closest = {}  # name -> (min_axis_dist, step, gripper_pos)
        for f in frames:
            eng = per_frame_engaged(f)
            if eng is not None:
                engaged_frames[eng[0]] += 1
            for name, o in f.get("oranges", {}).items():
                d = o.get("axis_dist")
                if d is None:
                    continue
                if name not in closest or d < closest[name][0]:
                    closest[name] = (d, f.get("step"), f.get("gripper_pos"))

        tried = [n for n, c in engaged_frames.items() if c >= MIN_ENGAGED_FRAMES]
        tried_per_ep.append(len(tried))
        distinct_hist[len(tried)] += 1
        total_eng = sum(engaged_frames[n] for n in tried)
        if total_eng:
            dominant_shares.append(max(engaged_frames[n] for n in tried) / total_eng)
        for n, (d, _s, _g) in closest.items():
            closest_by_orange[n].append(d)

    n_used = len(tried_per_ep)
    if not n_used:
        print(f"  no geometry_trace in any episode (rerun eval with schema v4 to populate)")
        return

    print(f"  episodes with trace:                {n_used}/{len(eps)}"
          + (f"   ({no_trace} without trace)" if no_trace else ""))
    print(f"  distinct oranges TRIED / episode:   {st.mean(tried_per_ep):.2f}"
          f"   (median {st.median(tried_per_ep)})")
    dist = "  ".join(f"{k}:{distinct_hist[k]}" for k in sorted(distinct_hist))
    print(f"  fixation (distinct tried -> #eps):  {dist}")
    if dominant_shares:
        print(f"  share of engaged time on top orange: {100*st.mean(dominant_shares):.1f}%")
    closest_all = [d for v in closest_by_orange.values() for d in v]
    if closest_all:
        print(f"  median closest grip-axis approach:  {st.median(closest_all)*100:.1f} cm"
              f"   (ENGAGE_M={ENGAGE_M*100:.0f} cm, MIN_FRAMES={MIN_ENGAGED_FRAMES})")


if __name__ == "__main__":
    for m in MODELS:
        analyse(*m)
