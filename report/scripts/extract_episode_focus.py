"""Offline per-episode focus / retry / failure analysis from eval checkpoints.

Runs on the LAPTOP. Reads only the git-tracked ``checkpoint.json`` /
``flat_checkpoint.json`` files under ``isaac-inference/results/`` -- NO re-run of
the evaluation is needed. Everything is derived from the ``subtask_attempts`` and
``timeline`` already stored per episode.

Goal: compare, on a common footing, how each policy *focuses* on oranges during
an episode -- which orange it works, for how long, how often it comes back to the
same one, and why attempts fail -- for both the orchestrated subtask runs (target
is authoritative, read from ``requested_orange``) and the monotask / flat runs
(target is physics-inferred, read from ``inferred_target_orange``).

Two asymmetries are corrected so the comparison is fair:

1. The flat trace's inferred target is the *nearest* orange, which flickers when
   the gripper hovers between two near-equidistant oranges. We merge any
   engagement segment shorter than ``T_MIN`` steps into its neighbours before
   counting switches / visits (raw per-orange dwell is unaffected by this and is
   summed directly).
2. ``switches`` and ``visits`` are computed on the merged segment stream so the
   flat run is not penalised for nearest-orange jitter.

No third-party deps; pure stdlib. Just::

    python extract_episode_focus.py
"""

from __future__ import annotations

import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

# isaac-inference/results, resolved relative to this file (report/scripts/..).
RESULTS = Path(__file__).resolve().parents[2] / "isaac-inference" / "results"

# Flicker-merge threshold. Calibrated from the flat runs: successful grasps last
# ~220+ steps, nearest-orange flicker segments ~60 (62% are <100). 100 sits in
# the gap. Segments shorter than this are absorbed into their neighbours before
# counting target switches and per-orange visits.
T_MIN = 100

# Reasons that are observer/orchestrator bookkeeping rather than a physical
# failure of the policy, excluded from the failure histogram. ``inferred_*`` are
# flat-trace artifacts (the nearest-orange retarget, and place-without-lift
# inference); the truncation/success markers are episode-end bookkeeping.
NON_FAILURE_REASONS = {
    "inferred_target_changed", "inferred_place_without_lift",
    "env_truncated", "episode_ended", "success_3_oranges",
}

# model key -> (display name, results subdir, checkpoint filename, formulation)
MODELS = [
    ("Teleop / monotask", "Gal_split_nolang", "flat_checkpoint.json", "monotask"),
    ("Teleop / subtask", "Gal-pick-orange-tailedCH20", "checkpoint.json", "subtask"),
    ("Teleop+Auto / monotask", "Gal-merged-tailed-auto-no-lang-no-home", "flat_checkpoint.json", "monotask"),
    ("Teleop+Auto / subtask", "Gal-merged-tailed-auto", "checkpoint.json", "subtask"),
]


def target_of(attempt):
    """The orange an attempt is working: authoritative for subtask, inferred for flat."""
    return attempt.get("requested_orange") or attempt.get("inferred_target_orange")


def segments(episode):
    """Piecewise (target, start, end, dur) stream over the episode from attempts.

    Adjacent attempts on the same orange (e.g. GRASP->LIFT->PLACE, or a same-orange
    retry) are coalesced into one segment. Attempts with no target are dropped.
    """
    segs = []
    for a in episode["subtask_attempts"]:
        tgt = target_of(a)
        if tgt is None:
            continue
        dur = a.get("duration_steps") or 0
        if segs and segs[-1]["target"] == tgt:
            segs[-1]["end"] = a["end_step"]
            segs[-1]["dur"] += dur
        else:
            segs.append({"target": tgt, "start": a["start_step"], "end": a["end_step"], "dur": dur})
    return segs


def merge_flicker(segs, t_min):
    """Absorb sub-t_min segments into a neighbour, then coalesce equal neighbours.

    Iteratively removes the shortest segment under t_min, folding its steps into
    the adjacent segment that shares its target if possible, else the longer
    neighbour. Leaves raw per-orange dwell totals (computed separately) untouched;
    this only cleans the switch / visit structure.
    """
    segs = [dict(s) for s in segs]
    while len(segs) > 1:
        i = min(range(len(segs)), key=lambda k: segs[k]["dur"])
        if segs[i]["dur"] >= t_min:
            break
        left = segs[i - 1] if i > 0 else None
        right = segs[i + 1] if i < len(segs) - 1 else None
        # prefer a neighbour with the same target; else the longer-duration neighbour
        if left and left["target"] == segs[i]["target"]:
            keep = left
        elif right and right["target"] == segs[i]["target"]:
            keep = right
        elif left and right:
            keep = left if left["dur"] >= right["dur"] else right
        else:
            keep = left or right
        keep["dur"] += segs[i]["dur"]
        keep["start"] = min(keep["start"], segs[i]["start"])
        keep["end"] = max(keep["end"], segs[i]["end"])
        segs.pop(i)
        # coalesce any now-adjacent equal-target segments
        j = 1
        while j < len(segs):
            if segs[j]["target"] == segs[j - 1]["target"]:
                segs[j - 1]["dur"] += segs[j]["dur"]
                segs[j - 1]["end"] = segs[j]["end"]
                segs.pop(j)
            else:
                j += 1
    return segs


def analyse(name, subdir, fname, formulation):
    data = json.load(open(RESULTS / subdir / fname))
    eps = data["episodes"]
    n = len(eps)

    dwell_conc = []        # max single-orange dwell / total engaged steps
    n_distinct = []        # distinct oranges engaged per episode
    longest_visit = []     # longest single uninterrupted engagement (steps)
    n_switches = []        # target switches per episode (merged)
    revisit_eps = 0        # episodes that return to an already-left orange
    fail_reasons = Counter()
    grasp_attempts_per_placed = []  # grasp attempts taken per orange eventually placed

    for e in eps:
        raw = segments(e)
        if not raw:
            continue
        # raw per-orange dwell (flicker does not bias a sum over a target)
        dwell = defaultdict(int)
        for s in raw:
            dwell[s["target"]] += s["dur"]
        total = sum(dwell.values()) or 1
        dwell_conc.append(max(dwell.values()) / total)
        n_distinct.append(len(dwell))

        merged = merge_flicker(raw, T_MIN)
        longest_visit.append(max(s["dur"] for s in merged))
        n_switches.append(max(0, len(merged) - 1))
        seen = []
        revisited = False
        for s in merged:
            if seen and s["target"] in seen[:-1]:
                revisited = True
            seen.append(s["target"])
        revisit_eps += int(revisited)

        for a in e["subtask_attempts"]:
            r = a.get("failure_reason")
            if r and r not in NON_FAILURE_REASONS:
                fail_reasons[r] += 1

    def med(x):
        return st.median(x) if x else float("nan")

    print(f"\n=== {name}  ({formulation}, n={n}) ===")
    print(f"  focus concentration (max-orange dwell / engaged time): median {med(dwell_conc):.2f}")
    print(f"  distinct oranges engaged per episode:                  median {med(n_distinct):.0f}")
    print(f"  longest single engagement (steps, budget 5000):        median {med(longest_visit):.0f}")
    print(f"  target switches per episode (flicker-merged):          median {med(n_switches):.0f}")
    print(f"  episodes that revisit an already-left orange:          {revisit_eps}/{n}")

    if formulation == "subtask":
        retry = sum(e.get("n_local_retries", 0) for e in eps)
        redir = sum(e.get("n_redirections", 0) for e in eps)
        aband = sum(e.get("n_oranges_abandoned", 0) for e in eps)
        ep_retry = sum(1 for e in eps if e.get("n_local_retries", 0) > 0)
        ep_redir = sum(1 for e in eps if e.get("n_redirections", 0) > 0)
        print(f"  local retries (same orange): {retry} total, in {ep_retry}/{n} episodes")
        print(f"  target redirections:         {redir} total, in {ep_redir}/{n} episodes")
        print(f"  oranges abandoned:           {aband} total")

    top = ", ".join(f"{k} {v}" for k, v in fail_reasons.most_common())
    print(f"  failure reasons: {top}")


def main():
    print(f"Reading checkpoints from {RESULTS}   (T_MIN={T_MIN} steps)")
    for m in MODELS:
        analyse(*m)


if __name__ == "__main__":
    main()
