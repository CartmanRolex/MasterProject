"""The grasp->place chain, computed from the git-tracked eval checkpoints.

Story: grasping is the wall; once an orange is *secured* it is placed reliably.

Reliable physics only:
  * a SECURED grasp = a force-confirmed GRASP success (reliable for both formulations);
  * placement = the orange's final geometric position (reliable);
  * grasp-success RATE = secured / (secured + grasp timeouts) -- reported for the SUBTASK
    models only, where the orchestrator issues real, budgeted GRASP attempts. The monotask
    trace is heuristic (phase_debug: target re-guessed every 10 frames; failed grasps appear
    as `retargeted`), so it has NO trustworthy grasp rate or attempt count -- we never print one.

Per orange we read its force-confirmed grasp count and whether it ended in the plate, giving:
  * placement-once-secured (clean = placed on the first grasp, recovered = needed a re-grasp);
  * secured oranges / episode (reliable for both, since it counts force-confirmed grasps only).

  python compute_grasp_chain.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import statistics as st

RESULTS = Path(__file__).resolve().parents[2] / "isaac-inference" / "results"
MODELS = [
    ("Teleop",      "monotask", "Gal_split_nolang",                       "flat_checkpoint.json"),
    ("Teleop",      "subtask",  "Gal-pick-orange-tailedCH20",             "checkpoint.json"),
    ("Teleop+Auto", "monotask", "Gal-merged-tailed-auto-no-lang-no-home", "flat_checkpoint.json"),
    ("Teleop+Auto", "subtask",  "Gal-merged-tailed-auto",                 "checkpoint.json"),
]


def pct(a, b):
    return f"{100 * a / b:4.1f}% ({a}/{b})" if b else "   n/a"


def orange_of(a):
    return a.get("actual_orange") or a.get("requested_orange") or a.get("inferred_target_orange")


def analyse(ds, form, subdir, fname):
    data = json.load(open(RESULTS / subdir / fname))
    eps = list(data["episodes"].values()) if isinstance(data["episodes"], dict) else data["episodes"]

    g_succ = g_timeout = 0            # subtask grasp-rate denominator parts
    secured_per_ep = []
    secured = placed = clean = recovered = 0
    mism = 0

    for e in eps:
        secs = defaultdict(int)      # force-confirmed grasp successes per orange
        for a in e["subtask_attempts"]:
            if a.get("subtask") != "GRASP":
                continue
            r = a.get("result")
            if r == "success":
                g_succ += 1
                secs[orange_of(a)] += 1
            elif r == "timeout":
                g_timeout += 1
            # retargeted / env_truncated / episode_ended -> not a clean attempt; excluded
        secured_per_ep.append(len(secs))
        placed_set = {n for n, o in (e.get("final_scene", {}).get("oranges", {}) or {}).items()
                      if o.get("in_plate")}
        if len(placed_set) != e.get("oranges_in_plate"):
            mism += 1
        for o, k in secs.items():
            secured += 1
            if o in placed_set:
                placed += 1
                if k == 1:
                    clean += 1
                else:
                    recovered += 1

    print(f"\n##### {ds} {form}  ({subdir}) #####")
    if form == "subtask":
        print(f"  grasp success rate (secured / attempts):  {pct(g_succ, g_succ + g_timeout)}")
    else:
        print(f"  grasp success rate: n/a (monotask trace is heuristic; no trustworthy attempt count)")
    print(f"  secured oranges / episode:                 {st.mean(secured_per_ep):.2f}")
    print(f"  placement once secured:                    {pct(placed, secured)}"
          f"   [clean {clean} / recovered {recovered}]")
    print(f"  (final_scene vs oranges_in_plate mismatch: {mism}/{len(eps)})")


if __name__ == "__main__":
    for m in MODELS:
        analyse(*m)
