"""Grasp obedience for the obedience section --- per model, never silently averaged.

Obedience = the gripper closed on the *requested* orange (matched by identity), i.e. the
diagonal of the confusion matrices in ``plot_grasp_confusion.py``; we reuse that module's
``confusion()`` so the numbers match the figure exactly, over scene states 0 and 1.

Also prints the share of GRASP requests that are "right" by scene state, which shows *why*
"right" dominates: the plate is on the left and placed oranges accumulate there, so the
remaining targets skew right as the episode proceeds.

  python compute_obedience.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from plot_grasp_confusion import MODELS, confusion

RESULTS = Path(__file__).resolve().parents[2] / "isaac-inference" / "results"
ORDER = ["right", "middle", "left", "top right", "bottom right"]


def obedience(subdir: str, fname: str):
    agg = defaultdict(lambda: [0, 0])  # label -> [obeyed, total]
    for n_placed in (0, 1):
        conf = confusion(subdir, fname, n_placed)
        for req, row in conf.items():
            agg[req][0] += row.get(req, 0)
            agg[req][1] += sum(row.values())
    return agg


def main() -> None:
    print("Per-model obedience (scene states 0+1; matches the confusion figure):")
    for disp, subdir, fname in MODELS:
        agg = obedience(subdir, fname)
        tot_o = sum(v[0] for v in agg.values())
        tot_n = sum(v[1] for v in agg.values())
        print(f"\n  {disp}  --  overall {100 * tot_o / tot_n:.0f}% ({tot_o}/{tot_n})")
        for label in ORDER:
            o, t = agg[label]
            print(f"    {label:13s} {100 * o / t:4.0f}% ({o}/{t})" if t else f"    {label:13s} n/a")

    # Why "right" dominates: share of GRASP requests that are "right", by #oranges already placed.
    print("\nShare of GRASP requests that are 'right', by #oranges already placed (Teleop subtask):")
    data = json.load(open(RESULTS / "Gal-pick-orange-tailedCH20" / "checkpoint.json"))
    by_state = defaultdict(Counter)
    for e in data["episodes"]:
        for a in e["subtask_attempts"]:
            if a.get("subtask") == "GRASP" and a.get("requested_label"):
                by_state[a.get("n_placed_start")][a["requested_label"]] += 1
    for ns in sorted(by_state):
        tot = sum(by_state[ns].values())
        r = by_state[ns].get("right", 0)
        print(f"  {ns} placed: {100 * r / tot:3.0f}% ({r}/{tot})")


if __name__ == "__main__":
    main()
