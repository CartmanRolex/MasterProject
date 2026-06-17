"""Place success rate for all four models, read directly from the geometric
per-attempt fields recorded by the eval (schema EpisodeStory v2 / PhaseMonitor v3).

Reads the git-tracked eval checkpoints under ``isaac-inference/results``.

Each subtask attempt now carries ``target_in_plate_end`` -- whether the attempt's
orange was geometrically in the plate at the attempt's end -- and ``scene_start``
with ``n_in_plate`` (oranges already placed when the attempt began). So place
success is a direct per-attempt read, uniform for monotask and subtask, with no
anchoring to oranges_in_plate, no "not re-engaged" heuristic, and no event
reconstruction. (This replaces the brittle post-processing that the old
gripper-retraction confirmation flag forced; see isaac-inference/EVAL_LOGGING_REWRITE.md.)

A place attempt counts when ``target_in_plate_end`` is known and the attempt was
not interrupted by a new attempt; success = the orange ended in the plate.

Pure stdlib.  python compute_place_success.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "isaac-inference" / "results"
MODELS = [
    ("Teleop",      "monotask", "Gal_split_nolang",                       "flat_checkpoint.json"),
    ("Teleop",      "subtask",  "Gal-pick-orange-tailedCH20",             "checkpoint.json"),
    ("Teleop+Auto", "monotask", "Gal-merged-tailed-auto-no-lang-no-home", "flat_checkpoint.json"),
    ("Teleop+Auto", "subtask",  "Gal-merged-tailed-auto",                 "checkpoint.json"),
]


def pct(a, b):
    return f"{100 * a / b:.1f}% ({a}/{b})" if b else "n/a"


def analyse(ds, form, subdir, fname):
    data = json.load(open(RESULTS / subdir / fname))
    overall = [0, 0]
    by_state = defaultdict(lambda: [0, 0])
    consistency_mismatch = 0
    for e in data["episodes"]:
        # sanity: recorded final scene must agree with the geometric count
        fin = sum(1 for o in e["final_scene"]["oranges"].values() if o.get("in_plate"))
        if fin != e.get("oranges_in_plate"):
            consistency_mismatch += 1
        for a in e["subtask_attempts"]:
            if a.get("subtask") != "PLACE":
                continue
            if a.get("failure_reason") == "interrupted_by_new_attempt":
                continue
            placed = a.get("target_in_plate_end")
            if placed is None:
                continue
            ns = (a.get("scene_start") or {}).get("n_in_plate")
            overall[1] += 1
            overall[0] += bool(placed)
            by_state[ns][1] += 1
            by_state[ns][0] += bool(placed)
    states = "  ".join(
        f"{k} placed {pct(*by_state[k])}"
        for k in sorted(by_state, key=lambda x: (x is None, x))
    )
    print(f"{ds:11s} {form:9s} place success {pct(*overall):>16s}   | {states}"
          f"   [final_scene/count mismatch {consistency_mismatch}/100]")


if __name__ == "__main__":
    for m in MODELS:
        analyse(*m)
