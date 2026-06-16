"""Place-success rate for the orchestrated (subtask) models, anchored to the
geometric oranges_in_plate count.

Reads the git-tracked eval checkpoints under ``isaac-inference/results``.

Why this is needed. A PLACE is only *confirmed* when the orange is in the plate
AND the gripper has opened AND its tip has retracted >= PLACE_GRIPPER_Z_MIN above
the plate (eval_utils ``_check_place``). That gripper-retraction gate curates
clean hand-off poses; it means an orange resting in the plate with the gripper
still low is never confirmed and, after its 500-step budget, is logged as a
``timeout`` failure despite being placed. The raw success flag therefore
undercounts placements.

We anchor to the trustworthy geometric final count instead: exactly
``oranges_in_plate`` placements truly succeeded, so

    place success rate = sum(oranges_in_plate) / sum(genuine place attempts),

where a genuine place attempt is any PLACE attempt that ran to its own terminal
outcome -- a confirmed success, a drop, or a *subtask* timeout. Full-episode
truncations (env_truncated / episode_ended / plate_flipped) are excluded.
No per-attempt attribution or identity guessing is involved, so the "place that
ran out only because the whole episode ended" edge case cannot inflate the rate.

Genuine failures decompose exactly: every drop is a failed attempt, so
    timeout failures = (attempts - placed) - drops,   placed-via-timeout = timeouts - timeout failures
the last being the gripper-not-retracted artifact.

This is meaningful only for the orchestrated runs, which log an authoritative
PLACE subtask. The monotask placements are observer-inferred with no discrete
place attempt, so a place-subtask rate is not well-defined there.

Pure stdlib.  python compute_place_success.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "isaac-inference" / "results"
MODELS = [
    ("Teleop",      "Gal-pick-orange-tailedCH20", "checkpoint.json"),
    ("Teleop+Auto", "Gal-merged-tailed-auto",     "checkpoint.json"),
]
BOOK = {"episode_ended", "plate_flipped", "env_truncated", "interrupted_by_new_attempt"}


def analyse(ds, subdir, fname):
    data = json.load(open(RESULTS / subdir / fname))
    placed = attempts = drops = timeouts = recorded_success = 0
    for e in data["episodes"]:
        placed += e.get("oranges_in_plate", 0)
        for a in e["subtask_attempts"]:
            if a.get("subtask") != "PLACE":
                continue
            r, fr = a.get("result"), a.get("failure_reason")
            if r == "failure" and fr in BOOK:
                continue
            attempts += 1
            if r == "success":
                recorded_success += 1
            elif fr == "timeout":
                timeouts += 1
            else:
                drops += 1
    fail = attempts - placed
    timeout_fail = fail - drops
    placed_via_timeout = timeouts - timeout_fail
    print(f"{ds:11s}  place success = {100*placed/attempts:.1f}%  ({placed}/{attempts})")
    print(f"{'':11s}  genuine failures: {drops} drops + {timeout_fail} subtask-timeouts = {fail}")
    print(f"{'':11s}  of {timeouts} raw timeouts, {placed_via_timeout} were placed (gripper not retracted); "
          f"check {recorded_success}+{placed_via_timeout}={recorded_success+placed_via_timeout} vs placed {placed}")


if __name__ == "__main__":
    for m in MODELS:
        analyse(*m)
