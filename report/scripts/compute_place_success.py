"""Place-success rates (raw vs. corrected) for the four evaluated models.

Reads the git-tracked eval checkpoints under ``isaac-inference/results``. Prints
the place success rate overall and by scene state (oranges already in the plate
when the place began), for the values in the ``tab:place_success`` table.

Why a correction is needed (subtask runs only). The orchestrator confirms a
PLACE only when the orange is in the plate AND the gripper has opened AND its tip
has retracted >= PLACE_GRIPPER_Z_MIN above the plate (eval_utils ``_check_place``).
That gripper-retraction gate was added to curate clean hand-off poses for the
next subtask; it means an orange resting in the plate with the gripper still low
is never confirmed and, after the 500-step budget, is logged as a ``timeout``
failure despite being placed. We therefore recount a subtask place as a success
when the orange ends up in the plate, detected behaviourally: a failed/timeout
place whose orange the orchestrator never re-engages (no later attempt targets it)
was placed and dropped from the sampling pool. This predicted placed-set matches
the trustworthy geometric ``oranges_in_plate`` count in >=86/100 episodes.

The monotask traces use ``PhaseMonitor`` confirmation, which has no gripper-Z
gate and no per-subtask timeout, so their rates need no correction; their scene
state is reconstructed from placement events (the flat traces omit it).

Pure stdlib.  python compute_place_success.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "isaac-inference" / "results"

MODELS = [
    ("Teleop",      "monotask", "Gal_split_nolang",                          "flat_checkpoint.json"),
    ("Teleop",      "subtask",  "Gal-pick-orange-tailedCH20",                "checkpoint.json"),
    ("Teleop+Auto", "monotask", "Gal-merged-tailed-auto-no-lang-no-home",    "flat_checkpoint.json"),
    ("Teleop+Auto", "subtask",  "Gal-merged-tailed-auto",                    "checkpoint.json"),
]
BOOK = {"episode_ended", "plate_flipped", "env_truncated", "interrupted_by_new_attempt"}


def tgt(a):
    return a.get("actual_orange") or a.get("requested_orange") or a.get("inferred_target_orange")


def pct(a, b):
    return f"{100 * a / b:.1f}% ({a}/{b})" if b else "n/a"


def place_attempts(atts):
    return [a for a in atts if a.get("subtask") == "PLACE"]


def n_placed_of(attempt, episode, atts):
    """n_placed_start if logged (subtask), else reconstruct from place_success events (monotask)."""
    if attempt.get("n_placed_start") is not None:
        return attempt["n_placed_start"]
    steps = sorted(ev["step"] for ev in episode.get("timeline", []) if ev.get("event_type") == "place_success")
    return sum(1 for s in steps if s < attempt["start_step"])


def corrected_success(attempt, o, atts):
    """A place is a (corrected) success if confirmed, or if it is the orange's last
    place attempt and the orchestrator never re-engages that orange afterwards."""
    if attempt.get("result") == "success":
        return True
    patts = [x for x in atts if x.get("subtask") == "PLACE" and tgt(x) == o]
    last_end = max(x["end_step"] for x in patts)
    if attempt["end_step"] != last_end:
        return False
    return not any(tgt(x) == o and x["start_step"] > last_end for x in atts)


def analyse(ds, form, subdir, fname):
    data = json.load(open(RESULTS / subdir / fname))
    overall = [0, 0]
    by_state = defaultdict(lambda: [0, 0])
    for e in data["episodes"]:
        atts = sorted(e["subtask_attempts"], key=lambda a: a.get("start_step", 0))
        for a in place_attempts(atts):
            if a.get("result") == "failure" and a.get("failure_reason") in BOOK:
                continue
            ok = corrected_success(a, tgt(a), atts) if form == "subtask" else (a.get("result") == "success")
            ns = n_placed_of(a, e, atts)
            overall[1] += 1; overall[0] += ok
            by_state[ns][1] += 1; by_state[ns][0] += ok
    tag = "corrected" if form == "subtask" else "raw"
    line = f"{ds:11s} {form:9s} overall {pct(*overall):>16s} ({tag})"
    line += "   | " + "  ".join(f"{k} placed {pct(*by_state[k])}" for k in sorted(by_state))
    print(line)


if __name__ == "__main__":
    for m in MODELS:
        analyse(*m)
