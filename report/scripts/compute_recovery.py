"""Leveled recovery analysis: Level-1 (same orange) and Level-2 (different orange
/ task progress), monotask vs subtask, two datasets (4 models).

Runs on the LAPTOP. Reads only the git-tracked ``checkpoint.json`` /
``flat_checkpoint.json`` files under ``isaac-inference/results/`` -- NO re-run of
the evaluation is needed. Everything is derived from ``subtask_attempts`` and
``final_scene`` already stored per episode (schema EpisodeStory v2 / PhaseMonitor
v3). Pure stdlib::

    python compute_recovery.py

------------------------------------------------------------------------------
Definitions (see report methodology, subsec:failure_taxonomy). Every failure and
recovery is pinned to a level of the goal hierarchy
Task > per-orange goal (GRASP->LIFT->PLACE) > subtask, so nothing is called
"recovery" loosely. A *failure event* is a physical setback read from privileged
state, defined identically for both formulations:

  * LIFT drop  = LIFT attempt, result 'failure', reason 'dropped_during_lift'.
  * PLACE slip = PLACE attempt (not interrupted by a new attempt) whose
                 target_in_plate_end is False (orange not in plate at attempt end).

The dropped/slipped orange's identity is known from contact (``actual_orange``),
so "same orange" needs no target inference for either formulation.

LEVEL-1 recovery -- subtask / SAME orange (implemented by local retry):
  * Lift  : of oranges whose FIRST genuine lift dropped, the fraction eventually
            lifted later in the same episode (a placed orange counts as lifted).
            Reproduces lift_stats.py / tab:lift_recovery.
  * Place : of oranges whose FIRST place attempt failed, the fraction eventually
            placed later in the episode. Reproduces compute_place_success.py /
            tab:place_recovery.
  * Re-engaged same orange: behavioural signature of the orchestrator's local
            retry (next attempt returns to the same orange). Reported for context,
            NOT as a fair head-to-head number -- the orchestrator is built to do
            exactly this, the monotask has no fixed target.

LEVEL-2 recovery -- task / DIFFERENT orange (implemented by target redirection):
  The fair, mechanism-agnostic head-to-head. Trigger is a drop/slip (identically
  detectable for both formulations); the recovery criterion is a later successful
  placement, read from physics. Two views:
  * Score recovery (per episode): of episodes with >=1 drop/slip, the fraction
            whose placed count still increased after the first setback ("kept
            scoring after a setback"; same- or different-orange).
  * Different-orange recovery (per event): of drop/slip events for which >=1
            OTHER orange was still unplaced at the setback, the fraction followed
            by a successful placement of a DIFFERENT orange.

Opportunity is inherent to the score view (the dropped orange is itself unplaced)
and explicit in the different-orange view (>=1 other orange free). The 5000-step
budget is identical for both formulations, so budget exhaustion is symmetric.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "isaac-inference" / "results"

# (dataset, formulation, results subdir, checkpoint filename)
MODELS = [
    ("Teleop",      "monotask", "Gal_split_nolang",                       "flat_checkpoint.json"),
    ("Teleop",      "subtask",  "Gal-pick-orange-tailedCH20",             "checkpoint.json"),
    ("Teleop+Auto", "monotask", "Gal-merged-tailed-auto-no-lang-no-home", "flat_checkpoint.json"),
    ("Teleop+Auto", "subtask",  "Gal-merged-tailed-auto",                 "checkpoint.json"),
]

DROP_REASON = "dropped_during_lift"


def pct(a, b):
    return f"{100 * a / b:5.1f}% ({a}/{b})" if b else "   n/a"


def target_of(a):
    """Worked orange: authoritative for subtask, contact/inferred for flat."""
    return a.get("actual_orange") or a.get("requested_orange") or a.get("inferred_target_orange")


def classify_lift(a):
    """success | drop | timeout | book, for a LIFT attempt (matches lift_stats.py)."""
    r, fr = a.get("result"), a.get("failure_reason")
    if r in ("success", "skipped"):   # place-without-lift -> treated as lifted
        return "success"
    if r == "timeout":
        return "timeout"
    if r == "failure" and fr == DROP_REASON:
        return "drop"
    return "book"                     # episode_ended, env_truncated, interrupted, ...


def placed_in_final(e):
    """Set of oranges geometrically in the plate at episode end."""
    return {n for n, o in e.get("final_scene", {}).get("oranges", {}).items() if o.get("in_plate")}


def all_oranges(e):
    return set(e.get("final_scene", {}).get("oranges", {}).keys())


def successful_places(e):
    """[(start_step, orange)] for genuine successful placements, ordered."""
    out = []
    for a in sorted(e["subtask_attempts"], key=lambda a: a.get("start_step", 0)):
        if a.get("subtask") != "PLACE":
            continue
        if a.get("failure_reason") == "interrupted_by_new_attempt":
            continue
        if a.get("target_in_plate_end"):
            out.append((a.get("start_step", 0), target_of(a)))
    return out


def setbacks(e):
    """[(start_step, orange)] for LIFT drops and PLACE slips, ordered."""
    out = []
    for a in sorted(e["subtask_attempts"], key=lambda a: a.get("start_step", 0)):
        st = a.get("subtask")
        if st == "LIFT" and classify_lift(a) == "drop":
            out.append((a.get("start_step", 0), target_of(a)))
        elif st == "PLACE":
            if a.get("failure_reason") == "interrupted_by_new_attempt":
                continue
            if a.get("target_in_plate_end") is False:
                out.append((a.get("start_step", 0), target_of(a)))
    return out


def analyse(ds, form, subdir, fname):
    data = json.load(open(RESULTS / subdir / fname))
    eps = data["episodes"]

    # --- Level-1 lift (same orange) -------------------------------------------
    l1_lift = [0, 0]
    # --- Level-1 place (same orange) + re-engagement ---------------------------
    l1_place = [0, 0]
    reeng_place = [0, 0]
    # --- Level-2 ---------------------------------------------------------------
    l2_score = [0, 0]      # per episode: kept scoring after first setback
    l2_diff = [0, 0]       # per event: different orange placed after a setback
    mismatch = 0

    for e in eps:
        if sum(1 for o in e.get("final_scene", {}).get("oranges", {}).values() if o.get("in_plate")) \
                != e.get("oranges_in_plate"):
            mismatch += 1

        placed_final = placed_in_final(e)
        atts = sorted(e["subtask_attempts"], key=lambda a: a.get("start_step", 0))

        # Level-1 LIFT: first genuine lift per orange dropped -> eventually lifted?
        lift_seq = defaultdict(list)
        for a in atts:
            if a.get("subtask") == "LIFT":
                c = classify_lift(a)
                if c in ("success", "drop"):
                    lift_seq[target_of(a)].append(c)
        for o, seq in lift_seq.items():
            if seq and seq[0] == "drop":
                l1_lift[1] += 1
                l1_lift[0] += int(("success" in seq) or (o in placed_final))

        # Level-1 PLACE: first place per orange failed -> eventually placed?
        places = [
            a for a in atts
            if a.get("subtask") == "PLACE"
            and a.get("failure_reason") != "interrupted_by_new_attempt"
            and a.get("target_in_plate_end") is not None
        ]
        per = defaultdict(list)
        for a in places:
            per[target_of(a)].append(bool(a.get("target_in_plate_end")))
        for o, seq in per.items():
            if seq and seq[0] is False:
                l1_place[1] += 1
                l1_place[0] += int(any(seq) or o in placed_final)
        for i, a in enumerate(places):
            if a.get("target_in_plate_end"):
                continue
            if i + 1 < len(places):
                reeng_place[1] += 1
                reeng_place[0] += int(target_of(places[i + 1]) == target_of(a))

        # Level-2: setbacks vs subsequent successful placements
        sb = setbacks(e)
        pl = successful_places(e)
        if sb:
            s0 = sb[0][0]
            l2_score[1] += 1
            l2_score[0] += int(any(ps > s0 for ps, _ in pl))
        oranges = all_oranges(e)
        for se, oe in sb:
            placed_by = {o for ps, o in pl if ps <= se}
            others_free = oranges - placed_by - {oe}
            if not others_free:
                continue
            l2_diff[1] += 1
            l2_diff[0] += int(any(ps > se and po != oe for ps, po in pl))

    head = f"{ds:11s} {form:9s}"
    print(f"{head}  L1 lift recovered {pct(*l1_lift):>16s}   "
          f"L1 place recovered {pct(*l1_place):>15s}   "
          f"[re-engaged same place {pct(*reeng_place)}]")
    print(f"{'':21s} L2 score recovery (episode) {pct(*l2_score):>15s}   "
          f"L2 different-orange (event) {pct(*l2_diff):>15s}"
          + (f"   [final/count mismatch {mismatch}/{len(eps)}]" if mismatch else ""))


if __name__ == "__main__":
    print("Leveled recovery  (Level-1 = same orange; Level-2 = different orange / score)\n")
    for m in MODELS:
        analyse(*m)
