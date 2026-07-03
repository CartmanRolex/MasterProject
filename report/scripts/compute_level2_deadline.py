"""Level-2 recovery, deadline-anchored (feeds the §4.2.1 prose + tab:level2_deadline).

The monotask-vs-subtask recovery comparison cannot use grasp attempts (monotask
intent is unobservable) nor post-grasp failures (those trigger Level-1, not
Level-2). The matched evidence is TIME: a **stall** = one full grasp budget
(700 steps) of consecutive fruitless commitment to a single orange.

Commitment
  subtask : the requested orange of a run of consecutive GRASP prompts
            (orchestrator log, ground truth). Deadline = end of the run's first
            timed-out GRASP attempt — exact, the GRASP budget is 700 steps.
  monotask: INFERRED as the nearest not-yet-placed orange to the gripper, from
            the v4 geometry_trace (samples every 10 steps). A commitment spell =
            consecutive samples with the same nearest unplaced orange and no
            force-confirmed grasp (>=10 N both tips, gripper closing within
            3 cm — same thresholds as plot_recovery_outcomes.py). Deadline =
            spell start + THR_MONO.

THR_MONO = 600, not 700: the monotask clock only starts once the target becomes
nearest, i.e. after part of the approach. Calibration from successful
acquisitions — subtask median 286–290 steps (home -> secured) vs monotask
140–170 steps (nearest -> secured) — puts that head start at ~130–150 steps.
The sensitivity sweep (500/600/700) shows no conclusion depends on it.

Filters (both formulations, identically)
  opportunity gate: >=1 orange OTHER than the committed one unplaced at the
            deadline — otherwise redirection is impossible by construction.
  censor  : >= MIN_REMAINING steps of the 5000-step episode budget left after
            the deadline. The report uses 1500, which exceeds the median
            steps-to-place, so observed recovery is not clipped by episode end
            (it also makes subtask redirection an exact 100%: the only
            exceptions ever observed were stalls < 100 steps before the end).

Metrics over kept stalls
  redirected     : share whose next worked-on orange differs from the committed
                   one (the Level-2 mechanism firing / monotask switching).
  placed after   : share followed by ANY successful placement, decomposed
                   additively by the identity of the FIRST post-deadline
                   placement: the SAME committed orange vs a DIFFERENT orange
                   (the Level-2 payoff). A second additive decomposition by
                   response (via redirect / via persistence) is printed for the
                   prose; note the axes differ — a persisted-then-secured
                   stall can still first place a different orange later.
  steps to place : deadline -> first subsequent placement (mean and median),
                   among placed-after stalls.

Placement timing: subtask from genuine successful PLACE attempts
(target_in_plate_end, interruptions excluded); monotask from the per-orange
in_plate flip in the geometry trace. Self-validates final_scene vs
oranges_in_plate per episode like the sibling compute_* scripts.

  python compute_level2_deadline.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

RESULTS = Path(__file__).resolve().parents[2] / "isaac-inference" / "results"

# (dataset, formulation, results subdir, checkpoint filename)
MODELS = [
    ("Teleop",      "monotask", "Gal_split_nolang",                       "flat_checkpoint.json"),
    ("Teleop",      "subtask",  "Gal-pick-orange-tailedCH20",             "checkpoint.json"),
    ("Teleop+Auto", "monotask", "Gal-merged-tailed-auto-no-lang-no-home", "flat_checkpoint.json"),
    ("Teleop+Auto", "subtask",  "Gal-merged-tailed-auto",                 "checkpoint.json"),
]

CLOSED, REACH, FORCE_MIN = 0.60, 0.03, 10.0   # force-confirmed-grasp thresholds
EPISODE_BUDGET = 5000
THR_MONO_REPORT = 600                          # calibrated monotask deadline
CENSOR_REPORT = 1500                           # report uses this censor level


def episodes(d):
    eps = d["episodes"]
    return list(eps.values()) if isinstance(eps, dict) else eps


def validate(e):
    """1 if final_scene disagrees with oranges_in_plate, else 0."""
    n_final = sum(1 for o in (e.get("final_scene", {}).get("oranges", {}) or {}).values()
                  if o.get("in_plate"))
    return int(n_final != e.get("oranges_in_plate"))


# ---------------- monotask: commitment spells from the geometry trace ----------------

def mono_trace(e):
    """(spells, placed, secured_durations).
    spell = (orange, start_step, end_step, outcome) with outcome
    S = a force-confirmed grasp ended the spell, R = commitment moved to a
    different orange, E = episode ended still committed.
    placed = [(step, orange)] first in_plate flip per orange."""
    gt = e.get("geometry_trace") or {}
    cols, rows = gt.get("columns", []), gt.get("rows", [])
    if not rows:
        return [], [], []
    si = cols.index("step")
    gp, gf, jf = cols.index("gripper_pos"), cols.index("grip_force_n"), cols.index("jaw_force_n")
    tip = {n[:-9]: i for i, n in enumerate(cols) if n.endswith(":tip_dist")}
    ax = {n[:-10]: i for i, n in enumerate(cols) if n.endswith(":axis_dist")}
    ip = {n[:-9]: i for i, n in enumerate(cols) if n.endswith(":in_plate")}

    placed = []
    for n, i in ip.items():
        for r in rows:
            if r[i]:
                placed.append((r[si], n))
                break

    spells, secured_dur = [], []
    cur = None  # [orange, start_step, end_step]

    def close(outcome):
        nonlocal cur
        if cur is not None:
            spells.append((cur[0], cur[1], cur[2], outcome))
            if outcome == "S":
                secured_dur.append(cur[2] - cur[1])
        cur = None

    for r in rows:
        free = [(n, r[tip[n]]) for n in tip if not r[ip[n]]]
        eng = min(free, key=lambda t: t[1])[0] if free else None
        secure = r[gp] < CLOSED and min(r[gf], r[jf]) >= FORCE_MIN and \
            any(r[ax[n]] < REACH for n in ax if not r[ip[n]])
        if secure:
            if cur is not None:
                cur[2] = r[si]
            close("S")
            continue
        if eng is None:
            continue                      # far from every orange / all placed
        if cur is None:
            cur = [eng, r[si], r[si]]
        elif eng == cur[0]:
            cur[2] = r[si]
        else:
            close("R")
            cur = [eng, r[si], r[si]]
    close("E")
    return spells, placed, secured_dur


# ---------------- subtask: stalls from the orchestrator log ----------------

def sub_events(e):
    """(stalls, placed, secured_durations).
    stall = (orange, deadline_step, outcome S/R/E), one per run of
    consecutive same-target GRASP prompts containing >=1 timeout before any
    success; deadline = end of that first timed-out attempt."""
    atts = sorted(e.get("subtask_attempts", []), key=lambda a: a.get("start_step", 0))
    grasps = [a for a in atts if a.get("subtask") == "GRASP"]
    placed = [(a.get("end_step", 0), a.get("actual_orange") or a.get("requested_orange"))
              for a in atts
              if a.get("subtask") == "PLACE"
              and a.get("failure_reason") != "interrupted_by_new_attempt"
              and a.get("target_in_plate_end")]
    secured_dur = [a.get("duration_steps", 0) for a in grasps if a.get("result") == "success"]
    events, i = [], 0
    while i < len(grasps):
        o = grasps[i].get("requested_orange")
        j, secured, deadline = i, False, None
        while j < len(grasps) and grasps[j].get("requested_orange") == o:
            if grasps[j].get("result") == "success":
                secured = True
            if grasps[j].get("result") == "timeout" and deadline is None and not secured:
                deadline = grasps[j].get("end_step", 0)
            j += 1
        if deadline is not None:
            outcome = "S" if secured else ("R" if j < len(grasps) else "E")
            events.append((o, deadline, outcome))
        i = j
    return events, placed, secured_dur


# ---------------- report ----------------

def pct(a, b):
    return f"{100*a/b:5.1f}% ({a:3d}/{b:3d})" if b else "      n/a(0)"


def report(thr_mono, min_remaining):
    print(f"=== monotask threshold {thr_mono} | censor: >= {min_remaining} steps left after deadline ===")
    for src, form, subdir, fname in MODELS:
        data = json.load(open(RESULTS / subdir / fname))
        n = rec = same = diff = gated_out = censored = mismatch = 0
        R = E = S = 0
        rec_r, rec_p = [0, 0], [0, 0]      # placed-after count per response branch
        ttp, sec_all = [], []
        for e in episodes(data):
            mismatch += validate(e)
            if form == "monotask":
                spells, placed, sdur = mono_trace(e)
                events = [(o, s0 + thr_mono, outc) for o, s0, s1, outc in spells
                          if s1 - s0 >= thr_mono]
            else:
                events, placed, sdur = sub_events(e)
            sec_all += sdur
            oranges = set((e.get("final_scene", {}).get("oranges", {}) or {}).keys())
            for o, dl, outc in events:
                placed_before = {nm for s, nm in placed if s <= dl}
                if not (oranges - placed_before - {o}):
                    gated_out += 1
                    continue
                if EPISODE_BUDGET - dl < min_remaining:
                    censored += 1
                    continue
                n += 1
                S += outc == "S"
                R += outc == "R"
                E += outc == "E"
                branch = rec_r if outc == "R" else rec_p
                branch[1] += 1
                after = [(s, nm) for s, nm in placed if s > dl]
                if after:
                    rec += 1
                    branch[0] += 1
                    first_step, first_orange = min(after)
                    same += first_orange == o
                    diff += first_orange != o
                    ttp.append(first_step - dl)
        print(f"{src:11s} {form:8s}: stalls={n:3d} (gated out {gated_out}, censored {censored}) | "
              f"redirected {pct(R, n)} | "
              f"placed-after {pct(rec, n)} "
              f"= same orange {100*same/n:4.1f}% ({same:3d}) + different {100*diff/n:4.1f}% ({diff:3d}) | "
              f"steps-to-place mean {mean(ttp):6.0f} med {median(ttp):6.0f} (n={len(ttp)})")
        print(f"{'':21s} by response: via redirect {100*rec_r[0]/n:4.1f}% ({rec_r[0]:3d}) "
              f"+ via persist {100*rec_p[0]/n:4.1f}% ({rec_p[0]:3d}) | "
              f"persisted past deadline: {pct(S + E, n)} | "
              f"[successful-grasp dur med {median(sec_all):4.0f}]"
              + (f"   [final/count mismatch {mismatch}]" if mismatch else ""))
    print()


print(">>> REPORT NUMBERS (tab:level2_deadline / §4.2.1 prose)")
report(THR_MONO_REPORT, CENSOR_REPORT)

print(">>> SENSITIVITY: censor level (monotask threshold fixed at 600)")
for censor in (0, 500, 1000):
    report(THR_MONO_REPORT, censor)

print(">>> SENSITIVITY: monotask deadline threshold (censor fixed at 1500)")
for thr in (500, 700):
    report(thr, CENSOR_REPORT)
