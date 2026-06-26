"""The grasp->place chain + leveled recovery, from the git-tracked eval checkpoints.

Story: grasping is the wall; once an orange is *secured* it is placed reliably; the
orchestrator's redirection (Level-2) lets it try a different orange after giving up.

Per-formulation "secured/grasped" (deliberate, physically motivated):
  * SUBTASK  : a force-confirmed GRASP success -- the orchestrator only lifts after force.
  * MONOTASK : the orange was LIFTED (a LIFT success) -- the end-to-end policy lifts without
               full force, so "got it off the table" is the honest criterion.

Metrics (all from privileged physics; reliable for both formulations unless noted):
  * Grasp success rate -- subtask only (real budgeted attempts; the monotask trace is
    heuristic, target re-guessed every few frames, so no trustworthy grasp rate).
  * Secured oranges / episode; Place once secured, split clean vs recovered (Level-1:
    re-secured the SAME orange before placing it).
  * Level-2 redirection (subtask only): of GRASP timeouts, the fraction followed by placing
    a DIFFERENT orange. Monotask has no timeout/give-up event -> reported as pending.
  * Distinct oranges TRIED (subtask only): distinct requested_orange per episode -> redirection
    breadth. Monotask pending (inferred target flickers; needs per-step proximity).
  * Empty-episode rate (0/3 placed) -- the fair cross-formulation outcome for redirection.

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


def org(a):
    return a.get("actual_orange") or a.get("requested_orange") or a.get("inferred_target_orange")


def secured_events(e, form):
    """{orange: [steps]} of 'secured' events -- force-confirmed GRASP (subtask) or LIFT (mono)."""
    want = "GRASP" if form == "subtask" else "LIFT"
    out = defaultdict(list)
    for a in sorted(e["subtask_attempts"], key=lambda a: a.get("start_step", 0)):
        if a.get("subtask") == want and a.get("result") == "success":
            out[org(a)].append(a.get("start_step", 0))
    return out


def placed_set(e):
    return {n for n, o in (e.get("final_scene", {}).get("oranges", {}) or {}).items() if o.get("in_plate")}


def successful_places(e):
    return [(a.get("start_step", 0), org(a)) for a in sorted(e["subtask_attempts"], key=lambda a: a.get("start_step", 0))
            if a.get("subtask") == "PLACE" and a.get("target_in_plate_end")
            and a.get("failure_reason") != "interrupted_by_new_attempt"]


def analyse(ds, form, subdir, fname):
    data = json.load(open(RESULTS / subdir / fname))
    eps = list(data["episodes"].values()) if isinstance(data["episodes"], dict) else data["episodes"]

    g_succ = g_timeout = 0
    secured_per_ep, tried_per_ep = [], []
    secured = placed = clean = recovered = 0
    l2_events = l2_diff = 0
    empty = mism = 0

    for e in eps:
        atts = sorted(e["subtask_attempts"], key=lambda a: a.get("start_step", 0))
        placed = placed  # noqa (kept for clarity)
        pf = placed_set(e)
        if len(pf) != e.get("oranges_in_plate"):
            mism += 1
        if e.get("oranges_in_plate") == 0:
            empty += 1
        places = successful_places(e)

        secs = secured_events(e, form)
        secured_per_ep.append(len(secs))
        for o, steps in secs.items():
            secured += 1
            o_place = next((ps for ps, po in places if po == o), None)
            if o_place is not None:
                placed += 1
                n_before = sum(1 for s in steps if s <= o_place)
                if n_before <= 1:
                    clean += 1
                else:
                    recovered += 1

        if form == "subtask":
            tried = set()
            for a in atts:
                if a.get("subtask") != "GRASP":
                    continue
                if a.get("result") == "success":
                    g_succ += 1
                elif a.get("result") == "timeout":
                    g_timeout += 1
                if a.get("requested_orange"):
                    tried.add(a.get("requested_orange"))
            tried_per_ep.append(len(tried))
            # Level-2: GRASP timeout -> different orange placed afterwards
            for a in atts:
                if a.get("subtask") == "GRASP" and a.get("result") == "timeout":
                    l2_events += 1
                    s, A = a.get("start_step", 0), org(a)
                    if any(ps > s and po != A for ps, po in places):
                        l2_diff += 1

    print(f"\n##### {ds} {form}  ({subdir}) #####")
    if form == "subtask":
        print(f"  grasp success rate (force-confirmed):      {pct(g_succ, g_succ + g_timeout)}")
        print(f"  distinct oranges TRIED / episode:          {st.mean(tried_per_ep):.2f}")
        print(f"  Level-2 redirection (timeout->diff orange):{pct(l2_diff, l2_events)}")
    else:
        print(f"  grasp success rate: n/a (heuristic trace)  |  distinct TRIED: pending (needs proximity)")
    print(f"  secured oranges / episode ({'grasp' if form=='subtask' else 'lift'}):    {st.mean(secured_per_ep):.2f}")
    print(f"  place once secured:                        {pct(placed, secured)}"
          f"   [clean {clean} / recovered {recovered}]")
    print(f"  empty episodes (0/3 placed):               {pct(empty, len(eps))}"
          f"   (final/count mismatch {mism}/{len(eps)})")


if __name__ == "__main__":
    for m in MODELS:
        analyse(*m)
