"""Failure-mode composition for LIFT and PLACE, and (monotask-only) failure geometry.

Reads the git-tracked eval checkpoints under ``isaac-inference/results`` (schema
EpisodeStory v2 / PhaseMonitor v3). Pure stdlib.

For each of the four models it prints, per subtask:
  * the genuine-attempt composition success / drop(slip) / timeout
    (episode-end truncations excluded from the denominator, as in tab:lift_recovery),
  * the raw ``failure_reason`` counts (sanity / label discovery).
For the monotask (flat) models, whose attempts carry the per-attempt ``metrics``
dict, it also prints the failure-geometry contrast (mean/median over attempts):
  * LIFT  : height_gain_m, success vs drop  (immediate slip = tiny gain)
  * PLACE : xy_distance_m  from plate centre, success vs slip (rim failure = large)

  python compute_failure_modes.py
"""

from __future__ import annotations

import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "isaac-inference" / "results"
MODELS = [
    ("Teleop",      "monotask", "Gal_split_nolang",                       "flat_checkpoint.json"),
    ("Teleop",      "subtask",  "Gal-pick-orange-tailedCH20",             "checkpoint.json"),
    ("Teleop+Auto", "monotask", "Gal-merged-tailed-auto-no-lang-no-home", "flat_checkpoint.json"),
    ("Teleop+Auto", "subtask",  "Gal-merged-tailed-auto",                 "checkpoint.json"),
]

DROP = {"LIFT": "dropped_during_lift", "PLACE": "dropped_during_place"}
TRUNC = {"episode_ended", "env_truncated", "truncated", None}


def pct(a, b):
    return f"{100 * a / b:4.1f}% ({a}/{b})" if b else "   n/a"


def composition(data, sub):
    """Geometric success (orange in plate / lifted) overrides the orchestrator flag:
    a timed-out attempt whose orange ended in the plate counts as a SUCCESS.
    Residual failures (orange NOT in plate at attempt end) split into drop/slip,
    genuine timeout (timed out, orange not placed), and other/truncation."""
    succ = drop = tmo = 0
    raw = Counter()
    timeout_but_placed = 0
    for e in data["episodes"]:
        for a in e["subtask_attempts"]:
            if a.get("subtask") != sub:
                continue
            if a.get("failure_reason") == "interrupted_by_new_attempt":
                continue
            res = a.get("result")
            fr = a.get("failure_reason")
            placed = a.get("target_in_plate_end")  # geometric ground truth (PLACE)
            # geometric override: orange in plate at end => success regardless of flag
            if res == "success" or placed is True:
                succ += 1
                if res != "success":
                    timeout_but_placed += 1
            elif fr == DROP[sub]:
                drop += 1; raw[fr] += 1
            elif res == "timeout" or fr == "timeout":
                tmo += 1; raw["timeout"] += 1
            else:
                raw[fr] += 1  # episode_ended / env_truncated -> excluded from genuine
    n = succ + drop + tmo
    return succ, drop, tmo, n, raw, timeout_but_placed


def grasp_composition(data):
    """GRASP has no drop/slip: an attempt either secures an orange (success, including
    wrong-orange grabs -- obedience is analysed separately) or never acquires one
    (timeout). Truncations and reconstruction artefacts (env_truncated, episode_ended,
    retargeted, ...) are excluded from the denominator. Trustworthy only for the
    orchestrated (subtask) models; the monotask offline reconstruction over-segments the
    grasp search (median ~9 segments/episode), so it has no per-attempt grasp rate."""
    succ = tmo = 0
    raw = Counter()
    timeout_but_placed = 0
    for e in data["episodes"]:
        for a in e["subtask_attempts"]:
            if a.get("subtask") != "GRASP":
                continue
            res, fr = a.get("result"), a.get("failure_reason")
            placed = a.get("target_in_plate_end")  # geometric ground truth
            # geometric override: if the target orange ended in the plate, the grasp
            # really worked, even when the orchestrator logged a timeout/failure.
            if res == "success" or placed is True:
                succ += 1
                if res != "success":
                    timeout_but_placed += 1
                elif fr:
                    raw[f"success:{fr}"] += 1   # wrong-orange grabs (an orange was secured)
            elif res == "timeout" or fr == "timeout":
                tmo += 1; raw["timeout"] += 1
            else:
                raw[fr or res] += 1         # env_truncated / episode_ended / retargeted -> excluded
    return succ, tmo, succ + tmo, raw, timeout_but_placed


def geometry(data, sub, key):
    """mean/median of metrics[key] for success vs the subtask's drop/slip (flat only)."""
    ok, bad = [], []
    for e in data["episodes"]:
        for a in e["subtask_attempts"]:
            if a.get("subtask") != sub:
                continue
            m = a.get("metrics")
            if not isinstance(m, dict) or key not in m or m[key] is None:
                continue
            if a.get("result") == "success":
                ok.append(m[key])
            elif a.get("failure_reason") == DROP[sub]:
                bad.append(m[key])
    def s(v):
        return f"mean {st.mean(v):.3f} median {st.median(v):.3f} (n={len(v)})" if v else "n/a"
    return s(ok), s(bad)


def analyse(ds, form, subdir, fname):
    data = json.load(open(RESULTS / subdir / fname))
    print(f"\n##### {ds} {form}  ({subdir}) #####")
    if form == "subtask":
        sg, tg, ng, graw, gtbp = grasp_composition(data)
        print(f"  GRASP genuine n={ng:3d} | success(acquired) {pct(sg,ng)}  "
              f"timeout {pct(tg,ng)}   [timeout-but-placed reclassified as success: {gtbp}]")
        print(f"        raw (excluded/other): {dict(graw)}")
    else:
        print("  GRASP : not measured per-attempt (monotask offline reconstruction over-segments the search)")
    for sub in ("LIFT", "PLACE"):
        succ, drop, tmo, n, raw, tbp = composition(data, sub)
        dlabel = "drop" if sub == "LIFT" else "slip"
        print(f"  {sub:5s} genuine n={n:3d} | success {pct(succ,n)}  "
              f"{dlabel} {pct(drop,n)}  genuine-timeout {pct(tmo,n)}"
              f"   [timeout-but-placed reclassified as success: {tbp}]")
        print(f"        raw failure_reason (excluded/other): {dict(raw)}")
    if form == "monotask":
        ok, bad = geometry(data, "LIFT", "height_gain_m")
        print(f"  GEOM LIFT  height_gain_m : success {ok} | drop {bad}")
        ok, bad = geometry(data, "PLACE", "xy_distance_m")
        print(f"  GEOM PLACE xy_distance_m : success {ok} | slip {bad}")
        ok, bad = geometry(data, "PLACE", "z_offset_m")
        print(f"  GEOM PLACE z_offset_m    : success {ok} | slip {bad}")


if __name__ == "__main__":
    for m in MODELS:
        analyse(*m)
