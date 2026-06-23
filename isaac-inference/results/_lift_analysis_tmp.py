"""Throwaway lift-failure / recovery analysis v2 (corrected). Monotask vs subtask.

Reads git-tracked eval checkpoints. Pure stdlib. NOT committed; delete when done.
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

MODELS = [
    ("Teleop  monotask", "Gal_split_nolang", "flat_checkpoint.json", "monotask"),
    ("Teleop  subtask ", "Gal-pick-orange-tailedCH20", "checkpoint.json", "subtask"),
    ("Tel+Auto monotask", "Gal-merged-tailed-auto-no-lang-no-home", "flat_checkpoint.json", "monotask"),
    ("Tel+Auto subtask ", "Gal-merged-tailed-auto", "checkpoint.json", "subtask"),
]

BOOK = {"env_truncated", "episode_ended", "interrupted_by_new_attempt",
        "no_confirmed_progress", "plate_flipped"}


def target_of(a):
    return a.get("actual_orange") or a.get("requested_orange") or a.get("inferred_target_orange")


def classify_lift(a):
    r, fr = a.get("result"), a.get("failure_reason")
    if r == "success":
        return "success"
    if r == "skipped":
        return "skip"            # inferred_place_without_lift -> placed, so lifted
    if fr in BOOK:
        return "book"
    if r in ("failure", "timeout"):
        return "drop"
    return "book"


def ever_placed_oranges(e):
    """Oranges that were in the plate at any point: place_success events + final scene."""
    placed = set()
    for ev in e.get("timeline", []):
        if ev.get("event_type") == "place_success":
            o = ev.get("actual_orange") or ev.get("requested_orange") or ev.get("inferred_target_orange")
            if o:
                placed.add(o)
    for name, info in e.get("final_scene", {}).get("oranges", {}).items():
        if info.get("status") == "placed":
            placed.add(name)
    return placed


def analyse(display, subdir, fname, formulation):
    data = json.load(open(HERE / subdir / fname))
    eps = data["episodes"]
    n = len(eps)

    lift_outcomes = Counter()
    result_timeout = 0

    # per-orange "never lifted"
    org_grasped = org_lifted = 0

    # Q2: oranges whose FIRST genuine lift attempt is a drop
    ff_total = 0                 # qualifying oranges
    ff_recovered = 0             # eventually lifted (success/skip/placed)
    ff_recov_via_place_only = 0  # eventually placed but no detected lift-success attempt
    retries = []                 # among recovered-with-lift-success: attempts after the first

    # merged Q1/Q4: after a drop, next grasp-success OR lift-start on same orange?
    reeng_same = reeng_diff = reeng_none = 0

    for e in eps:
        atts = sorted(e["subtask_attempts"], key=lambda a: a.get("start_step", 0))
        lifts = [a for a in atts if a.get("subtask") == "LIFT"]
        placed = ever_placed_oranges(e)

        # per-orange ordered lift-attempt classifications (exclude book)
        per_org = {}
        for a in lifts:
            c = classify_lift(a)
            lift_outcomes[c] += 1
            if a.get("result") == "timeout":
                result_timeout += 1
            if c == "book":
                continue
            per_org.setdefault(target_of(a), []).append(c)

        for o, seq in per_org.items():
            org_grasped += 1
            lifted = ("success" in seq) or ("skip" in seq) or (o in placed)
            org_lifted += int(lifted)

            # Q2: first genuine lift attempt is a drop
            if seq and seq[0] == "drop":
                ff_total += 1
                if lifted:
                    ff_recovered += 1
                    # first success/skip position within this orange's lift seq
                    pos = next((i for i, c in enumerate(seq) if c in ("success", "skip")), None)
                    if pos is not None:
                        retries.append(pos)         # = # attempts before success (all failed) = retries
                    else:
                        ff_recov_via_place_only += 1  # placed but no lift-success attempt

        # merged Q1/Q4: walk attempts; after each drop find next grasp-success or lift-start
        for i, a in enumerate(atts):
            if a.get("subtask") != "LIFT" or classify_lift(a) != "drop":
                continue
            o = target_of(a)
            nxt = None
            for b in atts[i + 1:]:
                if b.get("subtask") == "LIFT":
                    nxt = b; break
                if b.get("subtask") == "GRASP" and b.get("result") == "success":
                    nxt = b; break
            if nxt is None:
                reeng_none += 1
            elif target_of(nxt) == o:
                reeng_same += 1
            else:
                reeng_diff += 1

    # ---- report ----
    succ, drop = lift_outcomes["success"], lift_outcomes["drop"]
    skip, book = lift_outcomes["skip"], lift_outcomes["book"]
    genuine = succ + drop
    print(f"\n================  {display}  ({formulation}, {n} ep)  ================")
    print(f"LIFT attempts: success {succ} | drop {drop} | skip(place-w/o-lift) {skip} | bookkeeping {book}")
    print(f"  GLOBAL drop rate (drops / genuine attempts): {drop}/{genuine} = {pct(drop,genuine)}"
          + (f"   [{result_timeout} timeouts]" if result_timeout else ""))

    print(f"\n  Q3 per distinct orange grasped-into-lift: "
          f"never lifted {org_grasped-org_lifted}/{org_grasped} = {pct(org_grasped-org_lifted,org_grasped)}"
          f"   (ever lifted {pct(org_lifted,org_grasped)}; place counts as lifted)")

    print(f"\n  Q2 oranges whose FIRST lift attempt failed: {ff_total}")
    print(f"     eventually lifted later in episode: {ff_recovered}/{ff_total} = {pct(ff_recovered,ff_total)}"
          f"   (recovery rate)")
    if ff_recov_via_place_only:
        print(f"       (of those, {ff_recov_via_place_only} placed w/o a detected lift-success attempt)")
    if retries:
        dist = Counter(retries)
        m = sum(retries) / len(retries)
        print(f"     avg retries to first lift-success (not counting the first attempt): {m:.2f}"
              f"   over {len(retries)} oranges")
        print(f"       distribution: " + ", ".join(f"{k} retr->{dist[k]}" for k in sorted(dist)))

    print(f"\n  Q1=Q4 after a failed lift, next grasp-success/lift-start is on the SAME orange:")
    tot = reeng_same + reeng_diff
    print(f"     same {reeng_same} | different {reeng_diff} | no re-engagement {reeng_none}"
          f"   -> same = {pct(reeng_same,tot)} (of those that re-engaged)")


def pct(a, b):
    return f"{100*a/b:.1f}%" if b else "n/a"


if __name__ == "__main__":
    for m in MODELS:
        analyse(*m)
