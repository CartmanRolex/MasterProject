"""Lift-subtask statistics: monotask vs subtask, two datasets (4 models).

Reads git-tracked eval checkpoints (pure stdlib). Definitions (confirmed):

  * Overall lift success rate = successes / (successes + drops), per LIFT attempt.
    Bookkeeping aborts (episode_ended / env_truncated / interrupted) and timeouts
    are EXCLUDED from the denominator. Timeouts are reported separately.

  * drop = result 'failure' with failure_reason 'dropped_during_lift'.

  * Recovery population = distinct oranges whose FIRST genuine lift attempt
    (success or drop, ignoring bookkeeping) was a drop. Of those, what fraction
    were eventually lifted later in the same episode. An orange that was placed
    counts as lifted (edge case: place implies a successful lift).

  * Retries = number of failed (dropped) LIFT attempts on that orange before the
    first successful lift. Averaged over recovered oranges that had a lift-success
    (place-only recoveries reported separately, no countable retries).
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

MODELS = [
    ("Teleop  / monotask", "Gal_split_nolang", "flat_checkpoint.json"),
    ("Teleop  / subtask ", "Gal-pick-orange-tailedCH20", "checkpoint.json"),
    ("Tel+Auto/ monotask", "Gal-merged-tailed-auto-no-lang-no-home", "flat_checkpoint.json"),
    ("Tel+Auto/ subtask ", "Gal-merged-tailed-auto", "checkpoint.json"),
]

DROP_REASON = "dropped_during_lift"


def target_of(a):
    return a.get("actual_orange") or a.get("requested_orange") or a.get("inferred_target_orange")


def classify(a):
    """success | drop | timeout | book  (for LIFT attempts only)."""
    r, fr = a.get("result"), a.get("failure_reason")
    if r == "success":
        return "success"
    if r == "skipped":
        return "success"          # place-without-lift -> treated as lifted
    if r == "timeout":
        return "timeout"
    if r == "failure" and fr == DROP_REASON:
        return "drop"
    return "book"                 # episode_ended, env_truncated, interrupted, ...


def ever_placed(e):
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


def pct(a, b):
    return f"{100*a/b:5.1f}%" if b else "  n/a "


def analyse(display, subdir, fname):
    data = json.load(open(HERE / subdir / fname))
    eps = data["episodes"]

    outcomes = Counter()                       # success / drop / timeout / book

    ff_total = 0                               # oranges whose first genuine lift dropped
    ff_recovered = 0                           # ... eventually lifted
    ff_place_only = 0                          # ... recovered but no lift-success attempt
    retries = []                               # failed lifts before first success (recovered w/ lift)

    for e in eps:
        atts = sorted(e["subtask_attempts"], key=lambda a: a.get("start_step", 0))
        placed = ever_placed(e)

        per_org = {}                           # orange -> ordered genuine lift outcomes
        for a in atts:
            if a.get("subtask") != "LIFT":
                continue
            c = classify(a)
            outcomes[c] += 1
            if c in ("success", "drop"):       # genuine, target-attributable
                per_org.setdefault(target_of(a), []).append(c)

        for o, seq in per_org.items():
            if not seq or seq[0] != "drop":
                continue
            ff_total += 1
            lifted = ("success" in seq) or (o in placed)
            if not lifted:
                continue
            ff_recovered += 1
            first_succ = next((i for i, c in enumerate(seq) if c == "success"), None)
            if first_succ is None:
                ff_place_only += 1             # placed but never a detected lift-success
            else:
                retries.append(first_succ)     # drops before the success = retries

    succ, drop, to, book = (outcomes["success"], outcomes["drop"],
                            outcomes["timeout"], outcomes["book"])
    genuine = succ + drop

    print(f"\n===========  {display}  ({len(eps)} episodes)  ===========")
    print(f"  LIFT attempts:  success {succ:3d} | drop {drop:3d} | "
          f"timeout {to:3d} | bookkeeping {book:3d}")
    print(f"  Overall lift success rate (succ / succ+drop):  {pct(succ, genuine)}"
          f"   ({succ}/{genuine})")
    print(f"  Recovery: oranges whose 1st lift dropped:      {ff_total}")
    print(f"            of those eventually lifted:          {pct(ff_recovered, ff_total)}"
          f"   ({ff_recovered}/{ff_total})"
          + (f"   [{ff_place_only} via place only]" if ff_place_only else ""))
    if retries:
        dist = Counter(retries)
        print(f"  Avg retries before the successful lift:        {sum(retries)/len(retries):.2f}"
              f"   (over {len(retries)} recovered oranges)")
        print("            retry distribution:  "
              + ", ".join(f"{k}->{dist[k]}" for k in sorted(dist)))


if __name__ == "__main__":
    for m in MODELS:
        analyse(*m)
    print()
