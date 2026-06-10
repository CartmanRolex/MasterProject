"""
Analyze local retry and grasp attempt statistics from checkpoint.json.

For each episode, per orange:
  - local retries = number of local_retry timeline events for that orange
    (slip during LIFT/PLACE → back to GRASP for same orange)
  - total grasp attempts = number of GRASP subtask_attempts for that orange
  - placement order = 1st / 2nd / 3rd (from n_placed_start of successful PLACE)

Stats are broken down overall and by placement order.
"""

import json
from collections import defaultdict
import statistics

CHECKPOINT = "/home/students/Gal/MasterProject/isaac-inference/results/Gal-pick-orange-tailedCH20/checkpoint.json"

with open(CHECKPOINT) as f:
    data = json.load(f)

episodes = data["episodes"]

# Each record: {local_retries, grasp_attempts, placed, placement_order (1/2/3 or None)}
records = []  # one per (episode, orange) that had at least one GRASP attempt

for ep in episodes:
    ep_idx = ep["episode"]
    attempts = ep.get("subtask_attempts", [])
    timeline = ep.get("timeline", [])

    # Count local_retry events per orange
    local_retries_per_orange = defaultdict(int)
    for event in timeline:
        if event.get("event_type") == "local_retry":
            orange = event.get("requested_orange")
            if orange:
                local_retries_per_orange[orange] += 1

    # Gather all GRASP attempts per orange and their placement order
    grasp_attempts_per_orange = defaultdict(int)
    placement_order_per_orange = {}  # orange → 1-based order (from n_placed_start of successful PLACE)

    for att in attempts:
        subtask = att.get("subtask")
        orange = att.get("requested_orange")
        if not orange:
            continue

        if subtask == "GRASP":
            grasp_attempts_per_orange[orange] += 1

        if subtask == "PLACE" and att.get("result") == "success":
            order = att.get("n_placed_start", 0) + 1  # 0 already placed → 1st, etc.
            placement_order_per_orange[orange] = order

    # Collect all oranges that were attempted in this episode
    all_oranges = set(grasp_attempts_per_orange.keys())
    for orange in all_oranges:
        records.append({
            "episode": ep_idx,
            "orange": orange,
            "local_retries": local_retries_per_orange[orange],
            "grasp_attempts": grasp_attempts_per_orange[orange],
            "placed": orange in placement_order_per_orange,
            "placement_order": placement_order_per_orange.get(orange),  # None if not placed
        })


def summarize(subset, label):
    if not subset:
        print(f"\n{label}: no data")
        return
    n = len(subset)
    retries = [r["local_retries"] for r in subset]
    grasps = [r["grasp_attempts"] for r in subset]
    placed_count = sum(1 for r in subset if r["placed"])

    def fmt(vals):
        mn = min(vals)
        mx = max(vals)
        mean = statistics.mean(vals)
        median = statistics.median(vals)
        stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"mean={mean:.2f}  median={median:.1f}  stdev={stdev:.2f}  min={mn}  max={mx}"

    print(f"\n{'─'*60}")
    print(f"  {label}  (n={n} orange-episodes, placed={placed_count}/{n} = {100*placed_count/n:.1f}%)")
    print(f"{'─'*60}")
    print(f"  Local retries (back to GRASP after slip):")
    print(f"    {fmt(retries)}")
    dist = defaultdict(int)
    for v in retries:
        dist[v] += 1
    print(f"    Distribution: {dict(sorted(dist.items()))}")
    print(f"  Total GRASP attempts per orange:")
    print(f"    {fmt(grasps)}")
    dist2 = defaultdict(int)
    for v in grasps:
        dist2[v] += 1
    print(f"    Distribution: {dict(sorted(dist2.items()))}")


print("=" * 60)
print("  RETRY / GRASP ATTEMPT STATISTICS")
print(f"  Model: {data['model_id']}")
print(f"  Episodes: {data['completed_episodes']}")
print("=" * 60)

summarize(records, "ALL oranges")

for order in [1, 2, 3]:
    subset = [r for r in records if r["placement_order"] == order]
    # Also include unplaced oranges whose intended order we can infer from
    # their first GRASP n_placed_start. Grab those from the episodes directly.
    summarize(subset, f"Successfully placed as #{order}")

# For unplaced, we can only report their stats without order
unplaced = [r for r in records if not r["placed"]]
summarize(unplaced, "NOT placed (abandoned/timeout)")

# Breakdown by intended order (first GRASP attempt's n_placed_start)
print("\n\n--- Breakdown by INTENDED placement order (n_placed_start of first GRASP) ---")

intended_order_records = defaultdict(list)
for ep in episodes:
    ep_idx = ep["episode"]
    attempts = ep.get("subtask_attempts", [])
    timeline = ep.get("timeline", [])

    local_retries_per_orange = defaultdict(int)
    for event in timeline:
        if event.get("event_type") == "local_retry":
            orange = event.get("requested_orange")
            if orange:
                local_retries_per_orange[orange] += 1

    grasp_attempts_per_orange = defaultdict(int)
    first_grasp_order = {}
    placement_order_per_orange = {}

    for att in attempts:
        subtask = att.get("subtask")
        orange = att.get("requested_orange")
        if not orange:
            continue
        if subtask == "GRASP":
            grasp_attempts_per_orange[orange] += 1
            if orange not in first_grasp_order:
                first_grasp_order[orange] = att.get("n_placed_start", 0) + 1
        if subtask == "PLACE" and att.get("result") == "success":
            placement_order_per_orange[orange] = att.get("n_placed_start", 0) + 1

    for orange, intended in first_grasp_order.items():
        intended_order_records[intended].append({
            "episode": ep_idx,
            "orange": orange,
            "local_retries": local_retries_per_orange[orange],
            "grasp_attempts": grasp_attempts_per_orange[orange],
            "placed": orange in placement_order_per_orange,
        })

for order in [1, 2, 3]:
    subset = intended_order_records[order]
    summarize(subset, f"INTENDED as #{order} (first attempted when {order-1} already placed)")
