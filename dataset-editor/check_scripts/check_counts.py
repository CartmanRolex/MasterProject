import json
import sys
from collections import Counter


EXCLUDED_TASK = "Go back to start position"
EXPECTED_COUNT = 3


def check_episodes(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    flagged = []

    for ep_id, episode in data["episodes"].items():
        tasks = [edit["task"] for edit in episode["edits"] if edit["task"] != EXCLUDED_TASK]
        counts = Counter(tasks)

        bad_tasks = {task: count for task, count in counts.items() if count != EXPECTED_COUNT}
        if bad_tasks:
            flagged.append((ep_id, bad_tasks))

    if not flagged:
        print("All episodes look good — every task appears exactly 3 times.")
    else:
        print(f"Found {len(flagged)} episode(s) with task count issues:\n")
        for ep_id, bad_tasks in flagged:
            print(f"  Episode {ep_id}:")
            for task, count in bad_tasks.items():
                print(f"    '{task}' appears {count}x (expected {EXPECTED_COUNT})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_task_counts.py <path_to_json>")
        sys.exit(1)
    check_episodes(sys.argv[1])