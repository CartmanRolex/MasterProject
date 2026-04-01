import json
import sys
from collections import Counter


def analyze_tasks(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    all_tasks = [
        edit["task"]
        for episode in data["episodes"].values()
        for edit in episode["edits"]
    ]

    unique_tasks = sorted(set(all_tasks))
    counts = Counter(all_tasks)

    print(f"Total task instances : {len(all_tasks)}")
    print(f"Unique task strings  : {len(unique_tasks)}")
    print()
    for task in unique_tasks:
        print(f"  [{counts[task]:>3}x]  {task}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_tasks.py <path_to_json>")
        sys.exit(1)
    analyze_tasks(sys.argv[1])