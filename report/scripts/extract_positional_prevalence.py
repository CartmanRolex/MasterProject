"""Extract positional-instruction prevalence from the LeRobot training datasets.

Runs anywhere with network access --- it pulls only the two small metadata
files from the HuggingFace dataset repo (a few tens of kB total), never the
videos or per-frame data. For each training episode it reads the language task
string and, when that string is a GRASP instruction (built as
``"Grasp {label} orange"``, see ``build_task_prompt`` in
``inference_autonomous_orders.py``), extracts the spatial label, e.g.
``"Grasp bottom right orange"`` -> ``bottom right``. It counts how many GRASP
episodes carry each label, split into the Teleop part (episode indices 0-845)
and the Auto part (846-1259), the order ``merge_datasets.py`` concatenates them.

The point is to relate how often each positional referent appears in training to
how reliably the policy obeys it at evaluation (the obedience table in the
Results section). The measured numbers are baked into the report by hand.

Needs ``huggingface_hub`` + ``pyarrow``. Xet transfer is disabled because it
hangs on this network; the plain-HTTPS path serves these small files instantly.

    python extract_positional_prevalence.py
"""

from __future__ import annotations

import json
import os
from collections import Counter

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "MasterProject2026/Gal-merged-tailed-auto"
TELEOP_EPISODES = 846  # episodes 0-845 are Teleop; 846-1259 are Auto
EPISODES_META = "meta/episodes/chunk-000/file-000.parquet"


def grasp_label(task_string: str) -> str | None:
    """Positional label of a GRASP task string, or None if it is not a GRASP."""
    s = task_string.strip().lower()
    if not s.startswith("grasp"):
        return None
    s = s[len("grasp"):].strip()
    if s.endswith("orange"):
        s = s[: -len("orange")].strip()
    return s or "unspecified"


def main() -> None:
    path = hf_hub_download(REPO, EPISODES_META, repo_type="dataset")
    episodes = pq.read_table(path).to_pylist()

    counts = {"Teleop": Counter(), "Auto": Counter()}
    for ep in episodes:
        source = "Teleop" if ep["episode_index"] < TELEOP_EPISODES else "Auto"
        for task in ep["tasks"]:
            label = grasp_label(task)
            if label is not None:
                counts[source][label] += 1

    labels = sorted(
        set(counts["Teleop"]) | set(counts["Auto"]),
        key=lambda l: -(counts["Teleop"][l] + counts["Auto"][l]),
    )
    summary = {
        label: {
            "Teleop": counts["Teleop"].get(label, 0),
            "Auto": counts["Auto"].get(label, 0),
            "Combined": counts["Teleop"].get(label, 0) + counts["Auto"].get(label, 0),
        }
        for label in labels
    }
    summary["_totals"] = {
        "Teleop": sum(counts["Teleop"].values()),
        "Auto": sum(counts["Auto"].values()),
        "Combined": sum(counts["Teleop"].values()) + sum(counts["Auto"].values()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
