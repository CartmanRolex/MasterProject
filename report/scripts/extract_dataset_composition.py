"""Extract dataset-composition statistics from the LeRobot training datasets.

Run on the DESKTOP, where the LeRobot datasets live in the HuggingFace cache
(``~/.cache/huggingface/lerobot/MasterProject2026``). It reads the merged
subtask training set ``Gal-merged-tailed-auto`` (1260 episodes), classifies each
episode as GRASP / LIFT / PLACE from its language task string, and splits the
Teleop part (episode indices 0-845) from the Auto part (846-1259), which is the
order ``merge_datasets.py`` concatenates them in.

It prints the per-source x per-subtask episode-length summary used by
``plot_dataset_composition.py``. Episode lengths include the 20-frame terminal
hold tail that the subtask training set appends to every episode.

This script needs ``pyarrow`` + ``numpy`` (the ``lerobot`` conda env on the
desktop). The plotting script does NOT depend on it: the measured numbers are
baked into ``plot_dataset_composition.py`` so the figure regenerates on the
laptop without the datasets present.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DATASET_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / "MasterProject2026"
MERGED = "Gal-merged-tailed-auto"
TELEOP_EPISODES = 846  # episodes 0-845 are Teleop; 846-1259 are Auto


def classify(task_string: str) -> str:
    s = task_string.lower()
    if s.startswith("grasp"):
        return "GRASP"
    if s.startswith("pick"):
        return "LIFT"
    if s.startswith("place"):
        return "PLACE"
    if "back to start" in s:
        return "HOME"
    return "OTHER"


def main() -> None:
    dataset_dir = DATASET_ROOT / MERGED
    tasks = {
        row["task_index"]: row["__index_level_0__"]
        for row in pq.read_table(dataset_dir / "meta" / "tasks.parquet").to_pylist()
    }
    subtask_of = {idx: classify(text) for idx, text in tasks.items()}

    files = sorted(glob.glob(str(dataset_dir / "data" / "**" / "*.parquet"), recursive=True))
    table = pa.concat_tables([pq.read_table(f) for f in files])
    episode_index = np.array(table.column("episode_index"))
    task_index = np.array(table.column("task_index"))

    rows = []
    for episode in np.unique(episode_index):
        mask = episode_index == episode
        dominant_task = int(np.bincount(task_index[mask]).argmax())
        source = "Teleop" if episode < TELEOP_EPISODES else "Auto"
        rows.append((source, subtask_of[dominant_task], int(mask.sum())))

    summary = {}
    for source in ("Teleop", "Auto"):
        for subtask in ("GRASP", "LIFT", "PLACE"):
            lengths = np.array(
                [n for src, sub, n in rows if src == source and sub == subtask],
                dtype=float,
            )
            if lengths.size == 0:
                continue
            summary[f"{source}/{subtask}"] = {
                "n": int(lengths.size),
                "mean": round(float(lengths.mean()), 1),
                "p5": round(float(np.percentile(lengths, 5)), 1),
                "p25": round(float(np.percentile(lengths, 25)), 1),
                "p50": round(float(np.percentile(lengths, 50)), 1),
                "p75": round(float(np.percentile(lengths, 75)), 1),
                "p95": round(float(np.percentile(lengths, 95)), 1),
                "min": int(lengths.min()),
                "max": int(lengths.max()),
            }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
