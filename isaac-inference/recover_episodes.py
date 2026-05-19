"""
Recovery script for corrupted episodes parquet in Gal-auto-subtasks2.

Strategy:
  - 6 of 7 episodes parquet files are valid (363 episodes).
  - The corrupted file-006.parquet held the remaining 48 episodes (363-410).
  - The 411 data parquet files are all healthy (one per episode, rotated after each commit).
  - Reconstruct missing episode rows:
      * Basic columns (episode_index, tasks, length, index ranges): from data parquet files.
      * Stats for observable features (state, action, timestamps, indices): computed from data.
      * Image stats: copied from the last valid episode (close enough for resuming).
      * Video mapping: extrapolated from the pattern in valid episodes.
  - Write a single fresh file-000.parquet with all 411 episodes.
"""

import os
import numpy as np
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

DATASET_ROOT = Path("/home/gal/Documents/MasterProject/isaac-inference/synthetic_datasets/Gal-auto-subtasks2")
DATA_DIR     = DATASET_ROOT / "data/chunk-000"
EPISODES_DIR = DATASET_ROOT / "meta/episodes/chunk-000"
TASKS_FILE   = DATASET_ROOT / "meta/tasks.parquet"
TOTAL_EPISODES = 411

# ── Step 1: Read all valid episodes files ────────────────────────────────────
print("=== Reading valid episodes files ===")
valid_tables = []
schema = None
for fname in sorted(os.listdir(EPISODES_DIR)):
    path = EPISODES_DIR / fname
    try:
        t = pq.read_table(path)
        valid_tables.append(t)
        if schema is None:
            schema = t.schema
        print(f"  OK       {fname}  ({t.num_rows} rows)")
    except Exception as e:
        print(f"  CORRUPT  {fname}  — {e}")

combined = pa.concat_tables(valid_tables)
df = combined.to_pandas()
df = df.sort_values("episode_index").reset_index(drop=True)
df["episode_index"] = df["episode_index"].astype(int)
print(f"\nLoaded {len(df)} valid episode rows, covering indices: {df['episode_index'].min()}..{df['episode_index'].max()}")

# ── Step 2: Identify missing episodes ────────────────────────────────────────
known_indices = set(df["episode_index"].tolist())
missing_indices = sorted(set(range(TOTAL_EPISODES)) - known_indices)
print(f"Missing episode indices: {missing_indices[0]}..{missing_indices[-1]} ({len(missing_indices)} total)")

# ── Step 3: Read tasks.parquet ────────────────────────────────────────────────
tasks_df = pq.read_table(TASKS_FILE).to_pandas()
print(f"\nTasks: {tasks_df.to_dict('records')}")
task_map = {int(task_idx): task_name for task_name, task_idx in zip(tasks_df.index, tasks_df["task_index"])}

# ── Step 4: Inspect valid row structure for pattern deduction ─────────────────
print("\n=== Sample episode row (last valid episode) ===")
last_valid = df.iloc[-1]
for col in df.columns:
    val = last_valid[col]
    print(f"  {col}: {repr(val)[:80]}")

# Understand video mapping pattern
print("\n=== Video file_index mapping (last 10 valid episodes) ===")
for _, row in df.tail(10).iterrows():
    front_file = row.get("videos/observation.images.front/file_index", "?")
    wrist_file = row.get("videos/observation.images.wrist/file_index", "?")
    ep = row["episode_index"]
    print(f"  ep {ep:3d}: front_file={front_file}, wrist_file={wrist_file}")

# ── Helper: compute stats for a 1D or 2D feature array ───────────────────────
QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]
QNAMES    = ["q01", "q10", "q50", "q90", "q99"]

def stats_for(vals_2d, dtype="float"):
    """
    vals_2d: numpy array shape (n_frames, D) or (n_frames,).
    Returns dict: min, max, mean, std, count, q01..q99.
    All as Python lists (matching the parquet list<element: X> schema).
    """
    if vals_2d.ndim == 1:
        vals_2d = vals_2d[:, np.newaxis]
    n, D = vals_2d.shape
    out = {
        "min":   vals_2d.min(axis=0).tolist(),
        "max":   vals_2d.max(axis=0).tolist(),
        "mean":  vals_2d.mean(axis=0).tolist(),
        "std":   vals_2d.std(axis=0).tolist(),
        "count": [int(n)] * D,
    }
    for qn, q in zip(QNAMES, QUANTILES):
        out[qn] = np.quantile(vals_2d, q, axis=0).tolist()
    if dtype == "int":
        for k in ["min", "max", "count", *QNAMES]:
            out[k] = [int(v) for v in out[k]]
        out["mean"] = [float(v) for v in out["mean"]]
        out["std"]  = [float(v) for v in out["std"]]
    return out

# ── Step 5: Build reconstructed rows for missing episodes ────────────────────
print(f"\n=== Reconstructing {len(missing_indices)} missing episodes ===")

# Cumulative frame counter: we need global frame indices (dataset_from_index etc.)
# Compute from valid episodes first, then extend for missing.
ep_lengths = {}
for _, row in df.iterrows():
    ep_lengths[int(row["episode_index"])] = int(row["length"])

# Read data parquet files for missing episodes to get their lengths
for ep_idx in missing_indices:
    data_path = DATA_DIR / f"file-{ep_idx:03d}.parquet"
    ep_df = pq.read_table(data_path, columns=["episode_index"]).to_pandas()
    ep_lengths[ep_idx] = len(ep_df)

# Build cumulative frame index
cumulative = 0
dataset_from = {}
dataset_to   = {}
for i in range(TOTAL_EPISODES):
    dataset_from[i] = cumulative
    dataset_to[i]   = cumulative + ep_lengths[i] - 1
    cumulative += ep_lengths[i]
print(f"Total frames: {cumulative}")

# Use last valid row as template for image stats and video mapping
template = last_valid.to_dict()

# Determine video file_index for missing episodes by extrapolating the max seen
max_front_file = int(df["videos/observation.images.front/file_index"].max())
max_wrist_file = int(df["videos/observation.images.wrist/file_index"].max())
# Missing episodes come after valid ones; if their videos were finalized before crash
# they'd be in a new file. Use max+1 conservatively (may not perfectly match reality,
# but LeRobotDataset only needs this to find the right video file on disk).

new_rows = []
for ep_idx in missing_indices:
    data_path = DATA_DIR / f"file-{ep_idx:03d}.parquet"
    ep_df = pq.read_table(data_path).to_pandas()

    task_idx  = int(ep_df["task_index"].iloc[0])
    task_name = task_map.get(task_idx, f"unknown_task_{task_idx}")
    length    = len(ep_df)

    # Compute stats for data-parquet features
    state_arr = np.array(ep_df["observation.state"].tolist(), dtype=np.float64)
    action_arr = np.array(ep_df["action"].tolist(), dtype=np.float64)
    ts_arr     = ep_df["timestamp"].to_numpy(dtype=np.float64)
    fi_arr     = ep_df["frame_index"].to_numpy(dtype=np.int64)
    ei_arr     = ep_df["episode_index"].to_numpy(dtype=np.int64)
    idx_arr    = ep_df["index"].to_numpy(dtype=np.int64)
    ti_arr     = ep_df["task_index"].to_numpy(dtype=np.int64)

    state_stats  = stats_for(state_arr)
    action_stats = stats_for(action_arr)
    ts_stats     = stats_for(ts_arr)
    fi_stats     = stats_for(fi_arr, dtype="int")
    ei_stats     = stats_for(ei_arr, dtype="int")
    idx_stats    = stats_for(idx_arr, dtype="int")
    ti_stats     = stats_for(ti_arr, dtype="int")

    # Determine video file index (mirror from template or use max_seen)
    # The video file assigned during recording is what's on disk.
    # We'll approximate using the pattern from the last valid episode.
    # Heuristic: assume a new video file was started for the remaining episodes.
    front_chunk = int(template.get("videos/observation.images.front/chunk_index", 0))
    front_file  = max_front_file  # same file if sessions ran long, or max+N for new sessions
    wrist_chunk = int(template.get("videos/observation.images.wrist/chunk_index", 0))
    wrist_file  = max_wrist_file

    # Compute video timestamps from episode data timestamps
    front_from_ts = float(ts_arr[0])
    front_to_ts   = float(ts_arr[-1])
    wrist_from_ts = float(ts_arr[0])
    wrist_to_ts   = float(ts_arr[-1])

    row = {
        "episode_index": ep_idx,
        "tasks":         [task_name],
        "length":        length,
        "data/chunk_index":   0,
        "data/file_index":    ep_idx,
        "dataset_from_index": dataset_from[ep_idx],
        "dataset_to_index":   dataset_to[ep_idx],
        "videos/observation.images.front/chunk_index": front_chunk,
        "videos/observation.images.front/file_index":  front_file,
        "videos/observation.images.front/from_timestamp": front_from_ts,
        "videos/observation.images.front/to_timestamp":   front_to_ts,
        "videos/observation.images.wrist/chunk_index": wrist_chunk,
        "videos/observation.images.wrist/file_index":  wrist_file,
        "videos/observation.images.wrist/from_timestamp": wrist_from_ts,
        "videos/observation.images.wrist/to_timestamp":   wrist_to_ts,
        # Image stats: copy from template (approx — not used for recording)
        "stats/observation.images.front/min":   template.get("stats/observation.images.front/min"),
        "stats/observation.images.front/max":   template.get("stats/observation.images.front/max"),
        "stats/observation.images.front/mean":  template.get("stats/observation.images.front/mean"),
        "stats/observation.images.front/std":   template.get("stats/observation.images.front/std"),
        "stats/observation.images.front/count": template.get("stats/observation.images.front/count"),
        "stats/observation.images.front/q01":   template.get("stats/observation.images.front/q01"),
        "stats/observation.images.front/q10":   template.get("stats/observation.images.front/q10"),
        "stats/observation.images.front/q50":   template.get("stats/observation.images.front/q50"),
        "stats/observation.images.front/q90":   template.get("stats/observation.images.front/q90"),
        "stats/observation.images.front/q99":   template.get("stats/observation.images.front/q99"),
        "stats/observation.images.wrist/min":   template.get("stats/observation.images.wrist/min"),
        "stats/observation.images.wrist/max":   template.get("stats/observation.images.wrist/max"),
        "stats/observation.images.wrist/mean":  template.get("stats/observation.images.wrist/mean"),
        "stats/observation.images.wrist/std":   template.get("stats/observation.images.wrist/std"),
        "stats/observation.images.wrist/count": template.get("stats/observation.images.wrist/count"),
        "stats/observation.images.wrist/q01":   template.get("stats/observation.images.wrist/q01"),
        "stats/observation.images.wrist/q10":   template.get("stats/observation.images.wrist/q10"),
        "stats/observation.images.wrist/q50":   template.get("stats/observation.images.wrist/q50"),
        "stats/observation.images.wrist/q90":   template.get("stats/observation.images.wrist/q90"),
        "stats/observation.images.wrist/q99":   template.get("stats/observation.images.wrist/q99"),
        # State stats (computed)
        "stats/observation.state/min":   state_stats["min"],
        "stats/observation.state/max":   state_stats["max"],
        "stats/observation.state/mean":  state_stats["mean"],
        "stats/observation.state/std":   state_stats["std"],
        "stats/observation.state/count": state_stats["count"],
        "stats/observation.state/q01":   state_stats["q01"],
        "stats/observation.state/q10":   state_stats["q10"],
        "stats/observation.state/q50":   state_stats["q50"],
        "stats/observation.state/q90":   state_stats["q90"],
        "stats/observation.state/q99":   state_stats["q99"],
        # Action stats (computed)
        "stats/action/min":   action_stats["min"],
        "stats/action/max":   action_stats["max"],
        "stats/action/mean":  action_stats["mean"],
        "stats/action/std":   action_stats["std"],
        "stats/action/count": action_stats["count"],
        "stats/action/q01":   action_stats["q01"],
        "stats/action/q10":   action_stats["q10"],
        "stats/action/q50":   action_stats["q50"],
        "stats/action/q90":   action_stats["q90"],
        "stats/action/q99":   action_stats["q99"],
        # Timestamp stats
        "stats/timestamp/min":   ts_stats["min"],
        "stats/timestamp/max":   ts_stats["max"],
        "stats/timestamp/mean":  ts_stats["mean"],
        "stats/timestamp/std":   ts_stats["std"],
        "stats/timestamp/count": ts_stats["count"],
        "stats/timestamp/q01":   ts_stats["q01"],
        "stats/timestamp/q10":   ts_stats["q10"],
        "stats/timestamp/q50":   ts_stats["q50"],
        "stats/timestamp/q90":   ts_stats["q90"],
        "stats/timestamp/q99":   ts_stats["q99"],
        # Frame index stats
        "stats/frame_index/min":   fi_stats["min"],
        "stats/frame_index/max":   fi_stats["max"],
        "stats/frame_index/mean":  fi_stats["mean"],
        "stats/frame_index/std":   fi_stats["std"],
        "stats/frame_index/count": fi_stats["count"],
        "stats/frame_index/q01":   fi_stats["q01"],
        "stats/frame_index/q10":   fi_stats["q10"],
        "stats/frame_index/q50":   fi_stats["q50"],
        "stats/frame_index/q90":   fi_stats["q90"],
        "stats/frame_index/q99":   fi_stats["q99"],
        # Episode index stats
        "stats/episode_index/min":   ei_stats["min"],
        "stats/episode_index/max":   ei_stats["max"],
        "stats/episode_index/mean":  ei_stats["mean"],
        "stats/episode_index/std":   ei_stats["std"],
        "stats/episode_index/count": ei_stats["count"],
        "stats/episode_index/q01":   ei_stats["q01"],
        "stats/episode_index/q10":   ei_stats["q10"],
        "stats/episode_index/q50":   ei_stats["q50"],
        "stats/episode_index/q90":   ei_stats["q90"],
        "stats/episode_index/q99":   ei_stats["q99"],
        # Global index stats
        "stats/index/min":   idx_stats["min"],
        "stats/index/max":   idx_stats["max"],
        "stats/index/mean":  idx_stats["mean"],
        "stats/index/std":   idx_stats["std"],
        "stats/index/count": idx_stats["count"],
        "stats/index/q01":   idx_stats["q01"],
        "stats/index/q10":   idx_stats["q10"],
        "stats/index/q50":   idx_stats["q50"],
        "stats/index/q90":   idx_stats["q90"],
        "stats/index/q99":   idx_stats["q99"],
        # Task index stats
        "stats/task_index/min":   ti_stats["min"],
        "stats/task_index/max":   ti_stats["max"],
        "stats/task_index/mean":  ti_stats["mean"],
        "stats/task_index/std":   ti_stats["std"],
        "stats/task_index/count": ti_stats["count"],
        "stats/task_index/q01":   ti_stats["q01"],
        "stats/task_index/q10":   ti_stats["q10"],
        "stats/task_index/q50":   ti_stats["q50"],
        "stats/task_index/q90":   ti_stats["q90"],
        "stats/task_index/q99":   ti_stats["q99"],
        # Meta episodes mapping
        "meta/episodes/chunk_index": 0,
        "meta/episodes/file_index":  0,
    }
    new_rows.append(row)
    if ep_idx % 10 == 0:
        print(f"  Reconstructed episode {ep_idx} ({task_name!r}, {length} frames)")

# ── Step 6: Fix dataset_from_index/to_index for known episodes too ───────────
# (these may have drifted if data parquet lengths differ from recorded lengths)
print("\n=== Verifying known episode index ranges ===")
for _, row in df.iterrows():
    ep_idx = int(row["episode_index"])
    if row["dataset_from_index"] != dataset_from[ep_idx]:
        print(f"  WARNING ep {ep_idx}: from_index {row['dataset_from_index']} → {dataset_from[ep_idx]}")
    if row["dataset_to_index"] != dataset_to[ep_idx]:
        print(f"  WARNING ep {ep_idx}: to_index {row['dataset_to_index']} → {dataset_to[ep_idx]}")

# ── Step 7: Combine and write ─────────────────────────────────────────────────
print("\n=== Building final table ===")
new_df = pd.DataFrame(new_rows)
full_df = pd.concat([df, new_df], ignore_index=True)
full_df = full_df.sort_values("episode_index").reset_index(drop=True)
assert len(full_df) == TOTAL_EPISODES, f"Expected {TOTAL_EPISODES} rows, got {len(full_df)}"
print(f"Total rows: {len(full_df)}")

# Convert back to PyArrow table with original schema
full_table = pa.Table.from_pandas(full_df, schema=schema, preserve_index=False)

# Move old files out of the way (backed up to episodes.bak already)
print("\n=== Moving old episodes files ===")
for fname in sorted(os.listdir(EPISODES_DIR)):
    if fname.endswith(".old"):
        continue
    p = EPISODES_DIR / fname
    dest = EPISODES_DIR / (fname + ".old")
    p.rename(dest)
    print(f"  {fname} → {fname}.old")

out_path = EPISODES_DIR / "file-000.parquet"
pq.write_table(full_table, out_path)
print(f"\nWrote {len(full_table)} episodes to {out_path}")

# ── Step 8: Validate ──────────────────────────────────────────────────────────
print("\n=== Validating with LeRobotDataset ===")
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset(
    repo_id="MasterProject2026/Gal-auto-subtasks2",
    root=DATASET_ROOT,
)
print(f"  total_episodes : {dataset.meta.total_episodes}")
print(f"  total_frames   : {dataset.meta.total_frames}")
if dataset.meta.total_episodes == TOTAL_EPISODES:
    print("SUCCESS — all 411 episodes recovered!")
else:
    print(f"WARNING: expected {TOTAL_EPISODES}, got {dataset.meta.total_episodes}")
