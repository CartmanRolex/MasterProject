#!/usr/bin/env python3
"""
strip_lang_and_tail.py — produce a no-language-conditioning clone of a dataset
with the trailing N "frozen frames" of every episode removed.

Source: synthetic_datasets/Gal-merged-tailed-auto (1335 episodes, 191257 frames, 16 tasks).
Destination: synthetic_datasets/Gal-merged-tailed-auto-no-lang-no-home.

What changes:
  - Episodes labelled "Go back to start position" are removed.
  - All task labels collapse to a single string: "Place the orange into plate".
  - Last TAIL=20 frames of every kept episode are dropped from data parquet.
  - episodes meta: length, dataset_from/to_index, tasks, videos/*/to_timestamp updated.
  - tasks.parquet rewritten with a single row.
  - info.json: total_frames, total_episodes, splits, and total_tasks updated.
  - stats.json: scalar feature stats recomputed; image stats preserved (videos untouched).

What does NOT change:
  - Videos: chunk mp4s are symlinked from source. The trailing 20 frames of each
    episode become an unread gap between consecutive episodes' video slices.
  - Per-frame timestamp/frame_index values for kept rows.
  - Episode video from_timestamp (each episode still starts at the same offset
    in the shared chunk mp4).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from balance_dataset import CAMERAS

ROOT = Path(__file__).parent
SRC = ROOT.parent / "synthetic_datasets" / "Gal-merged-tailed-auto"
DST = ROOT.parent / "synthetic_datasets" / "Gal-merged-tailed-auto-no-lang-no-home"
TAIL = 20
NEW_TASK = "Place the orange into plate"
DROP_TASKS = {"Go back to start position"}
EXPECTED_KEPT_EPISODES = 1260
EXPECTED_TOTAL_FRAMES = 155357


def replace_col(t: pa.Table, name: str, values, typ) -> pa.Table:
    return t.set_column(t.schema.get_field_index(name), name, pa.array(values, type=typ))


def task_tuple(tasks) -> tuple[str, ...]:
    return tuple(str(task) for task in tasks)


def load_episode_filter(src_path: Path) -> tuple[dict[int, int], int, int, int]:
    """Return old->new episode ids plus source/drop/kept raw frame counts."""
    eps = pq.read_table(src_path)
    old_episode_indices = eps.column("episode_index").to_pylist()
    tasks = eps.column("tasks").to_pylist()
    lengths = eps.column("length").to_pylist()

    old_to_new: dict[int, int] = {}
    dropped_frames = 0
    kept_raw_frames = 0
    for old_ep, task_list, length in zip(old_episode_indices, tasks, lengths):
        if task_tuple(task_list) in {(task,) for task in DROP_TASKS}:
            dropped_frames += length
            continue
        old_to_new[int(old_ep)] = len(old_to_new)
        kept_raw_frames += length

    return old_to_new, len(old_episode_indices), dropped_frames, kept_raw_frames


def link_videos(src: Path, dst: Path) -> None:
    """Symlink each mp4 from src/videos/ into dst/videos/, preserving the tree."""
    src_videos = src / "videos"
    dst_videos = dst / "videos"
    for cam_dir in sorted(src_videos.iterdir()):
        if not cam_dir.is_dir():
            continue
        for chunk_dir in sorted(cam_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue
            out_chunk = dst_videos / cam_dir.name / chunk_dir.name
            out_chunk.mkdir(parents=True, exist_ok=True)
            for mp4 in sorted(chunk_dir.glob("*.mp4")):
                link = out_chunk / mp4.name
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(mp4.resolve())


def rewrite_data_parquet(
    src_path: Path,
    dst_path: Path,
    tail: int,
    old_to_new_episode: dict[int, int],
) -> int:
    """Drop filtered episodes and per-episode tail, then reset episode/task/global indices."""
    data = pq.read_table(src_path)
    ep_col = data.column("episode_index").to_pylist()

    # Per-episode total counts, then keep rows with per-episode-counter < length - tail.
    totals: dict[int, int] = {}
    for ep in ep_col:
        totals[ep] = totals.get(ep, 0) + 1

    keep_mask: list[bool] = []
    new_episode_indices: list[int] = []
    seen: dict[int, int] = {}
    for ep in ep_col:
        c = seen.get(ep, 0)
        keep = ep in old_to_new_episode and c < totals[ep] - tail
        keep_mask.append(keep)
        if keep:
            new_episode_indices.append(old_to_new_episode[ep])
        seen[ep] = c + 1

    data = data.filter(pa.array(keep_mask, type=pa.bool_()))
    n = data.num_rows
    data = replace_col(data, "episode_index", new_episode_indices, pa.int64())
    data = replace_col(data, "task_index", [0] * n, pa.int64())
    data = replace_col(data, "index", list(range(n)), pa.int64())
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data, dst_path)
    return n


def rewrite_episodes_parquet(
    src_path: Path,
    dst_path: Path,
    tail: int,
    fps: int,
    old_to_new_episode: dict[int, int],
) -> tuple[list[int], int]:
    """Update length, tasks, dataset_from/to_index, videos/*/to_timestamp.

    Returns (new_lengths, new_total_frames).
    """
    eps = pq.read_table(src_path)
    old_episode_indices = eps.column("episode_index").to_pylist()
    keep_mask = [ep in old_to_new_episode for ep in old_episode_indices]
    eps = eps.filter(pa.array(keep_mask, type=pa.bool_()))
    n_eps = eps.num_rows
    old_lengths = eps.column("length").to_pylist()
    new_lengths = [L - tail for L in old_lengths]
    assert all(L >= 1 for L in new_lengths), "An episode would become empty after trimming"

    eps = replace_col(eps, "episode_index", list(range(n_eps)), pa.int64())
    eps = replace_col(eps, "length", new_lengths, pa.int64())

    # tasks: single-element string list per row
    new_tasks = [[NEW_TASK] for _ in range(n_eps)]
    eps = replace_col(eps, "tasks", new_tasks, pa.list_(pa.string()))

    cum = 0
    from_idx, to_idx = [], []
    for L in new_lengths:
        from_idx.append(cum)
        cum += L
        to_idx.append(cum)
    eps = replace_col(eps, "dataset_from_index", from_idx, pa.int64())
    eps = replace_col(eps, "dataset_to_index", to_idx, pa.int64())

    # Shorten per-episode to_timestamp by TAIL/fps. from_timestamp stays the same:
    # each episode still begins at its original offset within the shared chunk mp4.
    shift = tail / fps
    for cam in CAMERAS:
        old_to = eps.column(f"videos/{cam}/to_timestamp").to_pylist()
        new_to = [t - shift for t in old_to]
        eps = replace_col(eps, f"videos/{cam}/to_timestamp", new_to, pa.float64())

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(eps, dst_path)
    return new_lengths, cum


def rewrite_tasks_parquet(dst_path: Path) -> None:
    """Single-row tasks table: task_index=0, task=NEW_TASK as the index (matches existing schema)."""
    tasks = pd.DataFrame({"task_index": [0]}, index=[NEW_TASK])
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_parquet(dst_path)


def rewrite_info_json(src_path: Path, dst_path: Path, total_frames: int, total_episodes: int) -> None:
    info = json.loads(src_path.read_text())
    info["total_frames"] = total_frames
    info["total_episodes"] = total_episodes
    info["total_tasks"] = 1
    info["splits"] = {"train": f"0:{total_episodes}"}
    with open(dst_path, "w") as f:
        json.dump(info, f, indent=4)


def _quantiles(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    return {
        "min":   arr.min(axis=0).tolist()  if arr.ndim > 1 else [float(arr.min())],
        "max":   arr.max(axis=0).tolist()  if arr.ndim > 1 else [float(arr.max())],
        "mean":  arr.mean(axis=0).tolist() if arr.ndim > 1 else [float(arr.mean())],
        "std":   arr.std(axis=0).tolist()  if arr.ndim > 1 else [float(arr.std())],
        "count": [int(len(arr))],
        "q01":   np.quantile(arr, 0.01, axis=0).tolist() if arr.ndim > 1 else [float(np.quantile(arr, 0.01))],
        "q10":   np.quantile(arr, 0.10, axis=0).tolist() if arr.ndim > 1 else [float(np.quantile(arr, 0.10))],
        "q50":   np.quantile(arr, 0.50, axis=0).tolist() if arr.ndim > 1 else [float(np.quantile(arr, 0.50))],
        "q90":   np.quantile(arr, 0.90, axis=0).tolist() if arr.ndim > 1 else [float(np.quantile(arr, 0.90))],
        "q99":   np.quantile(arr, 0.99, axis=0).tolist() if arr.ndim > 1 else [float(np.quantile(arr, 0.99))],
    }


def rewrite_stats_json(src_stats: Path, dst_stats: Path, data_parquet: Path) -> None:
    """Recompute scalar feature stats from the trimmed data parquet; keep image stats verbatim."""
    old = json.loads(src_stats.read_text())
    df = pd.read_parquet(data_parquet)

    new_stats = {}
    new_stats["action"] = _quantiles(np.stack(df["action"].values))
    new_stats["observation.state"] = _quantiles(np.stack(df["observation.state"].values))
    for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        new_stats[col] = _quantiles(df[col].to_numpy())

    # Image stats: copy from source (videos unchanged).
    for k in old:
        if k.startswith("observation.images."):
            new_stats[k] = old[k]

    with open(dst_stats, "w") as f:
        json.dump(new_stats, f, indent=4)


def verify(dst: Path) -> None:
    data = pd.read_parquet(dst / "data" / "chunk-000" / "file-000.parquet")
    eps = pd.read_parquet(next((dst / "meta" / "episodes").rglob("*.parquet")))
    tasks = pd.read_parquet(dst / "meta" / "tasks.parquet")
    info = json.loads((dst / "meta" / "info.json").read_text())
    fps = info["fps"]

    assert len(data) == int(eps["length"].sum()), \
        f"data rows {len(data)} != sum of episode lengths {eps['length'].sum()}"
    assert (data["index"].to_numpy() == np.arange(len(data))).all(), \
        "data 'index' column is not a contiguous range"
    assert (data["episode_index"].to_numpy() >= 0).all(), \
        "data contains negative episode_index values"
    assert sorted(data["episode_index"].unique()) == list(range(len(eps))), \
        "data episode_index values are not contiguous or do not match episodes metadata"
    assert (eps["episode_index"].to_numpy() == np.arange(len(eps))).all(), \
        "episodes metadata episode_index is not a contiguous range"
    assert list(data["task_index"].unique()) == [0], \
        f"unexpected task_index values: {data['task_index'].unique()}"
    assert len(tasks) == 1, f"tasks.parquet has {len(tasks)} rows, expected 1"
    assert info["total_frames"] == len(data)
    assert info["total_episodes"] == len(eps)
    assert info["total_tasks"] == 1
    assert info["splits"] == {"train": f"0:{len(eps)}"}
    assert len(eps) == EXPECTED_KEPT_EPISODES, \
        f"episodes {len(eps)} != expected {EXPECTED_KEPT_EPISODES}"
    assert len(data) == EXPECTED_TOTAL_FRAMES, \
        f"frames {len(data)} != expected {EXPECTED_TOTAL_FRAMES}"

    # Per-episode length consistency
    grouped = data.groupby("episode_index").agg(rows=("frame_index", "size"),
                                                 max_fi=("frame_index", "max"))
    for ep_idx, expected_len in zip(eps["episode_index"], eps["length"]):
        row = grouped.loc[ep_idx]
        assert row["rows"] == expected_len, \
            f"ep {ep_idx}: data has {row['rows']} rows, episodes meta says {expected_len}"
        assert row["max_fi"] == expected_len - 1, \
            f"ep {ep_idx}: max frame_index {row['max_fi']} != length-1 {expected_len - 1}"

    # tasks list uniformity
    task_strs = set(tuple(t) for t in eps["tasks"])
    assert task_strs == {(NEW_TASK,)}, f"non-uniform tasks in episodes meta: {task_strs}"
    assert not any(task in DROP_TASKS for task_tuple_ in task_strs for task in task_tuple_), \
        f"drop task still present in episodes meta: {task_strs}"

    # Video timestamp consistency
    for cam in CAMERAS:
        from_ts = eps[f"videos/{cam}/from_timestamp"].to_numpy()
        to_ts = eps[f"videos/{cam}/to_timestamp"].to_numpy()
        expected_dur = eps["length"].to_numpy() / fps
        diffs = np.abs((to_ts - from_ts) - expected_dur)
        assert diffs.max() < 1e-3, \
            f"{cam}: to-from timestamp duration mismatches length/fps by {diffs.max()}"

    print("OK — all self-checks passed.")
    print(f"  episodes:      {len(eps)}")
    print(f"  total frames:  {len(data)}")
    print(f"  unique tasks:  {len(tasks)}  → {tasks.index.tolist()}")


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    if DST.exists():
        raise FileExistsError(f"{DST} already exists; delete it first if you want to rebuild")

    info = json.loads((SRC / "meta" / "info.json").read_text())
    fps = info["fps"]
    old_to_new_episode, source_episodes, dropped_raw_frames, kept_raw_frames = load_episode_filter(
        SRC / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    kept_episodes = len(old_to_new_episode)
    expected_total = kept_raw_frames - kept_episodes * TAIL

    print(f"Source:      {SRC}")
    print(f"Destination: {DST}")
    print(f"fps={fps}, trimming last {TAIL} frames per episode, "
          f"collapsing all task labels to: {NEW_TASK!r}")
    print(f"Dropping tasks: {sorted(DROP_TASKS)}")
    print(f"Source episodes: {source_episodes}")
    print(f"Kept episodes:   {kept_episodes}")
    print(f"Dropped episodes:{source_episodes - kept_episodes}")
    print(f"Dropped frames:  {dropped_raw_frames}")
    print(f"Expected rows:   {expected_total}\n")

    assert kept_episodes == EXPECTED_KEPT_EPISODES, \
        f"kept episodes {kept_episodes} != expected {EXPECTED_KEPT_EPISODES}"
    assert expected_total == EXPECTED_TOTAL_FRAMES, \
        f"expected rows {expected_total} != expected {EXPECTED_TOTAL_FRAMES}"

    DST.mkdir(parents=True)
    (DST / "data" / "chunk-000").mkdir(parents=True)
    (DST / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (DST / "videos").mkdir(parents=True)

    print("Symlinking videos ...")
    link_videos(SRC, DST)

    print("Rewriting data parquet ...")
    new_total = rewrite_data_parquet(
        SRC / "data" / "chunk-000" / "file-000.parquet",
        DST / "data" / "chunk-000" / "file-000.parquet",
        TAIL,
        old_to_new_episode,
    )
    print(f"  → {new_total} rows")

    print("Rewriting episodes parquet ...")
    new_lengths, sum_lengths = rewrite_episodes_parquet(
        SRC / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        DST / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        TAIL, fps, old_to_new_episode,
    )
    assert sum_lengths == new_total, \
        f"episodes length sum ({sum_lengths}) != data rows ({new_total})"

    print("Rewriting tasks.parquet ...")
    rewrite_tasks_parquet(DST / "meta" / "tasks.parquet")

    print("Rewriting info.json ...")
    rewrite_info_json(SRC / "meta" / "info.json", DST / "meta" / "info.json", new_total, kept_episodes)

    print("Rewriting stats.json ...")
    rewrite_stats_json(
        SRC / "meta" / "stats.json",
        DST / "meta" / "stats.json",
        DST / "data" / "chunk-000" / "file-000.parquet",
    )

    print("\nVerifying ...")
    verify(DST)


if __name__ == "__main__":
    main()
