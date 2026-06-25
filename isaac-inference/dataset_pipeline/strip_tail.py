#!/usr/bin/env python3
"""
strip_tail.py — produce a TAIL-FREE clone of a subtask dataset: the trailing
N "frozen frames" of every episode are removed, but the language task labels are
kept (unlike strip_lang_and_tail.py, which also collapses language and drops the
"go back to start" episodes).

Source: HF-cache copy of MasterProject2026/Gal_split_tailed (846 episodes, 104886
        frames, 11 language tasks; the dataset behind Gal-pick-orange-tailedCH20).
Destination: synthetic_datasets/Gal_split_notail.

This isolates the frozen tail as the only changed variable, so a model trained on
the result is directly comparable to Gal-pick-orange-tailedCH20.

What changes:
  - Last TAIL=20 frames of every episode are dropped from the data parquet.
  - episodes meta: length, dataset_from/to_index, videos/*/to_timestamp updated.
  - info.json: total_frames updated (total_episodes / total_tasks unchanged).
  - stats.json: scalar feature stats recomputed; image stats preserved.
What does NOT change:
  - Task labels (tasks.parquet copied verbatim; per-frame task_index preserved).
  - Videos: chunk mp4s symlinked from source; the trailing 20 frames of each
    episode become an unread gap between consecutive episodes' video slices.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from balance_dataset import CAMERAS

HF_CACHE = Path.home() / ".cache" / "huggingface" / "lerobot" / "MasterProject2026"
SRC = HF_CACHE / "Gal_split_tailed"
DST = Path(__file__).parent.parent / "synthetic_datasets" / "Gal_split_notail"
TAIL = 20
EXPECTED_KEPT_EPISODES = 846
EXPECTED_TOTAL_FRAMES = 104886 - 846 * TAIL  # 87966


def replace_col(t: pa.Table, name: str, values, typ) -> pa.Table:
    return t.set_column(t.schema.get_field_index(name), name, pa.array(values, type=typ))


def link_videos(src: Path, dst: Path) -> None:
    """Symlink each mp4 from src/videos/ into dst/videos/, preserving the tree."""
    for cam_dir in sorted((src / "videos").iterdir()):
        if not cam_dir.is_dir():
            continue
        for chunk_dir in sorted(cam_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue
            out_chunk = dst / "videos" / cam_dir.name / chunk_dir.name
            out_chunk.mkdir(parents=True, exist_ok=True)
            for mp4 in sorted(chunk_dir.glob("*.mp4")):
                link = out_chunk / mp4.name
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(mp4.resolve())


def rewrite_data_parquet(src_path: Path, dst_path: Path, tail: int) -> int:
    """Drop the per-episode tail; reset global index. Keep episode_index + task_index."""
    data = pq.read_table(src_path)
    ep_col = data.column("episode_index").to_pylist()

    totals: dict[int, int] = {}
    for ep in ep_col:
        totals[ep] = totals.get(ep, 0) + 1

    seen: dict[int, int] = {}
    keep_mask: list[bool] = []
    for ep in ep_col:
        c = seen.get(ep, 0)
        keep_mask.append(c < totals[ep] - tail)
        seen[ep] = c + 1

    data = data.filter(pa.array(keep_mask, type=pa.bool_()))
    n = data.num_rows
    # episode_index already 0..N-1 contiguous (no episodes dropped); task_index kept as-is.
    data = replace_col(data, "index", list(range(n)), pa.int64())
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data, dst_path)
    return n


def rewrite_episodes_parquet(src_path: Path, dst_path: Path, tail: int, fps: int) -> tuple[list[int], int]:
    """Update length, dataset_from/to_index, videos/*/to_timestamp. Keep tasks."""
    eps = pq.read_table(src_path)
    old_lengths = eps.column("length").to_pylist()
    new_lengths = [L - tail for L in old_lengths]
    assert all(L >= 1 for L in new_lengths), "An episode would become empty after trimming"

    eps = replace_col(eps, "length", new_lengths, pa.int64())

    cum = 0
    from_idx, to_idx = [], []
    for L in new_lengths:
        from_idx.append(cum)
        cum += L
        to_idx.append(cum)
    eps = replace_col(eps, "dataset_from_index", from_idx, pa.int64())
    eps = replace_col(eps, "dataset_to_index", to_idx, pa.int64())

    shift = tail / fps  # from_timestamp unchanged; each episode still starts at its offset
    for cam in CAMERAS:
        old_to = eps.column(f"videos/{cam}/to_timestamp").to_pylist()
        eps = replace_col(eps, f"videos/{cam}/to_timestamp", [t - shift for t in old_to], pa.float64())

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(eps, dst_path)
    return new_lengths, cum


def _quantiles(arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    multi = arr.ndim > 1
    q = lambda p: (np.quantile(arr, p, axis=0).tolist() if multi else [float(np.quantile(arr, p))])
    return {
        "min":   arr.min(axis=0).tolist()  if multi else [float(arr.min())],
        "max":   arr.max(axis=0).tolist()  if multi else [float(arr.max())],
        "mean":  arr.mean(axis=0).tolist() if multi else [float(arr.mean())],
        "std":   arr.std(axis=0).tolist()  if multi else [float(arr.std())],
        "count": [int(len(arr))],
        "q01": q(0.01), "q10": q(0.10), "q50": q(0.50), "q90": q(0.90), "q99": q(0.99),
    }


def rewrite_stats_json(src_stats: Path, dst_stats: Path, data_parquet: Path) -> None:
    old = json.loads(src_stats.read_text())
    df = pd.read_parquet(data_parquet)
    new_stats = {
        "action": _quantiles(np.stack(df["action"].values)),
        "observation.state": _quantiles(np.stack(df["observation.state"].values)),
    }
    for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        new_stats[col] = _quantiles(df[col].to_numpy())
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

    assert len(data) == int(eps["length"].sum()), f"data {len(data)} != sum lengths {eps['length'].sum()}"
    assert (data["index"].to_numpy() == np.arange(len(data))).all(), "'index' not contiguous"
    assert sorted(data["episode_index"].unique()) == list(range(len(eps))), "episode_index not contiguous"
    assert (eps["episode_index"].to_numpy() == np.arange(len(eps))).all(), "episodes meta index not contiguous"
    assert info["total_frames"] == len(data) == EXPECTED_TOTAL_FRAMES, "total_frames mismatch"
    assert info["total_episodes"] == len(eps) == EXPECTED_KEPT_EPISODES, "total_episodes mismatch"
    # language preserved
    assert set(data["task_index"].unique()) <= set(range(len(tasks))), "task_index out of range"
    assert len(tasks) == info["total_tasks"] > 1, "tasks not preserved (language kept?)"

    grouped = data.groupby("episode_index").agg(rows=("frame_index", "size"), max_fi=("frame_index", "max"))
    for ep_idx, expected_len in zip(eps["episode_index"], eps["length"]):
        row = grouped.loc[ep_idx]
        assert row["rows"] == expected_len, f"ep {ep_idx}: {row['rows']} rows != {expected_len}"
        assert row["max_fi"] == expected_len - 1, f"ep {ep_idx}: max frame_index {row['max_fi']} != {expected_len-1}"

    for cam in CAMERAS:
        from_ts = eps[f"videos/{cam}/from_timestamp"].to_numpy()
        to_ts = eps[f"videos/{cam}/to_timestamp"].to_numpy()
        diffs = np.abs((to_ts - from_ts) - eps["length"].to_numpy() / fps)
        assert diffs.max() < 1e-3, f"{cam}: timestamp/length mismatch {diffs.max()}"

    print("OK — all self-checks passed.")
    print(f"  episodes:     {len(eps)}")
    print(f"  total frames: {len(data)}")
    print(f"  tasks kept:   {len(tasks)}")


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    if DST.exists():
        raise FileExistsError(f"{DST} already exists; delete it first to rebuild")

    info = json.loads((SRC / "meta" / "info.json").read_text())
    fps = info["fps"]
    print(f"Source:      {SRC}")
    print(f"Destination: {DST}")
    print(f"fps={fps}, trimming last {TAIL} frames per episode (language kept)\n")

    DST.mkdir(parents=True)
    (DST / "data" / "chunk-000").mkdir(parents=True)
    (DST / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (DST / "videos").mkdir(parents=True)

    print("Symlinking videos ...")
    link_videos(SRC, DST)

    print("Rewriting data parquet ...")
    new_total = rewrite_data_parquet(
        SRC / "data" / "chunk-000" / "file-000.parquet",
        DST / "data" / "chunk-000" / "file-000.parquet", TAIL)
    print(f"  → {new_total} rows")

    print("Rewriting episodes parquet ...")
    _, sum_lengths = rewrite_episodes_parquet(
        SRC / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        DST / "meta" / "episodes" / "chunk-000" / "file-000.parquet", TAIL, fps)
    assert sum_lengths == new_total, f"episodes length sum {sum_lengths} != data rows {new_total}"

    print("Copying tasks.parquet (language preserved) ...")
    shutil.copy(SRC / "meta" / "tasks.parquet", DST / "meta" / "tasks.parquet")

    print("Rewriting info.json ...")
    info["total_frames"] = new_total
    info["splits"] = {"train": f"0:{info['total_episodes']}"}
    with open(DST / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    print("Rewriting stats.json ...")
    rewrite_stats_json(SRC / "meta" / "stats.json", DST / "meta" / "stats.json",
                       DST / "data" / "chunk-000" / "file-000.parquet")

    # carry over any extra editor metadata if present (harmless)
    extra = SRC / "meta" / "edits.json"
    if extra.exists():
        shutil.copy(extra, DST / "meta" / "edits.json")

    print("\nVerifying ...")
    verify(DST)


if __name__ == "__main__":
    main()
