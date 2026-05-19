#!/usr/bin/env python3
"""
consolidate_dataset_videos.py — Merge per-episode video files into multi-episode chunk files.

LeRobot datasets recorded with SubtaskRecorder write one video file per episode for crash-safety.
This is fine for recording but slow for training: torchcodec reinitialises the decoder for every
file, so 315 files = 315x overhead vs 3-4 large files.

This script concatenates the individual per-episode MP4 files into larger files (no re-encoding)
and updates meta/episodes with the new from_timestamp / to_timestamp offsets for each episode.

Usage:
    python consolidate_dataset_videos.py <dataset_folder> [--episodes-per-file N]

    N = 0 (default): put all episodes in one file per camera — fastest for training.
    N > 0: split into files of at most N episodes each.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


CAMERAS = ["observation.images.front", "observation.images.wrist"]


def load_all_episode_meta(dataset_path: Path) -> tuple[pa.Table, list[Path]]:
    """Return (concatenated table, list of source parquet file paths) sorted by episode_index."""
    parquet_files = sorted((dataset_path / "meta" / "episodes").glob("chunk-*/*.parquet"))
    tables = []
    for f in parquet_files:
        try:
            tables.append(pq.read_table(f))
        except Exception as e:
            print(f"  Warning: skipping corrupted {f.name}: {e}")
    all_eps = pa.concat_tables(tables)
    # Sort by episode_index to guarantee order
    order = sorted(range(len(all_eps)), key=lambda i: all_eps.column("episode_index")[i].as_py())
    return all_eps.take(order), parquet_files


def concat_videos(src_paths: list[Path], dst_path: Path) -> None:
    """Losslessly concatenate src_paths into dst_path using ffmpeg concat demuxer."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_file = Path(f.name)
        for p in src_paths:
            # ffmpeg concat requires absolute paths or paths relative to the list file
            f.write(f"file '{p.resolve()}'\n")
    try:
        res = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", str(concat_file),
             "-c", "copy", str(dst_path)],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed:\n{res.stderr}")
    finally:
        concat_file.unlink(missing_ok=True)


def consolidate_videos(dataset_path: Path, episodes_per_file: int = 0) -> None:
    """
    In-place: merge per-episode video files into multi-episode chunk files.
    episodes_per_file=0 puts all episodes in a single file per camera.
    Updates meta/episodes timestamps. Deletes the old individual files on success.
    """
    with open(dataset_path / "meta" / "info.json") as f:
        info = json.load(f)
    fps: int = info["fps"]

    all_eps, meta_parquet_files = load_all_episode_meta(dataset_path)
    N = len(all_eps)

    if N == 0:
        print("  No episodes found — nothing to consolidate.")
        return

    batch_size = N if episodes_per_file <= 0 else episodes_per_file
    n_batches = (N + batch_size - 1) // batch_size
    print(f"  {N} episodes → {n_batches} file(s) per camera  (batch size: {batch_size})")

    # Gather per-episode info we need
    ep_indices = all_eps.column("episode_index").to_pylist()
    ep_lengths = all_eps.column("length").to_pylist()

    # Build new timestamp arrays for each camera
    new_file_index:  list[int]   = [0] * N
    new_chunk_index: list[int]   = [0] * N
    new_from_ts:     dict[str, list[float]] = {cam: [0.0] * N for cam in CAMERAS}
    new_to_ts:       dict[str, list[float]] = {cam: [0.0] * N for cam in CAMERAS}

    for cam in CAMERAS:
        src_chunk_col = all_eps.column(f"videos/{cam}/chunk_index").to_pylist()
        src_file_col  = all_eps.column(f"videos/{cam}/file_index").to_pylist()

        for batch_idx in range(n_batches):
            lo = batch_idx * batch_size
            hi = min(lo + batch_size, N)

            # Collect source video paths for this batch
            src_paths = []
            for i in range(lo, hi):
                c = src_chunk_col[i]
                fi = src_file_col[i]
                src_paths.append(dataset_path / f"videos/{cam}/chunk-{c:03d}/file-{fi:03d}.mp4")

            # Output path for the consolidated file
            dst = dataset_path / f"videos/{cam}/chunk-000/file-{batch_idx:03d}.mp4"

            if len(src_paths) == 1 and src_paths[0] == dst:
                # Nothing to do for this batch
                offset = 0.0
                new_file_index[lo] = batch_idx
                new_chunk_index[lo] = 0
                new_from_ts[cam][lo] = 0.0
                new_to_ts[cam][lo] = ep_lengths[lo] / fps
                continue

            print(f"    [{cam}] batch {batch_idx}: concat {len(src_paths)} clips → {dst.name}")

            # Write to a temp path first, then rename — keeps old files intact until success
            tmp_dst = dst.with_suffix(".tmp.mp4")
            concat_videos(src_paths, tmp_dst)
            tmp_dst.rename(dst)

            # Delete source files that are not the destination
            for p in src_paths:
                if p.resolve() != dst.resolve():
                    p.unlink(missing_ok=True)

            # Compute timestamps
            running_offset = 0.0
            for i in range(lo, hi):
                new_chunk_index[i]   = 0
                new_file_index[i]    = batch_idx
                new_from_ts[cam][i]  = running_offset
                running_offset      += ep_lengths[i] / fps
                new_to_ts[cam][i]    = running_offset

        # Clean up any now-empty chunk subdirectories (other than chunk-000)
        for chunk_dir in sorted((dataset_path / f"videos/{cam}").iterdir()):
            if chunk_dir.name != "chunk-000" and chunk_dir.is_dir():
                try:
                    chunk_dir.rmdir()  # only removes if empty
                except OSError:
                    pass

    # Update meta/episodes: replace the affected columns
    def replace_col(t: pa.Table, name: str, values: list, typ: pa.DataType) -> pa.Table:
        return t.set_column(t.schema.get_field_index(name), name, pa.array(values, type=typ))

    updated = all_eps
    for cam in CAMERAS:
        updated = replace_col(updated, f"videos/{cam}/chunk_index", new_chunk_index, pa.int64())
        updated = replace_col(updated, f"videos/{cam}/file_index",  new_file_index,  pa.int64())
        updated = replace_col(updated, f"videos/{cam}/from_timestamp", new_from_ts[cam], pa.float64())
        updated = replace_col(updated, f"videos/{cam}/to_timestamp",   new_to_ts[cam],   pa.float64())

    # Write back — keep the same chunk/file structure for meta parquets
    # (just overwrite the single existing meta parquet with updated content)
    if len(meta_parquet_files) == 1:
        pq.write_table(updated, meta_parquet_files[0])
    else:
        # Multiple meta parquet files: overwrite the first with all data, delete the rest
        pq.write_table(updated, meta_parquet_files[0])
        for f in meta_parquet_files[1:]:
            f.unlink(missing_ok=True)

    print(f"  Done. meta/episodes updated.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate per-episode video files for fast training.")
    parser.add_argument("dataset", help="Path to LeRobot dataset folder")
    parser.add_argument(
        "--episodes-per-file", type=int, default=0,
        help="Max episodes per output file (0 = all in one file, default: 0)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Consolidating videos in: {dataset_path.name}")
    consolidate_videos(dataset_path, args.episodes_per_file)
    print("All done.")


if __name__ == "__main__":
    main()
