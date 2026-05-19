#!/usr/bin/env python3
"""
consolidate_dataset_videos.py — Merge per-episode files into multi-episode chunk files.

LeRobot datasets recorded with SubtaskRecorder write one data parquet and one video file
per episode for crash-safety. Training is slow with many small files because torchcodec
reinitialises the decoder for every video file, and LeRobot derives video download paths
from data/file_index in meta/episodes.

This script merges all per-episode files into one large file per chunk (no re-encoding for
video, plain parquet concatenation for data) and updates meta/episodes accordingly.
Idempotent: safe to re-run; skips steps that are already done.

Usage:
    python consolidate_dataset_videos.py <dataset_folder> [--episodes-per-file N]
    N=0 (default): all episodes in one file — fastest for training.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path
import json

import pyarrow as pa
import pyarrow.parquet as pq


CAMERAS = ["observation.images.front", "observation.images.wrist"]


def _replace_col(t: pa.Table, name: str, values: list, typ: pa.DataType) -> pa.Table:
    return t.set_column(t.schema.get_field_index(name), name, pa.array(values, type=typ))


def _ffmpeg_concat(src_paths: list[Path], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_path = Path(f.name)
        for p in src_paths:
            f.write(f"file '{p.resolve()}'\n")
    try:
        res = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", str(list_path),
             "-c", "copy", str(dst)],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{res.stderr}")
    finally:
        list_path.unlink(missing_ok=True)


def consolidate(dataset_path: Path, episodes_per_file: int = 0) -> None:
    # ── Load meta/episodes ─────────────────────────────────────────────────
    meta_files = sorted((dataset_path / "meta" / "episodes").glob("chunk-*/*.parquet"))
    tables = []
    for f in meta_files:
        try:
            tables.append(pq.read_table(f))
        except Exception as e:
            print(f"  Warning: skipping corrupted {f.name}: {e}")
    if not tables:
        raise RuntimeError("No valid meta/episodes parquet files found.")

    all_eps = pa.concat_tables(tables)
    order   = sorted(range(len(all_eps)), key=lambda i: all_eps.column("episode_index")[i].as_py())
    all_eps = all_eps.take(order)
    N       = len(all_eps)

    with open(dataset_path / "meta" / "info.json") as f:
        fps = json.load(f)["fps"]

    ep_lengths = all_eps.column("length").to_pylist()
    batch_size = N if episodes_per_file <= 0 else episodes_per_file

    updated = all_eps   # will accumulate column replacements

    # ── Video consolidation ────────────────────────────────────────────────
    new_vid_chunk: list[int]              = [0] * N
    new_vid_file:  list[int]              = [0] * N
    new_from_ts:   dict[str, list[float]] = {cam: [0.0] * N for cam in CAMERAS}
    new_to_ts:     dict[str, list[float]] = {cam: [0.0] * N for cam in CAMERAS}

    for cam in CAMERAS:
        src_chunk_col = all_eps.column(f"videos/{cam}/chunk_index").to_pylist()
        src_file_col  = all_eps.column(f"videos/{cam}/file_index").to_pylist()

        # Build ordered list of UNIQUE source video files (in episode order)
        seen: set[tuple] = set()
        unique_srcs: list[Path] = []
        for i in range(N):
            key = (src_chunk_col[i], src_file_col[i])
            if key not in seen:
                seen.add(key)
                unique_srcs.append(dataset_path / f"videos/{cam}/chunk-{key[0]:03d}/file-{key[1]:03d}.mp4")

        n_batches = (len(unique_srcs) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            batch_srcs = unique_srcs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            dst = dataset_path / f"videos/{cam}/chunk-000/file-{batch_idx:03d}.mp4"

            if len(batch_srcs) == 1 and batch_srcs[0].resolve() == dst.resolve():
                print(f"  [video/{cam}] batch {batch_idx}: already consolidated — skipping concat")
            else:
                print(f"  [video/{cam}] batch {batch_idx}: concat {len(batch_srcs)} files → {dst.name}")
                tmp = dst.with_suffix(".tmp.mp4")
                _ffmpeg_concat(batch_srcs, tmp)
                tmp.rename(dst)
                for p in batch_srcs:
                    if p.resolve() != dst.resolve():
                        p.unlink(missing_ok=True)

        # Compute which batch each episode lands in, and its timestamp offset within that batch
        # A batch covers the episodes whose source file was in unique_srcs[lo:hi]
        src_key_to_batch = {}
        for batch_idx in range(n_batches):
            for src in unique_srcs[batch_idx * batch_size : (batch_idx + 1) * batch_size]:
                # recover the (chunk, file) key from the path
                parts = src.parts
                c  = int(parts[-2].replace("chunk-", ""))
                fi = int(src.stem.replace("file-", ""))
                src_key_to_batch[(c, fi)] = batch_idx

        batch_offsets: dict[int, float] = {b: 0.0 for b in range(n_batches)}
        for i in range(N):
            key       = (src_chunk_col[i], src_file_col[i])
            b         = src_key_to_batch[key]
            new_vid_chunk[i] = 0
            new_vid_file[i]  = b
            new_from_ts[cam][i] = batch_offsets[b]
            batch_offsets[b]   += ep_lengths[i] / fps
            new_to_ts[cam][i]   = batch_offsets[b]

        # Clean up empty non-chunk-000 video directories
        for d in sorted((dataset_path / f"videos/{cam}").iterdir()):
            if d.name != "chunk-000" and d.is_dir():
                try:
                    d.rmdir()
                except OSError:
                    pass

    for cam in CAMERAS:
        updated = _replace_col(updated, f"videos/{cam}/chunk_index",   new_vid_chunk,     pa.int64())
        updated = _replace_col(updated, f"videos/{cam}/file_index",    new_vid_file,      pa.int64())
        updated = _replace_col(updated, f"videos/{cam}/from_timestamp", new_from_ts[cam], pa.float64())
        updated = _replace_col(updated, f"videos/{cam}/to_timestamp",   new_to_ts[cam],   pa.float64())

    # ── Data parquet consolidation ─────────────────────────────────────────
    new_data_chunk: list[int] = [0] * N
    new_data_file:  list[int] = [0] * N

    for chunk_dir in sorted((dataset_path / "data").glob("chunk-*")):
        data_files = sorted(chunk_dir.glob("file-*.parquet"))
        if len(data_files) <= 1:
            print(f"  [data/{chunk_dir.name}] already 1 file — skipping")
            continue
        print(f"  [data/{chunk_dir.name}] merging {len(data_files)} parquets → file-000.parquet")
        merged = pa.concat_tables([pq.read_table(f) for f in data_files])
        tmp = chunk_dir / "file-000.tmp.parquet"
        pq.write_table(merged, tmp)
        tmp.rename(chunk_dir / "file-000.parquet")
        for f in data_files[1:]:
            f.unlink()

    updated = _replace_col(updated, "data/chunk_index", new_data_chunk, pa.int64())
    updated = _replace_col(updated, "data/file_index",  new_data_file,  pa.int64())

    # ── Write meta/episodes ────────────────────────────────────────────────
    pq.write_table(updated, meta_files[0])
    for f in meta_files[1:]:
        f.unlink(missing_ok=True)

    print("  Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate per-episode files into chunk files for fast training."
    )
    parser.add_argument("dataset", help="Path to LeRobot dataset folder")
    parser.add_argument("--episodes-per-file", type=int, default=0,
                        help="Max episodes per output file (0 = all in one, default: 0)")
    args = parser.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    print(f"Consolidating: {path.name}")
    consolidate(path, args.episodes_per_file)
    print("All done.")


if __name__ == "__main__":
    main()
