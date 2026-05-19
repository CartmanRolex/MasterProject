#!/usr/bin/env python3
"""
balance_dataset.py — Create a balanced LeRobot dataset subset.

Balance criterion: each of the 9 subtask slots in a fully successful 3-orange
episode gets equal representation:

  (Grasp orange, n_placed=0)     (Pick it up,        n_placed=0)  (Place into plate, n_placed=1)
  (Grasp orange, n_placed=1)     (Pick it up,        n_placed=1)  (Place into plate, n_placed=2)
  (Grasp orange, n_placed=2)     (Pick it up,        n_placed=2)  (Place into plate, n_placed=3)

N = min count across all 9 buckets.  Total output = 9 * N episodes.
"Go back to start position" is excluded (scripted primitive, not VLA).
"""

import argparse
import datetime
import json
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SLOTS = [
    ("Grasp orange", 0),
    ("Pick it up", 0),
    ("Place it into plate", 1),
    ("Grasp orange", 1),
    ("Pick it up", 1),
    ("Place it into plate", 2),
    ("Grasp orange", 2),
    ("Pick it up", 2),
    ("Place it into plate", 3),
]

CAMERAS = ["observation.images.front", "observation.images.wrist"]

SLOT_SET = set(SLOTS)


def group_task(task: str) -> str:
    return "Grasp orange" if task.startswith("Grasp") else task


def select_episodes(records: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)

    buckets: dict[tuple, list] = {slot: [] for slot in SLOTS}
    for rec in records:
        slot = (group_task(rec["task"]), rec["n_placed"])
        if slot in buckets:
            buckets[slot].append(rec)

    print("Bucket sizes:")
    for slot in SLOTS:
        print(f"  ({slot[0]!r:30s}, n_placed={slot[1]}): {len(buckets[slot])}")

    n = min(len(buckets[slot]) for slot in SLOTS)
    if n == 0:
        raise RuntimeError("One or more buckets are empty — cannot balance.")

    print(f"\nN per bucket: {n}  →  total: {len(SLOTS) * n} episodes")

    selected = []
    for slot in SLOTS:
        selected.extend(rng.sample(buckets[slot], n))

    selected.sort(key=lambda r: r["episode_index"])
    return selected


def load_episode_lookup(source: Path) -> tuple[dict[int, int], pa.Table]:
    """
    Return (row_of_ep, all_eps) where:
      - row_of_ep maps episode_index → row index in the concatenated meta/episodes table
      - all_eps is the concatenated pyarrow Table (all meta/episodes)
    Skips corrupted parquet files with a warning.
    """
    tables = []
    for f in sorted((source / "meta" / "episodes").glob("chunk-*/*.parquet")):
        try:
            tables.append(pq.read_table(f))
        except Exception as e:
            print(f"  Warning: skipping corrupted metadata file {f.name}: {e}")
    if not tables:
        raise RuntimeError("No valid meta/episodes parquet files found.")
    all_eps = pa.concat_tables(tables)
    ep_ids = all_eps.column("episode_index").to_pylist()
    row_of_ep = {ep: i for i, ep in enumerate(ep_ids)}
    return row_of_ep, all_eps


def copy_data_parquet(src: Path, dst: Path, new_ep_idx: int, frame_offset: int) -> int:
    table = pq.read_table(src)
    n = len(table)
    table = table.set_column(
        table.schema.get_field_index("episode_index"),
        "episode_index",
        pa.array([new_ep_idx] * n, type=pa.int64()),
    )
    table = table.set_column(
        table.schema.get_field_index("index"),
        "index",
        pa.array(range(frame_offset, frame_offset + n), type=pa.int64()),
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)
    return n


def probe_vcodec() -> str:
    """Return the best available hardware/software video encoder."""
    for codec in ("h264_nvenc", "libx264"):
        res = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=0.1:size=640x480:rate=30",
             "-c:v", codec, "-f", "null", "-"],
            capture_output=True,
        )
        if res.returncode == 0:
            return codec
    raise RuntimeError("No supported video encoder found (h264_nvenc or libx264)")


_VCODEC: str | None = None


def extract_video(src: Path, dst: Path, from_ts: float, to_ts: float) -> None:
    global _VCODEC
    if _VCODEC is None:
        _VCODEC = probe_vcodec()
        print(f"  Video encoder: {_VCODEC}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(from_ts), "-to", str(to_ts),
            "-i", str(src),
            "-c:v", _VCODEC, "-pix_fmt", "yuv420p",
            str(dst),
        ],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src}:\n{res.stderr}")


def build_meta_episodes_table(
    all_eps: pa.Table,
    row_of_ep: dict[int, int],
    selected: list[dict],
    per_ep_frames: list[int],
    frame_offsets: list[int],
    fps: int,
    chunks_size: int,
) -> pa.Table:
    N = len(selected)
    ordered_row_idxs = [row_of_ep[r["episode_index"]] for r in selected]
    table = all_eps.take(ordered_row_idxs)

    def replace(t: pa.Table, col: str, values: list, typ: pa.DataType) -> pa.Table:
        return t.set_column(t.schema.get_field_index(col), col, pa.array(values, type=typ))

    table = replace(table, "episode_index", list(range(N)), pa.int64())
    table = replace(table, "data/chunk_index", [i // chunks_size for i in range(N)], pa.int64())
    table = replace(table, "data/file_index", [i % chunks_size for i in range(N)], pa.int64())
    table = replace(table, "dataset_from_index", frame_offsets, pa.int64())
    table = replace(table, "dataset_to_index", [frame_offsets[i] + per_ep_frames[i] for i in range(N)], pa.int64())

    for cam in CAMERAS:
        table = replace(table, f"videos/{cam}/chunk_index", [i // chunks_size for i in range(N)], pa.int64())
        table = replace(table, f"videos/{cam}/file_index", [i % chunks_size for i in range(N)], pa.int64())
        table = replace(table, f"videos/{cam}/from_timestamp", [0.0] * N, pa.float64())
        table = replace(table, f"videos/{cam}/to_timestamp", [per_ep_frames[i] / fps for i in range(N)], pa.float64())

    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a 9-slot balanced LeRobot dataset from a subtask recording."
    )
    parser.add_argument("source", help="Source folder name inside synthetic_datasets/")
    parser.add_argument("output", help="Output folder name inside synthetic_datasets/")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    this_dir = Path(__file__).parent / "synthetic_datasets"
    source = this_dir / args.source
    dest = this_dir / args.output

    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if dest.exists():
        raise FileExistsError(f"Output already exists: {dest}  — delete it first")

    records = [
        json.loads(l)
        for l in (source / "subtask_metadata.jsonl").read_text().splitlines()
        if l.strip()
    ]
    with open(source / "meta" / "info.json") as f:
        info = json.load(f)
    fps: int = info["fps"]
    chunks_size: int = info["chunks_size"]

    row_of_ep, all_eps = load_episode_lookup(source)

    # Filter out episodes whose metadata was lost (e.g. corrupted parquet skipped above)
    n_before = len(records)
    records = [r for r in records if r["episode_index"] in row_of_ep]
    if len(records) < n_before:
        print(f"  Note: {n_before - len(records)} episode(s) missing from meta/episodes (skipped)")

    selected = select_episodes(records, args.seed)

    # Pre-load video file-mapping columns for fast lookup
    def col(name: str) -> list:
        return all_eps.column(name).to_pylist()

    data_chunk_col = col("data/chunk_index")
    data_file_col = col("data/file_index")
    vid_info: dict[str, dict[str, list]] = {}
    for cam in CAMERAS:
        vid_info[cam] = {
            "chunk": col(f"videos/{cam}/chunk_index"),
            "file": col(f"videos/{cam}/file_index"),
            "from_ts": col(f"videos/{cam}/from_timestamp"),
            "to_ts": col(f"videos/{cam}/to_timestamp"),
        }

    per_ep_frames: list[int] = []
    frame_offsets: list[int] = []
    frame_offset = 0

    print(f"\nProcessing {len(selected)} episodes (videos will be re-encoded)...")
    for new_ep_idx, rec in enumerate(selected):
        old_ep_idx = rec["episode_index"]
        ri = row_of_ep[old_ep_idx]

        src_parquet = source / f"data/chunk-{data_chunk_col[ri]:03d}/file-{data_file_col[ri]:03d}.parquet"
        dst_parquet = dest / f"data/chunk-{new_ep_idx // chunks_size:03d}/file-{new_ep_idx % chunks_size:03d}.parquet"
        n_frames = copy_data_parquet(src_parquet, dst_parquet, new_ep_idx, frame_offset)

        for cam in CAMERAS:
            vi = vid_info[cam]
            src_video = source / f"videos/{cam}/chunk-{vi['chunk'][ri]:03d}/file-{vi['file'][ri]:03d}.mp4"
            dst_video = dest / f"videos/{cam}/chunk-{new_ep_idx // chunks_size:03d}/file-{new_ep_idx % chunks_size:03d}.mp4"
            extract_video(src_video, dst_video, vi["from_ts"][ri], vi["to_ts"][ri])

        frame_offsets.append(frame_offset)
        per_ep_frames.append(n_frames)
        frame_offset += n_frames

        if (new_ep_idx + 1) % 25 == 0 or new_ep_idx == len(selected) - 1:
            print(f"  {new_ep_idx + 1}/{len(selected)}")

    total_frames = frame_offset

    # Write meta/episodes
    new_ep_table = build_meta_episodes_table(
        all_eps, row_of_ep, selected, per_ep_frames, frame_offsets, fps, chunks_size
    )
    n_out_chunks = (len(selected) + chunks_size - 1) // chunks_size
    for c in range(n_out_chunks):
        start = c * chunks_size
        chunk_table = new_ep_table.slice(start, min(chunks_size, len(selected) - start))
        chunk_dir = dest / "meta" / "episodes" / f"chunk-{c:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(chunk_table, chunk_dir / "file-000.parquet")

    # Copy unchanged meta files
    (dest / "meta").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "meta" / "tasks.parquet", dest / "meta" / "tasks.parquet")
    stats_src = source / "meta" / "stats.json"
    if stats_src.exists():
        shutil.copy2(stats_src, dest / "meta" / "stats.json")

    # Write info.json — update codec if we re-encoded to H.264
    new_info = dict(info)
    new_info["total_episodes"] = len(selected)
    new_info["total_frames"] = total_frames
    new_info["splits"] = {"train": f"0:{len(selected)}"}
    if _VCODEC and _VCODEC != "libsvtav1":
        codec_name = "h264" if "264" in _VCODEC else _VCODEC
        for cam in CAMERAS:
            if "features" in new_info and cam in new_info["features"]:
                new_info["features"][cam]["info"]["video.codec"] = codec_name
    with open(dest / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=4)

    # Write checkpoint.json
    with open(dest / "checkpoint.json", "w") as f:
        json.dump(
            {
                "total_subtask_recordings": len(selected),
                "last_commit": datetime.datetime.now().isoformat(timespec="seconds"),
            },
            f,
        )

    # Write subtask_metadata.jsonl
    with open(dest / "subtask_metadata.jsonl", "w") as f:
        for new_ep_idx, rec in enumerate(selected):
            f.write(
                json.dumps(
                    {
                        "episode_index": new_ep_idx,
                        "task": rec["task"],
                        "n_placed": rec["n_placed"],
                        "timestamp": rec["timestamp"],
                    }
                )
                + "\n"
            )

    N = len(selected)
    print(f"\nDone.")
    print(f"  Source:   {source.name}  ({len(records)} episodes)")
    print(f"  Output:   {dest.name}  ({N} episodes, {total_frames} frames)")
    print(f"\nVerify with:")
    print(f"  python plot_dataset_stats.py {args.output}")


if __name__ == "__main__":
    main()
