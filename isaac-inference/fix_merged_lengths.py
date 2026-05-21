#!/usr/bin/env python3
"""
fix_merged_lengths.py — repair per-episode length drift after merge_datasets.py.

extract_video uses ffmpeg fast-seek (-ss before -i), which can lose a frame at
av1 GOP boundaries. Result: a few per-episode mp4s end up shorter than their
declared `length` in meta/episodes, and the merged-video concat ends up shorter
than sum-of-lengths — which breaks consolidate()'s PTS walk.

This script:
  1. ffprobes each per-episode mp4 (per camera) for actual frame count.
  2. Picks min(front_count, wrist_count, declared_length) per episode.
  3. ffmpeg-trims both per-episode mp4s to that count (-c copy, no re-encode).
  4. Truncates the merged data parquet: keep first `actual_length` rows per episode.
  5. Rewrites meta/episodes lengths + cumulative offsets and meta/info.json totals.
  6. Re-runs consolidate(dest, episodes_per_file=0).
"""

import argparse
import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from balance_dataset import CAMERAS
from consolidate_dataset_videos import consolidate


def probe_nb_frames(mp4: Path) -> int:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {mp4}: {res.stderr}")
    return int(res.stdout.strip())


def trim_mp4(src: Path, n_frames: int) -> None:
    """Trim mp4 in-place to exactly n_frames via stream copy."""
    tmp = src.with_suffix(".trim.mp4")
    res = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(src),
         "-frames:v", str(n_frames),
         "-c", "copy",
         str(tmp)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg trim failed for {src}: {res.stderr}")
    tmp.rename(src)


def replace_col(t: pa.Table, name: str, values, typ) -> pa.Table:
    return t.set_column(t.schema.get_field_index(name), name, pa.array(values, type=typ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="Folder name under synthetic_datasets/ (or full path)")
    args = parser.parse_args()

    dest = Path(args.dataset)
    if not dest.is_absolute():
        dest = Path(__file__).parent / "synthetic_datasets" / args.dataset
    if not dest.exists():
        raise FileNotFoundError(dest)

    # Load meta/episodes (single consolidated file at this stage)
    meta_files = sorted((dest / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if len(meta_files) != 1:
        raise RuntimeError(f"Expected exactly one meta/episodes parquet, found {len(meta_files)}")
    eps = pq.read_table(meta_files[0])
    n_eps = eps.num_rows
    chunks_size = json.loads((dest / "meta" / "info.json").read_text())["chunks_size"]
    fps = json.loads((dest / "meta" / "info.json").read_text())["fps"]

    declared_lengths = eps.column("length").to_pylist()

    print(f"Probing {n_eps} per-episode clips across {len(CAMERAS)} cameras ...")
    actual_lengths: list[int] = []
    n_changed = 0
    for i in range(n_eps):
        chunk = i // chunks_size
        file = i % chunks_size
        mp4s = {cam: dest / f"videos/{cam}/chunk-{chunk:03d}/file-{file:03d}.mp4" for cam in CAMERAS}
        counts = {cam: probe_nb_frames(p) for cam, p in mp4s.items()}
        actual = min(min(counts.values()), declared_lengths[i])
        if actual != declared_lengths[i] or any(c != actual for c in counts.values()):
            print(f"  ep {i}: declared={declared_lengths[i]}, "
                  f"front={counts['observation.images.front']}, "
                  f"wrist={counts['observation.images.wrist']}, → {actual}")
            n_changed += 1
            for cam, p in mp4s.items():
                if counts[cam] != actual:
                    trim_mp4(p, actual)
        actual_lengths.append(actual)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n_eps}")

    new_total = sum(actual_lengths)
    old_total = sum(declared_lengths)
    print(f"\n{n_changed} episode(s) adjusted; total frames {old_total} → {new_total}")

    # Update meta/episodes lengths + cumulative offsets + provisional video to_timestamp
    eps = replace_col(eps, "length", actual_lengths, pa.int64())
    cum = 0
    from_idx, to_idx = [], []
    for L in actual_lengths:
        from_idx.append(cum)
        cum += L
        to_idx.append(cum)
    eps = replace_col(eps, "dataset_from_index", from_idx, pa.int64())
    eps = replace_col(eps, "dataset_to_index", to_idx, pa.int64())
    for cam in CAMERAS:
        eps = replace_col(eps, f"videos/{cam}/from_timestamp", [0.0] * n_eps, pa.float64())
        eps = replace_col(eps, f"videos/{cam}/to_timestamp", [L / fps for L in actual_lengths], pa.float64())
    pq.write_table(eps, meta_files[0])

    # Truncate data parquet per-episode
    data_files = sorted((dest / "data").glob("chunk-*/file-*.parquet"))
    if len(data_files) != 1:
        raise RuntimeError(f"Expected exactly one data parquet, found {len(data_files)}")
    data = pq.read_table(data_files[0])
    ep_col = data.column("episode_index").to_pylist()

    # Build keep mask via running counter per episode
    keep_mask: list[bool] = []
    seen: dict[int, int] = {}
    for ep in ep_col:
        c = seen.get(ep, 0)
        keep_mask.append(c < actual_lengths[ep])
        seen[ep] = c + 1
    n_keep = sum(keep_mask)
    print(f"data parquet: keeping {n_keep}/{data.num_rows} rows")
    if n_keep != new_total:
        # If any episode had MORE rows in the data parquet than actual_length asks for,
        # we drop the surplus. If FEWER rows (shouldn't happen since data parquet came
        # from sources that match declared length), bail.
        for ep in range(n_eps):
            available = ep_col.count(ep)
            if available < actual_lengths[ep]:
                raise RuntimeError(
                    f"Episode {ep}: data parquet has {available} rows but mp4 needs {actual_lengths[ep]}"
                )
    data = data.filter(pa.array(keep_mask, type=pa.bool_()))

    # Reissue `index` as a contiguous range (data parquet's original `index` no longer matches)
    data = replace_col(data, "index", list(range(data.num_rows)), pa.int64())
    pq.write_table(data, data_files[0])

    # Update meta/info.json total_frames
    info_path = dest / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] = new_total
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    print(f"\nRe-running consolidate ...")
    consolidate(dest, episodes_per_file=0)
    print(f"\nDone. Total: {n_eps} episodes, {new_total} frames")


if __name__ == "__main__":
    main()
