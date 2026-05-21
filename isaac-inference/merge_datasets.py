#!/usr/bin/env python3
"""
merge_datasets.py — Concatenate two (or more) LeRobot v3.0 datasets into one.

Episode order = source order on the CLI. The first source keeps its episode
indices 0..N0-1; subsequent sources are shifted up. Task strings are unioned
preserving first-seen order, and task_index in data parquets is remapped.

Videos are re-encoded to h264 (NVDEC works on every modern NVIDIA GPU; av1
hardware decode is RTX-40+ only — h264 is the faster training-time codec on
older hardware).

Usage:
    python merge_datasets.py <output_name> \\
        [--local FOLDER]... [--hub REPO_ID]... \\
        [--push REPO_ID]

Output goes to synthetic_datasets/<output_name>/.
"""

import argparse
import json
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from balance_dataset import CAMERAS, extract_video, probe_vcodec
import balance_dataset as _bd
from consolidate_dataset_videos import consolidate, push_to_hub


THIS_DIR = Path(__file__).parent
SYNTH_DIR = THIS_DIR / "synthetic_datasets"


# ── source resolution ─────────────────────────────────────────────────────────

def resolve_local(name: str) -> Path:
    p = SYNTH_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Local dataset not found: {p}")
    return p


def resolve_hub(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download
    print(f"  Downloading {repo_id} from the Hub ...")
    path = snapshot_download(repo_id=repo_id, repo_type="dataset")
    return Path(path)


# ── schema checks ─────────────────────────────────────────────────────────────

def load_info(src: Path) -> dict:
    with open(src / "meta" / "info.json") as f:
        return json.load(f)


def check_compatible(infos: list[dict]) -> None:
    base = infos[0]
    for i, info in enumerate(infos[1:], 1):
        if info["fps"] != base["fps"]:
            raise ValueError(f"fps mismatch: {base['fps']} vs {info['fps']} (source {i})")
        if info["robot_type"] != base["robot_type"]:
            raise ValueError(f"robot_type mismatch: {base['robot_type']} vs {info['robot_type']}")
        for key in ("observation.state", "action"):
            if base["features"][key]["shape"] != info["features"][key]["shape"]:
                raise ValueError(f"{key} shape mismatch (source {i})")
        for cam in CAMERAS:
            if base["features"][cam]["shape"] != info["features"][cam]["shape"]:
                raise ValueError(f"{cam} shape mismatch (source {i})")


# ── task table union ──────────────────────────────────────────────────────────

def load_tasks_dict(src: Path) -> dict[int, str]:
    """Return {old_task_index: task_string} from a source's meta/tasks.parquet."""
    t = pq.read_table(src / "meta" / "tasks.parquet")
    # Schema is ['task_index', '__index_level_0__'] in this codebase.
    str_col = "__index_level_0__" if "__index_level_0__" in t.schema.names else "task"
    return dict(zip(t.column("task_index").to_pylist(), t.column(str_col).to_pylist()))


def unify_tasks(source_tasks: list[dict[int, str]]) -> tuple[list[str], list[dict[int, int]]]:
    """Union task strings preserving first-seen order. Returns (unified_list, per-source remap)."""
    unified: OrderedDict[str, int] = OrderedDict()
    remaps: list[dict[int, int]] = []
    for tasks in source_tasks:
        remap: dict[int, int] = {}
        for old_idx, s in tasks.items():
            if s not in unified:
                unified[s] = len(unified)
            remap[old_idx] = unified[s]
        remaps.append(remap)
    return list(unified.keys()), remaps


def write_tasks_parquet(unified: list[str], dst: Path) -> None:
    table = pa.table({
        "task_index": pa.array(list(range(len(unified))), type=pa.int64()),
        "__index_level_0__": pa.array(unified, type=pa.string()),
    })
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)


# ── data parquet merge ────────────────────────────────────────────────────────

def load_concat_data_parquet(src: Path) -> pa.Table:
    files = sorted((src / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquets in {src}")
    return pa.concat_tables([pq.read_table(f) for f in files])


def remap_data_table(
    table: pa.Table,
    ep_offset: int,
    frame_offset: int,
    task_remap: dict[int, int],
) -> pa.Table:
    n = table.num_rows

    def replace(t: pa.Table, col: str, values, typ) -> pa.Table:
        return t.set_column(t.schema.get_field_index(col), col, pa.array(values, type=typ))

    old_eps = table.column("episode_index").to_pylist()
    table = replace(table, "episode_index", [e + ep_offset for e in old_eps], pa.int64())

    table = replace(table, "index", list(range(frame_offset, frame_offset + n)), pa.int64())

    old_ti = table.column("task_index").to_pylist()
    table = replace(table, "task_index", [task_remap[x] for x in old_ti], pa.int64())

    # info.json declares timestamp as float32; balance_dataset.py historically wrote
    # float64 — normalize so cross-source concat succeeds.
    ts_idx = table.schema.get_field_index("timestamp")
    if ts_idx != -1 and table.schema.field("timestamp").type != pa.float32():
        ts_vals = table.column("timestamp").cast(pa.float32())
        table = table.set_column(ts_idx, "timestamp", ts_vals)

    return table


# ── meta/episodes merge ───────────────────────────────────────────────────────

# Minimal columns we maintain. We strip per-episode stats/* columns because
# they require recomputation; downstream training reads global stats elsewhere.
EPISODES_CORE_COLS = [
    "episode_index",
    "tasks",
    "length",
    "data/chunk_index",
    "data/file_index",
    "dataset_from_index",
    "dataset_to_index",
] + [
    col
    for cam in CAMERAS
    for col in (
        f"videos/{cam}/chunk_index",
        f"videos/{cam}/file_index",
        f"videos/{cam}/from_timestamp",
        f"videos/{cam}/to_timestamp",
    )
]


def load_concat_episodes(src: Path) -> pa.Table:
    files = sorted((src / "meta" / "episodes").glob("chunk-*/*.parquet"))
    tables = []
    for f in files:
        try:
            tables.append(pq.read_table(f))
        except Exception as e:
            print(f"  Warning: skipping corrupted {f.name}: {e}")
    table = pa.concat_tables(tables, promote_options="default")
    # Ensure deterministic order by episode_index.
    order = sorted(range(table.num_rows), key=lambda i: table.column("episode_index")[i].as_py())
    return table.take(order)


def build_merged_episodes(
    per_source_eps: list[pa.Table],
    ep_offsets: list[int],
    frame_offsets_per_source: list[int],
    fps: int,
    chunks_size: int,
    override_lengths: list[int] | None = None,
) -> pa.Table:
    """Build merged episodes table with renumbered indices + cumulative frame offsets.
    Per-episode video file paths point to chunk-XXX/file-XXX.mp4 (one mp4 per episode);
    the final consolidate() pass merges them into one mp4 per camera and rewrites these
    indices + the from/to_timestamps from probed PTS.

    If override_lengths is given (flat list of length total_episodes), use those instead
    of the per-source declared lengths — for length reconciliation after extraction."""
    rows: dict[str, list] = {col: [] for col in EPISODES_CORE_COLS}

    global_frame_cursor = 0
    global_ep_cursor = 0
    for src_eps, ep_off in zip(per_source_eps, ep_offsets):
        n = src_eps.num_rows
        lengths = src_eps.column("length").to_pylist()
        tasks_col = src_eps.column("tasks").to_pylist()

        for i in range(n):
            ep_len = override_lengths[global_ep_cursor] if override_lengths is not None else lengths[i]
            rows["episode_index"].append(global_ep_cursor)
            rows["tasks"].append(tasks_col[i])
            rows["length"].append(ep_len)
            rows["data/chunk_index"].append(0)
            rows["data/file_index"].append(0)
            rows["dataset_from_index"].append(global_frame_cursor)
            rows["dataset_to_index"].append(global_frame_cursor + ep_len)
            for cam in CAMERAS:
                rows[f"videos/{cam}/chunk_index"].append(global_ep_cursor // chunks_size)
                rows[f"videos/{cam}/file_index"].append(global_ep_cursor % chunks_size)
                rows[f"videos/{cam}/from_timestamp"].append(0.0)
                rows[f"videos/{cam}/to_timestamp"].append(ep_len / fps)
            global_frame_cursor += ep_len
            global_ep_cursor += 1

    arrays = {
        "episode_index": pa.array(rows["episode_index"], type=pa.int64()),
        "tasks": pa.array(rows["tasks"], type=pa.list_(pa.string())),
        "length": pa.array(rows["length"], type=pa.int64()),
        "data/chunk_index": pa.array(rows["data/chunk_index"], type=pa.int64()),
        "data/file_index": pa.array(rows["data/file_index"], type=pa.int64()),
        "dataset_from_index": pa.array(rows["dataset_from_index"], type=pa.int64()),
        "dataset_to_index": pa.array(rows["dataset_to_index"], type=pa.int64()),
    }
    for cam in CAMERAS:
        arrays[f"videos/{cam}/chunk_index"] = pa.array(rows[f"videos/{cam}/chunk_index"], type=pa.int64())
        arrays[f"videos/{cam}/file_index"] = pa.array(rows[f"videos/{cam}/file_index"], type=pa.int64())
        arrays[f"videos/{cam}/from_timestamp"] = pa.array(rows[f"videos/{cam}/from_timestamp"], type=pa.float64())
        arrays[f"videos/{cam}/to_timestamp"] = pa.array(rows[f"videos/{cam}/to_timestamp"], type=pa.float64())

    return pa.table(arrays)


# ── video extract + merge ─────────────────────────────────────────────────────

def _probe_nb_frames(mp4: Path) -> int:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {mp4}: {res.stderr}")
    return int(res.stdout.strip())


def _trim_mp4(src: Path, n_frames: int) -> None:
    """Trim mp4 in-place to exactly n_frames via stream copy."""
    tmp = src.with_suffix(".trim.mp4")
    res = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-frames:v", str(n_frames), "-c", "copy", str(tmp)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg trim failed for {src}: {res.stderr}")
    tmp.rename(src)


def reconcile_clip_lengths(dest: Path, n_eps: int, chunks_size: int, declared_lengths: list[int]) -> list[int]:
    """Probe each per-episode mp4 (per camera); if actual frame count is less than
    declared length, trim peers to match. Fast-seek extraction occasionally loses a
    frame at av1 GOP boundaries, so the mp4 is the source of truth.

    Returns the reconciled per-episode lengths (same length as declared_lengths)."""
    print(f"Reconciling clip lengths across {len(CAMERAS)} cameras × {n_eps} episodes ...")
    actual_lengths: list[int] = []
    n_changed = 0
    for i in range(n_eps):
        chunk = i // chunks_size
        file = i % chunks_size
        counts = {
            cam: _probe_nb_frames(dest / f"videos/{cam}/chunk-{chunk:03d}/file-{file:03d}.mp4")
            for cam in CAMERAS
        }
        actual = min(min(counts.values()), declared_lengths[i])
        if actual != declared_lengths[i] or any(c != actual for c in counts.values()):
            print(f"  ep {i}: declared={declared_lengths[i]}, "
                  + ", ".join(f"{cam.split('.')[-1]}={c}" for cam, c in counts.items())
                  + f"  → {actual}")
            n_changed += 1
            for cam, c in counts.items():
                if c != actual:
                    _trim_mp4(dest / f"videos/{cam}/chunk-{chunk:03d}/file-{file:03d}.mp4", actual)
        actual_lengths.append(actual)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n_eps}")
    drift = sum(declared_lengths) - sum(actual_lengths)
    print(f"  {n_changed} clip(s) adjusted; {drift} frames trimmed total")
    return actual_lengths


def extract_per_episode_videos(
    sources: list[tuple[Path, str]],
    dest: Path,
    fps: int,
    chunks_size: int,
) -> None:
    """Extract each episode's slice (using its source's from_timestamp + length/fps)
    into dest/videos/<cam>/chunk-XXX/file-XXX.mp4 — one mp4 per episode per camera.
    consolidate() will later merge these and re-probe PTS. This guarantees only
    referenced frames appear in the merged video, even if a source mp4 had
    unreferenced content (e.g. tailed dataset)."""
    # Force balance_dataset's encoder probe once so all extract_video calls share it.
    if _bd._VCODEC is None:
        _bd._VCODEC = probe_vcodec()
        print(f"  Video encoder: {_bd._VCODEC}")

    for cam in CAMERAS:
        global_ep_idx = 0
        for src_idx, (src, codec) in enumerate(sources):
            src_eps = load_concat_episodes(src)
            n = src_eps.num_rows
            chunk_col = src_eps.column(f"videos/{cam}/chunk_index").to_pylist()
            file_col  = src_eps.column(f"videos/{cam}/file_index").to_pylist()
            from_ts_col = src_eps.column(f"videos/{cam}/from_timestamp").to_pylist()
            length_col  = src_eps.column("length").to_pylist()

            print(f"  [{cam}] source {src_idx} ({codec}): extracting {n} per-episode clips")
            for i in range(n):
                src_video = src / f"videos/{cam}/chunk-{chunk_col[i]:03d}/file-{file_col[i]:03d}.mp4"
                out = dest / f"videos/{cam}/chunk-{global_ep_idx // chunks_size:03d}/file-{global_ep_idx % chunks_size:03d}.mp4"
                extract_video(src_video, out, from_ts_col[i], length_col[i], fps)
                global_ep_idx += 1
                if (i + 1) % 100 == 0 or i == n - 1:
                    print(f"    {i + 1}/{n}")


# ── info.json ─────────────────────────────────────────────────────────────────

def write_info(template: dict, dest: Path, total_episodes: int, total_frames: int, total_tasks: int) -> None:
    new_info = dict(template)
    new_info["total_episodes"] = total_episodes
    new_info["total_frames"] = total_frames
    new_info["total_tasks"] = total_tasks
    new_info["splits"] = {"train": f"0:{total_episodes}"}
    for cam in CAMERAS:
        if cam in new_info.get("features", {}):
            feat = new_info["features"][cam]
            feat["info"]["video.codec"] = "h264"
            if "video_info" in feat:
                feat["video_info"]["video.codec"] = "h264"
    with open(dest / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=4)


# ── subtask_metadata.jsonl (first source only) ────────────────────────────────

def maybe_copy_subtask_metadata(first_source: Path, dest: Path) -> None:
    src = first_source / "subtask_metadata.jsonl"
    if src.exists():
        print(f"  Copying subtask_metadata.jsonl from source 0 ({src.parent.name})")
        shutil.copy2(src, dest / "subtask_metadata.jsonl")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", help="Output folder name inside synthetic_datasets/")
    parser.add_argument("--local", action="append", default=[],
                        help="Local dataset folder name under synthetic_datasets/ (repeatable)")
    parser.add_argument("--hub", action="append", default=[],
                        help="HuggingFace dataset repo_id (repeatable)")
    parser.add_argument("--push", metavar="REPO_ID",
                        help="After merging+consolidating, upload to this HF repo and update the v3.0 tag")
    args = parser.parse_args()

    # Resolve sources in argv order. Re-parse to preserve user order.
    import sys
    sources: list[Path] = []
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--local":
            sources.append(resolve_local(argv[i + 1]))
            i += 2
        elif argv[i] == "--hub":
            sources.append(resolve_hub(argv[i + 1]))
            i += 2
        elif argv[i] == "--push":
            i += 2
        else:
            i += 1

    if len(sources) < 2:
        parser.error("Need at least two sources (combine --local and/or --hub).")

    dest = SYNTH_DIR / args.output
    if dest.exists():
        raise FileExistsError(f"Output already exists: {dest}  — delete it first")

    # ── 1+2. Schemas ──────────────────────────────────────────────────────────
    print("Sources:")
    for s in sources:
        print(f"  {s}")
    infos = [load_info(s) for s in sources]
    check_compatible(infos)
    fps: int = infos[0]["fps"]
    source_codecs = [info["features"][CAMERAS[0]]["info"]["video.codec"] for info in infos]
    for s, codec, info in zip(sources, source_codecs, infos):
        print(f"  {s.name}: {info['total_episodes']} eps, {info['total_frames']} frames, codec={codec}")

    # ── 3. Unified task table ─────────────────────────────────────────────────
    source_tasks = [load_tasks_dict(s) for s in sources]
    unified_tasks, task_remaps = unify_tasks(source_tasks)
    print(f"\nUnified task table: {len(unified_tasks)} tasks")
    for idx, s in enumerate(unified_tasks):
        print(f"  {idx}: {s!r}")

    # ── 4. Per-source meta/episodes ───────────────────────────────────────────
    per_source_eps = [load_concat_episodes(s) for s in sources]

    # Compute per-source episode + frame offsets
    ep_offsets: list[int] = []
    frame_offsets: list[int] = []
    cum_ep = 0
    cum_frames = 0
    for eps_table in per_source_eps:
        ep_offsets.append(cum_ep)
        frame_offsets.append(cum_frames)
        cum_ep += eps_table.num_rows
        cum_frames += sum(eps_table.column("length").to_pylist())
    total_episodes = cum_ep
    total_frames = cum_frames
    print(f"\nMerged total: {total_episodes} episodes, {total_frames} frames")

    # ── 5. Build merged data table (in memory — write after length reconciliation) ─
    print("\nBuilding merged data parquet ...")
    merged_chunks: list[pa.Table] = []
    for src, ep_off, frame_off, remap in zip(sources, ep_offsets, frame_offsets, task_remaps):
        t = load_concat_data_parquet(src)
        t = remap_data_table(t, ep_off, frame_off, remap)
        merged_chunks.append(t)
    merged_data = pa.concat_tables(merged_chunks, promote_options="default")
    print(f"  {merged_data.num_rows} rows (will be reconciled against actual mp4 frame counts)")

    # ── 6. Extract per-episode videos into dest tree ──────────────────────────
    chunks_size: int = infos[0]["chunks_size"]
    print("\nExtracting per-episode video clips ...")
    src_codec_pairs = list(zip(sources, source_codecs))
    extract_per_episode_videos(src_codec_pairs, dest, fps, chunks_size)

    # ── 6b. Reconcile clip lengths (mp4 = source of truth) ────────────────────
    declared_lengths: list[int] = []
    for src_eps in per_source_eps:
        declared_lengths.extend(src_eps.column("length").to_pylist())
    actual_lengths = reconcile_clip_lengths(dest, total_episodes, chunks_size, declared_lengths)

    # ── 6c. Truncate data parquet per-episode if any clip shrank ──────────────
    if actual_lengths != declared_lengths:
        ep_col = merged_data.column("episode_index").to_pylist()
        keep_mask: list[bool] = []
        seen: dict[int, int] = {}
        for ep in ep_col:
            c = seen.get(ep, 0)
            keep_mask.append(c < actual_lengths[ep])
            seen[ep] = c + 1
        merged_data = merged_data.filter(pa.array(keep_mask, type=pa.bool_()))
        merged_data = merged_data.set_column(
            merged_data.schema.get_field_index("index"),
            "index",
            pa.array(list(range(merged_data.num_rows)), type=pa.int64()),
        )
        total_frames = sum(actual_lengths)
        print(f"  Reconciled total: {total_frames} frames ({merged_data.num_rows} parquet rows)")

    (dest / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(merged_data, dest / "data" / "chunk-000" / "file-000.parquet")
    print(f"  Wrote {dest / 'data/chunk-000/file-000.parquet'}")

    # ── 7. meta/episodes + meta/tasks + meta/info.json ────────────────────────
    print("\nWriting meta files ...")
    merged_eps = build_merged_episodes(
        per_source_eps, ep_offsets, frame_offsets, fps, chunks_size,
        override_lengths=actual_lengths,
    )
    (dest / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(merged_eps, dest / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    write_tasks_parquet(unified_tasks, dest / "meta" / "tasks.parquet")
    write_info(infos[0], dest, total_episodes, total_frames, len(unified_tasks))

    # ── 9. subtask_metadata.jsonl (source 0 only) ─────────────────────────────
    maybe_copy_subtask_metadata(sources[0], dest)

    # ── 10. Final consolidate pass: probe PTS, fix from/to_timestamp ──────────
    print("\nFinal consolidate pass (probe PTS + fix timestamps) ...")
    consolidate(dest, episodes_per_file=0)

    print(f"\nDone.")
    print(f"  Output: {dest}")
    print(f"  Episodes: {total_episodes}  Frames: {total_frames}  Tasks: {len(unified_tasks)}")

    if args.push:
        print(f"\nPushing to {args.push} ...")
        push_to_hub(dest, args.push)

    print(f"\nVerify with:")
    print(f"  python plot_dataset_stats.py {args.output}")


if __name__ == "__main__":
    main()
