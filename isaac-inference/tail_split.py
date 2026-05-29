#!/usr/bin/env python3
"""
tail_split.py — properly "tail" a LeRobot v3.0 dataset by APPENDING freeze frames.

This is the inverse of strip_lang_and_tail.py: instead of removing trailing
frozen frames, it appends a fixed number of freeze frames to the end of every
kept episode, reproducing the per-subtask freeze logic that dataset_recorder.py
applies at record time (so a teleop-derived split dataset ends up tailed the
same way as the auto-recorded balanced dataset):

  - observation.state: holds the last joint state across all freeze frames.
  - action:
      GRASP / "Pick it up" / "Pick up ...": arm joints hold the last state, but
          the gripper channel holds the last *commanded* action (closing force)
          — so the freeze action differs slightly from a pure pose-hold.
      everything else (PLACE, ...): action == state (normal freeze).

Episodes whose task is in DROP_TASKS ("Go back to start position") are removed.

Videos are re-encoded exactly like balance_dataset.extract_video (per-episode
slice via ffmpeg fast-seek), with the freeze frames synthesised by cloning the
last frame (ffmpeg tpad=stop_mode=clone) in the same pass. Per-episode clips are
length-reconciled (av1 fast-seek can drop a frame) and then merged with
consolidate_dataset_videos.consolidate().

Usage:
    python tail_split.py <output_name> (--hub REPO_ID | --local FOLDER) \
        [--freeze N] [--push REPO_ID]

Output goes to synthetic_datasets/<output_name>/.
"""

import argparse
import json
import subprocess
from collections import OrderedDict, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from balance_dataset import CAMERAS, probe_vcodec
from consolidate_dataset_videos import consolidate, push_to_hub
from merge_datasets import reconcile_clip_lengths
from strip_lang_and_tail import _quantiles


THIS_DIR = Path(__file__).parent
SYNTH_DIR = THIS_DIR / "synthetic_datasets"

DROP_TASKS = {"Go back to start position"}
DEFAULT_FREEZE = 20


# ── freeze classification ───────────────────────────────────────────────────

def is_gripper_hold(task: str) -> bool:
    """GRASP / LIFT subtasks keep the gripper's last commanded action during the
    freeze (closing force); everything else freezes with action == state."""
    t = task.strip().lower()
    return t.startswith("grasp") or t.startswith("pick it up") or t.startswith("pick up the")


# ── source resolution ───────────────────────────────────────────────────────

def resolve_local(name: str) -> Path:
    p = SYNTH_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Local dataset not found: {p}")
    return p


def resolve_hub(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download
    print(f"  Downloading {repo_id} from the Hub ...")
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))


def load_concat_episodes(src: Path) -> pa.Table:
    files = sorted((src / "meta" / "episodes").glob("chunk-*/*.parquet"))
    tables = []
    for f in files:
        try:
            tables.append(pq.read_table(f))
        except Exception as e:
            print(f"  Warning: skipping corrupted {f.name}: {e}")
    table = pa.concat_tables(tables, promote_options="default")
    order = sorted(range(table.num_rows), key=lambda i: table.column("episode_index")[i].as_py())
    return table.take(order)


def load_concat_data(src: Path) -> pa.Table:
    files = sorted((src / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquets in {src}")
    return pa.concat_tables([pq.read_table(f) for f in files])


# ── video extract + freeze ──────────────────────────────────────────────────

def extract_video_with_freeze(
    src: Path, dst: Path, from_ts: float, n_frames: int, fps: int, freeze: int, encoder: str
) -> None:
    """Extract n_frames starting at from_ts (identical to balance_dataset.extract_video)
    and append `freeze` clones of the last frame, in a single encode pass."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    timescale = fps * 1000
    vf = f"trim=end_frame={n_frames},setpts=PTS-STARTPTS"
    if freeze > 0:
        vf += f",tpad=stop={freeze}:stop_mode=clone"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{from_ts}",
        "-i", str(src),
        "-frames:v", str(n_frames + freeze),
        "-c:v", encoder, "-pix_fmt", "yuv420p", "-g", "2",
        *(["-bf", "0"] if "264" in encoder else []),
        "-vf", vf,
        "-r", str(fps),
        "-video_track_timescale", str(timescale),
        str(dst),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src}:\n{res.stderr}")


# ── stats.json ──────────────────────────────────────────────────────────────

def write_stats_json(src_stats: Path, dst_stats: Path, data_parquet: Path) -> None:
    """Recompute scalar feature stats from the tailed data parquet; copy image stats
    from source (adding cloned freeze frames barely changes the pixel distribution)."""
    import numpy as np
    old = json.loads(src_stats.read_text())
    df = pd.read_parquet(data_parquet)

    new_stats = {}
    new_stats["action"] = _quantiles(np.stack(df["action"].values))
    new_stats["observation.state"] = _quantiles(np.stack(df["observation.state"].values))
    for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        new_stats[col] = _quantiles(df[col].to_numpy())
    for k in old:
        if k.startswith("observation.images."):
            new_stats[k] = old[k]
    with open(dst_stats, "w") as f:
        json.dump(new_stats, f, indent=4)


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("output", help="Output folder name inside synthetic_datasets/")
    src_grp = parser.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--hub", metavar="REPO_ID", help="HuggingFace dataset repo_id")
    src_grp.add_argument("--local", metavar="FOLDER", help="Local folder under synthetic_datasets/")
    parser.add_argument("--freeze", type=int, default=DEFAULT_FREEZE,
                        help=f"Freeze frames to append per episode (default: {DEFAULT_FREEZE})")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N kept episodes (smoke test; 0 = all)")
    parser.add_argument("--push", metavar="REPO_ID",
                        help="After building, upload to this HF repo and update the v3.0 tag")
    args = parser.parse_args()

    src = resolve_hub(args.hub) if args.hub else resolve_local(args.local)
    dest = SYNTH_DIR / args.output
    if dest.exists():
        raise FileExistsError(f"Output already exists: {dest}  — delete it first")

    freeze = args.freeze
    info = json.loads((src / "meta" / "info.json").read_text())
    fps: int = info["fps"]
    chunks_size: int = info["chunks_size"]
    encoder = probe_vcodec()
    print(f"Source: {src}")
    print(f"Output: {dest}")
    print(f"fps={fps}, freeze={freeze}, encoder={encoder}")

    # ── 1. Episodes: filter HOME, renumber ────────────────────────────────────
    eps = load_concat_episodes(src)
    src_ep_idx = eps.column("episode_index").to_pylist()
    src_tasks  = [tuple(t) for t in eps.column("tasks").to_pylist()]
    src_length = eps.column("length").to_pylist()
    vid_chunk = {cam: eps.column(f"videos/{cam}/chunk_index").to_pylist() for cam in CAMERAS}
    vid_file  = {cam: eps.column(f"videos/{cam}/file_index").to_pylist() for cam in CAMERAS}
    vid_from  = {cam: eps.column(f"videos/{cam}/from_timestamp").to_pylist() for cam in CAMERAS}

    kept = []  # list of dicts describing each kept episode in output order
    for row in range(eps.num_rows):
        task_list = src_tasks[row]
        if len(task_list) == 1 and task_list[0] in DROP_TASKS:
            continue
        kept.append({
            "src_row": row,
            "src_ep": src_ep_idx[row],
            "task": task_list[0],
            "length": src_length[row],
        })
    if args.limit:
        kept = kept[:args.limit]
        print(f"\n  [smoke test] limiting to first {args.limit} kept episodes")
    n_out = len(kept)
    print(f"\nSource episodes: {eps.num_rows}   kept: {n_out}   dropped: {eps.num_rows - len(kept)}")

    # ── 2. Unified task table (kept tasks, first-seen order) ──────────────────
    unified: "OrderedDict[str, int]" = OrderedDict()
    for k in kept:
        if k["task"] not in unified:
            unified[k["task"]] = len(unified)
    task_to_idx = dict(unified)
    print(f"Tasks kept: {len(unified)}")
    for s, i in unified.items():
        print(f"  {i}: {s!r}")

    # ── 3. Build merged (tailed) data parquet in memory ───────────────────────
    full = load_concat_data(src)
    state_type  = full.schema.field("observation.state").type
    action_type = full.schema.field("action").type
    ts_type     = full.schema.field("timestamp").type
    ep_col = full.column("episode_index").to_pylist()
    fi_col = full.column("frame_index").to_pylist()
    rows_by_ep: "dict[int, list]" = defaultdict(list)
    for ridx, (e, fi) in enumerate(zip(ep_col, fi_col)):
        rows_by_ep[e].append((fi, ridx))
    for e in rows_by_ep:
        rows_by_ep[e].sort()

    print("\nBuilding tailed data parquet ...")
    states_all, actions_all = [], []
    ts_all, fi_all, ep_all, idx_all, ti_all = [], [], [], [], []
    declared_lengths = []
    global_idx = 0
    for new_idx, k in enumerate(kept):
        order = [ridx for _, ridx in rows_by_ep[k["src_ep"]]]
        ep_tbl = full.take(order)
        n = ep_tbl.num_rows
        states = ep_tbl.column("observation.state").to_pylist()
        actions = ep_tbl.column("action").to_pylist()
        last_state = list(states[-1])
        last_action = list(actions[-1])
        freeze_action = list(last_state)
        if is_gripper_hold(k["task"]):
            freeze_action[-1] = last_action[-1]
        states += [list(last_state)] * freeze
        actions += [list(freeze_action)] * freeze
        new_len = n + freeze
        declared_lengths.append(new_len)
        states_all.extend(states)
        actions_all.extend(actions)
        ts_all.extend([j / fps for j in range(new_len)])
        fi_all.extend(range(new_len))
        ep_all.extend([new_idx] * new_len)
        idx_all.extend(range(global_idx, global_idx + new_len))
        ti_all.extend([task_to_idx[k["task"]]] * new_len)
        global_idx += new_len
        if (new_idx + 1) % 100 == 0 or new_idx == n_out - 1:
            print(f"  {new_idx + 1}/{n_out}")

    merged_data = pa.table({
        "action": pa.array(actions_all, type=action_type),
        "observation.state": pa.array(states_all, type=state_type),
        "timestamp": pa.array(ts_all, type=ts_type),
        "frame_index": pa.array(fi_all, type=pa.int64()),
        "episode_index": pa.array(ep_all, type=pa.int64()),
        "index": pa.array(idx_all, type=pa.int64()),
        "task_index": pa.array(ti_all, type=pa.int64()),
    }).select(full.schema.names)

    # ── 4. Extract per-episode videos (slice + freeze clones) ─────────────────
    print("\nExtracting per-episode video clips (+freeze) ...")
    for cam in CAMERAS:
        print(f"  [{cam}] {n_out} clips")
        for new_idx, k in enumerate(kept):
            row = k["src_row"]
            src_video = src / f"videos/{cam}/chunk-{vid_chunk[cam][row]:03d}/file-{vid_file[cam][row]:03d}.mp4"
            out = dest / f"videos/{cam}/chunk-{new_idx // chunks_size:03d}/file-{new_idx % chunks_size:03d}.mp4"
            extract_video_with_freeze(src_video, out, vid_from[cam][row], k["length"], fps, freeze, encoder)
            if (new_idx + 1) % 100 == 0 or new_idx == n_out - 1:
                print(f"    {new_idx + 1}/{n_out}")

    # ── 5. Reconcile clip lengths (mp4 = source of truth) ─────────────────────
    actual_lengths = reconcile_clip_lengths(dest, n_out, chunks_size, declared_lengths)

    # ── 6. Truncate data parquet per-episode if any clip shrank ───────────────
    if actual_lengths != declared_lengths:
        ep_data = merged_data.column("episode_index").to_pylist()
        keep_mask, seen = [], {}
        for e in ep_data:
            c = seen.get(e, 0)
            keep_mask.append(c < actual_lengths[e])
            seen[e] = c + 1
        merged_data = merged_data.filter(pa.array(keep_mask, type=pa.bool_()))
        merged_data = merged_data.set_column(
            merged_data.schema.get_field_index("index"),
            "index", pa.array(range(merged_data.num_rows), type=pa.int64()),
        )
    total_frames = sum(actual_lengths)

    (dest / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(merged_data, dest / "data" / "chunk-000" / "file-000.parquet")
    print(f"\nWrote data parquet: {merged_data.num_rows} rows")

    # ── 7. meta/episodes ──────────────────────────────────────────────────────
    rows = {"episode_index": [], "tasks": [], "length": [],
            "data/chunk_index": [], "data/file_index": [],
            "dataset_from_index": [], "dataset_to_index": []}
    for cam in CAMERAS:
        for sfx in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            rows[f"videos/{cam}/{sfx}"] = []
    cursor = 0
    for new_idx, (k, L) in enumerate(zip(kept, actual_lengths)):
        rows["episode_index"].append(new_idx)
        rows["tasks"].append([k["task"]])
        rows["length"].append(L)
        rows["data/chunk_index"].append(0)
        rows["data/file_index"].append(0)
        rows["dataset_from_index"].append(cursor)
        rows["dataset_to_index"].append(cursor + L)
        for cam in CAMERAS:
            rows[f"videos/{cam}/chunk_index"].append(new_idx // chunks_size)
            rows[f"videos/{cam}/file_index"].append(new_idx % chunks_size)
            rows[f"videos/{cam}/from_timestamp"].append(0.0)
            rows[f"videos/{cam}/to_timestamp"].append(L / fps)
        cursor += L

    eps_arrays = {
        "episode_index": pa.array(rows["episode_index"], type=pa.int64()),
        "tasks": pa.array(rows["tasks"], type=pa.list_(pa.string())),
        "length": pa.array(rows["length"], type=pa.int64()),
        "data/chunk_index": pa.array(rows["data/chunk_index"], type=pa.int64()),
        "data/file_index": pa.array(rows["data/file_index"], type=pa.int64()),
        "dataset_from_index": pa.array(rows["dataset_from_index"], type=pa.int64()),
        "dataset_to_index": pa.array(rows["dataset_to_index"], type=pa.int64()),
    }
    for cam in CAMERAS:
        eps_arrays[f"videos/{cam}/chunk_index"] = pa.array(rows[f"videos/{cam}/chunk_index"], type=pa.int64())
        eps_arrays[f"videos/{cam}/file_index"] = pa.array(rows[f"videos/{cam}/file_index"], type=pa.int64())
        eps_arrays[f"videos/{cam}/from_timestamp"] = pa.array(rows[f"videos/{cam}/from_timestamp"], type=pa.float64())
        eps_arrays[f"videos/{cam}/to_timestamp"] = pa.array(rows[f"videos/{cam}/to_timestamp"], type=pa.float64())
    (dest / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(eps_arrays), dest / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    # ── 8. meta/tasks.parquet ─────────────────────────────────────────────────
    tasks_df = pd.DataFrame({"task_index": range(len(unified))}, index=list(unified.keys()))
    tasks_df.to_parquet(dest / "meta" / "tasks.parquet")

    # ── 9. meta/info.json ─────────────────────────────────────────────────────
    new_info = dict(info)
    new_info["total_episodes"] = n_out
    new_info["total_frames"] = total_frames
    new_info["total_tasks"] = len(unified)
    new_info["splits"] = {"train": f"0:{n_out}"}
    codec = "av1" if encoder == "libsvtav1" else "h264" if "264" in encoder else encoder
    for cam in CAMERAS:
        if cam in new_info.get("features", {}):
            new_info["features"][cam]["info"]["video.codec"] = codec
    with open(dest / "meta" / "info.json", "w") as f:
        json.dump(new_info, f, indent=4)

    # ── 10. Consolidate videos (merge per-episode + probe PTS) ────────────────
    print("\nConsolidating videos ...")
    consolidate(dest, episodes_per_file=0)

    # ── 11. meta/stats.json (after consolidate; reads final data parquet) ─────
    print("Writing stats.json ...")
    write_stats_json(src / "meta" / "stats.json", dest / "meta" / "stats.json",
                     dest / "data" / "chunk-000" / "file-000.parquet")

    print(f"\nDone.")
    print(f"  Output:   {dest}")
    print(f"  Episodes: {n_out}   Frames: {total_frames}   Tasks: {len(unified)}")

    if args.push:
        print(f"\nPushing to {args.push} ...")
        push_to_hub(dest, args.push)

    print(f"\nVerify with:\n  python plot_dataset_stats.py {args.output}")


if __name__ == "__main__":
    main()
