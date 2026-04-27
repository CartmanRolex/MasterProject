"""
Analyze and fix frozen tail frames in a LeRobot V3 dataset.

Usage:
  python tailer.py analyze DATASET_PATH
  python tailer.py fix     DATASET_PATH OUTPUT_PATH [--min-frozen N] [--task-filter STR]

A "frozen tail" is a run of trailing frames where the arm state (joints 0-4,
gripper excluded) is identical to the last frame. Useful for "Place" tasks
where the robot should dwell at the goal pose.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tasks(dataset_path: Path) -> pd.DataFrame:
    f = dataset_path / "meta" / "tasks.parquet"
    df = pd.read_parquet(f)
    if "task" not in df.columns:
        df = df.reset_index()
        # After reset_index the old string index lands in a column called 'index'
        for col in df.columns:
            if col not in ("task_index",) and df[col].dtype == object:
                df = df.rename(columns={col: "task"})
                break
    return df.rename(columns={"index": "task"}) if "task" not in df.columns else df


def _load_ep_meta(dataset_path: Path) -> tuple[list[Path], pd.DataFrame]:
    files = sorted((dataset_path / "meta" / "episodes").rglob("*.parquet"))
    parts = [pd.read_parquet(f) for f in files]
    return files, pd.concat(parts, ignore_index=True)


def _load_data(dataset_path: Path) -> tuple[list[Path], list[pa.Schema], pd.DataFrame]:
    files = sorted((dataset_path / "data").rglob("*.parquet"))
    schemas, parts = [], []
    for f in files:
        t = pq.read_table(f)
        schemas.append(t.schema)
        parts.append(t.to_pandas())
    return files, schemas, pd.concat(parts, ignore_index=True)


def count_frozen_tail(states: np.ndarray, tol: float = 0.1) -> int:
    """Count trailing frames whose arm state (joints 0-4) matches the last frame."""
    if len(states) == 0:
        return 0
    last = states[-1][:5]
    count = 0
    for row in reversed(states):
        if np.allclose(row[:5], last, atol=tol):
            count += 1
        else:
            break
    return count


def _task_label(task_str: str) -> str:
    """Return a short label for grouping (first 3 words)."""
    return " ".join(task_str.split()[:3])


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def cmd_analyze(dataset_path: Path) -> None:
    with open(dataset_path / "meta" / "info.json") as f:
        info = json.load(f)

    tasks_df = _load_tasks(dataset_path)
    idx2task = {int(r["task_index"]): str(r["task"]) for _, r in tasks_df.iterrows()}

    _, _, df = _load_data(dataset_path)

    # Per-episode frozen count
    rows = []
    for ep_idx, ep_df in df.groupby("episode_index"):
        ep_df = ep_df.sort_values("frame_index")
        states = np.stack(ep_df["observation.state"].values)
        frozen = count_frozen_tail(states)
        task_indices = ep_df["task_index"].unique()
        task_str = idx2task.get(int(task_indices[0]), "unknown") if len(task_indices) else "unknown"
        rows.append({"episode_index": ep_idx, "task": task_str, "frozen": frozen, "length": len(ep_df)})

    per_ep = pd.DataFrame(rows)

    print(f"\n=== {dataset_path} ({info.get('total_episodes', '?')} episodes, {info.get('fps', 30)} fps) ===\n")

    header = f"{'Task':<35} {'eps':>5} {'min':>5} {'mean':>6} {'max':>5} {'need_pad':>9}"
    print(header)
    print("-" * len(header))

    for task_str, group in per_ep.groupby("task"):
        frozen = group["frozen"]
        label = task_str if len(task_str) <= 35 else task_str[:32] + "..."
        need = (frozen < 24).sum()
        print(f"{label:<35} {len(group):>5} {frozen.min():>5} {frozen.mean():>6.1f} {frozen.max():>5} {need:>9}")

    print()
    total_need = (per_ep["frozen"] < 24).sum()
    print(f"Total episodes needing padding (< 24 frozen): {total_need} / {len(per_ep)}")


# ---------------------------------------------------------------------------
# fix
# ---------------------------------------------------------------------------

def _recompute_ep_stats(ep_data: pd.DataFrame, ep_meta_row: pd.Series, ep_meta_schema: pa.Schema) -> dict:
    row = ep_meta_row.to_dict()
    row["length"] = len(ep_data)

    skip_prefixes = ("stats/observation.images", "stats/observation.video",
                     "videos/", "meta/", "data/")
    array_cols = {"action", "observation.state"}
    scalar_stat_cols = {"timestamp", "frame_index", "index", "episode_index", "task_index"}

    stat_cols = [c for c in ep_meta_row.index if c.startswith("stats/")]
    for stat_col in stat_cols:
        if any(stat_col.startswith(p) for p in skip_prefixes):
            continue
        parts = stat_col.split("/")
        stat_type = parts[-1]
        feature = "/".join(parts[1:-1])

        if feature in array_cols and feature in ep_data.columns:
            stacked = np.stack(ep_data[feature].values)
            if stat_type == "min":
                row[stat_col] = stacked.min(axis=0).tolist()
            elif stat_type == "max":
                row[stat_col] = stacked.max(axis=0).tolist()
            elif stat_type == "mean":
                row[stat_col] = stacked.mean(axis=0).tolist()
            elif stat_type == "std":
                row[stat_col] = stacked.std(axis=0).tolist()
            elif stat_type == "count":
                row[stat_col] = [len(stacked)]

        elif feature in scalar_stat_cols and feature in ep_data.columns:
            vals = ep_data[feature].values.astype(float)
            if stat_type == "min":
                row[stat_col] = [float(vals.min())]
            elif stat_type == "max":
                row[stat_col] = [float(vals.max())]
            elif stat_type == "mean":
                row[stat_col] = [float(vals.mean())]
            elif stat_type == "std":
                row[stat_col] = [float(vals.std())]
            elif stat_type == "count":
                row[stat_col] = [len(vals)]

    return row


def cmd_fix(dataset_path: Path, output_path: Path, min_frozen: int, task_filter: str) -> None:
    if output_path.resolve() == dataset_path.resolve():
        sys.exit("ERROR: output path must differ from source.")

    print(f"Copying {dataset_path} → {output_path} ...")
    if output_path.exists():
        shutil.rmtree(output_path)
    shutil.copytree(dataset_path, output_path)

    with open(output_path / "meta" / "info.json") as f:
        info = json.load(f)
    fps = info.get("fps", 30)

    tasks_df = _load_tasks(output_path)
    idx2task = {int(r["task_index"]): str(r["task"]) for _, r in tasks_df.iterrows()}

    ep_meta_files, ep_meta = _load_ep_meta(output_path)
    ep_meta_schema = pq.read_table(ep_meta_files[0]).schema

    data_files, data_schemas, df = _load_data(output_path)

    # Find episodes to pad
    to_pad: dict[int, int] = {}  # ep_idx → n_frames_to_add
    for ep_idx, ep_df in df.groupby("episode_index"):
        ep_df = ep_df.sort_values("frame_index")
        task_idx = int(ep_df["task_index"].iloc[0])
        task_str = idx2task.get(task_idx, "")
        if task_filter and task_filter.lower() not in task_str.lower():
            continue
        states = np.stack(ep_df["observation.state"].values)
        frozen = count_frozen_tail(states)
        if frozen < min_frozen:
            to_pad[ep_idx] = min_frozen - frozen

    if not to_pad:
        print("No episodes need padding. Done.")
        return

    print(f"Padding {len(to_pad)} episodes (min_frozen={min_frozen}, filter='{task_filter}')...")

    # Build tail rows for each episode and append to df
    tail_chunks = []
    for ep_idx, n_add in to_pad.items():
        ep_df = df[df["episode_index"] == ep_idx].sort_values("frame_index")
        last = ep_df.iloc[-1].copy()
        last_fi = int(last["frame_index"])
        for i in range(1, n_add + 1):
            new_row = last.copy()
            new_row["frame_index"] = last_fi + i
            tail_chunks.append(new_row)

    tail_df = pd.DataFrame(tail_chunks)
    df = pd.concat([df, tail_df], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    df["index"] = range(len(df))

    # Recompute dataset_from_index / dataset_to_index for all episodes
    ep_ranges = df.groupby("episode_index")["index"].agg(["min", "max"])

    # Update episode metadata
    updated_meta_rows = []
    for _, meta_row in ep_meta.iterrows():
        ep_idx = int(meta_row["episode_index"])
        ep_data = df[df["episode_index"] == ep_idx].sort_values("frame_index")
        row = _recompute_ep_stats(ep_data, meta_row, ep_meta_schema)
        if ep_idx in ep_ranges.index:
            row["dataset_from_index"] = int(ep_ranges.loc[ep_idx, "min"])
            row["dataset_to_index"] = int(ep_ranges.loc[ep_idx, "max"]) + 1
        updated_meta_rows.append(row)

    new_ep_meta = pd.DataFrame(updated_meta_rows).sort_values("episode_index").reset_index(drop=True)

    # Write data parquets (all into first file for simplicity, preserving schema)
    original_schema = data_schemas[0]
    out_data_file = data_files[0]
    try:
        out_table = pa.Table.from_pandas(df, schema=original_schema, preserve_index=False)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        out_table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(out_table, out_data_file)

    # Remove extra data files if they existed (we merged into one)
    for f in data_files[1:]:
        f.unlink(missing_ok=True)

    # Write episode metadata
    arrow_cols = {}
    for col_name in ep_meta_schema.names:
        field_type = ep_meta_schema.field(col_name).type
        if col_name == "tasks":
            clean = []
            for val in new_ep_meta[col_name]:
                if isinstance(val, np.ndarray):
                    clean.append(val.tolist())
                elif isinstance(val, (list, tuple)):
                    clean.append(list(val))
                else:
                    clean.append([str(val)])
            arrow_cols[col_name] = pa.array(clean, type=field_type)
        elif col_name.startswith("stats/"):
            clean = [v.tolist() if isinstance(v, np.ndarray) else v for v in new_ep_meta[col_name]]
            try:
                arrow_cols[col_name] = pa.array(clean, type=field_type)
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                arrow_cols[col_name] = pa.array(clean)
        else:
            try:
                arrow_cols[col_name] = pa.array(new_ep_meta[col_name].values, type=field_type)
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                arrow_cols[col_name] = pa.array(new_ep_meta[col_name].values)

    pq.write_table(pa.table(arrow_cols), ep_meta_files[0])

    # Update info.json
    total_added = sum(to_pad.values())
    info["total_frames"] = int(info.get("total_frames", 0)) + total_added
    with open(output_path / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDone.")
    print(f"  Episodes padded : {len(to_pad)}")
    print(f"  Frames added    : {total_added}")
    print(f"  Total frames    : {info['total_frames']}")
    print(f"  Output          : {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_analyze = sub.add_parser("analyze", help="Print frozen-tail statistics per task")
    p_analyze.add_argument("dataset", type=Path)

    p_fix = sub.add_parser("fix", help="Pad episodes with insufficient frozen tail")
    p_fix.add_argument("dataset", type=Path)
    p_fix.add_argument("output", type=Path)
    p_fix.add_argument("--min-frozen", type=int, default=24,
                       help="Target number of frozen frames at episode end (default: 24)")
    p_fix.add_argument("--task-filter", type=str, default="",
                       help="Only pad episodes whose task contains this string (case-insensitive)")

    args = parser.parse_args()

    if args.cmd == "analyze":
        cmd_analyze(args.dataset)
    elif args.cmd == "fix":
        cmd_fix(args.dataset, args.output, args.min_frozen, args.task_filter)


if __name__ == "__main__":
    main()
