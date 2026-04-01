"""Export edited dataset to disk."""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data_loader import load_tasks
from .state import ProgressTracker


def export_dataset(
    dataset_path: Path,
    progress: ProgressTracker,
    output_path: Path | None = None,
) -> str:
    """Export the edited dataset across ALL episodes. Returns a status message string.

    If output_path is None, overwrites in place.
    If output_path == dataset_path, also overwrites in place.
    Otherwise copies meta/ and data/ to the new path.
    """
    if output_path is None:
        output_path = dataset_path

    overwrite = output_path.resolve() == dataset_path.resolve()
    messages = []

    # --- 0. Copy basic structure if exporting to a new directory ---
    if not overwrite:
        messages.append(f"Copying meta/ and data/ to {output_path} ...")
        for subdir in ["meta", "data"]:
            src = dataset_path / subdir
            dst = output_path / subdir
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".git"))
        videos_src = dataset_path / "videos"
        videos_dst = output_path / "videos"
        if videos_src.exists() and not videos_dst.exists():
            try:
                os.symlink(videos_src, videos_dst)
                messages.append(f"Symlinked videos/ → {videos_src}")
            except OSError:
                messages.append(f"Note: videos/ not copied. Original at: {videos_src}")

    # Gather all edits from all episodes
    all_episodes = progress.get_done_episodes()
    
# --- 1. Build new task table ---
    tasks_df = load_tasks(output_path)
    max_task_idx = int(tasks_df["task_index"].max()) if len(tasks_df) > 0 else -1

    edit_task_map: dict[str, int] = {}
    for _, row in tasks_df.iterrows():
        edit_task_map[str(row["task"])] = int(row["task_index"])

    new_tasks = []
    # Find any entirely new tasks introduced during editing
    for ep_idx in all_episodes:
        for edit in progress.get_edits_for_episode(ep_idx):
            if edit.task not in edit_task_map:
                max_task_idx += 1
                edit_task_map[edit.task] = max_task_idx
                new_tasks.append({"task_index": max_task_idx, "task": edit.task})

    if new_tasks:
        new_df = pd.DataFrame(new_tasks)
        tasks_df = pd.concat([tasks_df, new_df], ignore_index=True)

    out_tasks = tasks_df.set_index("task")[["task_index"]]
    
    # FIX: Strip the index name so PyArrow falls back to __index_level_0__
    out_tasks.index.name = None 
    
    out_tasks.to_parquet(output_path / "meta" / "tasks.parquet")
    messages.append(f"Updated tasks.parquet ({len(tasks_df)} total tasks)")


    # --- 2. Update per-frame task_index (Preserving PyArrow Schema!) ---
    # We loop through all data files and apply any edits that belong to them
    for f in sorted((output_path / "data").rglob("*.parquet")):
        # Read with PyArrow to capture the exact schema
        table = pq.read_table(f)
        original_schema = table.schema
        df = table.to_pandas()
        
        modified = False
        
        # Apply edits for all edited episodes found in this file
        for ep_idx in all_episodes:
            ep_mask = df["episode_index"] == ep_idx
            if not ep_mask.any():
                continue
            
            for edit in progress.get_edits_for_episode(ep_idx):
                tidx = edit_task_map[edit.task]
                frame_mask = ep_mask & df["frame_index"].between(edit.start, edit.end)
                if frame_mask.any():
                    df.loc[frame_mask, "task_index"] = tidx
                    modified = True

        if modified:
            # Write back using PyArrow and the ORIGINAL schema
            try:
                out_table = pa.Table.from_pandas(df, schema=original_schema, preserve_index=False)
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                out_table = pa.Table.from_pandas(df, preserve_index=False)
            
            pq.write_table(out_table, f)
            messages.append(f"Updated data file {f.name}")


    # --- 3. Update episode metadata ---
    # To be perfectly accurate, we re-scan the dataset to see what tasks belong to which episode now
    used_tasks_per_ep = {}
    for f in sorted((output_path / "data").rglob("*.parquet")):
        # Only read the two columns we care about to save time
        df_minimal = pq.read_table(f, columns=["episode_index", "task_index"]).to_pandas()
        for ep_idx, group in df_minimal.groupby("episode_index"):
            if ep_idx not in used_tasks_per_ep:
                used_tasks_per_ep[ep_idx] = set()
            used_tasks_per_ep[ep_idx].update(group["task_index"].unique())

    # Write the new task lists into the episode metadata
    for f in sorted((output_path / "meta" / "episodes").rglob("*.parquet")):
        original_table = pq.read_table(f)
        original_schema = original_table.schema
        eps = original_table.to_pandas()

        modified = False
        if "tasks" in eps.columns:
            for row_idx in eps.index:
                ep_idx = eps.at[row_idx, "episode_index"]
                if ep_idx in used_tasks_per_ep:
                    sample_val = eps.at[row_idx, "tasks"]
                    
                    is_string_type = False
                    if hasattr(sample_val, '__iter__') and len(sample_val) > 0:
                        first_elem = list(sample_val)[0]
                        is_string_type = isinstance(first_elem, str)

                    # Map indices back to strings if the dataset uses string labels
                    if is_string_type:
                        idx_to_str = {tidx: task_str for task_str, tidx in edit_task_map.items()}
                        new_val = sorted(idx_to_str.get(idx, str(idx)) for idx in used_tasks_per_ep[ep_idx])
                    else:
                        new_val = sorted(int(x) for x in used_tasks_per_ep[ep_idx])

                    eps.at[row_idx, "tasks"] = (
                        type(sample_val)(new_val)
                        if not isinstance(sample_val, np.ndarray)
                        else np.array(new_val, dtype=sample_val.dtype)
                    )
                    modified = True

        if modified:
            try:
                table = pa.Table.from_pandas(eps, schema=original_schema, preserve_index=False)
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                table = pa.Table.from_pandas(eps, preserve_index=False)
            pq.write_table(table, f)
            messages.append(f"Updated metadata file {f.name}")

    dest = "in place" if overwrite else str(output_path)
    messages.append(f"\nDone! Saved {dest}")
    return "\n".join(messages)