"""Export edited dataset to disk."""

import os
import shutil
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pandas import col
import pyarrow as pa
import pyarrow.parquet as pq
from typer import edit

from .data_loader import load_tasks
from .state import ProgressTracker

"""
Split function for LeRobot V3 datasets.

Add this function to exporter.py (add `import json` to imports too).
Then add the GUI button + handler snippets to gui.py.
"""

# === ADD TO exporter.py IMPORTS ===
# import json

# === ADD THIS FUNCTION TO exporter.py ===

def split_all_episodes(
    dataset_path: Path,
    progress: ProgressTracker,
    output_path: Path,
) -> str:
    """Split ALL episodes that have edits into sub-episodes.

    Writes to output_path (full copy first). Never modifies dataset_path.

    For each episode with edits, each edit region becomes a new episode.
    Frames outside edit regions are DROPPED.
    Episodes without edits are EXCLUDED from the output.

    Updates: data parquet, episode metadata (all columns including video
    timestamps and per-episode stats), tasks.parquet, and info.json.
    """
    import json

    all_done = progress.get_done_episodes()
    episodes_with_edits = [ep for ep in all_done if progress.get_edits_for_episode(ep)]

    if not episodes_with_edits:
        return "No episodes with edits to split."

    if output_path.resolve() == dataset_path.resolve():
        return "Output path must be different from source to avoid data loss."

    messages = []

    # --- 0. Copy entire dataset to output_path ---
    if output_path.exists():
        shutil.rmtree(output_path)
    shutil.copytree(dataset_path, output_path)
    messages.append(f"Copied dataset to {output_path}")

    # --- 1. Read data + episode metadata from the COPY ---
    data_file = sorted((output_path / "data").rglob("*.parquet"))[0]
    data_table = pq.read_table(data_file)
    data_schema = data_table.schema
    df = data_table.to_pandas()

    ep_meta_file = sorted((output_path / "meta" / "episodes").rglob("*.parquet"))[0]
    ep_meta_table = pq.read_table(ep_meta_file)
    ep_meta_schema = ep_meta_table.schema
    ep_meta = ep_meta_table.to_pandas()

    # Load info.json
    info_path = output_path / "meta" / "info.json"
    with open(info_path) as fj:
        info = json.load(fj)
    fps = info.get("fps", 30)

    # --- 2. Update tasks.parquet ---
    tasks_df = load_tasks(output_path)
    task_map = {str(row["task"]): int(row["task_index"]) for _, row in tasks_df.iterrows()}
    max_task_idx = int(tasks_df["task_index"].max()) if len(tasks_df) > 0 else -1
    new_tasks = []
    for ep_idx in episodes_with_edits:
        for edit in progress.get_edits_for_episode(ep_idx):
            if edit.task not in task_map:
                max_task_idx += 1
                task_map[edit.task] = max_task_idx
                new_tasks.append({"task_index": max_task_idx, "task": edit.task})
    if new_tasks:
        tasks_df = pd.concat([tasks_df, pd.DataFrame(new_tasks)], ignore_index=True)
        out_tasks = tasks_df.set_index("task")[["task_index"]]
        out_tasks.index.name = None
        out_tasks.to_parquet(output_path / "meta" / "tasks.parquet")
        messages.append(f"Added {len(new_tasks)} new task(s)")

    # --- 3. Build the new data rows ---
    eps_to_split = set(episodes_with_edits)
    kept_rows = df.iloc[0:0].copy()  # unannotated episodes are excluded

    new_data_chunks = []         # DataFrames for new sub-episodes
    new_meta_rows = []           # dicts for new episode metadata rows
    split_from_eps = set()       # original episode indices that got split

    # We'll assign temporary episode indices > max, then remap to contiguous
    next_ep_idx = int(df["episode_index"].max()) + 1

    for ep_idx in sorted(episodes_with_edits):
        ep_rows = df[df["episode_index"] == ep_idx]
        if ep_rows.empty:
            continue

        # Get the original episode metadata row
        orig_meta_row = ep_meta[ep_meta["episode_index"] == ep_idx]
        if orig_meta_row.empty:
            messages.append(f"Warning: no metadata for episode {ep_idx}, skipping")
            continue

        orig_meta = orig_meta_row.iloc[0]
        sorted_edits = sorted(progress.get_edits_for_episode(ep_idx), key=lambda e: e.start)
        split_from_eps.add(ep_idx)

        for edit in sorted_edits:
            chunk = ep_rows[
                ep_rows["frame_index"].between(edit.start, edit.end)
            ].copy()
            if chunk.empty:
                continue

            old_min_frame = int(chunk["frame_index"].min())
            old_min_ts = float(chunk["timestamp"].min())
            # Re-index the data
            chunk["frame_index"] = chunk["frame_index"] - old_min_frame
            chunk["timestamp"] = chunk["timestamp"] - old_min_ts
            chunk["episode_index"] = next_ep_idx
            chunk["task_index"] = task_map[edit.task]

            n_frames = len(chunk)

            new_data_chunks.append(chunk)

            # Build metadata row for this sub-episode
            meta_row = {}
            meta_row["episode_index"] = next_ep_idx
            meta_row["length"] = n_frames
            meta_row["tasks"] = np.array([edit.task], dtype=object)

            # Data file references (same file since we write everything to one)
            meta_row["data/chunk_index"] = 0
            meta_row["data/file_index"] = 0
            # dataset_from_index / dataset_to_index: will be set after final sort
            meta_row["dataset_from_index"] = -1  # placeholder
            meta_row["dataset_to_index"] = -1    # placeholder

            # Video references: same file as original, but adjusted timestamps
            for col_name in ep_meta.columns:
                if col_name.endswith("/chunk_index") and col_name.startswith("videos/"):
                    meta_row[col_name] = int(orig_meta[col_name])
                elif col_name.endswith("/file_index") and col_name.startswith("videos/"):
                    meta_row[col_name] = int(orig_meta[col_name])
                elif col_name.endswith("/from_timestamp") and col_name.startswith("videos/"):
                    orig_from = float(orig_meta[col_name])
                    meta_row[col_name] = orig_from + (edit.start / fps)
                elif col_name.endswith("/to_timestamp") and col_name.startswith("videos/"):
                    orig_from_col = col_name.replace("/to_timestamp", "/from_timestamp")
                    orig_from = float(orig_meta[orig_from_col])
                    meta_row[col_name] = orig_from + ((edit.end + 1) / fps)  # no tail extension: frozen frames seek within real footage

            # Episode metadata self-reference
            meta_row["meta/episodes/chunk_index"] = 0
            meta_row["meta/episodes/file_index"] = 0

            # Stats will be computed after we finalize the data
            # Store the chunk reference so we can compute stats later
            meta_row["_chunk_ref"] = chunk  # temporary, removed before writing

            new_meta_rows.append(meta_row)
            next_ep_idx += 1

    # --- 4. Assemble final data, remap episode indices to be contiguous ---
    result = pd.concat([kept_rows] + new_data_chunks, ignore_index=True)

    old_eps = sorted(result["episode_index"].unique())
    remap = {old: new for new, old in enumerate(old_eps)}
    result["episode_index"] = result["episode_index"].map(remap)
    result = result.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    result["index"] = range(len(result))

    # Update episode_index in metadata rows
    for row in new_meta_rows:
        row["episode_index"] = remap[row["episode_index"]]

    # Now compute dataset_from_index / dataset_to_index from the final data
    for row in new_meta_rows:
        ep_data = result[result["episode_index"] == row["episode_index"]]
        row["dataset_from_index"] = int(ep_data["index"].min())
        row["dataset_to_index"] = int(ep_data["index"].max()) + 1

    # --- 5. Write data parquet ---
    try:
        out_table = pa.Table.from_pandas(result, schema=data_schema, preserve_index=False)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        out_table = pa.Table.from_pandas(result, preserve_index=False)
    pq.write_table(out_table, data_file)
    messages.append(f"Wrote {len(old_eps)} episodes to {data_file.name}")

    # --- 6. Build complete episode metadata with stats ---
    # Unannotated episodes are excluded; only annotated sub-episodes are kept
    kept_meta = ep_meta.iloc[0:0].copy()

    # Compute per-episode stats for new sub-episodes
    stat_columns = [c for c in ep_meta.columns if c.startswith("stats/")]
    idx_to_str = {ti: tstr for tstr, ti in task_map.items()}

    final_new_meta = []
    for row in new_meta_rows:
        chunk_data = row.pop("_chunk_ref")  # remove temp reference
        # Recompute stats from the chunk data (with the re-indexed values)
        ep_i = row["episode_index"]
        ep_final = result[result["episode_index"] == ep_i]

        for stat_col in stat_columns:
            # Parse stat_col: "stats/{feature}/{stat_type}"
            parts = stat_col.split("/")
            # parts = ["stats", feature_name, stat_type]
            # But feature_name can contain '/' like "observation.images.front"
            stat_type = parts[-1]  # min, max, mean, std, count
            feature_name = "/".join(parts[1:-1])

            if feature_name not in ep_final.columns:
                # Copy from original metadata as fallback (e.g. image stats)
                orig_ep_meta = ep_meta[ep_meta["episode_index"].isin(split_from_eps)]
                if not orig_ep_meta.empty:
                    row[stat_col] = orig_ep_meta.iloc[0][stat_col]
                continue

            col_data = ep_final[feature_name]

            # Check if the column contains arrays/lists (like action, observation.state)
            first_val = col_data.iloc[0]
            is_array = isinstance(first_val, (list, np.ndarray))

            if is_array:
                stacked = np.stack(col_data.values)
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
            else:
                vals = col_data.values.astype(float)
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

        final_new_meta.append(row)

    # Combine kept + new metadata, sort by episode_index
    new_meta_df = pd.DataFrame(final_new_meta)
    all_meta = pd.concat([kept_meta, new_meta_df], ignore_index=True)
    all_meta = all_meta.sort_values("episode_index").reset_index(drop=True)

    # --- 7. Write episode metadata using the original Arrow schema ---
    # Build column-by-column to handle complex types (lists, nested lists)
    arrow_columns = {}
    for col_name in ep_meta_schema.names:
        field_type = ep_meta_schema.field(col_name).type

        if col_name == "tasks":
            # list<string> — force to plain Python lists
            clean = []
            for val in all_meta[col_name]:
                if isinstance(val, np.ndarray):
                    clean.append(val.tolist())
                elif isinstance(val, (list, tuple)):
                    clean.append(list(val))
                else:
                    clean.append([str(val)])
            arrow_columns[col_name] = pa.array(clean, type=field_type)

        elif col_name.startswith("stats/"):
            # Stats columns have various nested list types
            clean = []
            for val in all_meta[col_name]:
                if isinstance(val, np.ndarray):
                    clean.append(val.tolist())
                elif isinstance(val, (list, tuple)):
                    clean.append(val)
                else:
                    clean.append(val)
            try:
                arrow_columns[col_name] = pa.array(clean, type=field_type)
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                # If type casting fails, let PyArrow infer
                arrow_columns[col_name] = pa.array(clean)

        else:
            # Simple scalar columns (int64, float64)
            try:
                arrow_columns[col_name] = pa.array(
                    all_meta[col_name].values, type=field_type
                )
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                arrow_columns[col_name] = pa.array(all_meta[col_name].values)

    out_meta_table = pa.table(arrow_columns)
    pq.write_table(out_meta_table, ep_meta_file)
    messages.append(f"Updated episode metadata: {len(all_meta)} episodes")

    # --- 8. Update info.json ---
    total_episodes = len(all_meta)
    total_frames = len(result)
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = len(task_map)
    info["splits"] = {"train": f"0:{total_episodes}"}

    with open(info_path, "w") as fj:
        json.dump(info, fj, indent=2)
    messages.append(f"Updated info.json: {total_episodes} episodes, {total_frames} frames")

    messages.append(f"\nDone! Split {len(episodes_with_edits)} episodes → {len(new_data_chunks)} sub-episodes")
    messages.append(f"Total episodes: {total_episodes}, Total frames: {total_frames}")
    messages.append(f"Output: {output_path}")
    return "\n".join(messages)

def split_episode(
    dataset_path: Path,
    episode_idx: int,
    edits: list,  # list[TaskEdit]
) -> str:
    """Split one episode into N new episodes (one per edit region).

    Each edit defines a frame range (in frame_index space) + task string.
    Frames outside any edit are dropped. Videos are NOT touched.
    """
    if not edits:
        return "No edits to split on."

    sorted_edits = sorted(edits, key=lambda e: e.start)
    messages = []

    # --- 1. Read all data, find max episode_index ---
    data_files = sorted((dataset_path / "data").rglob("*.parquet"))
    all_tables = {}
    max_ep = -1
    for f in data_files:
        table = pq.read_table(f)
        df = table.to_pandas()
        all_tables[f] = (table.schema, df)
        if len(df) > 0:
            max_ep = max(max_ep, int(df["episode_index"].max()))

    new_ep_indices = list(range(max_ep + 1, max_ep + 1 + len(sorted_edits)))

    # --- 2. Update tasks.parquet ---
    tasks_df = load_tasks(dataset_path)
    task_map = {str(row["task"]): int(row["task_index"]) for _, row in tasks_df.iterrows()}
    max_task_idx = int(tasks_df["task_index"].max()) if len(tasks_df) > 0 else -1
    new_tasks = []
    for edit in sorted_edits:
        if edit.task not in task_map:
            max_task_idx += 1
            task_map[edit.task] = max_task_idx
            new_tasks.append({"task_index": max_task_idx, "task": edit.task})
    if new_tasks:
        tasks_df = pd.concat([tasks_df, pd.DataFrame(new_tasks)], ignore_index=True)
        out_tasks = tasks_df.set_index("task")[["task_index"]]
        out_tasks.index.name = None
        out_tasks.to_parquet(dataset_path / "meta" / "tasks.parquet")
        messages.append(f"Added {len(new_tasks)} new task(s)")

    # --- 3. Split data parquet rows ---
    for f, (original_schema, df) in all_tables.items():
        ep_mask = df["episode_index"] == episode_idx
        if not ep_mask.any():
            continue

        ep_rows = df[ep_mask]
        other_rows = df[~ep_mask]

        new_chunks = []
        for edit, new_ep_idx in zip(sorted_edits, new_ep_indices):
            chunk = ep_rows[
                ep_rows["frame_index"].between(edit.start, edit.end)
            ].copy()
            if chunk.empty:
                continue

            old_min_frame = int(chunk["frame_index"].min())
            old_min_ts = float(chunk["timestamp"].min())

            chunk["frame_index"] = chunk["frame_index"] - old_min_frame
            chunk["timestamp"] = chunk["timestamp"] - old_min_ts
            chunk["episode_index"] = new_ep_idx
            chunk["task_index"] = task_map[edit.task]
            new_chunks.append(chunk)

        if new_chunks:
            result = pd.concat([other_rows] + new_chunks, ignore_index=True)
            result = result.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
            # Recompute global 'index' column
            result["index"] = range(len(result))
            try:
                out_table = pa.Table.from_pandas(result, schema=original_schema, preserve_index=False)
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                out_table = pa.Table.from_pandas(result, preserve_index=False)
            pq.write_table(out_table, f)
            messages.append(f"Split data in {f.name}: ep {episode_idx} → {len(new_chunks)} new episodes")

    # --- 4. Update episode metadata ---
    for f in sorted((dataset_path / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(f)
        original_schema = table.schema
        eps = table.to_pandas()
        # eps has no episode_index column — row position IS the episode index

        if episode_idx >= len(eps):
            continue

        # Build new rows for the sub-episodes
        new_rows = []
        for edit, new_ep_idx in zip(sorted_edits, new_ep_indices):
            new_rows.append({"tasks": np.array([edit.task], dtype=object)})

        # Remove original episode row, append new ones
        eps = eps.drop(index=episode_idx).reset_index(drop=True)
        new_df = pd.DataFrame(new_rows)
        eps = pd.concat([eps, new_df], ignore_index=True)

        # Write back — manually build the tasks column as a PyArrow list array
        tasks_list = [list(row) for row in eps["tasks"]]
        tasks_array = pa.array(tasks_list, type=pa.list_(pa.string()))
        out_table = pa.table({"tasks": tasks_array})
        pq.write_table(out_table, f)
        messages.append(f"Updated episode metadata: removed ep {episode_idx}, added {len(new_rows)} new")

    # --- 5. Now we need to fix episode_index in data to be contiguous ---
    # After removing an episode and adding new ones at the end, the episode
    # indices in the data parquet may have gaps. Let's re-index everything.
    for f in data_files:
        table = pq.read_table(f)
        original_schema = table.schema
        df = table.to_pandas()

        # Build mapping: old episode_index → new contiguous index
        old_eps = sorted(df["episode_index"].unique())
        remap = {old: new for new, old in enumerate(old_eps)}
        df["episode_index"] = df["episode_index"].map(remap)
        df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        df["index"] = range(len(df))

        try:
            out_table = pa.Table.from_pandas(df, schema=original_schema, preserve_index=False)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            out_table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(out_table, f)

    messages.append(f"\nSplit episode {episode_idx} → {len(sorted_edits)} new episodes")
    return "\n".join(messages)

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