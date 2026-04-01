import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

def verify_meta_exports(old_dataset_path, new_dataset_path):
    old_meta = Path(old_dataset_path) / "meta"
    new_meta = Path(new_dataset_path) / "meta"

    # 1. Compare tasks.parquet
    print("=== COMPARING: tasks.parquet ===")
    old_tasks_file = old_meta / "tasks.parquet"
    new_tasks_file = new_meta / "tasks.parquet"
    
    old_tasks_schema = pq.read_table(old_tasks_file).schema
    new_tasks_schema = pq.read_table(new_tasks_file).schema
    
    if old_tasks_schema == new_tasks_schema:
        print("✅ Schemas match perfectly! (__index_level_0__ is correct)")
    else:
        print("❌ Schemas DO NOT match.")
        print(f"  Old columns: {[f.name for f in old_tasks_schema]}")
        print(f"  New columns: {[f.name for f in new_tasks_schema]}")

    old_tasks_df = pd.read_parquet(old_tasks_file)
    new_tasks_df = pd.read_parquet(new_tasks_file)
    print(f"📊 Task count: Original={len(old_tasks_df)}, Exported={len(new_tasks_df)}")

    # 2. Compare episodes/*.parquet
    print("\n=== COMPARING: episodes/*.parquet ===")
    old_ep_files = sorted((old_meta / "episodes").rglob("*.parquet"))
    new_ep_files = sorted((new_meta / "episodes").rglob("*.parquet"))
    
    if not old_ep_files or not new_ep_files:
        print("Missing episode files to compare.")
        return

    # Compare the first file's schema as representative
    old_ep_schema = pq.read_table(old_ep_files[0]).schema
    new_ep_schema = pq.read_table(new_ep_files[0]).schema

    if old_ep_schema == new_ep_schema:
        print("✅ Schemas match perfectly! (Lists and types are identical)")
    else:
        print("❌ Schemas DO NOT match.")
    
    # Check the actual task list changes
    old_eps_df = pd.read_parquet(old_ep_files[0])
    new_eps_df = pd.read_parquet(new_ep_files[0])
    
    diffs = 0
    print("\n🔍 Snapshot of modified episodes:")
    for idx in range(min(len(old_eps_df), len(new_eps_df))):
        ep_index = old_eps_df.iloc[idx]['episode_index']
        old_tasks = list(old_eps_df.iloc[idx]['tasks'])
        new_tasks = list(new_eps_df.iloc[idx]['tasks'])
        
        if sorted(old_tasks) != sorted(new_tasks):
            diffs += 1
            if diffs <= 3:  # Just print the first 3 so it doesn't spam your console
                print(f"  Episode {ep_index}:")
                print(f"    Old: {old_tasks}")
                print(f"    New: {new_tasks}\n")
    
    if diffs == 0:
        print("⚠️ No task changes found in the episodes metadata.")
    else:
        print(f"✅ Found {diffs} total episodes with updated sub-tasks.")

# --- Paths (Replace these if necessary) ---
old_path = r"leisaac-pick-orange-mimic-v0"
new_path = r"pick-orange-mimic-subtasked"

verify_meta_exports(old_path, new_path)