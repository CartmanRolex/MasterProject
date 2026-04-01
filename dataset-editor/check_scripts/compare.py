import pandas as pd
import numpy as np

def get_modified_episodes(file1_path, file2_path):
    # 1. Load data
    df1 = pd.read_parquet(file1_path)
    df2 = pd.read_parquet(file2_path)

    # 2. Define the keys that identify a specific moment
    keys = ['episode_index', 'frame_index']
    
    # 3. Merge files side-by-side
    merged = pd.merge(df1, df2, on=keys, suffixes=('_old', '_new'))

    # 4. Identify columns to compare (ignoring the keys)
    other_cols = [c for c in df1.columns if c not in keys]
    
    # We use a set to automatically prevent duplicate episode numbers
    modified_episodes = set()

    for col in other_cols:
        col_old = f"{col}_old"
        col_new = f"{col}_new"
        
        # Check for differences
        is_diff = merged.apply(lambda row: str(row[col_old]) != str(row[col_new]), axis=1)
        
        if is_diff.any():
            # Grab the episode numbers where this column changed and add them to the set
            diff_eps = merged[is_diff]['episode_index'].unique()
            modified_episodes.update(diff_eps)

    # 5. Print just the list of modified episodes
    if not modified_episodes:
        print("✅ No episodes were modified.")
    else:
        # Sort the episodes so they are easy to read
        sorted_episodes = sorted(list(modified_episodes))
        print(f"Found {len(sorted_episodes)} modified episodes. Here is the list:")
        print(sorted_episodes)

# --- File Paths ---
file_A = r"leisaac-pick-orange-mimic-v0\data\chunk-000\file-000.parquet"
file_B = r"pick-orange-mimic-subtasked\data\chunk-000\file-000.parquet"

get_modified_episodes(file_A, file_B)