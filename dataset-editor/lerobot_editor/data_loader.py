"""Dataset loading utilities for LeRobot V3 datasets."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def resolve_dataset_path(source: str) -> Path:
    """Return a local Path to the dataset, downloading from HF Hub if needed."""
    local = Path(source)
    if local.is_dir():
        return local.resolve()

    if "/" in source:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            sys.exit("huggingface_hub is required. Install with: pip install huggingface_hub")

        print(f"[INFO] Downloading dataset from HuggingFace: {source} ...")
        local_dir = snapshot_download(repo_id=source, repo_type="dataset")
        return Path(local_dir)

    sys.exit(f"[ERROR] '{source}' is not a valid local path or HuggingFace repo ID.")


def load_tasks(dataset_path: Path) -> pd.DataFrame:
    """Load meta/tasks.parquet → DataFrame with columns [task_index, task]."""
    tasks_file = dataset_path / "meta" / "tasks.parquet"
    if not tasks_file.exists():
        sys.exit(f"[ERROR] tasks.parquet not found at {tasks_file}")
    df = pd.read_parquet(tasks_file)

    if "task" not in df.columns and "task_index" in df.columns:
        df = df.reset_index()
        if "index" in df.columns and "task" not in df.columns:
            df = df.rename(columns={"index": "task"})
        for col in df.columns:
            if col != "task_index" and df[col].dtype == object:
                df = df.rename(columns={col: "task"})
                break

    if "task" not in df.columns or "task_index" not in df.columns:
        sys.exit(f"[ERROR] Cannot parse tasks.parquet. Columns: {df.columns.tolist()}")

    return df


def load_episodes_metadata(dataset_path: Path) -> pd.DataFrame:
    ep_dir = dataset_path / "meta" / "episodes"
    if not ep_dir.exists():
        sys.exit(f"[ERROR] Episodes metadata not found at {ep_dir}")
    frames = []
    for f in sorted(ep_dir.rglob("*.parquet")):
        frames.append(pd.read_parquet(f))
    return pd.concat(frames, ignore_index=True)


def load_episode_list(episodes_df: pd.DataFrame) -> list[int]:
    return sorted(episodes_df["episode_index"].unique().tolist())


def load_frame_data(dataset_path: Path, episode_idx: int) -> pd.DataFrame:
    frames = []
    needed_cols = ["frame_index", "episode_index", "task_index", "action"]
    for f in sorted((dataset_path / "data").rglob("*.parquet")):
        # Only read the columns we need
        df = pd.read_parquet(f, columns=needed_cols)
        ep_data = df[df["episode_index"] == episode_idx]
        if len(ep_data) > 0:
            frames.append(ep_data)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("frame_index").reset_index(drop=True)


def discover_cameras(dataset_path: Path) -> list[str]:
    videos_dir = dataset_path / "videos"
    if not videos_dir.exists():
        return []
    return [d.name for d in sorted(videos_dir.iterdir()) if d.is_dir()]


def find_video_for_episode(
    dataset_path: Path, camera: str, episode_idx: int, episodes_df: pd.DataFrame
) -> str | None:
    chunk_col = f"videos/{camera}/chunk_index"
    file_col = f"videos/{camera}/file_index"

    ep_row = episodes_df[episodes_df["episode_index"] == episode_idx]
    if len(ep_row) > 0 and chunk_col in ep_row.columns and file_col in ep_row.columns:
        chunk_idx = int(ep_row.iloc[0][chunk_col])
        file_idx = int(ep_row.iloc[0][file_col])
        video_path = (
            dataset_path / "videos" / camera
            / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
        )
        if video_path.exists():
            return str(video_path)

    video_dir = dataset_path / "videos" / camera
    all_mp4s = sorted(video_dir.rglob("*.mp4"))
    if all_mp4s and episode_idx < len(all_mp4s):
        return str(all_mp4s[episode_idx])
    return None


class VideoReader:
    """Wraps cv2.VideoCapture with frame seeking."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._current = 0

    def read_frame(self, idx: int) -> np.ndarray | None:
        idx = max(0, min(idx, self.total_frames - 1))
        if idx != self._current:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self._current = idx + 1
            return frame
        return None

    def close(self):
        self.cap.release()