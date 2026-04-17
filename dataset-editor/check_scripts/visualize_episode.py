"""
Visualize a single episode exactly as LeRobot training loads it — no PyTorch needed.

Uses the same backend (PyAV) and the same algorithm as LeRobot's training loader:
  lerobot_dataset.py  _query_videos()              → abs_ts = from_timestamp + frame["timestamp"]
  video_utils.py      decode_video_frames_torchvision() (pyav path):
    1. seek to nearest keyframe before abs_ts
    2. decode all frames forward until pts >= abs_ts
    3. return the frame whose pts is closest to abs_ts  (L1 argmin)

Usage:
    python visualize_episode.py --dataset PATH/TO/DATASET --episode 0 [--fps 10]
"""

import argparse
import json
from pathlib import Path

import av
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_frames(dataset_root: Path, episode_idx: int) -> pd.DataFrame:
    data_files = sorted(dataset_root.glob("data/**/*.parquet"))
    frames = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    return frames[frames["episode_index"] == episode_idx].sort_values("frame_index").reset_index(drop=True)


def load_episode_meta(dataset_root: Path, episode_idx: int) -> pd.Series:
    meta_files = sorted(dataset_root.glob("meta/episodes/**/*.parquet"))
    meta = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)
    row = meta[meta["episode_index"] == episode_idx]
    if row.empty:
        raise ValueError(f"Episode {episode_idx} not found in metadata.")
    return row.iloc[0]


def get_video_path(dataset_root: Path, ep_meta: pd.Series, cam_key: str) -> Path:
    chunk = int(ep_meta[f"videos/{cam_key}/chunk_index"])
    file_ = int(ep_meta[f"videos/{cam_key}/file_index"])
    return dataset_root / "videos" / cam_key / f"chunk-{chunk:03d}" / f"file-{file_:03d}.mp4"


def camera_keys_from_meta(ep_meta: pd.Series):
    keys = set()
    for col in ep_meta.index:
        if col.startswith("videos/") and col.endswith("/from_timestamp"):
            keys.add(col.removeprefix("videos/").removesuffix("/from_timestamp"))
    return sorted(keys)


# ---------------------------------------------------------------------------
# Frame decoding — mirrors decode_video_frames_torchvision() (pyav path)
# video_utils.py lines 157-250
# ---------------------------------------------------------------------------

_containers: dict[str, av.container.InputContainer] = {}


def seek_frame_lerobot(video_path: Path, abs_ts: float) -> np.ndarray | None:
    """
    Identical algorithm to LeRobot training (pyav backend):

    Step 1  reader.seek(abs_ts, keyframes_only=True)
            → container.seek to the nearest keyframe at or before abs_ts

    Step 2  for frame in reader: ...
            → decode all frames forward until pts >= abs_ts

    Step 3  dist = torch.cdist(query_ts, loaded_ts, p=1); argmin
            → pick frame with smallest |pts - abs_ts|
    """
    key = str(video_path)
    if key not in _containers:
        _containers[key] = av.open(key)
    container = _containers[key]
    stream = container.streams.video[0]

    # Step 1: keyframe seek  (keyframes_only=True → any_frame=False in av)
    container.seek(int(abs_ts / float(av.time_base)), backward=True, any_frame=False)

    # Step 2: decode forward
    loaded_frames: list[np.ndarray] = []
    loaded_ts: list[float] = []
    for frame in container.decode(stream):
        t = float(frame.pts * stream.time_base)
        loaded_frames.append(frame.to_ndarray(format="rgb24"))
        loaded_ts.append(t)
        if t >= abs_ts:
            break

    if not loaded_frames:
        return None

    # Step 3: closest frame by L1  (same as torch.cdist argmin)
    best = min(range(len(loaded_ts)), key=lambda i: abs(loaded_ts[i] - abs_ts))
    return loaded_frames[best]


def close_containers():
    for c in _containers.values():
        c.close()
    _containers.clear()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset root")
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS (default: dataset FPS)")
    args = parser.parse_args()

    root = Path(args.dataset)
    info_path = root / "meta" / "info.json"
    fps = args.fps or (json.loads(info_path.read_text())["fps"] if info_path.exists() else 30)

    frames   = load_frames(root, args.episode)
    ep_meta  = load_episode_meta(root, args.episode)
    cam_keys = camera_keys_from_meta(ep_meta)
    n_frames = len(frames)

    print(f"Episode {args.episode}: {n_frames} frames | cameras: {cam_keys} | playback fps: {fps}")

    # Build figure
    n_cams = len(cam_keys)
    fig, axes = plt.subplots(1, n_cams, figsize=(5 * n_cams, 5), squeeze=False)
    axes = axes[0]
    im_handles = []
    for ax, key in zip(axes, cam_keys):
        ax.set_title(key.split("/")[-1])
        ax.axis("off")
        im_handles.append(ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8)))

    info_text = fig.text(0.5, 0.01, "", ha="center", va="bottom", fontsize=8, family="monospace")
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.ion()
    plt.show()

    prev_ts = None
    for _, row in frames.iterrows():
        frame_ts  = float(row["timestamp"])
        frame_idx = int(row["frame_index"])
        frozen    = (prev_ts is not None and frame_ts == prev_ts)
        prev_ts   = frame_ts

        for handle, key in zip(im_handles, cam_keys):
            # ── Exact formula from lerobot_dataset.py _query_videos() ──────
            # shifted_query_ts = [from_timestamp + ts for ts in query_ts]
            from_ts = float(ep_meta[f"videos/{key}/from_timestamp"])
            abs_ts  = from_ts + frame_ts
            # ────────────────────────────────────────────────────────────────
            img = seek_frame_lerobot(root / get_video_path(root, ep_meta, key).relative_to(root), abs_ts)
            if img is not None:
                handle.set_data(img)

        action = np.array(row["action"]) if "action" in row else None
        state  = np.array(row["observation.state"]) if "observation.state" in row else None

        fig.suptitle(
            f"Episode {args.episode}  |  frame {frame_idx}  ({frames.index.get_loc(_) + 1}/{n_frames})"
            f"  |  t={frame_ts:.4f}s{'  *** FROZEN ***' if frozen else ''}",
            fontsize=10,
        )
        parts = []
        if action is not None:
            parts.append(f"action: {np.array2string(action, precision=3, suppress_small=True)}")
        if state is not None:
            parts.append(f"state:  {np.array2string(state, precision=3, suppress_small=True)}")
        info_text.set_text("\n".join(parts))

        fig.canvas.draw()
        plt.pause(1.0 / fps)

    close_containers()
    print("Done.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
