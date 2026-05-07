"""
Per-subtask dataset recorder for autonomous inference.

Buffers frames during each subtask phase and flushes them as a single
LeRobot episode when the subtask is confirmed. Failed or timed-out
attempts are discarded without writing anything to disk.

Resume support: pass resume=True to SubtaskRecorder.create() and it will
append to an existing partial dataset at local_path. If the existing
dataset is corrupted (e.g. from a hard crash), it falls back to a clean
start automatically.

Crash-safety: after every commit(), the parquet writer is explicitly
closed so the file has a valid footer on disk. A subsequent hard crash
(SIGKILL, kernel freeze) will not corrupt already-committed episodes.
"""

import json
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

DATASET_FEATURES = {
    "observation.images.front": {"dtype": "video", "shape": (480, 640, 3), "names": None},
    "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), "names": None},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "action":            {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
}

FREEZE_FRAMES = 20


class SubtaskRecorder:
    """Wraps a LeRobotDataset and manages a per-subtask frame buffer.

    Typical usage per step:
        recorder.record(frame_dict)          # always call

    On subtask confirmation:
        recorder.commit("Grasp left orange") # saves episode, clears buffer

    On failure / timeout / episode reset:
        recorder.discard()                   # drops buffer silently

    On new subtask phase starting:
        recorder.start()                     # arms the buffer

    At end of run:
        recorder.finalize()                  # encodes videos, pushes to Hub
    """

    def __init__(self, dataset: LeRobotDataset):
        self._dataset = dataset
        self._buffer: list[dict] = []
        self._active = False

    @classmethod
    def create(cls, repo_id: str, local_path: str, fps: int = 30, resume: bool = False) -> "SubtaskRecorder":
        local_path = Path(local_path)
        info_file = local_path / "meta" / "info.json"

        if resume and info_file.exists():
            try:
                with open(info_file) as f:
                    n_existing = json.load(f).get("total_episodes", 0)
                dataset = LeRobotDataset(
                    repo_id=repo_id,
                    root=local_path,
                    vcodec="libsvtav1",
                )
                print(f"  ▶ Resuming: {n_existing} episodes already saved — appending to {local_path}")
                return cls(dataset)
            except Exception as e:
                print(f"  ⚠  Could not load existing dataset ({type(e).__name__}: {e})")
                print("  ⚠  Starting fresh — deleting corrupted dataset.")
                shutil.rmtree(local_path, ignore_errors=True)

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=DATASET_FEATURES,
            root=local_path,
            robot_type="so101_follower",
            use_videos=True,
            vcodec="libsvtav1",
        )
        return cls(dataset)

    def start(self):
        """Arm the recorder for a new subtask. Any unfinished buffer is discarded."""
        self._buffer = []
        self._active = True

    def record(self, frame: dict):
        """Add one frame to the buffer. No-op if not armed."""
        if self._active:
            self._buffer.append({
                k: v.astype(np.float32) if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating) and v.dtype != np.float32 else v
                for k, v in frame.items()
            })

    def commit(self, task: str):
        """Flush the buffer as a new episode with the given task label."""
        if not self._active or not self._buffer:
            self._active = False
            self._buffer = []
            return

        last_frame = self._buffer[-1]
        if FREEZE_FRAMES > 0:
            last_state = last_frame.get("observation.state")
            last_action = last_frame.get("action")
            for _ in range(FREEZE_FRAMES):
                freeze_frame = dict(last_frame)
                if last_state is not None:
                    freeze_frame["observation.state"] = np.asarray(last_state, dtype=np.float32).copy()
                if last_action is not None:
                    freeze_frame["action"] = np.asarray(last_action, dtype=np.float32).copy()
                self._buffer.append(freeze_frame)

        for frame in self._buffer:
            self._dataset.add_frame({**frame, "task": task})
        self._dataset.save_episode()

        # Close the parquet writer immediately so the file gets a valid footer.
        # Without this, a hard crash (SIGKILL, kernel freeze) leaves an incomplete
        # parquet file whose footer is missing, making all data in it unreadable.
        # Setting _writer_closed_for_reading=True tells LeRobot to rotate to a new
        # file on the next commit instead of reopening (and overwriting) this one.
        self._dataset._close_writer()
        self._dataset._writer_closed_for_reading = True

        n = len(self._buffer)
        self._buffer = []
        self._active = False
        print(f"  📼 Saved episode: \"{task}\" ({n} frames)")

    def discard(self):
        """Drop the current buffer without saving."""
        self._buffer = []
        self._active = False

    def finalize(self):
        """Close all writers, compute stats, and push to HuggingFace Hub."""
        self.discard()
        if hasattr(self._dataset, "consolidate"):
            self._dataset.consolidate()
        elif hasattr(self._dataset, "finalize"):
            self._dataset.finalize()
        self._dataset.push_to_hub()
        print(f"  📤 Dataset pushed to Hub: {self._dataset.repo_id}")
