"""
Per-subtask dataset recorder for autonomous inference.

Buffers frames during each subtask phase and flushes them as a single
LeRobot episode when the subtask is confirmed. Failed or timed-out
attempts are discarded without writing anything to disk.
"""

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
    def create(cls, repo_id: str, local_path: str, fps: int = 30) -> "SubtaskRecorder":
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

        for frame in self._buffer:
            self._dataset.add_frame({**frame, "task": task})
        self._dataset.save_episode()

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
