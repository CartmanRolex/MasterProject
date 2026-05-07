"""
Per-subtask dataset recorder for autonomous inference.

Terminology
-----------
- inference run   : one full robot session (env reset → pick all oranges → done).
                    Controlled by n_inference_runs in inference_autonomous_orders.py.
- subtask recording: one successful subtask saved to the LeRobot dataset
                    (one GRASP, one LIFT, one PLACE, or one HOME movement).
                    Multiple subtask recordings are produced per inference run.
                    This is what LeRobot calls an "episode".

Crash-safety
------------
After every commit():
  - The data parquet writer is closed (writes a valid footer) and rotated to a
    new file, so previously committed subtask recordings are safe even after a
    hard crash (SIGKILL, kernel freeze).
  - A checkpoint.json is written atomically next to the dataset, recording the
    total number of committed subtask recordings. This survives corrupted parquets.
  - The metadata (episodes) parquet writer stays open between commits so it does
    not overwrite previous episodes; it is properly closed via close_writers(),
    which is always called in the finally block of inference_autonomous_orders.py.

Resume support
--------------
Pass resume=True to SubtaskRecorder.create(). If the dataset is corrupted,
a RuntimeError is raised with a clear message — no data is auto-deleted.
Set RECORD_OVERWRITE=True to explicitly wipe and start fresh.
"""

import datetime
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
    """
    Wraps a LeRobotDataset and manages a per-subtask frame buffer.

    Each successful subtask (GRASP / LIFT / PLACE / HOME) produces one
    subtask recording (= one LeRobot episode). Failed or timed-out
    attempts are discarded without writing anything to disk.

    Usage per step:
        recorder.record(frame_dict)           # buffer this frame
        recorder.commit("Grasp left orange")  # save as a subtask recording
        recorder.discard()                    # drop buffer on failure
        recorder.start()                      # arm for the next subtask
        recorder.close_writers()              # flush to disk (call in finally)
        recorder.push_to_hub()               # upload to HuggingFace
    """

    def __init__(self, dataset: LeRobotDataset):
        self._dataset = dataset
        self._buffer: list[dict] = []
        self._active = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        repo_id: str,
        local_path: str,
        fps: int = 30,
        resume: bool = False,
        overwrite: bool = False,
    ) -> "SubtaskRecorder":
        """
        Open or create a LeRobot dataset for recording subtask episodes.

        Args:
            resume:    If True and a dataset exists at local_path, append to it.
                       If the dataset is corrupted, raises RuntimeError (no auto-delete).
            overwrite: If True, delete any existing dataset and start fresh.
                       Takes precedence over resume.
        """
        local_path = Path(local_path)
        info_file = local_path / "meta" / "info.json"
        checkpoint_file = local_path / "checkpoint.json"

        if overwrite and local_path.exists():
            print(f"  🗑  RECORD_OVERWRITE=True — deleting existing dataset at {local_path}")
            shutil.rmtree(local_path)

        elif resume and info_file.exists():
            # Read checkpoint first — it's reliable even if parquet is corrupted
            saved_count = 0
            if checkpoint_file.exists():
                with open(checkpoint_file) as f:
                    saved_count = json.load(f).get("total_subtask_recordings", 0)

            try:
                dataset = LeRobotDataset(
                    repo_id=repo_id,
                    root=local_path,
                    vcodec="libsvtav1",
                )
                # Flush metadata immediately after every episode (not batched)
                # so the open meta writer is always up to date on disk.
                dataset.meta.metadata_buffer_size = 1
                n = dataset.meta.total_episodes
                print(f"  ▶ Resuming: {n} subtask recordings already saved — appending to {local_path}")
                return cls(dataset)

            except Exception as e:
                raise RuntimeError(
                    f"\n⚠  Dataset at {local_path} is corrupted and cannot be resumed.\n"
                    f"   Checkpoint shows {saved_count} subtask recording(s) were committed.\n"
                    f"   Options:\n"
                    f"     - Set RECORD_OVERWRITE=True to wipe and start fresh (deletes data)\n"
                    f"     - Manually delete {local_path} and restart\n"
                    f"   Original error: {type(e).__name__}: {e}"
                ) from e

        # Create a brand-new dataset
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=DATASET_FEATURES,
            root=local_path,
            robot_type="so101_follower",
            use_videos=True,
            vcodec="libsvtav1",
            metadata_buffer_size=1,  # flush meta immediately after every subtask recording
        )
        return cls(dataset)

    # ------------------------------------------------------------------
    # Per-step recording
    # ------------------------------------------------------------------

    def start(self):
        """Arm the recorder for a new subtask. Any previous unfinished buffer is dropped."""
        self._buffer = []
        self._active = True

    def record(self, frame: dict):
        """Buffer one frame. No-op if not armed."""
        if self._active:
            self._buffer.append({
                k: v.astype(np.float32) if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating) and v.dtype != np.float32 else v
                for k, v in frame.items()
            })

    def discard(self):
        """Drop the current buffer without saving anything."""
        self._buffer = []
        self._active = False

    # ------------------------------------------------------------------
    # Committing a subtask recording
    # ------------------------------------------------------------------

    def commit(self, task: str):
        """
        Flush the buffer as a new subtask recording (LeRobot episode).

        After this returns, the subtask recording is fully safe on disk:
          - data parquet  : writer closed (valid footer) and rotated to a new file
          - metadata      : flushed to the open meta writer (closed on close_writers())
          - checkpoint.json: written atomically with the updated count
        """
        if not self._active or not self._buffer:
            self._active = False
            self._buffer = []
            return

        # Append freeze frames (hold last pose) to pad the end of the subtask
        last_frame = self._buffer[-1]
        if FREEZE_FRAMES > 0:
            last_state  = last_frame.get("observation.state")
            last_action = last_frame.get("action")
            for _ in range(FREEZE_FRAMES):
                freeze = dict(last_frame)
                if last_state  is not None:
                    freeze["observation.state"] = np.asarray(last_state,  dtype=np.float32).copy()
                if last_action is not None:
                    freeze["action"]            = np.asarray(last_action, dtype=np.float32).copy()
                self._buffer.append(freeze)

        for frame in self._buffer:
            self._dataset.add_frame({**frame, "task": task})
        self._dataset.save_episode()

        # ── Data parquet crash-safety ──────────────────────────────────
        # Close the writer so the current data file gets a valid footer
        # immediately. Without this, a hard crash would leave the footer
        # missing and make ALL data in the file unreadable.
        # _writer_closed_for_reading tells LeRobot to start a new file on
        # the next commit instead of reopening (and overwriting) this one.
        self._dataset._close_writer()
        self._dataset._writer_closed_for_reading = True

        # ── Checkpoint ────────────────────────────────────────────────
        # Write an atomic JSON record of the total committed count.
        # This survives corrupted parquets and lets us report what was
        # saved even when LeRobotDataset itself cannot be reopened.
        self._write_checkpoint()

        n = len(self._buffer)
        self._buffer = []
        self._active = False
        total = self._dataset.meta.total_episodes
        print(f"  📼 Subtask recording saved: \"{task}\" ({n} frames) — total: {total}")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close_writers(self):
        """
        Close all parquet writers so every file gets a valid footer.

        Call this in the finally block of the main script so that both
        data and metadata parquets are readable even after Ctrl+C or a
        crash that reaches the finally block (i.e. any non-SIGKILL exit).
        """
        self.discard()
        self._dataset._close_writer()
        self._dataset.meta._close_writer()

    def push_to_hub(self):
        """Push the completed dataset to HuggingFace Hub."""
        if hasattr(self._dataset, "consolidate"):
            self._dataset.consolidate()
        elif hasattr(self._dataset, "finalize"):
            self._dataset.finalize()
        self._dataset.push_to_hub()
        print(f"  📤 Dataset pushed to Hub: {self._dataset.repo_id}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_checkpoint(self):
        """Atomically write checkpoint.json with the current subtask recording count."""
        root = Path(self._dataset.root)
        data = {
            "total_subtask_recordings": self._dataset.meta.total_episodes,
            "last_commit": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        tmp = root / "checkpoint.json.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        tmp.rename(root / "checkpoint.json")
