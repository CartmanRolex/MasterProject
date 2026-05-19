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
import contextlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset


JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

DATASET_FEATURES = {
    "observation.images.front": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
    "observation.images.wrist": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "action":            {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
}

HF_ORG = "MasterProject2026"
SYNTHETIC_DATASETS_DIR = Path(__file__).parent / "synthetic_datasets"
INCREMENTAL_PUSH_EVERY = 25

CAMERAS = ["observation.images.front", "observation.images.wrist"]


def merge_staging_into(staging_path: Path, main_path: Path) -> int:
    """
    Append all episodes from staging into the main dataset at the file level.
    Used by FULL_SUCCESS_DATA_GENERATION mode to atomically add a complete
    successful episode to the main dataset only after all subtasks are recorded.

    Must be called after staging_recorder.close_writers() so all parquet
    footers are valid.  Returns the number of episodes merged.
    """
    with open(staging_path / "meta" / "info.json") as f:
        staging_info = json.load(f)
    with open(main_path / "meta" / "info.json") as f:
        main_info = json.load(f)

    n_staging   = staging_info["total_episodes"]
    chunks_size = main_info["chunks_size"]
    ep_offset   = main_info["total_episodes"]
    frame_offset = main_info["total_frames"]

    if n_staging == 0:
        return 0

    # --- Load staging meta/episodes table ---
    staging_ep_tables = []
    for f in sorted((staging_path / "meta" / "episodes").glob("chunk-*/*.parquet")):
        try:
            staging_ep_tables.append(pq.read_table(f))
        except Exception:
            pass
    staging_ep_meta = pa.concat_tables(staging_ep_tables)
    src_ep_ids   = staging_ep_meta.column("episode_index").to_pylist()
    src_row_of   = {ep: i for i, ep in enumerate(src_ep_ids)}

    def col(table, name):
        return table.column(name).to_pylist()

    src_data_chunk = col(staging_ep_meta, "data/chunk_index")
    src_data_file  = col(staging_ep_meta, "data/file_index")
    src_vid = {
        cam: {
            "chunk":   col(staging_ep_meta, f"videos/{cam}/chunk_index"),
            "file":    col(staging_ep_meta, f"videos/{cam}/file_index"),
            "from_ts": col(staging_ep_meta, f"videos/{cam}/from_timestamp"),
            "to_ts":   col(staging_ep_meta, f"videos/{cam}/to_timestamp"),
        }
        for cam in CAMERAS
    }

    new_ep_frame_offsets: list[int] = []
    new_ep_n_frames: list[int]      = []
    running_frame_offset = frame_offset

    for src_ep in range(n_staging):
        dst_ep    = ep_offset + src_ep
        dst_chunk = dst_ep // chunks_size
        dst_file  = dst_ep % chunks_size
        ri = src_row_of[src_ep]

        # Copy + reindex data parquet
        src_parquet = staging_path / f"data/chunk-{src_data_chunk[ri]:03d}/file-{src_data_file[ri]:03d}.parquet"
        dst_parquet = main_path   / f"data/chunk-{dst_chunk:03d}/file-{dst_file:03d}.parquet"
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(src_parquet)
        n_frames = len(table)
        table = table.set_column(
            table.schema.get_field_index("episode_index"),
            "episode_index",
            pa.array([dst_ep] * n_frames, type=pa.int64()),
        )
        table = table.set_column(
            table.schema.get_field_index("index"),
            "index",
            pa.array(range(running_frame_offset, running_frame_offset + n_frames), type=pa.int64()),
        )
        pq.write_table(table, dst_parquet)

        # Copy video files
        for cam in CAMERAS:
            vc = src_vid[cam]
            src_video = staging_path / f"videos/{cam}/chunk-{vc['chunk'][ri]:03d}/file-{vc['file'][ri]:03d}.mp4"
            dst_video = main_path   / f"videos/{cam}/chunk-{dst_chunk:03d}/file-{dst_file:03d}.mp4"
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_video, dst_video)

        new_ep_frame_offsets.append(running_frame_offset)
        new_ep_n_frames.append(n_frames)
        running_frame_offset += n_frames

    total_new_frames = running_frame_offset - frame_offset

    # --- Update main meta/episodes ---
    def replace_col(t, name, values, typ):
        return t.set_column(t.schema.get_field_index(name), name, pa.array(values, type=typ))

    # Re-order staging meta to match src_ep 0..n_staging-1 order
    ordered_rows = [src_row_of[ep] for ep in range(n_staging)]
    new_ep_table = staging_ep_meta.take(ordered_rows)
    N = n_staging

    new_ep_table = replace_col(new_ep_table, "episode_index",    [ep_offset + i for i in range(N)], pa.int64())
    new_ep_table = replace_col(new_ep_table, "data/chunk_index", [(ep_offset + i) // chunks_size for i in range(N)], pa.int64())
    new_ep_table = replace_col(new_ep_table, "data/file_index",  [(ep_offset + i) % chunks_size  for i in range(N)], pa.int64())
    new_ep_table = replace_col(new_ep_table, "dataset_from_index", new_ep_frame_offsets, pa.int64())
    new_ep_table = replace_col(new_ep_table, "dataset_to_index",
                               [new_ep_frame_offsets[i] + new_ep_n_frames[i] for i in range(N)], pa.int64())

    for cam in CAMERAS:
        new_ep_table = replace_col(new_ep_table, f"videos/{cam}/chunk_index",
                                   [(ep_offset + i) // chunks_size for i in range(N)], pa.int64())
        new_ep_table = replace_col(new_ep_table, f"videos/{cam}/file_index",
                                   [(ep_offset + i) % chunks_size  for i in range(N)], pa.int64())
        new_ep_table = replace_col(new_ep_table, f"videos/{cam}/from_timestamp", [0.0] * N, pa.float64())
        fps = main_info["fps"]
        new_ep_table = replace_col(new_ep_table, f"videos/{cam}/to_timestamp",
                                   [new_ep_n_frames[i] / fps for i in range(N)], pa.float64())

    # Append new rows to the appropriate main meta/episodes chunk file
    last_chunk = (ep_offset + N - 1) // chunks_size
    for dst_chunk in range(ep_offset // chunks_size, last_chunk + 1):
        chunk_start = dst_chunk * chunks_size
        # Rows in new_ep_table that belong to this chunk
        lo = max(chunk_start - ep_offset, 0)
        hi = min((dst_chunk + 1) * chunks_size - ep_offset, N)
        if lo >= hi:
            continue
        chunk_slice = new_ep_table.slice(lo, hi - lo)

        chunk_dir = main_path / "meta" / "episodes" / f"chunk-{dst_chunk:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        existing_files = sorted(chunk_dir.glob("file-*.parquet"))
        if existing_files:
            # Append to the last existing metadata file in this chunk
            existing = pq.read_table(existing_files[-1])
            merged = pa.concat_tables([existing, chunk_slice])
            pq.write_table(merged, existing_files[-1])
        else:
            pq.write_table(chunk_slice, chunk_dir / "file-000.parquet")

    # --- Update main meta/info.json ---
    main_info["total_episodes"] = ep_offset + N
    main_info["total_frames"]   = frame_offset + total_new_frames
    main_info["splits"]         = {"train": f"0:{ep_offset + N}"}
    with open(main_path / "meta" / "info.json", "w") as f:
        json.dump(main_info, f, indent=4)

    # --- Update checkpoint.json ---
    with open(main_path / "checkpoint.json", "w") as f:
        json.dump({
            "total_subtask_recordings": ep_offset + N,
            "last_commit": datetime.datetime.now().isoformat(timespec="seconds"),
        }, f)

    # --- Append to subtask_metadata.jsonl ---
    staging_jsonl = staging_path / "subtask_metadata.jsonl"
    main_jsonl    = main_path    / "subtask_metadata.jsonl"
    staging_lines = [json.loads(l) for l in staging_jsonl.read_text().splitlines() if l.strip()]
    with open(main_jsonl, "a") as f:
        for i, rec in enumerate(staging_lines):
            rec["episode_index"] = ep_offset + i
            f.write(json.dumps(rec) + "\n")

    return N


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

    def __init__(self, dataset: LeRobotDataset, freeze_frames: int = 20):
        self._dataset = dataset
        self._buffer: list[dict] = []
        self._active = False
        self._freeze_frames = freeze_frames
        self._closed = False
        self._metadata_path = Path(dataset.root) / "subtask_metadata.jsonl"
        self._commits_since_push: int = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        dataset_name: str,
        fps: int = 30,
        resume: bool = False,
        overwrite: bool = False,
        freeze_frames: int = 20,
    ) -> "SubtaskRecorder":
        """
        Open or create a LeRobot dataset for recording subtask episodes.

        Args:
            dataset_name: Short name (e.g. "Gal-auto-subtasks"). The HuggingFace
                          repo id is derived as HF_ORG/dataset_name and the local
                          path as SYNTHETIC_DATASETS_DIR/dataset_name.
            resume:       If True and a dataset exists locally, append to it.
                          If the dataset is corrupted, raises RuntimeError (no auto-delete).
            overwrite:    If True, delete any existing dataset and start fresh.
                          Takes precedence over resume.
            freeze_frames: Number of freeze frames appended at the end of each episode.
        """
        repo_id    = f"{HF_ORG}/{dataset_name}"
        local_path = SYNTHETIC_DATASETS_DIR / dataset_name
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
            except Exception as e:
                raise RuntimeError(
                    f"\n⚠  Dataset at {local_path} is corrupted and cannot be resumed.\n"
                    f"   Checkpoint shows {saved_count} subtask recording(s) were committed.\n"
                    f"   Options:\n"
                    f"     - Set RECORD_OVERWRITE=True to wipe and start fresh (deletes data)\n"
                    f"     - Manually delete {local_path} and restart\n"
                    f"   Original error: {type(e).__name__}: {e}"
                ) from e

            n = dataset.meta.total_episodes
            # A SIGKILL crash can leave the meta parquet without a valid footer,
            # causing load_episodes() to fail and total_episodes to reset to 0.
            # If that happens, resumed episodes would get wrong episode_index values
            # and create duplicate indices in the data parquet — the exact failure
            # that corrupted episodes 0-3. The checkpoint is the ground truth.
            if checkpoint_file.exists() and saved_count != n:
                raise RuntimeError(
                    f"\n⚠  Checkpoint/metadata mismatch — dataset was likely corrupted by a hard crash.\n"
                    f"   checkpoint.json says {saved_count} episode(s) were committed.\n"
                    f"   Loaded metadata reports {n} episode(s).\n"
                    f"   Resuming now would assign wrong episode_index values and corrupt the dataset.\n"
                    f"   Options:\n"
                    f"     - Set RECORD_OVERWRITE=True to wipe and start fresh (deletes data)\n"
                    f"     - Manually delete {local_path} and restart"
                )
            print(f"  ▶ Resuming: {n} subtask recordings already saved — appending to {local_path}")
            return cls(dataset, freeze_frames=freeze_frames)

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
        return cls(dataset, freeze_frames=freeze_frames)

    # ------------------------------------------------------------------
    # Per-step recording
    # ------------------------------------------------------------------

    def start(self):
        """Arm the recorder for a new subtask. Any previous unfinished buffer is dropped."""
        if self._closed:
            return
        self._buffer = []
        self._active = True

    def record(self, frame: dict):
        """Buffer one frame. No-op if not armed."""
        if self._active and not self._closed:
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

    def commit(self, task: str, n_placed: int | None = None):
        """
        Flush the buffer as a new subtask recording (LeRobot episode).

        After this returns, the subtask recording is fully safe on disk:
          - data parquet  : writer closed (valid footer) and rotated to a new file
          - metadata      : flushed to the open meta writer (closed on close_writers())
          - checkpoint.json: written atomically with the updated count
          - subtask_metadata.jsonl: one line appended with task + n_placed tag
        """
        if self._closed:
            self._active = False
            self._buffer = []
            return
        if not self._active or not self._buffer:
            self._active = False
            self._buffer = []
            return

        # Append freeze frames (hold last pose) to pad the end of the subtask.
        #
        # observation.state: always copies the last joint state (robot holds its pose).
        # action: depends on the subtask —
        #   GRASP / LIFT ("Pick it up"): keep the last commanded action so the gripper
        #           maintains its closing force on the orange (action ≠ state because
        #           the gripper is still being driven closed).
        #   others: set action = observation.state so the robot commands "stay here"
        #           with no movement (position control: target = current position).
        last_frame = self._buffer[-1]
        if self._freeze_frames > 0:
            last_state  = last_frame.get("observation.state")
            last_action = last_frame.get("action")
            # Build the freeze action: arm joints always hold their current state,
            # gripper holds last_action only for GRASP (to maintain closing force).
            freeze_action = np.asarray(last_state, dtype=np.float32).copy()
            if (task.lower().startswith("grasp") or task.lower().startswith("pick it up")) and last_action is not None:
                freeze_action[-1] = np.asarray(last_action, dtype=np.float32)[-1]  # gripper only
            for _ in range(self._freeze_frames):
                freeze = dict(last_frame)
                if last_state is not None:
                    freeze["observation.state"] = np.asarray(last_state, dtype=np.float32).copy()
                if last_action is not None:
                    freeze["action"] = freeze_action.copy()
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
        self._append_metadata(task, n_placed)
        print(f"  📼 Subtask recording saved: \"{task}\" ({n} frames) — total: {total}")

        self._commits_since_push += 1
        if self._commits_since_push >= INCREMENTAL_PUSH_EVERY:
            self.push_incremental()

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
        if self._closed:
            return
        self.discard()
        self._dataset.finalize()
        self._closed = True

    def push_to_hub(self):
        """Push the completed dataset to HuggingFace Hub."""
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RevisionNotFoundError
        self.close_writers()
        self._validate_metadata_before_upload()
        hub_api = HfApi()
        hub_api.upload_large_folder(
            repo_id=self._dataset.repo_id,
            repo_type="dataset",
            folder_path=self._dataset.root,
            ignore_patterns=[
                "checkpoint.json",
                "checkpoint.json.tmp",
                "meta/episodes.bak/**",
                "meta/**/*.bak/**",
                "**/*.tmp",
            ],
        )
        # LeRobot loads datasets from the codebase-version tag by default
        # (for this install, v3.0). Keep that tag aligned with the uploaded
        # default branch so training does not read an old snapshot.
        with contextlib.suppress(RevisionNotFoundError):
            hub_api.delete_tag(self._dataset.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
        hub_api.create_tag(self._dataset.repo_id, tag=CODEBASE_VERSION, revision="main", repo_type="dataset")
        print(f"  📤 Dataset pushed to Hub: {self._dataset.repo_id} ({CODEBASE_VERSION} -> main)")

    def push_incremental(self):
        """
        Flush all writers, push current state to HF, then reopen for continued recording.

        Called automatically every INCREMENTAL_PUSH_EVERY commits. Uses
        upload_large_folder which is incremental — only new/changed files are sent.
        """
        from huggingface_hub import HfApi

        # finalize() writes valid parquet footers for all open writers
        # (data parquet is already rotated per commit; this closes the meta writer).
        self._dataset.finalize()
        self._validate_metadata_before_upload()

        hub_api = HfApi()
        hub_api.upload_large_folder(
            repo_id=self._dataset.repo_id,
            repo_type="dataset",
            folder_path=self._dataset.root,
            ignore_patterns=[
                "checkpoint.json",
                "checkpoint.json.tmp",
                "meta/episodes.bak/**",
                "meta/**/*.bak/**",
                "**/*.tmp",
            ],
        )
        total = self._dataset.meta.total_episodes
        print(f"  📤 Incremental push: {total} subtask recordings on Hub. Reopening dataset...")

        # Reopen in resume mode so recording continues with correct episode indices.
        self._dataset = LeRobotDataset(
            repo_id=self._dataset.repo_id,
            root=self._dataset.root,
            vcodec="libsvtav1",
        )
        self._dataset.meta.metadata_buffer_size = 1
        self._commits_since_push = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append_metadata(self, task: str, n_placed: int | None):
        """Append one line to subtask_metadata.jsonl with the episode tag."""
        entry = {
            "episode_index": self._dataset.meta.total_episodes - 1,
            "task": task,
            "n_placed": n_placed,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        with open(self._metadata_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

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

    def _validate_metadata_before_upload(self):
        """Check active episode metadata counts before publishing the dataset."""
        import pyarrow.parquet as pq

        root = Path(self._dataset.root)
        info_file = root / "meta" / "info.json"
        episodes_dir = root / "meta" / "episodes"
        checkpoint_file = root / "checkpoint.json"

        with open(info_file) as f:
            total_episodes = json.load(f)["total_episodes"]

        episode_files = sorted(episodes_dir.glob("chunk-*/*.parquet"))
        metadata_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in episode_files)

        mismatches = []
        if metadata_rows != total_episodes:
            mismatches.append(
                f"active episode metadata rows={metadata_rows}, info.json total_episodes={total_episodes}"
            )

        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                checkpoint_total = json.load(f).get("total_subtask_recordings")
            if checkpoint_total != total_episodes:
                mismatches.append(
                    f"checkpoint total_subtask_recordings={checkpoint_total}, info.json total_episodes={total_episodes}"
                )

        if mismatches:
            raise RuntimeError(
                "Dataset metadata validation failed before upload:\n  - "
                + "\n  - ".join(mismatches)
            )

        backup_dirs = sorted(path for path in (root / "meta").glob("*.bak") if path.is_dir())
        if backup_dirs:
            names = ", ".join(path.name for path in backup_dirs)
            print(f"  ⚠ Ignoring backup metadata directories during upload: {names}")

        print(f"  ✅ Dataset metadata validated: {total_episodes} active episode(s)")
