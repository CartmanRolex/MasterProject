"""Editor state, task edits, and progress tracking."""

import json
from pathlib import Path

import pandas as pd


class TaskEdit:
    """One pending edit: assign a task string to a frame range."""
    def __init__(self, start: int, end: int, task: str):
        self.start = start
        self.end = end
        self.task = task

    def __repr__(self):
        return f"TaskEdit({self.start}-{self.end}: '{self.task}')"


class EditorState:
    def __init__(self, tasks_df: pd.DataFrame, frame_data: pd.DataFrame):
        self.tasks_df = tasks_df.copy()
        self.frame_data = frame_data.copy()
        self.edits: list[TaskEdit] = []
        self.mark_start: int | None = None

        self.task_lookup: dict[int, str] = {}
        for _, row in tasks_df.iterrows():
            self.task_lookup[int(row["task_index"])] = str(row["task"])

        # Pre-build frame_index → task_index dict for O(1) lookup
        self._frame_task: dict[int, int] = {}
        self._frame_action: dict[int, list[float]] = {}
        for _, row in frame_data.iterrows():
            fi = int(row["frame_index"])
            self._frame_task[fi] = int(row["task_index"])
            if "action" in row.index and row["action"] is not None:
                self._frame_action[fi] = list(row["action"])

    def get_task_for_frame(self, frame_idx: int) -> str:
        for edit in reversed(self.edits):
            if edit.start <= frame_idx <= edit.end:
                return edit.task
        tidx = self._frame_task.get(frame_idx)
        if tidx is not None:
            return self.task_lookup.get(tidx, f"<unknown task_index={tidx}>")
        return "<no data>"

    def get_edit_index_for_frame(self, frame_idx: int) -> int | None:
        """Return the edit index if this frame is in an edit region, else None."""
        for i, edit in enumerate(reversed(self.edits)):
            if edit.start <= frame_idx <= edit.end:
                return len(self.edits) - 1 - i
        return None

    def add_edit(self, start: int, end: int, task: str):
        self.edits.append(TaskEdit(start, end, task))

    def undo_last_edit(self) -> TaskEdit | None:
        if self.edits:
            return self.edits.pop()
        return None


class ProgressTracker:
    """Tracks which episodes have been edited. Saves/loads from meta/edits.json."""

    def __init__(self, dataset_path: Path):
        self.path = dataset_path / "meta" / "edits.json"
        self.data: dict = {"episodes": {}}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    self.data = json.load(f)
                if "episodes" not in self.data:
                    self.data["episodes"] = {}
            except (json.JSONDecodeError, IOError):
                self.data = {"episodes": {}}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_edits_for_episode(self, episode_idx: int) -> list[TaskEdit]:
        key = str(episode_idx)
        if key not in self.data["episodes"]:
            return []
        return [
            TaskEdit(e["start"], e["end"], e["task"])
            for e in self.data["episodes"][key]["edits"]
        ]

    def save_episode_edits(self, episode_idx: int, edits: list[TaskEdit]):
        key = str(episode_idx)
        self.data["episodes"][key] = {
            "done": True,
            "edits": [{"start": e.start, "end": e.end, "task": e.task} for e in edits],
        }
        self.save()

    def is_done(self, episode_idx: int) -> bool:
        key = str(episode_idx)
        return key in self.data["episodes"] and self.data["episodes"][key].get("done", False)

    def get_done_episodes(self) -> list[int]:
        return [int(k) for k, v in self.data["episodes"].items() if v.get("done", False)]