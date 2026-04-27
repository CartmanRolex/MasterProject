"""Tkinter-based GUI for the LeRobot dataset editor."""

import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, filedialog
from pathlib import Path
from PIL import Image, ImageTk
import cv2
import numpy as np

from .data_loader import (
    VideoReader, discover_cameras, find_video_for_episode,
    load_episodes_metadata, load_episode_list, load_frame_data, load_tasks,
)
from .state import EditorState, ProgressTracker, TaskEdit
from .exporter import export_dataset


# Colors for edit regions (tkinter hex)
EDIT_COLORS = ["#00cc44", "#ff8800", "#3388ff", "#cccc00", "#cc44cc", "#44cccc", "#ff4444", "#8844ff"]
EDIT_COLORS_BGR = [
    (0, 204, 68), (0, 136, 255), (255, 136, 51), (0, 204, 204),
    (204, 68, 204), (204, 204, 68), (68, 68, 255), (255, 68, 136),
]


class EditorApp:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.root = tk.Tk()
        self.root.title(f"LeRobot Editor — {dataset_path.name}")
        self.root.configure(bg="#1e1e1e")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Load dataset metadata
        self.tasks_df = load_tasks(dataset_path)
        self.episodes_df = load_episodes_metadata(dataset_path)
        self.episodes = load_episode_list(self.episodes_df)
        self.cameras = discover_cameras(dataset_path)
        self.progress = ProgressTracker(dataset_path)

        # Current episode state
        self.ep_pos = 0
        self.state: EditorState | None = None
        self.readers: list[VideoReader] = []
        self.video_offsets: list[int] = []
        self.active_cameras: list[str] = []
        self.min_frame = 0
        self.total_frames = 1
        self.current_frame = 0

        self._build_ui()
        self._load_episode(self.episodes[0])
        self._update_display()

        # Bind keys
        self.root.bind("<Left>", lambda e: self._step(-1))
        self.root.bind("<Right>", lambda e: self._step(1))
        self.root.bind("<Shift-Left>", lambda e: self._step(-10))
        self.root.bind("<Shift-Right>", lambda e: self._step(10))
        self.root.bind("m", self._on_mark_start)
        self.root.bind("M", self._on_mark_end)
        self.root.bind("u", lambda e: self._on_undo())
        self.root.bind("n", lambda e: self._next_episode())
        self.root.bind("p", lambda e: self._prev_episode())
        self.root.bind("s", lambda e: self._on_export())
        self.root.bind("q", lambda e: self._on_close())
        # "f" + arrow → skip to next/prev state change (skips frozen frames)
        self._f_held = False
        self._f_arrow_used = False
        self.root.bind("<KeyPress-f>", self._on_f_press)
        self.root.bind("<KeyRelease-f>", self._on_f_release)

    # ── UI CONSTRUCTION ──────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#e0e0e0", font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#1e1e1e", foreground="#ffffff", font=("Segoe UI", 12, "bold"))
        style.configure("Task.TLabel", background="#1e1e1e", foreground="#88ff88", font=("Segoe UI", 11))
        style.configure("Status.TLabel", background="#2a2a2a", foreground="#aaaaaa", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 9))
        style.configure("Done.TLabel", background="#1e1e1e", foreground="#44bb44", font=("Segoe UI", 9))
        style.configure("Pending.TLabel", background="#1e1e1e", foreground="#888888", font=("Segoe UI", 9))

        # ── TOP: Episode selector + info ──
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=(8, 2))

        ttk.Label(top, text="Episode:", style="Header.TLabel").pack(side="left")

        self.progress_label = ttk.Label(top, text="", style="TLabel")
        self.progress_label.pack(side="right", padx=8)

        self.ep_combo = ttk.Combobox(top, width=8, state="readonly", font=("Segoe UI", 10))
        self.ep_combo.pack(side="left", padx=(4, 8))
        self._refresh_episode_list()
        self.ep_combo.bind("<<ComboboxSelected>>", self._on_episode_select)

        btn_frame = ttk.Frame(top)
        btn_frame.pack(side="left", padx=4)
        ttk.Button(btn_frame, text="◀ Prev (p)", command=self._prev_episode, width=10).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Next (n) ▶", command=self._next_episode, width=10).pack(side="left", padx=2)

        self.ep_info_label = ttk.Label(top, text="", style="TLabel")
        self.ep_info_label.pack(side="left", padx=12)

        # ── VIDEO CANVAS ──
        self.canvas = tk.Canvas(self.root, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=4)
        self._photo = None  # prevent garbage collection

        # ── TIMELINE BAR (drawn on canvas at bottom) ──
        self.timeline = tk.Canvas(self.root, bg="#111111", height=32, highlightthickness=0)
        self.timeline.pack(fill="x", padx=8, pady=(0, 2))
        self.timeline.bind("<Button-1>", self._on_timeline_click)
        self.timeline.bind("<B1-Motion>", self._on_timeline_click)

        # ── FRAME SLIDER ──
        slider_frame = ttk.Frame(self.root)
        slider_frame.pack(fill="x", padx=8, pady=2)

        self.frame_label = ttk.Label(slider_frame, text="Frame: 0/0", style="TLabel")
        self.frame_label.pack(side="left")

        self.slider = ttk.Scale(slider_frame, from_=0, to=100, orient="horizontal",
                                command=self._on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=8)

        # ── TASK DISPLAY ──
        task_frame = ttk.Frame(self.root)
        task_frame.pack(fill="x", padx=8, pady=2)

        ttk.Label(task_frame, text="Task:", style="Header.TLabel").pack(side="left")
        self.task_label = ttk.Label(task_frame, text="—", style="Task.TLabel", wraplength=800)
        self.task_label.pack(side="left", padx=8, fill="x", expand=True)

        # ── MARK / EDIT CONTROLS ──
        edit_frame = ttk.Frame(self.root)
        edit_frame.pack(fill="x", padx=8, pady=4)

        self.mark_btn = ttk.Button(edit_frame, text="Set Mark Start (m)", command=lambda: self._on_mark_start(None))
        self.mark_btn.pack(side="left", padx=4)

        self.mark_end_btn = ttk.Button(edit_frame, text="Set Mark End + Task (M)", command=lambda: self._on_mark_end(None))
        self.mark_end_btn.pack(side="left", padx=4)

        ttk.Button(edit_frame, text="Undo (u)", command=self._on_undo).pack(side="left", padx=4)
        ttk.Button(edit_frame, text="Split Episode", command=self._on_split_episode).pack(side="left", padx=4)
        self.mark_label = ttk.Label(edit_frame, text="No mark set", style="TLabel")
        self.mark_label.pack(side="left", padx=12)

        # ── EDITS LIST ──
        edits_outer = ttk.Frame(self.root)
        edits_outer.pack(fill="x", padx=8, pady=(2, 4))

        ttk.Label(edits_outer, text="Edits:", style="Header.TLabel").pack(anchor="w")
        self.edits_list = tk.Listbox(
            edits_outer, height=5, bg="#2a2a2a", fg="#e0e0e0",
            font=("Consolas", 9), selectbackground="#444444", borderwidth=0,
        )
        self.edits_list.pack(fill="x", pady=2)
        self.edits_list.bind("<Double-1>", self._on_edit_task_double_click)
        # ── BOTTOM: Export + status ──
        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x", padx=8, pady=(2, 8))

        ttk.Button(bottom, text="💾 Export / Save (s)", command=self._on_export).pack(side="left", padx=4)
        ttk.Button(bottom, text="Save Progress Only", command=self._on_save_progress).pack(side="left", padx=4)

        self.status_label = ttk.Label(bottom, text="Ready", style="Status.TLabel")
        self.status_label.pack(side="right", padx=8)

    def _refresh_episode_list(self):
        """Refresh combo box with done/pending markers."""
        done = set(self.progress.get_done_episodes())
        items = []
        for ep in self.episodes:
            tag = " ✓" if ep in done else ""
            items.append(f"{ep}{tag}")
        self.ep_combo["values"] = items
        if self.ep_pos < len(items):
            self.ep_combo.current(self.ep_pos)

        # Update progress label
        n_done = len(done)
        total = len(self.episodes)
        self.progress_label.config(text=f"Progress: {n_done}/{total}")

    def _on_edit_task_double_click(self, event):
        """Double-click an edit to rename its task string."""
        sel = self.edits_list.curselection()
        if not sel or self.state is None:
            return
        idx = sel[0]
        if idx >= len(self.state.edits):
            return
        edit = self.state.edits[idx]
        new_task = simpledialog.askstring(
            "Edit Task",
            f"Change task for frames {edit.start}–{edit.end}:",
            initialvalue=edit.task,
            parent=self.root,
        )
        if new_task and new_task.strip():
            edit.task = new_task.strip()
            self._refresh_edits_list()
            self._set_status(f"Edit [{idx}] task changed to '{edit.task}'")
            self._request_redraw()
    # ── EPISODE LOADING ──────────────────────────

    def _close_readers(self):
        for r in self.readers:
            r.close()
        self.readers = []
        self.video_offsets = []
        self.active_cameras = []

    def _load_episode(self, episode_idx: int):
        self._close_readers()

        frame_data = load_frame_data(self.dataset_path, episode_idx)
        if len(frame_data) == 0:
            self._set_status(f"No frame data for episode {episode_idx}")
            return

        frame_indices = sorted(frame_data["frame_index"].unique().tolist())
        self.min_frame = frame_indices[0]
        max_frame = frame_indices[-1]
        self.total_frames = max_frame - self.min_frame + 1

        ep_row = self.episodes_df[self.episodes_df["episode_index"] == episode_idx]
        video_frame_offset = int(ep_row.iloc[0].get("dataset_from_index", self.min_frame)) if len(ep_row) > 0 else self.min_frame

        for cam in self.cameras:
            vpath = find_video_for_episode(self.dataset_path, cam, episode_idx, self.episodes_df)
            if vpath:
                try:
                    reader = VideoReader(vpath)
                    self.readers.append(reader)
                    self.active_cameras.append(cam)

                    if reader.total_frames > self.total_frames * 1.5:
                        ts_col = f"videos/{cam}/from_timestamp"
                        if len(ep_row) > 0 and ts_col in ep_row.columns:
                            from_ts = float(ep_row.iloc[0][ts_col])
                            fps = reader.fps if reader.fps > 0 else 30.0
                            offset = round(from_ts * fps)
                        else:
                            offset = video_frame_offset
                        self.video_offsets.append(offset)
                    else:
                        self.video_offsets.append(0)
                except RuntimeError:
                    pass

        self.state = EditorState(self.tasks_df, frame_data)

        # Load previous edits
        previous = self.progress.get_edits_for_episode(episode_idx)
        if previous:
            self.state.edits = previous

        self.current_frame = 0
        self.slider.configure(to=max(self.total_frames - 1, 0))
        self.slider.set(0)

        done_tag = " [DONE]" if self.progress.is_done(episode_idx) else ""
        self.ep_info_label.config(
            text=f"Frames: {self.min_frame}–{max_frame} ({self.total_frames}){done_tag}  |  "
                 f"Cameras: {', '.join(self.active_cameras)}"
        )
        self._refresh_edits_list()
        self._set_status(f"Loaded episode {episode_idx}")
        self._request_redraw()

    # ── DISPLAY ──────────────────────────────────

    def _request_redraw(self):
        """Mark that the display needs updating."""
        self._dirty = True

    def _update_display(self):
        if not self.readers or self.state is None:
            self.root.after(50, self._update_display)
            return

        # Only re-render if something changed
        if not getattr(self, '_dirty', True):
            self.root.after(50, self._update_display)
            return

        self._dirty = False
        dataset_frame = self.current_frame + self.min_frame

        # Read and composite frames
        panels = []
        for i, reader in enumerate(self.readers):
            video_frame = self.current_frame + self.video_offsets[i]
            frame = reader.read_frame(video_frame)
            if frame is None:
                frame = np.zeros((360, 480, 3), dtype=np.uint8)
            panels.append(frame)

        if panels:
            target_h = 400
            resized = []
            for p in panels:
                h, w = p.shape[:2]
                scale = target_h / h
                resized.append(cv2.resize(p, (int(w * scale), target_h)))
            composite = np.hstack(resized)
        else:
            composite = np.zeros((400, 640, 3), dtype=np.uint8)

        # Overlay action values on the video
        JOINT_NAMES = ["sh_pan", "sh_lift", "elbow", "wr_flex", "wr_roll", "gripper"]
        action = self.state._frame_action.get(dataset_frame)
        if action:
            y0 = composite.shape[0] - 12 - len(action) * 18
            for j, (name, val) in enumerate(zip(JOINT_NAMES, action)):
                is_gripper = (j == len(action) - 1)
                color = (0, 255, 255) if is_gripper else (200, 200, 200)  # yellow vs gray (BGR)
                txt = f"{name}: {val:+.1f}"
                if is_gripper:
                    txt = f">> {name}: {val:+.1f} <<"
                cv2.putText(composite, txt, (8, y0 + j * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2)  # shadow
                cv2.putText(composite, txt, (8, y0 + j * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        # Convert BGR → RGB for tkinter
        rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Fit to canvas size
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        img_w, img_h = img.size
        scale = min(cw / img_w, ch / img_h)
        if scale < 1:
            img = img.resize((int(img_w * scale), int(img_h * scale)), Image.LANCZOS)

        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")

        # Update labels
        task = self.state.get_task_for_frame(dataset_frame)
        edit_idx = self.state.get_edit_index_for_frame(dataset_frame)
        if edit_idx is not None:
            color = EDIT_COLORS[edit_idx % len(EDIT_COLORS)]
            self.task_label.config(text=f"[EDITED] {task}", foreground=color)
        else:
            self.task_label.config(text=task, foreground="#88ff88")

        episode_idx = self.episodes[self.ep_pos]
        self.frame_label.config(text=f"Frame: {dataset_frame}  |  ep {episode_idx}  |  {self.current_frame}/{self.total_frames - 1} within ep")

        if self.state.mark_start is not None:
            self.mark_label.config(text=f"Mark start: {self.state.mark_start}", foreground="#ff6666")
        else:
            self.mark_label.config(text="No mark set", foreground="#888888")

        # Draw timeline
        self._draw_timeline()

        self.root.after(50, self._update_display)

    def _draw_timeline(self):
        self.timeline.delete("all")
        w = self.timeline.winfo_width()
        h = self.timeline.winfo_height()
        if w < 10 or self.state is None:
            return

        # Draw edit regions
        for i, edit in enumerate(self.state.edits):
            rel_start = max(0, edit.start - self.min_frame)
            rel_end = max(0, edit.end - self.min_frame)
            x1 = int(rel_start / max(1, self.total_frames - 1) * w)
            x2 = int(rel_end / max(1, self.total_frames - 1) * w)
            color = EDIT_COLORS[i % len(EDIT_COLORS)]
            self.timeline.create_rectangle(x1, 4, x2, h - 4, fill=color, outline="", stipple="")

        # Playhead
        px = int(self.current_frame / max(1, self.total_frames - 1) * w)
        self.timeline.create_line(px, 0, px, h, fill="#ffffff", width=2)

        # Mark start indicator
        if self.state and self.state.mark_start is not None:
            mx = int((self.state.mark_start - self.min_frame) / max(1, self.total_frames - 1) * w)
            self.timeline.create_line(mx, 0, mx, h, fill="#ff4444", width=2, dash=(4, 2))

    def _on_timeline_click(self, event):
        w = self.timeline.winfo_width()
        if w > 0 and self.total_frames > 0:
            frac = max(0, min(1, event.x / w))
            self.current_frame = int(frac * (self.total_frames - 1))
            self.slider.set(self.current_frame)
            self._request_redraw()

    # ── NAVIGATION ───────────────────────────────

    def _on_slider(self, val):
        new_frame = int(float(val))
        if new_frame != self.current_frame:
            self.current_frame = new_frame
            self._request_redraw()

    def _step(self, delta):
        if self._f_held:
            self._f_arrow_used = True
            self._jump_to_frozen_region_end(1 if delta > 0 else -1)
            return
        new_frame = max(0, min(self.total_frames - 1, self.current_frame + delta))
        if new_frame != self.current_frame:
            self.current_frame = new_frame
            self.slider.set(self.current_frame)
            self._request_redraw()

    def _on_f_press(self, _event=None):
        self._f_held = True
        self._f_arrow_used = False

    def _on_f_release(self, _event=None):
        self._f_held = False
        self._f_arrow_used = False

    def _jump_to_frozen_region_end(self, direction: int):
        """Jump to the last frame of the next (or previous) frozen block.

        A frozen block is a run of consecutive frames with identical arm state
        (joints 0-4, gripper excluded). Landing on the last frame of the block
        makes it easy to see exactly where the freeze ends.
        """
        if self.state is None:
            return
        df = self.current_frame + self.min_frame
        first = self.min_frame
        last = self.min_frame + self.total_frames - 1

        def same(a, b):
            return all(abs(x - y) <= 0.1 for x, y in zip(a, b))

        cur = self.state._frame_state.get(df)
        if cur is None:
            return

        if direction > 0:
            fi = df + 1
            # Exit the current frozen block (if we're inside one)
            while fi <= last:
                s = self.state._frame_state.get(fi)
                if s is None or not same(s, cur):
                    break
                fi += 1
            # Skip non-frozen (moving) frames to find the next frozen block
            while fi < last:
                s_cur = self.state._frame_state.get(fi)
                s_next = self.state._frame_state.get(fi + 1)
                if s_cur is not None and s_next is not None and same(s_cur, s_next):
                    break
                fi += 1
            if fi >= last:
                return
            # Walk to the last frame of this frozen block
            frozen_state = self.state._frame_state.get(fi)
            while fi + 1 <= last:
                s = self.state._frame_state.get(fi + 1)
                if s is None or not same(s, frozen_state):
                    break
                fi += 1
        else:
            fi = df - 1
            # Exit the current frozen block going backward
            while fi >= first:
                s = self.state._frame_state.get(fi)
                if s is None or not same(s, cur):
                    break
                fi -= 1
            # Skip non-frozen frames backward to find the previous frozen block.
            # Stop when state[fi] == state[fi-1]; fi is then the last frame of that block.
            while fi > first:
                s_cur = self.state._frame_state.get(fi)
                s_prev = self.state._frame_state.get(fi - 1)
                if s_cur is not None and s_prev is not None and same(s_cur, s_prev):
                    break
                fi -= 1
            if fi <= first:
                return

        target = fi - self.min_frame
        if 0 <= target < self.total_frames:
            self.current_frame = target
            self.slider.set(self.current_frame)
            self._request_redraw()

    def _on_episode_select(self, event):
        idx = self.ep_combo.current()
        if 0 <= idx < len(self.episodes):
            self.ep_pos = idx
            self._load_episode(self.episodes[idx])

    def _next_episode(self):
        self.ep_pos = min(self.ep_pos + 1, len(self.episodes) - 1)
        self.ep_combo.current(self.ep_pos)
        self._load_episode(self.episodes[self.ep_pos])

    def _prev_episode(self):
        self.ep_pos = max(self.ep_pos - 1, 0)
        self.ep_combo.current(self.ep_pos)
        self._load_episode(self.episodes[self.ep_pos])

    # ── EDITING ──────────────────────────────────
    def _on_split_episode(self):
        """Split ALL edited episodes into sub-episodes, saving to a new folder."""
        # Save current edits first
        episode_idx = self.episodes[self.ep_pos]
        if self.state and self.state.edits:
            self.progress.save_episode_edits(episode_idx, self.state.edits)

        all_done = self.progress.get_done_episodes()
        eps_with_edits = [ep for ep in all_done if self.progress.get_edits_for_episode(ep)]

        if not eps_with_edits:
            self._set_status("No edits to split on. Mark regions first.")
            return

        ok = messagebox.askyesno(
            "Split All Episodes",
            f"This will split {len(eps_with_edits)} episode(s) into sub-episodes.\n\n"
            f"Episodes: {sorted(eps_with_edits)}\n\n"
            f"You will choose an output folder (original dataset is NOT modified).\n\n"
            f"Continue?",
        )
        if not ok:
            return

        folder = filedialog.askdirectory(title="Choose output folder for split dataset")
        if not folder:
            return
        output_path = Path(folder)

        self._set_status("Splitting all episodes...")
        self.root.update()

        try:
            from .exporter import split_all_episodes
            msg = split_all_episodes(
                self.dataset_path,
                self.progress,
                output_path,
            )
            self._set_status("Split complete")
            messagebox.showinfo("Split Complete", msg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Split Error", str(e))
            self._set_status(f"Split failed: {e}")

    def _on_mark_start(self, event):
        if self.state is None:
            return
        dataset_frame = self.current_frame + self.min_frame
        self.state.mark_start = dataset_frame
        self._set_status(f"Mark start set to frame {dataset_frame}")
        self._request_redraw()

    def _on_mark_end(self, event):
        if self.state is None:
            return
        if self.state.mark_start is None:
            self._set_status("No mark start set. Press 'm' first.")
            return

        dataset_frame = self.current_frame + self.min_frame
        mark_start = self.state.mark_start
        mark_end = dataset_frame
        if mark_end < mark_start:
            mark_start, mark_end = mark_end, mark_start

        # Popup dialog for task string
        task_str = simpledialog.askstring(
            "Task String",
            f"Enter task for frames {mark_start}–{mark_end}:",
            parent=self.root,
        )
        if task_str and task_str.strip():
            self.state.add_edit(mark_start, mark_end, task_str.strip())
            self._refresh_edits_list()
            self._set_status(f"Added edit: frames {mark_start}-{mark_end} → '{task_str.strip()}'")
        else:
            self._set_status("Edit cancelled.")
        self.state.mark_start = None
        self._request_redraw()

    def _on_undo(self):
        if self.state is None:
            return
        removed = self.state.undo_last_edit()
        if removed:
            self._refresh_edits_list()
            self._set_status(f"Undone: {removed}")
        else:
            self._set_status("Nothing to undo.")
        self._request_redraw()

    def _refresh_edits_list(self):
        self.edits_list.delete(0, tk.END)
        if self.state:
            for i, e in enumerate(self.state.edits):
                color_name = EDIT_COLORS[i % len(EDIT_COLORS)]
                self.edits_list.insert(tk.END, f"  [{i}] frames {e.start}–{e.end}: {e.task}")
                self.edits_list.itemconfig(i, fg=color_name)

    # ── EXPORT ───────────────────────────────────

    def _on_export(self):
        # We don't block exporting anymore if state.edits is empty,
        # because the user might just want to export ALL previous work.
        
        episode_idx = self.episodes[self.ep_pos]

        # 1. Force-save the current episode's progress before exporting
        if self.state and self.state.edits:
            self.progress.save_episode_edits(episode_idx, self.state.edits)

        # 2. Ask: overwrite or new path?
        choice = messagebox.askyesnocancel(
            "Export",
            f"Export all dataset edits?\n\n"
            f"YES = Overwrite in place\n"
            f"NO = Choose a new folder\n"
            f"CANCEL = Abort",
        )
        if choice is None:
            return

        if choice:
            output_path = self.dataset_path
        else:
            folder = filedialog.askdirectory(title="Choose export folder")
            if not folder:
                return
            output_path = Path(folder)

        self._set_status("Exporting all edits...")
        self.root.update()

        try:
            # 3. PASS 'self.progress' INSTEAD OF 'self.state' AND DROP 'episode_idx'
            msg = export_dataset(self.dataset_path, self.progress, output_path)
            
            self._refresh_episode_list()
            self._set_status(f"Export Complete")
            messagebox.showinfo("Export Complete", msg)
        except Exception as e:
            import traceback
            traceback.print_exc() # Helps with debugging in the console
            messagebox.showerror("Export Error", str(e))
            self._set_status(f"Export failed: {e}")

    def _on_save_progress(self):
        if self.state is None or not self.state.edits:
            self._set_status("No edits to save.")
            return
        episode_idx = self.episodes[self.ep_pos]
        self.progress.save_episode_edits(episode_idx, self.state.edits)
        self._refresh_episode_list()
        self._set_status(f"Progress saved for episode {episode_idx}")

    # ── HELPERS ───────────────────────────────────

    def _set_status(self, text: str):
        self.status_label.config(text=text)

    def _on_close(self):
        self._close_readers()
        self.root.destroy()

    def run(self):
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        self.root.after(10, self._update_display)
        self.root.mainloop()