# Dataset Editor

GUI tool for manually annotating and correcting LeRobot V3 datasets. Runs on the laptop.

## Entry Points

| File | Purpose |
|------|---------|
| `editor.py` | Main GUI — frame-by-frame episode viewer with task label editor and export |
| `tailer.py` | Frozen tail analyzer and fixer — detects and strips duplicate trailing frames from Place episodes |

## Usage

```bash
python editor.py <dataset>
```

`<dataset>` can be a HuggingFace repo ID (e.g. `MasterProject2026/my-dataset`) or a local directory path.

**Keyboard shortcuts inside the editor:**

| Key | Action |
|-----|--------|
| Left / Right | ±1 frame |
| Shift + Left/Right | ±10 frames |
| `[` / `]` | ±10 frames (alternative to Shift+Left/Right) |
| `m` | Mark start frame |
| `M` | Mark end frame + enter task string |
| `u` | Undo last edit |
| `n` / `p` | Next / previous episode |
| `s` | Export / save |
| `q` | Quit |

## Package: `lerobot_editor/`

The backing Python package used by `editor.py`:

| Module | Role |
|--------|------|
| `data_loader.py` | Resolves dataset path, discovers cameras, loads parquet and video data |
| `exporter.py` | Writes edited metadata back to disk in LeRobot V3 format. **Note:** contains a `split_all_episodes()` function stub that is WIP / incomplete; the core export path works but the split feature was never finished. |
| `gui.py` | Tkinter/OpenCV GUI logic and event loop |
| `state.py` | In-memory edit state. Classes: `TaskEdit(start, end, task)` — a single labeled region; `EditorState` — all edits, frame-to-task lookup, action/state arrays; `ProgressTracker` — per-episode annotation progress. |

## Check Scripts (`check_scripts/`)

Standalone dataset inspection utilities — run directly, no GUI needed:

| Script | Purpose |
|--------|---------|
| `check_counts.py` | Verifies episode subtask counts (expects 3 non-HOME tasks per episode) |
| `compare.py` | Diffs two dataset parquet metadata files frame-by-frame |
| `edit_json.py` | Analyzes task distribution — counts and statistics across subtask labels. Read-only inspection tool (despite the name). |
| `inspect_par.py` | Compares metadata exports between two dataset versions (`verify_meta_exports()`). Validates `tasks.parquet` schema and episode parquet schemas across old/new exports. |
| `visualize_episode.py` | Plays back camera frames using PyAV — same decode path as the LeRobot training loader |

## Keeping this file current

Update this file **and** `CLAUDE.md` in the same commit as any structural change: new modules, deleted scripts, renamed files, or significant feature additions. The `.md` and the code must always agree. Always commit and push after changes — never leave the working tree dirty.

**Commit messages must not include "Co-Authored-By: Claude" or any AI attribution line.**
