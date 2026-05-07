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
| `exporter.py` | Writes edited metadata back to disk in LeRobot V3 format |
| `gui.py` | Tkinter/OpenCV GUI logic and event loop |
| `state.py` | In-memory edit state: mark ranges, task strings, undo stack |

## Check Scripts (`check_scripts/`)

Standalone dataset inspection utilities — run directly, no GUI needed:

| Script | Purpose |
|--------|---------|
| `check_counts.py` | Verifies episode subtask counts (expects 3 non-HOME tasks per episode) |
| `compare.py` | Diffs two dataset parquet metadata files frame-by-frame |
| `edit_json.py` | Batch-edits JSON metadata fields across episodes |
| `inspect_par.py` | Prints parquet file contents for a given episode |
| `visualize_episode.py` | Plays back camera frames using PyAV — same decode path as the LeRobot training loader |
