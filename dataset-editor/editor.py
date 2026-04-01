#!/usr/bin/env python3
"""
LeRobot V3 Dataset Inspector & Task Editor (GUI version)

Usage:
    python editor.py <dataset>

    <dataset> can be:
      - A HuggingFace repo ID:  lerobot/aloha_sim_insertion
      - A local directory path:  ./my_dataset

Keyboard shortcuts (when the window is focused):
    Left / Right         ±1 frame
    Shift+Left/Right     ±10 frames
    [ / ]                ±10 frames
    m                    Mark start frame
    M (Shift+m)          Mark end frame + enter task string
    u                    Undo last edit
    n / p                Next / previous episode
    s                    Export / save
    q                    Quit

Requirements:
    pip install opencv-python pandas pyarrow Pillow huggingface_hub
"""

import argparse
import sys

from lerobot_editor.data_loader import resolve_dataset_path, discover_cameras
from lerobot_editor.gui import EditorApp


def main():
    parser = argparse.ArgumentParser(
        description="LeRobot V3 Dataset Inspector & Task Editor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dataset", help="HuggingFace repo ID or local dataset path")
    args = parser.parse_args()

    dataset_path = resolve_dataset_path(args.dataset)

    cameras = discover_cameras(dataset_path)
    if not cameras:
        sys.exit("[ERROR] No camera video directories found under videos/")

    app = EditorApp(dataset_path)
    app.run()


if __name__ == "__main__":
    main()
