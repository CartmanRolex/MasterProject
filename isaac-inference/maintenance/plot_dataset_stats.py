import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def group_task(task: str) -> str:
    return "Grasp orange" if task.startswith("Grasp") else task


GROUP_ORDER = ["Grasp orange", "Pick it up", "Place it into plate", "Go back to start position"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def bar_labels(ax, rects, fmt="{:.0f}"):
    for rect in rects:
        w = rect.get_width() if hasattr(rect, "get_width") else rect.get_height()
        if hasattr(rect, "get_width"):
            ax.text(
                rect.get_width() + ax.get_xlim()[1] * 0.005,
                rect.get_y() + rect.get_height() / 2,
                fmt.format(w),
                va="center",
                fontsize=8,
            )
        else:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + ax.get_ylim()[1] * 0.005,
                fmt.format(w),
                ha="center",
                va="bottom",
                fontsize=8,
            )


def main():
    parser = argparse.ArgumentParser(description="Plot stats for a synthetic dataset folder.")
    parser.add_argument("folder", help="Path to dataset folder containing subtask_metadata.jsonl")
    args = parser.parse_args()

    folder = Path(args.folder)
    metadata_path = folder / "subtask_metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"subtask_metadata.jsonl not found in {folder}")

    records = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]

    raw_counts = Counter(r["task"] for r in records)
    grouped_counts = Counter(group_task(r["task"]) for r in records)
    nplaced_counts = Counter(r["n_placed"] for r in records)

    groups = [g for g in GROUP_ORDER if g in grouped_counts]
    nplaced_vals = sorted(nplaced_counts)
    group_nplaced = {
        g: Counter(r["n_placed"] for r in records if group_task(r["task"]) == g)
        for g in groups
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Dataset stats — {folder.name}  ({len(records)} episodes)", fontsize=14, fontweight="bold")

    # [0,0] Raw subtask counts — horizontal bar
    ax = axes[0, 0]
    raw_labels = sorted(raw_counts, key=lambda t: raw_counts[t])
    raw_vals = [raw_counts[t] for t in raw_labels]
    rects = ax.barh(raw_labels, raw_vals, color="#4C72B0", edgecolor="white")
    bar_labels(ax, rects)
    ax.set_xlabel("Count")
    ax.set_title("Subtask types (raw)")
    ax.set_xlim(0, max(raw_vals) * 1.15)

    # [0,1] Grouped subtask counts — vertical bar
    ax = axes[0, 1]
    group_vals = [grouped_counts[g] for g in groups]
    x = np.arange(len(groups))
    rects = ax.bar(x, group_vals, color=COLORS[: len(groups)], edgecolor="white")
    bar_labels(ax, rects)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Subtask types (grouped)")
    ax.set_ylim(0, max(group_vals) * 1.12)

    # Warm sequential palette for n_placed levels (YlOrRd: light yellow → dark red)
    nplaced_palette = [plt.cm.YlOrRd(0.2 + 0.25 * i) for i in range(4)]

    # [1,0] n_placed overall distribution
    ax = axes[1, 0]
    np_vals = [nplaced_counts.get(v, 0) for v in nplaced_vals]
    rects = ax.bar(
        [str(v) for v in nplaced_vals],
        np_vals,
        color=nplaced_palette[: len(nplaced_vals)],
        edgecolor="white",
    )
    bar_labels(ax, rects)
    ax.set_xlabel("Oranges placed (n_placed)")
    ax.set_ylabel("Count")
    ax.set_title("Oranges placed at time of subtask")
    ax.set_ylim(0, max(np_vals) * 1.12)

    # [1,1] n_placed per task group — stacked bar
    ax = axes[1, 1]
    bottom = np.zeros(len(groups))
    for i, nv in enumerate(nplaced_vals):
        color = nplaced_palette[i % len(nplaced_palette)]
        r, g, b, _ = color
        txt_color = "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.5 else "white"
        heights = np.array([group_nplaced[grp].get(nv, 0) for grp in groups])
        bars = ax.bar(
            x,
            heights,
            bottom=bottom,
            label=f"n_placed={nv}",
            color=color,
            edgecolor="white",
        )
        for j, (rect, h) in enumerate(zip(bars, heights)):
            if h > 0:
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    bottom[j] + h / 2,
                    str(h),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=txt_color,
                    fontweight="bold",
                )
        bottom += heights
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Oranges placed per task group (stacked)")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    out = folder / "subtask_stats.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
