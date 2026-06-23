"""Interactive episode phase viewer for eval checkpoints (monotask + subtask).

Runs on the LAPTOP against the git-tracked checkpoints under
``isaac-inference/results/``. Pick a model and an episode and it draws a clear
step-axis timeline of what the policy did:

  * Subtask lane    -- GRASP / LIFT / PLACE attempts as bars, coloured by
                       subtask; failed / timed-out spans get a thin red
                       underline instead of a heavy outline.
  * Target lane     -- which orange the attempt worked on, coloured per orange
                       (``requested_orange`` for subtask runs, post-processed
                       inferred target for monotask / flat runs). Blank gaps =
                       uncertain transit / home / spatial-reset.
  * Event markers   -- grasp/lift/place success, drops & failures, local retry,
                       target redirection, grasp timeout, wrong-orange grasp.
  * Spatial resets  -- shaded grey bands (subtask runs only).
  * Task progress   -- placed oranges plus grasp/lift/place thirds, so failed
                       attempts visibly drop back to the last confirmed count.
  * Progress strip  -- cumulative oranges in plate over the episode, derived
                       from place-success events.

Both checkpoint dialects are handled: orchestrated ``checkpoint.json`` and flat
``flat_checkpoint.json`` / ``act_checkpoint.json``.

Usage
-----
GUI (default)::

    python view_episode.py

Headless export of a single episode to PNG (no display needed)::

    python view_episode.py --export <model_dir_name> <episode_index> out.png

Needs ``matplotlib`` and ``tkinter`` (both present on the laptop).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "isaac-inference" / "results"

SUBTASK_COLORS = {"GRASP": "#4C72B0", "LIFT": "#DD8452", "PLACE": "#55A868"}
ORANGE_COLORS = {
    "Orange001": "#E69F00",
    "Orange002": "#56B4E9",
    "Orange003": "#009E73",
}
ORANGE_FALLBACK = ["#CC79A7", "#999999", "#000000"]

# Flat / monotask traces infer the target from nearest-orange geometry. The
# signal is useful only after the gripper commits to an approach, so the viewer
# smooths short flips and keeps already-placed oranges out of the target lane.
FAR_GRASP_DISTANCE_M = 0.14
STABLE_TARGET_STEPS = 120
SHORT_TARGET_JITTER_STEPS = 95
APPROACH_BACKFILL_STEPS = 260
EVENT_ROW_SPACING = 0.16
HIDDEN_EVENT_TYPES = {"grasp_retargeted"}
PROBLEM_STRIP_COLOR = "#CC2F27"
TASK_PROGRESS_COLOR = "#2F3437"
SUBTASK_FRACTION = {
    "grasp_started": 1 / 6,
    "grasp_success": 1 / 3,
    "lift_started": 1 / 3,
    "lift_success": 2 / 3,
    "place_started": 2 / 3,
}
ROLLBACK_EVENTS = {
    "grasp_timeout",
    "place_timeout",
    "grasp_failure",
    "lift_failure",
    "place_failure",
    "orange_dropped",
    "wrong_orange_grasped",
    "no_confirmed_progress",
}

# event_type -> (marker, colour, legend label). Covers both dialects.
EVENT_STYLE = {
    "grasp_success": ("^", "#2ca02c", "grasp ok"),
    "grasp_failure": ("x", "#d62728", "grasp fail"),
    "lift_success": ("^", "#17a36b", "lift ok"),
    "place_success": ("*", "#1f7a1f", "place ok"),
    "orange_dropped": ("v", "#d62728", "drop"),
    "lift_failure": ("x", "#d62728", "lift fail"),
    "place_failure": ("x", "#b22222", "place fail"),
    "grasp_timeout": ("s", "#ff7f0e", "grasp timeout"),
    "place_timeout": ("s", "#cc6600", "place timeout"),
    "local_retry": ("o", "#1f77b4", "local retry"),
    "target_redirection": ("D", "#9467bd", "redirection"),
    "grasp_retargeted": ("D", "#9467bd", "retarget (inferred)"),
    "wrong_orange_grasped": ("X", "#8c1a1a", "wrong orange"),
    "placed_orange_left_plate": ("v", "#d62728", "left plate"),
    "no_confirmed_progress": ("x", "#777777", "no progress"),
}


def discover_models():
    """List (label, checkpoint_path) for every non-archived result dir."""
    out = []
    if not RESULTS.exists():
        return out
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir() or d.name == "_archive":
            continue
        for fn in ("checkpoint.json", "flat_checkpoint.json", "act_checkpoint.json"):
            if (d / fn).exists():
                out.append((d.name, d / fn))
                break
    return out


def target_of(attempt):
    return attempt.get("requested_orange") or attempt.get("inferred_target_orange")


def orange_color(name, order):
    if name in ORANGE_COLORS:
        return ORANGE_COLORS[name]
    try:
        idx = order.index(name)
    except ValueError:
        idx = len(order)
    idx %= len(ORANGE_FALLBACK)
    return ORANGE_FALLBACK[idx]


def orange_label(name):
    if isinstance(name, str) and name.startswith("Orange"):
        suffix = name.removeprefix("Orange").lstrip("0") or "0"
        return f"O{suffix}"
    return str(name)


def _step(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _attempt_bounds(attempt):
    s = _step(attempt.get("start_step"))
    e = _step(attempt.get("end_step"), s)
    return s, max(e, s)


def _attempt_failed(attempt):
    return _attempt_problem(attempt) is not None


def _attempt_problem(attempt):
    result = attempt.get("result")
    if attempt.get("failure_reason") == "episode_ended":
        return None
    if result == "timeout":
        return "timeout"
    if result == "failure":
        return "failure"
    if attempt.get("failure_reason") and result not in ("success", "retargeted"):
        return "failure"
    return None


def _is_flat_episode(ep):
    summary = ep.get("episode_summary", {})
    if summary.get("inferred"):
        return True
    return any(
        a.get("requested_orange") is None and a.get("inferred_target_orange")
        for a in ep.get("subtask_attempts", [])
    )


def _plate_events(timeline):
    events = []
    for ev in timeline:
        event_type = ev.get("event_type")
        if event_type not in ("place_success", "placed_orange_left_plate"):
            continue
        orange = (
            ev.get("actual_orange")
            or ev.get("requested_orange")
            or ev.get("inferred_target_orange")
        )
        if orange:
            events.append((_step(ev.get("step")), orange, event_type == "place_success"))
    return sorted(events)


def _placed_before(plate_events, step):
    placed = set()
    for ev_step, orange, in_plate in plate_events:
        if ev_step > step:
            break
        if in_plate:
            placed.add(orange)
        else:
            placed.discard(orange)
    return placed


def _phase_segments(attempts, flat):
    segments = []
    for attempt in sorted(attempts, key=lambda a: _attempt_bounds(a)[0]):
        s, e = _attempt_bounds(attempt)
        if e <= s:
            continue
        subtask = attempt.get("subtask") or "?"
        problem = _attempt_problem(attempt)
        segment = {
            "start": s,
            "end": e,
            "subtask": subtask,
            "failed": problem is not None,
            "problem": problem,
        }
        if (
            flat
            and segments
            and segments[-1]["subtask"] == subtask
            and s <= segments[-1]["end"] + 1
        ):
            segments[-1]["end"] = max(segments[-1]["end"], e)
            if problem:
                segments[-1]["failed"] = True
                segments[-1]["problem"] = problem
        else:
            segments.append(segment)
    return segments


def _target_items(ep, attempts, timeline, flat):
    plate_events = _plate_events(timeline)
    items = []
    for attempt in sorted(attempts, key=lambda a: _attempt_bounds(a)[0]):
        s, e = _attempt_bounds(attempt)
        if e <= s:
            continue

        subtask = attempt.get("subtask")
        actual = attempt.get("actual_orange")
        raw_target = actual or target_of(attempt)
        locked = bool(actual or attempt.get("requested_orange"))
        far = False

        if flat and subtask == "GRASP":
            if raw_target in _placed_before(plate_events, s):
                raw_target = None
                locked = False
            metrics = attempt.get("metrics") or {}
            distance = metrics.get("center_distance_m")
            far = distance is not None and distance > FAR_GRASP_DISTANCE_M
            if far and not actual:
                raw_target = None

        items.append(
            {
                "start": s,
                "end": e,
                "target": raw_target,
                "locked": locked,
                "far": far,
            }
        )

    if flat:
        return _smooth_target_items(items)
    return _merge_target_items(items, keep_unknown=False)


def _merge_target_items(items, keep_unknown=True):
    merged = []
    for item in sorted(items, key=lambda s: s["start"]):
        start, end = item["start"], item["end"]
        if end <= start:
            continue
        target = item.get("target")
        if target is None and not keep_unknown:
            continue
        if (
            merged
            and merged[-1]["target"] == target
            and start <= merged[-1]["end"] + 1
        ):
            merged[-1]["end"] = max(merged[-1]["end"], end)
            merged[-1]["locked"] = merged[-1]["locked"] or item.get("locked", False)
            merged[-1]["far"] = merged[-1]["far"] and item.get("far", False)
            continue
        merged.append(
            {
                "start": start,
                "end": end,
                "target": target,
                "locked": item.get("locked", False),
                "far": item.get("far", False),
            }
        )
    for segment in merged:
        segment["duration"] = segment["end"] - segment["start"]
    return merged


def _target_stable(segment):
    return (
        segment.get("target") is not None
        and (segment.get("locked") or segment.get("duration", 0) >= STABLE_TARGET_STEPS)
    )


def _fill_approach_runs(segments):
    changed = False
    i = 0
    while i < len(segments):
        if _target_stable(segments[i]):
            i += 1
            continue

        start_i = i
        while i < len(segments) and not _target_stable(segments[i]):
            i += 1
        if i >= len(segments):
            break

        lead = segments[start_i:i]
        lead_span = segments[i]["start"] - lead[0]["start"]
        uncertain = all(
            (not seg.get("locked"))
            and (
                seg.get("target") is None
                or seg.get("duration", 0) <= SHORT_TARGET_JITTER_STEPS
                or seg.get("far")
            )
            for seg in lead
        )
        if lead_span <= APPROACH_BACKFILL_STEPS and uncertain:
            target = segments[i]["target"]
            for seg in lead:
                if seg.get("target") != target:
                    seg["target"] = target
                    seg["far"] = False
                    changed = True
    return changed


def _absorb_short_interruptions(segments):
    changed = False
    for i in range(1, len(segments) - 1):
        prev_seg, seg, next_seg = segments[i - 1], segments[i], segments[i + 1]
        if seg.get("locked") or seg.get("duration", 0) > SHORT_TARGET_JITTER_STEPS:
            continue
        neighbor_target = prev_seg.get("target")
        if neighbor_target is None or neighbor_target != next_seg.get("target"):
            continue
        if seg.get("target") != neighbor_target:
            seg["target"] = neighbor_target
            seg["far"] = False
            changed = True
    return changed


def _smooth_target_items(items):
    segments = _merge_target_items(items)
    for _ in range(12):
        changed = _fill_approach_runs(segments)
        segments = _merge_target_items(segments)
        changed = _absorb_short_interruptions(segments) or changed
        segments = _merge_target_items(segments)
        if not changed:
            break
    return [seg for seg in segments if seg.get("target") is not None]


def _event_style(ev):
    if ev.get("event_type") == "grasp_failure" and ev.get("reason") == "no_confirmed_progress":
        return ("x", "#777777", "no progress")
    return EVENT_STYLE.get(ev.get("event_type"))


def _visible_events(timeline):
    events = []
    for ev in timeline:
        if ev.get("event_type") in HIDDEN_EVENT_TYPES:
            continue
        if ev.get("reason") == "episode_ended" and ev.get("event_type", "").endswith("_failure"):
            continue
        if _event_style(ev):
            events.append(ev)
    return events


def _draw_interval(ax, start, end, row, height, **kwargs):
    width = end - start
    if width <= 0:
        return
    ax.broken_barh([(start, width)], (row - height / 2, height), **kwargs)


def _event_orange(ev):
    return (
        ev.get("actual_orange")
        or ev.get("requested_orange")
        or ev.get("inferred_target_orange")
    )


def _progress_points(ep, timeline, total):
    xs, ys = [0], [0]
    placed = set()
    count = 0

    def push(step, next_count):
        nonlocal count
        if next_count == count:
            return
        xs.append(step)
        ys.append(count)
        count = next_count
        xs.append(step)
        ys.append(count)

    for ev in sorted(timeline, key=lambda item: _step(item.get("step"))):
        event_type = ev.get("event_type")
        if event_type not in ("place_success", "placed_orange_left_plate"):
            continue
        orange = _event_orange(ev)
        if event_type == "place_success":
            if orange:
                placed.add(orange)
                push(_step(ev.get("step")), len(placed))
            else:
                push(_step(ev.get("step")), count + 1)
        elif orange and orange in placed:
            placed.remove(orange)
            push(_step(ev.get("step")), len(placed))
        elif not orange and count > 0:
            push(_step(ev.get("step")), count - 1)

    final_count = ep.get("oranges_in_plate")
    if isinstance(final_count, int):
        push(total, final_count)
    if xs[-1] != total or ys[-1] != count:
        xs.append(total)
        ys.append(count)
    return xs, ys


def _task_progress_points(ep, timeline, total):
    xs, ys = [0], [0.0]
    placed = set()
    placed_count = 0
    value = 0.0

    def set_placed_count(next_count):
        nonlocal placed_count
        placed_count = max(0, min(3, int(next_count)))

    def push(step, next_value):
        nonlocal value
        next_value = max(0.0, min(3.0, float(next_value)))
        if abs(next_value - value) < 1e-9:
            return
        xs.append(step)
        ys.append(value)
        value = next_value
        xs.append(step)
        ys.append(value)

    def baseline():
        return placed_count

    for ev in sorted(timeline, key=lambda item: _step(item.get("step"))):
        step = _step(ev.get("step"))
        event_type = ev.get("event_type")
        if ev.get("reason") == "episode_ended" and event_type in ROLLBACK_EVENTS:
            continue

        if event_type in SUBTASK_FRACTION:
            push(step, baseline() + SUBTASK_FRACTION[event_type])
            continue

        if event_type == "place_success":
            orange = _event_orange(ev)
            if orange:
                placed.add(orange)
                set_placed_count(max(placed_count, len(placed)))
            else:
                set_placed_count(placed_count + 1)
            push(step, baseline())
            continue

        if event_type == "placed_orange_left_plate":
            orange = _event_orange(ev)
            if orange and orange in placed:
                placed.remove(orange)
                set_placed_count(len(placed))
            elif not orange:
                set_placed_count(placed_count - 1)
            push(step, baseline())
            continue

        if event_type in ROLLBACK_EVENTS:
            push(step, baseline())

    final_count = ep.get("oranges_in_plate")
    if isinstance(final_count, int):
        set_placed_count(final_count)
        push(total, baseline())
    if xs[-1] != total or abs(ys[-1] - value) > 1e-9:
        xs.append(total)
        ys.append(value)
    return xs, ys


def render(fig, checkpoint_path: Path, ep_index: int):
    """Draw the episode timeline onto ``fig``. Returns the matplotlib figure."""
    fig.clear()
    with open(checkpoint_path, encoding="utf-8") as f:
        data = json.load(f)
    episodes = data["episodes"]
    if not (0 <= ep_index < len(episodes)):
        raise IndexError(f"episode {ep_index} out of range (0..{len(episodes) - 1})")
    ep = episodes[ep_index]
    attempts = ep.get("subtask_attempts", [])
    timeline = ep.get("timeline", [])
    summ = ep.get("episode_summary", {})
    total = ep.get("step_count", max((a.get("end_step", 0) for a in attempts), default=1))
    flat = _is_flat_episode(ep)
    phase_segments = _phase_segments(attempts, flat)
    target_segments = _target_items(ep, attempts, timeline, flat)
    visible_events = _visible_events(timeline)

    # stable orange ordering (by name) for consistent colours
    order = sorted({s["target"] for s in target_segments if s.get("target")})

    gs = fig.add_gridspec(3, 1, height_ratios=[3.1, 1.15, 0.9], hspace=0.10)
    ax = fig.add_subplot(gs[0])
    axg = fig.add_subplot(gs[1], sharex=ax)
    axp = fig.add_subplot(gs[2], sharex=ax)

    Y_TGT, Y_SUB, Y_EVT = 0.0, 1.0, 2.0
    H = 0.58

    # spatial-reset shaded bands (pair start->finish; subtask runs only)
    reset_start = None
    for ev in timeline:
        if ev["event_type"] == "spatial_reset":
            reset_start = ev["step"]
        elif ev["event_type"] == "spatial_reset_finished" and reset_start is not None:
            ax.axvspan(reset_start, ev["step"], color="0.88", zorder=0)
            reset_start = None

    # Subtask lane: flat traces are merged so nearest-target retargets do not
    # masquerade as separate behavioral attempts.
    for segment in phase_segments:
        sub = segment["subtask"]
        _draw_interval(
            ax,
            segment["start"],
            segment["end"],
            Y_SUB,
            H,
            facecolors=SUBTASK_COLORS.get(sub, "#aaaaaa"),
            edgecolors="white",
            linewidth=0.7,
            zorder=2,
        )
        if segment["failed"]:
            _draw_interval(
                ax,
                segment["start"],
                segment["end"],
                Y_SUB - H / 2 + 0.045,
                0.075,
                facecolors=PROBLEM_STRIP_COLOR,
                edgecolors="none",
                alpha=0.95,
                zorder=3,
            )

    # Target lane: requested target for orchestrated runs, smoothed target for
    # flat / monotask runs.
    for segment in target_segments:
        target = segment.get("target")
        if not target:
            continue
        _draw_interval(
            ax,
            segment["start"],
            segment["end"],
            Y_TGT,
            H,
            facecolors=orange_color(target, order),
            edgecolors="white",
            linewidth=0.5,
            alpha=0.95,
            zorder=2,
        )

    # Event markers get their own lane and are stacked when several events
    # share a step, which avoids timeout/redirection marker collisions.
    seen_labels = {}
    events_by_step = defaultdict(list)
    for ev in visible_events:
        events_by_step[_step(ev.get("step"))].append(ev)
    for step, events in sorted(events_by_step.items()):
        n_events = len(events)
        for i, ev in enumerate(events):
            style = _event_style(ev)
            if not style:
                continue
            y_ev = Y_EVT + (i - (n_events - 1) / 2) * EVENT_ROW_SPACING
            marker, color, label = style
            lbl = label if label not in seen_labels else "_nolegend_"
            seen_labels[label] = True
            ax.scatter(
                step,
                y_ev,
                marker=marker,
                c=color,
                s=68,
                linewidths=1.5,
                zorder=4,
                label=lbl,
                clip_on=False,
            )

    max_event_stack = max((len(v) for v in events_by_step.values()), default=1)
    event_pad = max(0.35, (max_event_stack - 1) * EVENT_ROW_SPACING / 2 + 0.25)
    ax.set_yticks([Y_TGT, Y_SUB, Y_EVT])
    ax.set_yticklabels(["target\norange", "subtask", "events"])
    ax.set_ylim(Y_TGT - H, Y_EVT + event_pad)
    x_pad = max(1.0, total * 0.015)
    ax.set_xlim(0, total + x_pad)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.tick_params(labelbottom=False)

    # legends: subtask colours + orange colours + event markers
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    sub_handles = [Patch(facecolor=c, label=k) for k, c in SUBTASK_COLORS.items()]
    org_handles = [Patch(facecolor=orange_color(o, order), label=orange_label(o)) for o in order]
    problem_handles = []
    if any(segment["failed"] for segment in phase_segments):
        problem_handles = [
            Line2D([0], [0], color=PROBLEM_STRIP_COLOR, lw=3,
                   label="failed / timed out span")
        ]
    leg1 = ax.legend(handles=sub_handles + org_handles + problem_handles, loc="upper left",
                     bbox_to_anchor=(1.005, 1.0), fontsize=8, title="phases / oranges",
                     frameon=False)
    ax.add_artist(leg1)
    event_handles, event_labels = ax.get_legend_handles_labels()
    if event_handles:
        ax.legend(event_handles, event_labels, loc="lower left",
                  bbox_to_anchor=(1.005, 0.0), fontsize=8,
                  title="events", frameon=False)

    # Task-progress curve: each orange advances through grasp/lift/place thirds
    # and rolls back to the last confirmed placed count on failed attempts.
    for base in range(3):
        axg.axhspan(base, base + 1 / 3, color=SUBTASK_COLORS["GRASP"],
                    alpha=0.045, linewidth=0)
        axg.axhspan(base + 1 / 3, base + 2 / 3, color=SUBTASK_COLORS["LIFT"],
                    alpha=0.045, linewidth=0)
        axg.axhspan(base + 2 / 3, base + 1, color=SUBTASK_COLORS["PLACE"],
                    alpha=0.045, linewidth=0)
    xs_task, ys_task = _task_progress_points(ep, timeline, total)
    axg.step(xs_task, ys_task, where="post", color=TASK_PROGRESS_COLOR, linewidth=2.0)
    axg.set_ylim(-0.08, 3.08)
    axg.set_yticks([0, 1, 2, 3])
    axg.set_yticks(
        [base + frac for base in range(3) for frac in (1 / 3, 2 / 3)],
        minor=True,
    )
    axg.set_ylabel("task\nstage")
    axg.grid(axis="x", linestyle=":", alpha=0.4)
    axg.grid(axis="y", which="major", linestyle="-", alpha=0.18)
    axg.grid(axis="y", which="minor", linestyle=":", alpha=0.16)
    axg.tick_params(labelbottom=False)

    # progress strip: cumulative placed from events, reconciled with the
    # episode summary for traces that finish as the last place stabilizes.
    xs, ys = _progress_points(ep, timeline, total)
    axp.step(xs, ys, where="post", color="#1f7a1f", linewidth=1.8)
    axp.fill_between(xs, ys, step="post", color="#1f7a1f", alpha=0.12)
    axp.set_ylim(-0.2, 3.2)
    axp.set_yticks([0, 1, 2, 3])
    axp.set_ylabel("in plate")
    axp.set_xlabel("environment step")
    axp.grid(linestyle=":", alpha=0.4)

    ok = summ.get("success")
    segment_note = (
        f"{len(phase_segments)} phases ({len(attempts)} raw)"
        if flat and len(phase_segments) != len(attempts)
        else f"{len(attempts)} attempts"
    )
    title = (f"{data.get('result_name') or checkpoint_path.parent.name}  |  episode {ep_index}"
             f"  (seed {ep.get('seed')})\n"
             f"final {ep.get('oranges_in_plate', '?')}/3 in plate  -  "
             f"{'SUCCESS' if ok else 'fail'}  -  end: {summ.get('end_reason', '?')}  -  "
             f"{total} steps  -  {segment_note}")
    ax.set_title(title, fontsize=10, loc="left")
    fig.subplots_adjust(left=0.085, right=0.80, top=0.89, bottom=0.09, hspace=0.10)
    return fig


def _label_of(ep, orange):
    sc = ep.get("initial_scene", {}).get("oranges", {})
    return sc.get(orange, {}).get("label", "?")


# --------------------------------------------------------------------------- GUI
def launch_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    models = discover_models()
    if not models:
        print(f"No checkpoints found under {RESULTS}", file=sys.stderr)
        return

    root = tk.Tk()
    root.title("Episode phase viewer")
    root.geometry("1300x820")

    bar = ttk.Frame(root, padding=6)
    bar.pack(side=tk.TOP, fill=tk.X)

    ttk.Label(bar, text="Model:").pack(side=tk.LEFT)
    model_box = ttk.Combobox(bar, values=[m[0] for m in models], width=42, state="readonly")
    model_box.current(0)
    model_box.pack(side=tk.LEFT, padx=(2, 12))

    ttk.Label(bar, text="Episode:").pack(side=tk.LEFT)
    ep_box = ttk.Combobox(bar, width=8, state="readonly")
    ep_box.pack(side=tk.LEFT, padx=2)

    fig = Figure(figsize=(13, 7.4))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    NavigationToolbar2Tk(canvas, root)

    def path_for(label):
        return dict(models)[label]

    def refresh_episodes(*_):
        data = json.load(open(path_for(model_box.get())))
        n = len(data["episodes"])
        ep_box["values"] = list(range(n))
        if ep_box.get() == "" or int(ep_box.get() or 0) >= n:
            ep_box.current(0)
        draw()

    def draw(*_):
        try:
            render(fig, path_for(model_box.get()), int(ep_box.get() or 0))
            canvas.draw()
        except Exception as exc:  # noqa: BLE001 - surface to the user
            messagebox.showerror("Render error", str(exc))

    model_box.bind("<<ComboboxSelected>>", refresh_episodes)
    ep_box.bind("<<ComboboxSelected>>", draw)
    ttk.Button(bar, text="Reload", command=draw).pack(side=tk.LEFT, padx=8)

    refresh_episodes()
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", nargs=3, metavar=("MODEL_DIR", "EP", "OUT_PNG"),
                    help="headless: render one episode to a PNG and exit")
    args = ap.parse_args()

    if args.export:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        model_dir, ep, out = args.export
        ckpt = dict(discover_models()).get(model_dir)
        if ckpt is None:
            sys.exit(f"unknown model dir '{model_dir}'. Available: "
                     f"{[m[0] for m in discover_models()]}")
        fig = Figure(figsize=(13, 7.4))
        render(fig, ckpt, int(ep))
        fig.savefig(out, dpi=130)
        print(f"wrote {out}")
        return

    launch_gui()


if __name__ == "__main__":
    main()
