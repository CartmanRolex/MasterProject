"""Composite-label (top right / bottom right) mixup geometry --- feeds the
"Inconsistent composite-label boundaries" Limitations item.

For every successful GRASP whose requested label is *top right* or *bottom
right* (scene states 0--1, all five subtask models pooled), reconstructs the
grasp-time scene, finds the close pair, and prints the lateral (x) and depth
(y) gap between the two, split by obeyed vs within-pair misgrab.

Message: misgrabs do NOT cluster near the 3 cm lateral threshold (obeyed and
misgrabbed requests share the same x-gap distribution) and the pair is always
clearly separated in depth --- so the mixups look like a training-supervision
gap (scarce, inconsistently bounded composite labels) rather than genuinely
ambiguous evaluation scenes. The teleop annotations themselves cannot be
audited against the rule here (no scene geometry stored with them).

Pure stdlib; reuses plot_grasp_confusion's scene reconstruction.

    python compute_composite_labels.py
"""

from __future__ import annotations

import json
import statistics as st

from plot_grasp_confusion import GRID, RESULTS, classify

COMPOSITE = ("top right", "bottom right")


def pair_gaps():
    rows = []
    subs = [s for _, subdirs in GRID for s in subdirs if s]
    for sub in subs:
        data = json.load(open(RESULTS / sub / "checkpoint.json"))
        for e in data["episodes"]:
            for a in e["subtask_attempts"]:
                if a.get("subtask") != "GRASP" or a.get("result") != "success":
                    continue
                if a.get("n_placed_start") not in (0, 1) or a.get("requested_label") not in COMPOSITE:
                    continue
                scene = {n: list(o["position"]) for n, o in e["initial_scene"]["oranges"].items()}
                plate = e["initial_scene"]["plate_position"]
                placed = []
                for ev in e.get("timeline", []):
                    if ev.get("event_type") == "place_success" and ev["step"] <= a["start_step"]:
                        o = ev.get("actual_orange") or ev.get("requested_orange")
                        if o and o not in placed:
                            placed.append(o)
                for po in placed[: a.get("n_placed_start", 0)]:
                    scene[po] = plate
                labels = classify(scene)
                inv = {v: k for k, v in labels.items()}
                if any(c not in inv for c in COMPOSITE):
                    continue  # pair separated (or scene changed) by grasp time
                br, tr = scene[inv["bottom right"]], scene[inv["top right"]]
                obeyed = bool(a.get("target_match"))
                grabbed = a["requested_label"] if obeyed else labels.get(a.get("actual_orange"), "?")
                rows.append((a["requested_label"], obeyed, grabbed,
                             abs(br[0] - tr[0]), abs(br[1] - tr[1])))
    return rows


def main() -> None:
    rows = pair_gaps()
    print("Composite-label grasp geometry (five subtask models pooled, states 0+1):")
    for req in COMPOSITE:
        sel = [r for r in rows if r[0] == req]
        obey = [r for r in sel if r[1]]
        mis = [r for r in sel if not r[1] and r[2] in COMPOSITE]  # within-pair misgrab
        print(f"\n  requested '{req}': n={len(sel)}, obeyed={len(obey)}, within-pair misgrab={len(mis)}")
        for name, grp in (("obeyed", obey), ("within-pair misgrab", mis)):
            if not grp:
                continue
            xg = [r[3] * 100 for r in grp]
            yg = [r[4] * 100 for r in grp]
            print(f"    {name:20s} x-gap cm median {st.median(xg):.1f} (range {min(xg):.1f}-{max(xg):.1f})"
                  f"   y-gap cm median {st.median(yg):.1f} (range {min(yg):.1f}-{max(yg):.1f})")


if __name__ == "__main__":
    main()
