"""Per-label grasp obedience for tab:obedience --- the prevalence-vs-obedience inversion.

Obedience = the gripper closed on the *requested* orange (matched by identity), i.e. the
diagonal of the confusion matrices in ``plot_grasp_confusion.py``. We reuse that module's
``confusion()`` so the definition matches the figure exactly, and aggregate over both subtask
models and the two informative scene states (0 and 1 oranges already placed).

Training prevalence (the Teleop/Auto/Combined counts) comes from
``extract_positional_prevalence.py`` and is unchanged; this script only adds the obeyed column.

  python compute_obedience.py
"""

from __future__ import annotations

from collections import defaultdict

from plot_grasp_confusion import MODELS, confusion


def main() -> None:
    agg = defaultdict(lambda: [0, 0])  # label -> [obeyed, total]
    for _disp, subdir, fname in MODELS:
        for n_placed in (0, 1):
            conf = confusion(subdir, fname, n_placed)
            for req, row in conf.items():
                agg[req][0] += row.get(req, 0)
                agg[req][1] += sum(row.values())

    order = ["right", "middle", "left", "top right", "bottom right"]
    print("Per-label obedience (both subtask models, scene states 0+1):")
    for label in order:
        ob, tot = agg[label]
        print(f"  {label:13s} {100 * ob / tot:4.0f}% ({ob}/{tot})" if tot else f"  {label:13s} n/a")
    tot_ob = sum(v[0] for v in agg.values())
    tot_n = sum(v[1] for v in agg.values())
    print(f"  {'overall':13s} {100 * tot_ob / tot_n:4.0f}% ({tot_ob}/{tot_n})")


if __name__ == "__main__":
    main()
