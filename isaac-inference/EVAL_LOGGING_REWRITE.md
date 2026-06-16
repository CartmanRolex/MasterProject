# Eval checkpoint logging rewrite — clean failure/timeout capture

Goal: derive subtask **failures and timeouts for both models** directly from
authoritative geometry ("is the orange in the plate"), instead of the brittle
post-processing the report analyses currently need. **Logging-only**: no control
flow changes, so the seeded rerun (`pick_orange_reference_100_v1`) reproduces
identical trajectories — existing numbers are confirmed, not shifted.

Decision (confirmed): **no subtask timeout for the monotask** runs — they have no
per-subtask budget, so their only outcomes are `success` / `drop` / episode-end.
Timeouts remain subtask-only.

## Bug found: `final_scene` uses post-reset positions

At episode end (`inference_autonomous_orders.py`):

- `oranges_in_plate` is computed from `final_positions` — `last_positions` when
  truncated, else a fresh `save_positions(env)` (line ~1286).
- but `finish_story_episode(...)` is passed the **live post-step**
  `plate_pos, orange_positions` from `_get_env_data` (line ~1299).

On truncation (the common ending) Isaac Lab auto-resets inside `env.step()`, so
those live positions are the **reset** state. Hence `final_scene` stores
oranges ~0.2–0.3 m from the plate while `oranges_in_plate` (from the pre-reset
snapshot) says they are placed. `final_scene` is therefore unusable for geometry,
and `status` is left `"unknown"`. **Fix:** build `final_scene` and the placed set
from the same `final_positions` snapshot used for the count.

## Brittle post-processing this rewrite removes

Lift (§ lift recovery):
- L1 outcome split by string-matching `result`/`failure_reason`.
- L2 monotask lift identity/began/success are physics-inferred (PhaseMonitor).
- L3 "dropped orange eventually lifted" rebuilt from `place_success` events +
  the broken `final_scene`.
- L4/L5 re-engagement / retries rely on inferred identity + attempt ordering.

Place (§ placing failures):
- P1 success conflates *in-plate* with *gripper-retracted* (the `_check_place`
  gate `PLACE_GRIPPER_Z_MIN`): placed-but-unretracted oranges logged as timeout.
- P2 no per-attempt in-plate state.
- P3 no placed-orange identities (only the count) → "not re-engaged ⇒ placed"
  heuristic, wrong when the episode just ran out.
- P4 `final_scene` positions wrong (bug above).
- P5 `n_placed_start` missing for monotask → reconstructed from events.
- P6/P7 monotask place inferred; bounce-outs (placed then left) need capping.

## Schema additions (additive; bump `SCHEMA_VERSION`)

A shared geometry snapshot, consistent with `count_oranges_in_plate` (tilt-aware,
upside-down ⇒ not placed):

```
scene = {
  "plate_position": [x,y,z],
  "n_in_plate": int,
  "oranges": { "OrangeNNN": {"in_plate": bool, "held": bool, "position": [x,y,z]} }
}
```

1. **Per subtask attempt:** `scene_start` and `scene_end` snapshots, plus a
   convenience `target_in_plate_end` (the requested/actual orange's `in_plate` at
   `scene_end`). A place/lift "failure" is then simply `not target_in_plate_end`;
   a `timeout` keeps its explicit terminal cause but is now decidable as
   placed-vs-not by `target_in_plate_end`. Works identically for both recorders.
2. **Per episode:** `final_scene` rebuilt from `final_positions` with per-orange
   `in_plate` and an authoritative `placed_oranges` identity set matching
   `oranges_in_plate`. Keep the gripper-gated `result` as-is (still drives the
   orchestrator), but it is no longer the success signal for analysis.

## Insertion points

- `eval_utils.py`: add `scene_geometry(positions)` (operates on a
  `save_positions()` dict, matching `count_oranges_in_plate`) and a tracker-based
  variant for mid-episode snapshots; `EpisodeStory.start_attempt/finish_attempt`
  attach the latest snapshot; `_scene` records `in_plate`; bump SCHEMA_VERSION.
- `phase_monitor.py` (monotask): same snapshot attach in `update`,
  `_start_attempt/_finish_attempt`; `_final_scene` already geometric — add
  `in_plate`; bump SCHEMA_VERSION.
- `inference_autonomous_orders.py`: pass `final_positions` (not the live
  post-step positions) to `finish_story_episode` at all three call sites; push a
  per-step scene snapshot into `episode_story`.
