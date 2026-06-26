# Eval logging rewrite — handoff for the Isaac machine

**Status: written and committed (`1582c8d`) on a laptop with NO Isaac/torch, so
it is NOT runtime-tested. Please verify on the desktop and fix as needed.**

## Goal

Make subtask **failures and timeouts decidable directly from geometry**
("is the orange in the plate?") for **both** the monotask (`PhaseMonitor`) and
subtask (orchestrator/`EpisodeStory`) runs, so the report analyses stop relying
on brittle string-matching / heuristic reconstruction.

Constraints agreed with the user:
- **Logging-only** — no control-flow changes, so a seeded rerun
  (`pick_orange_reference_100_v1`) must reproduce *identical* trajectories,
  just with richer fields. If you find a change that alters behaviour, that is a
  bug — flag it.
- **No subtask timeout for monotask** (it has no per-subtask budget).

## What was changed

### `eval_utils.py`
- **New** `scene_geometry(positions, orange_names=...)` — from a
  `save_positions()` dict, returns
  `{plate_position, plate_upside_down, n_in_plate, oranges:{name:{in_plate, position}}}`.
  Uses the *same* tilt-aware test as `count_oranges_in_plate`
  (`is_orange_position_in_plate` + `is_plate_upside_down`), so a recorded scene
  always agrees with `oranges_in_plate`.
- **`EpisodeStory`** (`SCHEMA_VERSION` 1 → 2):
  - new `self._scene_now`, method `note_scene(scene)`.
  - `start_attempt` stores `scene_start`; `finish_attempt` stores `scene_end` and
    derives `target_in_plate_end` (the attempt's actual/requested orange's
    `in_plate` at end).
  - `_scene` and `build_record` take an optional `in_plate_map` to add per-orange
    `in_plate` on the final scene.

### `phase_monitor.py` (`PhaseMonitor`, `SCHEMA_VERSION` 2 → 3)
- `reset` adds `self._scene_now`.
- `update` sets `self._scene_now = scene_geometry({"plate":plate_pos, "plate_quat":tracker._plate_quat, **orange_positions})` each step.
- `_start_attempt` / `_finish_attempt` attach `scene_start` / `scene_end` /
  `target_in_plate_end`.
- `build_record` re-snapshots from `final_positions` before finishing the active
  attempt (so the episode-end attempt is not the auto-reset state).
- `_scene` adds per-orange `in_plate`.

### `inference_autonomous_orders.py`
- **Bug fix:** `finish_story_episode` now takes `(final_positions, orange_names)`
  and builds `final_scene` + placed identities + `in_plate` from that **same
  pre-reset snapshot** used for `oranges_in_plate`. Previously it was passed the
  live post-step positions, which on truncation are the **auto-reset** state
  (that is why `final_scene` positions disagreed with the count). Three call
  sites updated (two manual-stop/reset, one normal end).
- Per step: `episode_story.note_scene(scene_geometry(last_positions, ...))`
  (uses the pre-step snapshot, which is never the auto-reset state).
- Imports `scene_geometry`.

## New per-attempt fields (both recorders)

```
"scene_start": <scene_geometry dict or null>,
"scene_end":   <scene_geometry dict or null>,
"target_in_plate_end": true | false | null   # null if the orange is unknown
```
`final_scene.oranges[name]` now also has `"in_plate": bool`.

## Verified on the laptop (no Isaac)
- `python -m py_compile` passes for all three files.
- None-safety: `scene_geometry` with `plate_quat=None` works
  (`plate_up_vector_z(None)→1.0`); this path is exercised by the existing
  `test_phase_monitor.py` via `final_positions={"plate_quat":None,...}`.
- All `build_record` / `finish_story_episode` call sites (incl. the monotask
  entry points `inference_flat_prompt.py`, `inference_act_flat_prompt.py`) pass
  `save_positions`-style dicts; changes are additive keys only.

## NOT verified — please check on the desktop
1. **Unit tests:** `python -m pytest test_phase_monitor.py` (or however it is run).
   Additive fields should not break existing assertions, but confirm.
2. **Behaviour preserved:** run a few episodes and confirm
   `oranges_in_plate`, `result`/`failure_reason`, and the trajectory are
   unchanged vs the previous build (logging-only).
3. **`final_scene` consistency (the bug fix):** on a *truncated* episode, check
   that `final_scene.n_in_plate` / placed `status` now match
   `episode_summary.final_oranges_in_plate`, and that positions are near the
   plate. Previously they were the reset state.
4. **`target_in_plate_end` sanity:** for a few PLACE attempts, confirm a recorded
   `timeout` whose orange was actually in the plate now has
   `target_in_plate_end == true` (the gripper-retraction artefact), and a real
   drop has `false`.
5. **Monotask snapshots:** confirm `scene_start`/`scene_end` populate for
   `PhaseMonitor` attempts and the episode-end attempt is not the reset state.

## Possible edge cases to review
- The orchestrator snapshot uses the **pre-step** `last_positions`; on the exact
  step a place/lift *succeeds* the orange may have entered the plate mid-step, so
  `scene_end` could lag by one step on a success. This is intended (successes are
  identified by `result`; `target_in_plate_end` matters for failures/timeouts,
  where the in-plate state is stable). If you prefer post-step accuracy, snapshot
  from a guarded post-step `save_positions` (skip on terminated/truncated).
- `fallback_scene()` in `inference_autonomous_orders.py` is now unused (its two
  call sites were removed). Safe to delete if you want.

## After verifying
- **Re-run** the seeded eval to populate the new fields (the committed
  checkpoints predate this). The two subtask models are the priority; monotask
  too if clean per-attempt scenes are wanted.
- Then the report scripts (`report/scripts/compute_place_success.py`, the lift
  analysis) can read `target_in_plate_end` / `final_scene[...].in_plate` directly
  with no post-processing — ping the laptop agent to rewrite them against v2/v3.

---

# Schema v4 — per-step `geometry_trace` (monotask / `PhaseMonitor` only)

**Status: implemented + smoke-tested on the desktop. Logging-only — verified a
seeded rerun reproduces identical initial scenes (oranges+camera, max|Δ|=0).**

## Why
The monotask (flat) policy names no target, so "which orange is it trying to grasp"
was unanswerable from the checkpoint (the live `PhaseMonitor` target flickers — it is
re-guessed every ~10 frames). The report left **"Oranges tried" = pending** and could
not test single-orange **fixation**. More generally, several monotask labels were
heuristic and not robust: the `argmax`-of-contact-bodies force grasp confirmation, the
proximity-based "held" (tip within 8 cm), and the arbitrary per-subtask timeout budgets.

## What was added (`phase_monitor.py`, `SCHEMA_VERSION` 3 → 4)
A per-step **raw** geometry trace, so every classification (target / held / grasp /
lift / place / timeout) becomes **offline post-processing** — a future idea is a script
change, never an eval rerun.

- `PhaseMonitor.__init__` reads env knobs: `TELEMETRY_TRACE` (default on),
  `TELEMETRY_EVERY_STEPS` (default 3, downsample), `TELEMETRY_PROXIMITY_M`
  (default 0.20 m, gate).
- `reset()` adds `self.geometry_trace = []`.
- `_record_geometry(...)` (called at the top of `update()`, reusing the tuple
  `_get_env_data` already returns) appends one rounded frame, **downsampled** by
  `every_steps` and **gated** to frames where some unplaced orange's grip-axis distance
  is `< proximity_m` **or** phase ≠ SEARCHING (far idle-search carries no aim info).
- `build_record()` emits a top-level `"geometry_trace"` key, which flows into the
  episode record via `inference_flat_prompt.py`'s `record.update(phase_debug)` — **no
  change to `inference_flat_prompt.py`**.

### Per-frame fields
```
{
  "step", "phase",
  "gripper_tip":[x,y,z], "jaw_tip":[x,y,z],     # grip axis -> any distance recomputable
  "gripper_pos",                                  # closure (joint value)
  "grip_force_n", "jaw_force_n",                  # raw per-tip contact force on the grip axis
  "plate":[x,y,z], "plate_quat":[w,x,y,z],
  "oranges": { name: {pos:[x,y,z], axis_dist, tip_dist, grip_axis_t, height_gain, in_plate} }
}
```
Raw positions/forces are the source of truth; distances are stored as convenience.

## Offline consumer
`report/scripts/compute_grasp_intent.py` reads `geometry_trace` and prints, with
**tunable** thresholds (`ENGAGE_M`, `MIN_ENGAGED_FRAMES`): distinct oranges **tried**/ep
(fills the `pending` cells in `experiments.tex`), **fixation** (distinct-engaged
distribution + share of engaged time on the top orange), and closest grip-axis approach.
Degrades gracefully (reports "no geometry_trace") on pre-v4 checkpoints.

## Size
Gated+downsampled+rounded; ~hundreds of frames/episode at the default gate (the smoke
run with the gate *disabled* was ~99 frames per 300-step episode → ~0.19 MB/episode).
Expect roughly +5–10 MB per 100-episode model with the default gate.

## Scope / caveats
Monotask only (subtask runs already name their target). The subtask force-confirmed
grasp / "held" metrics keep their existing documented caveat — not changed here.

## After verifying
Re-run the seeded monotask evals to populate `geometry_trace` (committed checkpoints
predate v4), then fill the `pending` cells in `experiments.tex` and the fixation caveat
in `results.tex` from `compute_grasp_intent.py`.
