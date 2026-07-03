# Data Dictionary — datasets & models

The historical names (`Gal-pick-orange-tailedCH20`, `Gal_split_nolang`, …) are **opaque and
easy to confuse**, and there are too many references to rename them safely. This file is the
source of truth for *what each artifact actually is*. All live under the HuggingFace org
[`MasterProject2026`](https://huggingface.co/MasterProject2026). Figures are read from each
dataset's `meta/info.json` and each model's `train_config.json` / eval `results/`.

## Report aliases (read this first)

The thesis talks in terms of **training source × task formulation**. The mapping to real names:

| Report alias | Dataset | Model (frozen) | Full-task success |
|---|---|---|---|
| **Teleop, subtask** | `Gal_split_tailed` | `Gal-pick-orange-tailedCH20` | 20% (mean 1.77) |
| **Teleop, monotask** | `Gal_split_nolang` | `Gal_split_nolang` | 18% (mean 1.11) |
| **Teleop+Auto, subtask** | `Gal-merged-tailed-auto` | `Gal-merged-tailed-auto` | 13% (mean 1.42) |
| **Teleop+Auto, monotask** | `Gal-merged-tailed-auto-no-lang-no-home` | `…-no-lang-no-home` | 26% (mean 1.50) |
| **LightwheelAI baseline** | `leisaac-pick-orange-mimic-v0` (upstream) | `pick-orange-mimic` (SmolVLA), `ACT-pick-orange` (ACT) | 41% / — |

"Subtask" = orchestrated GRASP/LIFT/PLACE with per-subtask language prompts. "Monotask" =
end-to-end under one fixed prompt ("Place the orange into plate"), no orchestrator.

## Datasets

Flags: **source** (teleop = hand-collected via gamepad/Quest3; auto = autonomously generated
successful subtasks; merged = teleop+auto). **lang** = per-subtask instructions kept; **nolang**
= all labels collapsed to one prompt. **tail** = 20 frozen "hold" frames appended per episode
(idle-on-completion). **home** = contains scripted "Go back to start position" episodes.

| Dataset | source | seg. | lang | tail | home | eps | frames | Notes |
|---|---|---|---|---|---|---|---|---|
| `Gal_split` | teleop | subtask | lang | no | yes | 921 | 98 380 | Raw teleop subtask recording (also a few full-task / go-back prompts). Parent of the three below. |
| `Gal_split_tailed` | teleop | subtask | lang | **yes** | no | 846 | 104 886 | Tailing dropped 75 "go back" eps and appended 20 hold frames/ep. **Flagship subtask set.** |
| `Gal_split_notail` | teleop | subtask | lang | **no** | no | 846 | 87 966 | `Gal_split_tailed` with the 20 tail frames removed, language kept. Isolates the tail. *(new)* |
| `Gal_split_nolang` | teleop | **monotask** | nolang | no | no | 846 | 87 966 | `Gal_split_tailed` with language stripped **and** tails removed. The Teleop monotask set. |
| `Gal-auto-subtasks2` | auto | subtask | lang | yes | yes | 676 | 140 219 | First autonomous-generation run (not used directly in the report). |
| `Gal-auto-subtasks3-balanced` | auto | subtask | lang | yes | yes | 414 | 84 045 | Balanced Auto subset — the **Auto** half merged into Teleop+Auto. |
| `Gal-merged-tailed-auto` | merged | subtask | lang | yes | yes | 1 260 | 188 931 | 846 Teleop + 414 Auto. **Teleop+Auto subtask set.** |
| `Gal-merged-tailed-auto-no-lang` | merged | monotask | nolang | yes | yes | 1 335 | 164 557 | Language-stripped merge, home kept. Trained but **not** evaluated in the report. |
| `Gal-merged-tailed-auto-no-lang-no-home` | merged | **monotask** | nolang | no | no | 1 260 | 155 357 | Language-stripped, tails + 75 "go back" eps removed. Teleop+Auto monotask set. |

`leisaac-pick-orange-mimic-v0` (LightwheelAI, upstream) — 60 full-task demonstrations, the
official reference dataset; not in this org. Single global prompt, no subtask segmentation.

## Models

Recipe: **frozen** = parameter-efficient (vision encoder + LM backbone frozen, only action
expert + state projection trained, `train_expert_only=true`); **unfrozen-VLM** = the 16-layer
language backbone is also trained (`train_expert_only=false`); **unfrozen-all** = everything
trained including the vision encoder (`train_expert_only=false`, `freeze_vision_encoder=false`),
and — unlike every other model — batch size 32 with 40k gradient steps instead of 64/20k
(the 450M parameters did not fit in VRAM at batch 64; same samples seen, twice the weight
updates). All SmolVLA models start from `lerobot/smolvla_base`; chunk = action-chunk size. Report surface names: frozen =
*standard*, unfrozen-VLM = *LM-tuned*, unfrozen-all = *fully-tuned*.

| Model | policy | formulation | recipe | chunk | trained on | full% / mean |
|---|---|---|---|---|---|---|
| `Gal-pick-orange-tailedCH20` | SmolVLA | subtask | frozen | 20 | `Gal_split_tailed` | 20% / 1.77 |
| `Gal-pick-orange-tailedCH20-unfrozen-vlm` | SmolVLA | subtask | unfrozen-VLM | 20 | `Gal_split_tailed` | 33% / 1.88 |
| `Gal-pick-orange-tailedCH20-unfrozen-all` | SmolVLA | subtask | unfrozen-all | 20 | `Gal_split_tailed` | **58% / 2.32** |
| `Gal-pick-orange-notailCH20` | SmolVLA | subtask | frozen | 20 | `Gal_split_notail` | 32% / 1.81 |
| `Gal_split_nolang` | SmolVLA | monotask | frozen | 20 | `Gal_split_nolang` | 14% / 1.02 |
| `Gal_split_nolang-unfrozen-vlm` | SmolVLA | monotask | unfrozen-VLM | 20 | `Gal_split_nolang` | 9% / 0.92 |
| `Gal-merged-tailed-auto` | SmolVLA | subtask | frozen | 20 | `Gal-merged-tailed-auto` | 13% / 1.42 |
| `Gal-merged-tailed-auto-unfrozen-vlm` | SmolVLA | subtask | unfrozen-VLM | 20 | `Gal-merged-tailed-auto` | 10% / 1.37 |
| `Gal-merged-tailed-auto-no-lang-no-home` | SmolVLA | monotask | frozen | 20 | `…-no-lang-no-home` | 18% / 1.29 |
| `…-no-lang-no-home-unfrozen-vlm` | SmolVLA | monotask | unfrozen-VLM | 20 | `…-no-lang-no-home` | 25% / 1.54 |
| `pick-orange-mimic` | SmolVLA | monotask | frozen | — | `leisaac-pick-orange-mimic-v0` | 41% / 1.95 |
| `ACT-pick-orange` | ACT | monotask | — | 20/100 | `leisaac-pick-orange-mimic-v0` | — |

Earlier/intermediate models kept for provenance (not in the report): `Gal-pick-orange`,
`Gal-pick-orangeCH20` (trained on `Gal_split`), `Gal-pick-orange-tailed` (chunk 50),
`Gal-merged-tailed-auto-no-lang`, `Gal-auto-subtasks2/3-balanced`.

## Naming convention (for future artifacts)

Existing names are frozen, but **new** datasets/models should be self-describing. Recommended:

```
<source>-<segmentation>[-<flags>]          datasets   e.g. teleop-subtask-notail
smolvla-<source>-<segmentation>[-<flags>]  models     e.g. smolvla-teleop-subtask-unfrozen
```
where `source ∈ {teleop, auto, teleop+auto, lightwheel}`, `segmentation ∈ {subtask, monotask}`,
and `flags` spell out any non-default of `{nolang, notail, nohome, unfrozen, ch50, …}`. When a
new artifact must mirror an old one for clarity (as `Gal_split_notail` mirrors `Gal_split_tailed`),
keep the stem and add a row here so the mapping stays unambiguous.
