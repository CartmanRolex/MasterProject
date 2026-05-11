# Project Notes — Enabling Recovery in VLA-Based Robotic Manipulation

Internal scratch pad. Do not copy directly into the report.

---

## Setup

- **Robot:** SO-101 arm
- **Environment:** Kitchen scene in Isaac Sim (privileged state information available)
- **Task:** Pick 3 oranges from the scene and place them in a plate
- **Base policy:** SmolVLA, fine-tuned per experiment

---

## Phase 1 — Baseline fine-tuning on public dataset

- Fine-tuned SmolVLA on a public HuggingFace dataset: 60 episodes, each a full 3-orange pick-and-place sequence.
- **Success rate ~30%**, comparable to ACT trained on the same data.
- Takeaway: scaling to a VLA without exploiting language conditioning yields no meaningful gain over ACT. The monolithic full-task framing leaves the language-conditioning capability unused.

---

## Phase 2 — Subtask decomposition on public dataset

- **Motivation:** A VLA's value is language conditioning. Decompose into subtasks ("grasp orange", "place in plate") so a higher-level orchestrator can sequence and interrupt on failure. Key failure mode: *phantom grasp* — gripper closes on air, then proceeds to place as if holding an orange.
- **Method:** Re-sliced 60 episodes → ~600 subtask episodes. Retrained SmolVLA.
- **Result:** No clear improvement (qualitative only — no formal success rate measured). Subtask mode is significantly slower per rollout, making evaluation expensive.
- **Diagnosed limitation:** Public dataset always picks oranges in the same order → no signal tying a language instruction to a specific orange → policy cannot be conditioned to target a particular orange. This is a prerequisite for the orchestrator vision in Phase 5.

---

## Phase 3 — Custom teleoperated dataset

- **Motivation:** Need a dataset that supports (1) targeted-grasp conditioning by relative position, (2) subtask-level episodes from the start, (3) diverse pick orders.
- **Teleoperation pipeline:** Built from scratch — Xbox controller → SO-101 in Isaac kitchen. Substantial engineering deliverable in its own right.
- **Dataset design:**
  - Each episode = one subtask (not full task). Advantages: easy to exclude failed teleoperation attempts; easy to annotate with relative-position instructions ("grasp the left orange", "grasp the middle orange", "grasp the right orange").
  - Pick order randomized across episodes → breaks spurious order correlation from public dataset.
- **Final size:** 921 subtask episodes.
- **Result:** SmolVLA fine-tuned on this dataset follows relative-position instructions and can target a specific orange. However, robustness is limited: in unseen configurations where it fails, retrying the same orange does not generally succeed (out-of-distribution, not noise). Redirecting to a different orange typically works. No formal success rate measured.

---

## Phase 4 — Orchestrator and success detection (current state)

- **Implementation:** Algorithmic orchestrator reading **privileged Isaac state** (not a VLM). Detects gripper occupancy after grasp and orange-in-plate after placement. Can later be swapped for a VLM, but privileged state avoids perception noise at this stage.
- **Capabilities:**
  1. **Phantom-grasp detection:** prevents proceeding to placement if gripper is empty after a grasp attempt.
  2. **Targeted redirection:** if a specific orange is repeatedly ungraspable, re-issue instruction pointing to a different orange.
- **Important framing:** Redirecting to a *different* orange is **not recovery** of the original goal. It is (a) a way to avoid getting stuck and improve task completion, and (b) a mechanism for diverse autonomous data collection (Phase 5).

---

## Phase 5 — Autonomous data generation (next step)

- **Motivation:** Manual teleoperation is the bottleneck on dataset size and diversity. Current model + orchestrator can run fully autonomous rollouts, including in unseen configurations.
- **Pipeline:** Fully autonomous. Orchestrator labels each subtask attempt as success/failure via Isaac privileged state. Only successful subtask trajectories are stored.
- **Why redirection matters here:** Without targeted-grasp control, an ungraspable orange forces an episode reset (wasted configuration). With redirection, the system pivots to a different orange in the same scene and continues collecting — improving data yield from otherwise-wasted configurations.
- **Goal:** Use auto-generated data to retrain SmolVLA for better generalization to out-of-distribution configurations, without further manual annotation.

---

## Deliverables to Date

1. Xbox-controller teleoperation pipeline for SO-101 in Isaac, with subtask segmentation and relative-position annotation.
2. 921-episode subtask-level dataset with relative-position language annotations and randomized pick orders.
3. SmolVLA fine-tuned on this dataset that follows relative-position instructions.
4. Algorithmic orchestrator (Isaac privileged state) that sequences subtasks, detects phantom grasps, and redirects targeting.
5. Fully autonomous data-generation pipeline that filters for successful subtasks.
