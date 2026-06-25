# MasterProject

Robotics research for a pick-and-place task: an SO-101 robot arm picks three oranges and places them on a plate in Isaac Sim, using a SmolVLA Vision-Language-Action policy driven by a scripted subtask orchestrator (GRASP / LIFT / PLACE). Built on Isaac Sim (via LeIsaac) + LeRobot.

## One machine

Everything now runs on a single basement desktop (RTX 5090): Isaac Sim evaluation, local SmolVLA training, and the report toolchain. The project historically spanned four machines (a Dell laptop, the room desktop, this desktop, and the EPFL SCITAS cluster); those are no longer used. **If a source file or dataset is missing here, it is on the room desktop or the Dell laptop** and should be copied over.

See **[WORKFLOW.md](WORKFLOW.md)** for how the project is driven (parallel agents, phone control, unattended runs) and **[WORKLOG.md](WORKLOG.md)** to coordinate active sessions.

## Repository structure

```
MasterProject/
├── isaac-inference/     ← Policy evaluation, orchestrator, dataset pipeline (Isaac Sim)
│   ├── inference_*.py         eval entry points (orchestrated + flat-prompt)
│   ├── eval_utils.py robot_utils.py phase_monitor.py dataset_recorder.py   shared libs
│   ├── remote.sh              Isaac Sim launcher
│   ├── dataset_pipeline/      one-shot dataset build/transform scripts
│   ├── maintenance/           diagnostic / repair / plotting utilities
│   ├── tests/                 unit tests
│   ├── docs/                  reference notes (commands.txt, …)
│   ├── results/              eval results (git-tracked)   ·   eval_seeds/
│   └── legacy/                old entry points
├── cluster-training/    ← SmolVLA training — now local on the 5090 (local_train.sh); SLURM scripts kept as reference
├── dataset-editor/      ← Manual dataset annotation GUI (editor.py) + lerobot_editor package
├── leisaac-mods/        ← Custom LeIsaac/Isaac modules (teleop devices, Quest3)
├── report/              ← Thesis (LaTeX main.tex + figure/analysis scripts/)
└── tooling/             ← Claude Code workflow scripts (worktrees, unattended, notifications)
```

Model weights and datasets are on the HuggingFace Hub under [`MasterProject2026`](https://huggingface.co/MasterProject2026), not in this repo. See the `CLAUDE.md` in each subdirectory for details and known issues.

## Quick start

**Evaluation (Isaac Sim)**
```bash
cd isaac-inference
./remote.sh inference_autonomous_orders.py
```

**Training (local, on the 5090)**
```bash
cd cluster-training
./local_train.sh          # SmolVLA fine-tune via lerobot-train on the local GPU
```

**Dataset editing**
```bash
cd dataset-editor
python editor.py MasterProject2026/my-dataset
```

**Report (LaTeX)**
```bash
cd report
export PATH=/home/students/texlive/2026/bin/x86_64-linux:$PATH
latexmk -pdf main.tex
```
