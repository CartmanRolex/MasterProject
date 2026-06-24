# Working with Claude Code on this project

Everything runs on **one machine now** — the basement desktop (RTX 5090), reached
over SSH. This guide is the playbook for getting the most out of Claude with a Max
account: how to parallelize, how to see which session needs you, how to drive Claude
from your phone, and how to leave runs going unattended.

---

## 0. Consolidation (one machine)

The project used to span four machines (Dell laptop; the room desktop where most
datasets were created; this basement desktop; the SCITAS cluster). **From now on
everything is here**, including SmolVLA training (SCITAS is dropped — it's slower
than this box).

Present on this desktop: all subdir code, the `outputs/train/` checkpoints (~50 GB),
and the LeRobot datasets in the HF cache (`~/.cache/huggingface/lerobot/MasterProject2026`,
~24 GB). **Not here:** the 236 GB raw `teleop-datasets/` HDF5 — it stayed on the room
desktop. You only need it to rebuild datasets from scratch; the trainable datasets are
already on the Hub/cache. **If anything else turns up missing, it's on the room desktop
or the Dell laptop** — `rsync` it over.

---

## 1. The model: one orchestrator + subagents, matched to task type

Not many co-equal terminals, not one magic terminal. Pick per task:

- **Autonomous, scoped, returns an artifact → delegate to a subagent.** It runs to
  completion and reports back; great for fan-out. Reusable ones live in
  `.claude/agents/`:
  - `report-plotter` — refresh `report/figures/*.pdf` / recompute eval tables.
  - `eval-analyzer` — read-only analysis of `isaac-inference/results/<model>/`.
  - `code-refactor` — independent code change, runs in its own worktree branch.
  Dispatch several at once (they run in parallel / in the background) from one
  **orchestrator** terminal; it tells you as each finishes — so for batch work you
  watch *one* terminal.
- **Interactive / hardware / GPU-in-the-loop → its own session.** Thesis prose,
  Quest3 teleop tuning, live Isaac Sim debugging. Subagents are fire-and-forget and
  can't be steered mid-task, so these want a real conversation.

| Task | How |
|------|-----|
| Regenerate figures / recompute tables | subagent `report-plotter` |
| Analyze / compare eval results | subagent `eval-analyzer` |
| Independent refactor / cleanup / small fix | subagent `code-refactor` (own worktree) |
| Broad "where is X" search | `Explore` subagent |
| Thesis writing (`report/`) | dedicated session |
| Teleop tuning (`leisaac-mods/`) | dedicated session (GPU) |
| Live Isaac Sim eval/debug (`isaac-inference/`) | dedicated session (GPU) |
| Local SmolVLA training | background job + `/loop` to watch (GPU) |

**GPU rule:** the single 5090 is shared by sim, teleop, and training — run only
**one** GPU-heavy job at a time. Non-GPU work (writing, analysis) can run alongside.

---

## 2. Parallel work without conflicts — git worktrees

Two sessions editing the same area will clobber each other in one working tree. Give
each its own worktree (separate checkout, own branch):

```bash
tooling/cc-worktree.sh new myfeature        # ../MasterProject-wt/myfeature on cc/myfeature
tooling/cc-worktree.sh new bigrun --with-data  # also symlinks the gitignored data dirs
tooling/cc-worktree.sh list
tooling/cc-worktree.sh rm myfeature         # removes worktree (+ branch if merged)
```

The `code-refactor` subagent does this automatically (`isolation: worktree`). Merge a
branch back only after `/code-review`. Note: a fresh worktree has **no gitignored
data** (datasets, checkpoints) unless you pass `--with-data` — so use worktrees for
code/writing/analysis and the main tree for live data-touching runs.

---

## 3. "Which Claude is waiting for me?" — push, not poll

Stop scanning terminals. Sessions announce themselves via the `Notification`/`Stop`
hooks (`.claude/settings.json` → `tooling/notify.sh`):

- **One dashboard (works over SSH):**
  ```bash
  tail -f ~/.claude/cc-waiting.log
  ```
  Every session's "⏸ waiting / ✅ done" events land here with the directory.
- **Phone push:** `export CC_NTFY_TOPIC=<long-random-string>` before starting a
  session and subscribe to that topic in the **ntfy** app (free). Unattended sessions
  push automatically; for attended ones add `export CC_NTFY_ALWAYS=1`.
  ⚠️ ntfy topics are public — use a long random name, never send secrets.
- **In tmux:** the hook flashes a status-bar message; turn on
  `set -g monitor-activity on` to flag windows that moved.
- **The orchestrator** is itself a dashboard for background subagents — it reports
  "task 2 done, task 3 needs a decision" without you visiting each.

---

## 4. Drive Claude from your phone — Remote Control

Steer a session running here from the Claude mobile app, from anywhere, no VPN and no
open ports (needs Claude Code ≥ 2.1.51; this box runs 2.1.187).

1. Install the **Claude app** (iOS/Android); sign in with the same account.
2. In the session (ideally inside tmux — see §5), run **`/rc`**.
3. Scan the QR code. You can now watch live, approve/reject edits, redirect, and
   monitor multiple attached sessions — including which one is waiting.

---

## 5. Leave it running unattended

Run inside **tmux** so it survives SSH/laptop disconnects, isolated on its own branch,
with bypass-permissions so it doesn't stall:

```bash
export CC_NTFY_TOPIC=<your-topic>          # so you get phone pings
tooling/cc-unattended.sh nightly "refactor eval_utils.py per the plan and run its tests"
# -> detached tmux session cc-nightly on branch cc/nightly
tmux attach -t cc-nightly     # then /rc to steer from phone
tail -f ~/.claude/cc-waiting.log
```

Work lands on `cc/nightly`, never `main` — review with `/code-review` before merging,
so an unattended mistake stays on a throwaway branch. Pre-authorize permissions (or
use bypass, as the launcher does) so a background session doesn't block on a prompt.

**Recurring autonomy:**
- `/loop` — repeat a task on an interval or self-paced (e.g. watch a long training run
  and ping when a checkpoint lands or it crashes).
- `/schedule` — cron-style routines (e.g. nightly: tail the training log, push a
  summary to ntfy, free the GPU).

---

## 6. Other habits worth keeping

- `/fewer-permission-prompts` — extend the allowlist from your real transcripts.
- Run long training / Isaac launches as **background jobs** so the session stays
  responsive and re-notifies on exit.
- **Plan mode** for anything structural; `/code-review` as the merge gate for branches.
- Keep `CLAUDE.md`/`AGENTS.md` mirrors in sync and update them in the same commit as
  code changes (existing repo policy). Commits never include an AI attribution line.
