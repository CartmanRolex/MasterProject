# Tooling

Cross-cutting workflow scripts for running Claude Code on this desktop:
parallel worktrees, unattended tmux sessions, and centralized notifications.
Machine-agnostic; everything now runs on the basement desktop (see root `CLAUDE.md`).

## Scripts

| File | Purpose |
|------|---------|
| `cc-worktree.sh` | Manage git worktrees: `new <name> [--with-data]`, `list`, `rm <name>`. Each worktree is an isolated checkout on branch `cc/<name>` so parallel sessions never clobber each other. `--with-data` symlinks the big gitignored data dirs (`isaac-inference/teleop-datasets`, `synthetic_datasets`) into the worktree for data-touching tasks. |
| `cc-unattended.sh` | `<name> "<prompt>" [--with-data]` — create a worktree and launch a detached tmux session `cc-<name>` running `claude --dangerously-skip-permissions`. Sets `CC_UNATTENDED=1` so notifications reach your phone. Attach + run `/rc` to steer from the Claude app. |
| `notify.sh` | `<waiting\|done\|text>` — hook target (see `.claude/settings.json`). Reads the hook JSON on stdin and fans out to: a central log (`$CC_NOTIFY_LOG`, default `~/.claude/cc-waiting.log`), tmux status, desktop `notify-send`, and ntfy.sh phone push (when `CC_NTFY_TOPIC` set and unattended/`CC_NTFY_ALWAYS`). Always exits 0. |

## "Which Claude is waiting?" — one place to look

```bash
tail -f ~/.claude/cc-waiting.log     # every session's waiting/done events
```
Reliable over SSH (no GUI needed). For phone alerts, `export CC_NTFY_TOPIC=<long-random>`
and subscribe to that topic in the ntfy app; topics are public, so don't send secrets.

## Environment variables

| Var | Effect |
|-----|--------|
| `CC_NOTIFY_LOG` | Central log path (default `~/.claude/cc-waiting.log`). |
| `CC_NTFY_TOPIC` | ntfy.sh topic for phone pushes. |
| `CC_UNATTENDED` | Set to `1` by `cc-unattended.sh`; gates phone pushes. |
| `CC_NTFY_ALWAYS` | Set to `1` to also push phone alerts from attended sessions. |

## Keeping this file current

Update this file **and** `AGENTS.md` in the same commit as any change to the
scripts (new flags, new scripts, renamed files). The `.md` and the code must always
agree. Always commit and push after changes — never leave the working tree dirty.

**Commit messages must not include "Co-Authored-By: Claude" or any AI attribution line.**
