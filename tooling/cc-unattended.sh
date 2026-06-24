#!/usr/bin/env bash
# cc-unattended.sh <name> "<task prompt>" [--with-data]
#
# Launch an unattended Claude in a detached tmux session, isolated on its own
# git worktree/branch (cc/<name>), running with bypass-permissions so it doesn't
# stall waiting for approvals. Notifications go to the central log + ntfy (set
# CC_NTFY_TOPIC first and subscribe to that topic in the ntfy app).
#
# After it starts, attach once and run /rc to steer it from your phone:
#     tmux attach -t cc-<name>
#
# Review its branch before merging:  git log cc/<name> ;  /code-review
set -euo pipefail

name="${1:-}"; prompt="${2:-}"
[ -n "$name" ] && [ -n "$prompt" ] || { sed -n '2,14p' "$0"; exit 1; }
with_data=""; [ "${3:-}" = "--with-data" ] && with_data="--with-data"

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
wt="$(dirname "$repo")/$(basename "$repo")-wt/$name"

# create the isolated worktree
"$here/cc-worktree.sh" new "$name" $with_data

if [ -z "${CC_NTFY_TOPIC:-}" ]; then
  echo "warning: CC_NTFY_TOPIC is not set — you won't get phone pushes."
  echo "         export CC_NTFY_TOPIC=<long-random-topic> and subscribe in the ntfy app."
fi

session="cc-$name"
tmux new-session -d -s "$session" -c "$wt"
tmux setenv -t "$session" CC_UNATTENDED 1
[ -n "${CC_NTFY_TOPIC:-}" ] && tmux setenv -t "$session" CC_NTFY_TOPIC "$CC_NTFY_TOPIC"
# fresh shell so the setenv vars are present, then launch claude
tmux send-keys -t "$session" "export CC_UNATTENDED=1 CC_NTFY_TOPIC='${CC_NTFY_TOPIC:-}'" C-m
tmux send-keys -t "$session" "claude --dangerously-skip-permissions $(printf '%q' "$prompt")" C-m

echo
echo "started unattended session: $session   (worktree: $wt, branch cc/$name)"
echo "  steer from phone:  tmux attach -t $session   then run  /rc"
echo "  watch progress:    tail -f \${CC_NOTIFY_LOG:-\$HOME/.claude/cc-waiting.log}"
echo "  when done:         review cc/$name, then  tooling/cc-worktree.sh rm $name"
