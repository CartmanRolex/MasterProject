#!/usr/bin/env bash
# cc-worktree.sh — manage git worktrees for parallel / unattended Claude work.
#
#   cc-worktree.sh new <name> [--with-data]   create ../MasterProject-wt/<name>
#                                             on a fresh branch cc/<name>
#   cc-worktree.sh list                       list worktrees
#   cc-worktree.sh rm <name>                  remove the worktree (+ branch if merged)
#
# --with-data symlinks the big gitignored data dirs from the main tree into the
# worktree, so a data-touching task can run there. Code/writing/analysis tasks
# do NOT need it.
set -euo pipefail

repo="$(git rev-parse --show-toplevel)"
wt_root="$(dirname "$repo")/$(basename "$repo")-wt"

# gitignored heavy dirs (relative to repo root) to symlink with --with-data
DATA_DIRS=(
  "isaac-inference/teleop-datasets"
  "isaac-inference/synthetic_datasets"
)

usage() { sed -n '2,12p' "$0"; exit 1; }

cmd="${1:-}"; shift || true
case "$cmd" in
  new)
    name="${1:-}"; [ -n "$name" ] || usage; shift || true
    with_data=0; [ "${1:-}" = "--with-data" ] && with_data=1
    dest="$wt_root/$name"
    git -C "$repo" worktree add "$dest" -b "cc/$name"
    if [ "$with_data" = 1 ]; then
      for d in "${DATA_DIRS[@]}"; do
        src="$repo/$d"
        if [ -e "$src" ]; then
          mkdir -p "$(dirname "$dest/$d")"
          ln -sfn "$src" "$dest/$d"
          echo "  linked $d -> $src"
        else
          echo "  skip $d (not present in main tree)"
        fi
      done
    fi
    echo "worktree ready: $dest  (branch cc/$name)"
    echo "cd '$dest' && claude"
    ;;
  list)
    git -C "$repo" worktree list
    ;;
  rm)
    name="${1:-}"; [ -n "$name" ] || usage
    dest="$wt_root/$name"
    git -C "$repo" worktree remove "$dest" --force
    if git -C "$repo" branch --merged main | grep -qx "  cc/$name"; then
      git -C "$repo" branch -d "cc/$name" && echo "deleted merged branch cc/$name"
    else
      echo "branch cc/$name kept (not merged into main; delete with: git branch -D cc/$name)"
    fi
    ;;
  *) usage ;;
esac
