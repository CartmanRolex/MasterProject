#!/usr/bin/env bash
# notify.sh <type>   (type: waiting | done | <free text>)
#
# Centralized "which Claude needs me" notifier. Wired to Claude Code Stop /
# Notification hooks (see .claude/settings.json). Reads the hook's JSON event
# from stdin (fields: cwd, session_id, message, hook_event_name) and fans out:
#
#   1. Central log  -> $CC_NOTIFY_LOG (default ~/.claude/cc-waiting.log).
#      `tail -f ~/.claude/cc-waiting.log` in one pane = your dashboard across
#      every session. Reliable over SSH (no GUI needed).
#   2. tmux         -> flags the pane's window so the status bar shows activity.
#   3. Desktop      -> notify-send popup IF a GUI session is reachable.
#   4. Phone        -> ntfy.sh push IF $CC_NTFY_TOPIC is set and (we're an
#      unattended run OR $CC_NTFY_ALWAYS=1). ntfy topics are PUBLIC unless you
#      self-host: use a long random topic and never send secrets.
#
# Never fails the hook: always exits 0.

type="${1:-info}"
log="${CC_NOTIFY_LOG:-$HOME/.claude/cc-waiting.log}"

# --- parse the hook JSON on stdin (best effort; works without jq) ----------
payload="$(cat 2>/dev/null || true)"
read -r cwd message <<EOF
$(printf '%s' "$payload" | python3 -c '
import sys, json, os
try:
    d = json.load(sys.stdin)
except Exception:
    d = {}
cwd = d.get("cwd") or os.getcwd()
msg = (d.get("message") or "").replace("\n", " ")
print(cwd, msg)
' 2>/dev/null)
EOF
[ -n "${cwd:-}" ] || cwd="$(pwd)"
short="${cwd/#$HOME/~}"

case "$type" in
  waiting) icon="⏸"; label="waiting for input" ;;
  done)    icon="✅"; label="finished" ;;
  *)       icon="🔔"; label="$type" ;;
esac
line="$icon $(date '+%H:%M:%S')  $label  $short"
[ -n "$message" ] && line="$line  —  $message"

# 1. central log -----------------------------------------------------------
mkdir -p "$(dirname "$log")" 2>/dev/null
printf '%s\n' "$line" >>"$log" 2>/dev/null

# 2. tmux window flag + message -------------------------------------------
if [ -n "${TMUX:-}" ]; then
  tmux display-message -d 4000 "$line" 2>/dev/null
fi

# 3. desktop popup (only if a GUI is reachable) ----------------------------
if command -v notify-send >/dev/null 2>&1; then
  DISPLAY="${DISPLAY:-:0}" notify-send -u normal "Claude $label" "$short${message:+ — $message}" 2>/dev/null &
fi

# 4. phone push via ntfy ---------------------------------------------------
if [ -n "${CC_NTFY_TOPIC:-}" ] && { [ "${CC_UNATTENDED:-}" = "1" ] || [ "${CC_NTFY_ALWAYS:-}" = "1" ]; }; then
  curl -fsS --max-time 6 \
       -H "Title: Claude $label" \
       -H "Tags: robot" \
       -d "$short${message:+ — $message}" \
       "https://ntfy.sh/${CC_NTFY_TOPIC}" >/dev/null 2>&1 &
fi

exit 0
