#!/bin/bash
set -euo pipefail

# Configuration pour le streaming à distance. Callers may override these when
# launching concurrent Isaac jobs that need separate service ports.
export LIVESTREAM="${LIVESTREAM:-2}"
export ENABLE_LIVESTREAM="${ENABLE_LIVESTREAM:-1}"
export ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"

export LEISAAC_ASSETS_ROOT="$HOME/Documents/leisaac/assets"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <python-script> [args...]" >&2
    exit 2
fi

if [ -n "${REMOTE_LOG_FILE:-}" ]; then
    LOG_FILE="$REMOTE_LOG_FILE"
else
    LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_$(basename "${1%.py}").log"
fi
mkdir -p "$(dirname "$LOG_FILE")"
echo "Logging to $LOG_FILE"

cd "$SCRIPT_DIR"
PYTHON_CMD=(conda run --no-capture-output -n "${CONDA_ENV:-leisaac_envhub}" python "$@")
printf -v COMMAND '%q ' "${PYTHON_CMD[@]}"
script -q -e -f "$LOG_FILE" -c "$COMMAND"
