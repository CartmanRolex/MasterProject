#!/bin/bash

# Configuration pour le streaming à distance
export LIVESTREAM=2
export ENABLE_CAMERAS=1

export LEISAAC_ASSETS_ROOT="$HOME/Documents/leisaac/assets"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_$(basename "${1%.py}").log"
echo "Logging to $LOG_FILE"

script -q -e -f "$LOG_FILE" -c "python $(printf '%q ' "$@")"
