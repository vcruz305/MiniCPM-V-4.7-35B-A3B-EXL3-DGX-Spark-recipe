#!/usr/bin/env bash
# Source this only from trusted recipe scripts. No environment is installed here.
_RECIPE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export RECIPE_ROOT="${RECIPE_ROOT:-$_RECIPE_ROOT}"
export RECIPE_HOME="${RECIPE_HOME:-$HOME/minicpm47-exl3}"
export MODEL_DIR="${MODEL_DIR:-$HOME/models/MiniCPM-V-4.7-35B-A3B-EXL3-SAGE-3.74bpw}"
export STATE_DIR="${STATE_DIR:-$RECIPE_ROOT/.state}"
export PROFILE="${PROFILE:-single}"
export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8898}"
