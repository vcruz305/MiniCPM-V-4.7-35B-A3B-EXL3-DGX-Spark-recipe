#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  exec python3 "$RECIPE_ROOT/tools/manage.py" preview "$@"
fi
exec python3 "$RECIPE_ROOT/tools/manage.py" serve "$@"
