#!/usr/bin/env bash
# Build, validate, and deploy Azul. --dry-run never writes to AWS.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
AZUL_PYTHON="${AZUL_PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
if [[ ! -x "$AZUL_PYTHON" ]]; then
    AZUL_PYTHON="$(command -v python3)"
fi
exec "$AZUL_PYTHON" -m infra.deploy "$@"
