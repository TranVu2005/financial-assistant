#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if command -v uv >/dev/null 2>&1; then
  exec uv run --frozen --no-sync financial-report-qa download-data "$@"
fi

if command -v python3 >/dev/null 2>&1; then
  export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  exec python3 scripts/download_dataset.py "$@"
fi

echo "error: install uv or Python 3.11 before running this script" >&2
exit 127
