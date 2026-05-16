#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${1:-${PROJECT_ROOT}/examples/example.env}"
OUTPUT_ROOT="${2:-${PROJECT_ROOT}/outputs}"
cd "${PROJECT_ROOT}"

if command -v litbase-ai >/dev/null 2>&1; then
  CMD=(litbase-ai)
else
  CMD=(python -m litbase_ai.cli)
fi

if grep -Eq 'your_email@example.com|your_llm_api_key_here' "${ENV_FILE}"; then
  echo "[WARN] ${ENV_FILE} still contains placeholder values. Doctor will still run, but API-backed features may be skipped."
fi

echo "[INFO] Running doctor"
"${CMD[@]}" doctor --env-file "${ENV_FILE}" --output-dir "${OUTPUT_ROOT}/doctor"

echo "[INFO] Running demo search"
"${CMD[@]}" search \
  --env-file "${ENV_FILE}" \
  --topic "climate change integrated assessment model" \
  --limit 20 \
  --year-from 2018 \
  --disable-cnki \
  --progress-style rich \
  --output-dir "${OUTPUT_ROOT}/demo"
