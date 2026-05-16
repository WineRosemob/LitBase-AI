#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if command -v litbase-ai >/dev/null 2>&1; then
  CMD=(litbase-ai)
else
  CMD=(python -m litbase_ai.cli)
fi

"${CMD[@]}" doctor --env-file examples/example.env --output-dir outputs/doctor
"${CMD[@]}" search \
  --env-file examples/example.env \
  --topic "climate change integrated assessment model" \
  --limit 20 \
  --year-from 2018 \
  --disable-cnki \
  --progress-style rich \
  --output-dir outputs/demo
