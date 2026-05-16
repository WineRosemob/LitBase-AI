#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${LITBASE_CONDA_ENV:-litbase-ai}"
ENV_FILE="${DEEPSEEK_ENV_FILE:-${PROJECT_ROOT}/.env.deepseek}"

if [[ ! -f "${ENV_FILE}" ]]; then
  ENV_FILE="${PROJECT_ROOT}/examples/example.env"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Install Miniconda or Anaconda first."
  exit 1
fi

if [[ $# -eq 0 ]]; then
  TOPIC="integrated assessment model climate adaptation"
  LIMIT=30
  YEAR_FROM=2018
  OUTPUT_DIR="outputs/deepseek_run"
else
  TOPIC="${1}"
  LIMIT="${2:-30}"
  YEAR_FROM="${3:-2018}"
  OUTPUT_DIR="${4:-outputs/deepseek_run}"
fi

echo "[INFO] Running LitBase-AI with env file: ${ENV_FILE}"
conda run -n "${ENV_NAME}" \
  litbase-ai search \
  --env-file "${ENV_FILE}" \
  --topic "${TOPIC}" \
  --limit "${LIMIT}" \
  --year-from "${YEAR_FROM}" \
  --output-dir "${OUTPUT_DIR}"
