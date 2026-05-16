#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-litbase-ai}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE_TEMPLATE="${PROJECT_ROOT}/environment.yml"
TEMP_ENV_FILE="$(mktemp --suffix=.yml)"
trap 'rm -f "${TEMP_ENV_FILE}"' EXIT

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Install Miniconda or Anaconda first."
  exit 1
fi

sed "s/^name: .*/name: ${ENV_NAME}/" "${ENV_FILE_TEMPLATE}" > "${TEMP_ENV_FILE}"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Target conda environment: ${ENV_NAME}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Updating existing conda environment from environment.yml"
  if ! conda env update -f "${TEMP_ENV_FILE}" --prune; then
    echo "[WARN] conda env update failed; falling back to a minimal defaults-based refresh"
    conda install -n "${ENV_NAME}" --override-channels -c defaults python=3.11 pip -y
  fi
else
  echo "[INFO] Creating conda environment from environment.yml"
  if ! conda env create -f "${TEMP_ENV_FILE}"; then
    echo "[WARN] conda env create failed; falling back to a minimal defaults-based environment"
    if ! conda create -n "${ENV_NAME}" --override-channels -c defaults python=3.11 pip -y; then
      if [[ "${ENV_NAME}" != "litbase-ai" ]] && conda env list | awk '{print $1}' | grep -qx "litbase-ai"; then
        echo "[WARN] defaults-based create failed too; cloning local 'litbase-ai' environment as a last-resort fallback"
        conda create -n "${ENV_NAME}" --clone litbase-ai -y
      else
        echo "[ERROR] Unable to create conda environment via environment.yml, defaults, or local clone fallback."
        exit 1
      fi
    fi
  fi
fi

echo "[INFO] Upgrading pip"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip

echo "[INFO] Installing Python dependencies from requirements.txt"
conda run -n "${ENV_NAME}" python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

echo "[INFO] Installing project in editable mode"
conda run -n "${ENV_NAME}" python -m pip install -e "${PROJECT_ROOT}"

echo "[INFO] Installing Playwright Firefox browser"
conda run -n "${ENV_NAME}" python -m playwright install firefox

echo "[INFO] Running CLI health check"
conda run -n "${ENV_NAME}" litbase-ai --help >/dev/null

echo "[INFO] Setup completed successfully."
echo "[INFO] Next steps:"
echo "       conda run -n ${ENV_NAME} litbase-ai doctor --env-file examples/example.env --output-dir outputs/doctor"
echo "       conda run -n ${ENV_NAME} litbase-ai search --env-file examples/example.env --topic \"climate change integrated assessment model\" --limit 20 --year-from 2018 --disable-cnki --output-dir outputs/demo"
