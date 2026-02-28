#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_BASE_URL="${PYGAD_DATA_BASE_URL:-https://github.com/Migelo/pygad/releases/download/pygad-data}"
DATA_DIR="${ROOT_DIR}/data"
mkdir -p "${DATA_DIR}"

download_data_file() {
  local file="$1"
  local target="${DATA_DIR}/${file}"
  local tmp="${target}.tmp"
  local url="${DATA_BASE_URL%/}/${file}"

  if [ -s "${target}" ]; then
    echo "[setup] using cached ${target}"
    return 0
  fi

  echo "[setup] downloading ${url}"
  curl -fL --retry 3 --retry-delay 2 --connect-timeout 30 \
    --output "${tmp}" "${url}"
  mv "${tmp}" "${target}"
}

extract_if_missing() {
  local file="$1"
  local expected_path="$2"

  if [ -e "${ROOT_DIR}/pygad/${expected_path}" ]; then
    echo "[setup] pygad/${expected_path} already present; skipping extraction"
    return 0
  fi

  echo "[setup] extracting ${file} into pygad/"
  tar -xzf "${DATA_DIR}/${file}" -C "${ROOT_DIR}/pygad"
}

if [ "${PYGAD_SKIP_DATA_BOOTSTRAP:-0}" != "1" ]; then
  download_data_file "z_0.000_highres.tar.gz"
  download_data_file "iontbls.tar.gz"
  download_data_file "snaps.tar.gz"
  download_data_file "bc03.tar.gz"

  extract_if_missing "z_0.000_highres.tar.gz" "CoolingTables/z_0.000.hdf5"
  extract_if_missing "iontbls.tar.gz" "iontbls"
  extract_if_missing "snaps.tar.gz" "snaps"
  extract_if_missing "bc03.tar.gz" "bc03"
else
  echo "[setup] skipping data bootstrap (PYGAD_SKIP_DATA_BOOTSTRAP=1)"
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "[setup] creating virtual environment .venv"
  python -m venv .venv
fi

echo "[setup] installing Python dependencies into .venv"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install ipython ipykernel jupyter

echo "[setup] running import health check"
.venv/bin/python -c "import pygad; print('pygad', pygad.__version__)"

echo "[setup] done"
