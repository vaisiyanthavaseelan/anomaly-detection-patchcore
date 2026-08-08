#!/usr/bin/env bash
# Runs the API locally (outside Docker) with the OpenMP env vars required on
# macOS to avoid a torch/faiss crash -- see src/env_setup.py.
set -euo pipefail
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
./venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
