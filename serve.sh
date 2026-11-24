#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then echo 'usage: bash serve.sh MODEL [--listen tcp://127.0.0.1:8770]' >&2; exit 2; fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
model="$1"; shift
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "${PYTHON:-python}" -B -m service.serve --model "$model" "$@"
