#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:?Usage: bash run_spark.sh /absolute/path/to/3DAR_EV [output_directory]}"
RUN_OUTPUT="${2:-${PACKAGE_DIR}/runs/baseline}"
cd -- "$PACKAGE_DIR"
python scripts/verify_environment.py
python -c 'import torch; assert torch.cuda.is_available(), "Use the Spark CUDA PyTorch environment; CPU fallback is disabled."'
python -m unittest discover -s tests -v
python -u -m ev3dar run --data-root "$DATA_ROOT" --out "$RUN_OUTPUT" --config configs/ev.json
