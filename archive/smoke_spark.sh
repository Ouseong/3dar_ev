#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:?Usage: bash smoke_spark.sh /absolute/path/to/3DAR_EV [output_directory]}"
RUN_OUTPUT="${2:-${PACKAGE_DIR}/runs/smoke_seed42}"
cd -- "$PACKAGE_DIR"
python -c 'import torch; assert torch.cuda.is_available(), "Use Spark CUDA, not the laptop."'
# This IS training: run only on the authorized Spark host.
python -u -m ev3dar run --data-root "$DATA_ROOT" --out "$RUN_OUTPUT" --seed 42 --epochs 1 --config configs/ev.json
