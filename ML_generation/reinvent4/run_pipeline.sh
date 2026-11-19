#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="ML_generation/reinvent4/tl_v1"
PRIOR_FILE="ML_generation/reinvent4/priors/base.prior"
OUTPUT_CSV="data/generated_csv/reinvent4_tl_v1.csv"

r4_run() {
    conda run --no-capture-output -n reinvent4 "$@"
}

echo "=== [1/4] Checking GPU inside reinvent4 ==="
r4_run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('Using GPU:', torch.cuda.get_device_name(0))"

if [[ ! -f "$PRIOR_FILE" ]]; then
    echo "ERROR: Base prior not found at $PRIOR_FILE"
    exit 1
fi

echo "=== [2/4] Running Transfer Learning ==="
r4_run reinvent -l "$RUN_DIR/tl.log" "$RUN_DIR/transfer_learning.toml"

echo "=== [3/4] Running De Novo Sampling ==="
r4_run reinvent -l "$RUN_DIR/sampling.log" "$RUN_DIR/sampling.toml"

echo "=== [4/4] Done ==="
echo "Output saved to: $OUTPUT_CSV ($(($(wc -l < "$OUTPUT_CSV") - 1)) molecules)"
head -n 6 "$OUTPUT_CSV"
