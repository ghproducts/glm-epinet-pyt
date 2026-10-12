#!/bin/bash
# Score the epinet capacity-ablation checkpoints on the evaluation datasets.
#
# Each variant is written to its own cell label (conv_epinet_abl_<variant>) so it
# does not collide with the full-epinet conv_epinet cells.
#
#   GPU=1 bash run_ablation_eval.sh
#   VARIANTS="nopriors" DATASETS="boundary" GPU=1 bash run_ablation_eval.sh
set -uo pipefail

SW="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/scratch/home/glh52/venvs/aleatoric_boundary_venv/bin/python}"
BACKBONES="${BACKBONES:-NT_transformer,DNABERT2,hyenaDNA,CARMANIA}"
VARIANTS="${VARIANTS:-noprior noconv nopriors}"
DATASETS="${DATASETS:-boundary,alisim}"
SEEDS="${SEEDS:-1}"
GPU="${GPU:-0}"

for V in $VARIANTS; do
  echo "=== variant: $V ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$SW/run_sweep.py" \
    --backbones "$BACKBONES" --methods conv_epinet \
    --datasets "$DATASETS" --seeds "$SEEDS" \
    --epinet-kind "epinet_abl_${V}" 2>&1 \
    | grep -E "wrote .*acc=|FAILED|cells succeeded"
done
echo "ablation evaluation complete"
