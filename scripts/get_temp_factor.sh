#!/bin/bash
#set -euo pipefail

# ============================================================
# File to calculate temperature scaling factor for an input dataset
# ============================================================

export PYTHONUNBUFFERED=1
cd ../

# Paths
DATA="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/promoter_all"
BASE_CKPT="trained_models/model_1/checkpoint-1688" 

# DNABERT: set the number as 0.25 * your sequence length. 
# NT: set the number as 1/6 * your sequence length
# hyenaDNA/CARMANIA: use full sequence length
MODEL="DNABERT2"
MAX_LENGTH=75

# make sure seed here is the same as the seed in the base net training
SEED=2


python -m nn_proj.models.${MODEL}.scaling \
    --data_path "${DATA}" \
    --checkpoint "${BASE_CKPT}" \
    --model_max_length "${MAX_LENGTH}" \
    --run_name "${MODEL}_temp_scaling" \
    --per_device_eval_batch_size 32 \
    --seed ${SEED} \
    --data_seed ${SEED}