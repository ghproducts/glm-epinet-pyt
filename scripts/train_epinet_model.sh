#!/bin/bash
#set -euo pipefail

export PYTHONUNBUFFERED=1
cd ../

# ============================================================
# File to train epinet on top of frozen pretrained base
# ============================================================

# Paths
DATA="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/promoter_all"
BASE_CKPT="trained_models/model_1/checkpoint-1688" # checkpoint of your fine-tuned base model
EPI_CKPT="trained_models/model_1/epinet" #ckpt of output epinet model

# hyperparams
MODEL="DNABERT2"
SEED=2
LR=2e-5

# DNABERT: set the number as 0.25 * your sequence length. 
# NT: set the number as 1/6 * your sequence length
# hyenaDNA/CARMANIA: use full sequence length
MAX_LENGTH=75

echo "Begin training ${MODEL} epinet on seed: ${SEED}"

python -m nn_proj.models.${MODEL}.train_epinet \
    --data_path "${DATA}" \
    --checkpoint "${BASE_CKPT}" \
    --run_name "${MODEL}_epinet" \
    --model_max_length "${MAX_LENGTH}" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LR} \
    --num_train_epochs 2 \
    --fp16 \
    --output_dir "${EPI_CKPT}" \
    --eval_strategy epoch \
    --save_strategy epoch \
    --warmup_steps 50 \
    --logging_steps 100 \
    --overwrite_output_dir True \
    --log_level info \
    --find_unused_parameters False \
    --seed ${SEED} \
    --data_seed ${SEED}