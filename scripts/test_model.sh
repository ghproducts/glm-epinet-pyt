#!/bin/bash
#set -euo pipefail

export PYTHONUNBUFFERED=1
cd ../

# ============================================================
# File to evaluate trained model with UQ
# ============================================================


# Paths
DATA="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/promoter_all"
NUM_LABELS=2 # make sure to set num labels properly for the given dataset
BASE_CKPT="trained_models/model_1/checkpoint-1688" # If epinet is being run, make sure to point to the trained epinet model 
# BASE_CKPT="trained_models/model1/epinet/checkpoint-1688" # <- example epinet checkpoint
OUT_PATH="trained_models/model_1/inference_results"

# DNABERT: set the number as 0.25 * your sequence length. 
# NT: set the number as 1/6 * your sequence length
# hyenaDNA/CARMANIA: use full sequence length
MODEL="DNABERT2"
MAX_LENGTH=75
SEED=2

# UQ methods
# If using mc_dropout or epinet, make sure to set the K_samples
# if using temp scaling, make sure to set your your temp scaling param
# if running epinet, make sure that BASE_CKPT points to the epinet model
# UQ_method options:
# base, mc_dropout, epinet, base_scaled
UQ_method="base" 
TEMP=1 
K_SAMPLES=10


python -m nn_proj.models.${MODEL}.inference \
    --data_path "${DATA}" \
    --checkpoint "${BASE_CKPT}" \
    --model_max_length "${MAX_LENGTH}" \
    --temperature "${TEMP}" \
    --run_name "${MODEL}_${UQ_method}" \
    --per_device_eval_batch_size 32 \
    --num_samples "${K_SAMPLES}" \
    --uncertainty_method "${UQ_method}" \
    --num_labels "${NUM_LABELS}" \
    --output_dir "${OUT_PATH}" \
    --seed ${SEED} \
    --data_seed ${SEED}