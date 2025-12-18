#!/bin/bash
set -euo pipefail

# Activate env and cd to project
source ~/py311-pyt/bin/activate
cd /scratch/home/glh52/temp-glm-epinet

# Shared config
CKPT_ROOT="trained_models"
OUT_ROOT="inference_results"
K_SAMPLES=1   # number of MC dropout samples (adjust as needed)

mkdir -p logs "$OUT_ROOT"

###############################################################################
# PBSIM (long reads) — MC Dropout inference only
###############################################################################
(
    export CUDA_VISIBLE_DEVICES=9

    # Model/data args
    MODEL=DNABERT2
    NUM_CLASSES=1344
    MAX_LENGTH=1500
    DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/pbsim"
    BASE_CHECKPOINT_PATH="${CKPT_ROOT}/base/${MODEL}/pbsim/checkpoint-90684"

    # Output (separate subdir to avoid clobbering epinet outputs)
    OUT_DIR="${OUT_ROOT}/${MODEL}/pbsim_base"
    mkdir -p "$OUT_DIR"

    # Tasks
    declare -a TASKS=(
    "partial"
    )
    #"test"
    #"witheld"     # note: intentional spelling per your dataset
    #"partial"
    #)

    # Run inference over each task file
    for TEST_FILE in "${TASKS[@]}"; do
    echo "PBSIM | MC Dropout | ${TEST_FILE}"
    python -m nn_proj.models.$MODEL.inference \
        --data_path  "${DATA_PATH}/${TEST_FILE}.csv" \
        --checkpoint "${BASE_CHECKPOINT_PATH}" \
        --model_max_length "${MAX_LENGTH}" \
        --output_dir "${OUT_DIR}/${TEST_FILE}" \
        --uncertainty_method None \
        --num_samples "${K_SAMPLES}" \
        --num_labels "${NUM_CLASSES}" \
        --per_device_eval_batch_size 8
    done
) > "logs/pbsim_base_partial_dnabert.log" 2>&1 & pid2=$!

(
    export CUDA_VISIBLE_DEVICES=8

    # NT transformer
    # Model/data args
    MODEL=NT_transformer
    NUM_CLASSES=1344
    MAX_LENGTH=1000
    DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/pbsim"
    BASE_CHECKPOINT_PATH="${CKPT_ROOT}/base/${MODEL}/pbsim/checkpoint-90684"

    # Output (separate subdir to avoid clobbering epinet outputs)
    OUT_DIR="${OUT_ROOT}/${MODEL}/pbsim_base"
    mkdir -p "$OUT_DIR"

    # Tasks
    declare -a TASKS=(
    "partial"
    )
    #"test"
    #"witheld"     # note: intentional spelling per your dataset
    #)

    # Run inference over each task file
    for TEST_FILE in "${TASKS[@]}"; do
    echo "PBSIM | MC Dropout | ${TEST_FILE}"
    python -m nn_proj.models.$MODEL.inference \
        --data_path  "${DATA_PATH}/${TEST_FILE}.csv" \
        --checkpoint "${BASE_CHECKPOINT_PATH}" \
        --model_max_length "${MAX_LENGTH}" \
        --output_dir "${OUT_DIR}/${TEST_FILE}" \
        --uncertainty_method None \
        --num_samples "${K_SAMPLES}" \
        --num_labels "${NUM_CLASSES}" \
        --per_device_eval_batch_size 8
    done

) > "logs/pbsim_base_partial_NT.log" 2>&1 & pid1=$!

###############################################################################
# gene_taxa — MC Dropout inference only
###############################################################################
# (
#     export CUDA_VISIBLE_DEVICES=6
# 
#     # Model/data args
#     MODEL=DNABERT2
#     NUM_CLASSES=437
#     MAX_LENGTH=750
#     DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/gene_taxa"
#     BASE_CHECKPOINT_PATH="${CKPT_ROOT}/base/${MODEL}/gene_taxa/checkpoint-123194"
# 
#     # Output (separate subdir to avoid clobbering epinet outputs)
#     OUT_DIR="${OUT_ROOT}/${MODEL}/gene_taxa_base"
#     mkdir -p "$OUT_DIR"
# 
#     # Tasks
#     declare -a TASKS=(
#     "test"
#     "taxa_out"
#     "gene_out"
#     )
# 
#     # Run inference over each task file
#     for TEST_FILE in "${TASKS[@]}"; do
#     echo "gene_taxa | MC Dropout | ${TEST_FILE}"
#     python -m nn_proj.models.$MODEL.inference \
#         --data_path  "${DATA_PATH}/${TEST_FILE}.csv" \
#         --checkpoint "${BASE_CHECKPOINT_PATH}" \
#         --model_max_length "${MAX_LENGTH}" \
#         --output_dir "${OUT_DIR}/${TEST_FILE}" \
#         --uncertainty_method None \
#         --num_samples "${K_SAMPLES}" \
#         --num_labels "${NUM_CLASSES}" \
#         --per_device_eval_batch_size 8
#     done
# 
# ) > "logs/gene_taxa_base_dnabert.log" 2>&1 & pid2=$!
# 
# (
#     export CUDA_VISIBLE_DEVICES=7
# 
#     # Model/data args
#     MODEL=NT_transformer
#     NUM_CLASSES=437
#     MAX_LENGTH=500
#     DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/gene_taxa"
#     BASE_CHECKPOINT_PATH="${CKPT_ROOT}/base/${MODEL}/gene_taxa/checkpoint-123194"
# 
#     # Output (separate subdir to avoid clobbering epinet outputs)
#     OUT_DIR="${OUT_ROOT}/${MODEL}/gene_taxa_base"
#     mkdir -p "$OUT_DIR"
# 
#     # Tasks
#     declare -a TASKS=(
#     "test"
#     "taxa_out"
#     "gene_out"
#     )
# 
#     # Run inference over each task file
#     for TEST_FILE in "${TASKS[@]}"; do
#     echo "gene_taxa | MC Dropout | ${TEST_FILE}"
#     python -m nn_proj.models.$MODEL.inference \
#         --data_path  "${DATA_PATH}/${TEST_FILE}.csv" \
#         --checkpoint "${BASE_CHECKPOINT_PATH}" \
#         --model_max_length "${MAX_LENGTH}" \
#         --output_dir "${OUT_DIR}/${TEST_FILE}" \
#         --uncertainty_method None \
#         --num_samples "${K_SAMPLES}" \
#         --num_labels "${NUM_CLASSES}" \
#         --per_device_eval_batch_size 8
#     done

# ) > "logs/gene_taxa_base_NT.log" 2>&1 & pid1=$!

# Propagate failures from background jobs
wait "$pid1" "$pid2"
echo "MC Dropout inference completed for PBSIM and gene_taxa."
