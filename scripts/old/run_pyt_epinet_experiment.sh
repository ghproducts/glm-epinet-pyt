#!/bin/bash
set -euo pipefail

source ~/py311-pyt/bin/activate
cd /scratch/home/glh52/temp-glm-epinet



(
    #PBSIM long reads
    export CUDA_VISIBLE_DEVICES=9
    K_SAMPLES=10                   
    MODEL=NT_transformer
    CKPT_ROOT="trained_models"
    OUT_ROOT="inference_results/epinet"

    mkdir -p logs "$CKPT_ROOT" "$OUT_ROOT"
    # model args
    NUM_CLASSES=1344
    MAX_LENGTH=1000
    LR=1e-5

    # data paths
    DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/pbsim"
    TRAIN_DATA="train.csv"

    BASE_CHECKPOINT_PATH="${CKPT_ROOT}/base/${MODEL}/pbsim/checkpoint-90684"
    EPINET_CHECKPOINT_PATH="${CKPT_ROOT}/epinet/${MODEL}/pbsim"
    OUT_DIR="${OUT_ROOT}/${MODEL}/pbsim"
    mkdir -p "$EPINET_CHECKPOINT_PATH" "$OUT_DIR"

    declare -a TASKS=(
    "partial"
    )

    # # train epinet
    # python -m nn_proj.models.$MODEL.train_epinet \
    #     --data_path "${DATA_PATH}/${TRAIN_DATA}"\
    #     --checkpoint "$BASE_CHECKPOINT_PATH" \
    #     --run_name "${MODEL}_$(basename "$DATA_PATH")" \
    #     --model_max_length "${MAX_LENGTH}" \
    #     --per_device_train_batch_size 32 \
    #     --per_device_eval_batch_size 16 \
    #     --gradient_accumulation_steps 1 \
    #     --learning_rate "${LR}" \
    #     --num_train_epochs 2 \
    #     --fp16 \
    #     --output_dir "$EPINET_CHECKPOINT_PATH" \
    #     --eval_strategy epoch \
    #     --save_strategy epoch \
    #     --warmup_steps 50 \
    #     --logging_steps 100 \
    #     --overwrite_output_dir True \
    #     --log_level info \
    #     --find_unused_parameters False

    # find latest checkpoint
    CKPT_DIR="$(ls -1d "$EPINET_CHECKPOINT_PATH"/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true)"
    if [[ -z "${CKPT_DIR}" ]]; then
        echo "ERROR: No checkpoints found in $EPINET_CHECKPOINT_PATH" >&2
        exit 1
    fi
    echo "Using checkpoint: $CKPT_DIR"

    # test epinet
    for TEST_FILE in "${TASKS[@]}"; do
        echo "Testing on $TEST_FILE"
        python -m nn_proj.models.$MODEL.inference \
            --data_path  "${DATA_PATH}/${TEST_FILE}.csv" \
            --checkpoint "${CKPT_DIR}" \
            --model_max_length "${MAX_LENGTH}" \
            --output_dir "${OUT_DIR}/${TEST_FILE}" \
            --uncertainty_method epinet \
            --num_samples "${K_SAMPLES}" \
            --num_labels "${NUM_CLASSES}" \
            --per_device_eval_batch_size 8
    done

) > "logs/pbsim_epinet_NT.log" 2>&1 & pid1=$!



(
    #PBSIM long reads
    export CUDA_VISIBLE_DEVICES=8

    K_SAMPLES=10                   
    MODEL=DNABERT2
    CKPT_ROOT="trained_models"
    OUT_ROOT="inference_results/epinet"

    mkdir -p logs "$CKPT_ROOT" "$OUT_ROOT"

    # model args
    NUM_CLASSES=1344
    MAX_LENGTH=1500
    LR=1e-5

    # data paths
    DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/pbsim"
    TRAIN_DATA="train.csv"

    BASE_CHECKPOINT_PATH="${CKPT_ROOT}/base/${MODEL}/pbsim/checkpoint-90684"
    EPINET_CHECKPOINT_PATH="${CKPT_ROOT}/epinet/${MODEL}/pbsim"
    OUT_DIR="${OUT_ROOT}/${MODEL}/pbsim"
    mkdir -p "$EPINET_CHECKPOINT_PATH" "$OUT_DIR"

    declare -a TASKS=(
    "partial"
    )

    # # train epinet
    # python -m nn_proj.models.$MODEL.train_epinet \
    #     --data_path "${DATA_PATH}/${TRAIN_DATA}"\
    #     --checkpoint "$BASE_CHECKPOINT_PATH" \
    #     --run_name "${MODEL}_$(basename "$DATA_PATH")" \
    #     --model_max_length "${MAX_LENGTH}" \
    #     --per_device_train_batch_size 32 \
    #     --per_device_eval_batch_size 16 \
    #     --gradient_accumulation_steps 1 \
    #     --learning_rate "${LR}" \
    #     --num_train_epochs 2 \
    #     --fp16 \
    #     --output_dir "$EPINET_CHECKPOINT_PATH" \
    #     --eval_strategy epoch \
    #     --save_strategy epoch \
    #     --warmup_steps 50 \
    #     --logging_steps 100 \
    #     --overwrite_output_dir True \
    #     --log_level info \
    #     --find_unused_parameters False

    # find latest checkpoint
    CKPT_DIR="$(ls -1d "$EPINET_CHECKPOINT_PATH"/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true)"
    if [[ -z "${CKPT_DIR}" ]]; then
        echo "ERROR: No checkpoints found in $EPINET_CHECKPOINT_PATH" >&2
        exit 1
    fi
    echo "Using checkpoint: $CKPT_DIR"

    # test epinet
    for TEST_FILE in "${TASKS[@]}"; do
        echo "Testing on $TEST_FILE"
        python -m nn_proj.models.$MODEL.inference \
            --data_path  "${DATA_PATH}/${TEST_FILE}.csv" \
            --checkpoint "${CKPT_DIR}" \
            --model_max_length "${MAX_LENGTH}" \
            --output_dir "${OUT_DIR}/${TEST_FILE}" \
            --uncertainty_method epinet \
            --num_samples "${K_SAMPLES}" \
            --num_labels "${NUM_CLASSES}" \
            --per_device_eval_batch_size 8
    done

) > "logs/pbsim_epinet_DNABERT2.log" 2>&1 & pid2=$!


# (
#     # gene_taxa
#     export CUDA_VISIBLE_DEVICES=6
#     # model args
#     NUM_CLASSES=437  
#     MAX_LENGTH=500
#     LR=1e-5
# 
#     # data paths
#     DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/gene_taxa"
#     TRAIN_DATA="train.csv"
# 
#     BASE_CHECKPOINT_PATH="${CKPT_ROOT}/base/${MODEL}/gene_taxa/checkpoint-123194"
#     EPINET_CHECKPOINT_PATH="${CKPT_ROOT}/epinet/${MODEL}/gene_taxa"
#     OUT_DIR="${OUT_ROOT}/${MODEL}/gene_taxa"
#     mkdir -p "$EPINET_CHECKPOINT_PATH" "$OUT_DIR"
# 
#     declare -a TASKS=(
#     "test"
#     "taxa_out"   
#     "gene_out"
#     )
# 
#     # # train epinet
#     # python -m nn_proj.models.$MODEL.train_epinet \
#     #     --data_path "${DATA_PATH}/${TRAIN_DATA}"\
#     #     --checkpoint "$BASE_CHECKPOINT_PATH" \
#     #     --run_name "${MODEL}_$(basename "$DATA_PATH")" \
#     #     --model_max_length "${MAX_LENGTH}" \
#     #     --per_device_train_batch_size 32 \
#     #     --per_device_eval_batch_size 16 \
#     #     --gradient_accumulation_steps 1 \
#     #     --learning_rate "${LR}" \
#     #     --num_train_epochs 2 \
#     #     --fp16 \
#     #     --output_dir "$EPINET_CHECKPOINT_PATH" \
#     #     --eval_strategy epoch \
#     #     --save_strategy epoch \
#     #     --warmup_steps 50 \
#     #     --logging_steps 100 \
#     #     --overwrite_output_dir True \
#     #     --log_level info \
#     #     --find_unused_parameters False
# 
#     # find latest checkpoint
#     CKPT_DIR="$(ls -1d "$EPINET_CHECKPOINT_PATH"/checkpoint-* 2>/dev/null | sort -V | tail -n1 || true)"
#     if [[ -z "${CKPT_DIR}" ]]; then
#         echo "ERROR: No checkpoints found in $EPINET_CHECKPOINT_PATH" >&2
#         exit 1
#     fi
#     echo "Using checkpoint: $CKPT_DIR"
# 
#     # test epinet
#     for TEST_FILE in "${TASKS[@]}"; do
#         echo "Testing on $TEST_FILE"
#         python -m nn_proj.models.$MODEL.inference \
#             --data_path  "${DATA_PATH}/${TEST_FILE}.csv" \
#             --checkpoint "${CKPT_DIR}" \
#             --model_max_length "${MAX_LENGTH}" \
#             --output_dir "${OUT_DIR}/${TEST_FILE}" \
#             --uncertainty_method epinet \
#             --num_samples "${K_SAMPLES}" \
#             --num_labels "${NUM_CLASSES}" \
#             --per_device_eval_batch_size 8
#     done
# 
# ) > "logs/gene_taxa_epinet_${MODEL}.log" 2>&1 & pid2=$!


wait "$pid1" "$pid2"
echo "Epinet experiment completed for PBSIM and gene_taxa."
