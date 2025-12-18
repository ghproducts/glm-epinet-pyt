#!/bin/bash
#set -euo pipefail
(
    export CUDA_VISIBLE_DEVICES=6
    source ~/py311-pyt/bin/activate
    cd /scratch/home/glh52/glm-epinet

    # ===== Inputs / outputs =====
    INPUT_CSV="DATA/gene_taxa/test.csv"           # <- set to your eval CSV
    CHECKPOINT_DIR="models/dnabert2/gene/checkpoint-136882"     # <- matches your training output_dir
    OUT_DIR="pyt_inference/dnabert2/gene_taxa"
    mkdir -p "${OUT_DIR}"
    OUTPUT_CSV="${OUT_DIR}/test_mc.csv"
    SAVE_LOGITS_NPZ=""                             # optional: e.g., "${OUT_DIR}/test_logits.npz"

    # ===== Model / data knobs (mirror training) =====
    BASE_MODEL="zhihan1996/DNABERT-2-117M"
    MAX_LENGTH=750
    BATCH_SIZE=32
    KMER=-1
    K_SAMPLES=5    # set >1 to enable MC-dropout

    python pytorch_dnabert_epinet/inference_base.py \
    --checkpoint "${CHECKPOINT_DIR}" \
    --base_model "${BASE_MODEL}" \
    --input_csv "${INPUT_CSV}" \
    --output_csv "${OUTPUT_CSV}" \
    --batch_size ${BATCH_SIZE} \
    --max_len ${MAX_LENGTH} \
    --kmer ${KMER} \
    --k_samples ${K_SAMPLES} \
    --save_logits_npz "${SAVE_LOGITS_NPZ}"
) > test.log 2>&1 &

(
    export CUDA_VISIBLE_DEVICES=5
    source ~/py311-pyt/bin/activate
    cd /scratch/home/glh52/glm-epinet


    # ===== Inputs / outputs =====
    INPUT_CSV="DATA/gene_taxa/gene_out.csv"           # <- set to your eval CSV
    CHECKPOINT_DIR="models/dnabert2/gene/checkpoint-136882"     # <- matches your training output_dir
    OUT_DIR="pyt_inference/dnabert2/gene_taxa"
    mkdir -p "${OUT_DIR}"
    OUTPUT_CSV="${OUT_DIR}/gene_out_preds.csv"
    SAVE_LOGITS_NPZ=""                             # optional: e.g., "${OUT_DIR}/test_logits.npz"

    # ===== Model / data knobs (mirror training) =====
    BASE_MODEL="zhihan1996/DNABERT-2-117M"
    MAX_LENGTH=750
    BATCH_SIZE=32
    KMER=-1
    K_SAMPLES=1    # set >1 to enable MC-dropout

    python pytorch_dnabert_epinet/inference_base.py \
    --checkpoint "${CHECKPOINT_DIR}" \
    --base_model "${BASE_MODEL}" \
    --input_csv "${INPUT_CSV}" \
    --output_csv "${OUTPUT_CSV}" \
    --batch_size ${BATCH_SIZE} \
    --max_len ${MAX_LENGTH} \
    --kmer ${KMER} \
    --k_samples ${K_SAMPLES} \
    --save_logits_npz "${SAVE_LOGITS_NPZ}"



    # ===== Inputs / outputs =====
    INPUT_CSV="DATA/gene_taxa/gene_out.csv"           # <- set to your eval CSV
    CHECKPOINT_DIR="models/dnabert2/gene/checkpoint-136882"     # <- matches your training output_dir
    OUT_DIR="pyt_inference/dnabert2/gene_taxa"
    mkdir -p "${OUT_DIR}"
    OUTPUT_CSV="${OUT_DIR}/gene_out_mc.csv"

    # ===== Model / data knobs (mirror training) =====
    BASE_MODEL="zhihan1996/DNABERT-2-117M"
    MAX_LENGTH=750
    BATCH_SIZE=32
    KMER=-1
    K_SAMPLES=5    # set >1 to enable MC-dropout

    python pytorch_dnabert_epinet/inference_base.py \
    --checkpoint "${CHECKPOINT_DIR}" \
    --base_model "${BASE_MODEL}" \
    --input_csv "${INPUT_CSV}" \
    --output_csv "${OUTPUT_CSV}" \
    --batch_size ${BATCH_SIZE} \
    --max_len ${MAX_LENGTH} \
    --kmer ${KMER} \
    --k_samples ${K_SAMPLES} \
) > gene_out.log 2>&1 &


(
    export CUDA_VISIBLE_DEVICES=4
    source ~/py311-pyt/bin/activate
    cd /scratch/home/glh52/glm-epinet

    # ===== Inputs / outputs =====
    INPUT_CSV="DATA/gene_taxa/taxa_out.csv"           # <- set to your eval CSV
    CHECKPOINT_DIR="models/dnabert2/gene/checkpoint-136882"     # <- matches your training output_dir
    OUT_DIR="pyt_inference/dnabert2/gene_taxa"
    mkdir -p "${OUT_DIR}"
    OUTPUT_CSV="${OUT_DIR}/test_out_preds.csv"

    # ===== Model / data knobs (mirror training) =====
    BASE_MODEL="zhihan1996/DNABERT-2-117M"
    MAX_LENGTH=750
    BATCH_SIZE=32
    KMER=-1
    K_SAMPLES=1    # set >1 to enable MC-dropout

    python pytorch_dnabert_epinet/inference_base.py \
    --checkpoint "${CHECKPOINT_DIR}" \
    --base_model "${BASE_MODEL}" \
    --input_csv "${INPUT_CSV}" \
    --output_csv "${OUTPUT_CSV}" \
    --batch_size ${BATCH_SIZE} \
    --max_len ${MAX_LENGTH} \
    --kmer ${KMER} \
    --k_samples ${K_SAMPLES} \



    # ===== Inputs / outputs =====
    INPUT_CSV="DATA/gene_taxa/taxa_out.csv"           # <- set to your eval CSV
    CHECKPOINT_DIR="models/dnabert2/gene/checkpoint-136882"     # <- matches your training output_dir
    OUT_DIR="pyt_inference/dnabert2/gene_taxa"
    mkdir -p "${OUT_DIR}"
    OUTPUT_CSV="${OUT_DIR}/test_out_mc.csv"

    # ===== Model / data knobs (mirror training) =====
    BASE_MODEL="zhihan1996/DNABERT-2-117M"
    MAX_LENGTH=750
    BATCH_SIZE=32
    KMER=-1
    K_SAMPLES=5    # set >1 to enable MC-dropout

    python pytorch_dnabert_epinet/inference_base.py \
    --checkpoint "${CHECKPOINT_DIR}" \
    --base_model "${BASE_MODEL}" \
    --input_csv "${INPUT_CSV}" \
    --output_csv "${OUTPUT_CSV}" \
    --batch_size ${BATCH_SIZE} \
    --max_len ${MAX_LENGTH} \
    --kmer ${KMER} \
    --k_samples ${K_SAMPLES} \
) > taxa_out.log 2>&1 &

wait