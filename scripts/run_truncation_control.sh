#!/usr/bin/env bash
# Read-length truncation control.
#
# Darkmatter reads are ~150bp; the pbsim_family checkpoint was fine-tuned on
# ~6000bp reads. That shift is confounded with `up` (percent unknown) in the
# darkmatter analysis: the base model sits at 0.92-0.95 of maximum entropy on
# every darkmatter read, and we cannot tell whether that is novelty or simply
# short reads.
#
# This runs the SAME in-distribution split (pbsim id_novel_genus, novel genus
# but known family -- so no novelty at the family rank the head predicts)
# through the SAME checkpoint at two read lengths that differ in nothing else.
# If 150bp alone reproduces darkmatter-like saturation, read length is the
# cause and the darkmatter numbers say nothing specific about dark matter.
#
# Usage: cd scripts && bash run_truncation_control.sh
set -euo pipefail
cd ../
VENV=${VENV:-/scratch/home/glh52/venvs/glm_epinet_venv}
# shellcheck disable=SC1091
source "$VENV/bin/activate"

BASE_CKPT=old_models/trained_models_1/NT_transformer/pbsim_family/checkpoint-60708
EPI_CKPT=old_models/trained_models_1/conv_epinet/NT_transformer/pbsim_family/checkpoint-15177
TAXA_DF=/scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv
OUT_ROOT=results/truncation_control
LOG_DIR=logs/truncation_control
NUM_LABELS=429
MAX_LENGTH=1000
BATCH=64
SEED=1
mkdir -p "$LOG_DIR"

gpu=3
for LEN in 150 6000; do
    DATA=data_gen/pbsim/truncation_control/id_novel_genus_${LEN}bp.csv
    for SPEC in "base 1 base" "mc_dropout 10 mcd" "epinet 16 epinet"; do
        set -- $SPEC
        METHOD=$1; K=$2; TAG=$3
        EXTRA=()
        [ "$METHOD" = "epinet" ] && EXTRA=(--epinet_path "$EPI_CKPT")
        CUDA_VISIBLE_DEVICES=$gpu python -m nn_proj.models.NT_transformer.inference \
            --data_path "$DATA" --checkpoint "$BASE_CKPT" \
            --taxa_df "$TAXA_DF" --taxa_rank family \
            --model_max_length $MAX_LENGTH --uncertainty_method "$METHOD" \
            --num_samples $K --num_labels $NUM_LABELS \
            --per_device_eval_batch_size $BATCH --seed $SEED --data_seed $SEED \
            --run_name "trunc_${LEN}bp_${METHOD}" \
            --output_dir "$OUT_ROOT/${LEN}bp/$METHOD" \
            "${EXTRA[@]}" > "$LOG_DIR/${LEN}bp_${TAG}.log" 2>&1 &
        gpu=$((gpu+1))
    done
done
wait
echo "truncation control done -> $OUT_ROOT"
