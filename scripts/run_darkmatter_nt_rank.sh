#!/usr/bin/env bash
# Darkmatter sweep with NT_transformer at a chosen taxonomic rank.
#
# Generalisation of run_darkmatter_nt.sh. The family run (429 labels) left the
# model pinned at 0.92-0.95 of maximum entropy on every darkmatter read, so the
# `up` gradient had no room to express itself. Coarser ranks have far fewer
# classes (phylum 36, class 83, order 183) and should keep the model off that
# ceiling, making it possible to tell a genuine null from a saturation artifact.
#
# Usage: cd scripts && bash run_darkmatter_nt_rank.sh <rank>
#        rank in {phylum, class, order, family}
set -euo pipefail
cd ../
VENV=${VENV:-/scratch/home/glh52/venvs/glm_epinet_venv}
# shellcheck disable=SC1091
source "$VENV/bin/activate"

RANK=${1:?usage: run_darkmatter_nt_rank.sh <phylum|class|order|family>}
case "$RANK" in
    phylum) NUM_LABELS=36  ;;
    class)  NUM_LABELS=83  ;;
    order)  NUM_LABELS=183 ;;
    family) NUM_LABELS=429 ;;
    *) echo "unknown rank: $RANK" >&2; exit 1 ;;
esac

DATA=data_gen/darkmatter/csv_data/darkmatter_reads_all.csv
BASE_CKPT=$(ls -1d old_models/trained_models_1/NT_transformer/pbsim_${RANK}/checkpoint-* | sort -V | tail -n1)
EPI_CKPT=$(ls -1d old_models/trained_models_1/conv_epinet/NT_transformer/pbsim_${RANK}/checkpoint-* | sort -V | tail -n1)
OUT_ROOT=results/darkmatter_nt_${RANK}
LOG_DIR=logs/darkmatter_nt_${RANK}

MAX_LENGTH=1000     # NT is 6-mer tokenized: ~1/6 x sequence length
BATCH=64
SEED=1
MCD_SAMPLES=10
EPI_SAMPLES=16
GPU_BASE=${GPU_BASE:-0}
GPU_MCD=${GPU_MCD:-1}
GPU_EPI=${GPU_EPI:-2}

echo "rank=$RANK labels=$NUM_LABELS base=$BASE_CKPT epinet=$EPI_CKPT"
mkdir -p "$LOG_DIR"

run () {
    local gpu=$1 method=$2 log=$3; shift 3
    CUDA_VISIBLE_DEVICES=$gpu python -m nn_proj.models.NT_transformer.inference \
        --data_path "$DATA" --checkpoint "$BASE_CKPT" \
        --model_max_length $MAX_LENGTH --uncertainty_method "$method" \
        --num_labels $NUM_LABELS --per_device_eval_batch_size $BATCH \
        --seed $SEED --data_seed $SEED \
        --run_name "darkmatter_${method}_nt_${RANK}" \
        --output_dir "$OUT_ROOT/$method" "$@" > "$LOG_DIR/$log.log" 2>&1
}

run $GPU_BASE base       base   --num_samples 1 &
run $GPU_MCD  mc_dropout mcd    --num_samples $MCD_SAMPLES &
run $GPU_EPI  epinet     epinet --num_samples $EPI_SAMPLES --epinet_path "$EPI_CKPT" &
wait

python data_gen/darkmatter/summarize_uncertainty.py --results-dir "$OUT_ROOT" \
    --title "Darkmatter (all 61 files): uncertainty vs. \`up\` (NT_transformer, PBsim ${RANK}, seed 1)"
