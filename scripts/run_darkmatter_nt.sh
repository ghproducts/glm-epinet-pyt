#!/usr/bin/env bash
# Darkmatter OOD sweep with NT_transformer (PBsim-family checkpoint, seed 1).
#
# The 2026-09-07 version of this run was issued ad-hoc from a shell and only
# covered the 15 `unmatched_2` files that were in data_gen/darkmatter/ at the
# time; the other 46 files of the drop arrived 2026-09-14. This script is that
# run, re-issued over the whole drop (61 files), and committed so it does not
# have to be reconstructed from a transcript again.
#
# Prereq: data_gen/darkmatter/build_csv.py has been run to produce $DATA.
# Usage:  cd scripts && bash run_darkmatter_nt.sh
set -euo pipefail
cd ../

# The repo's pipeline runs in this venv; the base conda python has a
# libstdc++/CXXABI mismatch that breaks scipy on import.
VENV=${VENV:-/scratch/home/glh52/venvs/glm_epinet_venv}
# shellcheck disable=SC1091
source "$VENV/bin/activate"

DATA=data_gen/darkmatter/csv_data/darkmatter_reads_all.csv
BASE_CKPT=old_models/trained_models_1/NT_transformer/pbsim_family/checkpoint-60708
EPI_CKPT=old_models/trained_models_1/conv_epinet/NT_transformer/pbsim_family/checkpoint-15177
OUT_ROOT=results/darkmatter_nt_all
LOG_DIR=logs/darkmatter_nt_all

# NT_transformer is 6-mer tokenized, so MAX_LENGTH ~ 1/6 x sequence length;
# 1000 is what the 2026-09-07 run used and leaves headroom for these ~150bp
# reads. NUM_LABELS must match the pbsim_family checkpoint's head (429).
MAX_LENGTH=1000
NUM_LABELS=429
BATCH=64
SEED=1
MCD_SAMPLES=10     # K stochastic forward passes for mc_dropout
EPI_SAMPLES=16     # K epinet index samples

# One GPU per method so the three run concurrently; A100s 6-8 were idle.
GPU_BASE=6
GPU_MCD=7
GPU_EPI=8

mkdir -p "$LOG_DIR"

run () {  # run <gpu> <method> <logname> <extra args...>
    local gpu=$1 method=$2 log=$3; shift 3
    CUDA_VISIBLE_DEVICES=$gpu python -m nn_proj.models.NT_transformer.inference \
        --data_path "$DATA" \
        --checkpoint "$BASE_CKPT" \
        --model_max_length $MAX_LENGTH \
        --uncertainty_method "$method" \
        --num_labels $NUM_LABELS \
        --per_device_eval_batch_size $BATCH \
        --seed $SEED --data_seed $SEED \
        --run_name "darkmatter_${method}_nt_all" \
        --output_dir "$OUT_ROOT/$method" \
        "$@" \
        > "$LOG_DIR/$log.log" 2>&1
}

run $GPU_BASE base   base      --num_samples 1  &
run $GPU_MCD  mc_dropout mcd   --num_samples $MCD_SAMPLES &
run $GPU_EPI  epinet epinet    --num_samples $EPI_SAMPLES --epinet_path "$EPI_CKPT" &
wait

python data_gen/darkmatter/summarize_uncertainty.py \
    --results-dir "$OUT_ROOT" \
    --title "Darkmatter (all 61 files): uncertainty vs. \`up\` (NT_transformer, PBsim family, seed 1)"

# Distinct --out so the 2026-09-07 unmatched_2-only figure is not clobbered.
python data_gen/darkmatter/plot_epinet_scatter.py \
    --results-dir "$OUT_ROOT" \
    --out .docs/latex/figs/darkmatter_confidence_vs_up_all.pdf \
    --title "Darkmatter (all 61 files): epistemic uncertainty vs. % unknown
(NT_transformer, PBsim family, seed 1)"
