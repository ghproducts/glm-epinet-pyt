#!/usr/bin/env bash
# Retrain the pbsim taxonomic classifiers on the LEAKAGE-FREE training set.
#
# The existing checkpoints (old_models/trained_models_1/.../pbsim_*) were
# trained on the original train.csv, which contains every read later carved out
# into id_test.csv -- verified: 300/300 sampled id_test reads appear in it. Any
# evaluation using id_test as the in-distribution tier against those
# checkpoints is therefore scoring training data, which inflates ID accuracy
# and confidence and makes every ID-vs-novel AUROC optimistic.
#
# pbsim_id_test/train.csv (256,588 reads) has those reads removed, so a model
# trained here makes id_test a genuine holdout.
#
# Hyperparameters match the original run (scripts/full_experiments/
# run_pyt_train_base_models.sh): 6kb reads -> model_max_length 1000,
# LR 1e-5, per-device batch 8, 2 epochs.
set -euo pipefail
cd ../
VENV=${VENV:-/scratch/home/glh52/venvs/glm_epinet_venv}
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RANK=${1:?usage: retrain_pbsim_clean.sh <phylum|family> [gpu]}
GPU=${2:-0}
case "$RANK" in
  phylum) NUM_LABELS=36  ;;
  family) NUM_LABELS=429 ;;
  *) echo "rank must be phylum or family" >&2; exit 1 ;;
esac

DATA=/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test/train.csv
TAXA=/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test/full_basic_lineage.csv
OUT=checkpoints/pbsim_clean/${RANK}
LOG=logs/pbsim_clean
mkdir -p "$LOG" "$OUT"

echo "retraining ${RANK} (${NUM_LABELS} labels) on leakage-free data, gpu ${GPU}"
CUDA_VISIBLE_DEVICES=$GPU python -m nn_proj.models.NT_transformer.train_base \
    --data_path "$DATA" \
    --taxa_df "$TAXA" --taxa_rank "$RANK" \
    --num_labels $NUM_LABELS \
    --model_max_length 1000 \
    --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 --num_train_epochs 2 --fp16 \
    --eval_strategy epoch --save_strategy epoch --save_total_limit 1 \
    --load_best_model_at_end True \
    --warmup_steps 50 --logging_steps 500 --overwrite_output_dir True \
    --find_unused_parameters False \
    --run_name "pbsim_clean_${RANK}" --output_dir "$OUT" \
    --seed 1 --data_seed 1 > "$LOG/${RANK}.log" 2>&1
echo "done -> $OUT"
