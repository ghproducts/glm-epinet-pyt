#!/usr/bin/env bash
# Leave-phyla-out experiment: train 10 NT_transformer models, each on a
# different 5-phylum subset, then evaluate every model on (a) held-out pbsim
# reads with known phylum (ground-truth positive control) and (b) the darkmatter
# `up` sets.
#
# The question: does the darkmatter `up` trend depend on WHICH phyla a model
# knows? If the trend is identical across label spaces the effect is
# compositional; if it varies with membership, the unknown fraction is drawn
# from phyla inside the label space -- the hypothesis under test.
#
# Everything runs at 150bp, matching the darkmatter reads, so unlike previous
# runs there is no train/test read-length shift.
#
# Usage: cd scripts && bash run_leave_phyla_out.sh [train|eval|all]
set -euo pipefail
cd ../
VENV=${VENV:-/scratch/home/glh52/venvs/glm_epinet_venv}
# shellcheck disable=SC1091
source "$VENV/bin/activate"

STAGE=${1:-all}
SUB_DIR=data_gen/leave_phyla_out/subsets
CKPT_ROOT=checkpoints/leave_phyla_out
OUT_ROOT=results/leave_phyla_out
LOG_DIR=logs/leave_phyla_out
EVAL_PBSIM=data_gen/leave_phyla_out/eval_pbsim_150bp.csv
EVAL_DM=data_gen/darkmatter/csv_data/darkmatter_reads_all.csv
N_SUBSETS=10
NUM_LABELS=5
# NT is 6-mer tokenized: 150bp -> ~25 tokens. 48 leaves headroom for specials.
MAX_LENGTH=48
LR=2e-5
EPOCHS=3
SEED=1
mkdir -p "$LOG_DIR" "$CKPT_ROOT" "$OUT_ROOT"

if [ "$STAGE" = "train" ] || [ "$STAGE" = "all" ]; then
  echo "=== training $N_SUBSETS models (one per GPU) ==="
  for i in $(seq 0 $((N_SUBSETS-1))); do
    CUDA_VISIBLE_DEVICES=$i python -m nn_proj.models.NT_transformer.train_base \
      --data_path "$SUB_DIR/subset_${i}.csv" \
      --num_labels $NUM_LABELS \
      --model_max_length $MAX_LENGTH \
      --per_device_train_batch_size 64 \
      --per_device_eval_batch_size 64 \
      --gradient_accumulation_steps 1 \
      --learning_rate $LR \
      --num_train_epochs $EPOCHS \
      --fp16 \
      --eval_strategy epoch --save_strategy epoch \
      --save_total_limit 1 --load_best_model_at_end True \
      --warmup_steps 50 --logging_steps 100 \
      --overwrite_output_dir True --find_unused_parameters False \
      --run_name "lpo_subset_${i}" \
      --output_dir "$CKPT_ROOT/subset_${i}" \
      --seed $SEED --data_seed $SEED \
      > "$LOG_DIR/train_subset_${i}.log" 2>&1 &
  done
  wait
  echo "training done"
fi

if [ "$STAGE" = "eval" ] || [ "$STAGE" = "all" ]; then
  echo "=== evaluating (base UQ only, per plan) ==="
  for i in $(seq 0 $((N_SUBSETS-1))); do
    CKPT="$(ls -1d "$CKPT_ROOT/subset_${i}"/checkpoint-* 2>/dev/null | sort -V | tail -n1)"
    [ -z "$CKPT" ] && { echo "no checkpoint for subset_$i, skipping"; continue; }
    (
      for SPEC in "pbsim $EVAL_PBSIM" "darkmatter $EVAL_DM"; do
        set -- $SPEC
        CUDA_VISIBLE_DEVICES=$i python -m nn_proj.models.NT_transformer.inference \
          --data_path "$2" --checkpoint "$CKPT" \
          --model_max_length $MAX_LENGTH --uncertainty_method base --num_samples 1 \
          --num_labels $NUM_LABELS --per_device_eval_batch_size 256 \
          --seed $SEED --data_seed $SEED \
          --run_name "lpo_${i}_$1" \
          --output_dir "$OUT_ROOT/subset_${i}/$1" \
          > "$LOG_DIR/eval_${i}_$1.log" 2>&1
      done
    ) &
  done
  wait
  echo "evaluation done -> $OUT_ROOT"
fi
