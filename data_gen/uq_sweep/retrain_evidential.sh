#!/bin/bash
# Train evidential heads for the seeds the sweep is missing.
#
# Hyperparameters match the original DNABERT2 evidential run exactly (recovered
# from its training_args.bin): lr 2e-5, batch 64, 6 epochs, fp16, warmup 50,
# weight_decay 0.01, softplus/mse/annealing_step 3.
#
# 6 epochs is kept even though load_best_model_at_end always selects epoch 1
# (eval_loss rises after): num_train_epochs sets the LR decay horizon, so
# shortening it would change the epoch-1 weights and make seeds
# non-comparable.
#
#   bash retrain_evidential.sh                    # all backbones, seeds 2,3
#   SEEDS="2" BACKBONES="hyenaDNA" GPU=9 bash retrain_evidential.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_ROOT="${GLM_CKPT_ROOT:-/scratch/home/glh52/glm-epinet-pyt/checkpoints}"
DATA="${DATA:-/scratch/home/glh52/glm-epinet-pyt/data_gen/label_noise/csv_data_r00/train.csv}"
PY="${PY:-/scratch/home/glh52/venvs/aleatoric_boundary_venv/bin/python}"
BACKBONES="${BACKBONES:-DNABERT2 NT_transformer hyenaDNA CARMANIA}"
SEEDS="${SEEDS:-2 3}"
EPOCHS="${EPOCHS:-6}"
GPU="${GPU:-0}"
LOGDIR="${LOGDIR:-$REPO/data_gen/uq_sweep/retrain_logs}"

max_length_for() {
  case "$1" in
    NT_transformer) echo 50  ;;
    DNABERT2)       echo 75  ;;
    hyenaDNA)       echo 300 ;;
    CARMANIA)       echo 300 ;;
    *) echo "unknown backbone $1" >&2; return 1 ;;
  esac
}

mkdir -p "$LOGDIR"
cd "$REPO" || exit 1
[ -f "$DATA" ] || { echo "FATAL: training data not found: $DATA" >&2; exit 1; }

# see retrain_epinet.sh: transformers 4.30.2 wants --evaluation_strategy, and
# config.py's `eval_strategy` field is silently ignored there
EVAL_FLAG=$("$PY" - <<'EOF'
import dataclasses, transformers
f = {x.name for x in dataclasses.fields(transformers.TrainingArguments)}
print("--evaluation_strategy" if "evaluation_strategy" in f else "--eval_strategy")
EOF
)

fail=0
for M in $BACKBONES; do
  ML=$(max_length_for "$M") || exit 1
  for S in $SEEDS; do
    OUT="$CKPT_ROOT/seed_$S/$M/label_noise_r00/evidential"
    LOG="$LOGDIR/${M}_evidential_seed${S}.log"
    if [ -f "$OUT/pytorch_model.bin" ] || [ -f "$OUT/model.safetensors" ]; then
      echo "[skip] $M seed $S: evidential already trained"; continue
    fi
    echo "[run ] $M evidential seed $S (max_length=$ML) -> $OUT"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" -m "nn_proj.models.${M}.train_evidential" \
      --data_path "$DATA" \
      --run_name "${M}_evidential_seed${S}" \
      --model_max_length "$ML" \
      --per_device_train_batch_size 64 \
      --per_device_eval_batch_size 64 \
      --gradient_accumulation_steps 1 \
      --learning_rate 2e-5 \
      --num_train_epochs "$EPOCHS" \
      --weight_decay 0.01 \
      --fp16 \
      --output_dir "$OUT" \
      "$EVAL_FLAG" epoch \
      --save_strategy epoch \
      --warmup_steps 50 \
      --logging_steps 100 \
      --overwrite_output_dir True \
      --find_unused_parameters False \
      --save_model True \
      --evidence_activation softplus \
      --evidential_loss_type mse \
      --annealing_step 3 \
      --seed "$S" --data_seed "$S" > "$LOG" 2>&1

    if [ -f "$OUT/pytorch_model.bin" ] || [ -f "$OUT/model.safetensors" ]; then
      echo "[ ok ] $M evidential seed $S"
      grep -oE "'eval_accuracy': [0-9.]+" "$LOG" | tail -1 || true
    else
      echo "[FAIL] $M evidential seed $S -- see $LOG"; tail -5 "$LOG"; fail=1
    fi
  done
done

echo
[ "$fail" -eq 0 ] && echo "all requested evidential checkpoints present" \
                  || { echo "one or more evidential runs FAILED"; exit 1; }
