#!/bin/bash
# Retrain base checkpoints at the CORRECT model_max_length.
#
# configs/experiments.yaml derives model_max_length as seq_length *
# tokens_per_base, which omits special tokens. Measured over the train, test and
# both AliSim corpora (all 300bp), the real requirement is:
#
#   NT_transformer  52  (<cls> + 50 six-mers + eos)   -- was 50, lost 2 six-mers = 12bp
#   DNABERT2        77  (BPE, max observed)           -- was 75, truncated 2 test sequences
#   hyenaDNA       301  (300 chars + [SEP])           -- was 300, lost 1 base
#   CARMANIA       300  (exact, no special tokens)    -- was 300, correct already
#
# Output goes to a sibling `base_len<N>` directory so the originals survive for
# a like-for-like comparison. Resumable.
#
#   BACKBONES="NT_transformer" SEEDS="1" GPU=8 bash retrain_base.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_ROOT="${GLM_CKPT_ROOT:-/scratch/home/glh52/glm-epinet-pyt/checkpoints}"
DATA="${DATA:-/scratch/home/glh52/glm-epinet-pyt/data_gen/label_noise/csv_data_r00/train.csv}"
PY="${PY:-/scratch/home/glh52/venvs/aleatoric_boundary_venv/bin/python}"
BACKBONES="${BACKBONES:-NT_transformer hyenaDNA DNABERT2}"
SEEDS="${SEEDS:-1 2 3}"
GPU="${GPU:-0}"
# OUTNAME/MAXLEN let us add ensemble members matching the existing seeds 1-3
# (which use the pre-correction max_length); mixing lengths inside one
# ensemble would be a confound.
OUTNAME="${OUTNAME:-}"
MAXLEN="${MAXLEN:-}"
LOGDIR="${LOGDIR:-$REPO/data_gen/uq_sweep/retrain_logs}"

corrected_len() {
  case "$1" in
    NT_transformer) echo 52  ;;
    DNABERT2)       echo 77  ;;
    hyenaDNA)       echo 301 ;;
    CARMANIA)       echo 300 ;;
    *) echo "unknown backbone $1" >&2; return 1 ;;
  esac
}

mkdir -p "$LOGDIR"
cd "$REPO" || exit 1
[ -f "$DATA" ] || { echo "FATAL: training data not found: $DATA" >&2; exit 1; }

# transformers 4.30.2 (required for DNABERT2) wants --evaluation_strategy;
# config.py declares eval_strategy, which that version silently ignores
EVAL_FLAG=$("$PY" - <<'EOF'
import dataclasses, transformers
f = {x.name for x in dataclasses.fields(transformers.TrainingArguments)}
print("--evaluation_strategy" if "evaluation_strategy" in f else "--eval_strategy")
EOF
)

fail=0
for M in $BACKBONES; do
  ML=$(corrected_len "$M") || exit 1
  # MAXLEN overrides the per-backbone length; set it for one backbone at a time
  [ -n "$MAXLEN" ] && ML="$MAXLEN"
  for S in $SEEDS; do
    SUB="${OUTNAME:-base_len${ML}}"
    OUT="$CKPT_ROOT/seed_$S/$M/label_noise_r00/$SUB"
    LOG="$LOGDIR/${M}_${SUB}_seed${S}.log"
    if [ -f "$OUT/model.safetensors" ] || [ -f "$OUT/pytorch_model.bin" ] \
       || ls "$OUT"/checkpoint-*/pytorch_model.bin "$OUT"/checkpoint-*/model.safetensors >/dev/null 2>&1; then
      echo "[skip] $M seed $S: $SUB already trained"; continue
    fi
    echo "[run ] $M seed $S at model_max_length=$ML -> $OUT"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" -m "nn_proj.models.${M}.train_base" \
      --data_path "$DATA" \
      --run_name "${M}_${SUB}_seed${S}" \
      --model_max_length "$ML" \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 16 \
      --gradient_accumulation_steps 1 \
      --learning_rate 2e-5 \
      --num_train_epochs 2 \
      --fp16 \
      --output_dir "$OUT" \
      "$EVAL_FLAG" epoch \
      --save_strategy epoch \
      --warmup_steps 50 \
      --logging_steps 100 \
      --overwrite_output_dir True \
      --log_level info \
      --find_unused_parameters False \
      --save_model True \
      --seed "$S" --data_seed "$S" > "$LOG" 2>&1

    if [ -f "$OUT/model.safetensors" ] || [ -f "$OUT/pytorch_model.bin" ] \
       || ls "$OUT"/checkpoint-*/pytorch_model.bin "$OUT"/checkpoint-*/model.safetensors >/dev/null 2>&1; then
      echo "[ ok ] $M seed $S"
      grep -oE "'eval_accuracy': [0-9.]+" "$LOG" | tail -1 || true
    else
      echo "[FAIL] $M seed $S -- see $LOG"; tail -5 "$LOG"; fail=1
    fi
  done
done

echo
[ "$fail" -eq 0 ] && echo "all requested base_len checkpoints present" \
                  || { echo "one or more base retrains FAILED"; exit 1; }
