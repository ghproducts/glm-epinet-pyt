#!/bin/bash
# Retrain epinet heads under the per-example-z fix, for every backbone/seed.
#
# The existing `epinet` checkpoints all predate commit fb38241 (the z fix), so
# they were trained with one z shared across each batch. Evaluating them with
# the fixed code measures neither pipeline, which is why run_sweep.py refuses
# them. Output goes to a sibling `epinet_zfix` dir; the originals are left
# untouched for provenance.
#
# Resumable: a seed/backbone whose epinet_zfix already has weights is skipped.
#
#   bash retrain_epinet.sh                      # all backbones, seeds 1,2,3
#   BACKBONES="hyenaDNA" SEEDS="1" bash retrain_epinet.sh
#   GPU=4 bash retrain_epinet.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_ROOT="${GLM_CKPT_ROOT:-/scratch/home/glh52/glm-epinet-pyt/checkpoints}"
DATA="${DATA:-/scratch/home/glh52/glm-epinet-pyt/data_gen/label_noise/csv_data_r00/train.csv}"
PY="${PY:-/scratch/home/glh52/venvs/aleatoric_boundary_venv/bin/python}"
BACKBONES="${BACKBONES:-DNABERT2 NT_transformer hyenaDNA CARMANIA}"
SEEDS="${SEEDS:-1 2 3}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-2}"
GPU="${GPU:-0}"
LOGDIR="${LOGDIR:-$REPO/data_gen/uq_sweep/retrain_logs}"

# model_max_length per backbone: seq_length(300) * tokens_per_base
max_length_for() {
  case "$1" in
    NT_transformer) echo 50  ;;   # 6-mer  -> 1/6
    DNABERT2)       echo 75  ;;   # BPE    -> ~1/4
    hyenaDNA)       echo 300 ;;   # char-level
    CARMANIA)       echo 300 ;;   # char-level
    *) echo "unknown backbone $1" >&2; return 1 ;;
  esac
}

mkdir -p "$LOGDIR"
cd "$REPO" || exit 1
[ -f "$DATA" ] || { echo "FATAL: training data not found: $DATA" >&2; exit 1; }

# transformers renamed this arg: <=4.4x wants --evaluation_strategy, newer wants
# --eval_strategy. config.py declares `eval_strategy`, so on an older
# transformers the flag is accepted but silently ignored, evaluation_strategy
# stays "no", and load_best_model_at_end then hard-fails on the strategy
# mismatch. Detect the real field name instead of guessing.
EVAL_FLAG=$("$PY" - <<'EOF'
import dataclasses, transformers
f = {x.name for x in dataclasses.fields(transformers.TrainingArguments)}
print("--evaluation_strategy" if "evaluation_strategy" in f else "--eval_strategy")
EOF
)
echo "using $EVAL_FLAG (transformers $("$PY" -c 'import transformers;print(transformers.__version__)'))"

fail=0
for M in $BACKBONES; do
  ML=$(max_length_for "$M") || exit 1
  for S in $SEEDS; do
    BASE="$CKPT_ROOT/seed_$S/$M/label_noise_r00/base"
    OUT="$CKPT_ROOT/seed_$S/$M/label_noise_r00/epinet_zfix"
    LOG="$LOGDIR/${M}_seed${S}.log"

    if [ ! -d "$BASE" ]; then
      echo "[skip] $M seed $S: no base checkpoint at $BASE"; continue
    fi
    if [ -f "$OUT/model_epinet.pt" ] || [ -f "$OUT/model.safetensors" ]; then
      echo "[skip] $M seed $S: epinet_zfix already trained"; continue
    fi

    echo "[run ] $M seed $S (max_length=$ML) -> $OUT   log: $LOG"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" -m "nn_proj.models.${M}.train_epinet" \
      --data_path "$DATA" \
      --checkpoint "$BASE" \
      --run_name "${M}_epinet_zfix_seed${S}" \
      --model_max_length "$ML" \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 16 \
      --gradient_accumulation_steps 1 \
      --learning_rate "$LR" \
      --num_train_epochs "$EPOCHS" \
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
      --seed "$S" \
      --data_seed "$S" > "$LOG" 2>&1

    if [ -f "$OUT/model_epinet.pt" ] || [ -f "$OUT/model.safetensors" ]; then
      echo "[ ok ] $M seed $S"
      grep -oE "eval_accuracy[^,}]*" "$LOG" | tail -1 || true
    else
      echo "[FAIL] $M seed $S -- see $LOG"; tail -5 "$LOG"; fail=1
    fi
  done
done

echo
[ "$fail" -eq 0 ] && echo "all requested epinet_zfix checkpoints present" \
                  || { echo "one or more retrains FAILED"; exit 1; }
