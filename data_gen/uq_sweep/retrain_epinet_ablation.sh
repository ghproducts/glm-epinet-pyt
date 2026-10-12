#!/bin/bash
# Epinet capacity ablation (R2-7's remaining sub-item).
#
# The reviewer asks whether epinet's calibration gains come from epistemic
# modelling or just from the extra parameters and generic confidence shrinkage.
# The randomized priors are what make an epinet an ENN rather than an ordinary
# extra head, so zeroing them holds capacity fixed and removes the mechanism:
#
#   full      prior_scale=1 conv_prior_scale=1   (= the epinet_zfix baseline)
#   noprior   prior_scale=0 conv_prior_scale=1   MLP prior removed
#   noconv    prior_scale=1 conv_prior_scale=0   conv prior ensemble removed
#   nopriors  prior_scale=0 conv_prior_scale=0   both removed -> matched-capacity
#                                                control: same parameter count,
#                                                no randomized-prior mechanism
#
# If calibration gains survive `nopriors`, they are capacity/shrinkage, not
# epistemic modelling -- which is exactly what the reviewer suspects.
#
#   bash retrain_epinet_ablation.sh
#   BACKBONES="DNABERT2" VARIANTS="nopriors" GPU=8 bash retrain_epinet_ablation.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_ROOT="${GLM_CKPT_ROOT:-/scratch/home/glh52/glm-epinet-pyt/checkpoints}"
DATA="${DATA:-/scratch/home/glh52/glm-epinet-pyt/data_gen/label_noise/csv_data_r00/train.csv}"
PY="${PY:-/scratch/home/glh52/venvs/aleatoric_boundary_venv/bin/python}"
BACKBONES="${BACKBONES:-DNABERT2 NT_transformer hyenaDNA CARMANIA}"
VARIANTS="${VARIANTS:-noprior noconv nopriors}"
SEEDS="${SEEDS:-1}"
EPOCHS="${EPOCHS:-2}"
GPU="${GPU:-0}"
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

scales_for() {  # -> "<prior_scale> <conv_prior_scale>"
  case "$1" in
    noprior)  echo "0.0 1.0" ;;
    noconv)   echo "1.0 0.0" ;;
    nopriors) echo "0.0 0.0" ;;
    *) echo "unknown variant $1" >&2; return 1 ;;
  esac
}

mkdir -p "$LOGDIR"
cd "$REPO" || exit 1
[ -f "$DATA" ] || { echo "FATAL: training data not found: $DATA" >&2; exit 1; }

EVAL_FLAG=$("$PY" - <<'EOF'
import dataclasses, transformers
f = {x.name for x in dataclasses.fields(transformers.TrainingArguments)}
print("--evaluation_strategy" if "evaluation_strategy" in f else "--eval_strategy")
EOF
)

fail=0
for M in $BACKBONES; do
  ML=$(corrected_len "$M") || exit 1
  for V in $VARIANTS; do
    read -r PS CPS <<<"$(scales_for "$V")" || exit 1
    for S in $SEEDS; do
      BASE="$CKPT_ROOT/seed_$S/$M/label_noise_r00/base"
      OUT="$CKPT_ROOT/seed_$S/$M/label_noise_r00/epinet_abl_${V}"
      LOG="$LOGDIR/${M}_epinet_abl_${V}_seed${S}.log"
      [ -d "$BASE" ] || { echo "[skip] $M/$V seed $S: no base checkpoint"; continue; }
      if [ -f "$OUT/model_epinet.pt" ] || [ -f "$OUT/model.safetensors" ]; then
        echo "[skip] $M/$V seed $S: already trained"; continue
      fi
      echo "[run ] $M $V seed $S (prior_scale=$PS conv_prior_scale=$CPS)"
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" -m "nn_proj.models.${M}.train_epinet" \
        --data_path "$DATA" \
        --checkpoint "$BASE" \
        --run_name "${M}_epinet_abl_${V}_seed${S}" \
        --model_max_length "$ML" \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --learning_rate 2e-5 \
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
        --prior_scale "$PS" \
        --conv_prior_scale "$CPS" \
        --seed "$S" --data_seed "$S" > "$LOG" 2>&1

      if [ -f "$OUT/model_epinet.pt" ] || [ -f "$OUT/model.safetensors" ]; then
        echo "[ ok ] $M $V seed $S"
        grep -oE "'eval_accuracy': [0-9.]+" "$LOG" | tail -1 || true
      else
        echo "[FAIL] $M $V seed $S -- see $LOG"; tail -4 "$LOG"; fail=1
      fi
    done
  done
done

echo
[ "$fail" -eq 0 ] && echo "all requested ablation checkpoints present" \
                  || { echo "one or more ablation runs FAILED"; exit 1; }
