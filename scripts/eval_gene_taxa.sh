#!/usr/bin/env bash
# Calibration + UQ evaluation on the Scorpio gene_taxa task (NT_transformer).
#
# Structure differs from pbsim in ways that decide what is runnable:
#   test      140,524  437 labels, all seen in train   ID            labels VALID
#   taxa_out   11,800  436 labels, all seen in train   novel taxon   labels VALID
#   gene_out   43,417  every row label=1 (a dummy;     novel gene    labels DUMMY
#                      label 1 is 0.25% of test)
# So accuracy/ECE are reported for test and taxa_out only. gene_out is still run
# because its uncertainty scores are needed for OOD detection, where no labels
# are used -- but any accuracy printed for it is meaningless, by construction.
#
# The two tiers are orthogonal novelty axes, which is the point of this task:
# taxa_out is covariate shift (same gene families, unseen organisms), gene_out
# is semantic shift (gene families absent from the label space entirely).
#
# Leakage check before writing this: 0/43,417 gene_out, 0/11,800 taxa_out and
# 166/140,524 (0.12%) test sequences appear verbatim in train.csv. The 0.12% are
# duplicate sequences in the source corpus, not a split bug -- left alone.
#
# model_max_length 500 = 3000bp / 6 (NT is 6-mer); sequences are ~930bp median,
# so ~all reads pass untruncated. No 6kb windowing as in pbsim.
#
# Tiers are capped at CAP reads, sampled ONCE with a fixed seed and reused by
# every method, so methods are compared on identical reads. taxa_out is under
# the cap and is used whole.
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-/scratch/home/glh52/venvs/glm_epinet_venv}
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SRC=/scratch/home/glh52/glm-epinet/DATA/gene_taxa
BASE=old_models/trained_models_1/NT_transformer/gene_taxa/checkpoint-30800
EPI=old_models/trained_models_1/conv_epinet/NT_transformer/gene_taxa/checkpoint-30799
N=437
MAXLEN=500
CAP=${CAP:-25000}
D=temp/gene_taxa_eval          # capped copies of the tiers live here
OUT=results/gene_taxa_eval
LOG=logs/gene_taxa_eval
STATUS=$LOG/status.txt
mkdir -p "$LOG" "$OUT" "$D"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

say "PREP capped tiers (cap=$CAP, seed 1) + temperature-fit split"
python - "$SRC" "$D" "$CAP" <<'PY'
import sys, pandas as pd
src, dst, cap = sys.argv[1], sys.argv[2], int(sys.argv[3])
from pathlib import Path
Path(dst).mkdir(parents=True, exist_ok=True)
for t in ["test", "taxa_out", "gene_out"]:
    d = pd.read_csv(f"{src}/{t}.csv")
    n0 = len(d)
    if len(d) > cap:
        d = d.sample(cap, random_state=1)
    d.to_csv(f"{dst}/{t}.csv", index=False)
    print(f"  {t:9s} {n0:>7d} -> {len(d):>6d}")
# Temperature is fit on a 20k slice of the TRAINING split, matching the pbsim
# protocol (scaling.py there also reads train.csv) so the two tasks' base_scaled
# numbers are produced the same way. Guo et al. fit on a validation split; there
# is no usable one here (dev.csv is 100 rows, all the dummy label), and fitting
# on held-out test would make ID calibration partly in-sample.
tr = pd.concat(list(pd.read_csv(f"{src}/train.csv", chunksize=100_000)), ignore_index=True)
tr.sample(20_000, random_state=1).to_csv(f"{dst}/train_20k.csv", index=False)
print(f"  train     {len(tr):>7d} ->  20000  (temperature fit only)")
PY

# GPU pool. scripts/eval_clean_all.sh (pbsim family) runs concurrently and also
# picks GPUs by free memory; with no pool both scripts choose the same idle
# devices and the doubled-up jobs OOM. Holding this run to the upper half leaves
# the lower half to that job, which then sees these as busy and picks 0-4.
POOL=${POOL:-5,6,7,8,9}
pick_gpus () {   # pick_gpus <count> <min_free_mb>
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
      | awk -F', ' -v need="$2" -v pool=",$POOL," \
            '{f=$3-$2; if (f>=need && index(pool,","$1",")) print f, $1}' \
      | sort -rn | head -n "$1" | awk '{print $2}'
}

say "FIT temperature"
T=$(CUDA_VISIBLE_DEVICES=$(pick_gpus 1 20000) \
    python -m nn_proj.models.NT_transformer.scaling \
      --data_path "$D/train_20k.csv" --checkpoint "$BASE" \
      --model_max_length $MAXLEN --num_labels $N \
      --per_device_eval_batch_size 32 --seed 1 --data_seed 1 \
      --output_dir "$OUT/_scaling" 2>"$LOG/scaling.err" \
    | grep -oE "^T = [0-9.]+" | awk '{print $3}')
if [ -z "${T:-}" ]; then say "FAILED temperature fit (see $LOG/scaling.err)"; T=1.0; fi
say "  T = $T"


for TIER in test taxa_out gene_out; do
  say "TIER $TIER"
  QUEUE=("base 1 base" "base 1 base_scaled" "mc_dropout 10 mc_dropout" "epinet 16 epinet")
  qi=0
  while [ "$qi" -lt "${#QUEUE[@]}" ]; do
    mapfile -t GPUS < <(pick_gpus 4 20000)
    if [ "${#GPUS[@]}" -eq 0 ]; then say "  no GPU with >=20GB free; waiting"; sleep 180; continue; fi
    launched=0
    for g in "${GPUS[@]}"; do
      [ "$qi" -lt "${#QUEUE[@]}" ] || break
      set -- ${QUEUE[$qi]}; M=$1; K=$2; TAG=$3
      if [ -f "$OUT/$TIER/$TAG/inference_uncertainty.csv" ]; then
        say "  skip $TIER/$TAG (already present)"; qi=$((qi+1)); continue
      fi
      EX=()
      [ "$TAG" = base_scaled ] && EX=(--temperature "$T")
      [ "$TAG" = epinet ] && EX=(--epinet_path "$EPI")
      CUDA_VISIBLE_DEVICES=$g python -m nn_proj.models.NT_transformer.inference \
        --data_path "$D/${TIER}.csv" --checkpoint "$BASE" \
        --model_max_length $MAXLEN --uncertainty_method "$M" --num_samples $K \
        --num_labels $N --per_device_eval_batch_size 32 \
        --seed 1 --data_seed 1 --run_name "gt_${TIER}_${TAG}" \
        --output_dir "$OUT/$TIER/$TAG" "${EX[@]}" \
        > "$LOG/${TIER}_${TAG}.log" 2>&1 &
      say "  $TIER/$TAG -> gpu $g"
      qi=$((qi+1)); launched=$((launched+1))
    done
    [ "$launched" -gt 0 ] && wait
  done
  say "  tier $TIER complete"
done
wait
ok=0; miss=0
for TIER in test taxa_out gene_out; do for TAG in base base_scaled mc_dropout epinet; do
  if [ -f "$OUT/$TIER/$TAG/inference_uncertainty.csv" ]; then ok=$((ok+1));
  else miss=$((miss+1)); say "FAILED $TIER/$TAG"; fi
done; done
say "DONE gene_taxa: $ok/12 outputs written${miss:+, $miss missing}"
