#!/usr/bin/env bash
# Full evaluation on the LEAKAGE-FREE checkpoints.
#
# Base and epinet models were retrained on pbsim_id_test/train.csv, which
# excludes every read in id_test.csv. Against the previous checkpoints id_test
# was training data (300/300 sampled reads found in the old train.csv), so all
# earlier ID-tier numbers were optimistic. Everything here is recomputed.
#
# Four tiers, increasing novelty:
#   id_test           training species, held-out reads   (labels valid)
#   id_novel_genus    novel genus, phylum known          (labels valid)
#   ood_novel_family  novel family, phylum mostly known  (labels valid at phylum)
#   ood_nonbacterial  archaea / eukaryota / viruses      (no valid label)
#
# Four methods: base, base_scaled (temperature fit by scaling.py on the
# training split), mc_dropout K=10, epinet K=16.
# Reads are ~10.7kb; model_max_length 1000 truncates to ~6kb, matching training.
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-/scratch/home/glh52/venvs/glm_epinet_venv}
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RANK=${1:-phylum}
case "$RANK" in
  phylum) N=36  ;;
  family) N=429 ;;
  *) echo "usage: eval_clean_all.sh <phylum|family>" >&2; exit 1 ;;
esac
D=/scratch/home/glh52/glm-epinet/DATA/pbsim_id_test
TAXA=$D/full_basic_lineage.csv
BASE=checkpoints/pbsim_clean/$RANK/checkpoint-57734
EPI=checkpoints/pbsim_clean_epinet/$RANK/checkpoint-57734
OUT=results/clean_eval/$RANK
LOG=logs/clean_eval
STATUS=$LOG/status_${RANK}.txt
mkdir -p "$LOG" "$OUT"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

say "FIT temperature on the training split ($RANK)"
T=$(CUDA_VISIBLE_DEVICES=0 python -m nn_proj.models.NT_transformer.scaling \
      --data_path "$D/train.csv" --checkpoint "$BASE" \
      --taxa_df "$TAXA" --taxa_rank "$RANK" \
      --model_max_length 1000 --num_labels $N \
      --per_device_eval_batch_size 8 --seed 1 --data_seed 1 \
      --output_dir "$OUT/_scaling" 2>"$LOG/scaling_${RANK}.err" \
    | grep -oE "^T = [0-9.]+" | awk '{print $3}')
if [ -z "${T:-}" ]; then say "FAILED temperature fit (see $LOG/scaling_${RANK}.err)"; T=1.0; fi
say "  T = $T"

# One job per GPU, one tier at a time. 16 concurrent jobs across 10 GPUs put
# two 1000-token runs on the same device and every doubled-up job OOM'd --
# epinet (K=16) and mc_dropout (K=10) hold many times the activations of base.
# Pick GPUs by free memory at run time. Fixed indices failed: another user's
# job took 76GB on the devices this had hardcoded, so those runs OOM'd with
# ~2GB available while other GPUs sat idle.
pick_gpus () {   # pick_gpus <count> <min_free_mb>
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
      | awk -F', ' -v need="$2" '{f=$3-$2; if (f>=need) print f, $1}' \
      | sort -rn | head -n "$1" | awk '{print $2}'
}

for TIER in id_test id_novel_genus ood_novel_family ood_nonbacterial; do
  say "TIER $TIER"
  QUEUE=("base 1 base" "base 1 base_scaled" "mc_dropout 10 mc_dropout" "epinet 16 epinet")
  qi=0
  while [ "$qi" -lt "${#QUEUE[@]}" ]; do
    mapfile -t GPUS < <(pick_gpus 4 20000)
    if [ "${#GPUS[@]}" -eq 0 ]; then
      say "  no GPU with >=20GB free; waiting"
      sleep 180
      continue
    fi
    # launch as many as there are free GPUs, one job per device
    launched=0
    for g in "${GPUS[@]}"; do
      [ "$qi" -lt "${#QUEUE[@]}" ] || break
      set -- ${QUEUE[$qi]}; M=$1; K=$2; TAG=$3
      EX=()
      [ "$TAG" = base_scaled ] && EX=(--temperature "$T")
      [ "$TAG" = epinet ] && EX=(--epinet_path "$EPI")
      if [ -f "$OUT/$TIER/$TAG/inference_uncertainty.csv" ]; then
        say "  skip $TIER/$TAG (already present)"; qi=$((qi+1)); continue
      fi
      CUDA_VISIBLE_DEVICES=$g python -m nn_proj.models.NT_transformer.inference \
        --data_path "$D/${TIER}.csv" --checkpoint "$BASE" \
        --taxa_df "$TAXA" --taxa_rank "$RANK" \
        --model_max_length 1000 --uncertainty_method "$M" --num_samples $K \
        --num_labels $N --per_device_eval_batch_size 4 \
        --seed 1 --data_seed 1 --run_name "clean_${RANK}_${TIER}_${TAG}" \
        --output_dir "$OUT/$TIER/$TAG" "${EX[@]}" \
        > "$LOG/${RANK}_${TIER}_${TAG}.log" 2>&1 &
      say "  $TIER/$TAG -> gpu $g"
      qi=$((qi+1)); launched=$((launched+1))
    done
    [ "$launched" -gt 0 ] && wait
  done
  say "  tier $TIER complete"
done
wait
ok=0; miss=0
for TIER in id_test id_novel_genus ood_novel_family ood_nonbacterial; do
  for TAG in base base_scaled mc_dropout epinet; do
    if [ -f "$OUT/$TIER/$TAG/inference_uncertainty.csv" ]; then ok=$((ok+1));
    else miss=$((miss+1)); say "FAILED $TIER/$TAG"; fi
  done
done
say "DONE $RANK: $ok/16 outputs written${miss:+, $miss missing}"
