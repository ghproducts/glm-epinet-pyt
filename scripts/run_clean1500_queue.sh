#!/usr/bin/env bash
# Train + evaluate the clean1500 grid from a shared job queue, N workers per GPU.
# usage: bash scripts/run_clean1500_queue.sh <queue_file> <gpu_list e.g. 0,1,2> <workers_per_gpu> <log_dir>
# Queue lines: "<backbone> <rank|gene_taxa> <seed>", consumed top to bottom (put the longest first).
# Env passthrough: CKPT_ROOT, OUT_ROOT (defaults checkpoints/ and results/clean1500_es), and
# EPOCHS / PATIENCE for train_clean1500.sh (default early stopping: cap 20, patience 3).
# A job is skipped if its evaluation already finished; training is skipped if both
# checkpoints exist, so the queue can be re-run after an interruption.
set -uo pipefail
cd "$(dirname "$0")/.."
Q=$1; GPUS=$2; W=$3; L=$4; mkdir -p "$L"
P=/scratch/home/glh52/venvs/glm_epinet_venv/bin/python; export PYTHONPATH=$PWD
export CKPT_ROOT=${CKPT_ROOT:-checkpoints/clean1500_es} EPOCHS=${EPOCHS:-20} PATIENCE=${PATIENCE:-3}; OUT_ROOT=${OUT_ROOT:-results/clean1500_es}
pop() { flock "$Q.lock" bash -c "head -1 '$Q'; sed -i '1d' '$Q'"; }
worker() { G=$1
  while true; do
    job=$(pop); [ -z "$job" ] && break; set -- $job; B=$1; R=$2; S=$3
    [ -f $OUT_ROOT/${B}_${R}_s${S}/detection.csv ] && { echo "[$(date +%T)] skip $job (evaluated)"; continue; }
    CK=$CKPT_ROOT/$B/$R/seed$S
    if ! ls -d $CK/epinet/checkpoint-* >/dev/null 2>&1; then
      echo "[$(date +%T)] gpu$G train $job"
      bash scripts/train_clean1500.sh $B $R $S $G > $L/train_${B}_${R}_s$S.log 2>&1
      grep -q DONE $L/train_${B}_${R}_s$S.log || { echo "[$(date +%T)] FAILED train $job"; continue; }
    fi
    echo "[$(date +%T)] gpu$G eval $job"
    CUDA_VISIBLE_DEVICES=$G $P data_gen/clean_fragments/eval_clean1500.py --backbone $B --rank $R --seed $S --ckpt-root $CKPT_ROOT --out-root $OUT_ROOT > $L/eval_${B}_${R}_s$S.log 2>&1 \
      && echo "[$(date +%T)] DONE $job" || echo "[$(date +%T)] FAILED eval $job"
  done; }
for G in ${GPUS//,/ }; do for i in $(seq $W); do worker $G & sleep 2; done; done
wait; echo "[$(date +%T)] QUEUE EMPTY"
