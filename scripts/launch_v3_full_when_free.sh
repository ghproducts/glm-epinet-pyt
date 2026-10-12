#!/usr/bin/env bash
# Launch the 6kb leave-phyla-out training with FULL fine-tuning, as soon as
# enough GPU memory frees up.
#
# Full fine-tuning of NT-500M at 1000 tokens needs roughly 25-30GB per model:
# optimizer state for all 494M parameters plus backward activations for all 29
# layers. EsmForSequenceClassification does not implement
# gradient_checkpointing_enable, so there is no way to trade compute for memory
# here -- the only option is to wait for the memory.
#
# Polls free memory per GPU and starts one model per GPU that clears the
# threshold. Deliberately does NOT use `pgrep -f` to detect running jobs: the
# watcher's own command line contains whatever pattern it would search for, so
# it matches itself and waits forever.
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-/scratch/home/glh52/venvs/glm_epinet_venv}
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NEED_MB=${NEED_MB:-34000}     # per-GPU free memory required
N_MODELS=10
OUT=checkpoints/lpo_v3_full
LOG=logs/lpo_v3_full
mkdir -p "$OUT" "$LOG"

echo "[watch] waiting for ${N_MODELS} GPUs with >=${NEED_MB}MB free ..."
while true; do
    mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used,memory.total \
                        --format=csv,noheader,nounits | awk -F', ' '{print $1" "$3-$2}')
    AVAIL=()
    for row in "${FREE[@]}"; do
        set -- $row
        if [ "$2" -ge "$NEED_MB" ]; then AVAIL+=("$1"); fi
    done
    if [ "${#AVAIL[@]}" -ge "$N_MODELS" ]; then break; fi
    echo "[watch] $(date +%H:%M:%S) only ${#AVAIL[@]} GPU(s) free enough; sleeping"
    sleep 120
done
echo "[watch] GPUs available: ${AVAIL[*]}"

for i in $(seq 0 $((N_MODELS-1))); do
    g=${AVAIL[$i]}
    CUDA_VISIBLE_DEVICES=$g nohup python -m nn_proj.models.NT_transformer.train_base \
        --data_path data_gen/leave_phyla_out_v3/subsets/subset_${i}.csv \
        --num_labels 5 --model_max_length 1000 \
        --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5 --num_train_epochs 2 --fp16 \
        --eval_strategy epoch --save_strategy epoch --save_total_limit 1 \
        --load_best_model_at_end True \
        --warmup_steps 50 --logging_steps 200 --overwrite_output_dir True \
        --find_unused_parameters False \
        --run_name v3full_${i} --output_dir "$OUT/subset_${i}" \
        --seed 1 --data_seed 1 > "$LOG/train_${i}.log" 2>&1 &
    echo "[watch] subset_${i} -> gpu ${g}"
    sleep 5
done
wait
echo "[watch] FULL 6kb TRAINING COMPLETE"
