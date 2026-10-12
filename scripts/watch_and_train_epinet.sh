#!/usr/bin/env bash
# Wait for the clean pbsim base trainings to finish, verify they produced
# checkpoints, then train the Epinet heads on them.
#
# Detection notes (this has bitten us repeatedly):
#   * waiting is done on PIDs via `kill -0`, never `pgrep -f <pattern>` -- the
#     watching shell's own command line contains any pattern it searches for,
#     so pgrep matches itself and the wait never ends;
#   * completion is confirmed by the ARTIFACT (a checkpoint directory), not by
#     process exit, because a crashed run also "finishes";
#   * every milestone and every failure is echoed to a status file so an
#     external monitor sees both, rather than only the happy path.
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-/scratch/home/glh52/venvs/glm_epinet_venv}
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STATUS=${STATUS:-logs/pbsim_clean/status.txt}
mkdir -p "$(dirname "$STATUS")" logs/pbsim_clean_epinet checkpoints/pbsim_clean_epinet
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

PHY_PID=${1:?need phylum pid}
FAM_PID=${2:?need family pid}

say "WAIT base training: phylum pid=$PHY_PID family pid=$FAM_PID"
while kill -0 "$PHY_PID" 2>/dev/null || kill -0 "$FAM_PID" 2>/dev/null; do sleep 60; done
say "BASE TRAINING PROCESSES EXITED"

declare -A CK
ok=1
for r in phylum family; do
    c=$(ls -1d checkpoints/pbsim_clean/$r/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -z "$c" ] || [ ! -f "$c/model.safetensors" ]; then
        say "FAILED base $r: no usable checkpoint (see logs/pbsim_clean/$r.log)"
        ok=0
    else
        acc=$(tr '\r' '\n' < logs/pbsim_clean/$r.log | grep -o "'eval_accuracy': [0-9.]*" | tail -1 | cut -d' ' -f2)
        say "OK base $r -> $c  eval_accuracy=${acc:-unknown}"
        CK[$r]=$c
    fi
done
[ "$ok" -eq 1 ] || { say "ABORT: not training epinet because a base run failed"; exit 1; }

say "START epinet heads on the clean checkpoints"
declare -A PIDS
gpu=0
for r in phylum family; do
    n=36; [ "$r" = family ] && n=429
    CUDA_VISIBLE_DEVICES=$gpu nohup python -m nn_proj.models.NT_transformer.train_epinet \
        --data_path /scratch/home/glh52/glm-epinet/DATA/pbsim_id_test/train.csv \
        --taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim_id_test/full_basic_lineage.csv \
        --taxa_rank "$r" --checkpoint "${CK[$r]}" --num_labels $n \
        --model_max_length 1000 \
        --per_device_train_batch_size 8 --per_device_eval_batch_size 8 \
        --learning_rate 1e-3 --num_train_epochs 2 --fp16 \
        --eval_strategy epoch --save_strategy epoch --save_total_limit 1 \
        --load_best_model_at_end True \
        --warmup_steps 50 --logging_steps 500 --overwrite_output_dir True \
        --find_unused_parameters False \
        --run_name "epinet_clean_${r}" --output_dir "checkpoints/pbsim_clean_epinet/$r" \
        --seed 1 --data_seed 1 > "logs/pbsim_clean_epinet/${r}.log" 2>&1 &
    PIDS[$r]=$!
    say "  epinet $r -> gpu $gpu (pid ${PIDS[$r]})"
    gpu=$((gpu+1))
done
for r in phylum family; do wait "${PIDS[$r]}" || say "epinet $r exited nonzero"; done

for r in phylum family; do
    c=$(ls -1d checkpoints/pbsim_clean_epinet/$r/checkpoint-* 2>/dev/null | sort -V | tail -1)
    if [ -z "$c" ]; then say "FAILED epinet $r (see logs/pbsim_clean_epinet/$r.log)"
    else say "OK epinet $r -> $c"; fi
done
say "ALL DONE: clean base + epinet checkpoints ready"
