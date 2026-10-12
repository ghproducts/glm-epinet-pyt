#!/usr/bin/env bash
# Full-scale label-noise grid: 4 backbones x 5 noise rates x 3 seeds x
# (base, base_scaled, mc_dropout, conv_epinet). See
# .docs/LABEL_NOISE_PILOT.md "Recommendation for the full-scale run".
#
# Structure (mirrors CLAUDE.md's known constraints):
#   Phase 1: train -> epinet, per backbone, backbones run in PARALLEL
#            (each backbone pinned to its own GPU; train then epinet run
#            sequentially *within* a backbone's own background process,
#            since epinet needs that backbone's own base checkpoint).
#   Phase 2: temperature fitting, ALL backbones in a SINGLE sequential
#            process (run_grid.py's own execution loop is one Python
#            process iterating jobs one at a time) -- required because of
#            the unlocked read-modify-write race in record_temperature()
#            writing checkpoints/temperatures.json.
#   Phase 3: inference, per backbone, backbones run in PARALLEL again.
#
# Run this script itself with nohup/disown so it survives the launching
# shell exiting; it manages its own phase ordering and backgrounding.

set -uo pipefail
cd "$(dirname "$0")/.."   # repo root
REPO_ROOT="$(pwd)"
source /scratch/home/glh52/venvs/glm_epinet_venv/bin/activate

LOGDIR="$REPO_ROOT/logs/label_noise_full"
mkdir -p "$LOGDIR"

TASKS="label_noise_r00 label_noise_r05 label_noise_r10 label_noise_r20 label_noise_r40"
SEEDS="1 2 3"

# Backbone -> GPU pin. Chosen from GPUs confirmed idle (0% util, no other
# user's process) at launch time: 0,1,2,3,4,9 were idle; 5,6,7,8 are running
# this same user's unrelated amigo/two-headed job and are left alone.
declare -A GPU=( [DNABERT2]=0 [NT_transformer]=1 [hyenaDNA]=2 [CARMANIA]=3 )

echo "[$(date)] === Phase 1: train + epinet (parallel per backbone) ===" | tee -a "$LOGDIR/master.log"
pids=()
for BB in DNABERT2 NT_transformer hyenaDNA CARMANIA; do
  (
    export CUDA_VISIBLE_DEVICES="${GPU[$BB]}"
    echo "[$(date)] $BB starting on GPU ${GPU[$BB]} (train)"
    python scripts/run_grid.py --stage train --backbone "$BB" --seed $SEEDS \
      --task $TASKS --skip-existing --continue-on-error --run
    echo "[$(date)] $BB train stage done, starting epinet"
    python scripts/run_grid.py --stage epinet --backbone "$BB" --seed $SEEDS \
      --task $TASKS --skip-existing --continue-on-error --run
    echo "[$(date)] $BB epinet stage done"
  ) > "$LOGDIR/phase1_${BB}.log" 2>&1 &
  pids+=($!)
  echo "  $BB -> PID $! -> GPU ${GPU[$BB]} -> $LOGDIR/phase1_${BB}.log" | tee -a "$LOGDIR/master.log"
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "[$(date)] === Phase 1 complete (fail=$fail) ===" | tee -a "$LOGDIR/master.log"

echo "[$(date)] === Phase 2: temperature fitting (single sequential process, all backbones) ===" | tee -a "$LOGDIR/master.log"
export CUDA_VISIBLE_DEVICES=0
python scripts/run_grid.py --stage temperature --seed $SEEDS \
  --task $TASKS --continue-on-error --run > "$LOGDIR/phase2_temperature.log" 2>&1
echo "[$(date)] === Phase 2 complete (exit=$?) ===" | tee -a "$LOGDIR/master.log"

echo "[$(date)] === Phase 3: inference (parallel per backbone) ===" | tee -a "$LOGDIR/master.log"
pids=()
for BB in DNABERT2 NT_transformer hyenaDNA CARMANIA; do
  (
    export CUDA_VISIBLE_DEVICES="${GPU[$BB]}"
    echo "[$(date)] $BB starting on GPU ${GPU[$BB]} (inference)"
    python scripts/run_grid.py --stage inference --backbone "$BB" --seed $SEEDS \
      --task $TASKS --skip-existing --continue-on-error --run
    echo "[$(date)] $BB inference stage done"
  ) > "$LOGDIR/phase3_${BB}.log" 2>&1 &
  pids+=($!)
  echo "  $BB -> PID $! -> GPU ${GPU[$BB]} -> $LOGDIR/phase3_${BB}.log" | tee -a "$LOGDIR/master.log"
done

fail3=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail3=1
done
echo "[$(date)] === Phase 3 complete (fail=$fail3) ===" | tee -a "$LOGDIR/master.log"
echo "[$(date)] === ALL_DONE (phase1_fail=$fail phase3_fail=$fail3) ===" | tee -a "$LOGDIR/master.log"
