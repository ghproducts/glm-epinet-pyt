#!/usr/bin/env bash
# Calibration on the standard pbsim simulated-read task.
#
# This deliberately uses the ORIGINAL full-label-space checkpoints (36-way
# phylum, 429-way family) rather than the leave-phyla-out subsets. Those
# subsets exist to make the label space an independent variable for the
# OOD/darkmatter question; calibration has no need for that and is better
# measured on the standard task the paper reports.
#
# Tiers:
#   ID       id_test  -- training species, held-out reads (new; built by
#            data_gen/pbsim/make_id_test_from_train.py)
#   near-ID  id_novel_genus -- novel genus, label still in the label space
# near/far OOD are excluded: no correct class exists there, so accuracy and
# hence ECE are undefined.
#
# Methods: base, base_scaled (temperature from temp_scaling_factors_1.tsv),
# mc_dropout (K=10), conv_epinet (K=16). The near-ID arm already exists under
# inference_results_1/; this only fills in the ID arm.
set -euo pipefail
cd ../
VENV=${VENV:-/scratch/home/glh52/venvs/glm_epinet_venv}
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RANK=${1:-phylum}
case "$RANK" in
  phylum) NUM_LABELS=36;  TEMP=1.459559 ;;
  family) NUM_LABELS=429; TEMP=0.875454 ;;
  *) echo "usage: run_calibration_pbsim.sh <phylum|family>" >&2; exit 1 ;;
esac

DATA=data_gen/leave_phyla_out_v3/eval_id_test_6000bp.csv   # 6kb, native length
BASE=$(ls -1d old_models/trained_models_1/NT_transformer/pbsim_${RANK}/checkpoint-* | sort -V | tail -1)
EPI=$(ls -1d old_models/trained_models_1/conv_epinet/NT_transformer/pbsim_${RANK}/checkpoint-* | sort -V | tail -1)
TAXA=/scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv
OUT=results/calibration_pbsim/${RANK}/id_test
LOG=logs/calibration_pbsim
mkdir -p "$LOG"
echo "rank=$RANK labels=$NUM_LABELS temp=$TEMP"

run () {  # run <gpu> <method> <k> <tag> <extra...>
    local gpu=$1 method=$2 k=$3 tag=$4; shift 4
    CUDA_VISIBLE_DEVICES=$gpu python -m nn_proj.models.NT_transformer.inference \
        --data_path "$DATA" --checkpoint "$BASE" \
        --taxa_df "$TAXA" --taxa_rank "$RANK" \
        --model_max_length 1000 --uncertainty_method "$method" --num_samples "$k" \
        --num_labels $NUM_LABELS --per_device_eval_batch_size 8 \
        --seed 1 --data_seed 1 --run_name "calib_${RANK}_${tag}" \
        --output_dir "$OUT/$tag" "$@" > "$LOG/${RANK}_${tag}.log" 2>&1
}

run 0 base       1  base                            &
run 1 base       1  base_scaled  --temperature $TEMP &
run 2 mc_dropout 10 mc_dropout                      &
run 3 epinet     16 conv_epinet  --epinet_path "$EPI" &
wait
echo "done -> $OUT"
