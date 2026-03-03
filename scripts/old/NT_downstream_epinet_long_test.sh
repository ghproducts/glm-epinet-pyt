set -euo pipefail

export CUDA_VISIBLE_DEVICES=9
source ~/py311-pyt/bin/activate
cd /scratch/home/glh52/glm-epinet-pyt

# paths 
DATA_ROOT="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
MODEL="hyenaDNA" # DNABERT2, NT_transformer, hyenaDNA
CKPT_ROOT="trained_models"
OUT_ROOT="inference_results"
BATCH_SIZE=32
EPINET_NAME="conv_epinet_ablation_test"

# training params
LR=1e-6
EPOCHS=10

# testing params
K_SAMPLES=10
UQ_method="epinet"

mkdir -p "$CKPT_ROOT"

TRAINED=False  # set True to skip training and only run evaluation

declare -a TRAIN_TASKS=(
    "promoter_all 300 2"
    "enhancers_types 400 3"
    "splice_sites_acceptors 600 2"
    # "splice_sites_donors 600 2"
)

declare -a EVAL_TASKS=(
    # Promoters (300 bp)
    "promoter_all promoter_all 300 2"                # ID (subtype)
    "promoter_all enhancers 300 2"                    # Far-OOD (regulatory)
    # "promoter_all promoter_tata 300 2"                # Near-ID (subtype)
    # "promoter_all promoter_no_tata 300 2"             # Near-ID (subtype)

    # # Enhancers (400 bp)
     "enhancers_types enhancers_types 400 3"           # ID (3-class)
     "enhancers_types splice_sites_all 400 3"         # Far-OOD (3-class match)

    # # Splice sites (600 bp)
     "splice_sites_acceptors splice_sites_acceptors 600 2"  # ID
     "splice_sites_acceptors splice_sites_donors 600 2"

    # "splice_sites_donors splice_sites_donors 600 2"         # ID
    # "splice_sites_donors splice_sites_acceptors 600 2"      # Near-OOD (paired subtype)
)

# train
if [ "$TRAINED" = True ] ; then
    echo "Skipping training as TRAINED is set to True"
else
    echo "Starting training phase"
    echo "Note: experiment_epinet runs inference at the end of each epoch and writes CSVs into --output_dir."

    #for ENTRY in "${TRAIN_TASKS[@]}"; do
    for ENTRY in "${EVAL_TASKS[@]}"; do
        set -- $ENTRY
        TRAIN_TASK="$1"; MAX_LEN="$2"; NUM_LABELS="$3"
        TRAIN_TASK="$1"; EVAL_TASK="$2"; MAX_LEN="$3"; NUM_LABELS="$4"

        TRAIN_DATA_PATH="${DATA_ROOT}/${TRAIN_TASK}"
        TEST_DATA_PATH="${DATA_ROOT}/${EVAL_TASK}"

        BASE_PATH="$(ls -1d "${CKPT_ROOT}/${MODEL}/${TRAIN_TASK}"/checkpoint-* 2>/dev/null | sort -V | tail -n1)"
        EPI_PATH="${CKPT_ROOT}/${EPINET_NAME}/${MODEL}/${TRAIN_TASK}/${EVAL_TASK}"

        python -m nn_proj.models.${MODEL}.experiment_epinet \
            --data_path "${TRAIN_DATA_PATH}" \
            --test_path "${TEST_DATA_PATH}" \
            --checkpoint "${BASE_PATH}" \
            --run_name "${MODEL}_${TRAIN_TASK}" \
            --model_max_length "${MAX_LEN}" \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${LR} \
            --num_train_epochs ${EPOCHS} \
            --fp16 \
            --output_dir "${EPI_PATH}" \
            --eval_strategy epoch \
            --save_strategy epoch \
            --warmup_steps 50 \
            --logging_steps 100 \
            --overwrite_output_dir True \
            --log_level info \
            --num_labels "${NUM_LABELS}" \
            --find_unused_parameters False \
            --uncertainty_method "${UQ_method}" \
            --num_samples "${K_SAMPLES}"
    done
    echo "Training phase completed"
fi


echo "Done. Per-epoch inference CSVs are written inside each task output_dir under ${CKPT_ROOT}/${EPINET_NAME}/${MODEL}/<TRAIN_TASK>."