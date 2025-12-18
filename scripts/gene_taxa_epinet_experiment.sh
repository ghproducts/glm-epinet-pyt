
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4
source ~/py311-pyt/bin/activate
cd /scratch/home/glh52/temp-glm-epinet

# paths 
DATA_ROOT="/scratch/home/glh52/glm-epinet/DATA"
MODEL="NT_transformer" # DNABERT2, NT_transformer, hyenaDNA
CKPT_ROOT="trained_models"
OUT_ROOT="inference_results"
BATCH_SIZE=4
EPINET_NAME="conv_epinet"

# training params
LR=1e-6
EPOCHS=1

# testing params
K_SAMPLES=10
UQ_method="epinet"

mkdir -p "$CKPT_ROOT"

TRAINED=False

declare -a TRAIN_TASKS=(
    "gene_taxa 3000 437"
    "pbsim 6000 429"
)

declare -a EVAL_TASKS=(
    "gene_taxa test 3000 437"                
    "gene_taxa gene_out 3000 437"
    "gene_taxa taxa_out 3000 437"

    "pbsim id_novel_genus 6000 429"
    "pbsim ood_novel_family 6000 429"
    "pbsim ood_nonbacterial 6000 429"
)

# train
if [ "$TRAINED" = True ] ; then
    echo "Skipping training as TRAINED is set to True"
else
    echo "Starting training phase"

    for ENTRY in "${TRAIN_TASKS[@]}"; do
        set -- $ENTRY
        TRAIN_TASK="$1"; MAX_LEN="$2"; NUM_LABELS="$3"

        if [ """$MODEL" = "DNABERT2" ] ; then
            MAX_LEN=$(( MAX_LEN / 4 ))
        elif [ "$MODEL" = "NT_transformer" ] ; then
            MAX_LEN=$(( MAX_LEN / 6 ))
        fi

        TRAIN_DATA_PATH="${DATA_ROOT}/${TRAIN_TASK}/train.csv"
        BASE_PATH="$(ls -1d "${CKPT_ROOT}/${MODEL}/${TRAIN_TASK}"/checkpoint-* 2>/dev/null | sort -V | tail -n1)"
        EPI_PATH="${CKPT_ROOT}/${EPINET_NAME}/${MODEL}/${TRAIN_TASK}"
        
        python -m nn_proj.models.${MODEL}.train_epinet \
            --data_path "${TRAIN_DATA_PATH}" \
            --checkpoint "${BASE_PATH}" \
            --run_name "${MODEL}_${TRAIN_TASK}" \
            --model_max_length "${MAX_LEN}" \
            --per_device_train_batch_size "${BATCH_SIZE}" \
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
            --find_unused_parameters False
    done
    echo "Training phase completed"
fi

# eval base
echo "Starting evaluation phase"
for ENTRY in "${EVAL_TASKS[@]}"; do
    set -- $ENTRY
    TRAIN_TASK="$1"; EVAL_TASK="$2"; MAX_LEN="$3"; NUM_LABELS="$4"
    
    if [ """$MODEL" = "DNABERT2" ] ; then
        MAX_LEN=$(( MAX_LEN / 4 ))
    elif [ "$MODEL" = "NT_transformer" ] ; then
        MAX_LEN=$(( MAX_LEN / 6 ))
    fi


    #CKPT_PATH="${CKPT_ROOT}/${MODEL}/${TRAIN_TASK}"
    BASE_PATH="$(ls -1d "${CKPT_ROOT}/${MODEL}/${TRAIN_TASK}"/checkpoint-* 2>/dev/null | sort -V | tail -n1)"
    EPI_PATH="$(ls -1d "${CKPT_ROOT}/${EPINET_NAME}/${MODEL}/${TRAIN_TASK}"/checkpoint-* 2>/dev/null | sort -V | tail -n1)"
    EVAL_DATA_PATH="${DATA_ROOT}/${TRAIN_TASK}/${EVAL_TASK}.csv"
    OUT_PATH="${OUT_ROOT}/${MODEL}/${EPINET_NAME}/${TRAIN_TASK}/${EVAL_TASK}"

    python -m nn_proj.models.${MODEL}.inference \
        --data_path "${EVAL_DATA_PATH}" \
        --checkpoint "${BASE_PATH}" \
        --epinet_path "${EPI_PATH}" \
        --model_max_length "${MAX_LEN}" \
        --run_name "${MODEL}_${TRAIN_TASK}_to_${EVAL_TASK}" \
        --per_device_eval_batch_size ${BATCH_SIZE} \
        --num_samples "${K_SAMPLES}" \
        --uncertainty_method "${UQ_method}" \
        --num_labels "${NUM_LABELS}" \
        --output_dir "${OUT_PATH}" 
done