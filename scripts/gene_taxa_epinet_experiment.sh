
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
source ~/py311-pyt/bin/activate
cd /scratch/home/glh52/glm-epinet-pyt

# seed
SEED=1
echo "Seed: ${SEED}"

# paths 
DATA_ROOT="/scratch/home/glh52/glm-epinet/DATA"
#MODEL="CARMANIA" # DNABERT2, NT_transformer, hyenaDNA
CKPT_ROOT="trained_models_${SEED}"
OUT_ROOT="inference_results_${SEED}"
BATCH_SIZE=16
EPINET_NAME="conv_epinet"

# training params
LR=1e-6
EPOCHS=1

# testing params
K_SAMPLES=10
UQ_method="epinet"

mkdir -p "$CKPT_ROOT"

TRAINED=False

MODELS="CARMANIA" #"hyenaDNA DNABERT2 NT_transormer CARMANIA"

declare -a TRAIN_TASKS=(
    #"gene_taxa 3000 437 None"
    # "pbsim 6000 429 family"
    # "pbsim 6000 429 order"
    "pbsim 6000 429 class"
    "pbsim 6000 429 phylum"
)

declare -a EVAL_TASKS=(
    "gene_taxa test 3000 437 None"                
    "gene_taxa gene_out 3000 437 None"
    "gene_taxa taxa_out 3000 437 None"

    "pbsim id_novel_genus 6000 429 family"     
    "pbsim id_novel_genus 6000 429 order"                
    "pbsim id_novel_genus 6000 429 class"     
    "pbsim id_novel_genus 6000 429 phylum"     

    "pbsim ood_novel_family 6000 429 family"
    "pbsim ood_novel_family 6000 429 order"             
    "pbsim ood_novel_family 6000 429 class"             
    "pbsim ood_novel_family 6000 429 phylum"             
    
    "pbsim ood_nonbacterial 6000 429 family"              
    "pbsim ood_nonbacterial 6000 429 order"              
    "pbsim ood_nonbacterial 6000 429 class"              
    "pbsim ood_nonbacterial 6000 429 phylum"      
)

for MODEL in $MODELS
do
    # train
    if [ "$TRAINED" = True ] ; then
        echo "Skipping training as TRAINED is set to True"
    else
        echo "Starting training phase"

        for ENTRY in "${TRAIN_TASKS[@]}"; do
            set -- $ENTRY
            TRAIN_TASK="$1"; MAX_LEN="$2"; NUM_LABELS="$3"; RANK="$4"

            if [ """$MODEL" = "DNABERT2" ] ; then
                MAX_LEN=$(( MAX_LEN / 4 ))
            elif [ "$MODEL" = "NT_transformer" ] ; then
                MAX_LEN=$(( MAX_LEN / 6 ))
            fi

            if [ "$RANK" = "None" ] ; then
                RANK_SUFFIX=""
            else
                RANK_SUFFIX="_${RANK}"
            fi 

            TRAIN_DATA_PATH="${DATA_ROOT}/${TRAIN_TASK}/train.csv"
            BASE_PATH="$(ls -1d "${CKPT_ROOT}/${MODEL}/${TRAIN_TASK}${RANK_SUFFIX}"/checkpoint-* 2>/dev/null | sort -V | tail -n1)"
            EPI_PATH="${CKPT_ROOT}/${EPINET_NAME}/${MODEL}/${TRAIN_TASK}${RANK_SUFFIX}"

            echo "Base path: ${BASE_PATH}"
            echo "Epi path: ${EPI_PATH}"

            python -m nn_proj.models.${MODEL}.train_epinet \
                --data_path "${TRAIN_DATA_PATH}" \
                --checkpoint "${BASE_PATH}" \
                --taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
                --taxa_rank ${RANK} \
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
                --find_unused_parameters False \
                --seed ${SEED} \
                --data_seed ${SEED}

        done
        echo "Training phase completed"
    fi

    # eval base
    echo "Starting evaluation phase"
    for ENTRY in "${EVAL_TASKS[@]}"; do
        set -- $ENTRY
        TRAIN_TASK="$1"; EVAL_TASK="$2"; MAX_LEN="$3"; NUM_LABELS="$4"; RANK="$5"
        
        if [ """$MODEL" = "DNABERT2" ] ; then
            MAX_LEN=$(( MAX_LEN / 4 ))
        elif [ "$MODEL" = "NT_transformer" ] ; then
            MAX_LEN=$(( MAX_LEN / 6 ))
        fi

        if [ "$RANK" = "None" ] ; then
            RANK_SUFFIX=""
        else
            RANK_SUFFIX="_${RANK}"
        fi 

        BASE_PATH="$(ls -1d "${CKPT_ROOT}/${MODEL}/${TRAIN_TASK}${RANK_SUFFIX}"/checkpoint-* 2>/dev/null | sort -V | tail -n1)"
        EPI_PATH="$(ls -1d "${CKPT_ROOT}/${EPINET_NAME}/${MODEL}/${TRAIN_TASK}${RANK_SUFFIX}"/checkpoint-* 2>/dev/null | sort -V | tail -n1)"
        EVAL_DATA_PATH="${DATA_ROOT}/${TRAIN_TASK}/${EVAL_TASK}.csv"
        OUT_PATH="${OUT_ROOT}/${MODEL}/${EPINET_NAME}/${TRAIN_TASK}/${EVAL_TASK}${RANK_SUFFIX}"

        python -m nn_proj.models.${MODEL}.inference \
            --data_path "${EVAL_DATA_PATH}" \
            --checkpoint "${BASE_PATH}" \
            --epinet_path "${EPI_PATH}" \
            --taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
            --taxa_rank ${RANK} \
            --model_max_length "${MAX_LEN}" \
            --run_name "${MODEL}_${TRAIN_TASK}_to_${EVAL_TASK}" \
            --per_device_eval_batch_size ${BATCH_SIZE} \
            --num_samples "${K_SAMPLES}" \
            --uncertainty_method "${UQ_method}" \
            --num_labels "${NUM_LABELS}" \
            --output_dir "${OUT_PATH}" \
            --seed ${SEED} \
            --data_seed ${SEED}
    done
done