#!/bin/bash
#set -euo pipefail

export CUDA_VISIBLE_DEVICES=9
source ~/py311-pyt/bin/activate
cd /scratch/home/glh52/glm-epinet-pyt

#SEED
SEED=3
echo "Getting temp factors for seed ${SEED}"

# paths 
#MODEL="DNABERT2" # DNABERT2, NT_transformer, hyenaDNA, CARMANIA
CKPT_ROOT="trained_models_${SEED}"
OUT_ROOT="inference_results_${SEED}"
BATCH_SIZE=8

MODELS="hyenaDNA DNABERT2 NT_transformer CARMANIA"

for MODEL in $MODELS; do
    # regulatory tasks
    DATA_ROOT="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
    declare -a TRAIN_TASKS=(
        "promoter_all 300 2"
        "enhancers_types 400 3"
        "splice_sites_acceptors 600 2"
    )


    for ENTRY in "${TRAIN_TASKS[@]}"; do
        set -- $ENTRY
        TRAIN_TASK="$1"; MAX_LEN="$2"; NUM_LABELS="$3"

        #CKPT_PATH="${CKPT_ROOT}/${MODEL}/${TRAIN_TASK}"
        CKPT_PATH="$(ls -1d "${CKPT_ROOT}/${MODEL}/${TRAIN_TASK}"/checkpoint-* 2>/dev/null | sort -V | tail -n1)"
        TRAIN_DATA_PATH="${DATA_ROOT}/${TRAIN_TASK}"
        OUT_PATH="${OUT_ROOT}/temp"

        echo "$CKPT_PATH"

        python -m nn_proj.models.${MODEL}.scaling \
            --data_path "${TRAIN_DATA_PATH}" \
            --checkpoint "${CKPT_PATH}" \
            --model_max_length "${MAX_LEN}" \
            --run_name "${MODEL}_${TRAIN_TASK}_temp_scaling" \
            --per_device_eval_batch_size ${BATCH_SIZE} \
            --output_dir "${OUT_PATH}" 
    done


    # metagenomic tasks
    DATA_ROOT="/scratch/home/glh52/glm-epinet/DATA"
    declare -a TRAIN_TASKS=(
        "gene_taxa 3000 437 None"
        "pbsim 6000 429 family"
        "pbsim 6000 429 order"
        "pbsim 6000 429 class"
        "pbsim 6000 429 phylum"
    )

    for ENTRY in "${TRAIN_TASKS[@]}"; do
        set -- $ENTRY
        TRAIN_TASK="$1"; MAX_LEN="$2"; NUM_LABELS="$3"; RANK="$4"

        if [ "$MODEL" = "DNABERT2" ] ; then
            MAX_LEN=$(( MAX_LEN / 4 ))
        elif [ "$MODEL" = "NT_transformer" ] ; then
            MAX_LEN=$(( MAX_LEN / 6 ))
        fi

        if [ "$RANK" = "None" ] ; then
            RANK_SUFFIX=""
        else
            RANK_SUFFIX="_${RANK}"
        fi 

        
        CKPT_PATH="$(ls -1d "${CKPT_ROOT}/${MODEL}/${TRAIN_TASK}${RANK_SUFFIX}"/checkpoint-* 2>/dev/null | sort -V | tail -n1)"
        TRAIN_DATA_PATH="${DATA_ROOT}/${TRAIN_TASK}/train.csv"
        OUT_PATH="${OUT_ROOT}/temp"
        

        echo "$CKPT_PATH"

        python -m nn_proj.models.${MODEL}.scaling \
            --data_path "${TRAIN_DATA_PATH}" \
            --checkpoint "${CKPT_PATH}" \
            --model_max_length "${MAX_LEN}" \
            --taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
            --taxa_rank ${RANK} \
            --run_name "${MODEL}_${TRAIN_TASK}_temp_scaling" \
            --per_device_eval_batch_size ${BATCH_SIZE} \
            --output_dir "${OUT_PATH}" \
            --seed ${SEED} \
            --data_seed ${SEED}
    done
done