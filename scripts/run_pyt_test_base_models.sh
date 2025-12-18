
set -euo pipefail

export CUDA_VISIBLE_DEVICES=6
source ~/py311-pyt/bin/activate
cd /scratch/home/glh52/temp-glm-epinet

# paths 
DATA_ROOT="/scratch/home/glh52/glm-epinet/DATA"
MODEL="DNABERT2" # DNABERT2, NT_transformer, hyenaDNA
CKPT_ROOT="trained_models"
OUT_ROOT="inference_results"
BATCH_SIZE=1

# testing params
UQ_method="mc_dropout" # epinet, mc_dropout, base
K_SAMPLES=10

mkdir -p "$CKPT_ROOT"
mkdir -p "$OUT_ROOT"

declare -a EVAL_TASKS=(
    # PBsim reads, taxa family 6kbp
    "pbsim id_novel_genus 6000 429"                
    "pbsim ood_novel_family 6000 429"             
    "pbsim ood_nonbacterial 6000 429"              

    # Scorpio gene_taxa 3kbp
    #"gene_taxa test 3000 437"           
    #"gene_taxa gene_out 3000 437"           
    #"gene_taxa taxa_out 3000 437"           
)


# eval
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
    CKPT_PATH="$(ls -1d "${CKPT_ROOT}/${MODEL}/${TRAIN_TASK}"/checkpoint-* 2>/dev/null | sort -V | tail -n1)"
    EVAL_DATA_PATH="${DATA_ROOT}/${TRAIN_TASK}/${EVAL_TASK}.csv"
    OUT_PATH="${OUT_ROOT}/${MODEL}/${UQ_method}/${TRAIN_TASK}/${EVAL_TASK}"

    python -m nn_proj.models.${MODEL}.inference \
        --data_path "${EVAL_DATA_PATH}" \
        --checkpoint "${CKPT_PATH}" \
        --model_max_length "${MAX_LEN}" \
        --run_name "${MODEL}_${TRAIN_TASK}_to_${EVAL_TASK}" \
        --per_device_eval_batch_size ${BATCH_SIZE} \
        --num_samples "${K_SAMPLES}" \
        --uncertainty_method "${UQ_method}" \
        --num_labels "${NUM_LABELS}" \
        --output_dir "${OUT_PATH}" 
done