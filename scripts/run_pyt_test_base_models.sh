set -euo pipefail

export CUDA_VISIBLE_DEVICES=7
source ~/py311-pyt/bin/activate
cd /scratch/home/glh52/glm-epinet-pyt

# seed
SEED=4
echo "Evaluating on seed ${SEED}"

# paths 
DATA_ROOT="/scratch/home/glh52/glm-epinet/DATA"
#MODEL="DNABERT2" # DNABERT2, NT_transformer, hyenaDNA
CKPT_ROOT="trained_models_${SEED}"
OUT_ROOT="inference_results_${SEED}"
BATCH_SIZE=32

# testing params
UQ_method="mc_dropout" #mc_dropout, base
K_SAMPLES=10

mkdir -p "$CKPT_ROOT"
mkdir -p "$OUT_ROOT"


MODELS="DNABERT2" #"CARMANIA hyenaDNA DNABERT2 NT_transformer"

declare -a EVAL_TASKS=(
    # PBsim reads, taxa family 6kbp
    # "pbsim id_novel_genus 6000 429 family"     
    # "pbsim id_novel_genus 6000 429 order"                
    # "pbsim id_novel_genus 6000 429 class"     
    # "pbsim id_novel_genus 6000 429 phylum"     

    # "pbsim ood_novel_family 6000 429 family"
    # "pbsim ood_novel_family 6000 429 order"             
    # "pbsim ood_novel_family 6000 429 class"             
    # "pbsim ood_novel_family 6000 429 phylum"             
    
    # "pbsim ood_nonbacterial 6000 429 family"              
    # "pbsim ood_nonbacterial 6000 429 order"              
    # "pbsim ood_nonbacterial 6000 429 class"              
    # "pbsim ood_nonbacterial 6000 429 phylum"              
    
    # Scorpio gene_taxa 3kbp
    "gene_taxa test 3000 437 None"           
    "gene_taxa gene_out 3000 437 None"           
    "gene_taxa taxa_out 3000 437 None"           
)


# eval
echo "Starting evaluation phase"
for MODEL in $MODELS
do
    for ENTRY in "${EVAL_TASKS[@]}"; do
        set -- $ENTRY
        TRAIN_TASK="$1"; EVAL_TASK="$2"; MAX_LEN="$3"; NUM_LABELS="$4"; RANK="$5"
        
        if [ "$MODEL" = "DNABERT2" ] ; then
            MAX_LEN=$(( MAX_LEN / 4 ))
        elif [ "$MODEL" = "NT_transformer" ] ; then
            MAX_LEN=$(( MAX_LEN / 6 ))
        fi

        #CKPT_PATH="${CKPT_ROOT}/${MODEL}/${TRAIN_TASK}"
        if [ "$RANK" = "None" ] ; then
            RANK_SUFFIX=""
        else
            RANK_SUFFIX="_${RANK}"
        fi 


        echo "Evaluating ${MODEL} trained on ${TRAIN_TASK} on ${EVAL_TASK}. UQ: ${UQ_method}"
        echo "searching for checkpoints in ${CKPT_ROOT}/${MODEL}/${TRAIN_TASK}${RANK_SUFFIX}/checkpoint-*"

        CKPT_PATH="$(ls -1d "${CKPT_ROOT}/${MODEL}/${TRAIN_TASK}${RANK_SUFFIX}"/checkpoint-* 2>/dev/null | sort -V | tail -n1)"
        EVAL_DATA_PATH="${DATA_ROOT}/${TRAIN_TASK}/${EVAL_TASK}.csv"
        OUT_PATH="${OUT_ROOT}/${MODEL}/${UQ_method}/${TRAIN_TASK}/${EVAL_TASK}${RANK_SUFFIX}"

        echo "found checkpoint ${CKPT_PATH}"

        python -m nn_proj.models.${MODEL}.inference \
            --data_path "${EVAL_DATA_PATH}" \
            --checkpoint "${CKPT_PATH}" \
            --taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
            --taxa_rank ${RANK} \
            --model_max_length "${MAX_LEN}" \
            --run_name "${MODEL}_${TRAIN_TASK}_to_${EVAL_TASK}${RANK}" \
            --per_device_eval_batch_size ${BATCH_SIZE} \
            --num_samples "${K_SAMPLES}" \
            --uncertainty_method "${UQ_method}" \
            --num_labels "${NUM_LABELS}" \
            --output_dir "${OUT_PATH}" \
            --seed ${SEED} \
            --data_seed ${SEED}
    done
done