set -euo pipefail

export CUDA_VISIBLE_DEVICES=6
echo "training on GPU: ${CUDA_VISIBLE_DEVICES}"
source ~/py311-pyt/bin/activate
cd /scratch/home/glh52/glm-epinet-pyt

# func to get temp scaling param
get_temp() {
  local model="$1" task="$2"
  awk -F'\t' -v m="$model" -v t="$task" '
    NR>1 && $1==m && $2==t {print $3; exit}
  ' "$TEMP_FILE"
}

# SEED
SEED=4
echo "Seed: ${SEED}"

# paths 
TEMP_FILE="temp_scaling_factors_${SEED}.tsv"
CKPT_ROOT="trained_models_${SEED}"
OUT_ROOT="inference_results_${SEED}"
BATCH_SIZE=32

# testing params
UQ_method="base_scaled" # epinet, mc_dropout, base
temp_scaling=True
K_SAMPLES=1

mkdir -p "$OUT_ROOT"

MODELS="CARMANIA" # DNABERT2, NT_transformer, hyenaDNA, CARMANINT_transformer" # DNABER

DATA_ROOT="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
declare -a REGULATORY_TASKS=(
    # Promoters
    # "promoter_all enhancers 300 2"
    # "promoter_all promoter_all 300 2"

    # # Enhancers 
    # "enhancers_types enhancers_types 400 3"
    # "enhancers_types splice_sites_all 400 3"

    # # Splice sites 
    # "splice_sites_acceptors splice_sites_acceptors 600 2" 
    # "splice_sites_acceptors splice_sites_donors 600 2"
)


# eval
echo "Starting regulatory evaluation phase"

for MODEL in $MODELS
do
    for ENTRY in "${REGULATORY_TASKS[@]}"; do
        set -- $ENTRY
        TRAIN_TASK="$1"; EVAL_TASK="$2"; MAX_LEN="$3"; NUM_LABELS="$4"
        
        echo "Evaluating ${MODEL} trained on ${TRAIN_TASK} on ${EVAL_TASK}"
        echo "searching for checkpoints in ${CKPT_ROOT}/${MODEL}/${TRAIN_TASK}/checkpoint-*"

        CKPT_PATH="$(ls -1d "${CKPT_ROOT}/${MODEL}/${TRAIN_TASK}"/checkpoint-* 2>/dev/null | sort -V | tail -n1)"
        EVAL_DATA_PATH="${DATA_ROOT}/${EVAL_TASK}"
        OUT_PATH="${OUT_ROOT}/${MODEL}/${UQ_method}/${TRAIN_TASK}/${EVAL_TASK}"

        echo "found checkpoint ${CKPT_PATH}"
        TEMP="$(get_temp "$MODEL" "$TRAIN_TASK")"
        echo "T = ${TEMP}"

        python -m nn_proj.models.${MODEL}.inference \
            --data_path "${EVAL_DATA_PATH}" \
            --checkpoint "${CKPT_PATH}" \
            --model_max_length "${MAX_LEN}" \
            --temperature "${TEMP}" \
            --run_name "${MODEL}_${TRAIN_TASK}_to_${EVAL_TASK}" \
            --per_device_eval_batch_size ${BATCH_SIZE} \
            --num_samples "${K_SAMPLES}" \
            --uncertainty_method "${UQ_method}" \
            --num_labels "${NUM_LABELS}" \
            --output_dir "${OUT_PATH}"  \
            --seed ${SEED} \
            --data_seed ${SEED}

    done
done

DATA_ROOT="/scratch/home/glh52/glm-epinet/DATA"
declare -a METAGENOMIC_TASKS=(
    # PBsim reads, taxa family 6kbp
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
    
    # Scorpio gene_taxa 3kbp
    "gene_taxa test 3000 437 None"           
    "gene_taxa gene_out 3000 437 None"           
    "gene_taxa taxa_out 3000 437 None"           
)


# eval
echo "Starting metagenomic evaluation phase"

for MODEL in $MODELS
do
    for ENTRY in "${METAGENOMIC_TASKS[@]}"; do
        set -- $ENTRY
        TRAIN_TASK="$1"; EVAL_TASK="$2"; MAX_LEN="$3"; NUM_LABELS="$4"; RANK="$5"

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

        echo "Evaluating ${MODEL} trained on ${TRAIN_TASK} on ${EVAL_TASK}"
        echo "searching for checkpoints in ${CKPT_ROOT}/${MODEL}/${TRAIN_TASK}${RANK_SUFFIX}/checkpoint-*"

        CKPT_PATH="$(ls -1d "${CKPT_ROOT}/${MODEL}/${TRAIN_TASK}${RANK_SUFFIX}"/checkpoint-* 2>/dev/null | sort -V | tail -n1)"
        EVAL_DATA_PATH="${DATA_ROOT}/${TRAIN_TASK}/${EVAL_TASK}.csv"
        OUT_PATH="${OUT_ROOT}/${MODEL}/${UQ_method}/${TRAIN_TASK}/${EVAL_TASK}${RANK_SUFFIX}"

        echo "found checkpoint ${CKPT_PATH}"
        TEMP="$(get_temp "$MODEL" "${TRAIN_TASK}${RANK_SUFFIX}")"

        echo "T = ${TEMP}"

        python -m nn_proj.models.${MODEL}.inference \
            --data_path "${EVAL_DATA_PATH}" \
            --checkpoint "${CKPT_PATH}" \
        	--taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
        	--taxa_rank ${RANK} \
            --model_max_length "${MAX_LEN}" \
            --temperature "${TEMP}"\
            --run_name "${MODEL}_${TRAIN_TASK}_to_${EVAL_TASK}${RANK}" \
            --per_device_eval_batch_size ${BATCH_SIZE} \
            --num_samples "${K_SAMPLES}" \
            --uncertainty_method "${UQ_method}" \
            --num_labels "${NUM_LABELS}" \
            --output_dir "${OUT_PATH}"  \
            --seed ${SEED} \
            --data_seed ${SEED}

    done
done