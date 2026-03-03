#!/bin/bash
#set -euo pipefail

export CUDA_VISIBLE_DEVICES=9
source ~/py311-pyt/bin/activate
cd /scratch/home/glh52/glm-epinet-pyt

# paths 
DATA_ROOT="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"
MODEL="CARMANIA" # DNABERT2, NT_transformer, hyenaDNA, CARMANIA
CKPT_ROOT="trained_models"
OUT_ROOT="inference_results"
BATCH_SIZE=8

# training params
TRAINED=False #or True
LR=1e-6
EPOCHS=5

# testing params
UQ_method="base" # base, epinet, mc_dropout
K_SAMPLES=10


TRAIN_TASK="promoter_all"
MAX_LEN=300

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
    --num_samples "${K_SAMPLES}" \
    --uncertainty_method "${UQ_method}" \
    --output_dir "${OUT_PATH}" 
# 
# echo "Training CARMANIA on genes"
# export CUDA_VISIBLE_DEVICES=5
# export DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/gene_taxa/train.csv" 
# # for DNABERT/ Please set the number as 0.25 * your sequence length. 
# # DNABERT values: 1500 for pbsim #750 for gene_taxa
# # for NT, set the number as 1/6 * your sequence length
# # for hyenaDNA, use full length 6000
# export MAX_LENGTH=3000
# export LR=1e-5
# 
# # gene 
# python -m nn_proj.models.CARMANIA.train_base \
#  --data_path  ${DATA_PATH} \
#  --run_name CARMANIA_${DATA_PATH} \
#  --model_max_length ${MAX_LENGTH} \
#  --per_device_train_batch_size 8 \
#  --per_device_eval_batch_size 8 \
#  --gradient_accumulation_steps 1 \
#  --learning_rate ${LR} \
#  --num_train_epochs 2 \
#  --fp16 \
#  --output_dir trained_models/CARMANIA/gene_taxa \
#  --eval_strategy epoch \
#  --save_strategy epoch \
#  --warmup_steps 50 \
#  --logging_steps 100 \
#  --overwrite_output_dir True \
#  --log_level info \
#  --find_unused_parameters False

# export DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/gene_taxa/dev.csv" 
# # for DNABERT/ Please set the number as 0.25 * your sequence length. 
# # DNABERT values: 1500 for pbsim #750 for gene_taxa
# # for NT, set the number as 1/6 * your sequence length
# export MAX_LENGTH=500 
# export LR=1e-5
# 
# # gene 
# python -m nn_proj.models.CAUDUCEUS.train_base \
#     --data_path  ${DATA_PATH} \
#     --run_name DNABERT2_${DATA_PATH} \
#     --model_max_length ${MAX_LENGTH} \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate ${LR} \
#     --num_train_epochs 1 \
#     --fp16 \
#     --output_dir trained_models/CAUDUCEUS/gene \
#     --eval_strategy epoch \
#     --save_strategy epoch \
#     --warmup_steps 50 \
#     --logging_steps 100 \
#     --overwrite_output_dir True \
#     --log_level info \
#     --find_unused_parameters False
