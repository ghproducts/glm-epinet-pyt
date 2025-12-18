#!/bin/bash
#set -euo pipefail

source ~/py311-pyt/bin/activate
cd /scratch/home/glh52/temp-glm-epinet

export CUDA_VISIBLE_DEVICES=9
export DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/gene_taxa/train.csv" 
# for DNABERT/ Please set the number as 0.25 * your sequence length. 
# DNABERT values: 1500 for pbsim #750 for gene_taxa
# for NT, set the number as 1/6 * your sequence length
export MAX_LENGTH=750
export LR=8e-5

# gene 
python -m nn_proj.models.DNABERT2.train_epinet \
    --data_path  ${DATA_PATH} \
    --checkpoint trained_models/base/dnabert2/gene_taxa/checkpoint-123194 \
    --run_name DNABERT2_${DATA_PATH} \
    --model_max_length ${MAX_LENGTH} \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LR} \
    --num_train_epochs 1 \
    --fp16 \
    --output_dir trained_models/epinet/dnabert2/gene_taxa \
    --eval_strategy epoch \
    --save_strategy epoch \
    --warmup_steps 50 \
    --logging_steps 100 \
    --overwrite_output_dir True \
    --log_level info \
    --find_unused_parameters False \