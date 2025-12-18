#!/bin/bash
#set -euo pipefail

source ~/py311-pyt/bin/activate
cd /scratch/home/glh52/temp-glm-epinet

export CUDA_VISIBLE_DEVICES=8
export DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/gene_taxa/dev.csv" 
# for DNABERT/ Please set the number as 0.25 * your sequence length. 
# DNABERT values: 1500 for pbsim #750 for gene_taxa
# for NT, set the number as 1/6 * your sequence length
export MAX_LENGTH=750


python -m nn_proj.models.DNABERT2.inference \
    --data_path  ${DATA_PATH} \
    --checkpoint trained_models/base/dnabert2/gene_taxa/checkpoint-123194 \
    --model_max_length ${MAX_LENGTH} \
    --output_dir inference_results/gene_taxa \
    --uncertainty_method mc_dropout \
    --num_samples 5 \
    --num_labels 437 \
    --per_device_eval_batch_size 8 \