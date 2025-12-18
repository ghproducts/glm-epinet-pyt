#!/bin/bash
#set -euo pipefail

source ~/py311-pyt/bin/activate
cd /scratch/home/glh52/temp-glm-epinet

export PYTHONUNBUFFERED=1

(	
	echo "Training hyenaDNA on genes"
	export CUDA_VISIBLE_DEVICES=9
	export DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/gene_taxa/train.csv" 
	# PBsim - 6000bp, gene_taxa - 3000bp
	# for DNABERT/ Please set the number as 0.25 * your sequence length. 
	# DNABERT values: 1500 for pbsim #750 for gene_taxa
	# for NT, set the number as 1/6 * your sequence length
	# for hyenaDNA, use full length
	export MAX_LENGTH=3000
	export LR=2e-5

	# gene 
	python -m nn_proj.models.hyenaDNA.train_base \
	    --data_path  ${DATA_PATH} \
	    --run_name hyenaDNA_${DATA_PATH} \
	    --model_max_length ${MAX_LENGTH} \
	    --per_device_train_batch_size 32 \
	    --per_device_eval_batch_size 16 \
	    --gradient_accumulation_steps 1 \
	    --learning_rate ${LR} \
	    --num_train_epochs 2 \
	    --fp16 \
	    --output_dir trained_models/hyenaDNA/gene_taxa \
	    --eval_strategy epoch \
	    --save_strategy epoch \
	    --warmup_steps 50 \
	    --logging_steps 100 \
	    --overwrite_output_dir True \
	    --log_level info \
	    --find_unused_parameters False
) > logs/gene_hyenaDNA.log 2>&1 & pid1=$!

(
    echo "Training hyenaDNAQ on taxa reads"
    export CUDA_VISIBLE_DEVICES=8
    export DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/pbsim/train.csv"
    # for DNABERT/ Please set the number as 0.25 * your sequence length. 
    # DNABERT values: 1500 for pbsim #750 for gene_taxa
    # for NT, set the number as 1/6 * your sequence length
    # for hyenaDNA, use full length 6000
    export MAX_LENGTH=6000
    export LR=2e-5

    python -m nn_proj.models.hyenaDNA.train_base \
        --data_path  ${DATA_PATH} \
        --run_name hyenaDNA_${DATA_PATH} \
        --model_max_length ${MAX_LENGTH} \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --learning_rate ${LR} \
        --num_train_epochs 1 \
        --fp16 \
        --output_dir trained_models/hyenaDNA/pbsim \
        --eval_strategy epoch \
        --save_strategy epoch \
        --warmup_steps 50 \
        --logging_steps 100 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False
) > logs/pbsim_hyenaDNA.log 2>&1 & pid2=$!

# (	
# 	echo "Training DNABERT2 on genes"
# 	export CUDA_VISIBLE_DEVICES=6
# 	export DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/gene_taxa/train.csv" 
# 	# for DNABERT/ Please set the number as 0.25 * your sequence length. 
# 	# DNABERT values: 1500 for pbsim #750 for gene_taxa
# 	# for NT, set the number as 1/6 * your sequence length
# 	# for hyenaDNA, use full length 6000
# 	export MAX_LENGTH=750 
# 	export LR=2e-5
# 
# 	# gene 
# 	python -m nn_proj.models.DNABERT2.train_base \
# 	    --data_path  ${DATA_PATH} \
# 	    --run_name DNABERT2_${DATA_PATH} \
# 	    --model_max_length ${MAX_LENGTH} \
# 	    --per_device_train_batch_size 32 \
# 	    --per_device_eval_batch_size 16 \
# 	    --gradient_accumulation_steps 1 \
# 	    --learning_rate ${LR} \
# 	    --num_train_epochs 2 \
# 	    --fp16 \
# 	    --output_dir trained_models/DNABERT2/gene_taxa \
# 	    --eval_strategy epoch \
# 	    --save_strategy epoch \
# 	    --warmup_steps 50 \
# 	    --logging_steps 100 \
# 	    --overwrite_output_dir True \
# 	    --log_level info \
# 	    --find_unused_parameters False
# ) > logs/gene_dnabert.log 2>&1 & pid1=$!
# 
# (
# echo "Training DNABERT2 on taxa reads"
# export CUDA_VISIBLE_DEVICES=7
# export DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/pbsim/train.csv"
# # for DNABERT/ Please set the number as 0.25 * your sequence length. 
# # DNABERT values: 1500 for pbsim #750 for gene_taxa
# # for NT, set the number as 1/6 * your sequence length
# # for hyenaDNA, use full length 6000
# export MAX_LENGTH=1500 
# export LR=2e-5
# 
# python -m nn_proj.models.DNABERT2.train_base \
#     --data_path  ${DATA_PATH} \
#     --run_name DNABERT2_${DATA_PATH} \
#     --model_max_length ${MAX_LENGTH} \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate ${LR} \
#     --num_train_epochs 1 \
#     --fp16 \
#     --output_dir trained_models/DNABERT2/pbsim \
#     --eval_strategy epoch \
#     --save_strategy epoch \
#     --warmup_steps 50 \
#     --logging_steps 100 \
#     --overwrite_output_dir True \
#     --log_level info \
#     --find_unused_parameters False
# ) > logs/pbsim_dnabert.log 2>&1 & pid2=$!
# 
# (
# 	echo "Training NT transformer on genes"
# 	export CUDA_VISIBLE_DEVICES=8
# 	export DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/gene_taxa/train.csv" 
# 	# for DNABERT/ Please set the number as 0.25 * your sequence length. 
# 	# NT values: 1000 for pbsim #500 for gene_taxa
# 	# for NT, set the number as 1/6 * your sequence length
# 	# for hyenaDNA, use full length 6000
# 	export MAX_LENGTH=500
# 	export LR=2e-5
# 
# 	python -m nn_proj.models.NT_transformer.train_base \
# 	    --data_path  ${DATA_PATH} \
# 	    --run_name DNABERT2_${DATA_PATH} \
# 	    --model_max_length ${MAX_LENGTH} \
# 	    --per_device_train_batch_size 32 \
# 	    --per_device_eval_batch_size 16 \
# 	    --gradient_accumulation_steps 1 \
# 	    --learning_rate ${LR} \
# 	    --num_train_epochs 2 \
# 	    --fp16 \
# 	    --output_dir trained_models/NT_transformer/gene_taxa \
# 	    --eval_strategy epoch \
# 	    --save_strategy epoch \
# 	    --warmup_steps 50 \
# 	    --logging_steps 100 \
# 	    --overwrite_output_dir True \
# 	    --log_level info \
# 	    --find_unused_parameters False
# ) > logs/gene_NT.log 2>&1 & pid3=$!
# 
# (
# 	echo "Training NT transformer on taxa reads"
# 	export CUDA_VISIBLE_DEVICES=9
# 	export DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/pbsim/train.csv" 
# 	# for DNABERT/ Please set the number as 0.25 * your sequence length. 
# 	# DNABERT values: 1000 for pbsim #500 for gene_taxa
# 	# for NT, set the number as 1/6 * your sequence length
# 	# for hyenaDNA, use full length 6000
# 	export MAX_LENGTH=1000 
# 	export LR=1e-5
# 
# 	python -m nn_proj.models.NT_transformer.train_base \
# 	    --data_path  ${DATA_PATH} \
# 	    --run_name DNABERT2_${DATA_PATH} \
# 	    --model_max_length ${MAX_LENGTH} \
# 	    --per_device_train_batch_size 8 \
# 	    --per_device_eval_batch_size 16 \
# 	    --gradient_accumulation_steps 1 \
# 	    --learning_rate ${LR} \
# 	    --num_train_epochs 1 \
# 	    --fp16 \
# 	    --output_dir trained_models/NT_transformer/pbsim \
# 	    --eval_strategy epoch \
# 	    --save_strategy epoch \
# 	    --warmup_steps 50 \
# 	    --logging_steps 100 \
# 	    --overwrite_output_dir True \
# 	    --log_level info \
# 	    --find_unused_parameters False
# ) > logs/pbsim_NT.log 2>&1 & pid4=$!


wait "$pid1" "$pid2" # "$pid3" "$pid4"
echo "PBSIM and gene_taxa base training complete."
