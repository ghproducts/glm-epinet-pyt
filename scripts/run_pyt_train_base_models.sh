#!/bin/bash
#set -euo pipefail

source ~/py311-pyt/bin/activate
cd /scratch/home/glh52/glm-epinet-pyt

export PYTHONUNBUFFERED=1
SEED=2
echo "Training gene_taxa and PBsim on seed: ${SEED}"

# Gene training
# (	
# 	export CUDA_VISIBLE_DEVICES=0
# 	export DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/gene_taxa/train.csv" 
# 	export LR=2e-5
# 	# PBsim - 6000bp, gene_taxa - 3000bp
# 	# for DNABERT/ Please set the number as 0.25 * your sequence length. 
# 	# DNABERT values: 1500 for pbsim #750 for gene_taxa
# 	# for NT, set the number as 1/6 * your sequence length
# 	# for hyenaDNA, use full length
# 	
# 	# hyenaDNA ============================================================	
# 	echo "Training hyenaDNA on genes"
# 	export MAX_LENGTH=3000
# 
# 	python -m nn_proj.models.hyenaDNA.train_base \
# 	    --data_path  ${DATA_PATH} \
# 	    --run_name hyenaDNA_${DATA_PATH} \
# 	    --model_max_length ${MAX_LENGTH} \
# 	    --per_device_train_batch_size 32 \
# 	    --per_device_eval_batch_size 16 \
# 	    --gradient_accumulation_steps 1 \
# 	    --learning_rate ${LR} \
# 	    --num_train_epochs 2 \
# 	    --fp16 \
# 	    --output_dir trained_models_${SEED}/hyenaDNA/gene_taxa \
# 	    --eval_strategy epoch \
# 	    --save_strategy epoch \
# 	    --warmup_steps 50 \
# 	    --logging_steps 100 \
# 	    --overwrite_output_dir True \
# 	    --log_level info \
# 	    --find_unused_parameters False \
# 		--seed ${SEED} \
# 		--data_seed ${SEED}
# 
# 
# 	# CARMANIA ================================================
#  	echo "Training CARMANIA on genes"
#  	export MAX_LENGTH=3000
#  
#  	python -m nn_proj.models.CARMANIA.train_base \
#  	    --data_path  ${DATA_PATH} \
#  	    --run_name CARMANIA_${DATA_PATH} \
#  	    --model_max_length ${MAX_LENGTH} \
#  	    --per_device_train_batch_size 16 \
#  	    --per_device_eval_batch_size 16 \
#  	    --gradient_accumulation_steps 1 \
#  	    --learning_rate ${LR} \
#  	    --num_train_epochs 2 \
#  	    --fp16 \
#  	    --output_dir trained_models_${SEED}/CARMANIA/gene_taxa \
#  	    --eval_strategy epoch \
#  	    --save_strategy epoch \
#  	    --warmup_steps 50 \
#  	    --logging_steps 100 \
#  	    --overwrite_output_dir True \
#  	    --log_level info \
#  	    --find_unused_parameters False \
# 		--seed ${SEED} \
# 		--data_seed ${SEED}
# 
# 
# 	# DNABERT2 ============================================
# 	echo "Training DNABERT2 on genes"
# 	export MAX_LENGTH=750 
# 
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
# 	    --output_dir trained_models_${SEED}/DNABERT2/gene_taxa \
# 	    --eval_strategy epoch \
# 	    --save_strategy epoch \
# 	    --warmup_steps 50 \
# 	    --logging_steps 100 \
# 	    --overwrite_output_dir True \
# 	    --log_level info \
# 	    --find_unused_parameters False \
# 		--seed ${SEED} \
# 		--data_seed ${SEED}
# 
# 	# NT transformer
#  	echo "Training NT_transformer on genes"
#  	export MAX_LENGTH=500
#  
#  	python -m nn_proj.models.NT_transformer.train_base \
#  	    --data_path  ${DATA_PATH} \
#  	    --run_name NT_transformer_${DATA_PATH} \
#  	    --model_max_length ${MAX_LENGTH} \
#  	    --per_device_train_batch_size 32 \
#  	    --per_device_eval_batch_size 16 \
#  	    --gradient_accumulation_steps 1 \
#  	    --learning_rate ${LR} \
#  	    --num_train_epochs 2 \
#  	    --fp16 \
#  	    --output_dir trained_models_${SEED}/NT_transformer/gene_taxa \
#  	    --eval_strategy epoch \
#  	    --save_strategy epoch \
#  	    --warmup_steps 50 \
#  	    --logging_steps 100 \
#  	    --overwrite_output_dir True \
#  	    --log_level info \
#  	    --find_unused_parameters False \
# 		--seed ${SEED} \
# 		--data_seed ${SEED}
# 
# 
# ) > logs/gene_${SEED}.log 2>&1 & pid5=$!

# # family-level training
# (
#     export CUDA_VISIBLE_DEVICES=6
#     export DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/pbsim/train.csv"
#     # for DNABERT/ Please set the number as 0.25 * your sequence length. 
#     # DNABERT values: 1500 for pbsim #750 for gene_taxa
#     # for NT, set the number as 1/6 * your sequence length
#     # for hyenaDNA, use full length 6000
# 	export rank="family"
# 
# 	# CARMANIA ========================================================
#     echo "Training CARMANIA on taxa reads: rank=${rank}"
#     export MAX_LENGTH=6000
#     export LR=1e-5
#     python -m nn_proj.models.CARMANIA.train_base \
#     	--data_path  ${DATA_PATH} \
#     	--taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
#     	--taxa_rank ${rank} \
#     	--run_name CARMANIA_${DATA_PATH} \
#     	--model_max_length ${MAX_LENGTH} \
#     	--per_device_train_batch_size 8 \
#     	--per_device_eval_batch_size 8 \
#     	--gradient_accumulation_steps 1 \
#     	--learning_rate ${LR} \
#     	--num_train_epochs 2 \
#     	--fp16 \
#     	--output_dir trained_models_${SEED}/CARMANIA/pbsim_${rank} \
#     	--eval_strategy epoch \
#     	--save_strategy epoch \
#     	--warmup_steps 50 \
#     	--logging_steps 100 \
#     	--overwrite_output_dir True \
#     	--log_level info \
#     	--find_unused_parameters False \
# 		--seed ${SEED} \
# 		--data_seed ${SEED}
# 
# 	# HYENADNA ========================================================
#     echo "Training hyenaDNA on taxa reads: rank=${rank}"
#     export MAX_LENGTH=6000
#     export LR=2e-5
#     python -m nn_proj.models.hyenaDNA.train_base \
#     	--data_path  ${DATA_PATH} \
#     	--taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
#     	--taxa_rank ${rank} \
#     	--run_name hyenaDNA_${DATA_PATH} \
#     	--model_max_length ${MAX_LENGTH} \
#     	--per_device_train_batch_size 32 \
#     	--per_device_eval_batch_size 32 \
#     	--gradient_accumulation_steps 1 \
#     	--learning_rate ${LR} \
#     	--num_train_epochs 2 \
#     	--fp16 \
#     	--output_dir trained_models_${SEED}/hyenaDNA/pbsim_${rank} \
#     	--eval_strategy epoch \
#     	--save_strategy epoch \
#     	--warmup_steps 50 \
#     	--logging_steps 100 \
#     	--overwrite_output_dir True \
#     	--log_level info \
#     	--find_unused_parameters False \
# 		--seed ${SEED} \
# 		--data_seed ${SEED}
# 
# 	# DNABERT2 ========================================================
# 	echo "Training DNABERT2 on taxa reads: rank=${rank}"
# 	export MAX_LENGTH=1500
# 	export LR=2e-5
# 	python -m nn_proj.models.DNABERT2.train_base \
# 		--data_path  ${DATA_PATH} \
# 		--taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
# 		--taxa_rank ${rank} \
# 		--run_name DNABERT2_${DATA_PATH} \
# 		--model_max_length ${MAX_LENGTH} \
# 		--per_device_train_batch_size 32 \
# 		--per_device_eval_batch_size 32 \
# 		--gradient_accumulation_steps 1 \
# 		--learning_rate ${LR} \
# 		--num_train_epochs 2 \
# 		--fp16 \
# 		--output_dir trained_models_${SEED}/DNABERT2/pbsim_${rank} \
# 		--eval_strategy epoch \
# 		--save_strategy epoch \
# 		--warmup_steps 50 \
# 		--logging_steps 100 \
# 		--overwrite_output_dir True \
# 		--log_level info \
# 		--find_unused_parameters False \
# 		--seed ${SEED} \
# 		--data_seed ${SEED}
# 
# 	# NT TRANSFORMER ========================================================
# 	echo "Training NT transformer on taxa reads: rank=${rank}"
# 	export MAX_LENGTH=1000
# 	export LR=1e-5
# 	python -m nn_proj.models.NT_transformer.train_base \
# 		--data_path  ${DATA_PATH} \
# 		--taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
# 		--taxa_rank ${rank} \
# 		--run_name NT_${DATA_PATH} \
# 		--model_max_length ${MAX_LENGTH} \
# 		--per_device_train_batch_size 8 \
# 		--per_device_eval_batch_size 8 \
# 		--gradient_accumulation_steps 1 \
# 		--learning_rate ${LR} \
# 		--num_train_epochs 2 \
# 		--fp16 \
# 		--output_dir trained_models_${SEED}/NT_transformer/pbsim_${rank} \
# 		--eval_strategy epoch \
# 		--save_strategy epoch \
# 		--warmup_steps 50 \
# 		--logging_steps 100 \
# 		--overwrite_output_dir True \
# 		--log_level info \
# 		--find_unused_parameters False \
# 		--seed ${SEED} \
# 		--data_seed ${SEED}
# 
# ) > logs/pbsim_family_${SEED}.log 2>&1 & pid1=$!

# order-level training
(
    export CUDA_VISIBLE_DEVICES=2
    export DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/pbsim/train.csv"
    # for DNABERT/ Please set the number as 0.25 * your sequence length. 
    # DNABERT values: 1500 for pbsim #750 for gene_taxa
    # for NT, set the number as 1/6 * your sequence length
    # for hyenaDNA, use full length 6000
	export rank="order"
    
	# # CARMANIA ========================================================
	# echo "Training CARMANIA on taxa reads: rank=${rank}"
	# export MAX_LENGTH=6000
	# export LR=1e-5
	# python -m nn_proj.models.CARMANIA.train_base \
	# 	--data_path  ${DATA_PATH} \
	# 	--taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
	# 	--taxa_rank ${rank} \
	# 	--run_name CARMANIA_${DATA_PATH} \
	# 	--model_max_length ${MAX_LENGTH} \
	# 	--per_device_train_batch_size 8 \
	# 	--per_device_eval_batch_size 8 \
	# 	--gradient_accumulation_steps 1 \
	# 	--learning_rate ${LR} \
	# 	--num_train_epochs 2 \
	# 	--fp16 \
	# 	--output_dir trained_models_${SEED}/CARMANIA/pbsim_${rank} \
	# 	--eval_strategy epoch \
	# 	--save_strategy epoch \
	# 	--warmup_steps 50 \
	# 	--logging_steps 100 \
	# 	--overwrite_output_dir True \
	# 	--log_level info \
	# 	--find_unused_parameters False \
	# 	--seed ${SEED} \
	# 	--data_seed ${SEED}

	# # HYENADNA ========================================================
	# echo "Training hyenaDNA on taxa reads: rank=${rank}"
	# export MAX_LENGTH=6000
	# export LR=2e-5
	# python -m nn_proj.models.hyenaDNA.train_base \
	# 	--data_path  ${DATA_PATH} \
	# 	--taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
	# 	--taxa_rank ${rank} \
	# 	--run_name hyenaDNA_${DATA_PATH} \
	# 	--model_max_length ${MAX_LENGTH} \
	# 	--per_device_train_batch_size 32 \
	# 	--per_device_eval_batch_size 32 \
	# 	--gradient_accumulation_steps 1 \
	# 	--learning_rate ${LR} \
	# 	--num_train_epochs 3 \
	# 	--fp16 \
	# 	--output_dir trained_models_${SEED}/hyenaDNA/pbsim_${rank} \
	# 	--eval_strategy epoch \
	# 	--save_strategy epoch \
	# 	--warmup_steps 50 \
	# 	--logging_steps 100 \
	# 	--overwrite_output_dir True \
	# 	--log_level info \
	# 	--find_unused_parameters False \
	# 	--seed ${SEED} \
	# 	--data_seed ${SEED}

	# # # DNABERT2 ========================================================
	# echo "Training DNABERT2 on taxa reads: rank=${rank}"
	# export MAX_LENGTH=1500
	# export LR=2e-5
	# python -m nn_proj.models.DNABERT2.train_base \
	# 	--data_path  ${DATA_PATH} \
	# 	--taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
	# 	--taxa_rank ${rank} \
	# 	--run_name DNABERT2_${DATA_PATH} \
	# 	--model_max_length ${MAX_LENGTH} \
	# 	--per_device_train_batch_size 32 \
	# 	--per_device_eval_batch_size 32 \
	# 	--gradient_accumulation_steps 1 \
	# 	--learning_rate ${LR} \
	# 	--num_train_epochs 2 \
	# 	--fp16 \
	# 	--output_dir trained_models_${SEED}/DNABERT2/pbsim_${rank} \
	# 	--eval_strategy epoch \
	# 	--save_strategy epoch \
	# 	--warmup_steps 50 \
	# 	--logging_steps 100 \
	# 	--overwrite_output_dir True \
	# 	--log_level info \
	# 	--find_unused_parameters False \
	# 	--seed ${SEED} \
	# 	--data_seed ${SEED}

	# NT TRANSFORMER ========================================================
	echo "Training NT transformer on taxa reads: rank=${rank}"
	export MAX_LENGTH=1000
	export LR=1e-5
	python -m nn_proj.models.NT_transformer.train_base \
		--data_path  ${DATA_PATH} \
		--taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
		--taxa_rank ${rank} \
		--run_name NT_${DATA_PATH} \
		--model_max_length ${MAX_LENGTH} \
		--per_device_train_batch_size 8 \
		--per_device_eval_batch_size 8 \
		--gradient_accumulation_steps 1 \
		--learning_rate ${LR} \
		--num_train_epochs 2 \
		--fp16 \
		--output_dir trained_models_${SEED}/NT_transformer/pbsim_${rank} \
		--eval_strategy epoch \
		--save_strategy epoch \
		--warmup_steps 50 \
		--logging_steps 100 \
		--overwrite_output_dir True \
		--log_level info \
		--find_unused_parameters False \
		--seed ${SEED} \
		--data_seed ${SEED}

) > logs/pbsim_order_${SEED}.log 2>&1 & pid2=$!

# class-level training
# (
#     export CUDA_VISIBLE_DEVICES=5
#     export DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/pbsim/train.csv"
#     # for DNABERT/ Please set the number as 0.25 * your sequence length. 
#     # DNABERT values: 1500 for pbsim #750 for gene_taxa
#     # for NT, set the number as 1/6 * your sequence length
#     # for hyenaDNA, use full length 6000
# 	export rank="class"
# 	
#     # # CARMANIA ========================================================
#     # echo "Training CARMANIA on taxa reads: rank=${rank}"
#     # export MAX_LENGTH=6000
#     # export LR=1e-5
#     # python -m nn_proj.models.CARMANIA.train_base \
#     #     --data_path  ${DATA_PATH} \
#     #     --taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
#     #     --taxa_rank ${rank} \
#     #     --run_name CARMANIA_${DATA_PATH} \
#     #     --model_max_length ${MAX_LENGTH} \
#     #     --per_device_train_batch_size 8 \
#     #     --per_device_eval_batch_size 8 \
#     #     --gradient_accumulation_steps 1 \
#     #     --learning_rate ${LR} \
#     #     --num_train_epochs 2 \
#     #     --fp16 \
#     #     --output_dir trained_models_${SEED}/CARMANIA/pbsim_${rank} \
#     #     --eval_strategy epoch \
#     #     --save_strategy epoch \
#     #     --warmup_steps 50 \
#     #     --logging_steps 100 \
#     #     --overwrite_output_dir True \
#     #     --log_level info \
#     #     --find_unused_parameters False \
#     #     --seed ${SEED} \
#     #     --data_seed ${SEED}
# 	
#     # # #HYENADNA ========================================================
#     # echo "Training hyenaDNA on taxa reads: rank=${rank}"
#     # export MAX_LENGTH=6000
#     # export LR=2e-5
#     # python -m nn_proj.models.hyenaDNA.train_base \
#     #     --data_path  ${DATA_PATH} \
#     #     --taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
#     #     --taxa_rank ${rank} \
#     #     --run_name hyenaDNA_${DATA_PATH} \
#     #     --model_max_length ${MAX_LENGTH} \
#     #     --per_device_train_batch_size 32 \
#     #     --per_device_eval_batch_size 32 \
#     #     --gradient_accumulation_steps 1 \
#     #     --learning_rate ${LR} \
#     #     --num_train_epochs 3 \
#     #     --fp16 \
#     #     --output_dir trained_models_${SEED}/hyenaDNA/pbsim_${rank} \
#     #     --eval_strategy epoch \
#     #     --save_strategy epoch \
#     #     --warmup_steps 50 \
#     #     --logging_steps 100 \
#     #     --overwrite_output_dir True \
#     #     --log_level info \
#     #     --find_unused_parameters False \
#     #     --seed ${SEED} \
#     #     --data_seed ${SEED}
# 	 
#     # # # DNABERT2 ========================================================
#     # echo "Training DNABERT2 on taxa reads: rank=${rank}"
#     # export MAX_LENGTH=1500
#     # export LR=2e-5
#     # python -m nn_proj.models.DNABERT2.train_base \
#     #     --data_path  ${DATA_PATH} \
#     #     --taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
#     #     --taxa_rank ${rank} \
#     #     --run_name DNABERT2_${DATA_PATH} \
#     #     --model_max_length ${MAX_LENGTH} \
#     #     --per_device_train_batch_size 32 \
#     #     --per_device_eval_batch_size 32 \
#     #     --gradient_accumulation_steps 1 \
#     #     --learning_rate ${LR} \
#     #     --num_train_epochs 2 \
#     #     --fp16 \
#     #     --output_dir trained_models_${SEED}/DNABERT2/pbsim_${rank} \
#     #     --eval_strategy epoch \
#     #     --save_strategy epoch \
#     #     --warmup_steps 50 \
#     #     --logging_steps 100 \
#     #     --overwrite_output_dir True \
#     #     --log_level info \
#     #     --find_unused_parameters False \
#     #     --seed ${SEED} \
#     #     --data_seed ${SEED}
# 	
# 	# NT TRANSFORMER ========================================================
# 	echo "Training NT transformer on taxa reads: rank=${rank}"
# 	export MAX_LENGTH=1000
# 	export LR=1e-5
# 	python -m nn_proj.models.NT_transformer.train_base \
# 		--data_path  ${DATA_PATH} \
# 		--taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
# 		--taxa_rank ${rank} \
# 		--run_name NT_${DATA_PATH} \
# 		--model_max_length ${MAX_LENGTH} \
# 		--per_device_train_batch_size 8 \
# 		--per_device_eval_batch_size 8 \
# 		--gradient_accumulation_steps 1 \
# 		--learning_rate ${LR} \
# 		--num_train_epochs 2 \
# 		--fp16 \
# 		--output_dir trained_models_${SEED}/NT_transformer/pbsim_${rank} \
# 		--eval_strategy epoch \
# 		--save_strategy epoch \
# 		--warmup_steps 50 \
# 		--logging_steps 100 \
# 		--overwrite_output_dir True \
# 		--log_level info \
# 		--find_unused_parameters False \
# 		--seed ${SEED} \
# 		--data_seed ${SEED}
# 
# ) > logs/pbsim_class_${SEED}.log 2>&1 & pid3=$!

# phylum-level training
# (
#     export CUDA_VISIBLE_DEVICES=9
#     export DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/pbsim/train.csv"
#     # for DNABERT/ Please set the number as 0.25 * your sequence length. 
#     # DNABERT values: 1500 for pbsim #750 for gene_taxa
#     # for NT, set the number as 1/6 * your sequence length
#     # for hyenaDNA, use full length 6000
# 	export rank="phylum"
# 
# 
# 	# CARMANIA ========================================================
#     echo "Training CARMANIA on taxa reads: rank=${rank}"
#     export MAX_LENGTH=6000
#     export LR=1e-5
#     python -m nn_proj.models.CARMANIA.train_base \
#     	--data_path  ${DATA_PATH} \
#     	--taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
#     	--taxa_rank ${rank} \
#     	--run_name CARMANIA_${DATA_PATH} \
#     	--model_max_length ${MAX_LENGTH} \
#     	--per_device_train_batch_size 8 \
#     	--per_device_eval_batch_size 8 \
#     	--gradient_accumulation_steps 1 \
#     	--learning_rate ${LR} \
#     	--num_train_epochs 2 \
#     	--fp16 \
#     	--output_dir trained_models_${SEED}/CARMANIA/pbsim_${rank} \
#     	--eval_strategy epoch \
#     	--save_strategy epoch \
#     	--warmup_steps 50 \
#     	--logging_steps 100 \
#     	--overwrite_output_dir True \
#     	--log_level info \
#     	--find_unused_parameters False \
# 		--seed ${SEED} \
# 		--data_seed ${SEED}
# 
# 	# HYENADNA ========================================================
# 	echo "Training hyenaDNA on taxa reads: rank=${rank}"
# 	export MAX_LENGTH=6000
# 	export LR=2e-5
# 	python -m nn_proj.models.hyenaDNA.train_base \
# 		--data_path  ${DATA_PATH} \
# 		--taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
# 		--taxa_rank ${rank} \
# 		--run_name hyenaDNA_${DATA_PATH} \
# 		--model_max_length ${MAX_LENGTH} \
# 		--per_device_train_batch_size 32 \
# 		--per_device_eval_batch_size 32 \
# 		--gradient_accumulation_steps 1 \
# 		--learning_rate ${LR} \
# 		--num_train_epochs 3 \
# 		--fp16 \
# 		--output_dir trained_models_${SEED}/hyenaDNA/pbsim_${rank} \
# 		--eval_strategy epoch \
# 		--save_strategy epoch \
# 		--warmup_steps 50 \
# 		--logging_steps 100 \
# 		--overwrite_output_dir True \
# 		--log_level info \
# 		--find_unused_parameters False \
# 		--seed ${SEED} \
# 		--data_seed ${SEED}
# 
# 	# DNABERT2 ========================================================
# 	echo "Training DNABERT2 on taxa reads: rank=${rank}"
# 	export MAX_LENGTH=1500
# 	export LR=2e-5
# 	python -m nn_proj.models.DNABERT2.train_base \
# 		--data_path  ${DATA_PATH} \
# 		--taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
# 		--taxa_rank ${rank} \
# 		--run_name DNABERT2_${DATA_PATH} \
# 		--model_max_length ${MAX_LENGTH} \
# 		--per_device_train_batch_size 32 \
# 		--per_device_eval_batch_size 32 \
# 		--gradient_accumulation_steps 1 \
# 		--learning_rate ${LR} \
# 		--num_train_epochs 2 \
# 		--fp16 \
# 		--output_dir trained_models_${SEED}/DNABERT2/pbsim_${rank} \
# 		--eval_strategy epoch \
# 		--save_strategy epoch \
# 		--warmup_steps 50 \
# 		--logging_steps 100 \
# 		--overwrite_output_dir True \
# 		--log_level info \
# 		--find_unused_parameters False \
# 		--seed ${SEED} \
# 		--data_seed ${SEED}
# 
# 	# NT TRANSFORMER ========================================================
# 	echo "Training NT transformer on taxa reads: rank=${rank}"
# 	export MAX_LENGTH=1000
# 	export LR=1e-5
# 	python -m nn_proj.models.NT_transformer.train_base \
# 		--data_path  ${DATA_PATH} \
# 		--taxa_df /scratch/home/glh52/glm-epinet/DATA/pbsim/full_basic_lineage.csv \
# 		--taxa_rank ${rank} \
# 		--run_name NT_${DATA_PATH} \
# 		--model_max_length ${MAX_LENGTH} \
# 		--per_device_train_batch_size 8 \
# 		--per_device_eval_batch_size 8 \
# 		--gradient_accumulation_steps 1 \
# 		--learning_rate ${LR} \
# 		--num_train_epochs 2 \
# 		--fp16 \
# 		--output_dir trained_models_${SEED}/NT_transformer/pbsim_${rank} \
# 		--eval_strategy epoch \
# 		--save_strategy epoch \
# 		--warmup_steps 50 \
# 		--logging_steps 100 \
# 		--overwrite_output_dir True \
# 		--log_level info \
# 		--find_unused_parameters False \
# 		--seed ${SEED} \
# 		--data_seed ${SEED}
# 
# ) > logs/pbsim_phylum_${SEED}.log 2>&1 & pid4=$!


wait "$pid2" "$pid3" #"$pid3" "$pid4" "$pid5"
echo "PBSIM and gene_taxa base training complete."
