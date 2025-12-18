echo "Training cauduceus on pbsim reads"
export CUDA_VISIBLE_DEVICES=9
export DATA_PATH="/scratch/home/glh52/glm-epinet/DATA/gene_taxa/dev.csv" 
# for DNABERT/ Please set the number as 0.25 * your sequence length. 
# DNABERT values: 1500 for pbsim #750 for gene_taxa
# for NT, set the number as 1/6 * your sequence length
export MAX_LENGTH=500 
export LR=1e-5

# gene 
python -m nn_proj.models.CAUDUCEUS.train_base \
    --data_path  ${DATA_PATH} \
    --run_name DNABERT2_${DATA_PATH} \
    --model_max_length ${MAX_LENGTH} \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LR} \
    --num_train_epochs 1 \
    --fp16 \
    --output_dir trained_models/CAUDUCEUS/gene \
    --eval_strategy epoch \
    --save_strategy epoch \
    --warmup_steps 50 \
    --logging_steps 100 \
    --overwrite_output_dir True \
    --log_level info \
    --find_unused_parameters False
