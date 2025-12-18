source ~/py311-pyt/bin/activate
cd /scratch/home/glh52/temp-glm-epinet

# python -m nn_proj.common.mmseqs\
#   --train_data_path "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/promoter_all" \
#   --test_data_path  "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised/enhancers" \
#   --train_split train \
#   --test_split  test \
#   --outdir mmseqs



python -m nn_proj.common.mmseqs \
  --train_data_path  "/scratch/home/glh52/glm-epinet/DATA/pbsim/train.csv" \
  --test_data_path  "/scratch/home/glh52/glm-epinet/DATA/pbsim/ood_novel_family.csv" \
  --train_split train \
  --test_split  test \
  --outdir mmseqs

