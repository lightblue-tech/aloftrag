# Download datasets
HF_HOME=/workspace/hf_home/  python 0_download_datasets.py

# Main experiment with step 1 and 3 included
HF_HOME=/workspace/hf_home/ TENSOR_PARALLEL_SIZE=4 python loftrag/1_run_generate.py --do_qa_rating
HF_HOME=/workspace/hf_home/ TENSOR_PARALLEL_SIZE=4 python loftrag/2_prepare_training.py
HF_HOME=/workspace/hf_home/ TENSOR_PARALLEL_SIZE=4 python loftrag/3_run_training.py --wandb_username peterd
HF_HOME=/workspace/hf_home/ TENSOR_PARALLEL_SIZE=4 python loftrag/4_run_evaluation.py

# Experiment with step 1 excluded
HF_HOME=/workspace/hf_home/ TENSOR_PARALLEL_SIZE=4 python loftrag/2_prepare_training.py --min_text_score 0 --output_dir ./data_notxtfilt/train
HF_HOME=/workspace/hf_home/ TENSOR_PARALLEL_SIZE=4 python loftrag/3_run_training.py --wandb_username peterd --training_data_path ./data_notxtfilt/train/\*.json --output_dir ./models_notxtfilt
HF_HOME=/workspace/hf_home/ TENSOR_PARALLEL_SIZE=4 python loftrag/4_run_evaluation.py --adapter_paths ./models_notxtfilt/model_weights/\* --output_dir ./data_notxtfilt/eval

# Experiment with step 3 excluded
HF_HOME=/workspace/hf_home/ TENSOR_PARALLEL_SIZE=4 python loftrag/2_prepare_training.py --min_q_score 0 --min_a_score 0 --output_dir ./data_noqafilt/train
HF_HOME=/workspace/hf_home/ TENSOR_PARALLEL_SIZE=4 python loftrag/3_run_training.py --wandb_username peterd --training_data_path ./data_noqafilt/train/\*.json --output_dir ./models_noqafilt
HF_HOME=/workspace/hf_home/ TENSOR_PARALLEL_SIZE=4 python loftrag/4_run_evaluation.py --adapter_paths ./models_noqafilt/model_weights/\* --output_dir ./data_noqafilt/eval

# Experiment with 2 chunks instead of 10
HF_HOME=/workspace/hf_home/ TENSOR_PARALLEL_SIZE=4 python loftrag/2_prepare_training.py --output_dir ./data_2chunk/train --num_docs 2
HF_HOME=/workspace/hf_home/ TENSOR_PARALLEL_SIZE=4 python loftrag/3_run_training.py --wandb_username peterd --training_data_path ./data_2chunk/train/\*.json --output_dir ./models_2chunk
HF_HOME=/workspace/hf_home/ TENSOR_PARALLEL_SIZE=4 python loftrag/4_run_evaluation.py --adapter_paths ./models_2chunk/model_weights/\* --output_dir ./data_2chunk/eval  --num_docs 2 --do_run_base

# Experiment with 5 chunks instead of 10
HF_HOME=/workspace/hf_home/ TENSOR_PARALLEL_SIZE=4 python loftrag/2_prepare_training.py --output_dir ./data_5chunk/train --num_docs 5
HF_HOME=/workspace/hf_home/ TENSOR_PARALLEL_SIZE=4 python loftrag/3_run_training.py --wandb_username peterd --training_data_path ./data_5chunk/train/\*.json --output_dir ./models_5chunk
HF_HOME=/workspace/hf_home/ TENSOR_PARALLEL_SIZE=4 python loftrag/4_run_evaluation.py --adapter_paths ./models_5chunk/model_weights/\* --output_dir ./data_5chunk/eval  --num_docs 5 --do_run_base