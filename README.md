# How to perform ALoFTRAG on your data

In order to make your LLM fit to your text data, you need to first make a parquet file with your text file, with all the text under a column named `context`. Then run the below commands to run data generation and training.

```bash
python 1_run_generate.py --data_path ./path_to_your_parquet_file_of_contexts.parquet --llm_model_name Qwen/Qwen2.5-7B-Instruct --default_language English --output_dir ./data/generated
python 2_prepare_training.py --flagembedding_model_name BAAI/bge-m3 --data_path ./data/generated/*.parquet --output_dir ./data/train --min_q_score 0 --min_a_score 0 --default_language English
python 3_run_training.py --llm_model_name Qwen/Qwen2.5-7B-Instruct --chat_template qwen-7b-chat --training_data_path ./data/train/*.json --output_dir ./models
```

Change the language, LLM name, embedder name, and other parameters as necessary for your use-case.

Your trained model will then be available in `./models`.

# Experiment replication

If you would like to replicate the experiments from the paper, simply run:
```bash
./run_experiments.sh
```
