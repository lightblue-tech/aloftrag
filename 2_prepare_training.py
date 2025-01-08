from FlagEmbedding import BGEM3FlagModel
from datasets import load_dataset
import os
from pathlib import Path
from glob import glob
import argparse

from train.training_data import format_rag_training_data
from utils.util import get_language_from_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flagembedding_model_name", default="BAAI/bge-m3")
    parser.add_argument("--output_dir", default="./data/train")
    parser.add_argument("--data_path", default="./data/generated/*.parquet")
    parser.add_argument("--num_docs", default=10, type=int)
    parser.add_argument("--default_language", default=None)
    parser.add_argument("--min_text_score", default=8, type=float)
    parser.add_argument("--min_q_score", default=8, type=float)
    parser.add_argument("--min_a_score", default=8, type=float)
    args = parser.parse_args()

    flagembedding_model_name = args.flagembedding_model_name
    output_dir = args.output_dir
    data_path = args.data_path
    num_docs = args.num_docs
    default_language = args.default_language
    min_text_score = args.min_text_score
    min_q_score = args.min_q_score
    min_a_score = args.min_a_score

    os.makedirs(output_dir, exist_ok=True)

    embed_model = BGEM3FlagModel(
        flagembedding_model_name, 
        use_fp16=True
    )

    for data_path in sorted(glob(data_path)):
        print(data_path)

        language = get_language_from_path(data_path) if default_language is None else default_language
        
        dataset = load_dataset(
            "parquet", 
            data_files={"train": data_path}, 
            split="train"
        )
        dataset = format_rag_training_data(embed_model, dataset, num_docs, language, min_text_score, min_q_score, min_a_score)
        if dataset is None:
            print(f"Skipping formatting {data_path} as it has too few entries to perform RAG with {num_docs} chunks")
            continue
        dataset.to_json(f"{output_dir}/{Path(data_path).stem}.json")