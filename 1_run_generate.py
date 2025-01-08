from datasets import load_dataset
import os
from pathlib import Path
from glob import glob
import argparse
from vllm import LLM

from utils.util import get_language_from_path
from train.data_generation import generate_rag_ft_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_model_name", default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--default_language", default=None)
    parser.add_argument("--output_dir", default="./data/generated")
    parser.add_argument("--data_path", default="./data/raw/*.parquet")
    parser.add_argument("--max_model_len", default=20_000)
    parser.add_argument("--do_qa_rating", action='store_true', help="Enable QA rating")
    args = parser.parse_args()

    llm_model_name = args.llm_model_name
    default_language = args.default_language
    output_dir = args.output_dir
    data_path = args.data_path
    max_model_len = args.max_model_len

    os.makedirs(output_dir, exist_ok=True)

    llm = LLM(
        model=llm_model_name, 
        max_model_len=max_model_len, 
        tensor_parallel_size=int(os.environ.get("TENSOR_PARALLEL_SIZE", 1)),
    )

    for data_path in sorted(glob(data_path)):
        print(data_path)

        filepath = Path(data_path)
        filename_suffix = filepath.suffix.strip(".")
        filename_stem = filepath.stem

        language = get_language_from_path(data_path) if default_language is None else default_language

        dataset = load_dataset(
            filename_suffix, 
            data_files={"train": data_path}, 
            split="train"
        )
        dataset = generate_rag_ft_data(llm, dataset, language)
        dataset.to_parquet(f"{output_dir}/{filename_stem}.parquet")