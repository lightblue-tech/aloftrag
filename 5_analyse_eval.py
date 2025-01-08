from datasets import load_dataset
import os
from pathlib import Path
from glob import glob
import argparse

from openai import AzureOpenAI, BadRequestError

from eval.analysis import run_answer_analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dataset_dir", default="./data/eval/*.parquet")
    parser.add_argument("--eval_analysis_dir", default="./data/analysis/")

    args = parser.parse_args()
    
    eval_dataset_dir = args.eval_dataset_dir
    eval_analysis_dir = args.eval_analysis_dir

    client = AzureOpenAI(
        api_key=os.environ["AZURE_API_KEY"],
        api_version = "2024-04-01-preview",
        azure_endpoint = os.environ["AZURE_ENDPOINT"],
    )

    os.makedirs(eval_analysis_dir, exist_ok=True)

    for data_path in sorted(glob(eval_dataset_dir)):
        print(data_path)

        filepath = Path(data_path)
        filename_suffix = filepath.suffix.strip(".")
        filename_stem = filepath.stem

        dataset = load_dataset(
            filename_suffix, 
            data_files={"train": data_path}, 
            split="train"
        )

        dataset = dataset.map(lambda x: {"lora_answer_analysis": run_answer_analysis(client, x, "lora")}, num_proc=4)
        if "base_gen_ans" in dataset[0].keys():
            dataset = dataset.map(lambda x: {"base_answer_analysis": run_answer_analysis(client, x, "base")}, num_proc=4)

        dataset.to_parquet(f"{eval_analysis_dir}/{Path(data_path).stem}.parquet")