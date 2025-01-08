from vllm import LLM
from datasets import load_dataset
import os
from pathlib import Path
from glob import glob
import argparse

from utils.util import get_language_from_path
from eval.eval import add_contexts, add_gen_answers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flagembedding_model_name", default="BAAI/bge-m3")
    parser.add_argument("--llm_model_name", default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--adapter_paths", default="./models/model_weights/*")
    parser.add_argument("--output_dir", default="./data/eval")
    parser.add_argument("--eval_dataset_dir", default="./data/raw/*.parquet")
    parser.add_argument("--num_docs", default=10, type=int)
    parser.add_argument("--max_model_len", default=20_000, type=int)
    parser.add_argument("--default_language", default=None)
    parser.add_argument("--do_run_base", action='store_true', help="Also run inference on base model")

    args = parser.parse_args()
    
    flagembedding_model_name = args.flagembedding_model_name
    llm_model_name = args.llm_model_name
    adapter_paths = args.adapter_paths
    output_dir = args.output_dir
    eval_dataset_dir = args.eval_dataset_dir
    num_docs = args.num_docs
    max_model_len = args.max_model_len
    default_language = args.default_language
    do_run_base = args.do_run_base

    os.makedirs(output_dir, exist_ok=True)

    llm = LLM(
        model=llm_model_name, 
        max_model_len=max_model_len, 
        tensor_parallel_size=int(os.environ.get("TENSOR_PARALLEL_SIZE", 1)),
        gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", 0.7)), # By default, we make the GPU utilization 0.7 so we leave space to do text embedding too.
        enable_lora=True,
        max_lora_rank=64,
    )

    from FlagEmbedding import BGEM3FlagModel

    embed_model = BGEM3FlagModel(
        flagembedding_model_name, 
        use_fp16=True
    )

    adapter_paths = sorted(glob(adapter_paths))
    eval_dataset_dir = sorted(glob(eval_dataset_dir))

    # Find the dataset that corresponds with the adapter path. They should both have the same name.gh
    find_dataset = lambda adapter_path: [x for x in eval_dataset_dir if Path(x).stem == os.path.basename(adapter_path)][0]
    adapter_eval_datasets_paired = [(x, find_dataset(x)) for x in adapter_paths]

    for adapter_path, data_path in adapter_eval_datasets_paired:
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
        
        tokenizer = llm.llm_engine.tokenizer.tokenizer
        dataset = add_contexts(dataset, embed_model, num_docs, tokenizer, max_model_len)
        dataset = add_gen_answers(dataset, llm, language, adapter_path, do_run_base=do_run_base)
        dataset.to_parquet(f"{output_dir}/{Path(data_path).stem}.parquet")