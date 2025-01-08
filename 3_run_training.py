import os
from pathlib import Path
from glob import glob
import argparse

from utils.util import make_axolotl_yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_model_name", default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--chat_template", default="qwen-7b-chat")
    parser.add_argument("--training_data_path", default="./data/train/*.json")
    parser.add_argument("--output_dir", default="./models")
    parser.add_argument("--wandb_username", default="")
    args = parser.parse_args()

    llm_model_name = args.llm_model_name
    chat_template = args.chat_template
    training_data_path = args.training_data_path
    output_dir = args.output_dir
    wandb_username = args.wandb_username

    os.makedirs(output_dir, exist_ok=True)

    # for data_path in sorted(glob(training_data_path)):
    paths = sorted(glob(training_data_path))
    for data_path in paths:
        print(data_path)
        
        yaml_filename = make_axolotl_yaml(
            Path(data_path).stem, 
            llm_model_name, 
            data_path, 
            output_dir, 
            wandb_username, 
            chat_template=chat_template)

        hf_home = os.environ.get("HF_HOME", None)
        env_vars = f"HF_HOME={hf_home}" if hf_home is not None else ""
        os.system(f"{env_vars} accelerate launch -m axolotl.cli.train {yaml_filename}")
        os.system(f"rm -r {output_dir}/model_weights/*/checkpoint-*")