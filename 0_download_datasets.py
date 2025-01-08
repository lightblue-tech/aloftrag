from tqdm.auto import tqdm
from datasets import load_dataset, concatenate_datasets, Dataset
from cryptography.fernet import Fernet
import os
import pandas as pd
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
from io import StringIO
from transformers import AutoTokenizer
tqdm.pandas()

def parse_calmqa(row):
    return {
        "context": row["answer"].strip(),
        "question": row["question"].strip(),
        "answer": row["answer"].strip()
    }

def parse_squad(row):
    return {
        "context": row["context"].strip(),
        "question": row["question"].strip(),
        "answer": row["answers"]["text"][0].strip()
    }

def prepare_calmqa():
    # Need to prove that the generated questions in English and Japanese are not the same as the original question.
    dataset_dict = {}

    lang_map = {0: 'Afar', 1: 'Arabic',
                2: 'Balochi', 3: 'Chinese',
                4: 'English', 5: 'Faroese',
                6: 'Fijian', 7: 'German',
                8: 'Hebrew', 9: 'Hiligaynon',
                10: 'Hindi', 11: 'Hungarian',
                12: 'Japanese', 13: 'Kirundi',
                14: 'Korean', 15: 'Papiamento',
                16: 'Pashto', 17: 'Russian',
                18: 'Samoan', 19: 'Spanish',
                20: 'Tongan', 21: 'Tswana',
                22: 'Wolof'}

    for lang_num, lang_name in tqdm(lang_map.items()):
        ds = load_dataset(
            "shanearora/CaLMQA",
            split="train",
            trust_remote_code=True).filter(
                lambda x: x["language"] == lang_num and x["answer"] is not None,
                num_proc=16
            ).map(
                lambda x: {"language": lang_map[x["language"]]},
                num_proc=16
            )

        if len(ds) < 1:
            continue

        ds = ds.map(parse_calmqa)
        dataset_dict["calmqa." + lang_name] = ds

    return dataset_dict

def prepare_m2qa():
    dataset_dict = {}

    langs = [
        "chinese",
        "german",
        "turkish",
    ]

    domains = [
        "creative_writing",
        "news",
        "product_reviews"
    ]

    for lang in tqdm(langs):
        ds = concatenate_datasets([
            load_dataset(
                "UKPLab/m2qa",
                f"m2qa.{lang}.{x}",
                split="validation",
                trust_remote_code=True
                ) for x in domains])

        ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0, num_proc=16)

        # Decrypt it
        fernet = Fernet(b"aRY0LZZb_rPnXWDSiSJn9krCYezQMOBbGII2eGkN5jo=")

        def decrypt(example):
            example["question"] = fernet.decrypt(example["question"].encode()).decode()
            example["context"] = fernet.decrypt(example["context"].encode()).decode()
            example["answers"]["text"] = [fernet.decrypt(answer.encode()).decode() for answer in example["answers"]["text"]]
            return example

        ds = ds.map(decrypt)
        ds = ds.map(parse_squad)

        dataset_dict["m2qa." + lang] = ds

    return dataset_dict

def prepare_mlqa():
    dataset_dict = {}

    langs = ['ar', 'de', 'en', 'es', 'hi', 'vi', 'zh']

    for lang in tqdm(langs):
        ds = load_dataset(
            "facebook/mlqa",
            f"mlqa.{lang}.{lang}",
            split="test",
            trust_remote_code=True
            )
        ds = ds.map(parse_squad)
        dataset_dict["mlqa." + lang] = ds

    return dataset_dict

def prepare_xquad():

    dataset_dict = {}

    langs = ['xquad.ar', 'xquad.de',
            'xquad.el', 'xquad.en',
            'xquad.es', 'xquad.hi',
            'xquad.ro', 'xquad.ru',
            'xquad.th', 'xquad.tr',
            'xquad.vi', 'xquad.zh']

    for lang in tqdm(langs):
        ds = load_dataset(
            "google/xquad",
            lang,
            split="validation",
            trust_remote_code=True
        )
        ds = ds.map(parse_squad)
        dataset_dict[lang] = ds

    return dataset_dict

def prepare_tydiqa_goldp():

    dataset_dict = {}

    langs = ['arabic', 'bengali',
             'english', 'finnish',
             'indonesian', 'korean',
             'russian', 'swahili',
             'telugu']

    for lang in tqdm(langs):
        ds = load_dataset(
            "google-research-datasets/tydiqa",
            "secondary_task",
            split="validation",
            trust_remote_code=True).filter(
                lambda x: x["id"].split("-")[0] == lang,
                num_proc=16
            )
        ds = ds.map(lambda x: {"context": x["title"] + "\n" + x["context"]}, num_proc=16)
        ds = ds.map(parse_squad)
        dataset_dict["tydigoldp." + lang] = ds

    return dataset_dict

def prepare_skquad():
    ds = load_dataset("TUKE-DeutscheTelekom/skquad", split="validation")
    ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0 and len(x["answers"]["text"][0].strip()) > 0, num_proc=16)
    ds = ds.map(lambda x: {"context": x["title"] + "\n" + x["context"]}, num_proc=16)
    ds = ds.map(parse_squad, num_proc=16).select_columns(["context", "question", "answer"])
    dataset_dict = {"skquad.sk": ds}
    return dataset_dict

def prepare_arcd():
    ds = load_dataset("hsseinmz/arcd", split="validation")
    ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0 and len(x["answers"]["text"][0].strip()) > 0, num_proc=16)
    ds = ds.map(lambda x: {"context": x["title"] + "\n" + x["context"]}, num_proc=16)
    ds = ds.map(parse_squad, num_proc=16).select_columns(["context", "question", "answer"])
    dataset_dict = {"arcd.ar": ds}
    return dataset_dict

def prepare_persianqa():
    ds = load_dataset("SajjadAyoubi/persian_qa", split="validation")
    ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0 and len(x["answers"]["text"][0].strip()) > 0, num_proc=16)
    ds = ds.map(lambda x: {"context": x["title"] + "\n" + x["context"]}, num_proc=16)
    ds = ds.map(parse_squad, num_proc=16).select_columns(["context", "question", "answer"])
    dataset_dict = {"persianqa.fa": ds}
    return dataset_dict

def prepare_amharicqa():
    df = pd.read_json("https://raw.githubusercontent.com/semantic-systems/amharic-qa/main/test_data.json")
    df = pd.DataFrame(pd.DataFrame(df.data.tolist()).explode("paragraphs").paragraphs.tolist()).explode("qas")
    df["question"] = df.qas.apply(lambda x: x["question"])
    df["answer"] = df.qas.apply(lambda x: x["answers"][0]["text"])
    ds = Dataset.from_pandas(df).select_columns(["context", "question", "answer"])
    dataset_dict = {"amharicqa.am": ds}
    return dataset_dict

def prepare_chaii():
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files('chaii-hindi-and-tamil-question-answering', path='.')

    zip_path = './chaii-hindi-and-tamil-question-answering.zip'

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:    
        with zip_ref.open("train.csv") as file:
            content = file.read().decode('utf-8')
            df = pd.read_csv(StringIO(content))

    df["answer"] = df["answer_text"]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    max_tok_size = 1_500
    df["lens"] = df.context.progress_apply(lambda x: len(tokenizer.encode(x)))
    df = df[df.lens < max_tok_size]

    ta_ds = Dataset.from_pandas(df[df["language"] == "tamil"]).select_columns(["context", "question", "answer"])
    hi_ds = Dataset.from_pandas(df[df["language"] == "hindi"]).select_columns(["context", "question", "answer"])

    dataset_dict = {"chaii.ta": ta_ds, "chaii.hi": hi_ds}
    return dataset_dict

def prepare_sberquad():
    ds = load_dataset("kuznetsoffandrey/sberquad", split="validation", trust_remote_code=True)
    ds = ds.filter(lambda x: bool(len(x["answers"]["text"]) > 0) and bool(len(x["answers"]["text"][0]) > 0), num_proc=16)
    ds = ds.map(lambda x: {"context": x["context"]}, num_proc=16)
    ds = ds.map(parse_squad, num_proc=16).select_columns(["context", "question", "answer"])

    dataset_dict = {"sberquad.ru": ds}
    return dataset_dict

def prepare_pira():
    ds = load_dataset("paulopirozelli/pira", "default", split="test")

    en_ds = ds.map(lambda x: {
            "context": x["abstract"].strip(),
            "question": x["question_en_origin"].strip(),
            "answer": x["answer_en_origin"].strip(),
        }, num_proc=16).select_columns(["context", "question", "answer"])
    pt_ds = ds.map(lambda x: {
            "context": x["abstract_translated_pt"].strip(),
            "question": x["question_pt_origin"].strip(),
            "answer": x["answer_pt_origin"].strip(),
        }, num_proc=16).select_columns(["context", "question", "answer"])
    
    dataset_dict = {"pira.en": en_ds, "pira.pt": pt_ds}
    return dataset_dict

def prepare_jglue():
    ds = load_dataset("shunk031/JGLUE", "JSQuAD", split="validation", trust_remote_code=True)
    ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0 and len(x["answers"]["text"][0].strip()) > 0, num_proc=16)
    ds = ds.map(lambda x: {"context": x["context"].replace("[SEP]", "\n")}, num_proc=16)
    ds = ds.map(parse_squad, num_proc=16).select_columns(["context", "question", "answer"])
    
    dataset_dict = {"jglue.ja": ds}
    return dataset_dict

def prepare_korquad():
    ds = load_dataset("KorQuAD/squad_kor_v1", split="validation")
    ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0 and len(x["answers"]["text"][0].strip()) > 0, num_proc=16)
    ds = ds.map(lambda x: {"context": x["title"] + "\n" + x["context"]}, num_proc=16)
    ds = ds.map(parse_squad, num_proc=16).select_columns(["context", "question", "answer"])
    
    dataset_dict = {"korquad.ko": ds}
    return dataset_dict

def prepare_tquad():
    df = pd.read_json("https://raw.githubusercontent.com/TQuad/turkish-nlp-qa-dataset/master/dev-v0.1.json")
    df = pd.DataFrame(df.data.apply(lambda x: x["paragraphs"]).explode())
    df["context"] = df.data.apply(lambda x: x["context"])
    df["data"] = df.data.apply(lambda x: x["qas"])
    df = df.explode("data")
    df["question"] = df["data"].apply(lambda x: x["question"] if isinstance(x, dict) else None)
    df["answer"] = df["data"].apply(lambda x: x["answers"][0]["text"] if isinstance(x, dict) else None)
    df = df.dropna()
    ds = Dataset.from_pandas(df).select_columns(["context", "question", "answer"])
    
    dataset_dict = {"tquad.tr": ds}
    return dataset_dict

def prepare_sqac():
    df = pd.read_json("https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC/resolve/main/test.json")
    df = pd.DataFrame(df.data.apply(lambda x: x["paragraphs"]).explode())
    df["context"] = df.data.apply(lambda x: x["context"])
    df["data"] = df.data.apply(lambda x: x["qas"])
    df = df.explode("data")
    df["question"] = df["data"].apply(lambda x: x["question"] if isinstance(x, dict) else None)
    df["answer"] = df["data"].apply(lambda x: x["answers"][0]["text"] if isinstance(x, dict) else None)
    df = df.dropna()
    ds = Dataset.from_pandas(df).select_columns(["context", "question", "answer"])
    
    dataset_dict = {"sqac.es": ds}
    return dataset_dict

def prepare_germanquad():
    ds = load_dataset("deepset/germanquad", split="test", trust_remote_code=True)
    ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0 and len(x["answers"]["text"][0].strip()) > 0, num_proc=16)
    ds = ds.map(parse_squad, num_proc=16).select_columns(["context", "question", "answer"])
    
    dataset_dict = {"germanquad.de": ds}
    return dataset_dict

def prepare_kenswquad():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
    max_tok_size = 1_500
    
    ds = load_dataset("lightblue/KenSwQuAD", split="train")
    ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0 and len(x["answers"]["text"][0].strip()) > 0, num_proc=16)
    ds = ds.map(parse_squad, num_proc=16).select_columns(["context", "question", "answer"])
    ds = ds.filter(lambda x: len(tokenizer.encode(x["context"])) < max_tok_size, num_proc=16)
    
    dataset_dict = {"kenswquad.sw": ds}
    return dataset_dict

def prepare_drcd():
    ds = load_dataset("voidful/DRCD", split="test")
    df = pd.DataFrame(ds.to_pandas().explode("paragraphs").paragraphs.tolist()).explode("qas")
    df["question"] = df.qas.apply(lambda x: x["question"])
    df["answer"] = df.qas.apply(lambda x: x["answers"][0]["text"])
    ds = Dataset.from_pandas(df).select_columns(["context", "question", "answer"])
    
    dataset_dict = {"drcd.zh": ds}
    return dataset_dict

def prepare_publichealthqa():
    dataset_dict = {}
    langs = ['arabic', 'chinese', 'english', 'french', 'korean', 'russian', 'spanish', 'vietnamese']

    for lang in langs:
        ds = load_dataset("xhluca/publichealth-qa", lang, split="test")
        ds = ds.filter(lambda x: isinstance(x["question"], str) and len(x["question"].strip()) > 0, num_proc=16)
        ds = ds.filter(lambda x: isinstance(x["answer"], str) and len(x["answer"].strip()) > 0, num_proc=16)
        
        ds = ds.map(lambda x: {
            "context": x["answer"].strip(),
            "question": x["question"].strip(),
            "answer": x["answer"].strip(),
        }, num_proc=16).select_columns(["context", "question", "answer"])
        
        dataset_dict[f"publichealthqa.{lang}"] = ds
    return dataset_dict

def prepare_narrativeqa():

    ds = load_dataset("deepmind/narrativeqa", split="test").map(
        lambda x: {
            "context": x["document"]["summary"]["text"].strip(),
            "question": x["question"]["text"].strip(),
            "answer": x["answers"][0]["text"],
            }, num_proc=16
        ).select_columns(["context", "question", "answer"])
    
    dataset_dict = {"narrativeqa.en": ds}
    return dataset_dict

if __name__ == "__main__":
    dataset_dict = dict(
        **prepare_calmqa(),
        **prepare_m2qa(),
        **prepare_mlqa(),
        **prepare_xquad(),
        **prepare_tydiqa_goldp(),
        **prepare_skquad(),
        **prepare_arcd(),
        **prepare_persianqa(),
        # **prepare_amharicqa(),
        **prepare_chaii(),
        **prepare_narrativeqa(),
        **prepare_sberquad(),
        **prepare_pira(),
        **prepare_jglue(),
        **prepare_korquad(),
        **prepare_tquad(),
        **prepare_sqac(),
        **prepare_germanquad(),
        **prepare_kenswquad(),
        **prepare_drcd(),
        **prepare_publichealthqa(),
    )

    dataset_dict = {k: v.select_columns(
            ["question", "answer", "context"]
        ) for k, v in dataset_dict.items()}
    
    for k, v in dataset_dict.items():
        raw_data_dir = "./data/raw"
        os.makedirs(raw_data_dir, exist_ok=True)
        v.to_parquet(f"{raw_data_dir}/{k}.parquet")