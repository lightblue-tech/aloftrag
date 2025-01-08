from pathlib import Path
import os
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from tqdm.auto import tqdm

language_code_map = {'aa': 'Afar', 'ab': 'Abkhazian', 'ae': 'Avestan', 'af': 'Afrikaans', 'ak': 'Akan', 'am': 'Amharic', 'an': 'Aragonese', 'ar': 'Arabic', 'as': 'Assamese', 'av': 'Avaric', 'ay': 'Aymara', 'az': 'Azerbaijani', 'ba': 'Bashkir', 'be': 'Belarusian', 'bg': 'Bulgarian', 'bh': 'Bihari language', 'bi': 'Bislama', 'bm': 'Bambara', 'bn': 'Bengali', 'bo': 'Tibetan', 'br': 'Breton', 'bs': 'Bosnian', 'ca': 'Catalan', 'ce': 'Chechen', 'ch': 'Chamorro', 'co': 'Corsican', 'cr': 'Cree', 'cs': 'Czech', 'cu': 'Old Church Slavonic', 'cv': 'Chuvash', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'dv': 'Dhivehi', 'dz': 'Dzongkha', 'ee': 'Ewe', 'el': 'Greek', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'fa': 'Persian', 'ff': 'Fulah', 'fi': 'Finnish', 'fj': 'Fijian', 'fo': 'Faroese', 'fr': 'French', 'fy': 'Western Frisian', 'ga': 'Irish', 'gd': 'Scottish Gaelic', 'gl': 'Galician', 'gn': 'Guarani', 'gu': 'Gujarati', 'gv': 'Manx', 'ha': 'Hausa', 'he': 'Hebrew', 'hi': 'Hindi', 'ho': 'Hiri Motu', 'hr': 'Croatian', 'ht': 'Haitian', 'hu': 'Hungarian', 'hy': 'Armenian', 'hz': 'Herero', 'ia': 'Interlingua (International Auxiliary Language Association)', 'id': 'Indonesian', 'ie': 'Interlingue', 'ig': 'Igbo', 'ii': 'Sichuan Yi', 'ik': 'Inupiaq', 'io': 'Ido', 'is': 'Icelandic', 'it': 'Italian', 'iu': 'Inuktitut', 'ja': 'Japanese', 'jv': 'Javanese', 'ka': 'Georgian', 'kg': 'Kongo', 'ki': 'Kikuyu', 'kj': 'Kwanyama', 'kk': 'Kazakh', 'kl': 'Kalaallisut', 'km': 'Central Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'kr': 'Kanuri', 'ks': 'Kashmiri', 'ku': 'Kurdish', 'kv': 'Komi', 'kw': 'Cornish', 'ky': 'Kyrgyz', 'la': 'Latin', 'lb': 'Luxembourgish', 'lg': 'Ganda', 'li': 'Limburgan', 'ln': 'Lingala', 'lo': 'Lao', 'lt': 'Lithuanian', 'lu': 'Luba-Katanga', 'lv': 'Latvian', 'mg': 'Malagasy', 'mh': 'Marshallese', 'mi': 'Maori', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian', 'mr': 'Marathi', 'ms': 'Malay', 'mt': 'Maltese', 'my': 'Burmese', 'na': 'Nauru', 'nb': 'Norwegian Bokmål', 'nd': 'Ndebele, North', 'ne': 'Nepali', 'ng': 'Ndonga', 'nl': 'Dutch', 'nn': 'Norwegian Nynorsk', 'no': 'Norwegian', 'nr': 'Ndebele, South', 'nv': 'Navajo', 'oc': 'Occitan', 'oj': 'Ojibwa', 'om': 'Oromo', 'or': 'Oriya', 'os': 'Ossetian', 'pa': 'Panjabi', 'pi': 'Pali', 'pl': 'Polish', 'ps': 'Pushto', 'pt': 'Portuguese', 'qu': 'Quechua', 'rm': 'Romansh', 'rn': 'Rundi', 'ro': 'Romanian', 'ru': 'Russian', 'rw': 'Kinyarwanda', 'sa': 'Sanskrit', 'sc': 'Sardinian', 'sd': 'Sindhi', 'se': 'Northern Sami', 'sg': 'Sango', 'si': 'Sinhala', 'sk': 'Slovak', 'sl': 'Slovenian', 'sm': 'Samoan', 'sn': 'Shona', 'so': 'Somali', 'sq': 'Albanian', 'sr': 'Serbian', 'ss': 'Swati', 'st': 'Sotho, Southern', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu', 'tg': 'Tajik', 'th': 'Thai', 'ti': 'Tigrinya', 'tk': 'Turkmen', 'tl': 'Tagalog', 'tn': 'Tswana', 'to': 'Tongan', 'tr': 'Turkish', 'ts': 'Tsonga', 'tt': 'Tatar', 'tw': 'Twi', 'ty': 'Tahitian', 'ug': 'Uyghur', 'uk': 'Ukrainian', 'ur': 'Urdu', 'uz': 'Uzbek', 've': 'Venda', 'vi': 'Vietnamese', 'vo': 'Volapük', 'wa': 'Walloon', 'wo': 'Wolof', 'xh': 'Xhosa', 'yi': 'Yiddish', 'yo': 'Yoruba', 'za': 'Zhuang', 'zh': 'Chinese', 'zu': 'Zulu'}

def prompt_llm(llm, prompts, system_message, max_tokens, generation_prompt="", adapter_path=""):
    # Parse input prompts to conversations. If the input prompt is None, we make the conversation None
    sys_msg_chat = [{"role": "system", "content": system_message}]
    chats = [sys_msg_chat + [{"role": "user", "content": p}] if isinstance(p, str) else None for p in prompts]

    # Filter out None conversations
    str_chats = [x for x in chats if x is not None]

    # Parse conversations to string with correct chat template tokens
    in_texts = [llm.llm_engine.tokenizer.tokenizer.apply_chat_template(
        c,
        tokenize=False,
        add_generation_prompt=True
    ) + generation_prompt for c in tqdm(str_chats)]

    # Generate text from LLM
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    if len(adapter_path) > 0:
        outputs = llm.generate(
            in_texts,
            sampling_params,
            lora_request=LoRARequest("lora_adapter", 1, adapter_path)
        )
    else:
        outputs = llm.generate(
            in_texts, 
            sampling_params
        )

    # Make any outputs which do not finish None (they should all have reasonably finished within the token limit so have probably bugged out).
    finished_outputs = [o.outputs[0].text if o.outputs[0].finish_reason == "stop" else None for o in outputs]

    # Make the outputs to correspond with the original inputted prompts, where if any inputs are None, then the outputs are also None.  
    output_gen = iter(finished_outputs)
    return [next(output_gen) if c is not None else None for c in chats]

def make_input_text(context_list, question):
    context_texts = "\n\n".join([f"### {i+1}\n" + x for i, x in enumerate(context_list)])

    input_text = f"""{context_texts}

### Question
{question}"""
    return input_text

def get_language_from_path(path):
    filepath = Path(path)
    filename_stem = filepath.stem
    language = filename_stem.split(".")[-1]
    language = language_code_map[language] if language in language_code_map.keys() else language
    language = language.capitalize()
    return language

def make_axolotl_yaml(expt_name, llm_model_name, data_dir, output_dir, wandb_username, chat_template="qwen-7b-chat"):
    wandb_bool = "true" if len(wandb_username) > 0 else "false"
    wandb_project = "axolotl" if len(wandb_username) > 0 else ""
    wandb_name = f"self_rag_train_axolotl_{expt_name}" if len(wandb_username) > 0 else ""
    yaml_data = f"""base_model: {llm_model_name}
trust_remote_code: true

load_in_8bit: false
load_in_4bit: false
strict: false

datasets:
  - path: {data_dir}
    ds_type: json # see other options below
    type: sharegpt
    conversation: {chat_template}
dataset_prepared_path:
val_set_size: 0.1
output_dir: {output_dir}/model_weights/{expt_name}

sequence_len: 20000
sample_packing: false
eval_sample_packing: false
pad_to_sequence_len: true

use_wandb: {wandb_bool}
wandb_project: {wandb_project}
wandb_entity: {wandb_username}
wandb_name: {wandb_name}

adapter: lora
lora_model_dir:
lora_r: 64
lora_alpha: 32
lora_dropout: 0.05
lora_target_linear: true
lora_fan_in_fan_out:

gradient_accumulation_steps: 1
micro_batch_size: 1
num_epochs: 1
optimizer: adamw_torch
lr_scheduler: cosine
learning_rate: 0.0002

train_on_inputs: false
group_by_length: false
bf16: auto
fp16:
tf32: false

gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true

warmup_steps: 0
evals_per_epoch: 10
saves_per_epoch: 0
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:"""
    save_dir = "./train_configs"
    os.makedirs(save_dir, exist_ok=True)
    filepath = f"{save_dir}/{expt_name}.yaml"
    with open(filepath, "w") as f:
        f.write(yaml_data)
    return filepath