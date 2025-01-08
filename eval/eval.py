from tqdm.auto import tqdm
import numpy as np
import random
random.seed(123)

from utils.util import prompt_llm
from utils.util import make_input_text

from system_message.system_messages import train_system_message_fn

def add_contexts(dataset, embed_model, num_docs, tokenizer, max_model_len):

    context_arr = np.unique(dataset["context"])
    context_emb = embed_model.encode(context_arr.tolist(), batch_size=2)["dense_vecs"]
    question_emb = embed_model.encode(dataset["question"], batch_size=2)["dense_vecs"]

    selected_contexts = []
    is_easy_question_list = []

    max_model_len_w_output_margin = max_model_len - 1000 # Keep 1000 tokens margin after the context to hold chat template tokens, question tokens, and answer tokens.

    for idx, emb in enumerate(tqdm(question_emb)):
        sim = context_emb @ emb.reshape([-1, 1])
        sorted_args = sim[:, 0].argsort()
        top_k_args = sorted_args[-num_docs:]


        context_list = []
        for arg in top_k_args[::-1]:
            context_list.append(context_arr[arg])
            if sum([len(tokenizer.tokenize(y)) for y in context_list]) > max_model_len_w_output_margin:
                break
        
        correct_context = dataset[idx]["context"]
        is_easy_question = correct_context in context_list

        if not is_easy_question:
            context_list[-1] = correct_context

        context_list = random.sample(context_list, len(context_list))
        selected_contexts.append(context_list)
        is_easy_question_list.append(is_easy_question)
        
    dataset = dataset.add_column("is_hard_question", is_easy_question_list)
    dataset = dataset.add_column("selected_contexts", selected_contexts)
    return dataset

def make_training_data(row, language):
    selected_contexts = row["selected_contexts"]
    question = row["question"]

    input_text = make_input_text(selected_contexts, question)    

    return {"conversations": [
        {"from": "system", "value": train_system_message_fn(language)},
        {"from": "human", "value": input_text},
    ]}


def add_gen_answers(dataset, llm, language, adapter_path, do_run_base=False, max_tokens = 2048):

    input_text = dataset.map(lambda x: {
        "input_text": make_input_text(x["selected_contexts"], x["question"])
    }, num_proc=16)["input_text"]

    sys_message = train_system_message_fn(language)

    lora_gen_ans = prompt_llm(
        llm, input_text, 
        sys_message, 
        max_tokens=max_tokens, 
        generation_prompt="### Reference\n", 
        adapter_path=adapter_path)
    
    dataset = dataset.add_column("lora_gen_ans", lora_gen_ans)
    
    if do_run_base:
        base_gen_ans = prompt_llm(
            llm, input_text, 
            sys_message, 
            max_tokens=max_tokens,
            generation_prompt="### Reference\n")
        
        dataset = dataset.add_column("base_gen_ans", base_gen_ans)

    return dataset