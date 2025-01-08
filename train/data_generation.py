from tqdm.auto import tqdm
import re
from datasets import Dataset

from system_message.system_messages import filter_system_message_fn, gen_system_message_fn, question_rating_system_message_fn, answer_rating_system_message_fn
from utils.util import prompt_llm

def parse_float(text):
    try:
        return float(text.split("\n")[0].strip("[] "))
    except:
        return None

def filter_texts(llm, dataset):
    filter_scores = prompt_llm(
        llm,
        dataset["context"],
        system_message=filter_system_message_fn(),
        max_tokens=20,
        generation_prompt="### Filter score\n",
    )

    float_scores = [
        parse_float(x) for x in tqdm(filter_scores)
    ]

    dataset = dataset.add_column("filter_score", float_scores)

    return dataset

def extract_question_answer(text):
    if text is None:
        return {
            "question": None, "answer": None
        }
    
    pattern = r"([\s\S]+?)### Answer\n([\s\S]+)"
    match_ = re.search(pattern, text)

    if match_:
        question = match_.group(1).strip()
        answer = match_.group(2).strip()
        return {
            "question": question, "answer": answer
        }
    else:
        return {
            "question": None, "answer": None
        }

def generate_qas(llm, dataset, language):
    gen_qas = prompt_llm(
        llm,
        dataset["context"],
        system_message=gen_system_message_fn(language),
        max_tokens=512,
        generation_prompt="### Question\n",
    )

    dataset = dataset.add_column("gen_qa", gen_qas)
    dataset = dataset.map(lambda x: extract_question_answer(x["gen_qa"]), num_proc=16)
    return dataset

def make_context_question_str(text, question):
    if question is None:
        return None
    else:
        return "### Context\n" + text + "\n\n### Question\n" + question

def rate_questions(llm, dataset, language):

    rating_q_context_list = dataset.map(
        lambda x: {
            "rating_q_temp": make_context_question_str(x["context"], x["question"])
        },
        num_proc=16
    )["rating_q_temp"]

    q_rating = prompt_llm(
        llm,
        rating_q_context_list,
        system_message=question_rating_system_message_fn(language),
        max_tokens=16,
        generation_prompt="### Question rating score\n",
    )

    float_scores = [
        parse_float(x) for x in tqdm(q_rating)
    ]

    dataset = dataset.add_column("q_rating", float_scores)

    return dataset

def make_context_question_answer_str(text, question, answer):
    if question is None or answer is None:
        return None
    else:
        return "### Context\n" + text + "\n\n### Question\n" + question + "\n\n### Answer\n" + answer

def rate_answer(llm, dataset, language):

    rating_a_context_list = dataset.map(
        lambda x: {
            "rating_a_temp": make_context_question_answer_str(x["context"], x["question"], x["answer"])
        },
        num_proc=16
    )["rating_a_temp"]

    a_rating = prompt_llm(
        llm,
        rating_a_context_list,
        system_message=answer_rating_system_message_fn(language),
        max_tokens=16,
        generation_prompt="### Answer rating score\n",
    )

    float_scores = [
        parse_float(x) for x in tqdm(a_rating)
    ]

    dataset = dataset.add_column("a_rating", float_scores)

    return dataset

def generate_rag_ft_data(llm, dataset, language, rate_qa=True):
    dataset = Dataset.from_pandas(dataset.to_pandas().groupby("context").first()).select_columns(["context"])

    dataset = filter_texts(llm, dataset)
    dataset = generate_qas(llm, dataset, language)
    if rate_qa:
        dataset = rate_questions(llm, dataset, language)
        dataset = rate_answer(llm, dataset, language)
    return dataset