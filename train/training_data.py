# from FlagEmbedding import BGEM3FlagModel
import numpy as np
from tqdm.auto import tqdm
import random

from utils.util import make_input_text
from system_message.system_messages import train_system_message_fn


def select_neg_context(embed_model, dataset, num_docs=10):
    # Select the n chunks that are most similar to the question without being the same as the context
    context_emb = embed_model.encode(dataset["context"])['dense_vecs']
    question_emb = embed_model.encode(dataset["question"])['dense_vecs']

    contexts_arr = np.unique(dataset["context"])

    neg_contexts = []

    for emb, context in tqdm(zip(question_emb, dataset["context"])):
        # Make mask to select all contexts that are not the current context
        non_match_mask = contexts_arr != context

        sim = context_emb[non_match_mask] @ emb.reshape([-1, 1])

        sorted_args = sim[:, 0].argsort()

        top_k_args = sorted_args[-num_docs:]

        neg_contexts.append(
            contexts_arr[non_match_mask][top_k_args]
        )

    return neg_contexts

def make_training_data(row, num_docs, language):
    negative_contexts = row["neg_contexts"]
    positive_context = row["context"]
    question = row["question"]
    answer = row["answer"]

    correct_idx = random.randrange(0, num_docs)
    contexts = random.sample(negative_contexts, len(negative_contexts))
    contexts[correct_idx] = positive_context

    input_text = make_input_text(contexts, question)    
    output_text = f"""### Reference
{correct_idx+1}

### Answer
{answer}"""

    return {"conversations": [
        {"from": "system", "value": train_system_message_fn(language)},
        {"from": "human", "value": input_text},
        {"from": "gpt", "value": output_text}
    ]}

def format_rag_training_data(embedding_model, dataset, num_docs, language, min_text_score, min_q_score, min_a_score):
    dataset = dataset.filter(lambda x: all([
        isinstance(x['filter_score'], float),
        isinstance(x['q_rating'], float),
        isinstance(x['a_rating'], float)
    ]), num_proc=16)
    dataset = dataset.filter(lambda x: all([
        x['filter_score'] >= min_text_score,
        x['q_rating'] >= min_q_score,
        x['a_rating'] >= min_a_score
    ]), num_proc=16)
    if len(dataset) <= num_docs:
        return None
    neg_contexts = select_neg_context(embedding_model, dataset, num_docs=num_docs)
    dataset = dataset.add_column("neg_contexts", neg_contexts)
    dataset = dataset.map(lambda x: make_training_data(x, num_docs, language), num_proc=16)
    dataset = dataset.select_columns(["conversations"])
    return dataset