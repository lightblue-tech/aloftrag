sys_message = """You are a answer checking AI.
Given a context passage, a question, a correct reference answer, and a generated answer as inputs, determine whether the generated answer is correct based on the context given.

If the answer is not correct, output only FALSE.
If the answer is correct, output only TRUE."""

make_input = lambda c, q, ra, ca: f"""# Context
{c}

# Question
{q}

# Reference answer
{ra}

# Generated answer
{ca}"""

row_parse = lambda x, model: make_input(x["context"], x["question"], x["answer"], x[f"{model}_gen_ans"].split("### Answer")[1].strip())

def call_gpt(client, row, model):

    response = client.chat.completions.create(
        model="gpt4o-2024-05-13",
        messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": sys_message
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": row_parse(row, model)
            }
        ]
        }
        ],
        temperature=0,
        max_tokens=12,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=0
    )

    return response.choices[0].message.content

def run_answer_analysis(client, row, model):
    try:
        return call_gpt(client, row, model)
    except:
        return None