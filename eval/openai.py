# import the OpenAI Python library for calling the OpenAI API
import os
import openai

SYSTEM0 = """You are a helpful assistant that measures the accuracy of large language model responses to a prompt."""

INSTRUCTION = """Your task is to evaluate another assistant's response to a prompt.
Evaluation criteria include: response's helpfulness, relevance, accuracy, and level of detail.
An overall numeric score of 0 means the response did not answer the question, or did poorly in a criteria.
An overall numeric score of 0.5 means the response somewhat answered the question, or somewhat satisfied the criteria.
An overall numeric score of 1.0 means the response perfectly answered the question, and satisfied all criteria.
Let AAA be the overall score you evaluated the response,
BBB be the numeric score for helpfulness,
CCC be the numeric score for relevance,
DDD be the numeric score for accuracy,
EEE be the numeric score for level of detail.
Let DETAILS be an additional paragraph containing a comprehensive explanation of your evaluation.
Format your response as:
====
AAA, BBB, CCC, EEE
====
DETAILS
"""
RESPONSE = "Text to Score"
END_KEY = "### End"
INTRO = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)
PROMPT0 = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO,
    instruction_key=INSTRUCTION,
    instruction="{prompt}",
    response_key=RESPONSE,
)

def get_gpt_score(prompt, system=SYSTEM0, api_key=None, model="gpt-3.5-turbo",
                  max_tokens_prompt=384):
    api_key = api_key or os.environ.get("OPENAI_KEY")
    openai.api_key = api_key

    # Example OpenAI Python library request
    # can be chain of role's that will continue with
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system",
             "content": system},
            {"role": "user",
             "content": prompt[:max_tokens_prompt*4]},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    # get
    response1 = response['choices'][0]['message']['content']
    response_split = response1.split("\n")
    score = response_split[0]
    score = score.lower().replace("score:", "").strip()
    score = float(score)
    return score
