"""
Client test.

Run server:

python generate.py  --base_model=h2oai/h2ogpt-oig-oasst1-512-6.9b

NOTE: For private models, add --use-auth_token=True

NOTE: --infer_devices=True (default) must be used for multi-GPU in case see failures with cuda:x cuda:y mismatches.
Currently, this will force model to be on a single GPU.

Then run this client as:

python client_test.py
"""

debug = False

import os
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
from gradio_client import Client

client = Client("http://localhost:7860")
if debug:
    print(client.view_api(all_endpoints=True))

instruction = ''  # only for chat=True
iinput = ''  # only for chat=True
context = ''
# streaming output is supported, loops over and outputs each generation in streaming mode
# but leave stream_output=False for simple input/output mode
stream_output = False
prompt_type = 'human_bot'
temperature = 0.1
top_p = 0.75
top_k = 40
num_beams = 1
max_new_tokens = 50
min_new_tokens = 0
early_stopping = False
max_time = 20
repetition_penalty = 1.0
num_return_sequences = 1
do_sample = True
# only these 2 below used if pass chat=False
chat = False
instruction_nochat = "Who are you?"
iinput_nochat = ''


def test_client_basic():
    args = [instruction,
            iinput,
            context,
            stream_output,
            prompt_type,
            temperature,
            top_p,
            top_k,
            num_beams,
            max_new_tokens,
            min_new_tokens,
            early_stopping,
            max_time,
            repetition_penalty,
            num_return_sequences,
            do_sample,
            chat,
            instruction_nochat,
            iinput_nochat,
            ]
    api_name = '/submit_nochat'
    res = client.predict(
        *tuple(args),
        api_name=api_name,
    )
    res_dict = dict(instruction_nochat=instruction_nochat, iinput_nochat=iinput_nochat, response=md_to_text(res))
    print(res_dict)


import markdown  # pip install markdown
from bs4 import BeautifulSoup  # pip install beautifulsoup4


def md_to_text(md):
    html = markdown.markdown(md)
    soup = BeautifulSoup(html, features='html.parser')
    return soup.get_text()


if __name__ == '__main__':
    test_client_basic()
