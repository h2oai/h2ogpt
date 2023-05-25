"""
Client test.

Run server:

python generate.py  --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b

NOTE: For private models, add --use-auth_token=True

NOTE: --infer_devices=True (default) must be used for multi-GPU in case see failures with cuda:x cuda:y mismatches.
Currently, this will force model to be on a single GPU.

Then run this client as:

python client_test.py



For HF spaces:

HOST="https://h2oai-h2ogpt-chatbot.hf.space" python client_test.py

Result:

Loaded as API: https://h2oai-h2ogpt-chatbot.hf.space ✔
{'instruction_nochat': 'Who are you?', 'iinput_nochat': '', 'response': 'I am h2oGPT, a large language model developed by LAION.'}


For demo:

HOST="https://gpt.h2o.ai" python client_test.py

Result:

Loaded as API: https://gpt.h2o.ai ✔
{'instruction_nochat': 'Who are you?', 'iinput_nochat': '', 'response': 'I am h2oGPT, a chatbot created by LAION.'}

"""
import time
import os
import markdown  # pip install markdown
import pytest
from bs4 import BeautifulSoup  # pip install beautifulsoup4

debug = False

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'


def get_client(serialize=True):
    from gradio_client import Client

    client = Client(os.getenv('HOST', "http://localhost:7860"), serialize=serialize)
    if debug:
        print(client.view_api(all_endpoints=True))
    return client


def get_args(prompt, prompt_type, chat=False, stream_output=False, max_new_tokens=50):
    from collections import OrderedDict
    kwargs = OrderedDict(instruction=prompt if chat else '',  # only for chat=True
                         iinput='',  # only for chat=True
                         context='',
                         # streaming output is supported, loops over and outputs each generation in streaming mode
                         # but leave stream_output=False for simple input/output mode
                         stream_output=stream_output,
                         prompt_type=prompt_type,
                         temperature=0.1,
                         top_p=0.75,
                         top_k=40,
                         num_beams=1,
                         max_new_tokens=max_new_tokens,
                         min_new_tokens=0,
                         early_stopping=False,
                         max_time=20,
                         repetition_penalty=1.0,
                         num_return_sequences=1,
                         do_sample=True,
                         chat=chat,
                         instruction_nochat=prompt if not chat else '',
                         iinput_nochat='',  # only for chat=False
                         langchain_mode='Disabled',
                         top_k_docs=4,
                         document_choice=['All'],
                         )
    if chat:
        # add chatbot output on end.  Assumes serialize=False
        kwargs.update(dict(chatbot=[['', None]]))

    return kwargs, list(kwargs.values())


@pytest.mark.skip(reason="For manual use against some server, no server launched")
def test_client_basic():
    return run_client_nochat(prompt='Who are you?', prompt_type='human_bot', max_new_tokens=50)


def run_client_nochat(prompt, prompt_type, max_new_tokens):
    kwargs, args = get_args(prompt, prompt_type, chat=False, max_new_tokens=max_new_tokens)

    api_name = '/submit_nochat'
    client = get_client(serialize=True)
    res = client.predict(
        *tuple(args),
        api_name=api_name,
    )
    res_dict = dict(prompt=kwargs['instruction_nochat'], iinput=kwargs['iinput_nochat'],
                    response=md_to_text(res))
    print(res_dict)
    return res_dict


@pytest.mark.skip(reason="For manual use against some server, no server launched")
def test_client_chat():
    return run_client_chat(prompt='Who are you?', prompt_type='human_bot', stream_output=False, max_new_tokens=50)


def run_client_chat(prompt, prompt_type, stream_output, max_new_tokens):
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output, max_new_tokens=max_new_tokens)

    client = get_client(serialize=False)

    res = client.predict(*tuple(args), api_name='/instruction')
    args[-1] += [res[-1]]

    res_dict = kwargs
    res_dict['prompt'] = prompt
    if not kwargs['stream_output']:
        res = client.predict(*tuple(args), api_name='/instruction_bot')
        res_dict['response'] = res[0][-1][1]
        print(md_to_text(res_dict['response']))
        return res_dict
    else:
        job = client.submit(*tuple(args), api_name='/instruction_bot')
        res1 = ''
        while not job.done():
            outputs_list = job.communicator.job.outputs
            if outputs_list:
                res = job.communicator.job.outputs[-1]
                res1 = res[0][-1][-1]
                res1 = md_to_text(res1)
                print(res1)
            time.sleep(0.1)
        print(job.outputs())
        res_dict['response'] = res1
        return res_dict


def md_to_text(md):
    assert md is not None, "Markdown is None"
    html = markdown.markdown(md)
    soup = BeautifulSoup(html, features='html.parser')
    return soup.get_text()


if __name__ == '__main__':
    test_client_basic()
