"""
Client test.

Run server:

python generate.py  --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b

NOTE: For private models, add --use-auth_token=True

NOTE: --use_gpu_id=True (default) must be used for multi-GPU in case see failures with cuda:x cuda:y mismatches.
Currently, this will force model to be on a single GPU.

Then run this client as:

python src/client_test.py



For HF spaces:

HOST="https://h2oai-h2ogpt-chatbot.hf.space" python src/client_test.py

Result:

Loaded as API: https://h2oai-h2ogpt-chatbot.hf.space ✔
{'instruction_nochat': 'Who are you?', 'iinput_nochat': '', 'response': 'I am h2oGPT, a large language model developed by LAION.', 'sources': ''}


For demo:

HOST="https://gpt.h2o.ai" python src/client_test.py

Result:

Loaded as API: https://gpt.h2o.ai ✔
{'instruction_nochat': 'Who are you?', 'iinput_nochat': '', 'response': 'I am h2oGPT, a chatbot created by LAION.', 'sources': ''}

NOTE: Raw output from API for nochat case is a string of a python dict and will remain so if other entries are added to dict:

{'response': "I'm h2oGPT, a large language model by H2O.ai, the visionary leader in democratizing AI.", 'sources': ''}


"""
import ast
import time
import os
import markdown  # pip install markdown
import pytest
from bs4 import BeautifulSoup  # pip install beautifulsoup4

from enums import DocumentChoices, LangChainAction

debug = False

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'


def get_client(serialize=True):
    from gradio_client import Client

    client = Client(os.getenv('HOST', "http://localhost:7860"), serialize=serialize)
    if debug:
        print(client.view_api(all_endpoints=True))
    return client


def get_args(prompt, prompt_type, chat=False, stream_output=False,
             max_new_tokens=50,
             top_k_docs=3,
             langchain_mode='Disabled',
             langchain_action=LangChainAction.QUERY.value,
             langchain_agents=[],
             prompt_dict=None):
    from collections import OrderedDict
    kwargs = OrderedDict(instruction=prompt if chat else '',  # only for chat=True
                         iinput='',  # only for chat=True
                         context='',
                         # streaming output is supported, loops over and outputs each generation in streaming mode
                         # but leave stream_output=False for simple input/output mode
                         stream_output=stream_output,
                         prompt_type=prompt_type,
                         prompt_dict=prompt_dict,
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
                         langchain_mode=langchain_mode,
                         langchain_action=langchain_action,
                         langchain_agents=langchain_agents,
                         top_k_docs=top_k_docs,
                         chunk=True,
                         chunk_size=512,
                         document_subset=DocumentChoices.Relevant.name,
                         document_choice=[],
                         )
    from evaluate_params import eval_func_param_names
    assert len(set(eval_func_param_names).difference(set(list(kwargs.keys())))) == 0
    if chat:
        # add chatbot output on end.  Assumes serialize=False
        kwargs.update(dict(chatbot=[]))

    return kwargs, list(kwargs.values())


@pytest.mark.skip(reason="For manual use against some server, no server launched")
def test_client_basic(prompt_type='human_bot'):
    return run_client_nochat(prompt='Who are you?', prompt_type=prompt_type, max_new_tokens=50)


def run_client_nochat(prompt, prompt_type, max_new_tokens):
    kwargs, args = get_args(prompt, prompt_type, chat=False, max_new_tokens=max_new_tokens)

    api_name = '/submit_nochat'
    client = get_client(serialize=True)
    res = client.predict(
        *tuple(args),
        api_name=api_name,
    )
    print("Raw client result: %s" % res, flush=True)
    res_dict = dict(prompt=kwargs['instruction_nochat'], iinput=kwargs['iinput_nochat'],
                    response=md_to_text(res))
    print(res_dict)
    return res_dict, client


@pytest.mark.skip(reason="For manual use against some server, no server launched")
def test_client_basic_api(prompt_type='human_bot'):
    return run_client_nochat_api(prompt='Who are you?', prompt_type=prompt_type, max_new_tokens=50)


def run_client_nochat_api(prompt, prompt_type, max_new_tokens):
    kwargs, args = get_args(prompt, prompt_type, chat=False, max_new_tokens=max_new_tokens)

    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    client = get_client(serialize=True)
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    print("Raw client result: %s" % res, flush=True)
    res_dict = dict(prompt=kwargs['instruction_nochat'], iinput=kwargs['iinput_nochat'],
                    response=md_to_text(ast.literal_eval(res)['response']),
                    sources=ast.literal_eval(res)['sources'])
    print(res_dict)
    return res_dict, client


@pytest.mark.skip(reason="For manual use against some server, no server launched")
def test_client_basic_api_lean(prompt_type='human_bot'):
    return run_client_nochat_api_lean(prompt='Who are you?', prompt_type=prompt_type, max_new_tokens=50)


def run_client_nochat_api_lean(prompt, prompt_type, max_new_tokens):
    kwargs = dict(instruction_nochat=prompt)

    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    client = get_client(serialize=True)
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    print("Raw client result: %s" % res, flush=True)
    res_dict = dict(prompt=kwargs['instruction_nochat'],
                    response=md_to_text(ast.literal_eval(res)['response']),
                    sources=ast.literal_eval(res)['sources'])
    print(res_dict)
    return res_dict, client


@pytest.mark.skip(reason="For manual use against some server, no server launched")
def test_client_basic_api_lean_morestuff(prompt_type='human_bot'):
    return run_client_nochat_api_lean_morestuff(prompt='Who are you?', prompt_type=prompt_type, max_new_tokens=50)


def run_client_nochat_api_lean_morestuff(prompt, prompt_type='human_bot', max_new_tokens=512):
    kwargs = dict(
        instruction='',
        iinput='',
        context='',
        stream_output=False,
        prompt_type=prompt_type,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=256,
        min_new_tokens=0,
        early_stopping=False,
        max_time=20,
        repetition_penalty=1.0,
        num_return_sequences=1,
        do_sample=True,
        chat=False,
        instruction_nochat=prompt,
        iinput_nochat='',
        langchain_mode='Disabled',
        langchain_action=LangChainAction.QUERY.value,
        langchain_agents=[],
        top_k_docs=4,
        document_subset=DocumentChoices.Relevant.name,
        document_choice=[],
    )

    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    client = get_client(serialize=True)
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    print("Raw client result: %s" % res, flush=True)
    res_dict = dict(prompt=kwargs['instruction_nochat'],
                    response=md_to_text(ast.literal_eval(res)['response']),
                    sources=ast.literal_eval(res)['sources'])
    print(res_dict)
    return res_dict, client


@pytest.mark.skip(reason="For manual use against some server, no server launched")
def test_client_chat(prompt_type='human_bot'):
    return run_client_chat(prompt='Who are you?', prompt_type=prompt_type, stream_output=False, max_new_tokens=50,
                           langchain_mode='Disabled',
                           langchain_action=LangChainAction.QUERY.value,
                           langchain_agents=[])


@pytest.mark.skip(reason="For manual use against some server, no server launched")
def test_client_chat_stream(prompt_type='human_bot'):
    return run_client_chat(prompt="Tell a very long kid's story about birds.", prompt_type=prompt_type,
                           stream_output=True, max_new_tokens=512,
                           langchain_mode='Disabled',
                           langchain_action=LangChainAction.QUERY.value,
                           langchain_agents=[])


def run_client_chat(prompt, prompt_type, stream_output, max_new_tokens,
                    langchain_mode, langchain_action, langchain_agents,
                    prompt_dict=None):
    client = get_client(serialize=False)

    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens,
                            langchain_mode=langchain_mode,
                            langchain_action=langchain_action,
                            langchain_agents=langchain_agents,
                            prompt_dict=prompt_dict)
    return run_client(client, prompt, args, kwargs)


def run_client(client, prompt, args, kwargs, do_md_to_text=True, verbose=False):
    assert kwargs['chat'], "Chat mode only"
    res = client.predict(*tuple(args), api_name='/instruction')
    args[-1] += [res[-1]]

    res_dict = kwargs
    res_dict['prompt'] = prompt
    if not kwargs['stream_output']:
        res = client.predict(*tuple(args), api_name='/instruction_bot')
        res_dict['response'] = res[0][-1][1]
        print(md_to_text(res_dict['response'], do_md_to_text=do_md_to_text))
        return res_dict, client
    else:
        job = client.submit(*tuple(args), api_name='/instruction_bot')
        res1 = ''
        while not job.done():
            outputs_list = job.communicator.job.outputs
            if outputs_list:
                res = job.communicator.job.outputs[-1]
                res1 = res[0][-1][-1]
                res1 = md_to_text(res1, do_md_to_text=do_md_to_text)
                print(res1)
            time.sleep(0.1)
        full_outputs = job.outputs()
        if verbose:
            print('job.outputs: %s' % str(full_outputs))
        # ensure get ending to avoid race
        # -1 means last response if streaming
        # 0 means get text_output, ignore exception_text
        # 0 means get list within text_output that looks like [[prompt], [answer]]
        # 1 means get bot answer, so will have last bot answer
        res_dict['response'] = md_to_text(full_outputs[-1][0][0][1], do_md_to_text=do_md_to_text)
        return res_dict, client


@pytest.mark.skip(reason="For manual use against some server, no server launched")
def test_client_nochat_stream(prompt_type='human_bot'):
    return run_client_nochat_gen(prompt="Tell a very long kid's story about birds.", prompt_type=prompt_type,
                                 stream_output=True, max_new_tokens=512,
                                 langchain_mode='Disabled',
                                 langchain_action=LangChainAction.QUERY.value,
                                 langchain_agents=[])


def run_client_nochat_gen(prompt, prompt_type, stream_output, max_new_tokens,
                          langchain_mode, langchain_action, langchain_agents):
    client = get_client(serialize=False)

    kwargs, args = get_args(prompt, prompt_type, chat=False, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                            langchain_action=langchain_action, langchain_agents=langchain_agents)
    return run_client_gen(client, prompt, args, kwargs)


def run_client_gen(client, prompt, args, kwargs, do_md_to_text=True, verbose=False):
    res_dict = kwargs
    res_dict['prompt'] = prompt
    if not kwargs['stream_output']:
        res = client.predict(str(dict(kwargs)), api_name='/submit_nochat_api')
        res_dict['response'] = res[0]
        print(md_to_text(res_dict['response'], do_md_to_text=do_md_to_text))
        return res_dict, client
    else:
        job = client.submit(str(dict(kwargs)), api_name='/submit_nochat_api')
        while not job.done():
            outputs_list = job.communicator.job.outputs
            if outputs_list:
                res = job.communicator.job.outputs[-1]
                res_dict = ast.literal_eval(res)
                print('Stream: %s' % res_dict['response'])
            time.sleep(0.1)
        res_list = job.outputs()
        assert len(res_list) > 0, "No response, check server"
        res = res_list[-1]
        res_dict = ast.literal_eval(res)
        print('Final: %s' % res_dict['response'])
        return res_dict, client


def md_to_text(md, do_md_to_text=True):
    if not do_md_to_text:
        return md
    assert md is not None, "Markdown is None"
    html = markdown.markdown(md)
    soup = BeautifulSoup(html, features='html.parser')
    return soup.get_text()


def run_client_many(prompt_type='human_bot'):
    ret1, _ = test_client_chat(prompt_type=prompt_type)
    ret2, _ = test_client_chat_stream(prompt_type=prompt_type)
    ret3, _ = test_client_nochat_stream(prompt_type=prompt_type)
    ret4, _ = test_client_basic(prompt_type=prompt_type)
    ret5, _ = test_client_basic_api(prompt_type=prompt_type)
    ret6, _ = test_client_basic_api_lean(prompt_type=prompt_type)
    ret7, _ = test_client_basic_api_lean_morestuff(prompt_type=prompt_type)
    return ret1, ret2, ret3, ret4, ret5, ret6, ret7


if __name__ == '__main__':
    run_client_many()
