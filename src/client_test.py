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

from src.utils import is_gradio_version4

try:
    from enums import DocumentSubset, LangChainAction
except:
    from src.enums import DocumentSubset, LangChainAction

from tests.utils import get_inf_server

debug = False

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'


def get_client(serialize=not is_gradio_version4):
    from gradio_client import Client

    client = Client(get_inf_server(), serialize=serialize)
    if debug:
        print(client.view_api(all_endpoints=True))
    return client


def get_args(prompt, prompt_type=None, chat=False, stream_output=False,
             max_new_tokens=50,
             top_k_docs=3,
             langchain_mode='Disabled',
             add_chat_history_to_context=True,
             langchain_action=LangChainAction.QUERY.value,
             langchain_agents=[],
             prompt_dict=None,
             version=None,
             h2ogpt_key=None,
             visible_models=None,
             system_prompt='',  # default of no system prompt triggered by empty string
             add_search_to_context=False,
             chat_conversation=None,
             text_context_list=None,
             document_choice=[],
             document_source_substrings=[],
             document_source_substrings_op='and',
             document_content_substrings=[],
             document_content_substrings_op='and',
             max_time=20,
             repetition_penalty=1.0,
             do_sample=True,
             ):
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
                         penalty_alpha=0,
                         num_beams=1,
                         max_new_tokens=max_new_tokens,
                         min_new_tokens=0,
                         early_stopping=False,
                         max_time=max_time,
                         repetition_penalty=repetition_penalty,
                         num_return_sequences=1,
                         do_sample=do_sample,
                         chat=chat,
                         instruction_nochat=prompt if not chat else '',
                         iinput_nochat='',  # only for chat=False
                         langchain_mode=langchain_mode,
                         add_chat_history_to_context=add_chat_history_to_context,
                         langchain_action=langchain_action,
                         langchain_agents=langchain_agents,
                         top_k_docs=top_k_docs,
                         chunk=True,
                         chunk_size=512,
                         document_subset=DocumentSubset.Relevant.name,
                         document_choice=[] or document_choice,
                         document_source_substrings=[] or document_source_substrings,
                         document_source_substrings_op='and' or document_source_substrings_op,
                         document_content_substrings=[] or document_content_substrings,
                         document_content_substrings_op='and' or document_content_substrings_op,
                         pre_prompt_query=None,
                         prompt_query=None,
                         pre_prompt_summary=None,
                         prompt_summary=None,
                         hyde_llm_prompt=None,
                         system_prompt=system_prompt,
                         image_audio_loaders=None,
                         pdf_loaders=None,
                         url_loaders=None,
                         jq_schema=None,
                         extract_frames=None,
                         llava_prompt=None,
                         visible_models=visible_models,
                         h2ogpt_key=h2ogpt_key,
                         add_search_to_context=add_search_to_context,
                         chat_conversation=chat_conversation,
                         text_context_list=text_context_list,
                         docs_ordering_type=None,
                         min_max_new_tokens=None,
                         max_input_tokens=None,
                         max_total_input_tokens=None,
                         docs_token_handling=None,
                         docs_joiner=None,
                         hyde_level=0,
                         hyde_template=None,
                         hyde_show_only_final=False,
                         doc_json_mode=False,

                         chatbot_role='None',
                         speaker='None',
                         tts_language='autodetect',
                         tts_speed=1.0,
                         )
    diff = 0
    if version is None:
        # latest
        version = 1
    if version == 0:
        diff = 1
    if version >= 1:
        kwargs.update(dict(system_prompt=system_prompt))
        diff = 0

    from evaluate_params import eval_func_param_names
    assert len(set(eval_func_param_names).difference(set(list(kwargs.keys())))) == diff
    if chat:
        # add chatbot output on end.  Assumes serialize=False
        kwargs.update(dict(chatbot=[]))

    return kwargs, list(kwargs.values())


@pytest.mark.skip(reason="For manual use against some server, no server launched")
def test_client_basic(prompt_type='human_bot', version=None, visible_models=None, prompt='Who are you?',
                      h2ogpt_key=None):
    return run_client_nochat(prompt=prompt, prompt_type=prompt_type, max_new_tokens=50, version=version,
                             visible_models=visible_models, h2ogpt_key=h2ogpt_key)


"""
time HOST=https://gpt-internal.h2o.ai PYTHONPATH=. pytest -n 20 src/client_test.py::test_client_basic_benchmark
32 seconds to answer 20 questions at once with 70B llama2 on 4x A100 80GB using TGI 0.9.3
"""


@pytest.mark.skip(reason="For manual use against some server, no server launched")
@pytest.mark.parametrize("id", range(20))
def test_client_basic_benchmark(id, prompt_type='human_bot', version=None):
    return run_client_nochat(prompt="""
/nfs4/llm/h2ogpt/h2ogpt/bin/python /home/arno/pycharm-2022.2.2/plugins/python/helpers/pycharm/_jb_pytest_runner.py --target src/client_test.py::test_client_basic
Testing started at 8:41 AM ...
Launching pytest with arguments src/client_test.py::test_client_basic --no-header --no-summary -q in /nfs4/llm/h2ogpt

============================= test session starts ==============================
collecting ...
src/client_test.py:None (src/client_test.py)
ImportError while importing test module '/nfs4/llm/h2ogpt/src/client_test.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
h2ogpt/lib/python3.10/site-packages/_pytest/python.py:618: in _importtestmodule
    mod = import_path(self.path, mode=importmode, root=self.config.rootpath)
h2ogpt/lib/python3.10/site-packages/_pytest/pathlib.py:533: in import_path
    importlib.import_module(module_name)
/usr/lib/python3.10/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
<frozen importlib._bootstrap>:1050: in _gcd_import
    ???
<frozen importlib._bootstrap>:1027: in _find_and_load
    ???
<frozen importlib._bootstrap>:1006: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:688: in _load_unlocked
    ???
h2ogpt/lib/python3.10/site-packages/_pytest/assertion/rewrite.py:168: in exec_module
    exec(co, module.__dict__)
src/client_test.py:51: in <module>
    from enums import DocumentSubset, LangChainAction
E   ModuleNotFoundError: No module named 'enums'


collected 0 items / 1 error

=============================== 1 error in 0.14s ===============================
ERROR: not found: /nfs4/llm/h2ogpt/src/client_test.py::test_client_basic
(no name '/nfs4/llm/h2ogpt/src/client_test.py::test_client_basic' in any of [<Module client_test.py>])


Process finished with exit code 4

What happened?
""", prompt_type=prompt_type, max_new_tokens=100, version=version)


def run_client_nochat(prompt, prompt_type, max_new_tokens, version=None, h2ogpt_key=None, visible_models=None):
    kwargs, args = get_args(prompt, prompt_type, chat=False, max_new_tokens=max_new_tokens, version=version,
                            visible_models=visible_models, h2ogpt_key=h2ogpt_key)

    api_name = '/submit_nochat'
    client = get_client(serialize=not is_gradio_version4)
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
def test_client_basic_api(prompt_type='human_bot', version=None, h2ogpt_key=None):
    return run_client_nochat_api(prompt='Who are you?', prompt_type=prompt_type, max_new_tokens=50, version=version,
                                 h2ogpt_key=h2ogpt_key)


def run_client_nochat_api(prompt, prompt_type, max_new_tokens, version=None, h2ogpt_key=None):
    kwargs, args = get_args(prompt, prompt_type, chat=False, max_new_tokens=max_new_tokens, version=version,
                            h2ogpt_key=h2ogpt_key)

    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    client = get_client(serialize=not is_gradio_version4)
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
def test_client_basic_api_lean(prompt='Who are you?', prompt_type='human_bot', version=None, h2ogpt_key=None,
                               chat_conversation=None, system_prompt=''):
    return run_client_nochat_api_lean(prompt=prompt, prompt_type=prompt_type, max_new_tokens=50,
                                      version=version, h2ogpt_key=h2ogpt_key,
                                      chat_conversation=chat_conversation,
                                      system_prompt=system_prompt)


def run_client_nochat_api_lean(prompt, prompt_type, max_new_tokens, version=None, h2ogpt_key=None,
                               chat_conversation=None, system_prompt=''):
    kwargs = dict(instruction_nochat=prompt, h2ogpt_key=h2ogpt_key, chat_conversation=chat_conversation,
                  system_prompt=system_prompt)

    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    client = get_client(serialize=not is_gradio_version4)
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    print("Raw client result: %s" % res, flush=True)
    res_dict = dict(prompt=kwargs['instruction_nochat'],
                    response=md_to_text(ast.literal_eval(res)['response']),
                    sources=ast.literal_eval(res)['sources'],
                    h2ogpt_key=h2ogpt_key)
    print(res_dict)
    return res_dict, client


@pytest.mark.skip(reason="For manual use against some server, no server launched")
def test_client_basic_api_lean_morestuff(prompt_type='human_bot', version=None, h2ogpt_key=None):
    return run_client_nochat_api_lean_morestuff(prompt='Who are you?', prompt_type=prompt_type, max_new_tokens=50,
                                                version=version, h2ogpt_key=h2ogpt_key)


def run_client_nochat_api_lean_morestuff(prompt, prompt_type='human_bot', max_new_tokens=512, version=None,
                                         h2ogpt_key=None):
    kwargs = dict(
        instruction='',
        iinput='',
        context='',
        stream_output=False,
        prompt_type=prompt_type,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        penalty_alpha=0,
        num_beams=1,
        max_new_tokens=1024,
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
        add_chat_history_to_context=True,
        langchain_action=LangChainAction.QUERY.value,
        langchain_agents=[],
        top_k_docs=4,
        document_subset=DocumentSubset.Relevant.name,
        document_choice=[],
        document_source_substrings=[],
        document_source_substrings_op='and',
        document_content_substrings=[],
        document_content_substrings_op='and',
        h2ogpt_key=h2ogpt_key,
        add_search_to_context=False,
    )

    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    client = get_client(serialize=not is_gradio_version4)
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    print("Raw client result: %s" % res, flush=True)
    res_dict = dict(prompt=kwargs['instruction_nochat'],
                    response=md_to_text(ast.literal_eval(res)['response']),
                    sources=ast.literal_eval(res)['sources'],
                    h2ogpt_key=h2ogpt_key)
    print(res_dict)
    return res_dict, client


@pytest.mark.skip(reason="For manual use against some server, no server launched")
def test_client_chat(prompt_type='human_bot', version=None, h2ogpt_key=None):
    return run_client_chat(prompt='Who are you?', prompt_type=prompt_type, stream_output=False, max_new_tokens=50,
                           langchain_mode='Disabled',
                           langchain_action=LangChainAction.QUERY.value,
                           langchain_agents=[],
                           version=version,
                           h2ogpt_key=h2ogpt_key)


@pytest.mark.skip(reason="For manual use against some server, no server launched")
def test_client_chat_stream(prompt_type='human_bot', version=None, h2ogpt_key=None):
    return run_client_chat(prompt="Tell a very long kid's story about birds.", prompt_type=prompt_type,
                           stream_output=True, max_new_tokens=512,
                           langchain_mode='Disabled',
                           langchain_action=LangChainAction.QUERY.value,
                           langchain_agents=[],
                           version=version,
                           h2ogpt_key=h2ogpt_key)


def run_client_chat(prompt='',
                    stream_output=None,
                    max_new_tokens=128,
                    langchain_mode='Disabled',
                    langchain_action=LangChainAction.QUERY.value,
                    langchain_agents=[],
                    prompt_type=None, prompt_dict=None,
                    version=None,
                    h2ogpt_key=None,
                    chat_conversation=None,
                    system_prompt='',
                    document_choice=[],
                    document_content_substrings=[],
                    document_content_substrings_op='and',
                    document_source_substrings=[],
                    document_source_substrings_op='and',
                    top_k_docs=3,
                    max_time=20,
                    repetition_penalty=1.0,
                    do_sample=True):
    client = get_client(serialize=False)

    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens,
                            langchain_mode=langchain_mode,
                            langchain_action=langchain_action,
                            langchain_agents=langchain_agents,
                            prompt_dict=prompt_dict,
                            version=version,
                            h2ogpt_key=h2ogpt_key,
                            chat_conversation=chat_conversation,
                            system_prompt=system_prompt,
                            document_choice=document_choice,
                            document_source_substrings=document_source_substrings,
                            document_source_substrings_op=document_source_substrings_op,
                            document_content_substrings=document_content_substrings,
                            document_content_substrings_op=document_content_substrings_op,
                            top_k_docs=top_k_docs,
                            max_time=max_time,
                            repetition_penalty=repetition_penalty,
                            do_sample=do_sample)
    return run_client(client, prompt, args, kwargs)


def run_client(client, prompt, args, kwargs, do_md_to_text=True, verbose=False):
    if is_gradio_version4:
        kwargs['answer_with_sources'] = True
        kwargs['show_accordions'] = True
        kwargs['append_sources_to_answer'] = True
        kwargs['append_sources_to_chat'] = False
        kwargs['show_link_in_sources'] = True
        res_dict, client = run_client_gen(client, kwargs, do_md_to_text=do_md_to_text)
        res_dict['response'] += str(res_dict['sources_str'])
        return res_dict, client
        # FIXME: https://github.com/gradio-app/gradio/issues/6592

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
            outputs_list = job.outputs().copy()
            if outputs_list:
                res = outputs_list[-1]
                res1 = res[0][-1][-1]
                res1 = md_to_text(res1, do_md_to_text=do_md_to_text)
                print(res1)
            time.sleep(0.1)
        full_outputs = job.outputs().copy()
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
def test_client_nochat_stream(prompt_type='human_bot', version=None, h2ogpt_key=None):
    return run_client_nochat_gen(prompt="Tell a very long kid's story about birds.", prompt_type=prompt_type,
                                 stream_output=True, max_new_tokens=512,
                                 langchain_mode='Disabled',
                                 langchain_action=LangChainAction.QUERY.value,
                                 langchain_agents=[],
                                 version=version,
                                 h2ogpt_key=h2ogpt_key)


def run_client_nochat_gen(prompt, prompt_type, stream_output, max_new_tokens,
                          langchain_mode, langchain_action, langchain_agents, version=None,
                          h2ogpt_key=None):
    client = get_client(serialize=False)

    kwargs, args = get_args(prompt, prompt_type, chat=False, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                            langchain_action=langchain_action, langchain_agents=langchain_agents,
                            version=version, h2ogpt_key=h2ogpt_key)
    return run_client_gen(client, kwargs)


def run_client_gen(client, kwargs, do_md_to_text=True):
    res_dict = kwargs
    res_dict['prompt'] = kwargs['instruction'] or kwargs['instruction_nochat']
    if not kwargs['stream_output']:
        res = client.predict(str(dict(kwargs)), api_name='/submit_nochat_api')
        res_dict1 = ast.literal_eval(res)
        res_dict.update(res_dict1)
        print(md_to_text(res_dict['response'], do_md_to_text=do_md_to_text))
        return res_dict, client
    else:
        job = client.submit(str(dict(kwargs)), api_name='/submit_nochat_api')
        while not job.done():
            outputs_list = job.outputs().copy()
            if outputs_list:
                res = outputs_list[-1]
                res_dict1 = ast.literal_eval(res)
                print('Stream: %s' % res_dict1['response'])
            time.sleep(0.1)
        res_list = job.outputs().copy()
        assert len(res_list) > 0, "No response, check server"
        res = res_list[-1]
        res_dict1 = ast.literal_eval(res)
        print('Final: %s' % res_dict1['response'])
        res_dict.update(res_dict1)
        return res_dict, client


def md_to_text(md, do_md_to_text=True):
    if not do_md_to_text:
        return md
    assert md is not None, "Markdown is None"
    html = markdown.markdown(md)
    soup = BeautifulSoup(html, features='html.parser')
    return soup.get_text()


def run_client_many(prompt_type='human_bot', version=None, h2ogpt_key=None):
    kwargs = dict(prompt_type=prompt_type, version=version, h2ogpt_key=h2ogpt_key)
    ret1, _ = test_client_chat(**kwargs)
    ret2, _ = test_client_chat_stream(**kwargs)
    ret3, _ = test_client_nochat_stream(**kwargs)
    ret4, _ = test_client_basic(**kwargs)
    ret5, _ = test_client_basic_api(**kwargs)
    ret6, _ = test_client_basic_api_lean(**kwargs)
    ret7, _ = test_client_basic_api_lean_morestuff(**kwargs)
    return ret1, ret2, ret3, ret4, ret5, ret6, ret7


if __name__ == '__main__':
    run_client_many()
