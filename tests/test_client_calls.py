import ast
import json
import os, sys
import shutil
import tempfile
import time

import pytest

from tests.utils import wrap_test_forked, make_user_path_test, get_llama, get_inf_server, get_inf_port, count_tokens, \
    count_tokens_llm
from src.client_test import get_client, get_args, run_client_gen
from src.enums import LangChainAction, LangChainMode, no_model_str, no_lora_str, no_server_str, DocumentChoice, \
    db_types_full
from src.utils import get_githash, remove, download_simple, hash_file, makedirs, lg_to_gr, FakeTokenizer


@wrap_test_forked
def test_client1():
    os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
    sys.modules.pop('gpt_langchain', None)
    sys.modules.pop('langchain', None)

    from src.gen import main
    main(base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b', prompt_type='human_bot', chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False)

    from src.client_test import test_client_basic
    res_dict, _ = test_client_basic()
    assert res_dict['prompt'] == 'Who are you?'
    assert res_dict['iinput'] == ''
    assert 'I am h2oGPT' in res_dict['response'] or "I'm h2oGPT" in res_dict['response'] or 'I’m h2oGPT' in res_dict[
        'response']


@wrap_test_forked
def test_client1_lock_choose_model():
    os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
    sys.modules.pop('gpt_langchain', None)
    sys.modules.pop('langchain', None)

    from src.gen import main
    base1 = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    base2 = 'distilgpt2'
    model_lock = [dict(base_model=base1, prompt_type='human_bot'),
                  dict(base_model=base2, prompt_type='plain')]
    main(chat=False, model_lock=model_lock,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False)

    from src.client_test import test_client_basic

    for prompt_type in ['human_bot', None, '']:
        for visible_models in [None, 0, base1]:
            prompt = 'Who are you?'
            res_dict, _ = test_client_basic(visible_models=visible_models, prompt=prompt,
                                            prompt_type=prompt_type)
            assert res_dict['prompt'] == prompt
            assert res_dict['iinput'] == ''
            assert 'I am h2oGPT' in res_dict['response'] or "I'm h2oGPT" in res_dict['response'] or 'I’m h2oGPT' in \
                   res_dict[
                       'response']

    for prompt_type in ['plain', None, '']:
        for visible_models in [1, base2]:
            prompt = 'The sky is'
            res_dict, _ = test_client_basic(visible_models=visible_models, prompt=prompt,
                                            prompt_type=prompt_type)
            assert res_dict['prompt'] == prompt
            assert res_dict['iinput'] == ''
            assert 'the limit of time' in res_dict['response']


@pytest.mark.parametrize("base_model", [
    # 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2',  # can't handle
    'llama',
])
@wrap_test_forked
def test_client1_context(base_model):
    os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
    sys.modules.pop('gpt_langchain', None)
    sys.modules.pop('langchain', None)

    from src.gen import main
    main(base_model=base_model, prompt_type='prompt_answer', chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False)

    from gradio_client import Client
    client = Client(get_inf_server())

    # string of dict for input
    prompt = 'Who are you?'
    if base_model == 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2':
        context = """<|answer|>I am a pixie filled with fairy dust<|endoftext|><|prompt|>What kind of pixie are you?<|endoftext|><|answer|>Magical<|endoftext|>"""
    else:
        # FYI llama70b even works with falcon prompt_answer context
        context = """[/INST] I am a pixie filled with fairy dust </s><s>[INST] What kind of pixie are you? [/INST] Magical"""
    kwargs = dict(instruction_nochat=prompt, context=context)
    res = client.predict(str(dict(kwargs)), api_name='/submit_nochat_api')

    # string of dict for output
    response = ast.literal_eval(res)['response']
    print(response)
    assert """I am a mischievous pixie, always up to no good! *wink* But don't worry, I won't play any tricks on you... unless you want me to, that is. *giggles*
As for my fairy dust, it's a special blend of sparkly, shimmering magic that can grant wishes and make dreams come true. *twinkle eyes* Would you like some? *offers a tiny vial of sparkles*""" in response or \
           """I am a mischievous pixie, always up to no good! *winks* But don't worry, I won't play any tricks on you... unless you want me to, that is. *giggles*
   As for my fairy dust, it's a special blend of sparkly, shimmering magic that can grant wishes and make dreams come true. *twinkle* Would you like some? *offers a tiny vial of sparkles*""" in response or \
           """I am a mischievous pixie""" in response


@wrap_test_forked
def test_client1api():
    os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
    sys.modules.pop('gpt_langchain', None)
    sys.modules.pop('langchain', None)

    from src.gen import main
    main(base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b', prompt_type='human_bot', chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False)

    from src.client_test import test_client_basic_api
    res_dict, _ = test_client_basic_api()
    assert res_dict['prompt'] == 'Who are you?'
    assert res_dict['iinput'] == ''
    assert 'I am h2oGPT' in res_dict['response'] or "I'm h2oGPT" in res_dict['response'] or 'I’m h2oGPT' in res_dict[
        'response']


@pytest.mark.parametrize("admin_pass", ['', 'foodoo1234'])
@pytest.mark.parametrize("save_dir", [None, 'save_foodoo1234'])
@wrap_test_forked
def test_client1api_lean(save_dir, admin_pass):
    from src.gen import main
    base_model = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    os.environ['ADMIN_PASS'] = admin_pass
    os.environ['GET_GITHASH'] = '1'
    main(base_model=base_model, prompt_type='human_bot', chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False,
         save_dir=save_dir)

    client1 = get_client(serialize=True)

    from gradio_utils.grclient import GradioClient
    client2 = GradioClient(get_inf_server())
    client2.refresh_client()  # test refresh

    for client in [client1, client2]:
        api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
        prompt = 'Who are you?'
        kwargs = dict(instruction_nochat=prompt)
        # pass string of dict.  All entries are optional, but expect at least instruction_nochat to be filled
        res = client.predict(str(dict(kwargs)), api_name=api_name)
        res = ast.literal_eval(res)
        if save_dir:
            assert 'base_model' in res['save_dict']
            assert res['save_dict']['base_model'] == base_model
            assert res['save_dict']['error'] in [None, '']
            assert 'extra_dict' in res['save_dict']
            assert res['save_dict']['extra_dict']['ntokens'] > 0
            assert res['save_dict']['extra_dict']['t_generate'] > 0
            assert res['save_dict']['extra_dict']['tokens_persecond'] > 0

        print("Raw client result: %s" % res, flush=True)
        response = res['response']

        assert 'I am h2oGPT' in response or "I'm h2oGPT" in response or 'I’m h2oGPT' in response

        api_name = '/system_info_dict'
        # pass string of dict.  All entries are optional, but expect at least instruction_nochat to be filled
        ADMIN_PASS = os.getenv('ADMIN_PASS', admin_pass)
        res = client.predict(ADMIN_PASS, api_name=api_name)
        res = json.loads(res)
        assert isinstance(res, dict)
        assert res['base_model'] == base_model, "Problem with res=%s" % res
        assert 'device' in res
        assert res['hash'] == get_githash()

        api_name = '/system_hash'
        res = client.predict(api_name=api_name)
        assert res == get_githash()

        res = client.predict(api_name=api_name)
        assert res == get_githash()

    client2.refresh_client()  # test refresh
    res = client.predict(api_name=api_name)
    assert res == get_githash()

    res = client2.get_server_hash()
    assert res == get_githash()


@wrap_test_forked
def test_client1api_lean_lock_choose_model():
    from src.gen import main
    base1 = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    base2 = 'distilgpt2'
    model_lock = [dict(base_model=base1, prompt_type='human_bot'),
                  dict(base_model=base2, prompt_type='plain')]
    save_dir = 'save_test'
    main(model_lock=model_lock, chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False,
         save_dir=save_dir)

    client = get_client(serialize=True)
    for prompt_type in ['human_bot', None, '', 'plain']:
        for visible_models in [None, 0, base1, 1, base2]:
            base_model = base1 if visible_models in [None, 0, base1] else base2
            if base_model == base1 and prompt_type == 'plain':
                continue
            if base_model == base2 and prompt_type == 'human_bot':
                continue

            api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
            if base_model == base1:
                prompt = 'Who are you?'
            else:
                prompt = 'The sky is'
            kwargs = dict(instruction_nochat=prompt, prompt_type=prompt_type, visible_models=visible_models)
            # pass string of dict.  All entries are optional, but expect at least instruction_nochat to be filled
            res = client.predict(str(dict(kwargs)), api_name=api_name)
            res = ast.literal_eval(res)
            assert save_dir
            assert 'base_model' in res['save_dict']
            assert res['save_dict']['base_model'] == base_model
            assert res['save_dict']['error'] in [None, '']
            assert 'extra_dict' in res['save_dict']
            assert res['save_dict']['extra_dict']['ntokens'] > 0
            assert res['save_dict']['extra_dict']['t_generate'] > 0
            assert res['save_dict']['extra_dict']['tokens_persecond'] > 0

            print("Raw client result: %s" % res, flush=True)
            response = res['response']

            if base_model == base1:
                assert 'I am h2oGPT' in response or "I'm h2oGPT" in response or 'I’m h2oGPT' in response
            else:
                assert 'the limit of time' in response

    api_name = '/model_names'
    res = client.predict(api_name=api_name)
    res = ast.literal_eval(res)
    assert [x['base_model'] for x in res] == [base1, base2]
    assert res == [{'base_model': 'h2oai/h2ogpt-oig-oasst1-512-6_9b', 'prompt_type': 'human_bot',
                    'prompt_dict': {'promptA': '', 'promptB': '', 'PreInstruct': '<human>: ', 'PreInput': None,
                                    'PreResponse': '<bot>:',
                                    'terminate_response': ['\n<human>:', '\n<bot>:', '<human>:', '<bot>:', '<bot>:'],
                                    'chat_sep': '\n', 'chat_turn_sep': '\n', 'humanstr': '<human>:', 'botstr': '<bot>:',
                                    'generates_leading_space': True, 'system_prompt': ''}, 'load_8bit': False,
                    'load_4bit': False, 'low_bit_mode': 1, 'load_half': True, 'load_gptq': '', 'load_awq': '',
                    'load_exllama': False, 'use_safetensors': False, 'revision': None, 'use_gpu_id': True, 'gpu_id': 0,
                    'compile_model': True, 'use_cache': None,
                    'llamacpp_dict': {'n_gpu_layers': 100, 'use_mlock': True, 'n_batch': 1024, 'n_gqa': 0,
                                      'model_path_llama': 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin',
                                      'model_name_gptj': 'ggml-gpt4all-j-v1.3-groovy.bin',
                                      'model_name_gpt4all_llama': 'ggml-wizardLM-7B.q4_2.bin',
                                      'model_name_exllama_if_no_config': 'TheBloke/Nous-Hermes-Llama2-GPTQ'},
                    'model_path_llama': 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin',
                    'model_name_gptj': 'ggml-gpt4all-j-v1.3-groovy.bin',
                    'model_name_gpt4all_llama': 'ggml-wizardLM-7B.q4_2.bin',
                    'model_name_exllama_if_no_config': 'TheBloke/Nous-Hermes-Llama2-GPTQ', 'rope_scaling': {},
                    'max_seq_len': 2048, 'exllama_dict': {}, 'gptq_dict': {}, 'attention_sinks': False, 'sink_dict': {},
                    'truncation_generation': False, 'hf_model_dict': {}},
                   {'base_model': 'distilgpt2', 'prompt_type': 'plain',
                    'prompt_dict': {'promptA': '', 'promptB': '', 'PreInstruct': '<human>: ', 'PreInput': None,
                                    'PreResponse': '<bot>:',
                                    'terminate_response': ['\n<human>:', '\n<bot>:', '<human>:', '<bot>:', '<bot>:'],
                                    'chat_sep': '\n', 'chat_turn_sep': '\n', 'humanstr': '<human>:', 'botstr': '<bot>:',
                                    'generates_leading_space': True, 'system_prompt': ''}, 'load_8bit': False,
                    'load_4bit': False, 'low_bit_mode': 1, 'load_half': True, 'load_gptq': '', 'load_awq': '',
                    'load_exllama': False, 'use_safetensors': False, 'revision': None, 'use_gpu_id': True, 'gpu_id': 0,
                    'compile_model': True, 'use_cache': None,
                    'llamacpp_dict': {'n_gpu_layers': 100, 'use_mlock': True, 'n_batch': 1024, 'n_gqa': 0,
                                      'model_path_llama': 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin',
                                      'model_name_gptj': 'ggml-gpt4all-j-v1.3-groovy.bin',
                                      'model_name_gpt4all_llama': 'ggml-wizardLM-7B.q4_2.bin',
                                      'model_name_exllama_if_no_config': 'TheBloke/Nous-Hermes-Llama2-GPTQ'},
                    'model_path_llama': 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin',
                    'model_name_gptj': 'ggml-gpt4all-j-v1.3-groovy.bin',
                    'model_name_gpt4all_llama': 'ggml-wizardLM-7B.q4_2.bin',
                    'model_name_exllama_if_no_config': 'TheBloke/Nous-Hermes-Llama2-GPTQ', 'rope_scaling': {},
                    'max_seq_len': 1024, 'exllama_dict': {}, 'gptq_dict': {}, 'attention_sinks': False, 'sink_dict': {},
                    'truncation_generation': False, 'hf_model_dict': {}}]


@wrap_test_forked
def test_client1api_lean_chat_server():
    from src.gen import main
    main(base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b', prompt_type='human_bot', chat=True,
         stream_output=True, gradio=True, num_beams=1, block_gradio_exit=False)

    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    prompt = 'Who are you?'

    kwargs = dict(instruction_nochat=prompt)
    client = get_client(serialize=True)
    # pass string of dict.  All entries are optional, but expect at least instruction_nochat to be filled
    res = client.predict(str(dict(kwargs)), api_name=api_name)

    print("Raw client result: %s" % res, flush=True)
    response = ast.literal_eval(res)['response']

    assert 'I am h2oGPT' in response or "I'm h2oGPT" in response or 'I’m h2oGPT' in response


@wrap_test_forked
def test_client_chat_nostream():
    res_dict, client = run_client_chat_with_server(stream_output=False)
    assert 'I am h2oGPT' in res_dict['response'] or "I'm h2oGPT" in res_dict['response'] or 'I’m h2oGPT' in res_dict[
        'response']


@wrap_test_forked
def test_client_chat_nostream_gpt4all():
    res_dict, client = run_client_chat_with_server(stream_output=False, base_model='gptj', prompt_type='gptj')
    assert 'I am a computer program designed to assist' in res_dict['response'] or \
           'I am a person who enjoys' in res_dict['response'] or \
           'I am a student at' in res_dict['response'] or \
           'I am a person who' in res_dict['response']


@wrap_test_forked
def test_client_chat_nostream_gpt4all_llama():
    res_dict, client = run_client_chat_with_server(stream_output=False, base_model='gpt4all_llama', prompt_type='gptj')
    assert 'What do you want from me?' in res_dict['response'] or \
           'What do you want?' in res_dict['response'] or \
           'What is your name and title?' in res_dict['response'] or \
           'I can assist you with any information' in res_dict['response'] or \
           'I can provide information or assistance' in res_dict['response'] or \
           'am a student' in res_dict['response']


@pytest.mark.need_tokens
@wrap_test_forked
def test_client_chat_nostream_llama7b():
    prompt_type, full_path = get_llama()
    res_dict, client = run_client_chat_with_server(stream_output=False, base_model='llama',
                                                   prompt_type=prompt_type, model_path_llama=full_path)
    assert "am a virtual assistant" in res_dict['response'] or \
           'am a student' in res_dict['response'] or \
           "My name is John." in res_dict['response'] or \
           "how can I assist" in res_dict['response']


def run_client_chat_with_server(prompt='Who are you?', stream_output=False, max_new_tokens=256,
                                base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b', prompt_type='human_bot',
                                langchain_mode='Disabled',
                                langchain_action=LangChainAction.QUERY.value,
                                langchain_agents=[],
                                user_path=None,
                                langchain_modes=['UserData', 'MyData', 'Disabled', 'LLM'],
                                model_path_llama='llama-2-7b-chat.ggmlv3.q8_0.bin',
                                docs_ordering_type='reverse_ucurve_sort'):
    if langchain_mode == 'Disabled':
        os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
        sys.modules.pop('gpt_langchain', None)
        sys.modules.pop('langchain', None)

    from src.gen import main
    main(base_model=base_model,
         model_path_llama=model_path_llama,
         prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         docs_ordering_type=docs_ordering_type)

    from src.client_test import run_client_chat
    res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''
    return res_dict, client


@wrap_test_forked
def test_client_chat_stream():
    run_client_chat_with_server(stream_output=True)


def run_client_nochat_with_server(prompt='Who are you?', stream_output=False, max_new_tokens=256,
                                  base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b', prompt_type='human_bot',
                                  langchain_mode='Disabled', langchain_action=LangChainAction.QUERY.value,
                                  langchain_agents=[],
                                  user_path=None,
                                  langchain_modes=['UserData', 'MyData', 'Disabled', 'LLM'],
                                  docs_ordering_type='reverse_ucurve_sort'):
    if langchain_mode == 'Disabled':
        os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
        sys.modules.pop('gpt_langchain', None)
        sys.modules.pop('langchain', None)

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, langchain_action=langchain_action, langchain_agents=langchain_agents,
         user_path=user_path,
         langchain_modes=langchain_modes,
         docs_ordering_type=docs_ordering_type)

    from src.client_test import run_client_nochat_gen
    res_dict, client = run_client_nochat_gen(prompt=prompt, prompt_type=prompt_type,
                                             stream_output=stream_output,
                                             max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                             langchain_action=langchain_action, langchain_agents=langchain_agents)
    assert 'Birds' in res_dict['response'] or \
           'and can learn new things' in res_dict['response'] or \
           'Once upon a time' in res_dict['response']
    return res_dict, client


@wrap_test_forked
def test_client_nochat_stream():
    run_client_nochat_with_server(stream_output=True, prompt="Tell a very long kid's story about birds.")


@wrap_test_forked
def test_client_chat_stream_langchain():
    user_path = make_user_path_test()
    prompt = "What is h2oGPT?"
    res_dict, client = run_client_chat_with_server(prompt=prompt, stream_output=True, langchain_mode="UserData",
                                                   user_path=user_path,
                                                   langchain_modes=['UserData', 'MyData', 'Disabled', 'LLM'],
                                                   docs_ordering_type=None,  # for 6_9 dumb model for testing
                                                   )
    # below wouldn't occur if didn't use LangChain with README.md,
    # raw LLM tends to ramble about H2O.ai and what it does regardless of question.
    # bad answer about h2o.ai is just becomes dumb model, why flipped context above,
    # but not stable over different systems
    assert 'h2oGPT is a large language model' in res_dict['response'] or \
           'H2O.ai is a technology company' in res_dict['response'] or \
           'an open-source project' in res_dict['response'] or \
           'h2oGPT is a project that allows' in res_dict['response'] or \
           'h2oGPT is a language model trained' in res_dict['response'] or \
           'h2oGPT is a large-scale' in res_dict['response']


@pytest.mark.parametrize("max_new_tokens", [256, 2048])
@pytest.mark.parametrize("top_k_docs", [3, 100])
@wrap_test_forked
def test_client_chat_stream_langchain_steps(max_new_tokens, top_k_docs):
    os.environ['VERBOSE_PIPELINE'] = '1'
    user_path = make_user_path_test()

    stream_output = True
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'  # 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    prompt_type = 'llama2'  # 'human_bot'
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'LLM', 'Disabled', 'LLM']

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         top_k_docs=top_k_docs,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         docs_ordering_type=None,  # for 6_9
         )

    from src.client_test import get_client, get_args, run_client
    client = get_client(serialize=False)

    # QUERY1
    prompt = "What is h2oGPT?"
    langchain_mode = 'UserData'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens,
                            top_k_docs=top_k_docs,
                            langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    assert ('a large language model' in res_dict['response'] or
            '2oGPT is an open-source, Apache V2 project' in res_dict['response'] or
            'language model trained' in res_dict['response'] or
            'H2O GPT is a language model' in res_dict['response'] or
            'H2O GPT is a chatbot framework' in res_dict['response'] or
            'H2O GPT is a chatbot that can be trained' in res_dict['response'] or
            'A large language model (LLM)' in res_dict['response'] or
            'GPT-based language model' in res_dict['response'] or
            'H2O.ai is a technology company' in res_dict['response'] or
            'an open-source project' in res_dict['response'] or
            'is a company that provides' in res_dict['response'] or
            'h2oGPT is a project that' in res_dict['response'] or
            'for querying and summarizing documents' in res_dict['response'] or
            'Python-based platform for training' in res_dict['response'] or
            'h2oGPT is an open-source' in res_dict['response']
            ) \
           and ('FAQ.md' in res_dict['response'] or 'README.md' in res_dict['response'])

    # QUERY1
    prompt = "What is Whisper?"
    langchain_mode = 'UserData'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens,
                            top_k_docs=top_k_docs,
                            langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    # wrong answer given wrong docs
    assert ('A secure chatbot that uses a large language' in res_dict['response'] or
            'Whisper is a chatbot' in res_dict['response'] or
            'Whisper is a privacy-focused chatbot platform' in res_dict['response'] or
            'h2oGPT' in res_dict['response'] or
            'A secure, private, and anonymous chat platform' in res_dict['response'] or
            'Whisper is a privacy-preserving' in res_dict['response'] or
            'A chatbot that uses a large language model' in res_dict['response'] or
            'This is a config file for Whisper' in res_dict['response'] or
            'Whisper is a secure messaging app' in res_dict['response'] or
            'secure, private, and anonymous chatbot' in res_dict['response'] or
            'Whisper is a secure, anonymous, and encrypted' in res_dict['response'] or
            'secure, decentralized, and anonymous chat platform' in res_dict['response'] or
            'A low-code development framework' in res_dict['response'] or
            'secure messaging app' in res_dict['response'] or
            'privacy-focused messaging app that allows' in res_dict['response'] or
            'A low-code AI app development framework' in res_dict['response'] or
            'anonymous communication platform' in res_dict['response'] or
            'A privacy-focused chat app' in res_dict['response'] or
            'A platform for deploying' in res_dict['response'] or
            'A language model that can be used to generate text.' in res_dict['response'] or
            'a chat app that' in res_dict['response']
            ) \
           and ('FAQ.md' in res_dict['response'] or 'README.md' in res_dict['response'])

    # QUERY2
    prompt = "What is h2oGPT?"
    langchain_mode = 'LLM'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens,
                            top_k_docs=top_k_docs,
                            langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    # i.e. answers wrongly without data, dumb model, but also no docs at all since cutoff entirely
    assert 'h2oGPT is a variant of the popular GPT' in res_dict['response'] and '.md' not in res_dict['response']

    # QUERY3
    prompt = "What is whisper?"
    langchain_mode = 'UserData'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens,
                            top_k_docs=top_k_docs,
                            langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    # odd answer since no whisper docs, but still shows some docs at very low score
    assert ('h2oGPT' in res_dict['response'] or
            'A chatbot that can whisper to you' in res_dict['response'] or
            'whisper is a simple' in res_dict['response'] or
            'Whisper is a tool for generating text from a model' in res_dict['response'] or
            'Whisper is a chatbot platform' in res_dict['response'] or
            'whisper is a chatbot framework' in res_dict['response'] or
            'whisper is a tool for training language models' in res_dict['response'] or
            'whisper is a secure messaging app' in res_dict['response'] or
            'LLaMa-based models are not commercially viable' in res_dict['response'] or
            'A text-based chatbot that' in res_dict['response'] or
            'A secure, private, and anonymous chat service' in res_dict['response'] or
            'LLaMa is a language' in res_dict['response'] or
            'chatbot that can' in res_dict['response'] or
            'A secure, private, and anonymous chatbot' in res_dict['response'] or
            'A secure, encrypted chat service that allows' in res_dict['response'] or
            'A secure, private, and encrypted chatbot' in res_dict['response'] or
            'A secret communication system used' in res_dict['response'] or
            'H2O AI Cloud is a cloud-based platform' in res_dict['response'] or
            'is a platform for deploying' in res_dict['response'] or
            'is a language model that is trained' in res_dict['response'] or
            'private, and anonymous communication' in res_dict['response'] or
            'The large language model is' in res_dict['response'] or
            'is a private, secure, and encrypted' in res_dict['response'] or
            'H2O AI is a cloud-based platform for building' in res_dict['response'] or
            'a private chat between' in res_dict['response'] or
            'whisper is a chat bot' in res_dict['response']
            ) \
           and '.md' in res_dict['response']


@pytest.mark.parametrize("system_prompt", ['', None, 'None', 'auto', 'You are a goofy lion who talks to kids'])
# @pytest.mark.parametrize("system_prompt", [None])
@pytest.mark.parametrize("chat_conversation",
                         [None, [('Who are you?', 'I am a big pig who loves to tell kid stories')]])
# @pytest.mark.parametrize("chat_conversation", [[('Who are you?', 'I am a big pig who loves to tell kid stories')]])
@wrap_test_forked
def test_client_system_prompts(system_prompt, chat_conversation):
    stream_output = True
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'  # 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    prompt_type = 'llama2'  # 'human_bot'

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         )

    from src.client_test import get_client, get_args, run_client
    client = get_client(serialize=False)

    # QUERY1
    prompt = "Who are you?"
    for client_type in ['chat', 'nochat']:
        if client_type == 'chat':
            kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                                    system_prompt=system_prompt,
                                    chat_conversation=chat_conversation)

            res_dict, client = run_client(client, prompt, args, kwargs)
        else:
            api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
            kwargs = dict(instruction_nochat=prompt,
                          system_prompt=system_prompt,
                          chat_conversation=chat_conversation)
            # pass string of dict.  All entries are optional, but expect at least instruction_nochat to be filled
            res = client.predict(str(dict(kwargs)), api_name=api_name)
            res_dict = ast.literal_eval(res)

        if not chat_conversation:
            if system_prompt == 'You are a goofy lion who talks to kids':
                assert 'ROAR!' in res_dict['response'] and 'respectful' not in res_dict[
                    'response'] and 'developed by Meta' not in res_dict['response']
            elif system_prompt == '':
                assert "developed by Meta" in res_dict['response'] and 'respectful' not in res_dict[
                    'response'] and 'ROAR!' not in res_dict['response']
            elif system_prompt in [None, 'auto', 'None']:
                assert 'respectful' in res_dict['response'] and 'ROAR!' not in res_dict[
                    'response'] and 'developed by Meta' not in res_dict['response']
        else:
            if system_prompt == 'You are a goofy lion who talks to kids':
                # system prompt overwhelms chat conversation
                assert "I'm a goofy lion" in res_dict['response'] or \
                       "goofiest lion" in res_dict['response'] or \
                       "I'm the coolest lion around" in res_dict['response']
            elif system_prompt == '':
                # empty system prompt gives room for chat conversation to control
                assert "My name is Porky" in res_dict['response']
            elif system_prompt in [None, 'auto', 'None']:
                # conservative default system_prompt makes it ignore chat
                assert "not a real person" in res_dict['response'] or \
                       "I don't have personal experiences or feelings" in res_dict['response']


@pytest.mark.need_tokens
@pytest.mark.parametrize("max_new_tokens", [256, 2048])
@pytest.mark.parametrize("top_k_docs", [3, 100])
@pytest.mark.parametrize("auto_migrate_db", [False, True])
@wrap_test_forked
def test_client_chat_stream_langchain_steps2(max_new_tokens, top_k_docs, auto_migrate_db):
    os.environ['VERBOSE_PIPELINE'] = '1'
    # full user data
    from src.make_db import make_db_main
    make_db_main(download_some=True)
    user_path = None  # shouldn't be necessary, db already made

    stream_output = True
    max_new_tokens = 256
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'  # 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    prompt_type = 'llama2'  # 'human_bot'
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'github h2oGPT', 'LLM', 'Disabled']

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         verbose=True,
         auto_migrate_db=auto_migrate_db)

    from src.client_test import get_client, get_args, run_client
    client = get_client(serialize=False)

    # QUERY1
    prompt = "Who are you?"
    langchain_mode = 'LLM'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    assert 'an AI assistant developed by Meta' in res_dict['response'] and 'FAQ.md' not in res_dict['response']

    # QUERY2
    prompt = "What is whisper?"
    langchain_mode = 'UserData'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    res1 = 'large-scale speech recognition model' in res_dict['response'] and 'whisper.pdf' in res_dict['response']
    res2 = 'speech recognition system' in res_dict['response'] and 'whisper.pdf' in res_dict['response']
    assert res1 or res2

    # QUERY3
    prompt = "What is h2oGPT"
    langchain_mode = 'github h2oGPT'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    assert ('h2oGPT is an open-source, fully permissive, commercially usable, and fully trained language model' in
            res_dict['response'] or
            'A new open-source language model that is fully permissive' in res_dict['response'] or
            'h2oGPT is an open-source' in res_dict['response'] or
            'h2oGPT is an open-source, fully permissive, commercially usable' in res_dict['response']
            ) and \
           'README.md' in res_dict['response']


@wrap_test_forked
def test_doc_hash():
    remove('langchain_modes.pkl')
    user_path = make_user_path_test()

    stream_output = True
    base_model = ''
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'LLM', 'Disabled']

    os.environ['SHOULD_NEW_FILES'] = '1'
    os.environ['GRADIO_SERVER_PORT'] = str(get_inf_port())
    from src.gen import main
    main(base_model=base_model, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         score_model='None',
         docs_ordering_type=None,  # for 6_9
         )

    # repeat, shouldn't reload
    os.environ.pop('SHOULD_NEW_FILES', None)
    os.environ['NO_NEW_FILES'] = '1'
    os.environ['GRADIO_SERVER_PORT'] = str(get_inf_port() + 1)
    from src.gen import main
    main(base_model=base_model, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         score_model='None',
         docs_ordering_type=None,  # for 6_9
         )


@wrap_test_forked
def test_client_chat_stream_long():
    prompt = 'Tell a very long story about cute birds for kids.'
    res_dict, client = run_client_chat_with_server(prompt=prompt, stream_output=True, max_new_tokens=1024)
    assert 'Once upon a time' in res_dict['response']


@wrap_test_forked
def test_autogptq():
    prompt = 'Who are you?'
    stream_output = False
    max_new_tokens = 256
    base_model = 'TheBloke/Nous-Hermes-13B-GPTQ'
    load_gptq = 'model'
    use_safetensors = True
    prompt_type = 'instruct'
    langchain_mode = 'Disabled'
    langchain_action = LangChainAction.QUERY.value
    langchain_agents = []
    user_path = None
    langchain_modes = ['UserData', 'MyData', 'LLM', 'Disabled']
    docs_ordering_type = 'reverse_sort'
    from src.gen import main
    main(base_model=base_model, load_gptq=load_gptq,
         use_safetensors=use_safetensors,
         prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         docs_ordering_type=docs_ordering_type)

    from src.client_test import run_client_chat
    res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''
    assert "am a virtual assistant" in res_dict['response']


@wrap_test_forked
def test_autoawq():
    prompt = 'Who are you?'
    stream_output = False
    max_new_tokens = 256
    base_model = 'TheBloke/Llama-2-13B-chat-AWQ'
    load_awq = 'model'
    use_safetensors = True
    prompt_type = 'llama2'
    langchain_mode = 'Disabled'
    langchain_action = LangChainAction.QUERY.value
    langchain_agents = []
    user_path = None
    langchain_modes = ['UserData', 'MyData', 'LLM', 'Disabled']
    docs_ordering_type = 'reverse_sort'
    from src.gen import main
    main(base_model=base_model, load_awq=load_awq,
         use_safetensors=use_safetensors,
         prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         docs_ordering_type=docs_ordering_type)

    from src.client_test import run_client_chat
    res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''
    assert "am a virtual assistant" in res_dict['response'] or \
           "Hello! My name is LLaMA, I'm a large language model trained by a team" in res_dict['response']


@pytest.mark.parametrize("mode", ['a', 'b', 'c'])
@wrap_test_forked
def test_exllama(mode):
    prompt = 'Who are you?'
    stream_output = False
    max_new_tokens = 256
    if mode == 'c':
        base_model = 'TheBloke/Llama-2-70B-chat-GPTQ'
        exllama_dict = {}
    elif mode == 'b':
        base_model = 'TheBloke/Llama-2-70B-chat-GPTQ'
        exllama_dict = {'set_auto_map': '20,20'}
    elif mode == 'a':
        base_model = 'TheBloke/Llama-2-7B-chat-GPTQ'
        exllama_dict = {}
    else:
        raise RuntimeError("Bad mode=%s" % mode)
    load_exllama = True
    prompt_type = 'llama2'
    langchain_mode = 'Disabled'
    langchain_action = LangChainAction.QUERY.value
    langchain_agents = []
    user_path = None
    langchain_modes = ['UserData', 'MyData', 'LLM', 'Disabled']
    docs_ordering_type = 'reverse_ucurve_sort'
    from src.gen import main
    main(base_model=base_model,
         load_exllama=load_exllama, exllama_dict=exllama_dict,
         prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         docs_ordering_type=docs_ordering_type)

    from src.client_test import run_client_chat
    res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''
    assert "I'm LLaMA, an AI assistant" in res_dict['response'] or \
           "I am LLaMA" in res_dict['response'] or \
           "Hello! My name is Llama, I'm a large language model trained by Meta AI." in res_dict['response']


@pytest.mark.parametrize("attention_sinks", [False, True])  # mistral goes beyond context just fine up to 32k
@pytest.mark.parametrize("max_seq_len", [4096, 8192])
@wrap_test_forked
def test_attention_sinks(max_seq_len, attention_sinks):
    # full user data
    from src.make_db import make_db_main
    make_db_main(download_some=True)
    user_path = None  # shouldn't be necessary, db already made

    prompt = 'Give an extremely detailed report that is well-structured with step-by-step sections (and elaborate details for each section) that describes the documents. Do not stop or end the report, just keep generating forever in never-ending report.'
    stream_output = True
    max_new_tokens = 100000
    max_max_new_tokens = max_new_tokens
    base_model = 'mistralai/Mistral-7B-Instruct-v0.1'
    prompt_type = 'mistral'
    langchain_mode = 'UserData'
    langchain_action = LangChainAction.QUERY.value
    langchain_agents = []
    langchain_modes = ['UserData', 'MyData', 'LLM', 'Disabled']
    docs_ordering_type = 'reverse_ucurve_sort'
    document_choice = ['user_path/./whisper.pdf']  # only exact matches allowed currently
    top_k_docs = -1
    from src.gen import main
    main(base_model=base_model,
         attention_sinks=attention_sinks,
         user_path=user_path,
         prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         max_max_new_tokens=max_max_new_tokens,
         langchain_mode=langchain_mode,
         langchain_modes=langchain_modes,
         top_k_docs=top_k_docs,  # has no effect for client if client passes different number
         max_seq_len=max_seq_len,
         # mistral is 32k if don't say, easily run GPU OOM even on 48GB (even with --use_gpu_id=False)
         docs_ordering_type=docs_ordering_type,
         cut_distance=1.8,  # probably should allow control via API/UI
         sink_dict={'attention_sink_size': 4, 'attention_sink_window_size': 4096} if attention_sinks else {},
         )

    from src.client_test import run_client_chat
    res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents,
                                       document_choice=document_choice, top_k_docs=top_k_docs,
                                       max_time=600, repetition_penalty=1.07, do_sample=False)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''
    assert len(res_dict['response']) > 2500, "%s %s" % (len(res_dict['response']), res_dict['response'])


@pytest.mark.skip(reason="Local file required")
@wrap_test_forked
def test_client_long():
    os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
    sys.modules.pop('gpt_langchain', None)
    sys.modules.pop('langchain', None)

    from src.gen import main
    main(base_model='mosaicml/mpt-7b-storywriter', prompt_type='plain', chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False)

    with open("/home/jon/Downloads/Gatsby_PDF_FullText.txt") as f:
        prompt = f.readlines()

    from src.client_test import run_client_nochat
    res_dict, _ = run_client_nochat(prompt=prompt, prompt_type='plain', max_new_tokens=86000)
    print(res_dict['response'])


@wrap_test_forked
def test_fast_up():
    from src.gen import main
    main(gradio=True, block_gradio_exit=False)


@wrap_test_forked
def test_fast_up_auth():
    from src.gen import main
    main(gradio=True, block_gradio_exit=False, score_model='', langchain_mode='LLM', auth=[('jonny', 'dude')])
    # doesn't test login, has to be done manually


@wrap_test_forked
def test_fast_up_auth2():
    from src.gen import main
    main(gradio=True, block_gradio_exit=False, score_model='', langchain_mode='LLM', auth='')
    # doesn't test login, has to be done manually


@pytest.mark.parametrize("visible_models",
                         [None,
                          [0, 1],
                          "[0,1]",
                          "['h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3','gpt-3.5-turbo']",
                          ['h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3', 'gpt-3.5-turbo']
                          ])
@wrap_test_forked
def test_lock_up(visible_models):
    from src.gen import main
    main(gradio=True,
         model_lock=[{'base_model': 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3'},
                     {'base_model': 'distilgpt2'},
                     {'inference_server': 'openai_chat', 'base_model': 'gpt-3.5-turbo'}],
         visible_models=visible_models,
         model_lock_columns=3,
         gradio_size='small',
         height=400,
         save_dir='save_gpt_test1',
         max_max_new_tokens=2048,
         max_new_tokens=1024,
         langchain_mode='MyData',
         block_gradio_exit=False)


@pytest.mark.skipif(not os.getenv('STRESS'), reason="Only for stress testing already-running server")
@pytest.mark.parametrize("repeat", list(range(0, 100)))
@wrap_test_forked
def test_client_stress(repeat):
    # pip install pytest-repeat  # license issues, don't put with requirements
    # pip install pytest-timeout  # license issues, don't put with requirements
    #
    # CUDA_VISIBLE_DEVICES=0 SCORE_MODEL=None python generate.py --base_model=h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2 --langchain_mode=UserData --user_path=user_path --debug=True --concurrency_count=8
    #
    # timeout to mimic client disconnecting and generation still going, else too clean and doesn't fail STRESS=1
    # pytest -s -v -n 8 --timeout=30 tests/test_client_calls.py::test_client_stress 2> stress1.log
    # HOST=http://192.168.1.46:9999 STRESS=1 pytest -s -v -n 8 --timeout=1000 tests/test_client_calls.py::test_client_stress 2> stress1.log

    prompt = "Tell a very long kid's story about birds."
    # prompt = "Say exactly only one word."

    client = get_client(serialize=True)
    kwargs = dict(
        instruction='',
        max_new_tokens=200,
        min_new_tokens=1,
        max_time=300,
        do_sample=False,
        instruction_nochat=prompt,
    )

    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    print("Raw client result: %s" % res, flush=True)
    assert isinstance(res, str)
    res_dict = ast.literal_eval(res)
    assert 'response' in res_dict and res_dict['response']


@pytest.mark.skipif(not os.getenv('STRESS'), reason="Only for stress testing already-running server")
@pytest.mark.parametrize("repeat", list(range(0, 100)))
@wrap_test_forked
def test_client_stress_stream(repeat):
    prompt = "Tell a very long kid's story about birds."
    max_new_tokens = 200
    prompt_type = None
    langchain_mode = 'Disabled'
    stream_output = True
    chat = False

    client = get_client(serialize=True)
    kwargs, args = get_args(prompt, prompt_type, chat=chat, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)
    res_dict, client = run_client_gen(client, prompt, args, kwargs, do_md_to_text=False, verbose=False)

    assert 'response' in res_dict and res_dict['response']


@pytest.mark.skipif(not os.getenv('SERVER'),
                    reason="For testing remote text-generatino-inference server")
@wrap_test_forked
def test_text_generation_inference_server1():
    """
    e.g.
    SERVER on 192.168.1.46
    (alpaca) jon@gpu:/data/jon/h2o-llm$ CUDA_VISIBLE_DEVICES=0,1 docker run --gpus all --shm-size 2g -e NCCL_SHM_DISABLE=1 -p 6112:80 -v $HOME/.cache/huggingface/hub/:/data  ghcr.io/huggingface/text-generation-inference:latest --model-id h2oai/h2ogpt-oasst1-512-12b --max-input-length 2048 --max-total-tokens 4096 --sharded=true --num-shard=2 --disable-custom-kernels --quantize bitsandbytes --trust-remote-code --max-stop-sequences=6

    CLIENT on separate system
    HOST=http://192.168.1.46:6112 SERVER=1 pytest -s -v tests/test_client_calls.py::test_text_generation_inference_server1

    :return:
    """

    # Python client test:
    from text_generation import Client

    host = os.getenv("HOST", "http://127.0.0.1:6112")
    client = Client(host)
    print(client.generate("What is Deep Learning?", max_new_tokens=17).generated_text)

    text = ""
    for response in client.generate_stream("What is Deep Learning?", max_new_tokens=17):
        if not response.token.special:
            text += response.token.text
    assert 'Deep learning is a subfield of machine learning' in text

    # Curl Test (not really pass fail yet)
    import subprocess
    output = subprocess.run(['curl', '%s/generate' % host, '-X', 'POST', '-d',
                             '{"inputs":"<|prompt|>What is Deep Learning?<|endoftext|><|answer|>","parameters":{"max_new_tokens": 20, "truncate": 1024, "do_sample": false, "temperature": 0.1, "repetition_penalty": 1.2}}',
                             '-H', 'Content-Type: application/json',
                             '--user', 'user:bhx5xmu6UVX4'],
                            check=True, capture_output=True).stdout.decode()
    text = ast.literal_eval(output)['generated_text']
    assert 'Deep learning is a subfield of machine learning' in text or \
           'Deep learning refers to a class of machine learning' in text


@pytest.mark.need_tokens
@wrap_test_forked
@pytest.mark.parametrize("loaders", ['all', None])
@pytest.mark.parametrize("enforce_h2ogpt_api_key", [False, True])
def test_client_chat_stream_langchain_steps3(loaders, enforce_h2ogpt_api_key):
    os.environ['VERBOSE_PIPELINE'] = '1'
    user_path = make_user_path_test()

    if loaders is None:
        loaders = tuple([None, None, None, None])
    else:
        image_loaders_options0, image_loaders_options, \
            pdf_loaders_options0, pdf_loaders_options, \
            url_loaders_options0, url_loaders_options = \
            lg_to_gr(enable_ocr=True, enable_captions=True, enable_pdf_ocr=True,
                     enable_pdf_doctr=True,
                     use_pymupdf=True,
                     enable_doctr=True,
                     enable_pix2struct=True,
                     max_quality=True)
        loaders = [image_loaders_options, pdf_loaders_options, url_loaders_options, None]

    stream_output = True
    max_new_tokens = 256
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'  # 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    prompt_type = 'llama2'  # 'human_bot'
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'github h2oGPT', 'LLM', 'Disabled']

    from src.gen import main
    main_kwargs = {}
    h2ogpt_key = 'foodoo#'
    if enforce_h2ogpt_api_key:
        main_kwargs.update(dict(enforce_h2ogpt_api_key=True, h2ogpt_api_keys=[h2ogpt_key]))
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         **main_kwargs,
         verbose=True)

    from src.client_test import get_client, get_args, run_client
    # serialize=False would lead to returning dict for some objects or files for get_sources
    client = get_client(serialize=False)

    url = 'https://www.africau.edu/images/default/sample.pdf'
    test_file1 = os.path.join('/tmp/', 'sample1.pdf')
    download_simple(url, dest=test_file1)
    res = client.predict(test_file1,
                         langchain_mode, True, 512, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    # note moves from /tmp to stable path, even though not /tmp/gradio upload from UI
    assert 'file/%s/sample1.pdf' % user_path in res[2] or 'file/%s\sample1.pdf' % user_path in res[2]
    assert res[3] == ''

    # control langchain_mode
    user_path2 = makedirs('user_path2', use_base=True)  # so base accounted for
    langchain_mode2 = 'UserData2'
    remove(user_path2)
    remove('db_dir_%s' % langchain_mode2)
    new_langchain_mode_text = '%s, %s, %s' % (langchain_mode2, 'shared', user_path2)
    res = client.predict(langchain_mode, new_langchain_mode_text, api_name='/new_langchain_mode_text')
    assert res[0]['value'] == langchain_mode2
    # odd gradio change
    res0_choices = [x[0] for x in res[0]['choices']]
    assert langchain_mode2 in res0_choices
    assert res[1] == ''
    assert res[2]['headers'] == ['Collection', 'Type', 'Path', 'Directory', 'Embedding', 'DB']
    res[2]['data'] = [[x[0], x[1], x[2]] for x in res[2]['data']]  # ignore persist_directory
    assert res[2]['data'] == [['UserData', 'shared', user_path],
                              ['github h2oGPT', 'shared', ''],
                              ['MyData', 'personal', ''],
                              [langchain_mode2, 'shared', user_path2]]

    # url = 'https://unec.edu.az/application/uploads/2014/12/pdf-sample.pdf'
    test_file1 = os.path.join('/tmp/', 'pdf-sample.pdf')
    # download_simple(url, dest=test_file1)
    shutil.copy('tests/pdf-sample.pdf', test_file1)
    res = client.predict(test_file1, langchain_mode2, True, 512, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode2
    assert 'file/%s/pdf-sample.pdf' % user_path2 in res[2] or 'file/%s\pdf-sample.pdf' % user_path2 in res[2]
    assert 'sample1.pdf' not in res[2]  # ensure no leakage
    assert res[3] == ''

    # QUERY1
    prompt = "Is more text boring?"
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                            h2ogpt_key=h2ogpt_key)

    res_dict, client = run_client(client, prompt, args, kwargs)
    assert ('Yes, more text can be boring' in res_dict['response'] or
            "can be considered boring" in res_dict['response'] or
            "the text in the provided PDF file is quite repetitive and boring" in res_dict['response'] or
            "the provided PDF file is quite boring" in res_dict['response']) \
           and 'sample1.pdf' in res_dict['response']
    # QUERY2
    prompt = "What is a universal file format?"
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode2,
                            h2ogpt_key=h2ogpt_key)

    res_dict, client = run_client(client, prompt, args, kwargs)
    assert 'PDF' in res_dict['response'] and 'pdf-sample.pdf' in res_dict['response']

    # check sources, and do after so would detect leakage
    res = client.predict(langchain_mode, api_name='/get_sources')
    # is not actual data!
    assert isinstance(res[1], str)
    res = res[0]
    with open(res['name'], 'rb') as f:
        sources = f.read().decode()
    sources_expected = f'{user_path}/FAQ.md\n{user_path}/README.md\n{user_path}/pexels-evg-kowalievska-1170986_small.jpg\n{user_path}/sample1.pdf'
    assert sources == sources_expected or sources.replace('\\', '/').replace('\r', '') == sources_expected.replace(
        '\\', '/').replace('\r', '')

    res = client.predict(langchain_mode2, api_name='/get_sources')
    assert isinstance(res[1], str)
    res = res[0]
    with open(res['name'], 'rb') as f:
        sources = f.read().decode()
    sources_expected = """%s/pdf-sample.pdf""" % user_path2
    assert sources == sources_expected or sources.replace('\\', '/').replace('\r', '') == sources_expected.replace(
        '\\', '/').replace('\r', '')

    # check sources, and do after so would detect leakage
    res = client.predict(langchain_mode, api_name='/get_viewable_sources')
    assert isinstance(res[1], str)
    res = res[0]
    # is not actual data!
    with open(res['name'], 'rb') as f:
        sources = f.read().decode()
    sources_expected = f'{user_path}/FAQ.md\n{user_path}/README.md\n{user_path}/pexels-evg-kowalievska-1170986_small.jpg\n{user_path}/sample1.pdf'
    assert sources == sources_expected or sources.replace('\\', '/').replace('\r', '') == sources_expected.replace(
        '\\', '/').replace('\r', '')

    res = client.predict(langchain_mode2, api_name='/get_viewable_sources')
    assert isinstance(res[1], str)
    res = res[0]
    with open(res['name'], 'rb') as f:
        sources = f.read().decode()
    sources_expected = """%s/pdf-sample.pdf""" % user_path2
    assert sources == sources_expected or sources.replace('\\', '/').replace('\r', '') == sources_expected.replace(
        '\\', '/').replace('\r', '')

    # refresh
    shutil.copy('tests/next.txt', user_path)
    res = client.predict(langchain_mode, True, 512,
                         *loaders,
                         api_name='/refresh_sources')
    sources_expected = 'file/%s/next.txt' % user_path
    assert sources_expected in res or sources_expected.replace('\\', '/').replace('\r', '') in res.replace('\\',
                                                                                                           '/').replace(
        '\r', '\n')

    res = client.predict(langchain_mode, api_name='/get_sources')
    assert isinstance(res[1], str)
    res = res[0]
    # is not actual data!
    with open(res['name'], 'rb') as f:
        sources = f.read().decode()
    sources_expected = f'{user_path}/FAQ.md\n{user_path}/README.md\n{user_path}/next.txt\n{user_path}/pexels-evg-kowalievska-1170986_small.jpg\n{user_path}/sample1.pdf'
    assert sources == sources_expected or sources.replace('\\', '/').replace('\r', '') == sources_expected.replace(
        '\\', '/').replace('\r', '')

    # check sources, and do after so would detect leakage
    sources = ast.literal_eval(client.predict(langchain_mode, api_name='/get_sources_api'))
    assert isinstance(sources, list)
    sources_expected = ['user_path_test/FAQ.md', 'user_path_test/README.md', 'user_path_test/next.txt',
                        'user_path_test/pexels-evg-kowalievska-1170986_small.jpg', 'user_path_test/sample1.pdf']
    assert sources == sources_expected

    file_to_get = sources_expected[3]
    view_raw_text = False
    text_context_list = None
    source_dict = ast.literal_eval(
        client.predict(langchain_mode, file_to_get, view_raw_text, text_context_list, api_name='/get_document_api'))
    assert len(source_dict['contents']) == 1
    assert len(source_dict['metadatas']) == 1
    assert isinstance(source_dict['contents'][0], str)
    assert 'a cat sitting on a window' in source_dict['contents'][0]
    assert isinstance(source_dict['metadatas'][0], str)
    assert sources_expected[3] in source_dict['metadatas'][0]

    view_raw_text = True  # dict of metadatas stays dict instead of string
    source_dict = ast.literal_eval(
        client.predict(langchain_mode, file_to_get, view_raw_text, text_context_list, api_name='/get_document_api'))
    assert len(source_dict['contents']) == 2  # chunk_id=0 (query) and -1 (summarization)
    assert len(source_dict['metadatas']) == 2  # chunk_id=0 (query) and -1 (summarization)
    assert isinstance(source_dict['contents'][0], str)
    assert 'a cat sitting on a window' in source_dict['contents'][0]
    assert isinstance(source_dict['metadatas'][0], dict)
    assert sources_expected[3] == source_dict['metadatas'][0]['source']

    # even normal langchain_mode  passed to this should get the other langchain_mode2
    res = client.predict(langchain_mode, api_name='/load_langchain')
    res0_choices = [x[0] for x in res[0]['choices']]
    assert res0_choices == [langchain_mode, 'MyData', 'github h2oGPT', 'LLM', langchain_mode2]
    assert res[0]['value'] == langchain_mode
    assert res[1]['headers'] == ['Collection', 'Type', 'Path', 'Directory', 'Embedding', 'DB']
    res[1]['data'] = [[x[0], x[1], x[2]] for x in res[1]['data']]  # ignore persist_directory
    assert res[1]['data'] == [['UserData', 'shared', user_path],
                              ['github h2oGPT', 'shared', ''],
                              ['MyData', 'personal', ''],
                              [langchain_mode2, 'shared', user_path2]]

    # for pure-UI things where just input -> output often, just make sure no failure, if can
    res = client.predict(api_name='/export_chats')
    assert res is not None

    url = 'https://research.google/pubs/pub334.pdf'
    res = client.predict(url, langchain_mode, True, 512, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_url')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert url in res[2]
    assert res[3] == ''

    text = "Yufuu is a wonderful place and you should really visit because there is lots of sun."
    res = client.predict(text, langchain_mode, True, 512, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_text')
    assert res[0] is None
    assert res[1] == langchain_mode
    user_paste_dir = makedirs('user_paste', use_base=True)
    remove(user_paste_dir)
    sources_expected = 'file/%s/' % user_paste_dir
    assert sources_expected in res[2] or sources_expected.replace('\\', '/').replace('\r', '') in res[2].replace('\\',
                                                                                                                 '/').replace(
        '\r', '\n')
    assert res[3] == ''

    langchain_mode_my = LangChainMode.MY_DATA.value
    url = 'https://www.africau.edu/images/default/sample.pdf'
    test_file1 = os.path.join('/tmp/', 'sample1.pdf')
    download_simple(url, dest=test_file1)
    res = client.predict(test_file1, langchain_mode_my, True, 512, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode_my
    # will just use source location, e.g. for UI will be /tmp/gradio
    sources_expected = 'file//tmp/sample1.pdf'
    assert sources_expected in res[2] or sources_expected.replace('\\', '/').replace('\r', '') in res[2].replace('\\',
                                                                                                                 '/').replace(
        '\r', '\n')
    assert res[3] == ''

    # control langchain_mode
    user_path2b = ''
    langchain_mode2 = 'MyData2'
    new_langchain_mode_text = '%s, %s, %s' % (langchain_mode2, 'personal', user_path2b)
    res = client.predict(langchain_mode2, new_langchain_mode_text, api_name='/new_langchain_mode_text')
    assert res[0]['value'] == langchain_mode2
    res0_choices = [x[0] for x in res[0]['choices']]
    assert langchain_mode2 in res0_choices
    assert res[1] == ''
    assert res[2]['headers'] == ['Collection', 'Type', 'Path', 'Directory', 'Embedding', 'DB']
    res[2]['data'] = [[x[0], x[1], x[2]] for x in res[2]['data']]  # ignore persist_directory
    assert res[2]['data'] == [['UserData', 'shared', user_path],
                              ['github h2oGPT', 'shared', ''],
                              ['MyData', 'personal', ''],
                              ['UserData2', 'shared', user_path2],
                              [langchain_mode2, 'personal', '']]

    # url = 'https://unec.edu.az/application/uploads/2014/12/pdf-sample.pdf'
    test_file1 = os.path.join('/tmp/', 'pdf-sample.pdf')
    # download_simple(url, dest=test_file1)
    shutil.copy('tests/pdf-sample.pdf', test_file1)
    res = client.predict(test_file1, langchain_mode2, True, 512, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode2
    sources_expected = 'file//tmp/pdf-sample.pdf'
    assert sources_expected in res[2] or sources_expected.replace('\\', '/').replace('\r', '') in res[2].replace('\\',
                                                                                                                 '/').replace(
        '\r', '\n')
    assert 'sample1.pdf' not in res[2]  # ensure no leakage
    assert res[3] == ''

    urls = ['https://h2o.ai/company/team/leadership-team/',
            'https://arxiv.org/abs/1706.03762',
            'https://github.com/h2oai/h2ogpt',
            'https://h2o.ai'
            ]
    with tempfile.TemporaryDirectory() as tmp_user_path:
        urls_file = os.path.join(tmp_user_path, 'list.urls')
        with open(urls_file, 'wt') as f:
            f.write('\n'.join(urls))
        res = client.predict(urls_file, langchain_mode2, True, 512, True,
                             *loaders,
                             h2ogpt_key,
                             api_name='/add_file_api')
        assert res[0] is None
        assert res[1] == langchain_mode2
        assert [x in res[2] or x.replace('https', 'http') in res[2] for x in urls]
        assert res[3] == ''

    langchain_mode3 = 'MyData3'
    user_path3 = ''
    new_langchain_mode_text = '%s, %s, %s' % (langchain_mode3, 'personal', user_path3)
    res = client.predict(langchain_mode3, new_langchain_mode_text, api_name='/new_langchain_mode_text')
    assert res[0]['value'] == langchain_mode3
    res0_choices = [x[0] for x in res[0]['choices']]
    assert langchain_mode3 in res0_choices
    assert res[1] == ''
    assert res[2]['headers'] == ['Collection', 'Type', 'Path', 'Directory', 'Embedding', 'DB']
    res[2]['data'] = [[x[0], x[1], x[2]] for x in res[2]['data']]  # ignore persist_directory
    assert res[2]['data'] == [['UserData', 'shared', user_path],
                              ['github h2oGPT', 'shared', ''],
                              ['MyData', 'personal', ''],
                              ['UserData2', 'shared', user_path2],
                              [langchain_mode2, 'personal', ''],
                              [langchain_mode3, 'personal', ''],
                              ]

    with tempfile.TemporaryDirectory() as tmp_user_path:
        res = client.predict(urls, langchain_mode3, True, 512, True,
                             *loaders,
                             h2ogpt_key,
                             api_name='/add_url')
        print(res)
        assert res[0] is None
        assert res[1] == langchain_mode3
        assert [x in res[2] or x.replace('https', 'http') in res[2] for x in urls]
        assert res[3] == ''

    sources_text = client.predict(langchain_mode3, api_name='/show_sources')
    assert isinstance(sources_text, str)
    assert [x in sources_text or x.replace('https', 'http') in sources_text for x in urls]

    source_list = ast.literal_eval(client.predict(langchain_mode3, api_name='/get_sources_api'))
    source_list_assert = [x.replace('v1', '').replace('v7', '') for x in source_list]  # for arxiv for asserts
    assert isinstance(source_list, list)
    assert [x in source_list_assert or x.replace('https', 'http') in source_list_assert for x in urls]

    sources_text_after_delete = client.predict(source_list[0], langchain_mode3, api_name='/delete_sources')
    source_list_assert = [x.replace('v1', '').replace('v7', '') for x in source_list]  # for arxiv for asserts
    assert source_list_assert[0] not in sources_text_after_delete

    sources_state_after_delete = ast.literal_eval(client.predict(langchain_mode3, api_name='/get_sources_api'))
    sources_state_after_delete = [x.replace('v1', '').replace('v7', '') for x in
                                  sources_state_after_delete]  # for arxiv for asserts
    assert isinstance(sources_state_after_delete, list)
    source_list_assert = [x.replace('v1', '').replace('v7', '') for x in source_list]  # for arxiv for asserts
    assert source_list_assert[0] not in sources_state_after_delete

    res = client.predict(langchain_mode3, langchain_mode3, api_name='/remove_langchain_mode_text')
    assert res[0]['value'] == langchain_mode3
    res0_choices = [x[0] for x in res[0]['choices']]
    assert langchain_mode2 in res0_choices
    assert res[1] == ''
    assert res[2]['headers'] == ['Collection', 'Type', 'Path', 'Directory', 'Embedding', 'DB']
    res[2]['data'] = [[x[0], x[1], x[2]] for x in res[2]['data']]  # ignore persist_directory
    assert res[2]['data'] == [['UserData', 'shared', user_path],
                              ['github h2oGPT', 'shared', ''],
                              ['MyData', 'personal', ''],
                              ['UserData2', 'shared', user_path2],
                              [langchain_mode2, 'personal', '']]

    assert os.path.isdir("db_dir_%s" % langchain_mode)
    res = client.predict(langchain_mode, langchain_mode, api_name='/purge_langchain_mode_text')
    assert not os.path.isdir("db_dir_%s" % langchain_mode)
    assert res[0]['value'] == langchain_mode
    res0_choices = [x[0] for x in res[0]['choices']]
    assert langchain_mode not in res0_choices
    assert res[1] == ''
    assert res[2]['headers'] == ['Collection', 'Type', 'Path', 'Directory', 'Embedding', 'DB']
    res[2]['data'] = [[x[0], x[1], x[2]] for x in res[2]['data']]  # ignore persist_directory
    assert res[2]['data'] == [['github h2oGPT', 'shared', ''],
                              ['MyData', 'personal', ''],
                              ['UserData2', 'shared', 'user_path2'],
                              ['MyData2', 'personal', ''],
                              ]


@pytest.mark.need_tokens
@wrap_test_forked
def test_client_load_unload_models():
    os.environ['VERBOSE_PIPELINE'] = '1'
    user_path = make_user_path_test()

    stream_output = True
    max_new_tokens = 256
    base_model = ''
    prompt_type = 'human_bot'
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'github h2oGPT', 'LLM', 'Disabled']

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         score_model='',
         verbose=True)

    from src.client_test import get_client, get_args, run_client
    # serialize=False would lead to returning dict for some objects or files for get_sources
    client = get_client(serialize=False)

    model_choice = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    lora_choice = ''
    server_choice = ''
    # model_state
    prompt_type = ''
    model_load8bit_checkbox = False
    model_load4bit_checkbox = True
    model_low_bit_mode = 1
    model_load_gptq = ''
    model_load_awq = ''
    model_load_exllama_checkbox = False
    model_safetensors_checkbox = False
    model_revision = ''
    model_use_gpu_id_checkbox = True
    model_gpu = 0
    max_seq_len = 2048
    rope_scaling = '{}'
    # GGML:
    model_path_llama = ''
    model_name_gptj = ''
    model_name_gpt4all_llama = ''
    n_gpu_layers = 100
    n_batch = 128
    n_gqa = 0  # llama2 needs 8
    llamacpp_dict_more = '{}'
    system_prompt = None
    args_list = [model_choice, lora_choice, server_choice,
                 # model_state,
                 prompt_type,
                 model_load8bit_checkbox, model_load4bit_checkbox, model_low_bit_mode,
                 model_load_gptq, model_load_awq, model_load_exllama_checkbox,
                 model_safetensors_checkbox, model_revision,
                 model_use_gpu_id_checkbox, model_gpu,
                 max_seq_len, rope_scaling,
                 model_path_llama, model_name_gptj, model_name_gpt4all_llama,
                 n_gpu_layers, n_batch, n_gqa, llamacpp_dict_more,
                 system_prompt]
    res = client.predict(*tuple(args_list), api_name='/load_model')
    res_expected = ('h2oai/h2ogpt-oig-oasst1-512-6_9b', '', '', 'human_bot', {'__type__': 'update', 'maximum': 256},
                    {'__type__': 'update', 'maximum': 256})
    assert res == res_expected
    model_used, lora_used, server_used, prompt_type, max_new_tokens, min_new_tokens = res_expected

    prompt = "Who are you?"
    kwargs = dict(stream_output=stream_output, instruction=prompt)
    res_dict, client = run_client_gen(client, prompt, None, kwargs)
    response = res_dict['response']
    assert 'What do you want to be?' in response

    # unload
    args_list[0] = no_model_str
    res = client.predict(*tuple(args_list), api_name='/load_model')
    res_expected = (no_model_str, no_lora_str, no_server_str, '', {'__type__': 'update', 'maximum': 256},
                    {'__type__': 'update', 'maximum': 256})
    assert res == res_expected


@pytest.mark.need_tokens
@wrap_test_forked
def test_client_chat_stream_langchain_openai_embeddings():
    os.environ['VERBOSE_PIPELINE'] = '1'
    user_path = make_user_path_test()
    remove('db_dir_UserData')

    stream_output = True
    max_new_tokens = 256
    base_model = 'distilgpt2'
    prompt_type = 'human_bot'
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'github h2oGPT', 'LLM', 'Disabled']

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         use_openai_embedding=True,
         verbose=True)

    from src.client_test import get_client, get_args, run_client
    # serialize=False would lead to returning dict for some objects or files for get_sources
    client = get_client(serialize=False)

    url = 'https://www.africau.edu/images/default/sample.pdf'
    test_file1 = os.path.join('/tmp/', 'sample1.pdf')
    download_simple(url, dest=test_file1)
    h2ogpt_key = ''
    res = client.predict(test_file1, langchain_mode, True, 512, True,
                         None, None, None, None,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    # note moves from /tmp to stable path, even though not /tmp/gradio upload from UI
    assert 'file/%s/sample1.pdf' % user_path in res[2] or 'file/%s\sample1.pdf' % user_path in res[2]
    assert res[3] == ''

    from src.gpt_langchain import load_embed
    got_embedding, use_openai_embedding, hf_embedding_model = load_embed(persist_directory='db_dir_UserData')
    assert use_openai_embedding
    assert hf_embedding_model == 'hkunlp/instructor-large'  # but not used
    assert got_embedding


@pytest.mark.parametrize("stream_output", [True, False])
@pytest.mark.need_tokens
@wrap_test_forked
def test_client_clone(stream_output):
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'
    from src.gen import main
    main(base_model=base_model, block_gradio_exit=False, verbose=True)

    from gradio_utils.grclient import GradioClient
    client1 = GradioClient(get_inf_server())
    client1.setup()
    client2 = client1.clone()

    for client in [client1, client2]:
        prompt = "Who are you?"
        kwargs = dict(stream_output=stream_output, instruction=prompt)
        res_dict, client = run_client_gen(client, prompt, None, kwargs)
        response = res_dict['response']
        assert len(response) > 0
        sources = res_dict['sources']
        assert sources == ''


@pytest.mark.parametrize("max_time", [1, 5])
@pytest.mark.parametrize("stream_output", [True, False])
@pytest.mark.need_tokens
@wrap_test_forked
def test_client_timeout(stream_output, max_time):
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'
    from src.gen import main
    main(base_model=base_model, block_gradio_exit=False, verbose=True)

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server())

    prompt = "Tell a very long kid's story about birds"
    kwargs = dict(stream_output=stream_output, instruction=prompt, max_time=max_time)
    t0 = time.time()
    res_dict, client = run_client_gen(client, prompt, None, kwargs)
    response = res_dict['response']
    assert len(response) > 0
    assert time.time() - t0 < max_time * 2
    sources = res_dict['sources']
    assert sources == ''

    # get file for client to upload
    url = 'https://cdn.openai.com/papers/whisper.pdf'
    test_file1 = os.path.join('/tmp/', 'my_test_pdf.pdf')
    download_simple(url, dest=test_file1)

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server())

    # upload file(s).  Can be list or single file
    test_file_local, test_file_server = client.predict(test_file1, api_name='/upload_api')

    chunk = True
    chunk_size = 512
    langchain_mode = 'MyData'
    h2ogpt_key = ''
    res = client.predict(test_file_server,
                         langchain_mode, chunk, chunk_size, True,
                         None, None, None, None,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert os.path.basename(test_file_server) in res[2]
    assert res[3] == ''

    # ask for summary, need to use same client if using MyData
    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    instruction = "Give a very long detailed step-by-step description of what is Whisper paper about."
    kwargs = dict(langchain_mode=langchain_mode,
                  langchain_action="Query",
                  top_k_docs=4,
                  document_subset='Relevant',
                  document_choice=DocumentChoice.ALL.value,
                  max_new_tokens=1024,
                  max_time=max_time,
                  do_sample=False,
                  stream_output=stream_output,
                  )
    t0 = time.time()
    res_dict, client = run_client_gen(client, instruction, None, kwargs)
    response = res_dict['response']
    assert len(response) > 0
    # assert len(response) < max_time * 20  # 20 tokens/sec
    assert time.time() - t0 < max_time * 2
    sources = [x['source'] for x in res_dict['sources']]
    # only get source not empty list if break in inner loop, not gradio_runner loop, so good test of that too
    # this is why gradio timeout adds 10 seconds, to give inner a chance to produce references or other final info
    assert 'my_test_pdf.pdf' in sources[0]


# pip install pytest-timeout
# HOST=http://192.168.1.46:9999 STRESS=1 pytest -s -v -n 8 --timeout=1000 tests/test_client_calls.py::test_client_chat_stream_langchain_fake_embeddings_stress 2> stress1.log
@pytest.mark.skipif(not os.getenv('STRESS'), reason="Only for stress testing already-running server")
@pytest.mark.parametrize("repeat", list(range(0, 100)))
@wrap_test_forked
def test_client_chat_stream_langchain_fake_embeddings_stress(repeat):
    data_kind = 'helium3'
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'  # presumes remote server is llama-2 chat based
    local_server = False
    inference_server = None
    # inference_server = 'http://localhost:7860'
    return run_client_chat_stream_langchain_fake_embeddings(data_kind, base_model, local_server, inference_server)


# pip install pytest-timeout
# HOST=http://192.168.1.46:9999 STRESS=1 pytest -s -v -n 8 --timeout=1000 tests/test_client_calls.py::test_client_upload_simple 2> stress1.log
@pytest.mark.skipif(not os.getenv('STRESS'), reason="Only for stress testing already-running server")
@pytest.mark.parametrize("repeat", list(range(0, 100)))
@wrap_test_forked
def test_client_upload_simple(repeat):
    data_kind = 'helium3'
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'  # fake, just for tokenizer
    local_server = False
    inference_server = None
    # used with go_upload_gradio (say on remote machine) to test add_text
    return run_client_chat_stream_langchain_fake_embeddings(data_kind, base_model, local_server, inference_server,
                                                            simple=True)


# pip install pytest-timeout
# HOST=http://192.168.1.46:9999 STRESS=1 pytest -s -v -n 8 --timeout=1000 tests/test_client_calls.py::test_client_chat_stream_langchain_fake_embeddings_stress_no_llm 2> stress1.log
@pytest.mark.skipif(not os.getenv('STRESS'), reason="Only for stress testing already-running server")
@pytest.mark.parametrize("repeat", list(range(0, 100)))
@wrap_test_forked
def test_client_chat_stream_langchain_fake_embeddings_stress_no_llm(repeat):
    data_kind = 'helium3'
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'  # presumes remote server is llama-2 chat based
    local_server = False
    chat = False
    inference_server = None
    return run_client_chat_stream_langchain_fake_embeddings(data_kind, base_model, local_server, inference_server,
                                                            chat=chat)


def go_upload_gradio():
    import gradio as gr
    import time

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])
        with gr.Accordion("Upload", open=False, visible=True):
            with gr.Column():
                with gr.Row(equal_height=False):
                    file = gr.File(show_label=False,
                                   file_count="multiple",
                                   scale=1,
                                   min_width=0,
                                   )

        def respond(message, chat_history):
            if not chat_history:
                chat_history = [[message, '']]
            chat_history[-1][1] = message
            for fake in range(0, 1000):
                chat_history[-1][1] += str(fake)
                time.sleep(0.1)
                yield "", chat_history
            return

        def gofile(x):
            print(x)
            return x

        user_text_text = gr.Textbox(label='Paste Text',
                                    interactive=True,
                                    visible=True)

        msg.submit(respond, [msg, chatbot], [msg, chatbot])

        def show_text(x):
            return str(x)

        user_text_text.submit(fn=show_text, inputs=user_text_text, outputs=user_text_text, api_name='add_text')

        eventdb1 = file.upload(gofile, file, api_name='file')

    if __name__ == "__main__":
        demo.queue(concurrency_count=64)
        demo.launch(server_name='0.0.0.0')


# NOTE: llama-7b on 24GB will go OOM for helium1/2 tests
@pytest.mark.parametrize("inference_server", [None, 'openai_chat', 'openai_azure_chat', 'replicate'])
# local_server=True
@pytest.mark.parametrize("base_model",
                         ['h2oai/h2ogpt-oig-oasst1-512-6_9b', 'h2oai/h2ogpt-4096-llama2-7b-chat', 'gpt-3.5-turbo'])
# local_server=False or True if inference_server used
# @pytest.mark.parametrize("base_model", ['h2oai/h2ogpt-4096-llama2-70b-chat'])
@pytest.mark.parametrize("data_kind", [
    'simple',
    'helium1',
    'helium2',
    'helium3',
    'helium4',
    'helium5',
])
@wrap_test_forked
def test_client_chat_stream_langchain_fake_embeddings(data_kind, base_model, inference_server):
    # local_server = False  # set to False to test local server, e.g. gradio connected to TGI server
    local_server = True  # for gradio connected to TGI, or if pass inference_server too then some remote vLLM/TGI using local server
    return run_client_chat_stream_langchain_fake_embeddings(data_kind, base_model, local_server, inference_server)


texts_simple = ['first', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'last']

texts_helium1 = [
    '464 $ \n453 \n$ \n97 \n$ 125 $ 131 \n$ \n96 \n$ 89 $ \n84 \n$ 2,417 \n$ 2,291 $ 2,260 \nAverage loans\n291 \n287 \n298 \n321 \n307 \n304 \n41 \n74 \n83 \n— \n— \n— \n653 \n668 \n685 \nAverage deposits\n830 \n828 \n780 \n435 \n417 \n358 \n52 \n82 \n81 \n16 \n8 \n11 \n1,333 \n1,335 1,230 \n(1) \nIncludes total Citi revenues, net of interest expense (excluding \nCorporate/Other\n), in North America of $34.4 billion, $34.4 billion and $37.1 billion; in EMEA of',
    'Legacy Franchises\nCorporate/Other\nTotal Citi\n2022\n2021\n2020\n2022\n2021\n2020\n2022\n2021\n2020\n2022\n2021\n2020\n2022\n2021\n2020\nIn millions of \ndollars, except \nidentifiable assets, \naverage loans and \naverage deposits in \nbillions\nNet interest \nincome\n$ 17,911 \n$ 14,999 $ 15,750 \n$ 22,656 \n$ 20,646 $ 22,326 \n$ 5,691 \n$ 6,250 $ 6,973 \n$ 2,410 \n$ 599 $ (298) \n$ 48,668 \n$ 42,494 $ 44,751 \nNon-interest \nrevenue\n23,295 \n24,837 25,343 \n1,561 \n2,681 2,814 \n2,781 \n2,001 2,481 \n(967) \n(129) \n112 \n26,670 \n29,390 30,750',
    'Personal Banking and Wealth Management\n24,217 \n23,327 \n25,140 \n4 \n(7) \nLegacy Franchises\n8,472 \n8,251 \n9,454 \n3 \n(13) \nCorporate/Other\n1,443 \n470 \n(186) \nNM\nNM\nTotal Citigroup net revenues\n$ \n75,338 \n$ \n71,884 $ \n75,501 \n5 %\n(5) %\nNM Not meaningful\nINCOME\n% Change\n% Change\n2022 vs. 2021\n2021 vs. 2020\nIn millions of dollars\n2022\n2021\n2020\nIncome (loss) from continuing operations\nInstitutional Clients Group\n$ \n10,738 \n$ \n14,308 $ \n10,811 \n(25) %\n32 %\nPersonal Banking and Wealth Management\n3,319 \n7,734 \n1,322',
    '(2)\n307 \n(140) \n(59) \nNM\nNM\nTotal Banking revenues (including gains (losses) on loan \nhedges)\n(2)\n$ \n6,071 \n$ \n9,378 $ \n7,233 \n(35) %\n30 %\nTotal \nICG\nrevenues, net of interest expense\n$ \n41,206 \n$ \n39,836 $ \n41,093 \n3 %\n(3) %\n(1) \nCiti assesses its Markets business performance on a total revenue basis, as offsets may occur across revenue line items. For example, securities that generate \nNet \ninterest income\nmay be risk managed by derivatives that are recorded in \nPrincipal transactions\nrevenue within',
    'higher revenues. Citigroup’s effective tax rate was 19.4% in \nthe current year versus 19.8% in the prior year. Earnings per \nshare (EPS) decreased 31%, reflecting the decrease in net \nincome, partially offset by a 4% decline in average diluted \nshares outstanding.\nAs discussed above, results for 2022 included divestiture-\n•\nCiti’s revenues increased 5% versus the prior year, \nincluding net gains on sales of Citi’s Philippines and \nThailand consumer banking businesses versus a loss on',
    'Citigroup reported net income of $14.8 billion, or $7.00 per \nshare, compared to net income of $22.0 billion, or $10.14 per \nshare in the prior year. The decrease in net income was \nprimarily driven by the higher cost of credit, resulting from \nloan growth in \nPersonal Banking and Wealth Management \n(PBWM)\nand a deterioration in macroeconomic assumptions, \n3\nPolicies and Significant Estimates—Citi’s Allowance for \nCredit Losses (ACL)” below.\nNet credit losses of $3.8 billion decreased 23% from the',
    'The Company’s operating leases, where Citi is a lessor, \nCommercial and industrial\n$ \n56,176 \n$ \n48,364 \nare not significant to the Consolidated Financial Statements.\nFinancial institutions\n43,399 \n49,804 \nMortgage and real estate\n(2)\n17,829 \n15,965 \nInstallment and other\n23,767 \n20,143 \nLease financing\n308 \n415 \nTotal\n$ \n141,479 \n$ \n134,691 \nIn offices outside North America\n(1)\nCommercial and industrial\n$ \n93,967 \n$ \n102,735 \nFinancial institutions\n21,931 \n22,158 \nMortgage and real estate\n(2)\n4,179 \n4,374',
    '$1.8 billion in assets, including $1.2 billion of loans (net of allowance of $80 million) and excluding goodwill. The total amount of liabilities was $1.3 billion, \nincluding $1.2 billion in deposits. The sale resulted in a pretax gain on sale of approximately $618 million ($290 million after-tax), subject to closing adjustments, \nrecorded in \nOther revenue\n. The income before taxes shown in the above table for the Philippines reflects Citi’s ownership through August 1, 2022.\n(4)',
    'net interest income—taxable equivalent basis\n(1)\n$ \n43,660 \n$ \n37,519 \n$ \n39,739 \n(1) \nInterest revenue\nand \nNet interest income\ninclude the taxable equivalent adjustments discussed in the table above.\nCiti’s net interest income in the fourth quarter of 2022 was \n$13.3 billion (also $13.3 billion on a taxable equivalent basis), \nan increase of $2.5 billion versus the prior year, primarily \ndriven by non-\nICG\nMarkets (approximately $2.2 billion), as \nICG\nMarkets was largely unchanged (up approximately $0.3',
    'Corporate/Other\nin 2022, see “\nCorporate/Other\n” below.\n7% versus the prior year. Branded cards revenues of $8.9 \nbillion increased 9%, driven by higher net interest income. In \nBranded cards, new account acquisitions increased 11%, card \nspend volumes increased 16% and average loans increased \n11%. Retail services revenues of $5.5 billion increased 7%, \n5\nCITI’S CONSENT ORDER COMPLIANCE\nCiti has embarked on a multiyear transformation, with the \ntarget outcome to change Citi’s business and operating models',
    '$ (38,765) \n$ (32,058) $ (36,318) \nCitigroup’s total other comprehensive income (loss)\n(8,297) \n(6,707) \n4,260 \nBalance, end of year\n$ (47,062) \n$ (38,765) $ (32,058) \nTotal Citigroup common stockholders’ equity\n$ 182,194 \n$ 182,977 $ 179,962 \n1,936,986 \n1,984,355 2,082,089 \nTotal Citigroup stockholders’ equity\n$ 201,189 \n$ 201,972 $ 199,442 \nNoncontrolling interests\nBalance, beginning of year\n$ \n700 \n$ \n758 $ \n704 \nTransactions between Citigroup and the noncontrolling-interest \nshareholders\n(34) \n(10)',
    'CONSOLIDATED STATEMENT OF COMPREHENSIVE INCOME\nCitigroup Inc. and Subsidiaries\nYears ended December 31,\nIn millions of dollars\n2022\n2021\n2020\nCitigroup’s net income\n$ \n14,845 \n$ \n21,952 $ \n11,047 \nAdd: Citigroup’s other comprehensive income (loss)\n(1)\nNet change in unrealized gains and losses on debt securities, net of taxes\n(2)\n$ \n(5,384) \n$ \n(3,934) $ \n3,585 \nNet change in debt valuation adjustment (DVA), net of taxes\n(3)\n2,029 \n232 \n(475) \nNet change in cash flow hedges, net of taxes\n(2,623) \n(1,492)',
    'Efficiency ratio (total operating expenses/total revenues, net)\n68.1 \n67.0 \n58.8 \n57.0 \n58.1 \nBasel III ratios\nCET1 Capital\n(4)\n13.03 %\n12.25 %\n11.51 %\n11.79 %\n11.86 %\nTier 1 Capital\n(4)\n14.80 \n13.91 \n13.06 \n13.33 \n13.43 \nTotal Capital\n(4)\n15.46 \n16.04 \n15.33 \n15.87 \n16.14 \nSupplementary Leverage ratio\n5.82 \n5.73 \n6.99 \n6.20 \n6.40 \nCitigroup common stockholders’ equity to assets\n7.54 %\n7.99 %\n7.96 %\n8.98 %\n9.27 %\nTotal Citigroup stockholders’ equity to assets\n8.33 \n8.81 \n8.82 \n9.90 \n10.23',
    'to contractually based performance thresholds that, if met, \nwould require Citi to make ongoing payments to the partner. \nThe threshold is based on the profitability of a program and is \ngenerally calculated based on predefined program revenues \n166\nThe following table presents \nCommissions and fees\nrevenue:\n2022\n2021\n2020\nIn millions of \ndollars\nICG\nPBWM\nLF\nTotal\nICG\nPBWM\nLF\nTotal\nICG\nPBWM\nLF\nTotal\nInvestment \nbanking\n$ 3,084 $ \n— $ \n— $ 3,084 \n$ 6,007 $ \n— $ \n— $ 6,007 $ 4,483 $ \n— $ \n— $ 4,483',
    '$742 billion and $684 billion; in Latin America of $184 billion, $179 billion and $180 billion; and in Asia of $588 billion, $572 billion and $572 billion in 2022, \n2021 and 2020, respectively. These regional numbers exclude \nCorporate/Other\n, which largely reflects U.S. activities. The Company’s long-lived assets for the \nperiods presented are not considered to be significant in relation to its total assets. The majority of Citi’s long-lived assets are located in the U.S.\n164',
    '32,517 \n58,170 \nMortgage-backed securities\n33,573 \n— \n33,573 \nAsset-backed securities\n1,681 \n— \n1,681 \nOther\n4,026 \n58 \n4,084 \nTotal\n$ \n305,597 $ \n33,029 $ \n338,626 \n193\n12. BROKERAGE RECEIVABLES AND BROKERAGE \nPAYABLES\nThe Company has receivables and payables for financial \ninstruments sold to and purchased from brokers, dealers and \ncustomers, which arise in the ordinary course of business. Citi \nis exposed to risk of loss from the inability of brokers, dealers',
    'Payables to customers\n$ \n55,747 \n$ \n52,158 \nPayables to brokers, dealers and \nclearing organizations\n13,471 \n9,272 \nTotal brokerage payables\n(1)\n$ \n69,218 \n$ \n61,430 \n(1) Includes brokerage receivables and payables recorded by Citi broker-\ndealer entities that are accounted for in accordance with the AICPA \nAccounting Guide for Brokers and Dealers in Securities as codified in \nASC 940-320.\n194\n13. INVESTMENTS\nThe following table presents Citi’s investments by category:\nDecember 31,\nIn millions of dollars',
    'investment banking fees generated across the industry (i.e., the \nrevenue wallet) from investment banking transactions in \nM&A, equity and debt underwriting, and loan syndications.\n326\nNotes\n327\nNotes\n328\nNotes\n329\nNotes\n330\nNotes\n331\nNotes\n332\nNotes\n333\nStockholder information\nExchange agent\nCitigroup common stock is listed on the NYSE under the \nticker symbol “C.” Citigroup preferred stock Series J and K \nare also listed on the NYSE.\nHolders of Golden State Bancorp, Associates First Capital',
    'Non-U.S. pretax earnings approximated $16.2 billion in 2022, \n$12.9 billion in 2021 and $13.8 billion in 2020. As a U.S. \ncorporation, Citigroup and its U.S. subsidiaries are currently \nsubject to U.S. taxation on all non-U.S. pretax earnings of \nnon-U.S. branches. Beginning in 2018, there is a separate \nforeign tax credit (FTC) basket for branches. Also, dividends \nfrom a non-U.S. subsidiary or affiliate are effectively exempt \nfrom U.S. taxation. The Company provides income taxes on',
    'Total comprehensive income\n$ \n15,307 $ \n3,050 $ \n13,286 $ \n(16,270) $ \n15,373 \n308\nCondensed Consolidating Balance Sheet\nDecember 31, 2022\nOther \nCitigroup \nCitigroup \nsubsidiaries \nparent \nand \nCitigroup \ncompany\nCGMHI\neliminations\nConsolidating \nadjustments\nconsolidated\nIn millions of dollars\nAssets\nCash and due from banks\n$ \n— $ \n955 $ \n29,622 $ \n— $ \n30,577 \nCash and due from banks—intercompany\n15 \n7,448 \n(7,463) \n— \n— \nDeposits with banks, net of allowance\n— \n7,902 \n303,546 \n— \n311,448',
    '817 $ \n852 \nIn billions of dollars\n4Q22\n3Q22\n4Q21\nLegacy Franchises\n(1)\n$ \n50 \n$ \n50 $ \n74 \nCorporate/Other\n$ \n32 \n$ \n21 $ \n7 \nPersonal Banking and Wealth \nManagement\nU.S. Retail banking\n$ \n37 \n$ \n36 $ \n34 \nTotal Citigroup deposits (AVG)\n$ 1,361 \n$ 1,316 $ 1,370 \nU.S. Cards\n143 \n138 \n128 \nTotal Citigroup deposits (EOP)\n$ 1,366 \n$ 1,306 $ 1,317 \nGlobal Wealth\n150 \n151 \n150 \nTotal\n$ \n330 \n$ \n325 $ \n312 \n(1)\nSee footnote 2 to the table in “Credit Risk—Consumer Credit—\nConsumer Credit Portfolio” above.',
    'Citigroup Inc. and Consolidated Subsidiaries\nIn millions of dollars, except per share amounts, ratios and direct staff\n2022\n2021\n2020\n2019\n2018\nAt December 31:\nTotal assets\n$ 2,416,676 \n$ 2,291,413 \n$ 2,260,090 \n$ 1,951,158 \n$ 1,917,383 \nTotal deposits \n1,365,954 \n1,317,230 \n1,280,671 \n1,070,590 \n1,013,170 \nLong-term debt\n271,606 \n254,374 \n271,686 \n248,760 \n231,999 \nCitigroup common stockholders’ equity\n182,194 \n182,977 \n179,962 \n175,262 \n177,760 \nTotal Citigroup stockholders’ equity\n201,189 \n201,972',
    'Net income from continuing operations (for EPS purposes)\n$ \n15,076 \n$ \n21,945 $ \n11,067 \nLoss from discontinued operations, net of taxes\n(231) \n7 \n(20) \nCitigroup’s net income\n$ \n14,845 \n$ \n21,952 $ \n11,047 \nLess: Preferred dividends\n(1)\n1,032 \n1,040 \n1,095 \nNet income available to common shareholders\n$ \n13,813 \n$ \n20,912 $ \n9,952 \nLess: Dividends and undistributed earnings allocated to employee restricted and deferred shares \nwith rights to dividends, applicable to basic EPS\n113 \n154 \n73',
    'During 2022, emerging markets revenues accounted for \napproximately 37% of Citi’s total revenues (Citi generally \ndefines emerging markets as countries in Latin America, Asia \n(other than Japan, Australia and New Zealand), and central \nand Eastern Europe, the Middle East and Africa in EMEA). \nCiti’s presence in the emerging markets subjects it to various \nrisks, such as limitations or unavailability of hedges on foreign \ninvestments; foreign currency volatility, including',
    'On November 1, 2022, Citi completed the sale of its Thailand consumer banking business, which was part of \nLegacy Franchises\n. The business had approximately \n$2.7 billion in assets, including $2.4 billion of loans (net of allowance of $67 million) and excluding goodwill. The total amount of liabilities was $1.0 billion, \nincluding $0.8 billion in deposits. The sale resulted in a pretax gain on sale of approximately $209 million ($115 million after-tax), subject to closing adjustments, \nrecorded in']

texts_helium2 = [
    'Efficiency ratio (total operating expenses/total revenues, net)\n68.1\n67.0\n58.8\n57.0\n58.1\nBasel III ratios\nCET1 Capital\n(4)\n13.03 %\n12.25 %\n11.51 %\n11.79 %\n11.86 %\nTier 1 Capital\n(4)\n14.80\n13.91\n13.06\n13.33\n13.43\nTotal Capital\n(4)\n15.46\n16.04\n15.33\n15.87\n16.14\nSupplementary Leverage ratio\n5.82\n5.73\n6.99\n6.20\n6.40\nCitigroup common stockholders’ equity to assets\n7.54 %\n7.99 %\n7.96 %\n8.98 %\n9.27 %\nTotal Citigroup stockholders’ equity to assets\n8.33\n8.81\n8.82\n9.90\n10.23',
    'Payables to customers\n$\n55,747\n$\n52,158\nPayables to brokers, dealers and\nclearing organizations\n13,471\n9,272\nTotal brokerage payables\n(1)\n$\n69,218\n$\n61,430\n(1) Includes brokerage receivables and payables recorded by Citi broker-\ndealer entities that are accounted for in accordance with the AICPA\nAccounting Guide for Brokers and Dealers in Securities as codified in\nASC 940-320.\n194\n13. INVESTMENTS\nThe following table presents Citi’s investments by category:\nDecember 31,\nIn millions of dollars',
    'Payables to customers\n$\n55,747\n$\n52,158\nPayables to brokers, dealers and\nclearing organizations\n13,471\n9,272\nTotal brokerage payables\n(1)\n$\n69,218\n$\n61,430\n(1) Includes brokerage receivables and payables recorded by Citi broker-\ndealer entities that are accounted for in accordance with the AICPA\nAccounting Guide for Brokers and Dealers in Securities as codified in\nASC 940-320.\n194\n13. INVESTMENTS\nThe following table presents Citi’s investments by category:\nDecember 31,\nIn millions of dollars',
    'Corporate/Other\nin 2022, see “\nCorporate/Other\n” below.\n7% versus the prior year. Branded cards revenues of $8.9\nbillion increased 9%, driven by higher net interest income. In\nBranded cards, new account acquisitions increased 11%, card\nspend volumes increased 16% and average loans increased\n11%. Retail services revenues of $5.5 billion increased 7%,\n5\nCITI’S CONSENT ORDER COMPLIANCE\nCiti has embarked on a multiyear transformation, with the\ntarget outcome to change Citi’s business and operating models',
    'Corporate/Other\nin 2022, see “\nCorporate/Other\n” below.\n7% versus the prior year. Branded cards revenues of $8.9\nbillion increased 9%, driven by higher net interest income. In\nBranded cards, new account acquisitions increased 11%, card\nspend volumes increased 16% and average loans increased\n11%. Retail services revenues of $5.5 billion increased 7%,\n5\nCITI’S CONSENT ORDER COMPLIANCE\nCiti has embarked on a multiyear transformation, with the\ntarget outcome to change Citi’s business and operating models',
    'Citigroup Inc. and Consolidated Subsidiaries\nIn millions of dollars, except per share amounts, ratios and direct staff\n2022\n2021\n2020\n2019\n2018\nAt December 31:\nTotal assets\n$ 2,416,676\n$ 2,291,413\n$ 2,260,090\n$ 1,951,158\n$ 1,917,383\nTotal deposits\n1,365,954\n1,317,230\n1,280,671\n1,070,590\n1,013,170\nLong-term debt\n271,606\n254,374\n271,686\n248,760\n231,999\nCitigroup common stockholders’ equity\n182,194\n182,977\n179,962\n175,262\n177,760\nTotal Citigroup stockholders’ equity\n201,189\n201,972',
    'Citigroup Inc. and Consolidated Subsidiaries\nIn millions of dollars, except per share amounts, ratios and direct staff\n2022\n2021\n2020\n2019\n2018\nAt December 31:\nTotal assets\n$ 2,416,676\n$ 2,291,413\n$ 2,260,090\n$ 1,951,158\n$ 1,917,383\nTotal deposits\n1,365,954\n1,317,230\n1,280,671\n1,070,590\n1,013,170\nLong-term debt\n271,606\n254,374\n271,686\n248,760\n231,999\nCitigroup common stockholders’ equity\n182,194\n182,977\n179,962\n175,262\n177,760\nTotal Citigroup stockholders’ equity\n201,189\n201,972',
    '32,517\n58,170\nMortgage-backed securities\n33,573\n—\n33,573\nAsset-backed securities\n1,681\n—\n1,681\nOther\n4,026\n58\n4,084\nTotal\n$\n305,597 $\n33,029 $\n338,626\n193\n12. BROKERAGE RECEIVABLES AND BROKERAGE\nPAYABLES\nThe Company has receivables and payables for financial\ninstruments sold to and purchased from brokers, dealers and\ncustomers, which arise in the ordinary course of business. Citi\nis exposed to risk of loss from the inability of brokers, dealers',
    '32,517\n58,170\nMortgage-backed securities\n33,573\n—\n33,573\nAsset-backed securities\n1,681\n—\n1,681\nOther\n4,026\n58\n4,084\nTotal\n$\n305,597 $\n33,029 $\n338,626\n193\n12. BROKERAGE RECEIVABLES AND BROKERAGE\nPAYABLES\nThe Company has receivables and payables for financial\ninstruments sold to and purchased from brokers, dealers and\ncustomers, which arise in the ordinary course of business. Citi\nis exposed to risk of loss from the inability of brokers, dealers',
    'Total comprehensive income\n$\n15,307 $\n3,050 $\n13,286 $\n(16,270) $\n15,373\n308\nCondensed Consolidating Balance Sheet\nDecember 31, 2022\nOther\nCitigroup\nCitigroup\nsubsidiaries\nparent\nand\nCitigroup\ncompany\nCGMHI\neliminations\nConsolidating\nadjustments\nconsolidated\nIn millions of dollars\nAssets\nCash and due from banks\n$\n— $\n955 $\n29,622 $\n— $\n30,577\nCash and due from banks—intercompany\n15\n7,448\n(7,463)\n—\n—\nDeposits with banks, net of allowance\n—\n7,902\n303,546\n—\n311,448',
    'Total comprehensive income\n$\n15,307 $\n3,050 $\n13,286 $\n(16,270) $\n15,373\n308\nCondensed Consolidating Balance Sheet\nDecember 31, 2022\nOther\nCitigroup\nCitigroup\nsubsidiaries\nparent\nand\nCitigroup\ncompany\nCGMHI\neliminations\nConsolidating\nadjustments\nconsolidated\nIn millions of dollars\nAssets\nCash and due from banks\n$\n— $\n955 $\n29,622 $\n— $\n30,577\nCash and due from banks—intercompany\n15\n7,448\n(7,463)\n—\n—\nDeposits with banks, net of allowance\n—\n7,902\n303,546\n—\n311,448',
    'its right as a clearing member to transform cash margin into\nother assets, (iii) Citi does not guarantee and is not liable to\nthe client for the performance of the CCP or the depository\ninstitution and (iv) the client cash balances are legally isolated\nfrom Citi’s bankruptcy estate. The total amount of cash initial\nmargin collected and remitted in this manner was\napproximately $18.0 billion and $18.7 billion as of\nDecember 31, 2022 and 2021, respectively.',
    'its right as a clearing member to transform cash margin into\nother assets, (iii) Citi does not guarantee and is not liable to\nthe client for the performance of the CCP or the depository\ninstitution and (iv) the client cash balances are legally isolated\nfrom Citi’s bankruptcy estate. The total amount of cash initial\nmargin collected and remitted in this manner was\napproximately $18.0 billion and $18.7 billion as of\nDecember 31, 2022 and 2021, respectively.',
    '817 $\n852\nIn billions of dollars\n4Q22\n3Q22\n4Q21\nLegacy Franchises\n(1)\n$\n50\n$\n50 $\n74\nCorporate/Other\n$\n32\n$\n21 $\n7\nPersonal Banking and Wealth\nManagement\nU.S. Retail banking\n$\n37\n$\n36 $\n34\nTotal Citigroup deposits (AVG)\n$ 1,361\n$ 1,316 $ 1,370\nU.S. Cards\n143\n138\n128\nTotal Citigroup deposits (EOP)\n$ 1,366\n$ 1,306 $ 1,317\nGlobal Wealth\n150\n151\n150\nTotal\n$\n330\n$\n325 $\n312\n(1)\nSee footnote 2 to the table in “Credit Risk—Consumer Credit—\nConsumer Credit Portfolio” above.',
    '$14.9 billion, $13.4 billion and $13.4 billion; in Latin America of $9.9 billion, $9.2 billion and $9.4 billion; and in Asia of $14.7 billion, $14.4 billion and\n$15.8 billion in 2022, 2021 and 2020, respectively. These regional numbers exclude\nCorporate/Other\n, which largely reflects U.S. activities.\n(2)\nIncludes total Citi identifiable assets (excluding\nCorporate/Other\n), in North America of $776 billion, $709 billion and $741 billion; in EMEA of $773 billion,',
    'Revenues, net of interest expense\n$\n75,338\n$\n71,884 $\n75,501 $\n75,067 $\n74,036\nOperating expenses\n51,292\n48,193\n44,374\n42,783\n43,023\nProvisions for credit losses and for benefits and claims\n5,239\n(3,778)\n17,495\n8,383\n7,568\nIncome from continuing operations before income taxes\n$\n18,807\n$\n27,469 $\n13,632 $\n23,901 $\n23,445\nIncome taxes\n3,642\n5,451\n2,525\n4,430\n5,357\nIncome from continuing operations\n$\n15,165\n$\n22,018 $\n11,107 $\n19,471 $\n18,088',
    'Revenues, net of interest expense\n$\n75,338\n$\n71,884 $\n75,501 $\n75,067 $\n74,036\nOperating expenses\n51,292\n48,193\n44,374\n42,783\n43,023\nProvisions for credit losses and for benefits and claims\n5,239\n(3,778)\n17,495\n8,383\n7,568\nIncome from continuing operations before income taxes\n$\n18,807\n$\n27,469 $\n13,632 $\n23,901 $\n23,445\nIncome taxes\n3,642\n5,451\n2,525\n4,430\n5,357\nIncome from continuing operations\n$\n15,165\n$\n22,018 $\n11,107 $\n19,471 $\n18,088',
    'approximately $400 million ($345 million after-tax) related to\nare inherently limited because they involve techniques,\nincluding the use of historical data in many circumstances,\nassumptions and judgments that cannot anticipate every\neconomic and financial outcome in the markets in which Citi\noperates, nor can they anticipate the specifics and timing of\n49\ninterconnectedness among financial institutions, concerns\nabout the creditworthiness of or defaults by a financial',
    'approximately $400 million ($345 million after-tax) related to\nare inherently limited because they involve techniques,\nincluding the use of historical data in many circumstances,\nassumptions and judgments that cannot anticipate every\neconomic and financial outcome in the markets in which Citi\noperates, nor can they anticipate the specifics and timing of\n49\ninterconnectedness among financial institutions, concerns\nabout the creditworthiness of or defaults by a financial',
    'to contractually based performance thresholds that, if met,\nwould require Citi to make ongoing payments to the partner.\nThe threshold is based on the profitability of a program and is\ngenerally calculated based on predefined program revenues\n166\nThe following table presents\nCommissions and fees\nrevenue:\n2022\n2021\n2020\nIn millions of\ndollars\nICG\nPBWM\nLF\nTotal\nICG\nPBWM\nLF\nTotal\nICG\nPBWM\nLF\nTotal\nInvestment\nbanking\n$ 3,084 $\n— $\n— $ 3,084\n$ 6,007 $\n— $\n— $ 6,007 $ 4,483 $\n— $\n— $ 4,483',
    'to contractually based performance thresholds that, if met,\nwould require Citi to make ongoing payments to the partner.\nThe threshold is based on the profitability of a program and is\ngenerally calculated based on predefined program revenues\n166\nThe following table presents\nCommissions and fees\nrevenue:\n2022\n2021\n2020\nIn millions of\ndollars\nICG\nPBWM\nLF\nTotal\nICG\nPBWM\nLF\nTotal\nICG\nPBWM\nLF\nTotal\nInvestment\nbanking\n$ 3,084 $\n— $\n— $ 3,084\n$ 6,007 $\n— $\n— $ 6,007 $ 4,483 $\n— $\n— $ 4,483',
    'On November 1, 2022, Citi completed the sale of its Thailand consumer banking business, which was part of\nLegacy Franchises\n. The business had approximately\n$2.7 billion in assets, including $2.4 billion of loans (net of allowance of $67 million) and excluding goodwill. The total amount of liabilities was $1.0 billion,\nincluding $0.8 billion in deposits. The sale resulted in a pretax gain on sale of approximately $209 million ($115 million after-tax), subject to closing adjustments,\nrecorded in',
    'On November 1, 2022, Citi completed the sale of its Thailand consumer banking business, which was part of\nLegacy Franchises\n. The business had approximately\n$2.7 billion in assets, including $2.4 billion of loans (net of allowance of $67 million) and excluding goodwill. The total amount of liabilities was $1.0 billion,\nincluding $0.8 billion in deposits. The sale resulted in a pretax gain on sale of approximately $209 million ($115 million after-tax), subject to closing adjustments,\nrecorded in',
    'Efficiency ratio (total operating expenses/total revenues, net)\n68.1\n67.0\n58.8\n57.0\n58.1\nBasel III ratios\nCET1 Capital\n(4)\n13.03 %\n12.25 %\n11.51 %\n11.79 %\n11.86 %\nTier 1 Capital\n(4)\n14.80\n13.91\n13.06\n13.33\n13.43\nTotal Capital\n(4)\n15.46\n16.04\n15.33\n15.87\n16.14\nSupplementary Leverage ratio\n5.82\n5.73\n6.99\n6.20\n6.40\nCitigroup common stockholders’ equity to assets\n7.54 %\n7.99 %\n7.96 %\n8.98 %\n9.27 %\nTotal Citigroup stockholders’ equity to assets\n8.33\n8.81\n8.82\n9.90\n10.23',
    'The Company’s operating leases, where Citi is a lessor,\nCommercial and industrial\n$\n56,176\n$\n48,364\nare not significant to the Consolidated Financial Statements.\nFinancial institutions\n43,399\n49,804\nMortgage and real estate\n(2)\n17,829\n15,965\nInstallment and other\n23,767\n20,143\nLease financing\n308\n415\nTotal\n$\n141,479\n$\n134,691\nIn offices outside North America\n(1)\nCommercial and industrial\n$\n93,967\n$\n102,735\nFinancial institutions\n21,931\n22,158\nMortgage and real estate\n(2)\n4,179\n4,374',
    '464 $\n453\n$\n97\n$ 125 $ 131\n$\n96\n$ 89 $\n84\n$ 2,417\n$ 2,291 $ 2,260\nAverage loans\n291\n287\n298\n321\n307\n304\n41\n74\n83\n—\n—\n—\n653\n668\n685\nAverage deposits\n830\n828\n780\n435\n417\n358\n52\n82\n81\n16\n8\n11\n1,333\n1,335 1,230\n(1)\nIncludes total Citi revenues, net of interest expense (excluding\nCorporate/Other\n), in North America of $34.4 billion, $34.4 billion and $37.1 billion; in EMEA of',
    '$14.9 billion, $13.4 billion and $13.4 billion; in Latin America of $9.9 billion, $9.2 billion and $9.4 billion; and in Asia of $14.7 billion, $14.4 billion and\n$15.8 billion in 2022, 2021 and 2020, respectively. These regional numbers exclude\nCorporate/Other\n, which largely reflects U.S. activities.\n(2)\nIncludes total Citi identifiable assets (excluding\nCorporate/Other\n), in North America of $776 billion, $709 billion and $741 billion; in EMEA of $773 billion,',
    'Legacy Franchises\nCorporate/Other\nTotal Citi\n2022\n2021\n2020\n2022\n2021\n2020\n2022\n2021\n2020\n2022\n2021\n2020\n2022\n2021\n2020\nIn millions of\ndollars, except\nidentifiable assets,\naverage loans and\naverage deposits in\nbillions\nNet interest\nincome\n$ 17,911\n$ 14,999 $ 15,750\n$ 22,656\n$ 20,646 $ 22,326\n$ 5,691\n$ 6,250 $ 6,973\n$ 2,410\n$ 599 $ (298)\n$ 48,668\n$ 42,494 $ 44,751\nNon-interest\nrevenue\n23,295\n24,837 25,343\n1,561\n2,681 2,814\n2,781\n2,001 2,481\n(967)\n(129)\n112\n26,670\n29,390 30,750',
    'Legacy Franchises\nCorporate/Other\nTotal Citi\n2022\n2021\n2020\n2022\n2021\n2020\n2022\n2021\n2020\n2022\n2021\n2020\n2022\n2021\n2020\nIn millions of\ndollars, except\nidentifiable assets,\naverage loans and\naverage deposits in\nbillions\nNet interest\nincome\n$ 17,911\n$ 14,999 $ 15,750\n$ 22,656\n$ 20,646 $ 22,326\n$ 5,691\n$ 6,250 $ 6,973\n$ 2,410\n$ 599 $ (298)\n$ 48,668\n$ 42,494 $ 44,751\nNon-interest\nrevenue\n23,295\n24,837 25,343\n1,561\n2,681 2,814\n2,781\n2,001 2,481\n(967)\n(129)\n112\n26,670\n29,390 30,750',
    'Personal Banking and Wealth Management\n24,217\n23,327\n25,140\n4\n(7)\nLegacy Franchises\n8,472\n8,251\n9,454\n3\n(13)\nCorporate/Other\n1,443\n470\n(186)\nNM\nNM\nTotal Citigroup net revenues\n$\n75,338\n$\n71,884 $\n75,501\n5 %\n(5) %\nNM Not meaningful\nINCOME\n% Change\n% Change\n2022 vs. 2021\n2021 vs. 2020\nIn millions of dollars\n2022\n2021\n2020\nIncome (loss) from continuing operations\nInstitutional Clients Group\n$\n10,738\n$\n14,308 $\n10,811\n(25) %\n32 %\nPersonal Banking and Wealth Management\n3,319\n7,734\n1,322',
    'Personal Banking and Wealth Management\n24,217\n23,327\n25,140\n4\n(7)\nLegacy Franchises\n8,472\n8,251\n9,454\n3\n(13)\nCorporate/Other\n1,443\n470\n(186)\nNM\nNM\nTotal Citigroup net revenues\n$\n75,338\n$\n71,884 $\n75,501\n5 %\n(5) %\nNM Not meaningful\nINCOME\n% Change\n% Change\n2022 vs. 2021\n2021 vs. 2020\nIn millions of dollars\n2022\n2021\n2020\nIncome (loss) from continuing operations\nInstitutional Clients Group\n$\n10,738\n$\n14,308 $\n10,811\n(25) %\n32 %\nPersonal Banking and Wealth Management\n3,319\n7,734\n1,322',
    '(2)\n307\n(140)\n(59)\nNM\nNM\nTotal Banking revenues (including gains (losses) on loan\nhedges)\n(2)\n$\n6,071\n$\n9,378 $\n7,233\n(35) %\n30 %\nTotal\nICG\nrevenues, net of interest expense\n$\n41,206\n$\n39,836 $\n41,093\n3 %\n(3) %\n(1)\nCiti assesses its Markets business performance on a total revenue basis, as offsets may occur across revenue line items. For example, securities that generate\nNet\ninterest income\nmay be risk managed by derivatives that are recorded in\nPrincipal transactions\nrevenue within',
    '(2)\n307\n(140)\n(59)\nNM\nNM\nTotal Banking revenues (including gains (losses) on loan\nhedges)\n(2)\n$\n6,071\n$\n9,378 $\n7,233\n(35) %\n30 %\nTotal\nICG\nrevenues, net of interest expense\n$\n41,206\n$\n39,836 $\n41,093\n3 %\n(3) %\n(1)\nCiti assesses its Markets business performance on a total revenue basis, as offsets may occur across revenue line items. For example, securities that generate\nNet\ninterest income\nmay be risk managed by derivatives that are recorded in\nPrincipal transactions\nrevenue within',
    '$1.8 billion in assets, including $1.2 billion of loans (net of allowance of $80 million) and excluding goodwill. The total amount of liabilities was $1.3 billion,\nincluding $1.2 billion in deposits. The sale resulted in a pretax gain on sale of approximately $618 million ($290 million after-tax), subject to closing adjustments,\nrecorded in\nOther revenue\n. The income before taxes shown in the above table for the Philippines reflects Citi’s ownership through August 1, 2022.\n(4)',
    '$1.8 billion in assets, including $1.2 billion of loans (net of allowance of $80 million) and excluding goodwill. The total amount of liabilities was $1.3 billion,\nincluding $1.2 billion in deposits. The sale resulted in a pretax gain on sale of approximately $618 million ($290 million after-tax), subject to closing adjustments,\nrecorded in\nOther revenue\n. The income before taxes shown in the above table for the Philippines reflects Citi’s ownership through August 1, 2022.\n(4)',
    'Citigroup reported net income of $14.8 billion, or $7.00 per\nshare, compared to net income of $22.0 billion, or $10.14 per\nshare in the prior year. The decrease in net income was\nprimarily driven by the higher cost of credit, resulting from\nloan growth in\nPersonal Banking and Wealth Management\n(PBWM)\nand a deterioration in macroeconomic assumptions,\n3\nPolicies and Significant Estimates—Citi’s Allowance for\nCredit Losses (ACL)” below.\nNet credit losses of $3.8 billion decreased 23% from the',
    'Citigroup reported net income of $14.8 billion, or $7.00 per\nshare, compared to net income of $22.0 billion, or $10.14 per\nshare in the prior year. The decrease in net income was\nprimarily driven by the higher cost of credit, resulting from\nloan growth in\nPersonal Banking and Wealth Management\n(PBWM)\nand a deterioration in macroeconomic assumptions,\n3\nPolicies and Significant Estimates—Citi’s Allowance for\nCredit Losses (ACL)” below.\nNet credit losses of $3.8 billion decreased 23% from the',
    'The Company’s operating leases, where Citi is a lessor,\nCommercial and industrial\n$\n56,176\n$\n48,364\nare not significant to the Consolidated Financial Statements.\nFinancial institutions\n43,399\n49,804\nMortgage and real estate\n(2)\n17,829\n15,965\nInstallment and other\n23,767\n20,143\nLease financing\n308\n415\nTotal\n$\n141,479\n$\n134,691\nIn offices outside North America\n(1)\nCommercial and industrial\n$\n93,967\n$\n102,735\nFinancial institutions\n21,931\n22,158\nMortgage and real estate\n(2)\n4,179\n4,374',
    '464 $\n453\n$\n97\n$ 125 $ 131\n$\n96\n$ 89 $\n84\n$ 2,417\n$ 2,291 $ 2,260\nAverage loans\n291\n287\n298\n321\n307\n304\n41\n74\n83\n—\n—\n—\n653\n668\n685\nAverage deposits\n830\n828\n780\n435\n417\n358\n52\n82\n81\n16\n8\n11\n1,333\n1,335 1,230\n(1)\nIncludes total Citi revenues, net of interest expense (excluding\nCorporate/Other\n), in North America of $34.4 billion, $34.4 billion and $37.1 billion; in EMEA of',
    '$ (38,765)\n$ (32,058) $ (36,318)\nCitigroup’s total other comprehensive income (loss)\n(8,297)\n(6,707)\n4,260\nBalance, end of year\n$ (47,062)\n$ (38,765) $ (32,058)\nTotal Citigroup common stockholders’ equity\n$ 182,194\n$ 182,977 $ 179,962\n1,936,986\n1,984,355 2,082,089\nTotal Citigroup stockholders’ equity\n$ 201,189\n$ 201,972 $ 199,442\nNoncontrolling interests\nBalance, beginning of year\n$\n700\n$\n758 $\n704\nTransactions between Citigroup and the noncontrolling-interest\nshareholders\n(34)\n(10)',
    '$ (38,765)\n$ (32,058) $ (36,318)\nCitigroup’s total other comprehensive income (loss)\n(8,297)\n(6,707)\n4,260\nBalance, end of year\n$ (47,062)\n$ (38,765) $ (32,058)\nTotal Citigroup common stockholders’ equity\n$ 182,194\n$ 182,977 $ 179,962\n1,936,986\n1,984,355 2,082,089\nTotal Citigroup stockholders’ equity\n$ 201,189\n$ 201,972 $ 199,442\nNoncontrolling interests\nBalance, beginning of year\n$\n700\n$\n758 $\n704\nTransactions between Citigroup and the noncontrolling-interest\nshareholders\n(34)\n(10)',
    'net interest income—taxable equivalent basis\n(1)\n$\n43,660\n$\n37,519\n$\n39,739\n(1)\nInterest revenue\nand\nNet interest income\ninclude the taxable equivalent adjustments discussed in the table above.\nCiti’s net interest income in the fourth quarter of 2022 was\n$13.3 billion (also $13.3 billion on a taxable equivalent basis),\nan increase of $2.5 billion versus the prior year, primarily\ndriven by non-\nICG\nMarkets (approximately $2.2 billion), as\nICG\nMarkets was largely unchanged (up approximately $0.3',
    'net interest income—taxable equivalent basis\n(1)\n$\n43,660\n$\n37,519\n$\n39,739\n(1)\nInterest revenue\nand\nNet interest income\ninclude the taxable equivalent adjustments discussed in the table above.\nCiti’s net interest income in the fourth quarter of 2022 was\n$13.3 billion (also $13.3 billion on a taxable equivalent basis),\nan increase of $2.5 billion versus the prior year, primarily\ndriven by non-\nICG\nMarkets (approximately $2.2 billion), as\nICG\nMarkets was largely unchanged (up approximately $0.3',
    'higher revenues. Citigroup’s effective tax rate was 19.4% in\nthe current year versus 19.8% in the prior year. Earnings per\nshare (EPS) decreased 31%, reflecting the decrease in net\nincome, partially offset by a 4% decline in average diluted\nshares outstanding.\nAs discussed above, results for 2022 included divestiture-\n•\nCiti’s revenues increased 5% versus the prior year,\nincluding net gains on sales of Citi’s Philippines and\nThailand consumer banking businesses versus a loss on',
    'higher revenues. Citigroup’s effective tax rate was 19.4% in\nthe current year versus 19.8% in the prior year. Earnings per\nshare (EPS) decreased 31%, reflecting the decrease in net\nincome, partially offset by a 4% decline in average diluted\nshares outstanding.\nAs discussed above, results for 2022 included divestiture-\n•\nCiti’s revenues increased 5% versus the prior year,\nincluding net gains on sales of Citi’s Philippines and\nThailand consumer banking businesses versus a loss on',
    '$742 billion and $684 billion; in Latin America of $184 billion, $179 billion and $180 billion; and in Asia of $588 billion, $572 billion and $572 billion in 2022,\n2021 and 2020, respectively. These regional numbers exclude\nCorporate/Other\n, which largely reflects U.S. activities. The Company’s long-lived assets for the\nperiods presented are not considered to be significant in relation to its total assets. The majority of Citi’s long-lived assets are located in the U.S.\n164',
    '$742 billion and $684 billion; in Latin America of $184 billion, $179 billion and $180 billion; and in Asia of $588 billion, $572 billion and $572 billion in 2022,\n2021 and 2020, respectively. These regional numbers exclude\nCorporate/Other\n, which largely reflects U.S. activities. The Company’s long-lived assets for the\nperiods presented are not considered to be significant in relation to its total assets. The majority of Citi’s long-lived assets are located in the U.S.\n164',
    'CONSOLIDATED STATEMENT OF COMPREHENSIVE INCOME\nCitigroup Inc. and Subsidiaries\nYears ended December 31,\nIn millions of dollars\n2022\n2021\n2020\nCitigroup’s net income\n$\n14,845\n$\n21,952 $\n11,047\nAdd: Citigroup’s other comprehensive income (loss)\n(1)\nNet change in unrealized gains and losses on debt securities, net of taxes\n(2)\n$\n(5,384)\n$\n(3,934) $\n3,585\nNet change in debt valuation adjustment (DVA), net of taxes\n(3)\n2,029\n232\n(475)\nNet change in cash flow hedges, net of taxes\n(2,623)\n(1,492)',
    'CONSOLIDATED STATEMENT OF COMPREHENSIVE INCOME\nCitigroup Inc. and Subsidiaries\nYears ended December 31,\nIn millions of dollars\n2022\n2021\n2020\nCitigroup’s net income\n$\n14,845\n$\n21,952 $\n11,047\nAdd: Citigroup’s other comprehensive income (loss)\n(1)\nNet change in unrealized gains and losses on debt securities, net of taxes\n(2)\n$\n(5,384)\n$\n(3,934) $\n3,585\nNet change in debt valuation adjustment (DVA), net of taxes\n(3)\n2,029\n232\n(475)\nNet change in cash flow hedges, net of taxes\n(2,623)\n(1,492)',
    '817 $\n852\nIn billions of dollars\n4Q22\n3Q22\n4Q21\nLegacy Franchises\n(1)\n$\n50\n$\n50 $\n74\nCorporate/Other\n$\n32\n$\n21 $\n7\nPersonal Banking and Wealth\nManagement\nU.S. Retail banking\n$\n37\n$\n36 $\n34\nTotal Citigroup deposits (AVG)\n$ 1,361\n$ 1,316 $ 1,370\nU.S. Cards\n143\n138\n128\nTotal Citigroup deposits (EOP)\n$ 1,366\n$ 1,306 $ 1,317\nGlobal Wealth\n150\n151\n150\nTotal\n$\n330\n$\n325 $\n312\n(1)\nSee footnote 2 to the table in “Credit Risk—Consumer Credit—\nConsumer Credit Portfolio” above.']

texts_helium3 = [
    '12 Assets under management (AUM) includes\n3\nAssets under management consist of cash and\nassets of the investment advisers affiliated\n6\nThe company’s general account investment\ninvested assets and separate account assets of the\nwith New York Life Insurance Company, other\nportfolio totaled $317.13 billion at December 31,\ncompany’s domestic and international insurance\nthan Kartesia Management, and Tristan Capital\n2022 (including $122.99 billion invested assets\noperations, and assets the company manages\nPartners, as of 12/31/2022. As of 12/31/2022\nfor NYLIAC and $8.39 billion invested assets\nfor third-party investors, including mutual funds,\nNew York Life Investments changed its AUM\nfor LINA). At December 31, 2022, total assets\nseparately managed accounts, retirement plans,\ncalculation methodology, and AUM now includes\nequaled $392.13 billion (including $184.99 billion\nSee Note 6 for and assets under certain assets, such as non-discretionary\ntotal assets for NYLIAC and $9.25 billion total\ninformation on the company’s general account\nAUM, external fund selection, and overlay\nassets for LINA). Total liabilities, excluding the\ninvestment',
    '| 0                               | 1      | 2             | 3      | 4             |\n|:--------------------------------|:-------|:--------------|:-------|:--------------|\n| Cash and Invested Assets        |        |               |        |               |\n| (In $ Billions)                 |        | Dec. 31, 2022 |        | Dec. 31, 2021 |\n| Bonds                           | $230.4 | 73%           | $221.4 | 74%           |\n| Mortgage Loans                  | 38.7   | 12%           | 35.2   | 12%           |\n| Equities                        | 15.3   | 5%            | 14.9   | 5%            |\n| Policy Loans                    | 12.6   | 4%            | 12.2   | 4%            |\n| Cash and Short-Term Investments | 9.9    | 3%            | 4.7    | 2%            |\n| Other Investments               | 4.4    | 1%            | 4.1    | 1%            |\n| Derivatives                     | 3.0    | 1%            | 1.6    | 1%            |\n| Investments in Subsidiaries     | 2.8    | 1%            | 2.9    | 1%            |\n| Total Cash and Invested Assets  | $317.1 | 100%          | $297.0 | 100%          |',
    'The portfolio is high\nmortgage loan portfolio is broadly diversified\nquality, with a loan-to-value ratio of by both property type and geographic\n$38.7\nBILLION10\n33% Multifamily\n4%\n27% Industrial\n19%\n23% Office\n24%\n9%\n15% Retail\n7%\n24%\n2% Other\n13%\nNEW YORK LIFE INSURANCE COMPANY\nNotes appear on page 15\n10\nIn particular, we utilize our extensive investment\npotential for value appreciation. We also\nEquities\ncapabilities in private equity and real estate to\ninvest in properties where opportunities exist\nadd value to the General to increase net operating income through\nWe maintain a 5%\ncapital investment and/or repositioning and\nPrivate Equities consist primarily of\nallocation to equities,\nthereby increase the property’s investments in small- and middle-market\nwhich offer higher\ncompanies through funds sponsored by\nPublic Equities are invested in a broad\nreturns and inflation\ntop-tier partners and spectrum of publicly listed companies. We\nprotection over the\nWe have extensive expertise and also long-\nutilize public equities to manage our overall\nlong standing relationships with high-performing\nallocation to equities.',
    'program, New York Life fully committed the $1\nbillion across various investments that are at\nthe heart of our impact thesis, and we continue\nto seek additional investment opportunities to\nexpand the program beyond our initial SURPLUS AND ASSET VALUATION RESERVE5\nCASH AND INVESTED ASSETS6\nIn $ Billions\nIn $ Billions\n317.1\n30.1\n2022\n2022\n297.0\n30.7\n2021\n2021\n284.2\n27.0\n2020\n2020\n268.0\n27.0\n2019\n2019\n2018\n2018\n256.1\n24.8\nNEW YORK LIFE INSURANCE COMPANY\nNotes appear on page 15\n6\nGeneral Account Investment Portfolio Overview\nNew York Life had\ncash and invested assets\nof $317.1 billion as of\nDecember 31, 2022.6\nNet Yield on Investment7\nNet yield on investment (net investment\nflow being invested at market income divided by the average of the current\nHowever, having the capability to originate\nand prior years’ invested assets) has declined\nprivate placement debt and mortgage loans\nslowly since reaching a peak in the helps mitigate the effect of a lower interest\nThis is attributable to the combined effect of\nrate higher-yielding assets maturing and new cash\n15%\nNew York Life Average\nAverage 10-Year',
    'Investment Capabilities\n$710 billion in assets under management.3\nExpertise that creates Our deep investment\nexperience and\nNew York Life had $710 billion of assets under\nNew York Life is able to access virtually all\ninvestment capabilities\nmanagement as of December 31, 2022. This\nasset classes, providing a broad universe of\nare put to work for\nincludes the $317 billion General Account—an\ninvestment opportunities to deliver long-\nour investment portfolio used to support claim\nterm, relatively stable returns. In particular, we\nand benefit payments made to clients. New\nhave the ability to originate private debt and\nYork Life’s investment boutiques manage\nequity investments. This expertise allows us\na broad array of fixed income, equity, asset\nto identify valuable investment opportunities\nallocation, sustainable investments, and\nunavailable in the public alternative investment General Account Investment Philosophy\nWe take a long-term We maintain At New York Life,\nour General Account\nWe invest for the long term because we make\nWe focus on maintaining safety and security\ninvestment philosophy\nlong-term commitments to our policy owners\nwhile pursuing superior investment',
    'Overview of\ninvestment managers13\nNewly unified alternatives investment firm\nBoutique offering a range of fixed income\nwith capabilities spanning private credit,\nstrategies, including investment grade, high\nprivate equity, GP stakes, private real assets,\nyield, bank loans, and municipals, as well as\nand long/short fundamental Specialists in cross-asset investing, leveraging\nBoutique with expertise in active the breadth and depth of the New York Life\nCapabilities across Australian equities\nInvestments’ multi-boutique and global small cap, natural resources, and\nlisted Provides investment management and\nfinancing solutions for New York Life and our\nESG-focused, active asset manager with\nvalued strategic partners, focused on fixed\nexpertise in fixed income, equity, thematic\nincome and real investing, absolute return, asset allocation,\nand liability-driven investing for pension\nfunds and insurance ~~ TRISTAN\nSs “CAPTTALPARTNERS\nReal estate investment management company\nspecializing in a wide range of property types\nPioneer and leading provider of exchange\nacross the UK and continental traded funds, granting investors access to\ninnovative solutions designed to deliver a\nsmarter approach to traditional',
    'dominated by high-\nquality investments,\nWe maintain a relatively small allocation\nwith 95% rated as\nto high yield issuers. These investments\ninvestment typically offer higher yields but have\ngreater risk of default. Our experienced\n$230.4\ninvestment team conducts thorough\nBILLION8\nresearch to identify companies with good\nbusiness fundamentals, making them\nless likely to default. We have historically\nachieved significant risk-adjusted returns\nfrom high yield investments, creating\nvalue for our NAIC 1:\nAAA to A-\n62%\nCorporate Bond Industry Diversification\nThe public and private\ncorporate bond\nportfolio, totaling\nOther\nIndustrial\nTechnology\n$142.6 billion, or\nFinance\n4%\n5%\n2%\n62% of the bond\nCable &\nportfolio, remains\nMedia\nPaper & Packaging\n7%\n5%\n4%\n2%\nConsumer\nwell diversified across\nEnergy\nProducts\nAutomotive\nthe broad industry\n2%\n16%\nspectrum, providing\n8%\nUtilities\n8%\nprotection throughout\nBanking/\nServices\nREITs\nBrokerage\n2%\nbusiness',
    'manages $661 billion in assets as of\nOur global capabilities combined with local\n12/31/22,12 including New York Life’s\npresence drive more nuanced perspective and\nGeneral Account investments and\na more personal experience for our third-party Insurance insights\nOur boutiques\nIn addition to offering investment expertise\nto our clients, our investment managers\nOur multi-boutique business model is built\npartner and collaborate with our core insurance\non the foundation of a long and stable history,\nbusiness to deliver deep insights on topics such\nwhich gives our clients proven performance\nas asset/liability management, liability-driven\nmanaging risk through multiple economic\ninvesting, and income-focused strategies, as\ncycles. With capabilities across virtually all asset\nwell as regulatory, rating agency, and accounting\nclasses, market segments, and geographies, our\nregimes. This partnership allows New York\nfamily of specialized, independent boutiques\nLife Investments to help meet the unique\nand investment teams allows us to deliver\ninvestment needs of insurance companies as\ncustomized strategies and integrated solutions\nwell as other institutional and retail for every client Investment Capabilities\nOur investment\nFixed Income\nETFs\nIndex Solutions\nEquities\nteams’ expertise\n• U.S.',
    'services, including ESG screening services,\nAsset Valuation Reserve (AVR), equaled $362.02\n4\nPolicy owner benefits primarily include death\nadvisory consulting services, white labeling\nbillion (including $174.56 billion total liabilities for\nclaims paid to beneficiaries and annuity investment management services, and model\nNYLIAC and $7.50 billion total liabilities for Dividends are payments made to eligible policy\nSee Note 5 for total portfolio delivery services, that do not qualify\nowners from divisible surplus. Divisible surplus is\nas Regulatory Assets Under Management,\n7\nThe chart represents the composite yield on\nthe portion of the company’s total surplus that\nas defined in the SEC’s Form ADV. AUM is\ninvested assets in the General Accounts of New\nis available, following each year’s operations, for\nreported in USD. AUM not denominated in USD\nYork Life and its subsidiaries. Although yields\ndistribution in the form of dividends. Dividends\nis converted at the spot rate as of shown are for a retail product (10-year are not guaranteed.',
    'Each year the board of\nThis total AUM figure is less than the sum of the\nTreasury bonds), New York Life’s net yield does\ndirectors votes on the amount and allocation of\nAUM of each affiliated investment adviser in the\nnot represent the yield of a retail product. The\nthe divisible surplus. Policy owner benefits and\ngroup because it does not count AUM where the\nchart shows how New York Life’s aggregate net\ndividends reflect the consolidated results of\nsame assets can be counted by more than one\nyield on invested assets has remained relatively\nNYLIC and its domestic insurance affiliated investment stable during periods of both rising and falling\nIntercompany transactions have been eliminated\n13 The products and services of New York Life\ninterest rates. It is indicative of New York Life’s\nin consolidation. NYLIC’s policy owner benefits\nInvestments Boutiques are not available to\nfinancial strength and does not reflect a rate of\nand dividends were $8.70 billion and $8.80 billion\nall clients in all jurisdictions or regions where\nreturn on any particular investment or insurance\nfor the years ended December 31, 2022 and 2021,\nsuch provisions would be contrary to local\nproduct.',
    '9%\nHealthcare/\nInsurance\n4%\nPharmaceuticals\n3%\nOther\nTelecommunications\n2%\nRetail\nAerospace & Defense\nTransportation\n3%\n2%\n4%\n$142.6\nConglomerates\nChemicals\nBILLION9\n5%\n3%\n2022 INVESTMENT REPORT\nNotes appear on page 15\n9\nSingle\nCorporate Bond Issuer Diversification\nLargest Issuer\n0.2%\nThe largest single issuer represents 0.2%\nThe corporate\nof cash and invested assets. Furthermore,\nbond portfolio is\nthe portfolio’s ten largest corporate bond\nmanaged to limit\nholdings represent only 1.5% of cash\nexposure to individual\nand invested assets. The corporate bond\nissuers according to\nportfolio is comprised of securities issued\ncredit quality and\nby over 3,300 individual other $317.1\nBILLION6\nCash and\nTop 10\nInvested Assets\nLargest Issuers\n100%\n1.5%\nThe company’s mortgage loan investment\nlocation. We maintain regional underwriting\nMortgage Loans\nstyle emphasizes conservative underwriting\noffices to ensure we have deep knowledge\nand a focus on high quality properties. The\nof our target markets.',
    'These holdings are\nprivate equity sponsors. In addition, our\ntypically highly liquid and offer higher return\nNYL Ventures team invests directly in\npotential in the long term compared with that\ninnovative technology partnerships focused\nof fixed income on impacting financial services, digital\nhealth, and enterprise software. We also\nmake opportunistic investments in a\nselect group of venture capital Real Estate\nPrivate\nReal Estate Equities primarily consist of\nEquities\nEquities\n36%\n53%\nhigh-quality, institutional-grade properties\ndiversified across property types and\n$15.3\ngeographic regions. We strategically focus\nBILLION11\non multifamily, industrial, office, and retail\nproperties in primary markets. These\nPublic\nEquities\ntypes of real estate investments generally\n11%\nprovide stable and predictable income, with\nAsset Class Returns and Diversification\nAs illustrated below, individual asset class benchmark returns vary from year to We maintain\nBy maintaining a diversified asset allocation, we invest throughout market cycles and\ndiversification across\ndon’t simply chase',
    'The New York Life net yield shown in this chart\n14 Based on revenue as reported by “Fortune\n5\nTotal surplus, which includes the AVR, is\nrepresents a composite net yield of the invested\n500 ranked within Industries, Insurance: Life,\none of the key indicators of the company’s\nassets of each of the following companies:\nHealth (Mutual),”Fortune magazine, long-term financial strength and stability\nNYLIC, NYLIAC, NYLAZ, LINA, and NYLGICNY,\nFor methodology, please see and is presented on a consolidated basis of\nnet of eliminations for certain intra-company\nthe company. NYLIC’s statutory surplus was\ntransactions. The curve shown represents only\n$23.89 billion and $24.57 billion at December\nNYLIC in years 1972–1979, NYLIC and NYLIAC in\n31, 2022 and 2021, respectively. Included in\nyears 1980–1986, NYLIC, NYLIAC, and NYLAZ in\n2022 INVESTMENT REPORT\n15\n',
    '7\nBonds\nThe majority of the\nPublic Corporate Bonds\n31%\nGeneral Account\ninvestment portfolio\nPrivate Corporate Bonds\n31%\nis allocated to bonds,\nwhich provide current\nAsset-Backed Securities\n10%\nincome to pay claims\nand benefits to policy\n$230.4\nCommerical Mortgage-Backed Securities\n10%\nBILLION8\nMunicipal Bonds\n7%\nResidential Mortgage-Backed Securities\n6%\nGovernment & Agency\n5%\nPublic Corporate Bonds, issued primarily\nResidential Mortgage-Backed Securities\nby investment grade companies, form the\nare investments in the residential real\ncore of our investment portfolio. We invest\nestate mortgage market. These securities\nacross a diverse group of industries. Public\nare typically pools of mortgages from a\ncorporate bonds are liquid and provide stable\ndiverse group of borrowers and geographic\ncurrent regions. A large portion of our holdings are\nissued and guaranteed by U.S. government–\nPrivate Corporate Bonds are originated by our\nsponsored dedicated team of investment This expertise allows us to identify valuable\nMunicipal Bonds provide opportunities\ninvestment opportunities unavailable in the\nto invest in states, counties, and local\npublic markets. In addition, these investments\nmunicipalities.',
    'We believe being a responsible investor is\ndisciplined approach\nWe invest in assets with similar interest rate\nconsistent with our goal to create long-term\nsensitivities and cash flow characteristics\nfinancial security for our clients and aligns our\nwhen investing the\nas our liabilities. This is done with the goal of\ninvestment activity with the broader objectives\nGeneral Account\nhaving funds available when we need to pay\nof society. Our holistic approach to investment\ninvestment benefits to clients and to protect the surplus\nanalysis incorporates a financial assessment\nof the company from adverse changes in\nas well as considering environmental, social,\ninterest rates. In addition, we maintain ample\nand governance (ESG) factors that are deemed\nliquidity in the event we need to meet large\nmaterial to a company’s performance. We\nand unexpected cash believe responsible investing is a journey that\nneeds to be thoughtfully implemented to\nWell-balanced and diversified investments\nbe effective in its outcomes, and we remain\nPortfolios with diversified asset allocations\ncommitted to sharing our progress as we',
    'Municipal investments include\nprovide further diversification, better\ngeneral obligation bonds supported by\nselectivity, and higher returns compared with\ntaxes, as well as revenue bonds that finance\nthose of public specific income-producing projects. These\ninvestments provide further diversification\nCommercial Mortgage-Backed Securities\nto our portfolio as well as exhibit longer\nprovide access to diversified pools of\nduration, high credit quality, and a historically\ncommercial mortgages that supplement our\nlow default commercial mortgage loan Government & Agency Bonds are highly\nAsset-Backed Securities are bonds backed\nliquid securities that help ensure we have\nby various types of financial receivables, such\nample funds available to pay large and\nas equipment leases, collateralized bank\nunexpected loans, royalties, or consumer NEW YORK LIFE INSURANCE COMPANY\nNotes appear on page 15\n8\nNAIC 2:\nNAIC 3–6:\nBond Portfolio Quality\nBBB+ to BBB-\nBB+ and below\n33%\n5%\nInvestment grade securities provide\nThe bond portfolio\nsafety and security while producing\ncontinues to be\nstable',
    'Net Investment Yield\nTreasury Bond Yield\n10%\n5%\n4.04%\n2.95%\n0%\n1975\n1980\n1985\n1990\n1995\n2000\n2005\n2010\n2015\n2020\n2022 INVESTMENT REPORT\nNotes appear on page 15\n',
    'is aligned with the\nand are not distracted by short-term results\nWe focus keenly on capital preservation and\nbest interests of our\nat the expense of long-term predictable investment results while seeking\nabove-market General Account Value Proposition\nDriving benefits.4\nDriving the The General Account\ninvestment portfolio\nInvestment return is a primary driver of\nOur investments positively impact the\nplays a dual role:\nbenefits paid to our clients. By staying true\neconomy—creating jobs, benefiting\nto our investment philosophy and principles,\ncommunities, supporting innovation, and\nwe create value, paying dividends to our\nfunding sustainable energy participating policy owners and growing\nour already strong 2022 INVESTMENT REPORT\nNotes appear on page 15\n5\nGeneral Account Investment Strategy and Approach\nAsset/liability management focus\nDelivering for clients and society through\nReflecting our\nresponsible investing\ninvestment philosophy,\nOur primary focuses are asset/liability\nwe take a highly\nmanagement and maintaining ample']

texts_helium4 = [
    "instructions] Please note, this -- this event is being recorded. I now like to turn the\nconference over to Mr.\nFoster, vice president of investor relations. go ahead, sir.\nFoster -- Vice President, Investor Relations\nGood afternoon and welcome to FedEx Corporation's first-quarter\nearnings conference call. The earnings release, Form 10-Q, and stat book were on our website at fedex.com. This and the accompanying\nslides are being streamed from our website, where the replay and slides will be available for about one\nyear. us on the call today are members of the media. During our question-and-answer session, callers\nwill be limited to one question in order to allow us to accommodate all those who would like to participate.\nstatements in this conference call, such as projections regarding future performance, may be\nconsidered forward-looking statements. Such statements are subject to risks, uncertainties,\nand other factors which could cause actual results to differ materially from those expressed or implied by such\nforward-looking statements. For information on these factors, please refer to our press releases and\nfilings\nwith the SEC. Please",
    "hit the ground running, and I'm very\nhappy that he has joined FedEx. So, now to the quarter. We entered fiscal\nyear '24 with strength and\nmomentum, delivering results ahead of expectations in what remains a dynamic environment.\nI'm proud what the FedEx team has accomplished over the last 12 months. Amid demand\ndisruption, we delivered on what we said we would do, driving over $2 billion in year-over-year cost savings in\nfiscal\n'23. We are now well in executing on that transmission to be the most efficient,\nflexible,\nand\nintelligent global network. Our first-quarter\ngives me great conviction in our ability to execute going\nforward. We came into the determined to provide excellent service to our customers despite the\nindustry dynamics.\nWe achieved that goal delivering innovative and data-driven solutions that further enhance the customer\nexperience. As a result, we are positioned as we prepare for the peak season. As you can see in our on Slide 6, our transformation is enhancing our profitability.\nGround was a bright spot higher revenue year\nover year driven by higher yields. On top of this growth,",
    "See the 10 stocks\n*Stock Advisor returns as of September 18, 2023\nIt has been a privilege being a longtime part of the FedEx team. I truly believe that FedEx's best days are ahead,\nbut I will be cheering from the sidelines as I am 67 years old and I want to spend more time with my family. With\nthat, I will now turn it over to Raj for him to share his views on the quarter.\nRaj Subramaniam -- President and Chief Executive Officer\nThank you, Mickey, and good afternoon. I would like to first\ncongratulate Mickey on his upcoming retirement.\nHe led our investor relations team for nearly 18 years spanning 70 earnings calls and, after tomorrow, 18\nannual meetings. He be missed by all and especially this audience.\nwe thank him for his outstanding service to FedEx over the years. And we also take this opportunity to\nwelcome John Dietrich, our chief financial\nofficer\nfor FedEx. With than 30 years of experience in the\naviation and air cargo industries, John brings a unique blend of financial\nand operational expertise to our\nleadership team at a very important time for this company. He's",
    "very impactful change, and customer feedback has been overwhelmingly\npositive. Small and medium are a high-value growth segment, and we are confident\nthat the\nimprovements underway will further enable share gains.\nAnd lastly, we've My FedEx Rewards beyond the United States into nearly 30 other countries, with\nnine more European countries to launch later this year. My FedEx Rewards is only loyalty program in the\nindustry and benefits|\nour customers by providing them with rewards they can invest in back into their business.\nThis website uses to deliver our services and to\nanalyze traffic.\nWe also share information your use\nof our site with advertising and other partners. Privacy\nPolicy\n||\nThey can use them to recognize their employees for a job well done or give back to their communities. My\nFedEx Rewards have been a successful program in the United States, and we've built lasting relationships as\nwe continue to invest in our customers. We are excited about the potential to replicate this success in Europe\nand around the world. Driving to anticipate customers' needs and provide them with superior service is deeply\nembedded in our FedEx culture.\n",
    "will we continue to provide our customers with the best\nservice and product offerings, but our plans to bring our businesses together through One FedEx and execute\non DRIVE and Network 2.0 initiatives will be truly transformative. These initiatives will leverage and optimize\neverything that the talented teams across FedEx have built over the last 50 years. It make us smarter; it will\nmake us more efficient;\nand it will enable us to serve our customers better.\nBefore into the numbers, I want to share a brief overview of the priorities that will guide me and the\nfinance\norganization as we move forward. First and I'm committed to setting stringent financial\ngoals\nthat  the significant\nopportunity we have to improve margins and returns. This be enabled by the\nDRIVE initiatives and the integration of Network 2.0 as we move toward One FedEx. I've really impressed\nby the tremendous amount of work already completed on DRIVE from the initiatives in place, the accountability\nembedded in the program, and the team's steadfast focus on execution. In terms",
    "Raj\nSubramaniam for any closing remarks. Please go ahead, sir.\nRaj Subramaniam -- President and Chief Executive Officer\nThank you very much, operator. me say that, in closing, how proud I am of our team for delivering such a\nstrong start for the year. execution of the structural cost reductions remain on track. as we prepare for\npeak, we will continue to make every FedEx experience outstanding for our customers. have proven that\nDRIVE is changing the way we work, and we are enabling continued transformation across FedEx as we build\nthe world's most flexible,\nefficient,\nand intelligent network.\nThank for your attention today. I will see you next time.\n[Operator signoff]\nDuration: 0 minutes\nCall participants:\nMickey Foster -- Vice President, Investor Relations\nRaj Subramaniam -- President and Chief Executive Officer\nBrie Carere -- Executive Vice President, Chief Customer Officer\nJohn Dietrich -- Executive Vice President, Chief Financial Officer\nJon Chappell -- Evercore ISI -- Analyst\nJack Atkins -- Stephens, Inc. -- Analyst\n",
    "I'm proud of how our teams work together to support our current customers, build relationships with new ones,\nand ensure that FedEx is positioned to succeed during the quarter. Now, I will turn it over to John to discuss the\nfinancials\nin more detail.\nDietrich -- Executive Vice President, Chief Financial Officer\nThank you, Brie, and good afternoon, everyone. I'm really excited to be here. been a full sprint these last few\nweeks as I continue to get up to speed with this great company. As of you may know, I've done business\nwith FedEx throughout my career.\nthat experience, I've always admired how FedEx literally created a new industry and has built a\ndifferentiated network that serves customers all over the world. also admired its great culture that has\nthrived through the people-service-profit,\nor PSP, philosophy. After only being here a few short weeks, I've seen\nthe incredible opportunity we have before us. Not",
    'captured upside as a result of these one-time events, we were highly\ndiscerning in terms of the business we accepted in keeping with our goal to drive high-quality\nrevenue. we expect to maintain the majority of the volume we added in the quarter. I want to thank\nour FedEx team for deftly navigating these conditions to execute on our disciplined strategy. Now to\nDRIVE.\nWe fundamentally changing the way we work, drivers taking cost out of our network, and we are on track to\ndeliver our targeted $1.8 billion in structural benefits|\nfrom DRIVE this fiscal\nyear. At Ground, DRIVE initiatives\nreduced costs by $130 million this quarter. These were primarily driven by lower third-party\ntransportation rates as a result of a newly implemented purchase bid system, as well as optimized rail usage,\nthe continued benefit\nfrom reduced Sunday coverage, and the consolidation of source. At Freight, continue\nto manage our cost base more effectively. For example, the quarter, Freight completed the planned\nclosure of 29 terminal locations during August. And at',
    "the enthusiasm from customers on how much easier it is to\nmanage as we collapse and make the -- not just the pickup experience, the physical pickup one, but we also will\nrationalize our pricing there. And we will automate pickups in a more streamlined fashion, so it's a better\ncustomer experience. To we do not -- we have not yet found opportunities to speed up the network from a\nNetwork 2.0 perspective.\nwe continue to iterate. we have found is that's a lot easier to respond and adapt in the network as we\nbring them together. And so, that has also been something that customers have asked for, especially in the B2B\nspace and healthcare. So, we are learning a lot, but the net takeaway is customers are actually very supportive\nand excited about Network 2.0.\nThis website uses cookies to deliver our services and to\nanalyze traffic.\nWe share information about your use\nof our site with advertising and other partners. Policy\n||\nThe next question will come from Ravi Shanker with Morgan Stanley. Please go ahead.\nRavi Shanker -- Morgan Stanley -- Analyst\nThanks, everyone.",
    "of our capital priorities, I'll\nfocus on maintaining a healthy balance sheet, returning cash to shareholders, and reinvesting in the business\nwith a focus on the highest returns. Our organization will partner closely with Raj and the leadership\nThis website uses cookies to deliver our services and to\nanalyze traffic.\nWe also information about your use\nof our site with advertising and other partners. Privacy\n||\nteam to ensure we deliver consistent progress toward these priorities with the goal of delivering significant\nvalue for our employees, partners, customers, and shareholders in the years to come. a guiding principle\nfor me will be to have open and transparent communication with all key stakeholders, including all of you in the\nfinancial\ncommunity.\nI know some of you from my prior roles. I forward to continuing to work together and engaging with\nthe rest of you in the weeks and months ahead. taking a closer look at our results. fiscal\nyear 2024 is\noff to an outstanding start as demonstrated by the strong operational execution in the first\nquarter. At Ground, DRIVE initiatives are taking hold, and we delivered the most profitable\nquarter in our history for that\nsegment on an adjusted basis. Adjusted",
    "are focused on harnessing the power of this rich data to make supply chains smarter for everyone, for our\ncustomers, for our customers' customers, and for ourselves. we move to the next phase of our\ntransformation, I've given the team three specific\nchallenges: to use data to make our network more efficient,\nmake our customer experiences better, and drive new profitable\nrevenue streams through digital. Looking\nahead to the rest of FY '24. We focused on delivering the highest-quality service and aggressively\nmanaging what is within our control. in better-than-expected first-quarter\nresults, we're increasing the\nmidpoint of our adjusted EPS outlook range.\nAs we to deliver on our commitments, I'm confident\nwe have the right strategy and the right team in\nplace to create significant\nvalue. With that, me turn the call over to Brie.\nBrie Carere Executive Vice President, Chief Customer Officer\nThank you, Raj, and good afternoon, everyone. In the first\nwe remain focused on revenue quality and\nbeing a valued partner to our customers. We did this in an",
    "We are well underway with plans to simplify our organization. In June 2024, FedEx Express, FedEx\nGround, and FedEx Services will consolidate into one company, Federal Express Corporation. The\nreorganization will reduce and optimize overhead, streamline our go-to-market capabilities, and improve the\ncustomer experience.\nTo date, we have implemented or announced Network 2.0 in several markets including Alaska, Hawaii, and\nCanada. As each market is different, we're continuously learning and tailoring the network to adapt to the\noperational characteristics unique to each region while delivering the highest-quality service for our\ncustomers. We continue to use both employee couriers and service providers for pickup and delivery\noperations across the network. As with any significant\ntransformation, these changes are being thoughtfully\nexecuted and will take time to complete. network that FedEx has built over the last 50 years provides us a\nfoundation that is unmatched. physical network enables us to transport millions of packages a day around\nthe world, generating terabytes of data that contain invaluable insights about the global supply chain.\n",
    "While we strive for our Foolish Best, there may be errors, omissions, or inaccuracies\nin this transcript. As with all our articles, The Motley Fool does not assume any responsibility for your use of this content, and we strongly encourage you to do your\nown research, including listening to the call yourself and reading the company's SEC filings.\nsee our Terms and Conditions for additional details, including\nour Obligatory Capitalized Disclaimers of Liability.\nMotley Fool has positions in and recommends FedEx. Motley Fool has a disclosure policy.\nwebsite uses cookies to deliver our services and to\nanalyze traffic.\nWe share information about your use\nof our site with advertising and other partners. Policy\n||\nPremium Investing Services\nInvest better with The Motley Fool. Get stock\nrecommendations, portfolio guidance, and more from The\nMotley Fool's premium services.\nView Premium Services\nMaking the world smarter, happier, and richer.\n© 1995 - 2023 The Motley Fool. All rights reserved.\nMarket data powered by Xignite.\n",
    "And, Mickey, good luck, and thanks for the help over the years. Brie, just one quick follow-up\nfor you. You said that pricing traction was good so far, and you're converting a pretty decent amount of the base\nrate increase.\nWhat percentage of that -- I think, historically has been, like, closer to 50%. Kind of what rate are you converting\nright now? And also, you said that the pricing environment remains pretty rational, but you saw the US Post\nOffice\nbasically say they're not going to have any pricing surcharges. the USPS -- the UPS changes were\nnoted on the call. I Amazon is launching some competitive service as well.\nyou think 2024 could be a tougher environment, pricing-wise, across the industry?\nCarere -- Executive Vice President, Chief Customer Officer\nOK, that was a lot, but I think -- I think I got it. Raj, jump in here if I don't get it all. So, a GRI perspective, if we\ngo back to last January, the answer is the vast majority of our customers pay the full GRI. That",
    "operating income at Ground was up 61%, and adjusted operating\nmargin expanded 480 basis points to 13.3%.\nThese results were driven by yield improvement and cost reductions, including lower line haul expense\nand improved first\nand last-mile productivity. As a cost per package was down more than 2%. At FedEx\nthe business was able to improve operating income despite a decline in revenue. This demonstrates that DRIVE is working. Adjusted income at Express was up 14%, and adjusted\noperating margin expanded 40 basis points to 2.1%.\nCost and transformation efforts at FedEx Express included structural flight\nreductions, alignment of\nstaffing\nwith volume levels, parking aircraft, and shifting to one delivery wave per day in the U.S., all of which\nmore than offset the impact of lower revenue. It's important note that expanding operating margins and\nreducing costs at Express will be a key focus for me and the team. At FedEx the team diligently\nmanaged costs and revenue quality amid a dynamic volume environment. Operating declined 290 basis\npoints based on lower fuel surcharges and shipments but remained strong at 21%. Now turning to",
    "onboarded new customers who\nvalued our service and were committed to a long-term partnership with FedEx. a result, we added\napproximately 400,000 in average daily volume by the end of the first\nquarter, and the team did an excellent job\nfocusing on commercial Ground business acquisition.\nAt Freight, revenue was down 16% driven by a 13% decline in volume. We significant\nimprovement in volume in August due to Yellow's closure. benefited\nfrom approximately 5,000\nincremental average daily shipments at attractive rates as we exited the quarter. As you can see on Slide 11,\nmonthly volumes have improved sequentially with Ground and international export volumes inflecting\npositively\non a year-over-year basis. We to continue benefiting\nfrom this quarter's market share gains throughout\nthe fiscal\nyear. We improved year-over-year growth rates, especially late in the fiscal\nyear, albeit\nwithin a muted demand environment.\nThe old we shared last quarter persisted, particularly at FedEx Express where we saw reduced fuel and\ndemand surcharges year over year. Product mix",
    "operating environment marked by continued but\nmoderating volume pressure, mixed yield dynamics, and unique developments in the competitive landscape.\nLet's take each in turn.\nThis website cookies to deliver our services and to\nanalyze traffic.\nWe also share about your use\nof our site with advertising and other partners. Privacy\nPolicy\n||\nAt FedEx Ground, first-quarter\nrevenue was up 3% year over year driven by a 1% increase in volume and 3%\nincrease in yield. at FedEx Express was down 9% year over year. remained pressured though\ntotal Express volume declines moderated sequentially. export package volumes were up 3% year\nover year. to the fourth quarter, parcel volume declines were most pronounced in the United States.\nU.S. pounds were down 27%, continuing the trend we mentioned last quarter tied to the\nchange in strategy by the United States Postal Service. the Ground and Express, volumes improved\nsequentially, aided by the threat of a strike at our primary competitor.",
    "integrate three customer platforms: customer service, marketing, and sales into one, giving the\ncustomer a more informed, efficient,\nand personalized experience when doing business with FedEx. We are\nnow offering our estimated delivery time window, which provides customers with a four-hour window for their\npackage delivery for 96% of inbound volume globally across 48 countries. This capability is nicely\ncomplemented by picture proof of delivery or, as we like to say, PPOD, which is expanded across Europe in the\nfirst\nquarter. Now in 53 markets, PPOD provides shippers with increased confidence\nin package\ndelivery and helps reduce the volume of customer calls and claims. One FedEx Network 2.0 will simplify\nhow we do business, which is particularly important for our small and medium customers.\nFor our current customer contracts reflect\nthree independent companies. One FedEx enable us to\nchange that, making doing business with FedEx and becoming a new customer easier. Network 2.0 be\nmore efficient\nfor FedEx but also more efficient\nfor our customers. When we integrate market with one truck\nin one neighborhood that's not just for deliveries, it also means a streamlined pickup experience, one pickup per\nday versus two. This is a simple"]

texts_helium5 = [
    "| 0                                              | 1   | 2     | 3   | 4     | 5                            | 6                                                                            | 7          | 8              | 9                    |\n|:-----------------------------------------------|:----|:------|:----|:------|:-----------------------------|:-----------------------------------------------------------------------------|:-----------|:---------------|:---------------------|\n| 3/28/23, 3:56 PM                               |     |       |     |       | Document                     |                                                                              |            |                |                      |\n|                                                |     |       |     |       |                              | derivative  and  non-derivative  financial  instruments)  and  interest      |            |                |                      |\n| Assets Measured at Fair Value                  |     |       |     |       |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | rate  derivative             | instruments                                                                  | to  manage | the            | impact  of  currency |\n|                                                |     | 2018  |     | 2017  |                              | exchange and interest rate fluctuations on earnings, cash flow and           |            |                |                      |\n|                                                |     |       |     |       |                              | equity. We do not enter into derivative instruments for speculative          |            |                |                      |\n| Cash and cash equivalents                      | $   | 3,616 | $   | 2,542 |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       |                              | purposes.  We  are  exposed  to  potential  credit  loss  in  the  event  of |            |                |                      |\n| Trading marketable securities                  |     | 118   |     | 121   |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       |                              | nonperformance  by  counterparties  on  our  outstanding  derivative         |            |                |                      |\n| Level 1 - Assets                               | $   | 3,734 | $   | 2,663 |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       |                              | instruments  but  do  not  anticipate  nonperformance  by  any  of  our      |            |                |                      |\n| Available-for-sale marketable securities:      |     |       |     |       |                              | counterparties.  Should  a  counterparty  default,  our  maximum             |            |                |                      |\n| Corporate and asset-backed debt securities     | $   | 38    | $   | 125   |                              | exposure to loss is the asset balance of the instrument.                     |            |                |                      |\n| Foreign government debt securities             |     | —     |     | 2     |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | 2018                         |                                                                              | Designated | Non-Designated | Total                |\n| United States agency debt securities           |     | 11    |     | 27    |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Gross notional amount        | $                                                                            | 870        |                | 5,466                |\n|                                                |     |       |     |       |                              |                                                                              | $          |                | $                    |\n|                                                |     |       |     |       |                              |                                                                              |            |                | 6,336                |\n| United States treasury debt securities         |     | 23    |     | 70    |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Maximum term in days         |                                                                              |            |                | 586                  |\n| Certificates of deposit                        |     | 11    |     | 27    |                              |                                                                              |            |                |                      |\n| Total available-for-sale marketable securities | $   | 83    | $   | 251   | Fair value:                  |                                                                              |            |                |                      |\n| Foreign currency exchange forward contracts    |     | 77    |     | 15    | Other current assets         | $                                                                            | 15         |                | 28                   |\n|                                                |     |       |     |       |                              |                                                                              | $          |                | $                    |\n|                                                |     |       |     |       |                              |                                                                              |            |                | 43                   |\n| Interest rate swap asset                       |     | —     |     | 49    | Other noncurrent assets      |                                                                              | 1          |                | 33                   |\n|                                                |     |       |     |       |                              |                                                                              |            |                | 34                   |\n|                                                |     |       |     |       | Other current liabilities    |                                                                              | (5)        |                | (15)                 |\n|                                                |     |       |     |       |                              |                                                                              |            |                | (20)                 |\n| Level 2 - Assets                               | $   | 160   | $   | 315   |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Other noncurrent liabilities |                                                                              | —          |                | —                    |\n|                                                |     |       |     |       |                              |                                                                              |            |                | —                    |\n| Total assets measured at fair value            | $   | 3,894 | $   | 2,978 |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Total fair value             | $                                                                            | 11         |                | 46                   |\n|                                                |     |       |     |       |                              |                                                                              | $          |                | $                    |\n|                                                |     |       |     |       |                              |                                                                              |            |                | 57                   |\n| Liabilities Measured at Fair Value             |     |       |     |       |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | 2017                         |                                                                              |            |                |                      |\n|                                                |     | 2018  |     | 2017  |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Gross notional amount        | $                                                                            | 1,104      |                | 4,767                |\n|                                                |     |       |     |       |                              |                                                                              | $          |                | $                    |\n|                                                |     |       |     |       |                              |                                                                              |            |                | 5,871                |\n| Deferred compensation arrangements             | $   | 118   | $   | 121   |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Maximum term in days         |                                                                              |            |                | 548                  |\n| Level 1 - Liabilities                          | $   | 118   | $   | 121   |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Fair value:                  |                                                                              |            |                |                      |\n| Foreign currency exchange forward contracts    | $   | 20    | $   | 37    |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Other current assets         | $                                                                            | 11         |                | 4                    |\n|                                                |     |       |     |       |                              |                                                                              | $          |                | $                    |\n|                                                |     |       |     |       |                              |                                                                              |            |                | 15                   |\n| Level 2 - Liabilities                          | $   | 20    | $   | 37    |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Other noncurrent assets      |                                                                              | 1          |                | —                    |\n|                                                |     |       |     |       |                              |                                                                              |            |                | 1                    |\n| Contingent consideration:                      |     |       |     |       |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Other current liabilities    |                                                                              | (7)        |                | (29)                 |\n|                                                |     |       |     |       |                              |                                                                              |            |                | (36)                 |\n| Beginning                                      | $   | 32    | $   | 86    |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Other noncurrent liabilities |                                                                              | (1)        |                | —                    |\n|                                                |     |       |     |       |                              |                                                                              |            |                | (1)                  |\n| Additions                                      |     | 77    |     | 3     |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Total fair value             | $                                                                            | 4          |                | (25) $               |\n|                                                |     |       |     |       |                              |                                                                              | $          |                | (21)                 |\n| Change in estimate                             |     | 15    |     | 2     |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       |                              | In November 2018 we designated the issuance of €2,250 of senior              |            |                |                      |\n| Settlements                                    |     | (7)   |     | (59)  |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       |                              | unsecured  notes  as  a  net  investment  hedge  to  selectively  hedge      |            |                |                      |\n| Ending                                         | $   | 117   | $   | 32    |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       |                              | portions of our investment in certain international subsidiaries. The        |            |                |                      |\n| Level 3 - Liabilities                          | $   | 117   | $   | 32    |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       |                              | currency effects of our euro-denominated senior unsecured notes              |            |                |                      |\n|                                                | $   | 255   | $   | 190   |                              |                                                                              |            |                |                      |\n| Total liabilities measured at fair value       |     |       |     |       |                              | are reflected in AOCI within shareholders' equity where they offset          |            |                |                      |\n|                                                |     |       |     |       |                              | gains  and  losses  recorded  on  our  net  investment  in  international    |            |                |                      |",
    '| 0        | 1                                                                                     |   2 |\n|:---------|:--------------------------------------------------------------------------------------|----:|\n| Item 7.  | Management’s Discussion and Analysis of Financial Condition and Results of Operations |   8 |\n| Item 7A. | Quantitative and Qualitative Disclosures About Market Risk                            |  15 |\n| Item 8.  | Financial Statements and Supplementary Data                                           |  16 |\n|          | Report of Independent Registered Public Accounting Firm                               |  16 |\n|          | Consolidated Statements of Earnings                                                   |  17 |\n|          | Consolidated Statements of Comprehensive Income                                       |  17 |\n|          | Consolidated Balance Sheets                                                           |  18 |\n|          | Consolidated Statements of Shareholders’ Equity                                       |  19 |\n|          | Consolidated Statements of Cash Flows                                                 |  20 |\n|          | Notes to Consolidated Financial Statements                                            |  21 |\n| Item 9.  | Changes in and Disagreements With Accountants on Accounting and Financial Disclosure  |  33 |']


def run_client_chat_stream_langchain_fake_embeddings(data_kind, base_model, local_server, inference_server,
                                                     simple=False, chat=True):
    t0 = time.time()

    os.environ['VERBOSE_PIPELINE'] = '1'
    remove('db_dir_UserData')

    stream_output = True
    max_new_tokens = 256
    # base_model = 'distilgpt2'
    if base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
        prompt_type = 'human_bot'
    elif base_model == 'h2oai/h2ogpt-4096-llama2-7b-chat':
        prompt_type = 'llama2'
    else:
        prompt_type = ''
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'github h2oGPT', 'LLM', 'Disabled']

    if inference_server == 'replicate':
        model_string = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
        inference_server = 'replicate:%s' % model_string
        base_model0 = 'h2oai/h2ogpt-4096-llama2-7b-chat'
        if base_model != base_model0:
            return
    elif inference_server and inference_server.startswith('openai'):
        base_model0 = 'gpt-3.5-turbo'
        if base_model != base_model0:
            return

        if inference_server == 'openai_azure_chat':
            # need at least deployment name added:
            deployment_name = 'h2ogpt'
            inference_server += ':%s:%s' % (deployment_name, 'h2ogpt.openai.azure.com/')
            if 'azure' in inference_server:
                assert 'OPENAI_AZURE_KEY' in os.environ, "Missing 'OPENAI_AZURE_KEY'"
                os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_AZURE_KEY']
    else:
        if base_model == 'gpt-3.5-turbo':
            return
        if local_server:
            assert inference_server is None

    assert base_model is not None
    if inference_server and inference_server.startswith('openai'):
        tokenizer = FakeTokenizer()
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    if local_server:
        assert not simple
        from src.gen import main
        main(base_model=base_model,
             inference_server=inference_server,
             prompt_type=prompt_type, chat=True,
             stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
             max_new_tokens=max_new_tokens,
             langchain_mode=langchain_mode,
             langchain_modes=langchain_modes,
             use_openai_embedding=True,
             verbose=True)
    print("TIME main: %s %s %s" % (data_kind, base_model, time.time() - t0), flush=True, file=sys.stderr)
    t0 = time.time()

    from src.client_test import get_client, get_args, run_client
    # serialize=False would lead to returning dict for some objects or files for get_sources
    client = get_client(serialize=False)
    print("TIME client: %s %s %s" % (data_kind, base_model, time.time() - t0), flush=True, file=sys.stderr)
    t0 = time.time()

    expect_response = True
    if data_kind == 'simple':
        texts = texts_simple
        expected_return_number = len(texts)
        expected_return_number2 = expected_return_number
        prompt = '\n'.join(texts[:expected_return_number])
        counts = count_tokens_llm(prompt, tokenizer=tokenizer)
        print('counts ', counts)
        prompt_when_texts = 'Documents'
    elif data_kind == 'helium1':
        texts = texts_helium1
        if base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
            expected_return_number = 10
            expected_return_number2 = expected_return_number
            tokens_expected = 1500
        else:
            if base_model == 'gpt-3.5-turbo':
                tokens_expected = 2600
                expected_return_number = 24  # i.e. out of 25
            elif inference_server and 'replicate' in inference_server:
                tokens_expected = 3400
                expected_return_number = 16  # i.e. out of 25
            else:
                tokens_expected = 3400
                expected_return_number = 16  # i.e. out of 25
            expected_return_number2 = expected_return_number
        prompt = '\n'.join(texts[:expected_return_number])
        counts = count_tokens_llm(prompt, tokenizer=tokenizer)
        assert counts['llm'] > tokens_expected, counts['llm']
        print('counts ', counts)
        prompt = '\n'.join(texts)
        countsall = count_tokens_llm(prompt, tokenizer=tokenizer)
        print('countsall ', countsall)
        prompt_when_texts = 'Documents'
    elif data_kind == 'helium2':
        texts = texts_helium2
        if base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
            expected_return_number = 10
            tokens_expected = 1500
            expected_return_number2 = expected_return_number
        else:
            if base_model == 'gpt-3.5-turbo':
                expected_return_number = 25 if local_server else 25
                tokens_expected = 2700 if local_server else 2700
                expected_return_number2 = 25
            elif inference_server and 'replicate' in inference_server:
                expected_return_number = 17 if local_server else 17
                tokens_expected = 3400 if local_server else 2900
                expected_return_number2 = 17
            else:
                expected_return_number = 17 if local_server else 17
                tokens_expected = 3400 if local_server else 2900
                expected_return_number2 = 17
        prompt = '\n'.join(texts[:expected_return_number])
        counts = count_tokens_llm(prompt, tokenizer=tokenizer)
        assert counts['llm'] > tokens_expected, counts['llm']
        print('counts ', counts)
        prompt = '\n'.join(texts)
        countsall = count_tokens_llm(prompt, tokenizer=tokenizer)
        print('countsall ', countsall)
        prompt_when_texts = 'Documents'
    elif data_kind == 'helium3':
        texts = texts_helium3
        if base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
            expected_return_number = 6
            tokens_expected = 1500
            expected_return_number2 = expected_return_number
        else:
            if base_model == 'gpt-3.5-turbo':
                tokens_expected = 3000 if local_server else 2900
                expected_return_number = 14 if local_server else 14
                expected_return_number2 = 15 if 'azure' not in inference_server else 14
            elif inference_server and 'replicate' in inference_server:
                tokens_expected = 3000 if local_server else 2900
                expected_return_number = 11 if local_server else 11
                expected_return_number2 = expected_return_number
            else:
                tokens_expected = 3500 if local_server else 2900
                expected_return_number = 11 if local_server else 11
                expected_return_number2 = expected_return_number
        prompt = '\n'.join(texts[:expected_return_number])
        counts = count_tokens_llm(prompt, tokenizer=tokenizer)
        assert counts['llm'] > tokens_expected, counts['llm']
        print('counts ', counts)
        prompt = '\n'.join(texts)
        countsall = count_tokens_llm(prompt, tokenizer=tokenizer)
        print('countsall ', countsall)
        prompt_when_texts = 'Documents'
    elif data_kind == 'helium4':
        texts = texts_helium4
        if base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
            expected_return_number = 5
            expected_return_number2 = 7
            expect_response = False  # fails to respond even though docs are present
            tokens_expected = 1200
        else:
            if inference_server and inference_server.startswith('replicate'):
                expected_return_number = 12 if local_server else 12
                expected_return_number2 = 14
            elif inference_server and inference_server.startswith('openai_azure'):
                expected_return_number = 14 if local_server else 14
                expected_return_number2 = 16
            elif inference_server and inference_server.startswith('openai'):
                expected_return_number = 14 if local_server else 14
                expected_return_number2 = 16
            else:
                expected_return_number = 12 if local_server else 12
                expected_return_number2 = 14
            tokens_expected = 2900 if local_server else 2900
        prompt = '\n'.join(texts[:expected_return_number])
        counts = count_tokens_llm(prompt, tokenizer=tokenizer)
        assert counts['llm'] > tokens_expected, counts['llm']
        print('counts ', counts)
        prompt = '\n'.join(texts)
        countsall = count_tokens_llm(prompt, tokenizer=tokenizer)
        print('countsall ', countsall)
        prompt_when_texts = """
Please rate the following transcript based on the tone and sentiment expressed. Express the answer as a table with the columns: "Rating" and "Reason for Rating".
Only respond with the table, no additional text. The table should be formatted like this:

| Reason | Reason for Rating |
|--------|-------------------|
| 5      | The tone of the transcript is generally positive, with expressions of optimism, enthusiasm, and pride. The speakers highlight FedEx's achievements, growth prospects, and commitment to improvement, indicating a positive outlook. However, there are also some mentions of challenges, headwinds, and areas for improvement, which prevent the tone from being entirely positive. |


Use the following scale:

1 (most negative): The transcript is overwhelmingly negative, with a critical or disapproving tone.

2 (somewhat negative): The transcript has a negative tone, but there are also some positive elements or phrases.

3 (neutral): The transcript has a balanced tone, with neither a predominantly positive nor negative sentiment.

4 (somewhat positive): The transcript has a positive tone, with more positive elements than negative ones.

5 (most positive): The transcript is overwhelmingly positive, with an enthusiastic or supportive tone."

Here's an example of how this prompt might be applied to a transcript:

"Transcript: 'I can't believe how terrible this product is. It doesn't work at all and the customer service is horrible.'

Rating: 1 (most negative)"

"Transcript: 'I have mixed feelings about this product. On the one hand, it's easy to use and the features are great, but on the other hand, it's a bit expensive and the quality could be better.'

Rating: 3 (neutral)"

"Transcript: 'I love this product! It's so intuitive and user-friendly, and the customer service is amazing. I'm so glad I bought it!'

Rating: 5 (most positive)"""
    elif data_kind == 'helium5':
        texts = texts_helium5
        if base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
            expected_return_number = 1
            expected_return_number2 = 1
            expect_response = False  # fails to respond even though docs are present
            tokens_expected = 1200
        else:
            expected_return_number = min(len(texts), 12) if local_server else min(len(texts), 12)
            expected_return_number2 = min(len(texts), 14)
            if base_model == 'gpt-3.5-turbo':
                tokens_expected = 2500 if local_server else 2500
            else:
                tokens_expected = 2900 if local_server else 2900
        prompt = '\n'.join(texts[:expected_return_number])
        counts = count_tokens_llm(prompt, tokenizer=tokenizer)
        assert counts['llm'] > tokens_expected, counts['llm']
        print('counts ', counts)
        prompt = '\n'.join(texts)
        countsall = count_tokens_llm(prompt, tokenizer=tokenizer)
        print('countsall ', countsall)
        prompt_when_texts = """Is the information on interest rate swaps present in paragraphs or tables in the document ?"""
    else:
        raise ValueError("No such data_kind=%s" % data_kind)

    if simple:
        print("TIME prep: %s %s %s" % (data_kind, base_model, time.time() - t0), flush=True, file=sys.stderr)
        # res = client.predict(texts, api_name='/file')
        res = client.predict(texts, api_name='/add_text')
        assert res is not None
        print("TIME add_text: %s %s %s" % (data_kind, base_model, time.time() - t0), flush=True, file=sys.stderr)
        return

    # for testing persistent database
    # langchain_mode = "UserData"
    # for testing ephemeral database
    langchain_mode = "MyData"
    embed = False
    chunk = False
    chunk_size = 512
    h2ogpt_key = ''
    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    print("TIME prep: %s %s %s" % (data_kind, base_model, time.time() - t0), flush=True, file=sys.stderr)
    t0 = time.time()

    prompt = "Documents"  # prompt when using langchain
    kwargs0 = dict(
        instruction='',
        max_new_tokens=200,
        min_new_tokens=1,
        max_time=300,
        do_sample=False,
        instruction_nochat=prompt,
        text_context_list=None,  # NOTE: If use same client instance and push to this textbox, will be there next call
    )

    # fast text doc Q/A
    kwargs = kwargs0.copy()
    kwargs.update(dict(
        langchain_mode=langchain_mode,
        langchain_action="Query",
        top_k_docs=-1,
        max_new_tokens=1024,
        document_subset='Relevant',
        document_choice=DocumentChoice.ALL.value,
        instruction_nochat=prompt_when_texts,
        text_context_list=texts,
    ))
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    print("Raw client result: %s" % res, flush=True)
    assert isinstance(res, str)
    res_dict = ast.literal_eval(res)
    assert 'response' in res_dict and res_dict['response']
    sources = res_dict['sources']
    texts_out = [x['content'] for x in sources]
    orig_indices = [x['orig_index'] for x in res_dict['sources']]
    texts_out = [x for _, x in sorted(zip(orig_indices, texts_out))]
    texts_expected = texts[:expected_return_number]
    assert len(texts_expected) == len(texts_out), "%s vs. %s" % (len(texts_expected), len(texts_out))
    if data_kind == 'helium5' and base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
        assert len(texts_out) == 1
        assert len(texts_expected[0]) >= len(texts_out[0])
    else:
        assert texts_expected == texts_out
    print("TIME nochat0: %s %s %s" % (data_kind, base_model, time.time() - t0), flush=True, file=sys.stderr)

    # Full langchain with db
    res = client.predict(texts,
                         langchain_mode, chunk, chunk_size, embed,
                         None, None, None, None,
                         h2ogpt_key,
                         api_name='/add_text')
    assert res[0] is None
    assert res[1] == langchain_mode
    if data_kind == 'simple':
        # else won't show entire string, so can't check this
        assert all([x in res[2] for x in texts])
    assert res[3] == ''
    print("TIME add_text: %s %s %s" % (data_kind, base_model, time.time() - t0), flush=True, file=sys.stderr)
    t0 = time.time()

    if local_server:
        from src.gpt_langchain import load_embed

        # even normal langchain_mode  passed to this should get the other langchain_mode2
        res = client.predict(langchain_mode, api_name='/load_langchain')
        persist_directory = res[1]['data'][2][3]
        if langchain_mode == 'UserData':
            persist_directory_check = 'db_dir_%s' % langchain_mode
            assert persist_directory == persist_directory_check
        got_embedding, use_openai_embedding, hf_embedding_model = load_embed(persist_directory=persist_directory)
        assert got_embedding
        assert not use_openai_embedding
        assert hf_embedding_model == 'fake'

    if not chat:
        return

    kwargs = kwargs0.copy()
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    print("Raw client result: %s" % res, flush=True)
    assert isinstance(res, str)
    res_dict = ast.literal_eval(res)
    assert 'response' in res_dict and res_dict['response']
    print("TIME nochat1: %s %s %s" % (data_kind, base_model, time.time() - t0), flush=True, file=sys.stderr)
    t0 = time.time()

    kwargs = kwargs0.copy()
    kwargs.update(dict(
        langchain_mode=langchain_mode,
        langchain_action="Query",
        top_k_docs=-1,
        document_subset='Relevant',
        document_choice=DocumentChoice.ALL.value,
    ))
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    print("Raw client result: %s" % res, flush=True)
    assert isinstance(res, str)
    res_dict = ast.literal_eval(res)
    assert 'response' in res_dict
    if expect_response:
        assert res_dict['response']
    sources = res_dict['sources']
    texts_out = [x['content'] for x in sources]
    orig_indices = [x['orig_index'] for x in res_dict['sources']]
    texts_out = [x for _, x in sorted(zip(orig_indices, texts_out))]
    texts_expected = texts[:expected_return_number2]
    assert len(texts_expected) == len(texts_out), "%s vs. %s" % (len(texts_expected), len(texts_out))
    if data_kind == 'helium5' and base_model != 'h2oai/h2ogpt-4096-llama2-7b-chat':
        pass
    else:
        assert texts_expected == texts_out
    print("TIME nochat2: %s %s %s" % (data_kind, base_model, time.time() - t0), flush=True, file=sys.stderr)


@pytest.mark.parametrize("db_type", db_types_full)
@pytest.mark.parametrize("langchain_action", ['Extract', 'Summarize'])
@pytest.mark.parametrize("instruction", ['', 'Technical key points'])
@pytest.mark.parametrize("stream_output", [False, True])
@pytest.mark.parametrize("top_k_docs", [4, -1])
@pytest.mark.parametrize("inference_server", ['https://gpt.h2o.ai', None, 'openai_chat', 'openai_azure_chat'])
@pytest.mark.parametrize("prompt_summary", [None, '', 'Summarize into single paragraph'])
@pytest.mark.need_tokens
@wrap_test_forked
def test_client_summarization(prompt_summary, inference_server, top_k_docs, stream_output, instruction,
                              langchain_action, db_type):
    # launch server
    local_server = True
    num_async = 10
    if local_server:
        if not inference_server:
            base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'
        elif inference_server == 'https://gpt.h2o.ai':
            base_model = 'h2oai/h2ogpt-4096-llama2-13b-chat'
        else:
            base_model = 'gpt-3.5-turbo'

        if inference_server == 'openai_azure_chat':
            # need at least deployment name added:
            deployment_name = 'h2ogpt'
            inference_server += ':%s:%s' % (deployment_name, 'h2ogpt.openai.azure.com/')
            if 'azure' in inference_server:
                assert 'OPENAI_AZURE_KEY' in os.environ, "Missing 'OPENAI_AZURE_KEY'"
                os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_AZURE_KEY']

        if inference_server == 'https://gpt.h2o.ai':
            model_lock = [
                dict(inference_server=inference_server, base_model=base_model, visible_models=base_model,
                     h2ogpt_key=os.getenv('H2OGPT_API_KEY'))]
            base_model = inference_server = None
        else:
            model_lock = None

        from src.gen import main
        main(base_model=base_model,
             inference_server=inference_server,
             chat=True, gradio=True, num_beams=1, block_gradio_exit=False, verbose=True,
             use_auth_token=True,
             num_async=num_async,
             model_lock=model_lock,
             db_type=db_type,
             h2ogpt_key=os.getenv('H2OGPT_KEY') or os.getenv('H2OGPT_H2OGPT_KEY'),
             )
        check_hashes = True
    else:
        # To test file is really handled remotely
        # export HOST=''  in CLI to set to some host
        check_hashes = False

    # get file for client to upload
    url = 'https://cdn.openai.com/papers/whisper.pdf'
    test_file1 = os.path.join('/tmp/', 'my_test_pdf.pdf')
    download_simple(url, dest=test_file1)

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server())

    # upload file(s).  Can be list or single file
    test_file_local, test_file_server = client.predict(test_file1, api_name='/upload_api')
    if check_hashes:
        # only makes sense if server and client on same disk
        # since co-located with server, can test that uploaded by comparing the two files
        hash_client = hash_file(test_file1)
        hash_local = hash_file(test_file_local)
        hash_server = hash_file(test_file_server)
        assert hash_client == hash_local
        assert hash_client == hash_server
    assert os.path.normpath(test_file_local) != os.path.normpath(test_file_server)

    chunk = True
    chunk_size = 512
    langchain_mode = 'MyData'
    h2ogpt_key = ''
    res = client.predict(test_file_server,
                         langchain_mode, chunk, chunk_size, True,
                         None, None, None, None,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert os.path.basename(test_file_server) in res[2]
    assert res[3] == ''

    # ask for summary, need to use same client if using MyData
    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    kwargs = dict(langchain_mode=langchain_mode,
                  langchain_action=langchain_action,  # uses full document, not vectorDB chunks
                  top_k_docs=top_k_docs,  # -1 for entire pdf
                  document_subset='Relevant',
                  document_choice=DocumentChoice.ALL.value,
                  max_new_tokens=1024,
                  max_time=1000,
                  do_sample=False,
                  prompt_summary=prompt_summary,
                  stream_output=stream_output,
                  instruction=instruction,
                  )
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    res = ast.literal_eval(res)
    summary = res['response']
    sources = res['sources']
    if langchain_action == 'Extract':
        assert isinstance(summary, list) or 'No relevant documents to extract from.' == summary
        summary = str(summary)  # for easy checking
    if instruction == 'Technical key points':
        if langchain_action == LangChainAction.SUMMARIZE_MAP.value:
            assert 'No relevant documents to summarize.' in summary or 'long-form transcription' in summary or 'text standardization' in summary or 'speech processing' in summary
        else:
            assert 'No relevant documents to extract from.' in summary or 'long-form transcription' in summary or 'text standardization' in summary or 'speech processing' in summary
    else:
        if prompt_summary == '':
            assert 'Whisper' in summary or \
                   'robust speech recognition system' in summary or \
                   'Robust speech recognition' in summary or \
                   'speech processing' in summary or \
                   'LibriSpeech dataset with weak supervision' in summary or \
                   'Large-scale weak supervision of speech' in summary or \
                   'text standardization' in summary
        else:
            assert 'various techniques and approaches in speech recognition' in summary or \
                   'capabilities of speech processing systems' in summary or \
                   'speech recognition' in summary or \
                   'capabilities of speech processing systems' in summary or \
                   'Large-scale weak supervision of speech' in summary or \
                   'text standardization' in summary
        assert 'Robust Speech Recognition' in [x['content'] for x in sources][0]
        assert 'my_test_pdf.pdf' in [x['source'] for x in sources][0]


@pytest.mark.need_tokens
@wrap_test_forked
def test_client_summarization_from_text():
    # launch server
    base_model = 'meta-llama/Llama-2-7b-chat-hf'
    from src.gen import main
    main(base_model=base_model, chat=True, gradio=True, num_beams=1, block_gradio_exit=False, verbose=True,
         use_auth_token=True,
         )

    # get file for client to upload
    url = 'https://cdn.openai.com/papers/whisper.pdf'
    test_file1 = os.path.join('/tmp/', 'my_test_pdf.pdf')
    download_simple(url, dest=test_file1)

    # Get text version of PDF
    from langchain.document_loaders import PyMuPDFLoader
    # load() still chunks by pages, but every page has title at start to help
    doc1 = PyMuPDFLoader(test_file1).load()
    all_text_contents = '\n\n'.join([x.page_content for x in doc1])

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server(), serialize=True)
    chunk = True
    chunk_size = 512
    langchain_mode = 'MyData'
    h2ogpt_key = ''
    res = client.predict(all_text_contents,
                         langchain_mode, chunk, chunk_size, True,
                         None, None, None, None,
                         h2ogpt_key,
                         api_name='/add_text')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert 'user_paste' in res[2]
    assert res[3] == ''

    # ask for summary, need to use same client if using MyData
    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    kwargs = dict(langchain_mode=langchain_mode,
                  langchain_action="Summarize",  # uses full document, not vectorDB chunks
                  top_k_docs=4,  # -1 for entire pdf
                  document_subset='Relevant',
                  document_choice=DocumentChoice.ALL.value,
                  max_new_tokens=256,
                  max_time=300,
                  do_sample=False)
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    res = ast.literal_eval(res)
    summary = res['response']
    sources = res['sources']
    assert 'Whisper' in summary or 'robust speech recognition system' in summary
    assert 'Robust Speech Recognition' in [x['content'] for x in sources][0]
    assert 'user_paste' in [x['source'] for x in sources][0]


@pytest.mark.parametrize("url", ['https://cdn.openai.com/papers/whisper.pdf', 'https://github.com/h2oai/h2ogpt'])
@pytest.mark.parametrize("top_k_docs", [4, -1])
@pytest.mark.need_tokens
@wrap_test_forked
def test_client_summarization_from_url(url, top_k_docs):
    # launch server
    base_model = 'meta-llama/Llama-2-7b-chat-hf'
    from src.gen import main
    main(base_model=base_model, chat=True, gradio=True, num_beams=1, block_gradio_exit=False, verbose=True,
         use_auth_token=True,
         )

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server(), serialize=True)
    chunk = True
    chunk_size = 512
    langchain_mode = 'MyData'
    h2ogpt_key = ''
    res = client.predict(url,
                         langchain_mode, chunk, chunk_size, True,
                         None, None, None, None,
                         h2ogpt_key,
                         api_name='/add_url')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert url in res[2]
    assert res[3] == ''

    # ask for summary, need to use same client if using MyData
    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    kwargs = dict(langchain_mode=langchain_mode,
                  langchain_action="Summarize",  # uses full document, not vectorDB chunks
                  top_k_docs=top_k_docs,  # -1 for entire pdf
                  document_subset='Relevant',
                  document_choice=DocumentChoice.ALL.value,
                  max_new_tokens=256,  # per LLM call internally, so affects both intermediate and final steps
                  max_time=300,
                  do_sample=False)
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    res = ast.literal_eval(res)
    summary = res['response']
    sources = res['sources']
    if 'whisper' in url:
        assert 'Whisper' in summary or \
               'robust speech recognition system' in summary or \
               'speech recognition' in summary
        assert 'Robust Speech Recognition' in [x['content'] for x in sources][0]
    if 'h2ogpt' in url:
        assert 'Accurate embeddings for private offline databases' in summary \
               or 'private offline database' in summary \
               or 'H2OGPT is an open-source project' in summary \
               or 'H2O GPT is an open-source project' in summary \
               or 'is an open-source project for document Q/A' in summary \
               or 'h2oGPT is an open-source project' in summary \
               or ('key results based on the provided document' in summary and 'h2oGPT' in summary)
        assert 'h2oGPT' in [x['content'] for x in sources][0]
    assert url in [x['source'] for x in sources][0]


@pytest.mark.parametrize("prompt_type", ['instruct_vicuna', 'one_shot'])
@pytest.mark.parametrize("bits", [None, 8, 4])
@pytest.mark.parametrize("stream_output", [True, False])
@pytest.mark.need_tokens
@wrap_test_forked
def test_fastsys(stream_output, bits, prompt_type):
    base_model = 'lmsys/fastchat-t5-3b-v1.0'
    from src.gen import main
    main(base_model=base_model,
         load_half=True if bits == 16 else None,
         load_4bit=bits == 4,
         load_8bit=bits == 8,
         chat=True, gradio=True, num_beams=1, block_gradio_exit=False, verbose=True,
         use_auth_token=True,
         )

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server())

    prompt = "Who are you?"
    kwargs = dict(stream_output=stream_output, instruction=prompt)
    res_dict, client = run_client_gen(client, prompt, None, kwargs)
    response = res_dict['response']
    assert """As  an  AI  language  model,  I  don't  have  a  physical  identity  or  a  physical  body.  I  exist  solely  to  assist  users  with  their  questions  and  provide  information  to  the  best  of  my  ability.  Is  there  something  specific  you  would  like  to  know  or  discuss?""" in response or \
           "As  an  AI  language  model,  I  don't  have  a  personal  identity  or  physical  presence.  I  exist  solely  to  provide  information  and  answer  questions  to  the  best  of  my  ability.  How  can  I  assist  you  today?" in response or \
           "As  an  AI  language  model,  I  don't  have  a  physical  identity  or  a  physical  presence.  I  exist  solely  to  provide  information  and  answer  questions  to  the  best  of  my  ability.  How  can  I  assist  you  today?" in response
    sources = res_dict['sources']
    assert sources == ''

    # get file for client to upload
    url = 'https://cdn.openai.com/papers/whisper.pdf'
    test_file1 = os.path.join('/tmp/', 'my_test_pdf.pdf')
    download_simple(url, dest=test_file1)

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server())

    # upload file(s).  Can be list or single file
    test_file_local, test_file_server = client.predict(test_file1, api_name='/upload_api')

    chunk = True
    chunk_size = 512
    langchain_mode = 'MyData'
    h2ogpt_key = ''
    res = client.predict(test_file_server,
                         langchain_mode, chunk, chunk_size, True,
                         None, None, None, None,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert os.path.basename(test_file_server) in res[2]
    assert res[3] == ''

    # ask for summary, need to use same client if using MyData
    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    instruction = "What is Whisper?"
    kwargs = dict(langchain_mode=langchain_mode,
                  langchain_action="Query",
                  top_k_docs=4,
                  document_subset='Relevant',
                  document_choice=DocumentChoice.ALL.value,
                  max_new_tokens=256,
                  max_time=300,
                  do_sample=False,
                  stream_output=stream_output,
                  )
    res_dict, client = run_client_gen(client, instruction, None, kwargs)
    response = res_dict['response']
    if bits is None:
        assert """Whisper is a machine learning model developed by OpenAI for speech recognition. It is trained on large amounts of text data from the internet and uses a minimalist approach to data pre-processing, relying on the expressiveness of sequence-to-sequence models to learn to map between words in a transcript. The model is designed to be able to predict the raw text of transcripts without any significant standardization, allowing it to learn to map between words in different languages without having to rely on pre-trained models.""" in response or \
               """Whisper  is  a  speech  processing  system  that  is  designed  to  generalize  well  across  domains,  tasks,  and  languages.  It  is  based  on  a  single  robust  architecture  that  is  trained  on  a  wide  set  of  existing  datasets,  and  it  is  able  to  generalize  well  across  domains,  tasks,  and  languages.  The  goal  of  Whisper  is  to  develop  a  single  robust  speech  processing  system  that  works  reliably  without  the  need  for  dataset-specific  fine-tuning  to  achieve  high-quality  results  on  specific  distributions.""" in response
    else:
        assert """single  robust  speech  processing  system  that  works""" in response or """Whisper""" in response
    sources = [x['source'] for x in res_dict['sources']]
    assert 'my_test_pdf.pdf' in sources[0]
