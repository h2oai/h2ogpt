import ast
import json
import os, sys
import shutil
import tempfile

import pytest

from tests.utils import wrap_test_forked, make_user_path_test, get_llama, get_inf_server, get_inf_port, count_tokens
from src.client_test import get_client, get_args, run_client_gen
from src.enums import LangChainAction, LangChainMode, no_model_str, no_lora_str, no_server_str, DocumentChoice
from src.utils import get_githash, remove, download_simple, hash_file, makedirs, lg_to_gr


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
                    'prompt_dict': {'promptA': '', 'promptB': '', 'PreInstruct': '<human>: ',
                                    'PreInput': None, 'PreResponse': '<bot>:',
                                    'terminate_response': ['\n<human>:', '\n<bot>:', '<human>:',
                                                           '<bot>:', '<bot>:'], 'chat_sep': '\n',
                                    'chat_turn_sep': '\n', 'humanstr': '<human>:', 'botstr': '<bot>:',
                                    'generates_leading_space': True, 'system_prompt': None},
                    'load_8bit': False, 'load_4bit': False, 'low_bit_mode': 1, 'load_half': True,
                    'load_gptq': '', 'load_exllama': False, 'use_safetensors': False,
                    'revision': None, 'use_gpu_id': True, 'gpu_id': 0, 'compile_model': True,
                    'use_cache': None,
                    'llamacpp_dict': {'n_gpu_layers': 100, 'use_mlock': True, 'n_batch': 1024,
                                      'n_gqa': 0,
                                      'model_path_llama': 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin',
                                      'model_name_gptj': 'ggml-gpt4all-j-v1.3-groovy.bin',
                                      'model_name_gpt4all_llama': 'ggml-wizardLM-7B.q4_2.bin',
                                      'model_name_exllama_if_no_config': 'TheBloke/Nous-Hermes-Llama2-GPTQ'},
                    'model_path_llama': 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin',
                    'model_name_gptj': 'ggml-gpt4all-j-v1.3-groovy.bin',
                    'model_name_gpt4all_llama': 'ggml-wizardLM-7B.q4_2.bin',
                    'model_name_exllama_if_no_config': 'TheBloke/Nous-Hermes-Llama2-GPTQ'},
                   {'base_model': 'distilgpt2', 'prompt_type': 'plain',
                    'prompt_dict': {'promptA': '', 'promptB': '', 'PreInstruct': '<human>: ',
                                    'PreInput': None, 'PreResponse': '<bot>:',
                                    'terminate_response': ['\n<human>:', '\n<bot>:', '<human>:',
                                                           '<bot>:', '<bot>:'], 'chat_sep': '\n',
                                    'chat_turn_sep': '\n', 'humanstr': '<human>:', 'botstr': '<bot>:',
                                    'generates_leading_space': True, 'system_prompt': None},
                    'load_8bit': False, 'load_4bit': False, 'low_bit_mode': 1, 'load_half': True,
                    'load_gptq': '', 'load_exllama': False, 'use_safetensors': False,
                    'revision': None, 'use_gpu_id': True, 'gpu_id': 0, 'compile_model': True,
                    'use_cache': None,
                    'llamacpp_dict': {'n_gpu_layers': 100, 'use_mlock': True, 'n_batch': 1024,
                                      'n_gqa': 0,
                                      'model_path_llama': 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin',
                                      'model_name_gptj': 'ggml-gpt4all-j-v1.3-groovy.bin',
                                      'model_name_gpt4all_llama': 'ggml-wizardLM-7B.q4_2.bin',
                                      'model_name_exllama_if_no_config': 'TheBloke/Nous-Hermes-Llama2-GPTQ'},
                    'model_path_llama': 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin',
                    'model_name_gptj': 'ggml-gpt4all-j-v1.3-groovy.bin',
                    'model_name_gpt4all_llama': 'ggml-wizardLM-7B.q4_2.bin',
                    'model_name_exllama_if_no_config': 'TheBloke/Nous-Hermes-Llama2-GPTQ'}]


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
           "My name is John." in res_dict['response']


def run_client_chat_with_server(prompt='Who are you?', stream_output=False, max_new_tokens=256,
                                base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b', prompt_type='human_bot',
                                langchain_mode='Disabled',
                                langchain_action=LangChainAction.QUERY.value,
                                langchain_agents=[],
                                user_path=None,
                                langchain_modes=['UserData', 'MyData', 'Disabled', 'LLM'],
                                model_path_llama='llama-2-7b-chat.ggmlv3.q8_0.bin',
                                reverse_docs=True):
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
         reverse_docs=reverse_docs)

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
                                  reverse_docs=True):
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
         reverse_docs=reverse_docs)

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
                                                   reverse_docs=False,  # for 6_9 dumb model for testing
                                                   )
    # below wouldn't occur if didn't use LangChain with README.md,
    # raw LLM tends to ramble about H2O.ai and what it does regardless of question.
    # bad answer about h2o.ai is just becomes dumb model, why flipped context above,
    # but not stable over different systems
    assert 'h2oGPT is a large language model' in res_dict['response'] or \
           'H2O.ai is a technology company' in res_dict['response'] or \
           'an open-source project' in res_dict['response'] or \
           'h2oGPT is a project that allows' in res_dict['response'] or \
           'h2oGPT is a language model trained' in res_dict['response']


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
         reverse_docs=False,  # for 6_9
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
            'h2oGPT is a project that' in res_dict['response']
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
         reverse_docs=False,  # for 6_9
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
         reverse_docs=False,  # for 6_9
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
    reverse_docs = True
    from src.gen import main
    main(base_model=base_model, load_gptq=load_gptq,
         use_safetensors=use_safetensors,
         prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         reverse_docs=reverse_docs)

    from src.client_test import run_client_chat
    res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''
    assert "am a virtual assistant" in res_dict['response']


@wrap_test_forked
def test_exllama():
    prompt = 'Who are you?'
    stream_output = False
    max_new_tokens = 256
    # base_model = 'TheBloke/Llama-2-70B-chat-GPTQ'
    base_model = 'TheBloke/Llama-2-7B-chat-GPTQ'
    load_exllama = True
    prompt_type = 'llama2'
    langchain_mode = 'Disabled'
    langchain_action = LangChainAction.QUERY.value
    langchain_agents = []
    user_path = None
    langchain_modes = ['UserData', 'MyData', 'LLM', 'Disabled']
    reverse_docs = True
    from src.gen import main
    main(base_model=base_model, load_exllama=load_exllama,
         prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         reverse_docs=reverse_docs)

    from src.client_test import run_client_chat
    res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''
    assert "I'm LLaMA, an AI assistant" in res_dict['response'] or "I am LLaMA" in res_dict['response']


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
    assert 'Yes, more text can be boring' in res_dict['response'] and 'sample1.pdf' in res_dict['response']

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
    source_dict = ast.literal_eval(
        client.predict(langchain_mode, file_to_get, view_raw_text, api_name='/get_document_api'))
    assert len(source_dict['contents']) == 1
    assert len(source_dict['metadatas']) == 1
    assert isinstance(source_dict['contents'][0], str)
    assert 'a cat sitting on a window' in source_dict['contents'][0]
    assert isinstance(source_dict['metadatas'][0], str)
    assert sources_expected[3] in source_dict['metadatas'][0]

    view_raw_text = True  # dict of metadatas stays dict instead of string
    source_dict = ast.literal_eval(
        client.predict(langchain_mode, file_to_get, view_raw_text, api_name='/get_document_api'))
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
    args_list = [model_choice, lora_choice, server_choice,
                 # model_state,
                 prompt_type,
                 model_load8bit_checkbox, model_load4bit_checkbox, model_low_bit_mode,
                 model_load_gptq, model_load_exllama_checkbox,
                 model_safetensors_checkbox, model_revision,
                 model_use_gpu_id_checkbox, model_gpu,
                 max_seq_len, rope_scaling,
                 model_path_llama, model_name_gptj, model_name_gpt4all_llama,
                 n_gpu_layers, n_batch, n_gqa, llamacpp_dict_more]
    res = client.predict(*tuple(args_list), api_name='/load_model')
    res_expected = ('h2oai/h2ogpt-oig-oasst1-512-6_9b', '', '', 'human_bot', {'__type__': 'update', 'maximum': 1024},
                    {'__type__': 'update', 'maximum': 1024})
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


# NOTE: llama-7b on 24GB will go OOM for helium1/2 tests
@pytest.mark.parametrize("data_kind", [
    'simple',
    'helium1',
    'helium2',
    'helium3',
])
# local_server=True
@pytest.mark.parametrize("base_model", ['h2oai/h2ogpt-oig-oasst1-512-6_9b', 'h2oai/h2ogpt-4096-llama2-7b-chat'])
# local_server=False
# @pytest.mark.parametrize("base_model", ['h2oai/h2ogpt-4096-llama2-70b-chat'])
@wrap_test_forked
def test_client_chat_stream_langchain_fake_embeddings(data_kind, base_model):
    os.environ['VERBOSE_PIPELINE'] = '1'
    remove('db_dir_UserData')

    stream_output = True
    max_new_tokens = 256
    # base_model = 'distilgpt2'
    if base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
        prompt_type = 'human_bot'
    else:
        prompt_type = 'llama2'
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'github h2oGPT', 'LLM', 'Disabled']

    local_server = True  # set to False to test local server, e.g. gradio connected to TGI server
    if local_server:
        from src.gen import main
        main(base_model=base_model, prompt_type=prompt_type, chat=True,
             stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
             max_new_tokens=max_new_tokens,
             langchain_mode=langchain_mode,
             langchain_modes=langchain_modes,
             use_openai_embedding=True,
             verbose=True)

    from src.client_test import get_client, get_args, run_client
    # serialize=False would lead to returning dict for some objects or files for get_sources
    client = get_client(serialize=False)

    if data_kind == 'simple':
        texts = ['first', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'last']
        expected_return_number = len(texts)
        counts = count_tokens('\n'.join(texts[:expected_return_number]), base_model=base_model)
        print('counts ', counts)
    elif data_kind == 'helium1':
        texts = [
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
        if base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
            expected_return_number = 10
            tokens_expected = 1500
        else:
            expected_return_number = 16  # i.e. out of 25
            tokens_expected = 3500
        counts = count_tokens('\n'.join(texts[:expected_return_number]), base_model=base_model)
        assert counts['llm'] > tokens_expected, counts['llm']
        print('counts ', counts)
        countsall = count_tokens('\n'.join(texts), base_model=base_model)
        print('countsall ', countsall)
    elif data_kind == 'helium2':
        texts = [
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
        if base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
            expected_return_number = 10
            tokens_expected = 1500
        else:
            expected_return_number = 16 if local_server else 17
            tokens_expected = 3500 if local_server else 2900
        counts = count_tokens('\n'.join(texts[:expected_return_number]), base_model=base_model)
        assert counts['llm'] > tokens_expected, counts['llm']
        print('counts ', counts)
        countsall = count_tokens('\n'.join(texts), base_model=base_model)
        print('countsall ', countsall)
    elif data_kind == 'helium3':
        texts = [
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
        if base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
            expected_return_number = 10
            tokens_expected = 1500
        else:
            expected_return_number = 16 if local_server else 11
            tokens_expected = 3500 if local_server else 2900
        counts = count_tokens('\n'.join(texts[:expected_return_number]), base_model=base_model)
        assert counts['llm'] > tokens_expected, counts['llm']
        print('counts ', counts)
        countsall = count_tokens('\n'.join(texts), base_model=base_model)
        print('countsall ', countsall)
    else:
        raise ValueError("No such data_kind=%s" % data_kind)

    # for testing persistent database
    # langchain_mode = "UserData"
    # for testing ephemeral database
    langchain_mode = "MyData"
    embed = False
    chunk = False
    chunk_size = 512
    h2ogpt_key = ''
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

    if local_server:
        from src.gpt_langchain import load_embed
        got_embedding, use_openai_embedding, hf_embedding_model = load_embed(
            persist_directory='db_dir_%s' % langchain_mode)
        assert not use_openai_embedding
        assert hf_embedding_model == 'fake'
        assert got_embedding

    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing

    prompt = "Documents"
    kwargs = dict(
        instruction='',
        max_new_tokens=200,
        min_new_tokens=1,
        max_time=300,
        do_sample=False,
        instruction_nochat=prompt,
    )
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    print("Raw client result: %s" % res, flush=True)
    assert isinstance(res, str)
    res_dict = ast.literal_eval(res)
    assert 'response' in res_dict and res_dict['response']

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
    assert 'response' in res_dict and res_dict['response']
    sources = res_dict['sources']
    texts_out = [x['content'] for x in sources]
    texts_expected = texts[:expected_return_number]
    assert len(texts_expected) == len(texts_out), "%s vs. %s" % (len(texts_expected), len(texts_out))
    assert texts_expected == texts_out


@pytest.mark.parametrize("prompt_summary", ['', 'Summarize into single paragraph'])
@pytest.mark.need_tokens
@wrap_test_forked
def test_client_summarization(prompt_summary):
    # launch server
    local_server = True
    if local_server:
        base_model = 'meta-llama/Llama-2-7b-chat-hf'
        from src.gen import main
        main(base_model=base_model, chat=True, gradio=True, num_beams=1, block_gradio_exit=False, verbose=True,
             use_auth_token=True,
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
                  langchain_action="Summarize",  # uses full document, not vectorDB chunks
                  top_k_docs=4,  # -1 for entire pdf
                  document_subset='Relevant',
                  document_choice=DocumentChoice.ALL.value,
                  max_new_tokens=256,
                  max_time=300,
                  do_sample=False,
                  prompt_summary=prompt_summary,
                  )
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    res = ast.literal_eval(res)
    summary = res['response']
    sources = res['sources']
    if prompt_summary == '':
        assert 'Whisper' in summary or \
               'robust speech recognition system' in summary or \
               'Robust speech recognition' in summary or \
               'speech processing' in summary or \
               'LibriSpeech dataset with weak supervision' in summary
    else:
        assert 'various techniques and approaches in speech recognition' in summary or \
               'capabilities of speech processing systems' in summary or \
               'speech recognition' in summary
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
               or 'h2oGPT is an open-source project' in summary
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
