import ast
import json
import os, sys
import shutil
import tempfile

import pytest

from tests.utils import wrap_test_forked, make_user_path_test, get_llama, get_inf_server, get_inf_port
from src.client_test import get_client, get_args, run_client_gen
from src.enums import LangChainAction, LangChainMode
from src.utils import get_githash, remove, download_simple, hash_file, makedirs


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
            assert res['save_dict']['error'] is None
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
    base_model = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    prompt_type = 'human_bot'
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
    assert 'H2O.ai is a technology company' in res_dict['response'] and '.md' not in res_dict['response']

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
            'whisper is a chat bot'  in res_dict['response']
            ) \
           and '.md' in res_dict['response']


@pytest.mark.need_tokens
@pytest.mark.parametrize("max_new_tokens", [256, 2048])
@pytest.mark.parametrize("top_k_docs", [3, 100])
@wrap_test_forked
def test_client_chat_stream_langchain_steps2(max_new_tokens, top_k_docs):
    os.environ['VERBOSE_PIPELINE'] = '1'
    # full user data
    from src.make_db import make_db_main
    make_db_main(download_some=True)
    user_path = None  # shouldn't be necessary, db already made

    stream_output = True
    max_new_tokens = 256
    base_model = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    prompt_type = 'human_bot'
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'github h2oGPT', 'LLM', 'Disabled']

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         verbose=True)

    from src.client_test import get_client, get_args, run_client
    client = get_client(serialize=False)

    # QUERY1
    prompt = "Who are you?"
    langchain_mode = 'LLM'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    assert 'a large language model' in res_dict['response'] and 'FAQ.md' not in res_dict['response']

    # QUERY2
    prompt = "What is whisper?"
    langchain_mode = 'UserData'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    assert 'large-scale speech recognition model' in res_dict['response'] and 'whisper.pdf' in res_dict['response']

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
    load_gptq = 'nous-hermes-13b-GPTQ-4bit-128g.no-act.order'
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
    (alpaca) jon@gpu:/data/jon/h2o-llm$ CUDA_VISIBLE_DEVICES=0,1 docker run --gpus all --shm-size 2g -e NCCL_SHM_DISABLE=1 -e TRANSFORMERS_CACHE="/.cache/" -p 6112:80 -v $HOME/.cache:/.cache/ -v $HOME/.cache/huggingface/hub/:/data  ghcr.io/huggingface/text-generation-inference:latest --model-id h2oai/h2ogpt-oasst1-512-12b --max-input-length 2048 --max-total-tokens 4096 --sharded=true --num-shard=2 --disable-custom-kernels --quantize bitsandbytes --trust-remote-code --max-stop-sequences=6

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
def test_client_chat_stream_langchain_steps3():
    os.environ['VERBOSE_PIPELINE'] = '1'
    user_path = make_user_path_test()

    stream_output = True
    max_new_tokens = 256
    base_model = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    prompt_type = 'human_bot'
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'github h2oGPT', 'LLM', 'Disabled']

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         verbose=True)

    from src.client_test import get_client, get_args, run_client
    # serialize=False would lead to returning dict for some objects or files for get_sources
    client = get_client(serialize=False)

    url = 'https://www.africau.edu/images/default/sample.pdf'
    test_file1 = os.path.join('/tmp/', 'sample1.pdf')
    download_simple(url, dest=test_file1)
    res = client.predict(test_file1, True, 512, langchain_mode, api_name='/add_file_api')
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
    assert langchain_mode2 in res[0]['choices']
    assert res[1] == ''
    assert res[2]['headers'] == ['Collection', 'Type', 'Path', 'Directory']
    res[2]['data'] = [[x[0], x[1], x[2]] for x in res[2]['data']]  # ignore persist_directory
    assert res[2]['data'] == [['UserData', 'shared', user_path],
                              ['github h2oGPT', 'shared', ''],
                              ['MyData', 'personal', ''],
                              [langchain_mode2, 'shared', user_path2]]

    # url = 'https://unec.edu.az/application/uploads/2014/12/pdf-sample.pdf'
    test_file1 = os.path.join('/tmp/', 'pdf-sample.pdf')
    # download_simple(url, dest=test_file1)
    shutil.copy('tests/pdf-sample.pdf', test_file1)
    res = client.predict(test_file1, True, 512, langchain_mode2, api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode2
    assert 'file/%s/pdf-sample.pdf' % user_path2 in res[2] or 'file/%s\pdf-sample.pdf' % user_path2 in res[2]
    assert 'sample1.pdf' not in res[2]  # ensure no leakage
    assert res[3] == ''

    # QUERY1
    prompt = "Is more text boring?"
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    assert 'Yes, it is.' in res_dict['response'] and 'sample1.pdf' in res_dict['response']

    # QUERY2
    prompt = "What is a universal file format?"
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode2)

    res_dict, client = run_client(client, prompt, args, kwargs)
    assert 'PDF' in res_dict['response'] and 'pdf-sample.pdf' in res_dict['response']

    # check sources, and do after so would detect leakage
    res = client.predict(langchain_mode, api_name='/get_sources')
    # is not actual data!
    with open(res['name'], 'rb') as f:
        sources = f.read().decode()
    sources_expected = f'{user_path}/FAQ.md\n{user_path}/README.md\n{user_path}/pexels-evg-kowalievska-1170986_small.jpg\n{user_path}/sample1.pdf'
    assert sources == sources_expected or sources.replace('\\', '/').replace('\r', '') == sources_expected.replace(
        '\\', '/').replace('\r', '')

    res = client.predict(langchain_mode2, api_name='/get_sources')
    with open(res['name'], 'rb') as f:
        sources = f.read().decode()
    sources_expected = """%s/pdf-sample.pdf""" % user_path2
    assert sources == sources_expected or sources.replace('\\', '/').replace('\r', '') == sources_expected.replace(
        '\\', '/').replace('\r', '')

    # check sources, and do after so would detect leakage
    res = client.predict(langchain_mode, api_name='/get_viewable_sources')
    # is not actual data!
    with open(res['name'], 'rb') as f:
        sources = f.read().decode()
    sources_expected = f'{user_path}/FAQ.md\n{user_path}/README.md\n{user_path}/pexels-evg-kowalievska-1170986_small.jpg\n{user_path}/sample1.pdf'
    assert sources == sources_expected or sources.replace('\\', '/').replace('\r', '') == sources_expected.replace(
        '\\', '/').replace('\r', '')

    res = client.predict(langchain_mode2, api_name='/get_viewable_sources')
    with open(res['name'], 'rb') as f:
        sources = f.read().decode()
    sources_expected = """%s/pdf-sample.pdf""" % user_path2
    assert sources == sources_expected or sources.replace('\\', '/').replace('\r', '') == sources_expected.replace(
        '\\', '/').replace('\r', '')

    # refresh
    shutil.copy('tests/next.txt', user_path)
    res = client.predict(langchain_mode, True, 512, api_name='/refresh_sources')
    sources_expected = 'file/%s/next.txt' % user_path
    assert sources_expected in res or sources_expected.replace('\\', '/').replace('\r', '') in res.replace('\\',
                                                                                                           '/').replace(
        '\r', '\n')

    res = client.predict(langchain_mode, api_name='/get_sources')
    # is not actual data!
    with open(res['name'], 'rb') as f:
        sources = f.read().decode()
    sources_expected = f'{user_path}/FAQ.md\n{user_path}/README.md\n{user_path}/next.txt\n{user_path}/pexels-evg-kowalievska-1170986_small.jpg\n{user_path}/sample1.pdf'
    assert sources == sources_expected or sources.replace('\\', '/').replace('\r', '') == sources_expected.replace(
        '\\', '/').replace('\r', '')

    # check sources, and do after so would detect leakage
    sources = ast.literal_eval(client.predict(langchain_mode, api_name='/get_sources_api'))
    assert isinstance(sources, list)
    sources_expected = ['user_path_test/FAQ.md', 'user_path_test/README.md', 'user_path_test/next.txt', 'user_path_test/pexels-evg-kowalievska-1170986_small.jpg', 'user_path_test/sample1.pdf']
    assert sources == sources_expected

    # even normal langchain_mode  passed to this should get the other langchain_mode2
    res = client.predict(langchain_mode, api_name='/load_langchain')
    assert res[0]['choices'] == [langchain_mode, 'MyData', 'github h2oGPT', 'LLM', langchain_mode2]
    assert res[0]['value'] == langchain_mode
    assert res[1]['headers'] == ['Collection', 'Type', 'Path', 'Directory']
    res[1]['data'] = [[x[0], x[1], x[2]] for x in res[1]['data']]  # ignore persist_directory
    assert res[1]['data'] == [['UserData', 'shared', user_path],
                              ['github h2oGPT', 'shared', ''],
                              ['MyData', 'personal', ''],
                              [langchain_mode2, 'shared', user_path2]]

    # for pure-UI things where just input -> output often, just make sure no failure, if can
    res = client.predict(api_name='/export_chats')
    assert res is not None

    url = 'https://research.google/pubs/pub334.pdf'
    res = client.predict(url, True, 512, langchain_mode, api_name='/add_url')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert url in res[2]
    assert res[3] == ''

    text = "Yufuu is a wonderful place and you should really visit because there is lots of sun."
    res = client.predict(text, True, 512, langchain_mode, api_name='/add_text')
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
    res = client.predict(test_file1, True, 512, langchain_mode_my, api_name='/add_file_api')
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
    assert langchain_mode2 in res[0]['choices']
    assert res[1] == ''
    assert res[2]['headers'] == ['Collection', 'Type', 'Path', 'Directory']
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
    res = client.predict(test_file1, True, 512, langchain_mode2, api_name='/add_file_api')
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
        res = client.predict(urls_file, True, 512, langchain_mode2, api_name='/add_file_api')
        assert res[0] is None
        assert res[1] == langchain_mode2
        assert [x in res[2] or x.replace('https', 'http') in res[2] for x in urls]
        assert res[3] == ''

    langchain_mode3 = 'MyData3'
    user_path3 = ''
    new_langchain_mode_text = '%s, %s, %s' % (langchain_mode3, 'personal', user_path3)
    res = client.predict(langchain_mode3, new_langchain_mode_text, api_name='/new_langchain_mode_text')
    assert res[0]['value'] == langchain_mode3
    assert langchain_mode3 in res[0]['choices']
    assert res[1] == ''
    assert res[2]['headers'] == ['Collection', 'Type', 'Path', 'Directory']
    res[2]['data'] = [[x[0], x[1], x[2]] for x in res[2]['data']]  # ignore persist_directory
    assert res[2]['data'] == [['UserData', 'shared', user_path],
                              ['github h2oGPT', 'shared', ''],
                              ['MyData', 'personal', ''],
                              ['UserData2', 'shared', user_path2],
                              [langchain_mode2, 'personal', ''],
                              [langchain_mode3, 'personal', ''],
                              ]

    with tempfile.TemporaryDirectory() as tmp_user_path:
        res = client.predict(urls, True, 512, langchain_mode3, api_name='/add_url')
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
    sources_state_after_delete = [x.replace('v1', '').replace('v7', '') for x in sources_state_after_delete]  # for arxiv for asserts
    assert isinstance(sources_state_after_delete, list)
    source_list_assert = [x.replace('v1', '').replace('v7', '') for x in source_list]  # for arxiv for asserts
    assert source_list_assert[0] not in sources_state_after_delete

    res = client.predict(langchain_mode3, langchain_mode3, api_name='/remove_langchain_mode_text')
    assert res[0]['value'] == langchain_mode3
    assert langchain_mode2 in res[0]['choices']
    assert res[1] == ''
    assert res[2]['headers'] == ['Collection', 'Type', 'Path', 'Directory']
    res[2]['data'] = [[x[0], x[1], x[2]] for x in res[2]['data']]  # ignore persist_directory
    assert res[2]['data'] == [['UserData', 'shared', user_path],
                              ['github h2oGPT', 'shared', ''],
                              ['MyData', 'personal', ''],
                              ['UserData2', 'shared', user_path2],
                              [langchain_mode2, 'personal', '']]

    # FIXME: could do MyData personal type, but would need to know path and don't have access via API
    assert os.path.isdir("db_dir_%s" % langchain_mode)
    res = client.predict(langchain_mode, langchain_mode, api_name='/purge_langchain_mode_text')
    assert not os.path.isdir("db_dir_%s" % langchain_mode)
    assert res[0]['value'] == langchain_mode
    assert langchain_mode not in res[0]['choices']
    assert res[1] == ''
    assert res[2]['headers'] == ['Collection', 'Type', 'Path', 'Directory']
    res[2]['data'] = [[x[0], x[1], x[2]] for x in res[2]['data']]  # ignore persist_directory
    assert res[2]['data'] == [['github h2oGPT', 'shared', ''],
                              ['MyData', 'personal', ''],
                              ['UserData2', 'shared', 'user_path2'],
                              ['MyData2', 'personal', ''],
                              ]

    # FIXME: Add load_model, unload_model, etc.


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
    res = client.predict(test_file1, True, 512, langchain_mode, api_name='/add_file_api')
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
    res = client.predict(test_file_server, chunk, chunk_size, langchain_mode, api_name='/add_file_api')
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
                  document_choice='All',
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
    assert 'my_test_pdf.pdf' in sources


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
    res = client.predict(all_text_contents, chunk, chunk_size, langchain_mode, api_name='/add_text')
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
                  document_choice='All',
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
    assert 'user_paste' in sources


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
    res = client.predict(url, chunk, chunk_size, langchain_mode, api_name='/add_url')
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
                  document_choice='All',
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
    if 'h2ogpt' in url:
        assert 'Accurate embeddings for private offline databases' in summary \
               or 'private offline database' in summary \
               or 'H2OGPT is an open-source project' in summary \
               or 'is an open-source project for document Q/A' in summary \
               or 'h2oGPT is an open-source project' in summary
    assert url in sources


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
    res = client.predict(test_file_server, chunk, chunk_size, langchain_mode, api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert os.path.basename(test_file_server) in res[2]
    assert res[3] == ''

    # ask for summary, need to use same client if using MyData
    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    kwargs = dict(langchain_mode=langchain_mode,
                  langchain_action="Query",
                  top_k_docs=4,
                  document_subset='Relevant',
                  document_choice='All',
                  max_new_tokens=256,
                  max_time=300,
                  do_sample=False,
                  stream_output=stream_output,
                  instruction="What is Whisper?",
                  )
    res_dict, client = run_client_gen(client, prompt, None, kwargs)
    response = res_dict['response']
    if bits is None:
        assert """Whisper is a machine learning model developed by OpenAI for speech recognition. It is trained on large amounts of text data from the internet and uses a minimalist approach to data pre-processing, relying on the expressiveness of sequence-to-sequence models to learn to map between words in a transcript. The model is designed to be able to predict the raw text of transcripts without any significant standardization, allowing it to learn to map between words in different languages without having to rely on pre-trained models.""" in response or \
            """Whisper  is  a  speech  processing  system  that  is  designed  to  generalize  well  across  domains,  tasks,  and  languages.  It  is  based  on  a  single  robust  architecture  that  is  trained  on  a  wide  set  of  existing  datasets,  and  it  is  able  to  generalize  well  across  domains,  tasks,  and  languages.  The  goal  of  Whisper  is  to  develop  a  single  robust  speech  processing  system  that  works  reliably  without  the  need  for  dataset-specific  fine-tuning  to  achieve  high-quality  results  on  specific  distributions.""" in response
    else:
        assert """single  robust  speech  processing  system  that  works""" in response or """Whisper""" in response
    sources = res_dict['sources']
    assert 'my_test_pdf.pdf' in sources
