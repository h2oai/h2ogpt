import ast
import json
import os, sys
import pytest

from tests.utils import wrap_test_forked, make_user_path_test, get_llama
from src.client_test import get_client, get_args, run_client_gen
from src.enums import LangChainAction
from src.utils import get_githash


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
@wrap_test_forked
def test_client1api_lean(admin_pass):
    from src.gen import main
    base_model = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    os.environ['ADMIN_PASS'] = admin_pass
    inf_port = os.environ['GRADIO_SERVER_PORT'] = "9999"
    main(base_model=base_model, prompt_type='human_bot', chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False)

    os.environ['HOST'] = "http://127.0.0.1:%s" % inf_port

    client1 = get_client(serialize=True)

    from gradio_utils.grclient import GradioClient
    client2 = GradioClient(os.environ['HOST'])
    client2.refresh_client()  # test refresh

    for client in [client1, client2]:
        api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
        prompt = 'Who are you?'
        kwargs = dict(instruction_nochat=prompt)
        # pass string of dict.  All entries are optional, but expect at least instruction_nochat to be filled
        res = client.predict(str(dict(kwargs)), api_name=api_name)

        print("Raw client result: %s" % res, flush=True)
        response = ast.literal_eval(res)['response']

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
    prompt_type = get_llama()
    res_dict, client = run_client_chat_with_server(stream_output=False, base_model='llama', prompt_type=prompt_type)
    assert "am a virtual assistant" in res_dict['response'] or \
           'am a student' in res_dict['response']


def run_client_chat_with_server(prompt='Who are you?', stream_output=False, max_new_tokens=256,
                                base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b', prompt_type='human_bot',
                                langchain_mode='Disabled',
                                langchain_action=LangChainAction.QUERY.value,
                                langchain_agents=[],
                                user_path=None,
                                visible_langchain_modes=['UserData', 'MyData'],
                                reverse_docs=True):
    if langchain_mode == 'Disabled':
        os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
        sys.modules.pop('gpt_langchain', None)
        sys.modules.pop('langchain', None)

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         visible_langchain_modes=visible_langchain_modes,
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
                                  visible_langchain_modes=['UserData', 'MyData'],
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
         visible_langchain_modes=visible_langchain_modes,
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
                                                   visible_langchain_modes=['UserData', 'MyData'],
                                                   reverse_docs=False,  # for 6_9 dumb model for testing
                                                   )
    # below wouldn't occur if didn't use LangChain with README.md,
    # raw LLM tends to ramble about H2O.ai and what it does regardless of question.
    # bad answer about h2o.ai is just becomes dumb model, why flipped context above,
    # but not stable over different systems
    assert 'h2oGPT is a large language model' in res_dict['response'] or \
           'H2O.ai is a technology company' in res_dict['response']


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
    visible_langchain_modes = ['UserData', 'MyData']

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         top_k_docs=top_k_docs,
         langchain_mode=langchain_mode, user_path=user_path,
         visible_langchain_modes=visible_langchain_modes,
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
            'H2O.ai is a technology company' in res_dict['response']
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
            'secure, decentralized, and anonymous chat platform' in res_dict['response']
            ) \
           and ('FAQ.md' in res_dict['response'] or 'README.md' in res_dict['response'])

    # QUERY2
    prompt = "What is h2oGPT?"
    langchain_mode = 'ChatLLM'
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
            'A secret communication system used' in res_dict['response']
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
    visible_langchain_modes = ['UserData', 'MyData', 'github h2oGPT']

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         visible_langchain_modes=visible_langchain_modes,
         verbose=True)

    from src.client_test import get_client, get_args, run_client
    client = get_client(serialize=False)

    # QUERY1
    prompt = "Who are you?"
    langchain_mode = 'ChatLLM'
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
            'h2oGPT is an open-source language model' in res_dict['response'] or
            'h2oGPT is an open-source, fully permissive, commercially usable' in res_dict['response']
            ) and \
           'README.md' in res_dict['response']


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
    visible_langchain_modes = ['UserData', 'MyData']
    reverse_docs = True
    from src.gen import main
    main(base_model=base_model, load_gptq=load_gptq,
         use_safetensors=use_safetensors,
         prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         visible_langchain_modes=visible_langchain_modes,
         reverse_docs=reverse_docs)

    from src.client_test import run_client_chat
    res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''
    assert "am a virtual assistant" in res_dict['response']


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
                    reason="For testing text-generatino-inference server")
@wrap_test_forked
def test_text_generation_inference_server1():
    """
    e.g.
    SERVER on 192.168.1.46
    (alpaca) jon@gpu:/data/jon/h2o-llm$ CUDA_VISIBLE_DEVICES=0,1 docker run --gpus all --shm-size 2g -e NCCL_SHM_DISABLE=1 -e TRANSFORMERS_CACHE="/.cache/" -p 6112:80 -v $HOME/.cache:/.cache/ -v $HOME/.cache/huggingface/hub/:/data  ghcr.io/huggingface/text-generation-inference:0.8.2 --model-id h2oai/h2ogpt-oasst1-512-12b --max-input-length 2048 --max-total-tokens 4096 --sharded=true --num-shard=2 --disable-custom-kernels --quantize bitsandbytes --trust-remote-code --max-stop-sequences=6

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
