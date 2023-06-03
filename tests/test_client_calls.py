import pytest
from tests.utils import wrap_test_forked, make_user_path_test, get_llama


@wrap_test_forked
def test_client1():
    import os, sys
    os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
    sys.modules.pop('gpt_langchain', None)
    sys.modules.pop('langchain', None)

    from generate import main
    main(base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b', prompt_type='human_bot', chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False)

    from client_test import test_client_basic
    res_dict = test_client_basic()
    assert res_dict['prompt'] == 'Who are you?'
    assert res_dict['iinput'] == ''
    assert 'I am h2oGPT' in res_dict['response'] or "I'm h2oGPT" in res_dict['response'] or 'I’m h2oGPT' in res_dict[
        'response']


@wrap_test_forked
def test_client_chat_nostream():
    res_dict, client = run_client_chat_with_server(stream_output=False)
    assert 'I am h2oGPT' in res_dict['response'] or "I'm h2oGPT" in res_dict['response'] or 'I’m h2oGPT' in res_dict[
        'response']


@wrap_test_forked
def test_client_chat_nostream_gpt4all():
    res_dict, client = run_client_chat_with_server(stream_output=False, base_model='gptj', prompt_type='plain')
    assert 'I am a computer program designed to assist' in res_dict['response']


@wrap_test_forked
def test_client_chat_nostream_gpt4all_llama():
    res_dict, client = run_client_chat_with_server(stream_output=False, base_model='gpt4all_llama', prompt_type='plain')
    assert 'What do you want from me?' in res_dict['response'] or 'What do you want?' in res_dict['response']


@wrap_test_forked
def test_client_chat_nostream_llama7b():
    prompt_type = get_llama()
    res_dict, client = run_client_chat_with_server(stream_output=False, base_model='llama', prompt_type=prompt_type)
    assert 'I’m a software engineer' in res_dict['response']


def run_client_chat_with_server(prompt='Who are you?', stream_output=False, max_new_tokens=256,
                                base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b', prompt_type='human_bot',
                                langchain_mode='Disabled', user_path=None,
                                visible_langchain_modes=['UserData', 'MyData']):
    import os, sys
    if langchain_mode == 'Disabled':
        os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
        sys.modules.pop('gpt_langchain', None)
        sys.modules.pop('langchain', None)

    from generate import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         visible_langchain_modes=visible_langchain_modes)

    from client_test import run_client_chat
    res_dict, client = run_client_chat(prompt=prompt, prompt_type='human_bot', stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''
    return res_dict, client


@wrap_test_forked
def test_client_chat_stream():
    run_client_chat_with_server(stream_output=True)


@wrap_test_forked
def test_client_chat_stream_langchain():
    user_path = make_user_path_test()
    prompt = "What is h2oGPT?"
    res_dict, client = run_client_chat_with_server(prompt=prompt, stream_output=True, langchain_mode="UserData",
                                                   user_path=user_path,
                                                   visible_langchain_modes=['UserData', 'MyData'])
    # below wouldn't occur if didn't use LangChain with README.md,
    # raw LLM tends to ramble about H2O.ai and what it does regardless of question.
    assert 'h2oGPT is a large language model' in res_dict['response']


@wrap_test_forked
def test_client_chat_stream_langchain_steps():
    user_path = make_user_path_test()

    stream_output = True
    max_new_tokens = 256
    base_model = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    prompt_type = 'human_bot'
    langchain_mode = 'UserData'
    visible_langchain_modes = ['UserData', 'MyData']

    from generate import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         visible_langchain_modes=visible_langchain_modes)

    from client_test import get_client, get_args, run_client
    client = get_client(serialize=False)

    # QUERY1
    prompt = "What is h2oGPT?"
    langchain_mode = 'UserData'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    assert 'a large language model' in res_dict['response'] and 'FAQ.md' in res_dict['response']

    # QUERY2
    prompt = "What is h2oGPT?"
    langchain_mode = 'ChatLLM'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    # i.e. answers wrongly without data, dumb model, but also no docs at all since cutoff entirely
    assert 'H2O.ai is a technology company' in res_dict['response'] and '.md' not in res_dict['response']

    # QUERY3
    prompt = "What is whisper?"
    langchain_mode = 'UserData'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    # odd answer since no whisper docs, but still shows some docs at very low score
    assert 'Whisper is a secure messaging app that allows' in res_dict['response'] and '.md' in res_dict['response']


@wrap_test_forked
def test_client_chat_stream_langchain_steps2():
    # full user data
    from make_db import make_db_main
    make_db_main(download_some=True)
    user_path = None  # shouldn't be necessary, db already made

    stream_output = True
    max_new_tokens = 256
    base_model = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    prompt_type = 'human_bot'
    langchain_mode = 'UserData'
    visible_langchain_modes = ['UserData', 'MyData']

    from generate import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         visible_langchain_modes=visible_langchain_modes,
         verbose=True)

    from client_test import get_client, get_args, run_client
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
    assert 'speech recognition system' in res_dict['response'] and 'whisper.pdf' in res_dict['response']


@wrap_test_forked
def test_client_chat_stream_long():
    prompt = 'Tell a very long story about cute birds for kids.'
    res_dict, client = run_client_chat_with_server(prompt=prompt, stream_output=True, max_new_tokens=1024)
    assert 'Once upon a time' in res_dict['response']


@pytest.mark.skip(reason="Local file required")
@wrap_test_forked
def test_client_long():
    import os, sys
    os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
    sys.modules.pop('gpt_langchain', None)
    sys.modules.pop('langchain', None)

    from generate import main
    main(base_model='mosaicml/mpt-7b-storywriter', prompt_type='plain', chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False)

    with open("/home/jon/Downloads/Gatsby_PDF_FullText.txt") as f:
        prompt = f.readlines()

    from client_test import run_client_nochat
    res_dict = run_client_nochat(prompt=prompt, prompt_type='plain', max_new_tokens=86000)
    print(res_dict['response'])
