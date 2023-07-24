import os
import subprocess
import time
from datetime import datetime
import pytest

from tests.utils import wrap_test_forked
from tests.test_langchain_units import have_openai_key
from src.client_test import run_client_many
from src.enums import PromptType, LangChainAction


@pytest.mark.parametrize("base_model",
                         ['h2oai/h2ogpt-oig-oasst1-512-6_9b',
                          'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2',
                          'llama', 'gptj']
                         )
@pytest.mark.parametrize("force_langchain_evaluate", [False, True])
@pytest.mark.parametrize("do_langchain", [False, True])
@wrap_test_forked
def test_gradio_inference_server(base_model, force_langchain_evaluate, do_langchain,
                                 prompt='Who are you?', stream_output=False, max_new_tokens=256,
                                 langchain_mode='Disabled', langchain_action=LangChainAction.QUERY.value,
                                 langchain_agents=[],
                                 user_path=None,
                                 visible_langchain_modes=['UserData', 'MyData'],
                                 reverse_docs=True):
    if force_langchain_evaluate:
        langchain_mode = 'MyData'
    if do_langchain:
        langchain_mode = 'UserData'
        from tests.utils import make_user_path_test
        user_path = make_user_path_test()
        # from src.gpt_langchain import get_some_dbs_from_hf
        # get_some_dbs_from_hf()

    if base_model in ['h2oai/h2ogpt-oig-oasst1-512-6_9b', 'h2oai/h2ogpt-oasst1-512-12b']:
        prompt_type = PromptType.human_bot.name
    elif base_model in ['h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2']:
        prompt_type = PromptType.prompt_answer.name
    elif base_model in ['llama']:
        prompt_type = PromptType.wizard2.name
    elif base_model in ['gptj']:
        prompt_type = PromptType.gptj.name
    else:
        raise NotImplementedError(base_model)

    main_kwargs = dict(base_model=base_model, prompt_type=prompt_type, chat=True,
                       stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
                       max_new_tokens=max_new_tokens,
                       langchain_mode=langchain_mode, langchain_action=langchain_action,
                       langchain_agents=langchain_agents,
                       user_path=user_path,
                       visible_langchain_modes=visible_langchain_modes,
                       reverse_docs=reverse_docs,
                       force_langchain_evaluate=force_langchain_evaluate)

    # inference server
    inf_port = os.environ['GRADIO_SERVER_PORT'] = "7860"
    from src.gen import main
    main(**main_kwargs)

    # server that consumes inference server
    client_port = os.environ['GRADIO_SERVER_PORT'] = "7861"
    from src.gen import main
    main(**main_kwargs, inference_server='http://127.0.0.1:%s' % inf_port)

    # client test to server that only consumes inference server
    from src.client_test import run_client_chat
    os.environ['HOST'] = "http://127.0.0.1:%s" % client_port
    res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''

    # will use HOST from above
    ret1, ret2, ret3, ret4, ret5, ret6, ret7 = run_client_many(prompt_type=None)  # client shouldn't have to specify
    if base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
        assert 'h2oGPT' in ret1['response']
        assert 'Birds' in ret2['response']
        assert 'Birds' in ret3['response']
        assert 'h2oGPT' in ret4['response']
        assert 'h2oGPT' in ret5['response']
        assert 'h2oGPT' in ret6['response']
        assert 'h2oGPT' in ret7['response']
    elif base_model == 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2':
        assert 'I am a language model trained' in ret1['response'] or \
               'I am an AI language model developed by' in ret1['response'] or \
               'I am a chatbot.' in ret1['response'] or \
               'a chat-based assistant that can answer questions' in ret1['response'] or \
               'I am an AI language model' in ret1['response'] or \
               'I am an AI assistant.' in ret1['response']
        assert 'Once upon a time' in ret2['response']
        assert 'Once upon a time' in ret3['response']
        assert 'I am a language model trained' in ret4['response'] or 'I am an AI language model developed by' in \
               ret4['response'] or 'I am a chatbot.' in ret4['response'] or \
               'a chat-based assistant that can answer questions' in ret4['response'] or \
               'I am an AI language model' in ret4['response'] or \
               'I am an AI assistant.' in ret4['response']
        assert 'I am a language model trained' in ret5['response'] or 'I am an AI language model developed by' in \
               ret5['response'] or 'I am a chatbot.' in ret5['response'] or \
               'a chat-based assistant that can answer questions' in ret5['response'] or \
               'I am an AI language model' in ret5['response'] or \
               'I am an AI assistant.' in ret5['response']
        assert 'I am a language model trained' in ret6['response'] or 'I am an AI language model developed by' in \
               ret6['response'] or 'I am a chatbot.' in ret6['response'] or \
               'a chat-based assistant that can answer questions' in ret6['response'] or \
               'I am an AI language model' in ret6['response'] or \
               'I am an AI assistant.' in ret6['response']
        assert 'I am a language model trained' in ret7['response'] or 'I am an AI language model developed by' in \
               ret7['response'] or 'I am a chatbot.' in ret7['response'] or \
               'a chat-based assistant that can answer questions' in ret7['response'] or \
               'I am an AI language model' in ret7['response'] or \
               'I am an AI assistant.' in ret7['response']
    elif base_model == 'llama':
        assert 'I am a bot.' in ret1['response'] or 'can I assist you today?' in ret1[
            'response'] or 'How can I assist you?' in ret1['response']
        assert 'Birds' in ret2['response'] or 'Once upon a time' in ret2['response']
        assert 'Birds' in ret3['response'] or 'Once upon a time' in ret3['response']
        assert 'I am a bot.' in ret4['response'] or 'can I assist you today?' in ret4[
            'response'] or 'How can I assist you?' in ret4['response']
        assert 'I am a bot.' in ret5['response'] or 'can I assist you today?' in ret5[
            'response'] or 'How can I assist you?' in ret5['response']
        assert 'I am a bot.' in ret6['response'] or 'can I assist you today?' in ret6[
            'response'] or 'How can I assist you?' in ret6['response']
        assert 'I am a bot.' in ret7['response'] or 'can I assist you today?' in ret7[
            'response'] or 'How can I assist you?' in ret7['response']
    elif base_model == 'gptj':
        assert 'I am a bot.' in ret1['response'] or 'can I assist you today?' in ret1[
            'response'] or 'a student at' in ret1['response'] or 'am a person who' in ret1['response'] or 'I am' in \
               ret1['response'] or "I'm a student at" in ret1['response']
        assert 'Birds' in ret2['response'] or 'Once upon a time' in ret2['response']
        assert 'Birds' in ret3['response'] or 'Once upon a time' in ret3['response']
        assert 'I am a bot.' in ret4['response'] or 'can I assist you today?' in ret4[
            'response'] or 'a student at' in ret4['response'] or 'am a person who' in ret4['response'] or 'I am' in \
               ret4['response'] or "I'm a student at" in ret4['response']
        assert 'I am a bot.' in ret5['response'] or 'can I assist you today?' in ret5[
            'response'] or 'a student at' in ret5['response'] or 'am a person who' in ret5['response'] or 'I am' in \
               ret5['response'] or "I'm a student at" in ret5['response']
        assert 'I am a bot.' in ret6['response'] or 'can I assist you today?' in ret6[
            'response'] or 'a student at' in ret6['response'] or 'am a person who' in ret6['response'] or 'I am' in \
               ret6['response'] or "I'm a student at" in ret6['response']
        assert 'I am a bot.' in ret7['response'] or 'can I assist you today?' in ret7[
            'response'] or 'a student at' in ret7['response'] or 'am a person who' in ret7['response'] or 'I am' in \
               ret7['response'] or "I'm a student at" in ret7['response']
    print("DONE", flush=True)


def run_docker(inf_port, base_model):
    datetime_str = str(datetime.now()).replace(" ", "_").replace(":", "_")
    msg = "Starting HF inference %s..." % datetime_str
    print(msg, flush=True)
    home_dir = os.path.expanduser('~')
    data_dir = '%s/.cache/huggingface/hub/' % home_dir
    cmd = ["docker"] + ['run',
                        '--gpus', 'device=0',
                        '--shm-size', '1g',
                        '-e', 'TRANSFORMERS_CACHE="/.cache/"',
                        '-p', '%s:80' % inf_port,
                        '-v', '%s/.cache:/.cache/' % home_dir,
                        '-v', '%s:/data' % data_dir,
                        'ghcr.io/huggingface/text-generation-inference:latest',
                        '--model-id', base_model,
                        '--max-input-length', '2048',
                        '--max-total-tokens', '4096',
                        '--max-stop-sequences', '6',
                        ]
    print(cmd, flush=True)
    p = subprocess.Popen(cmd,
                         stdout=None, stderr=subprocess.STDOUT,
                         )
    print("Done starting TGI server", flush=True)
    return p.pid


@pytest.mark.parametrize("base_model",
                         # FIXME: Can't get 6.9 or 12b (quantized or not) to work on home system, so do falcon only for now
                         # ['h2oai/h2ogpt-oig-oasst1-512-6_9b', 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2']
                         ['h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2']
                         )
@pytest.mark.parametrize("force_langchain_evaluate", [False, True])
@pytest.mark.parametrize("do_langchain", [False, True])
@pytest.mark.parametrize("pass_prompt_type", [False, True, 'custom'])
@pytest.mark.parametrize("do_model_lock", [False, True])
@wrap_test_forked
def test_hf_inference_server(base_model, force_langchain_evaluate, do_langchain, pass_prompt_type, do_model_lock,
                             prompt='Who are you?', stream_output=False, max_new_tokens=256,
                             langchain_mode='Disabled',
                             langchain_action=LangChainAction.QUERY.value,
                             langchain_agents=[],
                             user_path=None,
                             visible_langchain_modes=['UserData', 'MyData'],
                             reverse_docs=True):
    # HF inference server
    inf_port = "6112"
    inference_server = 'http://127.0.0.1:%s' % inf_port
    inf_pid = run_docker(inf_port, base_model)
    time.sleep(60)

    if force_langchain_evaluate:
        langchain_mode = 'MyData'
    if do_langchain:
        langchain_mode = 'UserData'
        from tests.utils import make_user_path_test
        user_path = make_user_path_test()
        # from src.gpt_langchain import get_some_dbs_from_hf
        # get_some_dbs_from_hf()

    if base_model in ['h2oai/h2ogpt-oig-oasst1-512-6_9b', 'h2oai/h2ogpt-oasst1-512-12b']:
        prompt_type = PromptType.human_bot.name
    else:
        prompt_type = PromptType.prompt_answer.name
    if isinstance(pass_prompt_type, str):
        prompt_type = 'custom'
        prompt_dict = """{'promptA': None, 'promptB': None, 'PreInstruct': None, 'PreInput': None, 'PreResponse': None, 'terminate_response': [], 'chat_sep': '', 'chat_turn_sep': '', 'humanstr': None, 'botstr': None, 'generates_leading_space': False}"""
    else:
        prompt_dict = None
        if not pass_prompt_type:
            prompt_type = None
    if do_model_lock:
        model_lock = [{'inference_server': inference_server, 'base_model': base_model}]
        base_model = None
        inference_server = None
    else:
        model_lock = None
    main_kwargs = dict(base_model=base_model,
                       prompt_type=prompt_type,
                       prompt_dict=prompt_dict,
                       chat=True,
                       stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
                       max_new_tokens=max_new_tokens,
                       langchain_mode=langchain_mode,
                       langchain_action=langchain_action,
                       langchain_agents=langchain_agents,
                       user_path=user_path,
                       visible_langchain_modes=visible_langchain_modes,
                       reverse_docs=reverse_docs,
                       force_langchain_evaluate=force_langchain_evaluate,
                       inference_server=inference_server,
                       model_lock=model_lock)

    try:
        # server that consumes inference server
        client_port = os.environ['GRADIO_SERVER_PORT'] = "7861"
        from src.gen import main
        main(**main_kwargs)

        # client test to server that only consumes inference server
        from src.client_test import run_client_chat
        os.environ['HOST'] = "http://127.0.0.1:%s" % client_port
        res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type,
                                           stream_output=stream_output,
                                           max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                           langchain_action=langchain_action,
                                           langchain_agents=langchain_agents,
                                           prompt_dict=prompt_dict)
        assert res_dict['prompt'] == prompt
        assert res_dict['iinput'] == ''

        # will use HOST from above
        ret1, ret2, ret3, ret4, ret5, ret6, ret7 = run_client_many(prompt_type=None)  # client shouldn't have to specify
        # here docker started with falcon before personalization

        if isinstance(pass_prompt_type, str):
            assert 'year old student from the' in ret1['response'] or 'I am a person who is asking you a question' in \
                   ret1['response']
            assert 'bird' in ret2['response']
            assert 'bird' in ret3['response']
            assert 'year old student from the' in ret4['response'] or 'I am a person who is asking you a question' in \
                   ret4['response']
            assert 'year old student from the' in ret5['response'] or 'I am a person who is asking you a question' in \
                   ret5['response']
            assert 'year old student from the' in ret6['response'] or 'I am a person who is asking you a question' in \
                   ret6['response']
            assert 'year old student from the' in ret7['response'] or 'I am a person who is asking you a question' in \
                   ret7['response']
        elif base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
            assert 'h2oGPT' in ret1['response']
            assert 'Birds' in ret2['response']
            assert 'Birds' in ret3['response']
            assert 'h2oGPT' in ret4['response']
            assert 'h2oGPT' in ret5['response']
            assert 'h2oGPT' in ret6['response']
            assert 'h2oGPT' in ret7['response']
        else:
            assert 'I am a language model trained' in ret1['response'] or 'I am an AI language model developed by' in \
                   ret1['response'] or 'a chat-based assistant' in ret1['response'] or 'am a student' in ret1[
                       'response']
            assert 'Once upon a time' in ret2['response']
            assert 'Once upon a time' in ret3['response']
            assert 'I am a language model trained' in ret4['response'] or 'I am an AI language model developed by' in \
                   ret4['response'] or 'a chat-based assistant' in ret4['response'] or 'am a student' in ret4[
                       'response']
            assert 'I am a language model trained' in ret5['response'] or 'I am an AI language model developed by' in \
                   ret5['response'] or 'a chat-based assistant' in ret5['response'] or 'am a student' in ret5[
                       'response']
            assert 'I am a language model trained' in ret6['response'] or 'I am an AI language model developed by' in \
                   ret6['response'] or 'a chat-based assistant' in ret6['response'] or 'am a student' in ret6[
                       'response']
            assert 'I am a language model trained' in ret7['response'] or 'I am an AI language model developed by' in \
                   ret7['response'] or 'a chat-based assistant' in ret7['response'] or 'am a student' in ret7[
                       'response']
        print("DONE", flush=True)
    finally:
        # take down docker server
        import signal
        try:
            os.kill(inf_pid, signal.SIGTERM)
            os.kill(inf_pid, signal.SIGKILL)
        except:
            pass

        os.system("docker ps | grep text-generation-inference | awk '{print $1}' | xargs docker stop ")


@pytest.mark.skipif(not have_openai_key, reason="requires OpenAI key to run")
@pytest.mark.parametrize("force_langchain_evaluate", [False, True])
@wrap_test_forked
def test_openai_inference_server(force_langchain_evaluate,
                                 prompt='Who are you?', stream_output=False, max_new_tokens=256,
                                 base_model='gpt-3.5-turbo',
                                 langchain_mode='Disabled',
                                 langchain_action=LangChainAction.QUERY.value,
                                 langchain_agents=[],
                                 user_path=None,
                                 visible_langchain_modes=['UserData', 'MyData'],
                                 reverse_docs=True):
    if force_langchain_evaluate:
        langchain_mode = 'MyData'

    main_kwargs = dict(base_model=base_model, chat=True,
                       stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
                       max_new_tokens=max_new_tokens,
                       langchain_mode=langchain_mode,
                       langchain_action=langchain_action,
                       langchain_agents=langchain_agents,
                       user_path=user_path,
                       visible_langchain_modes=visible_langchain_modes,
                       reverse_docs=reverse_docs)

    # server that consumes inference server
    client_port = os.environ['GRADIO_SERVER_PORT'] = "7861"
    from src.gen import main
    main(**main_kwargs, inference_server='openai_chat')

    # client test to server that only consumes inference server
    from src.client_test import run_client_chat
    os.environ['HOST'] = "http://127.0.0.1:%s" % client_port
    res_dict, client = run_client_chat(prompt=prompt, prompt_type='openai_chat', stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''

    # will use HOST from above
    ret1, ret2, ret3, ret4, ret5, ret6, ret7 = run_client_many(prompt_type=None)  # client shouldn't have to specify
    assert 'I am an AI language model' in ret1['response']
    assert 'Once upon a time, in a far-off land,' in ret2['response'] or 'Once upon a time' in ret2['response']
    assert 'Once upon a time, in a far-off land,' in ret3['response'] or 'Once upon a time' in ret3['response']
    assert 'I am an AI language model' in ret4['response']
    assert 'I am an AI language model' in ret5['response']
    assert 'I am an AI language model' in ret6['response']
    assert 'I am an AI language model' in ret7['response']
    print("DONE", flush=True)
