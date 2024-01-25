import os
import subprocess
import time
from datetime import datetime
import pytest

from src.utils import get_ngpus_vis, makedirs
from tests.utils import wrap_test_forked, get_inf_port, get_inf_server
from tests.test_langchain_units import have_openai_key, have_replicate_key
from src.client_test import run_client_many, test_client_basic_api_lean
from src.enums import PromptType, LangChainAction


@pytest.mark.parametrize("base_model",
                         ['h2oai/h2ogpt-oig-oasst1-512-6_9b',
                          'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2',
                          'llama', 'gptj']
                         )
@pytest.mark.parametrize("force_langchain_evaluate", [False, True])
@pytest.mark.parametrize("do_langchain", [False, True])
@pytest.mark.parametrize("enforce_h2ogpt_api_key", [False, True])
@pytest.mark.parametrize("enforce_h2ogpt_ui_key", [False, True])
@wrap_test_forked
def test_gradio_inference_server(base_model, force_langchain_evaluate, do_langchain,
                                 enforce_h2ogpt_ui_key, enforce_h2ogpt_api_key,
                                 prompt='Who are you?', stream_output=False, max_new_tokens=256,
                                 langchain_mode='Disabled', langchain_action=LangChainAction.QUERY.value,
                                 langchain_agents=[],
                                 user_path=None,
                                 langchain_modes=['UserData', 'MyData', 'LLM', 'Disabled'],
                                 docs_ordering_type='reverse_sort'):
    if enforce_h2ogpt_api_key and base_model != 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
        # no need for so many cases
        return
    if force_langchain_evaluate:
        langchain_mode = 'MyData'
    if do_langchain:
        langchain_mode = 'UserData'
        from tests.utils import make_user_path_test
        user_path = make_user_path_test()
        # from src.gpt_langchain import get_some_dbs_from_hf
        # get_some_dbs_from_hf()

    max_seq_len_client = None
    if base_model in ['h2oai/h2ogpt-oig-oasst1-512-6_9b', 'h2oai/h2ogpt-oasst1-512-12b']:
        prompt_type = PromptType.human_bot.name
    elif base_model in ['h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2']:
        prompt_type = PromptType.prompt_answer.name
    elif base_model in ['llama']:
        max_seq_len_client = 2048
        prompt_type = PromptType.llama2.name
    elif base_model in ['gptj']:
        max_seq_len_client = 2048
        prompt_type = PromptType.gptj.name
    else:
        raise NotImplementedError(base_model)

    main_kwargs = dict(base_model=base_model, prompt_type=prompt_type, chat=True,
                       stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
                       max_new_tokens=max_new_tokens,
                       langchain_mode=langchain_mode, langchain_action=langchain_action,
                       langchain_agents=langchain_agents,
                       user_path=user_path,
                       langchain_modes=langchain_modes,
                       docs_ordering_type=docs_ordering_type,
                       force_langchain_evaluate=force_langchain_evaluate,
                       system_prompt='',
                       verbose=True)

    # inference server
    from src.gen import main
    main(**main_kwargs)
    inference_server = get_inf_server()
    inf_port = get_inf_port()

    # server that consumes inference server has different port
    from src.gen import main
    client_port = inf_port + 2  # assume will not use +  2 in testing, + 1 reserved for non-gradio inference servers
    # only case when GRADIO_SERVER_PORT and HOST should appear in tests because using 2 gradio instances
    os.environ['GRADIO_SERVER_PORT'] = str(client_port)
    os.environ['HOST'] = "http://127.0.0.1:%s" % client_port

    h2ogpt_key = 'foodoo#'
    main_kwargs = main_kwargs.copy()
    if enforce_h2ogpt_api_key:
        main_kwargs.update(dict(enforce_h2ogpt_api_key=True, h2ogpt_api_keys=[h2ogpt_key]))
    main_kwargs.update(dict(max_seq_len=max_seq_len_client))
    main(**main_kwargs, inference_server=inference_server)

    # client test to server that only consumes inference server
    from src.client_test import run_client_chat
    res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''

    # will use HOST from above
    if enforce_h2ogpt_api_key:
        # try without key first
        ret1, ret2, ret3, ret4, ret5, ret6, ret7 = run_client_many(prompt_type=None)
        assert 'Invalid Access Key' in ret1['response']
        assert 'Invalid Access Key' in ret2['response']
        assert 'Invalid Access Key' in ret3['response']
        assert 'Invalid Access Key' in ret4['response']
        assert 'Invalid Access Key' in ret5['response']
        assert 'Invalid Access Key' in ret6['response']
        assert 'Invalid Access Key' in ret7['response']
        ret1, ret2, ret3, ret4, ret5, ret6, ret7 = run_client_many(prompt_type=None, h2ogpt_key='foo')
        assert 'Invalid Access Key' in ret1['response']
        assert 'Invalid Access Key' in ret2['response']
        assert 'Invalid Access Key' in ret3['response']
        assert 'Invalid Access Key' in ret4['response']
        assert 'Invalid Access Key' in ret5['response']
        assert 'Invalid Access Key' in ret6['response']
        assert 'Invalid Access Key' in ret7['response']

    # try normal or with key if enforcing
    ret1, ret2, ret3, ret4, ret5, ret6, ret7 = run_client_many(prompt_type=None,
                                                               h2ogpt_key=h2ogpt_key)  # client shouldn't have to specify
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
               'I am a helpful assistant' in ret1['response'] or \
               'I am a chatbot.' in ret1['response'] or \
               'a chat-based assistant that can answer questions' in ret1['response'] or \
               'I am an AI language model' in ret1['response'] or \
               'I am an AI assistant.' in ret1['response']
        assert 'Once upon a time' in ret2['response']
        assert 'Once upon a time' in ret3['response']
        assert 'I am a language model trained' in ret4['response'] or 'I am a helpful assistant' in \
               ret4['response'] or 'I am a chatbot.' in ret4['response'] or \
               'a chat-based assistant that can answer questions' in ret4['response'] or \
               'I am an AI language model' in ret4['response'] or \
               'I am an AI assistant.' in ret4['response']
        assert 'I am a language model trained' in ret5['response'] or 'I am a helpful assistant' in \
               ret5['response'] or 'I am a chatbot.' in ret5['response'] or \
               'a chat-based assistant that can answer questions' in ret5['response'] or \
               'I am an AI language model' in ret5['response'] or \
               'I am an AI assistant.' in ret5['response']
        assert 'I am a language model trained' in ret6['response'] or 'I am a helpful assistant' in \
               ret6['response'] or 'I am a chatbot.' in ret6['response'] or \
               'a chat-based assistant that can answer questions' in ret6['response'] or \
               'I am an AI language model' in ret6['response'] or \
               'I am an AI assistant.' in ret6['response']
        assert 'I am a language model trained' in ret7['response'] or 'I am a helpful assistant' in \
               ret7['response'] or 'I am a chatbot.' in ret7['response'] or \
               'a chat-based assistant that can answer questions' in ret7['response'] or \
               'I am an AI language model' in ret7['response'] or \
               'I am an AI assistant.' in ret7['response']
    elif base_model == 'llama':
        assert 'I am a bot.' in ret1['response'] or 'can I assist you today?' in ret1[
            'response'] or 'How can I assist you?' in ret1['response'] or "I'm LLaMA" in ret1['response']
        assert 'Birds' in ret2['response'] or 'Once upon a time' in ret2['response']
        assert 'Birds' in ret3['response'] or 'Once upon a time' in ret3['response']
        assert 'I am a bot.' in ret4['response'] or 'can I assist you today?' in ret4[
            'response'] or 'How can I assist you?' in ret4['response'] or "I'm LLaMA" in ret4['response']
        assert 'I am a bot.' in ret5['response'] or 'can I assist you today?' in ret5[
            'response'] or 'How can I assist you?' in ret5['response'] or "I'm LLaMA" in ret5['response']
        assert 'I am a bot.' in ret6['response'] or 'can I assist you today?' in ret6[
            'response'] or 'How can I assist you?' in ret6['response'] or "I'm LLaMA" in ret6['response']
        assert 'I am a bot.' in ret7['response'] or 'can I assist you today?' in ret7[
            'response'] or 'How can I assist you?' in ret7['response'] or "I'm LLaMA" in ret7['response']
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


def run_docker(inf_port, base_model, low_mem_mode=False, do_shared=True):
    datetime_str = str(datetime.now()).replace(" ", "_").replace(":", "_")
    msg = "Starting HF inference %s..." % datetime_str
    print(msg, flush=True)
    home_dir = os.path.expanduser('~')
    makedirs(os.path.join(home_dir, '.cache/huggingface/hub'))
    data_dir = '%s/.cache/huggingface/hub/' % home_dir
    n_gpus = get_ngpus_vis()
    cmd = ["docker"] + ['run',
                        '-d',
                        '--runtime', 'nvidia',
                        ] + gpus_cmd() + [
              '--shm-size', '1g',
              '-e', 'HUGGING_FACE_HUB_TOKEN=%s' % os.environ['HUGGING_FACE_HUB_TOKEN'],
              '-p', '%s:80' % inf_port,
              '-v', '%s/.cache/huggingface/hub/:/data' % home_dir,
              '-v', '%s:/data' % data_dir,
              'ghcr.io/huggingface/text-generation-inference:0.9.3',
              '--model-id', base_model,
              '--max-stop-sequences', '6',
              '--sharded', 'false' if n_gpus == 1 or not do_shared else 'true'
          ]
    if n_gpus > 1 and do_shared:
        cmd.extend(['--num-shard', '%s' % n_gpus])
    if low_mem_mode:
        cmd.extend(['--max-input-length', '1024',
                    '--max-total-tokens', '2048',
                    # '--cuda-memory-fraction', '0.3',  # for 0.9.4, but too memory hungry
                    ])
    else:
        cmd.extend(['--max-input-length', '4096',
                    '--max-total-tokens', '8192',
                    # '--cuda-memory-fraction', '0.8',  # for 0.9.4, but too memory hungry
                    ])

    print(cmd, flush=True)
    docker_hash = subprocess.check_output(cmd).decode().strip()
    import time
    connected = False
    while not connected:
        cmd = 'docker logs %s' % docker_hash
        o = subprocess.check_output(cmd, shell=True, timeout=15)
        connected = 'Connected' in o.decode("utf-8")
        time.sleep(5)
    print("Done starting TGI server: %s" % docker_hash, flush=True)
    return docker_hash


def gpus_cmd():
    n_gpus = get_ngpus_vis()
    if n_gpus == 1:
        return ['--gpus', 'device=%d' % int(os.getenv('CUDA_VISIBLE_DEVICES', '0'))]
    elif n_gpus > 2:
        # note below if joined loses ' needed
        return ['--gpus', '\"device=%s\"' % os.getenv('CUDA_VISIBLE_DEVICES',
                                                      str(list(range(0, n_gpus))).replace(']', '').replace('[',
                                                                                                           '').replace(
                                                          ' ', '')
                                                      )]


def run_vllm_docker(inf_port, base_model, tokenizer=None):
    if base_model == 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2':
        # 7b has 71 heads, not divisible
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.system("docker pull gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0")
    datetime_str = str(datetime.now()).replace(" ", "_").replace(":", "_")
    msg = "Starting vLLM inference %s..." % datetime_str
    print(msg, flush=True)
    home_dir = os.path.expanduser('~')
    makedirs(os.path.join(home_dir, '.cache/huggingface/hub'))
    n_gpus = get_ngpus_vis()
    cmd = ["docker"] + ['run',
                        '-d',
                        '--runtime', 'nvidia',
                        ] + gpus_cmd() + [
              '--shm-size', '10.24g',
              '-e', 'HUGGING_FACE_HUB_TOKEN=%s' % os.environ['HUGGING_FACE_HUB_TOKEN'],
              '-p', '%s:5000' % inf_port,
              '--entrypoint', '/h2ogpt_conda/vllm_env/bin/python3.10',
              '-e', 'NCCL_IGNORE_DISABLED_P2P=1',
              '-v', '/etc/passwd:/etc/passwd:ro',
              '-v', '/etc/group:/etc/group:ro',
              '-u', '%s:%s' % (os.getuid(), os.getgid()),
              '-v', '%s/.cache:/workspace/.cache' % home_dir,
              # '--network', 'host',
              'gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0',
              # 'h2ogpt',  # use when built locally with vLLM just freshly added
              # 'docker.io/library/h2ogpt',  # use when built locally with vLLM just freshly added
              '-m', 'vllm.entrypoints.openai.api_server',
              '--port=5000',
              '--host=0.0.0.0',
                    '--model=%s' % base_model,
                    '--tensor-parallel-size=%s' % n_gpus,
              '--seed', '1234',
              '--trust-remote-code',
              '--download-dir=/workspace/.cache/huggingface/hub',
          ]
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    if tokenizer:
        cmd.append('--tokenizer=%s' % tokenizer)

    print(cmd, flush=True)
    print(' '.join(cmd), flush=True)
    docker_hash = subprocess.check_output(cmd).decode().strip()
    import time
    connected = False
    while not connected:
        cmd = 'docker logs %s' % docker_hash
        o = subprocess.check_output(cmd, shell=True, timeout=15)
        connected = 'Uvicorn running on' in o.decode("utf-8")
        # somehow above message doesn't come up
        connected |= 'GPU blocks' in o.decode("utf-8")
        time.sleep(5)
    print("Done starting vLLM server: %s" % docker_hash, flush=True)
    return docker_hash


def run_h2ogpt_docker(port, base_model, inference_server=None, max_new_tokens=None):
    os.system("docker pull gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0")
    datetime_str = str(datetime.now()).replace(" ", "_").replace(":", "_")
    msg = "Starting h2oGPT %s..." % datetime_str
    print(msg, flush=True)
    home_dir = os.path.expanduser('~')
    makedirs(os.path.join(home_dir, '.cache/huggingface/hub'))
    makedirs(os.path.join(home_dir, 'save'))
    cmd = ["docker"] + ['run',
                        '-d',
                        '--runtime', 'nvidia',
                        ] + gpus_cmd() + [
              '--shm-size', '1g',
              '-p', '%s:7860' % port,
              '-v', '%s/.cache:/workspace/.cache/' % home_dir,
              '-v', '%s/save:/workspace/save' % home_dir,
              '-v', '/etc/passwd:/etc/passwd:ro',
              '-v', '/etc/group:/etc/group:ro',
              '-u', '%s:%s' % (os.getuid(), os.getgid()),
              '-e', 'HUGGING_FACE_HUB_TOKEN=%s' % os.environ['HUGGING_FACE_HUB_TOKEN'],
              '--network', 'host',
              'gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0',
              # 'h2ogpt',  # use when built locally with vLLM just freshly added
              '/workspace/generate.py',
                    '--base_model=%s' % base_model,
              '--use_safetensors=True',
              '--save_dir=/workspace/save/',
              '--score_model=None',
                    '--max_max_new_tokens=%s' % (max_new_tokens or 2048),
                    '--max_new_tokens=%s' % (max_new_tokens or 1024),
              '--num_async=10',
              '--num_beams=1',
              '--top_k_docs=-1',
              '--chat=True',
              '--stream_output=True',
              # '--debug=True',
          ]

    if inference_server:
        cmd.extend(['--inference_server=%s' % inference_server])

    print(cmd, flush=True)
    docker_hash = subprocess.check_output(cmd).decode().strip()
    print("Done starting h2oGPT server: %s" % docker_hash, flush=True)
    return docker_hash


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
                             langchain_modes=['UserData', 'MyData', 'LLM', 'Disabled'],
                             docs_ordering_type='reverse_sort'):
    # HF inference server
    gradio_port = get_inf_port()
    inf_port = gradio_port + 1
    inference_server = 'http://127.0.0.1:%s' % inf_port
    docker_hash = run_docker(inf_port, base_model, low_mem_mode=True, do_shared=False)

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
                       system_prompt='',
                       stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
                       max_new_tokens=max_new_tokens,
                       langchain_mode=langchain_mode,
                       langchain_action=langchain_action,
                       langchain_agents=langchain_agents,
                       user_path=user_path,
                       langchain_modes=langchain_modes,
                       docs_ordering_type=docs_ordering_type,
                       force_langchain_evaluate=force_langchain_evaluate,
                       inference_server=inference_server,
                       model_lock=model_lock)

    try:
        # server that consumes inference server
        from src.gen import main
        main(**main_kwargs)

        # client test to server that only consumes inference server
        from src.client_test import run_client_chat
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
            assert 'year old student from the' in ret1['response'] or \
                   'I am a person who is asking you a question' in ret1['response'] or \
                   'year old' in ret1['response'] or \
                   'AI language model' in ret1['response']
            assert 'bird' in ret2['response']
            assert 'bird' in ret3['response']
            assert 'year old student from the' in ret4['response'] or 'I am a person who is asking you a question' in \
                   ret4['response'] or 'year old' in ret4['response']
            assert 'year old student from the' in ret5['response'] or 'I am a person who is asking you a question' in \
                   ret5['response'] or 'year old' in ret5['response']
            assert 'year old student from the' in ret6['response'] or 'I am a person who is asking you a question' in \
                   ret6['response'] or 'year old' in ret6['response']
            assert 'year old student from the' in ret7['response'] or 'I am a person who is asking you a question' in \
                   ret7['response'] or 'year old' in ret7['response']
        elif base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
            assert 'h2oGPT' in ret1['response']
            assert 'Birds' in ret2['response']
            assert 'Birds' in ret3['response']
            assert 'h2oGPT' in ret4['response']
            assert 'h2oGPT' in ret5['response']
            assert 'h2oGPT' in ret6['response']
            assert 'h2oGPT' in ret7['response']
        else:
            assert 'I am a language model trained' in ret1['response'] or 'I am a helpful assistant' in \
                   ret1['response'] or 'a chat-based assistant' in ret1['response'] or 'am a student' in ret1[
                       'response'] or 'I am an AI language model' in ret1['response']
            assert 'Once upon a time' in ret2['response']
            assert 'Once upon a time' in ret3['response']
            assert 'I am a language model trained' in ret4['response'] or 'I am a helpful assistant' in \
                   ret4['response'] or 'a chat-based assistant' in ret4['response'] or 'am a student' in ret4[
                       'response'] or 'I am an AI language model' in ret4['response']
            assert 'I am a language model trained' in ret5['response'] or 'I am a helpful assistant' in \
                   ret5['response'] or 'a chat-based assistant' in ret5['response'] or 'am a student' in ret5[
                       'response'] or 'I am an AI language model' in ret5['response']
            assert 'I am a language model trained' in ret6['response'] or 'I am a helpful assistant' in \
                   ret6['response'] or 'a chat-based assistant' in ret6['response'] or 'am a student' in ret6[
                       'response'] or 'I am an AI language model' in ret6['response']
            assert 'I am a language model trained' in ret7['response'] or 'I am a helpful assistant' in \
                   ret7['response'] or 'a chat-based assistant' in ret7['response'] or 'am a student' in ret7[
                       'response'] or 'I am an AI language model' in ret7['response']
        print("DONE", flush=True)
    finally:
        os.system("docker stop %s" % docker_hash)


chat_conversation1 = [['Who are you?',
                       'I am an AI language model created by OpenAI, designed to assist with various tasks such as answering questions, generating text, and providing information.']]


@pytest.mark.skipif(not have_openai_key, reason="requires OpenAI key to run")
@pytest.mark.parametrize("system_prompt", ['You are a baby cat who likes to talk to people.', ''])
@pytest.mark.parametrize("chat_conversation", [chat_conversation1, []])
@pytest.mark.parametrize("force_langchain_evaluate", [False, True])
@pytest.mark.parametrize("inference_server", ['openai_chat', 'openai_azure_chat'])
@wrap_test_forked
def test_openai_inference_server(inference_server, force_langchain_evaluate, chat_conversation,
                                 system_prompt,
                                 prompt='Who are you?', stream_output=False, max_new_tokens=256,
                                 base_model='gpt-3.5-turbo',
                                 langchain_mode='Disabled',
                                 langchain_action=LangChainAction.QUERY.value,
                                 langchain_agents=[],
                                 user_path=None,
                                 langchain_modes=['UserData', 'MyData', 'LLM', 'Disabled'],
                                 docs_ordering_type='reverse_sort'):
    if force_langchain_evaluate:
        langchain_mode = 'MyData'
    if inference_server == 'openai_azure_chat':
        # need at least deployment name added:
        deployment_name = 'h2ogpt'
        inference_server += ':%s:%s' % (deployment_name, 'h2ogpt.openai.azure.com/')
    if 'azure' in inference_server:
        assert 'OPENAI_AZURE_KEY' in os.environ, "Missing 'OPENAI_AZURE_KEY'"
        os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_AZURE_KEY']

    main_kwargs = dict(base_model=base_model, chat=True,
                       stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
                       max_new_tokens=max_new_tokens,
                       langchain_mode=langchain_mode,
                       langchain_action=langchain_action,
                       langchain_agents=langchain_agents,
                       user_path=user_path,
                       langchain_modes=langchain_modes,
                       system_prompt='auto',
                       docs_ordering_type=docs_ordering_type,
                       # chat_conversation=chat_conversation # not enough if API passes [], API will override
                       )

    # server that consumes inference server
    from src.gen import main
    main(**main_kwargs, inference_server=inference_server)

    if chat_conversation:
        prompt = 'What did I ask?'

    # client test to server that only consumes inference server
    from src.client_test import run_client_chat
    res_dict, client = run_client_chat(prompt=prompt, prompt_type='openai_chat', stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents,
                                       chat_conversation=chat_conversation,
                                       system_prompt=system_prompt)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''

    if chat_conversation and system_prompt:
        # TODO: don't check yet, system_prompt ignored if response from LLM is as if no system prompt
        return

    if chat_conversation or system_prompt:
        ret6, _ = test_client_basic_api_lean(prompt=prompt, prompt_type=None,
                                             chat_conversation=chat_conversation,
                                             system_prompt=system_prompt)
        if system_prompt:
            assert 'baby cat' in res_dict['response'] and 'meow' in res_dict['response'].lower()
            assert 'baby cat' in ret6['response'] and 'meow' in ret6['response'].lower()
        else:
            options_response = ['You asked "Who are you?"', """You asked, \"Who are you?\""""]
            assert res_dict['response'] in options_response
            assert ret6['response'] in options_response

        return

    if system_prompt:
        # don't test rest, too many cases
        return

    # will use HOST from above
    ret1, ret2, ret3, ret4, ret5, ret6, ret7 = run_client_many(prompt_type=None)  # client shouldn't have to specify
    assert 'I am an AI language model' in ret1['response'] or 'I am a helpful assistant designed' in ret1[
        'response'] or 'I am an AI assistant designed to help answer questions and provide information' in ret1[
               'response']
    assert 'Once upon a time, in a far-off land,' in ret2['response'] or 'Once upon a time' in ret2['response']
    assert 'Once upon a time, in a far-off land,' in ret3['response'] or 'Once upon a time' in ret3['response']
    assert 'I am an AI language model' in ret4['response'] or 'I am a helpful assistant designed' in ret4[
        'response'] or 'I am an AI assistant designed to help answer questions and provide information' in ret4[
               'response']
    assert 'I am an AI language model' in ret5['response'] or 'I am a helpful assistant designed' in ret5[
        'response'] or 'I am an AI assistant designed to help answer questions and provide information' in ret5[
               'response']
    assert 'I am an AI language model' in ret6['response'] or 'I am a helpful assistant designed' in ret6[
        'response'] or 'I am an AI assistant designed to help answer questions and provide information' in ret6[
               'response']
    assert 'I am an AI language model' in ret7['response'] or 'I am a helpful assistant designed' in ret7[
        'response'] or 'I am an AI assistant designed to help answer questions and provide information' in ret7[
               'response']
    print("DONE", flush=True)


@pytest.mark.parametrize("base_model",
                         ['h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2', 'meta-llama/Llama-2-7b-chat-hf']
                         )
@wrap_test_forked
def test_gradio_tgi_docker(base_model):
    # HF inference server
    gradio_port = get_inf_port()
    inf_port = gradio_port + 1
    inference_server = 'http://127.0.0.1:%s' % inf_port
    docker_hash1 = run_docker(inf_port, base_model, low_mem_mode=True, do_shared=False)
    os.system('docker logs %s | tail -10' % docker_hash1)

    # h2oGPT server
    docker_hash2 = run_h2ogpt_docker(gradio_port, base_model, inference_server=inference_server)
    time.sleep(30)  # assumes image already downloaded, else need more time
    os.system('docker logs %s | tail -10' % docker_hash2)

    # test this version for now, until docker updated
    version = 1

    try:
        # client test to server that only consumes inference server
        prompt = 'Who are you?'
        print("Starting client tests with prompt: %s using %s" % (prompt, get_inf_server()))
        from src.client_test import run_client_chat
        res_dict, client = run_client_chat(prompt=prompt,
                                           stream_output=True,
                                           max_new_tokens=256,
                                           langchain_mode='Disabled',
                                           langchain_action=LangChainAction.QUERY.value,
                                           langchain_agents=[],
                                           version=version)
        assert res_dict['prompt'] == prompt
        assert res_dict['iinput'] == ''

        # will use HOST from above
        # client shouldn't have to specify
        ret1, ret2, ret3, ret4, ret5, ret6, ret7 = run_client_many(prompt_type=None, version=version)
        if 'llama' in base_model.lower():
            who = "I'm LLaMA, an AI assistant developed by Meta AI"
            assert who in ret1['response']
            assert who in ret1['response']
            assert 'Once upon a time' in ret2['response']
            assert 'Once upon a time' in ret3['response']
            assert who in ret4['response']
            assert who in ret5['response']
            assert who in ret6['response']
            assert who in ret7['response']
        else:
            who = 'I am an AI language model'
            assert who in ret1['response']
            assert 'Once upon a time' in ret2['response']
            assert 'Once upon a time' in ret3['response']
            assert who in ret4['response']
            assert who in ret5['response']
            assert who in ret6['response']
            assert who in ret7['response']
        print("DONE", flush=True)
    finally:
        os.system("docker stop %s" % docker_hash1)
        os.system("docker stop %s" % docker_hash2)


@pytest.mark.parametrize("base_model",
                         [
                             'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2',
                             'h2oai/h2ogpt-4096-llama2-7b-chat']  # avoid meta to avoid hassle of key
                         )
@wrap_test_forked
def test_gradio_vllm_docker(base_model):
    # HF inference server
    gradio_port = get_inf_port()
    inf_port = gradio_port + 1
    inference_server = 'vllm:127.0.0.1:%s' % inf_port
    if 'llama' in base_model:
        tokenizer = 'hf-internal-testing/llama-tokenizer'
    else:
        tokenizer = None

    docker_hash1 = run_vllm_docker(inf_port, base_model, tokenizer)
    os.system('docker logs %s | tail -10' % docker_hash1)

    # h2oGPT server
    docker_hash2 = run_h2ogpt_docker(gradio_port, base_model, inference_server=inference_server)
    time.sleep(30)  # assumes image already downloaded, else need more time
    os.system('docker logs %s | tail -10' % docker_hash2)

    # test this version for now, until docker updated
    version = 1

    try:
        # client test to server that only consumes inference server
        prompt = 'Who are you?'
        print("Starting client tests with prompt: %s using %s" % (prompt, get_inf_server()))
        from src.client_test import run_client_chat
        res_dict, client = run_client_chat(prompt=prompt,
                                           stream_output=True,
                                           max_new_tokens=256,
                                           langchain_mode='Disabled',
                                           langchain_action=LangChainAction.QUERY.value,
                                           langchain_agents=[],
                                           version=version)
        assert res_dict['prompt'] == prompt
        assert res_dict['iinput'] == ''

        # will use HOST from above
        # client shouldn't have to specify
        ret1, ret2, ret3, ret4, ret5, ret6, ret7 = run_client_many(prompt_type=None, version=version)
        if 'llama' in base_model.lower():
            who = "I'm LLaMA, an AI assistant developed by Meta AI"
            assert who in ret1['response']
            assert who in ret1['response']
            assert 'Once upon a time' in ret2['response']
            assert 'Once upon a time' in ret3['response']
            assert who in ret4['response']
            assert who in ret5['response']
            assert who in ret6['response']
            assert who in ret7['response']
        else:
            who = 'I am an AI language model'
            assert who in ret1['response']
            assert 'Once upon a time' in ret2['response']
            assert 'Once upon a time' in ret3['response']
            assert who in ret4['response']
            assert who in ret5['response']
            assert who in ret6['response']
            assert who in ret7['response']
        print("DONE", flush=True)
    finally:
        os.system("docker stop %s" % docker_hash1)
        os.system("docker stop %s" % docker_hash2)


@pytest.mark.skipif(not have_replicate_key, reason="requires Replicate key to run")
@pytest.mark.parametrize("system_prompt", ['You are a baby cat who likes to talk to people.', ''])
@pytest.mark.parametrize("chat_conversation", [chat_conversation1, []])
@pytest.mark.parametrize("force_langchain_evaluate", [False, True])
@wrap_test_forked
def test_replicate_inference_server(force_langchain_evaluate,
                                    chat_conversation,
                                    system_prompt,
                                    prompt='Who are you?', stream_output=False,
                                    max_new_tokens=128,  # limit cost
                                    base_model='h2oai/h2ogpt-4096-llama2-7b-chat',
                                    langchain_mode='Disabled',
                                    langchain_action=LangChainAction.QUERY.value,
                                    langchain_agents=[],
                                    user_path=None,
                                    langchain_modes=['UserData', 'MyData', 'LLM', 'Disabled'],
                                    docs_ordering_type='reverse_sort'):
    if force_langchain_evaluate:
        langchain_mode = 'MyData'

    main_kwargs = dict(base_model=base_model, chat=True,
                       stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
                       max_new_tokens=max_new_tokens,
                       langchain_mode=langchain_mode,
                       langchain_action=langchain_action,
                       langchain_agents=langchain_agents,
                       user_path=user_path,
                       langchain_modes=langchain_modes,
                       docs_ordering_type=docs_ordering_type)

    # server that consumes inference server
    from src.gen import main
    # https://replicate.com/lucataco/llama-2-7b-chat
    #model_string = "lucataco/llama-2-7b-chat:6ab580ab4eef2c2b440f2441ec0fc0ace5470edaf2cbea50b8550aec0b3fbd38"
    model_string = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
    main(**main_kwargs, inference_server='replicate:%s' % model_string)

    if chat_conversation:
        prompt = 'What did I ask?'

    # client test to server that only consumes inference server
    from src.client_test import run_client_chat
    res_dict, client = run_client_chat(prompt=prompt, prompt_type='llama2', stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents,
                                       chat_conversation=chat_conversation,
                                       system_prompt=system_prompt)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''

    if chat_conversation and system_prompt:
        # TODO: don't check yet, system_prompt ignored if response from LLM is as if no system prompt
        return

    if chat_conversation or system_prompt:
        ret6, _ = test_client_basic_api_lean(prompt=prompt, prompt_type=None,
                                             chat_conversation=chat_conversation,
                                             system_prompt=system_prompt)
        if system_prompt:
            assert 'baby cat' in res_dict['response'] and 'meow' in res_dict['response'].lower()
            assert 'baby cat' in ret6['response'] and 'meow' in ret6['response'].lower()
        else:
            options_response = ['You asked "Who are you?"',
                                """You asked, \"Who are you?\"""",
                                """You asked: \"Who are you?\"""",
                                ]
            assert res_dict['response'] in options_response
            assert ret6['response'] in options_response

        return

    if system_prompt:
        # don't test rest, too many cases
        return

    # will use HOST from above
    ret1, ret2, ret3, ret4, ret5, ret6, ret7 = run_client_many(prompt_type=None)  # client shouldn't have to specify
    who = 'an AI assistant'
    who2 = 'just an AI'
    assert who in ret1['response'] or who2 in ret1['response']
    assert 'Once upon a time, in a far-off land,' in ret2['response'] or 'Once upon a time' in ret2['response']
    assert 'Once upon a time, in a far-off land,' in ret3['response'] or 'Once upon a time' in ret3['response']
    assert who in ret4['response'] or 'I am a helpful assistant designed' in ret4['response'] or who2 in ret4['response']
    assert who in ret5['response'] or 'I am a helpful assistant designed' in ret5['response'] or who2 in ret5['response']
    assert who in ret6['response'] or 'I am a helpful assistant designed' in ret6['response'] or who2 in ret6['response']
    assert who in ret7['response'] or 'I am a helpful assistant designed' in ret7['response'] or who2 in ret7['response']
    print("DONE", flush=True)
