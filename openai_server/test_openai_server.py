import time

import pytest
import os

# to avoid copy-paste, only other external reference besides main() (for local_server=True)
from tests.utils import wrap_test_forked


def launch_openai_server():
    from openai_server.server import run
    run()


def test_openai_server():
    # for manual separate OpenAI server on existing h2oGPT, run:
    # Shell 1: CUDA_VISIBLE_DEVICES=0 python generate.py --verbose=True --score_model=None --pre_load_embedding_model=False --gradio_offline_level=2 --base_model=openchat/openchat-3.5-1210 --inference_server=vllm:ip:port --max_seq_len=4096 --save_dir=duder1 --verbose --openai_server=True --concurrency_count=64 --openai_server=False
    # Shell 2: pytest -s -v openai_server/test_openai_server.py::test_openai_server  # once client done, hit CTRL-C, should pass
    # Shell 3: pytest -s -v openai_server/test_openai_server.py::test_openai_client_test2  # should pass
    launch_openai_server()


# repeat0 = 100  # e.g. to test concurrency
repeat0 = 1


@pytest.mark.parametrize("stream_output", [False, True])
@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("local_server", [False])
@wrap_test_forked
def test_openai_client_test2(stream_output, chat, local_server):
    prompt = "Who are you?"
    api_key = 'EMPTY'
    enforce_h2ogpt_api_key = False
    repeat = 1
    run_openai_client(stream_output, chat, local_server, prompt, api_key, enforce_h2ogpt_api_key, repeat)


@pytest.mark.parametrize("stream_output", [False, True])
@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("local_server", [True])
@pytest.mark.parametrize("prompt", ["Who are you?", "Tell a very long kid's story about birds."])
@pytest.mark.parametrize("api_key", [None, "EMPTY", os.environ.get('H2OGPT_H2OGPT_KEY', 'EMPTY')])
@pytest.mark.parametrize("enforce_h2ogpt_api_key", [False, True])
@pytest.mark.parametrize("repeat", list(range(0, repeat0)))
@wrap_test_forked
def test_openai_client(stream_output, chat, local_server, prompt, api_key, enforce_h2ogpt_api_key, repeat):
    run_openai_client(stream_output, chat, local_server, prompt, api_key, enforce_h2ogpt_api_key, repeat)


def run_openai_client(stream_output, chat, local_server, prompt, api_key, enforce_h2ogpt_api_key, repeat):
    base_model = 'openchat/openchat-3.5-1210'

    if local_server:
        from src.gen import main
        main(base_model=base_model, chat=False,
             stream_output=stream_output, gradio=True,
             num_beams=1, block_gradio_exit=False,
             add_disk_models_to_ui=False,
             enable_tts=False,
             enable_stt=False,
             enforce_h2ogpt_api_key=enforce_h2ogpt_api_key,
             # or use file with h2ogpt_api_keys=h2ogpt_api_keys.json
             h2ogpt_api_keys=[api_key] if api_key else None,
             )
        time.sleep(10)
    else:
        # RUN something
        # e.g. CUDA_VISIBLE_DEVICES=0 python generate.py --verbose=True --score_model=None --gradio_offline_level=2 --base_model=openchat/openchat-3.5-1210 --inference_server=vllm:IP:port --max_seq_len=4096 --save_dir=duder1 --verbose --openai_server=True --concurency_count=64
        pass

    # api_key = "EMPTY"  # if gradio/openai server not keyed.  Can't pass '' itself, leads to httpcore.LocalProtocolError: Illegal header value b'Bearer '
    # Setting H2OGPT_H2OGPT_KEY does not key h2oGPT, just passes along key to gradio inference server, so empty key is valid test regardless of the H2OGPT_H2OGPT_KEY value
    # api_key = os.environ.get('H2OGPT_H2OGPT_KEY', 'EMPTY')  # if keyed and have this in env with same key
    print('api_key: %s' % api_key)
    # below should be consistent with server prefix, host, and port
    base_url = 'http://localhost:5000/v1'
    verbose = True
    system_prompt = "You are a helpful assistant."
    chat_conversation = []
    add_chat_history_to_context = True

    client_kwargs = dict(model=base_model,
                         max_tokens=200,
                         stream=stream_output)

    from openai import OpenAI, AsyncOpenAI
    client_args = dict(base_url=base_url, api_key=api_key)
    openai_client = OpenAI(**client_args)
    async_client = AsyncOpenAI(**client_args)

    try:
        test_chat(chat, openai_client, async_client, system_prompt, chat_conversation, add_chat_history_to_context,
                  prompt, client_kwargs, stream_output, verbose)
    except AssertionError:
        if enforce_h2ogpt_api_key and api_key is None:
            print("Expected to fail since no key but enforcing.")
        else:
            raise

    # MODELS
    model_info = openai_client.models.retrieve(base_model)
    assert model_info.base_model == base_model
    model_list = openai_client.models.list()
    assert model_list.data[0] == base_model


def test_chat(chat, openai_client, async_client, system_prompt, chat_conversation, add_chat_history_to_context,
              prompt, client_kwargs, stream_output, verbose):
    # COMPLETION

    if chat:
        client = openai_client.chat.completions
        async_client = async_client.chat.completions

        messages0 = []
        if system_prompt:
            messages0.append({"role": "system", "content": system_prompt})
        if chat_conversation and add_chat_history_to_context:
            for message1 in chat_conversation:
                if len(message1) == 2:
                    messages0.append(
                        {'role': 'user', 'content': message1[0] if message1[0] is not None else ''})
                    messages0.append(
                        {'role': 'assistant', 'content': message1[1] if message1[1] is not None else ''})
        messages0.append({'role': 'user', 'content': prompt if prompt is not None else ''})

        client_kwargs.update(dict(messages=messages0))
    else:
        client = openai_client.completions
        async_client = async_client.completions

        client_kwargs.update(dict(prompt=prompt))

    responses = client.create(**client_kwargs)

    if not stream_output:
        if chat:
            text = responses.choices[0].message.content
        else:
            text = responses.choices[0].text
        print(text)
    else:
        collected_events = []
        text = ''
        for event in responses:
            collected_events.append(event)  # save the event response
            if chat:
                delta = event.choices[0].delta.content
            else:
                delta = event.choices[0].text  # extract the text
            text += delta  # append the text
            if verbose:
                print('delta: %s' % delta)
        print(text)

    if "Who" in prompt:
        assert 'OpenAI' in text or 'chatbot' in text
    else:
        assert 'birds' in text


if __name__ == '__main__':
    launch_openai_server()
