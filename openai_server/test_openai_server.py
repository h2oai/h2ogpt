import json
import shutil
import sys
import time

import pytest
import os
import ast

# to avoid copy-paste, only other external reference besides main() (for local_server=True)
from tests.utils import wrap_test_forked


def launch_openai_server():
    from openai_server.server_start import run
    from openai_server.server import app as openai_app
    run(is_openai_server=True, workers=1, app=openai_app)


def test_openai_server():
    # for manual separate OpenAI server on existing h2oGPT, run:
    # Shell 1: CUDA_VISIBLE_DEVICES=0 python generate.py --verbose=True --score_model=None --pre_load_embedding_model=False --gradio_offline_level=2 --base_model=h2oai/h2o-danube2-1.8b-chat --inference_server=vllm:ip:port --max_seq_len=4096 --save_dir=duder1 --verbose --concurrency_count=64 --openai_server=False --add_disk_models_to_ui=False
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
    openai_workers = 1
    run_openai_client(stream_output, chat, local_server, openai_workers, prompt, api_key, enforce_h2ogpt_api_key,
                      repeat)


@pytest.mark.parametrize("stream_output", [False, True])
@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("local_server", [True])  # choose False if start local server
@pytest.mark.parametrize("openai_workers", [1, 0])  # choose 0 to test multi-worker case
@pytest.mark.parametrize("prompt", ["Who are you?", "Tell a very long kid's story about birds."])
@pytest.mark.parametrize("api_key", [None, "EMPTY", os.environ.get('H2OGPT_H2OGPT_KEY', 'EMPTY')])
@pytest.mark.parametrize("enforce_h2ogpt_api_key", [False, True])
@pytest.mark.parametrize("repeat", list(range(0, repeat0)))
@wrap_test_forked
def test_openai_client(stream_output, chat, local_server, openai_workers, prompt, api_key, enforce_h2ogpt_api_key,
                       repeat):
    run_openai_client(stream_output, chat, local_server, openai_workers, prompt, api_key, enforce_h2ogpt_api_key,
                      repeat)


def run_openai_client(stream_output, chat, local_server, openai_workers, prompt, api_key, enforce_h2ogpt_api_key,
                      repeat):
    base_model = 'h2oai/h2o-danube2-1.8b-chat'
    # base_model = 'gemini-pro'

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
             openai_workers=openai_workers,
             )
        time.sleep(10)
    else:
        # RUN something
        # e.g. CUDA_VISIBLE_DEVICES=0 python generate.py --verbose=True --score_model=None --gradio_offline_level=2 --base_model=h2oai/h2o-danube2-1.8b-chat --inference_server=vllm:IP:port --max_seq_len=4096 --save_dir=duder1 --verbose --openai_server=True --concurency_count=64
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
                  prompt, client_kwargs, stream_output, verbose, base_model)
    except AssertionError as e:
        if enforce_h2ogpt_api_key and api_key is None:
            print("Expected to fail since no key but enforcing.")
        else:
            raise AssertionError(str(e))
    except Exception as e:
        raise RuntimeError(str(e))

    # MODELS
    model_info = openai_client.models.retrieve(base_model)
    assert model_info.base_model == base_model
    model_list = openai_client.models.list()
    assert model_list.data[0].id == base_model

    os.system('pkill -f server_start.py --signal 9')
    os.system('pkill -f "h2ogpt/bin/python -c from multiprocessing" --signal 9')


def test_chat(chat, openai_client, async_client, system_prompt, chat_conversation, add_chat_history_to_context,
              prompt, client_kwargs, stream_output, verbose, base_model):
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

    if base_model == 'gemini-pro':
        if "Who" in prompt:
            assert 'Google' in text or 'model' in text
        else:
            assert 'birds' in text
    else:
        if "Who" in prompt:
            assert 'OpenAI' in text or 'chatbot' in text or 'model' in text or 'AI' in text
        else:
            assert 'birds' in text


def show_plot(text):
    # We can see the output scatter.png and the code file generated by the agent.
    pattern = r'<files>(.*?)</files>'
    # re.DOTALL allows dot (.) to match newlines as well
    import re
    files = ast.literal_eval(re.findall(pattern, text, re.DOTALL)[0])
    images = [x for x in files if x.endswith('.png') or x.endswith('.jpeg')]

    print(files)
    print(images)

    from PIL import Image
    im = Image.open(images[0])
    im.show()


def test_autogen():
    from openai import OpenAI

    client = OpenAI(base_url='http://0.0.0.0:5004/v1')

    # prompt = "2+2="
    import datetime
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt = f"Today is {today}.  Write Python code to plot TSLA's and META's stock price gains YTD vs. time per week, and save the plot to a file named 'stock_gains.png'."

    # vllm_chat:

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    # model = "mistralai/Mistral-7B-Instruct-v0.3"
    model = "gpt-4o"

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=2048,
        extra_body=dict(use_autogen=True),
    )

    text = response.choices[0].message.content

    print(text)
    show_plot(text)

    image_file_id = "file-abc-123"
    image_file = client.files.content(image_file_id)
    with open("plot.png", "wb") as f:
        f.write(image_file.content)

    # streaming:

    responses = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=4096,
        extra_body=dict(use_autogen=True),
    )

    text = ''
    for chunk in responses:
        delta = chunk.choices[0].delta.content
        if delta:
            text += delta
            print(delta)

    print(text)
    show_plot(text)

    ####

    # text completion:

    responses = client.completions.create(
        model=model,
        # response_format=dict(type=response_format),  Text Completions API can't handle
        prompt=prompt,
        stream=False,
        max_tokens=4096,
        extra_body=dict(use_autogen=True),
    )
    text = responses.choices[0].text

    print(text)
    show_plot(text)

    # streaming text completion:

    responses = client.completions.create(
        model=model,
        # response_format=dict(type=response_format),  Text Completions API can't handle
        prompt=prompt,
        stream=True,
        max_tokens=4096,
        extra_body=dict(use_autogen=True),
    )

    collected_events = []
    for event in responses:
        collected_events.append(event)  # save the event response
        delta = event.choices[0].text  # extract the text
        text += delta  # append the text
        if delta:
            print(delta)

    print(text)
    show_plot(text)


@pytest.fixture(scope="module")
def test_file():
    base_path = os.getenv('H2OGPT_OPENAI_BASE_FILE_PATH', './openai_files/')
    if base_path and base_path != './' and base_path != '.' and base_path != '/':
        shutil.rmtree(base_path)

    # Create a sample file for testing
    file_content = b"Sample file content"
    filename = "test_file.txt"
    with open(filename, "wb") as f:
        f.write(file_content)
    yield filename
    os.remove(filename)


def test_file_operations(test_file):
    api_key = "EMPTY"
    base_url = "http://0.0.0.0:5000/v1"
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)

    # Test file upload
    with open(test_file, "rb") as f:
        upload_response = client.files.create(file=f, purpose="assistants")
    print(upload_response)
    assert upload_response.id
    assert upload_response.object == "file"
    assert upload_response.purpose == "assistants"
    assert upload_response.created_at
    assert upload_response.bytes > 5
    assert upload_response.filename == "test_file.txt"

    file_id = upload_response.id

    # Test list files
    list_response = client.files.list().data
    assert isinstance(list_response, list)
    assert list_response[0].id == file_id
    assert list_response[0].object == "file"
    assert list_response[0].purpose == "assistants"
    assert list_response[0].created_at
    assert list_response[0].bytes > 5
    assert list_response[0].filename == "test_file.txt"

    # Test retrieve file
    retrieve_response = client.files.retrieve(file_id)
    assert retrieve_response.id == file_id
    assert retrieve_response.object == "file"

    # Test retrieve file content
    content = client.files.content(file_id)
    assert content == "Sample file content"

    # Test delete file
    delete_response = client.files.delete(file_id)
    assert delete_response.id == file_id
    assert delete_response.object == "file"
    assert delete_response.deleted is True


if __name__ == '__main__':
    launch_openai_server()
