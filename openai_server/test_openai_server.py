import json
import shutil
import sys
import tempfile
import time
import uuid

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
    # for manual separate OpenAI server on existing h2oGPT, run (choose vllm:ip:port and/or base_model):
    # Shell 1: CUDA_VISIBLE_DEVICES=0 python generate.py --verbose=True --score_model=None --pre_load_embedding_model=False --gradio_offline_level=2 --base_model=h2oai/h2o-danube2-1.8b-chat --inference_server=vllm:ip:port --max_seq_len=4096 --save_dir=duder1 --verbose --concurrency_count=64 --openai_server=False --add_disk_models_to_ui=False
    # Shell 2: pytest -s -v openai_server/test_openai_server.py::test_openai_server  # once client done, hit CTRL-C, should pass
    # Shell 3: pytest -s -v openai_server/test_openai_server.py::test_openai_client_test2  # should pass
    # for rest of tests:
    # Shell 1: pytest -s -v openai_server/test_openai_server.py -k 'serverless or needs_server or has_server or serverless'
    launch_openai_server()


# repeat0 = 100  # e.g. to test concurrency
repeat0 = 1


@pytest.mark.needs_server
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


@pytest.mark.has_server
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
    # base_model = 'claude-3-5-sonnet-20240620'

    if local_server:
        from src.gen import main
        main(base_model=base_model,
             # inference_server='anthropic',
             chat=False,
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
        run_test_chat(chat, openai_client, async_client, system_prompt, chat_conversation, add_chat_history_to_context,
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
    assert model_info.id == base_model
    model_list = openai_client.models.list()
    assert base_model in [x.id for x in model_list.data]

    os.system('pkill -f server_start.py --signal 9')
    os.system('pkill -f "h2ogpt/bin/python -c from multiprocessing" --signal 9')


def run_test_chat(chat, openai_client, async_client, system_prompt, chat_conversation, add_chat_history_to_context,
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


def show_plot_from_ids(usage, client):
    if not hasattr(usage, 'file_ids') or not usage.file_ids:
        return None
    file_ids = usage.file_ids

    list_response = client.files.list().data
    assert isinstance(list_response, list)
    response_dict = {item.id: {key: value for key, value in dict(item).items() if key != 'id'} for item in
                     list_response}

    test_dir = 'openai_files_testing_%s' % str(uuid.uuid4())
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)
    files = []
    for file_id in file_ids:
        test_filename = os.path.join(test_dir, os.path.basename(response_dict[file_id]['filename']))
        content = client.files.content(file_id).content
        with open(test_filename, 'wb') as f:
            f.write(content)
        files.append(test_filename)

    images = [x for x in files if x.endswith('.png') or x.endswith('.jpeg')]

    print(files)
    print(images, file=sys.stderr)

    from PIL import Image
    im = Image.open(images[0])
    print("START SHOW IMAGE: %s" % images[0], file=sys.stderr)
    im.show()
    print("FINISH SHOW IMAGE", file=sys.stderr)
    return images


# NOTE: Should test with --force_streaming_on_to_handle_timeouts=False and --force_streaming_on_to_handle_timeouts=True
@pytest.mark.needs_server
def test_autogen():
    if os.path.exists('./openai_files'):
        shutil.rmtree('./openai_files')

    from openai import OpenAI

    client = OpenAI(base_url='http://0.0.0.0:5004/v1')

    # prompt = "2+2="
    import datetime
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    prompt = f"Today is {today}.  Write Python code to plot TSLA's and META's stock price gains YTD vs. time per week, and save the plot to a file named 'stock_gains.png'."

    print("chat non-streaming", file=sys.stderr)

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
        extra_body=dict(use_agent=True),
    )

    text = response.choices[0].message.content
    print(text, file=sys.stderr)
    assert show_plot_from_ids(response.usage, client) is not None

    print("chat streaming", file=sys.stderr)

    responses = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=4096,
        extra_body=dict(use_agent=True),
    )

    text = ''
    usages = []
    for chunk in responses:
        delta = chunk.choices[0].delta.content
        if chunk.usage is not None:
            usages.append(chunk.usage)
        if delta:
            text += delta
            print(delta, end='')

    print(text)
    assert len(usages) == 1
    assert show_plot_from_ids(usages[0], client) is not None

    ####

    print("text non-streaming", file=sys.stderr)

    responses = client.completions.create(
        model=model,
        # response_format=dict(type=response_format),  Text Completions API can't handle
        prompt=prompt,
        stream=False,
        max_tokens=4096,
        extra_body=dict(use_agent=True),
    )
    text = responses.choices[0].text

    print(text)
    assert show_plot_from_ids(responses.usage, client) is not None

    print("text streaming", file=sys.stderr)

    responses = client.completions.create(
        model=model,
        # response_format=dict(type=response_format),  Text Completions API can't handle
        prompt=prompt,
        stream=True,
        max_tokens=4096,
        extra_body=dict(use_agent=True),
    )

    collected_events = []
    usages = []
    for event in responses:
        collected_events.append(event)  # save the event response
        if event.usage is not None:
            usages.append(event.usage)
        delta = event.choices[0].text  # extract the text
        text += delta  # append the text
        if delta:
            print(delta, end='')

    print(text)
    assert len(usages) == 1
    assert show_plot_from_ids(usages[0], client) is not None


@pytest.fixture(scope="module")
def text_file():
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


@pytest.fixture(scope="module")
def pdf_file():
    base_path = os.getenv('H2OGPT_OPENAI_BASE_FILE_PATH', './openai_files/')
    if base_path and base_path != './' and base_path != '.' and base_path != '/':
        shutil.rmtree(base_path)

    # Create a sample file for testing
    filename = "test_file.pdf"
    shutil.copy('tests/2403.09629.pdf', filename)
    yield filename
    os.remove(filename)


@pytest.fixture(scope="module")
def image_file():
    base_path = os.getenv('H2OGPT_OPENAI_BASE_FILE_PATH', './openai_files/')
    if base_path and base_path != './' and base_path != '.' and base_path != '/':
        shutil.rmtree(base_path)

    # Create a sample file for testing
    filename = "test_file.png"
    shutil.copy('tests/dental.png', filename)
    yield filename
    os.remove(filename)


@pytest.fixture(scope="module")
def python_file():
    base_path = os.getenv('H2OGPT_OPENAI_BASE_FILE_PATH', './openai_files/')
    if base_path and base_path != './' and base_path != '.' and base_path != '/':
        shutil.rmtree(base_path)

    filename = "test_file.py"
    shutil.copy('src/gen.py', filename)
    yield filename
    os.remove(filename)


@pytest.fixture(scope="module")
def video_file():
    base_path = os.getenv('H2OGPT_OPENAI_BASE_FILE_PATH', './openai_files/')
    if base_path and base_path != './' and base_path != '.' and base_path != '/':
        shutil.rmtree(base_path)

    filename = "test_file.mp4"
    shutil.copy('tests/videotest.mp4', filename)
    yield filename
    os.remove(filename)


@pytest.mark.needs_server
@pytest.mark.parametrize("test_file", ["text_file", "pdf_file", "image_file", "python_file", "video_file"])
def test_file_operations(request, test_file):
    test_file_type = test_file
    test_file = request.getfixturevalue(test_file)

    if test_file_type == "text_file":
        ext = '.txt'
    elif test_file_type == "pdf_file":
        ext = '.pdf'
    elif test_file_type == "image_file":
        ext = '.png'
    elif test_file_type == "python_file":
        ext = '.py'
    elif test_file_type == "video_file":
        ext = '.mp4'
    else:
        raise ValueError("no such file %s" % test_file_type)

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
    assert upload_response.filename == "test_file%s" % ext

    file_id = upload_response.id

    # Test list files
    list_response = client.files.list().data
    assert isinstance(list_response, list)
    assert list_response[0].id == file_id
    assert list_response[0].object == "file"
    assert list_response[0].purpose == "assistants"
    assert list_response[0].created_at
    assert list_response[0].bytes > 5
    assert list_response[0].filename == "test_file%s" % ext

    # Test retrieve file
    retrieve_response = client.files.retrieve(file_id)
    assert retrieve_response.id == file_id
    assert retrieve_response.object == "file"

    # Test retrieve file content
    content = client.files.content(file_id).content
    check_content(content, test_file_type, test_file)

    content = client.files.content(file_id, extra_body=dict(stream=True)).content
    check_content(content, test_file_type, test_file)

    # Test delete file
    delete_response = client.files.delete(file_id)
    assert delete_response.id == file_id
    assert delete_response.object == "file"
    assert delete_response.deleted is True


def check_content(content, test_file_type, test_file):
    if test_file_type in ["text_file", "python_file"]:
        # old
        with open(test_file, 'rb') as f:
            old_content = f.read()
        # new
        assert content.decode('utf-8') == old_content.decode('utf-8')
    elif test_file_type == 'pdf_file':
        import fitz
        # old
        assert fitz.open(test_file).is_pdf
        # new
        with tempfile.NamedTemporaryFile() as tmp_file:
            new_file = tmp_file.name
            with open(new_file, 'wb') as f:
                f.write(content)
            assert fitz.open(new_file).is_pdf
    elif test_file_type == 'image_file':
        from PIL import Image
        # old
        assert Image.open(test_file).format == 'PNG'
        # new
        with tempfile.NamedTemporaryFile() as tmp_file:
            new_file = tmp_file.name
            with open(new_file, 'wb') as f:
                f.write(content)
            assert Image.open(new_file).format == 'PNG'
    elif test_file_type == 'video_file':
        import cv2
        # old
        cap = cv2.VideoCapture(test_file)
        if not cap.isOpened():
            return False

        # Check if we can read the first frame
        ret, frame = cap.read()
        if not ret:
            return False
        cap.release()

        # new
        with tempfile.NamedTemporaryFile() as tmp_file:
            new_file = tmp_file.name
            with open(new_file, 'wb') as f:
                f.write(content)

            cap = cv2.VideoCapture(new_file)
            if not cap.isOpened():
                return False

            # Check if we can read the first frame
            ret, frame = cap.read()
            if not ret:
                return False
            cap.release()


@pytest.mark.serverless
def test_return_generator():
    import typing

    def generator_function() -> typing.Generator[str, None, str]:
        yield "Intermediate result 1"
        yield "Intermediate result 2"
        return "Final Result"

    # Example usage
    gen = generator_function()

    # Consume the generator
    ret_dict = None
    try:
        while True:
            value = next(gen)
            print(value)
    except StopIteration as e:
        ret_dict = e.value

    # Get the final return value
    assert ret_dict == "Final Result"


@pytest.mark.needs_server
def test_tool_use():
    from openai import OpenAI
    import json

    model1 = 'gpt-4o'
    client = OpenAI(base_url='http://localhost:5000/v1', api_key='EMPTY')

    # client = OpenAI()

    # Example dummy function hard coded to return the same weather
    # In production, this could be your backend API or an external API
    def get_current_weather(location, unit="fahrenheit"):
        """Get the current weather in a given location"""
        if "tokyo" in location.lower():
            return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
        elif "san francisco" in location.lower():
            return json.dumps(
                {"location": "San Francisco", "temperature": "72" if unit == "fahrenheit" else "25", "unit": unit})
        elif "paris" in location.lower():
            return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
        else:
            return json.dumps({"location": location, "temperature": "unknown"})

    def run_conversation(model):
        # Step 1: send the conversation and available functions to the model
        messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location", "unit"],
                    },
                },
            }
        ]

        model_info = client.models.retrieve(model)
        assert model_info.id == model
        model_list = client.models.list()
        assert model in [x.id for x in model_list.data]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",  # auto is default, but we'll be explicit
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        # Step 2: check if the model wanted to call a function
        if tool_calls:
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "get_current_weather": get_current_weather,
            }  # only one function in this example, but you can have multiple
            messages.append(response_message)  # extend conversation with assistant's reply
            # Step 4: send the info for each function call and function response to the model
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                    location=function_args.get("location"),
                    unit=function_args.get("unit"),
                )
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )  # extend conversation with function response
            second_response = client.chat.completions.create(
                model=model,
                messages=messages,
            )  # get a new response from the model where it can see the function response
            print(second_response)
            return second_response.choices[0].message.content

    print(run_conversation(model1))


@pytest.mark.needs_server
def test_tool_use2():
    from openai import OpenAI
    import json

    model = 'gpt-4o'
    client = OpenAI(base_url='http://localhost:5000/v1', api_key='EMPTY')
    # client = OpenAI()

    prompt = """"# Tool Name

get_current_weather
# Tool Description:

Get the current weather in a given location

# Prompt

What's the weather like in San Francisco, Tokyo, and Paris?


Choose the single tool that best solves the task inferred from the prompt.  Never choose more than one tool, i.e. act like parallel_tool_calls=False.  If no tool is a good fit, then only choose the noop tool.
"""
    messages = [{"role": "user", "content": prompt}]
    tools = [{'type': 'function',
              'function': {'name': 'get_current_weather', 'description': 'Get the current weather in a given location',
                           'parameters': {'type': 'object', 'properties': {'location': {'type': 'string',
                                                                                        'description': 'The city and state, e.g. San Francisco, CA'},
                                                                           'unit': {'type': 'string',
                                                                                    'enum': ['celsius', 'fahrenheit']}},
                                          'required': ['location']}}}]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        # parallel_tool_calls=False,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    assert tool_calls


if __name__ == '__main__':
    launch_openai_server()
