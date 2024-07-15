## Client APIs

A Gradio API and an OpenAI-compliant API are supported. You can also use `curl` to some extent for basic API.

## OpenAI Proxy client API

h2oGPT by default starts an [OpenAI compatible server](README_InferenceServers.md#openai-proxy-inference-server-client).  One communicates to it via OpenAI 1.x Python package.

### Chat and Text Completions

For example:
```python
from openai import OpenAI
base_url = 'https://localhost:5000/v1'
api_key = 'INSERT KEY HERE or set to EMPTY if no key set on h2oGPT server'
client_args = dict(base_url=base_url, api_key=api_key)
openai_client = OpenAI(**client_args)

messages = [{'role': 'user', 'content': 'Who are you?'}]
stream = False
client_kwargs = dict(model='h2oai/h2ogpt-4096-llama2-70b-chat', max_tokens=200, stream=stream, messages=messages)
client = openai_client.chat.completions

responses = client.create(**client_kwargs)
text = responses.choices[0].message.content
print(text)
```
or for streaming:
```python
from openai import OpenAI
base_url = 'http://localhost:5000/v1'
api_key = 'INSERT KEY HERE or set to EMPTY if no key set on h2oGPT server'
client_args = dict(base_url=base_url, api_key=api_key)
openai_client = OpenAI(**client_args)

messages = [{'role': 'user', 'content': 'Who are you?'}]
stream = True
client_kwargs = dict(model='h2oai/h2ogpt-4096-llama2-70b-chat', max_tokens=200, stream=stream, messages=messages)
client = openai_client.chat.completions

responses = client.create(**client_kwargs)
text = ''
for chunk in responses:
    delta = chunk.choices[0].delta.content
    if delta:
        text += delta
        print(delta, end='')
```
just as with OpenAI, and related API for text completion (non-chat) mode.

### Image Understanding

```python
from src.vision.utils_vision import img_to_base64

# local files would only work if server on same system as client
# for img_to_base64, str_bytes=True or False will work.  True is for internal use for LLaVa gradio communication only
urls = ['https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg',
        img_to_base64('tests/driverslicense.jpeg'),
        img_to_base64('tests/receipt.jpg'),
        img_to_base64('tests/dental.png'),
        ]
expecteds = ['tiger', 'license', 'receipt', ['Oral', 'Clinic']]
for expected, url in zip(expecteds, urls):
    # OpenAI API
    messages = [{
        'role':
            'user',
        'content': [{
            'type': 'text',
            'text': 'Describe the image please',
        }, {
            'type': 'image_url',
            'image_url': {
                'url':
                    url,
            },
        }],
    }]



    model = 'OpenGVLab/InternVL-Chat-V1-5'
    base_url = 'http://localhost:5000/v1'
    h2ogpt_key = 'fill or EMPTY'

    from openai import OpenAI
    client_args = dict(base_url=base_url,
                       api_key=h2ogpt_key)
    client = OpenAI(**client_args)

    # auth:
    # user = '%s:%s' % ('user', 'pass')
    # no auth:
    user = None

    client_kwargs = dict(model=model,
                         max_tokens=200,
                         stream=False,
                         messages=messages,
                         user=user,
                         )
    response = client.chat.completions.create(**client_kwargs)
    print(response)
    if isinstance(expected, list):
        assert any(x in response.choices[0].message.content for x in expected), "%s %s" % (url, response)
    else:
        assert expected in response.choices[0].message.content, "%s %s" % (url, response)
```

That that `str_bytes=True` leads to something like:
```text
b'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...'
```
which includes the b prefix indicating it's a byte string.
while `str_bytes=False` leads to something like
```text
data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...
```
without the b prefix, indicating it's a plain string.

Ensure the bytes encoded part does *not* itself have `b' '` around it.  i.e. if used:
```python
f"data:image/{iformat.lower()};base64,{img_str.decode('utf-8')}"
```
and `img_str = str(bytes_object)` that will not be correct.

#### Authentication

If h2oGPT has authentication enabled, then one passes `user` to OpenAI with the `username:password` as a string to access.  E.g.:
```python
from openai import OpenAI
base_url = 'http://localhost:5000/v1'
api_key = 'INSERT KEY HERE or set to EMPTY if no key set on h2oGPT server'
model = '<model name>'

client_args = dict(base_url=base_url, api_key=api_key)
openai_client = OpenAI(**client_args)

messages = [{'role': 'user', 'content': 'Who are you?'}]
stream = False
client_kwargs = dict(model=model, max_tokens=200, stream=stream, messages=messages,
                     user='username:password')
client = openai_client.chat.completions

responses = client.create(**client_kwargs)
text = responses.choices[0].message.content
print(text)
```
This is only required if `--auth_access=closed` was used, else for `--auth_access=open` we use guest access if that is allowed, else random uuid if no guest access.  Note that if access is closed, one cannot get model names or info.

**Note:** The default OpenAI proxy port for MacOS is set to `5001`, since ports 5000 and 7000 are being used by [AirPlay in MacOS](https://developer.apple.com/forums/thread/682332).

### extra_body

In order to control other parameters not normally part of OpenAI API, one can use `extra_body`, e.g.
```python
from openai import OpenAI

base_url = 'http://localhost:5000/v1'
api_key = 'INSERT KEY HERE or set to EMPTY if no key set on h2oGPT server'
model = '<model name>'

client_args = dict(base_url=base_url, api_key=api_key)
openai_client = OpenAI(**client_args)

messages = [{'role': 'user', 'content': 'Who are you?'}]
stream = False
client_kwargs = dict(model=model, max_tokens=200, stream=stream, messages=messages,
                     user='username:password',
                     extra_body=dict(langchain_mode='UserData'))
client = openai_client.chat.completions

responses = client.create(**client_kwargs)
text = responses.choices[0].message.content
print(text)
```
The OpenAI client does a login to the Gradio server as well, so one can access personal collections like `MyData` as well.

Any parameters normally passed to gradio client can be passed this way. See [H2oGPTParams](../openai_server/server.py) for complete list.

### Text to Speech

h2oGPT can do text-to-speech and speech-to-text if `--enable_tts=True` and `--enable_stt=True` as well
as `--pre_load_image_audio_models=True`, respectively. h2oGPT's OpenAI Proxy server follows OpenAI API
for [Text to Speech](https://platform.openai.com/docs/guides/text-to-speech), e.g.:

```python
from openai import OpenAI
client = OpenAI(base_url='http://0.0.0.0:5000/v1')

with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="",
        extra_body=dict(stream=True,
                        chatbot_role="Female AI Assistant",
                        speaker="SLT (female)",
                        stream_strip=True,
                        ),
        response_format='wav',
        input="Good morning! The sun is shining brilliantly today, casting a warm, golden glow that promises a day full of possibility and joy. It’s the perfect moment to embrace new opportunities and make the most of every cheerful, sunlit hour. What can I do to help you make today absolutely wonderful?",
) as response:
    response.stream_to_file("speech_local.wav")
```

Set `stream=False` to avoid streaming, e.g.:
```python
    from openai import OpenAI

    client = OpenAI(base_url='http://0.0.0.0:5000/v1')

    response = client.audio.speech.create(
            model="tts-1",
            voice="",
            extra_body=dict(stream=False,
                            chatbot_role="Female AI Assistant",
                            speaker="SLT (female)",
                            format='wav',
                            ),
            input="Today is a wonderful day to build something people love! Today is a wonderful day to build something people love! Today is a wonderful day to build something people love! Today is a wonderful day to build something people love! Today is a wonderful day to build something people love! Today is a wonderful day to build something people love! Today is a wonderful day to build something people love! Today is a wonderful day to build something people love! Today is a wonderful day to build something people love! Today is a wonderful day to build something people love! Today is a wonderful day to build something people love! Today is a wonderful day to build something people love! Today is a wonderful day to build something people love! Today is a wonderful day to build something people love! ",
    )
    response.stream_to_file("speech_local2.wav")
```

To stream the audio and play during streaming, one can use httpx and pygame:
```python
import openai
import httpx
import pygame

import pygame.mixer

pygame.mixer.init(frequency=16000, size=-16, channels=1)

sound_queue = []


def play_audio(audio):
    import io
    from pydub import AudioSegment

    sr = 16000
    s = io.BytesIO(audio)
    channels = 1
    sample_width = 2

    audio = AudioSegment.from_raw(s, sample_width=sample_width, frame_rate=sr, channels=channels)
    sound = pygame.mixer.Sound(io.BytesIO(audio.raw_data))
    sound_queue.append(sound)
    sound.play()

    # Wait for the audio to finish playing
    duration_ms = sound.get_length() * 1000  # Convert seconds to milliseconds
    pygame.time.wait(int(duration_ms))


# Ensure to clear the queue when done to free memory and resources
def clear_queue(sound_queue):
    for sound in sound_queue:
        sound.stop()


api_key = 'EMPTY'

# Initialize OpenAI and Pygame
client = openai.OpenAI(api_key=api_key)

# Set up the request headers and parameters
headers = {
    "Authorization": f"Bearer {client.api_key}",
    "Content-Type": "application/json",
}
data = {
    "model": "tts-1",
    "voice": "SLT (female)",
    "input": "Good morning! The sun is shining brilliantly today, casting a warm, golden glow that promises a day full of possibility and joy. It’s the perfect moment to embrace new opportunities and make the most of every cheerful, sunlit hour. What can I do to help you make today absolutely wonderful?",
    "stream": "true",
    "stream_strip": "false",
}

# base_url = "https://api.openai.com/v1"
base_url = "http://localhost:5000/v1/audio/speech"

# Start the HTTP session and stream the audio
with httpx.Client(timeout=None) as http_client:
    # Initiate a POST request and stream the response
    with http_client.stream("POST", base_url, headers=headers, json=data) as response:
        chunk_riff = b''
        for chunk in response.iter_bytes():
            if chunk.startswith(b'RIFF'):
                if chunk_riff:
                    play_audio(chunk_riff)
                chunk_riff = chunk
            else:
                chunk_riff += chunk
        # Play the last accumulated chunk
        if chunk_riff:
            play_audio(chunk_riff)
# done
clear_queue(sound_queue)
pygame.quit()
```

The streaming case writes the file (which could be to some buffer) each chunk (sentence) at a time, while non-streaming case does entire file at once and client waits till end to write the file.  For the streaming case, if it is a wave file, like OpenAI, the server artificially inflates the estimated duration of the audio so player will play through end of the audio.

### Speech to Text

Requires h2oGPT loaded with `--enable_stt=True --pre_load_image_audio_models=True`.

```python
from openai import OpenAI
client = OpenAI(base_url='http://0.0.0.0:5000/v1')

file = "speech.wav"
with open(file, "rb") as f:
    audio_file= f.read()
transcription = client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file
)
print(transcription.text)
```

Streaming STT is not natively supported by OpenAI client, but it can still be done via httpx:
```python
import json
import httpx
import asyncio

async def stream_audio_transcription(file_path, model="default-model"):
    url = "http://0.0.0.0:5000/v1/audio/transcriptions"
    headers = {"X-API-KEY": "your-api-key"}

    # Read the audio file
    with open(file_path, "rb") as f:

        # Create the multipart/form-data payload
        files = {
            "file": ("audio.wav", f, "audio/wav"),
            "model": (None, model),
            "stream": (None, "true"),  # Note the lowercase "true" as the server checks for this
            "response_format": (None, "text"),
            "chunk": (None, "none"),
        }

        text = ''
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=headers, files=files, timeout=120) as response:
                async for line in response.aiter_lines():
                    # Process each chunk of data as it is received
                    if line.startswith("data:"):
                        try:
                            # Remove "data: " prefix and strip any newlines or trailing whitespace
                            json_data = json.loads(line[5:].strip())
                            # Process the parsed JSON data
                            print('json_data: %s' % json_data)
                            text += json_data["text"]
                        except json.JSONDecodeError as e:
                            print("Error decoding JSON:", e)
        return text
# Run the client function
final_text = asyncio.run(stream_audio_transcription("/home/jon/h2ogpt/tests/test_speech.wav"))
print(final_text)
```

### Image Generation

Requires h2oGPT loaded with `--enable_image=True --pre_load_image_audio_models=True --visible_image_models=['sdxl_turbo']` or some selection of such image generation models.

```python
from openai import OpenAI
client = OpenAI(base_url='http://0.0.0.0:5000/v1')
# client = OpenAI()

response = client.images.generate(
  model="sdxl_turbo",  # should be empty if do not know which model, h2oGPT will choose first if exists
  prompt="A cute baby sea otter",
  n=1,
  size="1024x1024",
  response_format='b64_json',
)
import base64
image_data = base64.b64decode(response.data[0].b64_json.encode('utf-8'))
# Convert binary data to an image
from PIL import Image
import io
image = Image.open(io.BytesIO(image_data))
# Save the image to a file or display it
image.save('output_image.png')
image.show()  # This will open the default image viewer and display the image
```

### Embedding

Requires h2oGPT loaded with langchain enabled (not `--langchain_mode=Disabled`) and `--pre_load_embedding_model=True` and potentially some choice for `--hf_embedding_model` (default is used if no specified) and `--use_openai_embedding=False` to be set (default).

Note `model` is ignored currently, uses single embedding in h2oGPT.
```python
from openai import OpenAI
client = OpenAI(base_url='http://0.0.0.0:5000/v1')
#client = OpenAI()

response = client.embeddings.create(
    input="Your text string goes here",
    model="text-embedding-3-small"
)
print(response.data[0].embedding)

response = client.embeddings.create(
    input=["Your text string goes here", "Another text string goes here"],
    model="text-embedding-3-small"
)
print(response.data[0].embedding)
print(response.data[1].embedding)
```

### Curl for REST API

Or for curl, with api_key set or as `EMPTY` if not set, one can do:
```bash
export OPENAI_API_KEY=xxxx
curl https://localhost:5000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "prompt": "Who are you?",
    "max_tokens": 200,
    "temperature": 0,
    "seed": 1234,
    "h2ogpt_key": "$OPENAI_API_KEY"
  }'
```
where one should pass along the `h2ogpt_key` if gradio is itself protected for some queries.

Chat completion also works with curl like:
```bash
export OPENAI_API_KEY=xxxx
curl http://localhost:5000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $OPENAI_API_KEY" \
-d '{
  "messages": [
    {
      "role": "system",
      "content": "You are a beautiful dragon who likes to breath fire."
    },
    {
      "role": "user",
      "content": "Who are you?"
    }
  ],
  "max_tokens": 200,
  "temperature": 0,
  "seed": 1234,
  "h2ogpt_key": "$OPENAI_API_KEY"
}'
```

For streaming, just add `stream` bool, e.g.:
```bash
export OPENAI_API_KEY=xxxx
curl http://localhost:5000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $OPENAI_API_KEY" \
-d '{
  "messages": [
    {
      "role": "system",
      "content": "You are a beautiful dragon who likes to breath fire."
    },
    {
      "role": "user",
      "content": "Who are you?"
    }
  ],
  "max_tokens": 200,
  "temperature": 0,
  "seed": 1234,
  "h2ogpt_key": "$OPENAI_API_KEY",
  "stream": true
}'
```
which results in chunks of choices of delta like given in the OpenAI Python API.

The strings `prompt` and `max_tokens` are taken as OpenAI type names that are converted to `instruction` and `max_new_tokens`.  In either case, any additional parameters are passed along to the Gradio `submit_nochat_api` API.  Either `http` or `https` works if using ngrok or some proxy service, or setup directly in the OpenAI proxy server.  Replace 'localhost' with the http or https proxy (or direct SSL) server name or IP.  Replace 5000 with the assigned port.

## Gradio Client API

h2oGPT's `generate.py` by default runs a gradio server, which also gives access to client API using the [Gradio Python client](https://www.gradio.app/docs/python-client). You can use it with h2oGPT, or independently of h2oGPT repository by installing an env:
```bash
conda create -n gradioclient -y
conda activate gradioclient
conda install python=3.10 -y
pip install gradio_client==0.6.1

# Download Gradio Wrapper code if GradioClient class used, not needed for native Gradio Client
# No wheel for now
wget https://raw.githubusercontent.com/h2oai/h2ogpt/main/gradio_utils/grclient.py
mkdir -p gradio_utils
mv grclient.py gradio_utils
```

Run client code with Gradio's native client:
```python
from gradio_client import Client
import ast

HOST_URL = "http://localhost:7860"
client = Client(HOST_URL)

# string of dict for input
kwargs = dict(instruction_nochat='Who are you?')
res = client.predict(str(dict(kwargs)), api_name='/submit_nochat_api')

# string of dict for output
response = ast.literal_eval(res)['response']
print(response)
```

You can also stream the response. The following is a complete example code of streaming each updated text fragment to the console so that they appear to stream in the console:
```python
from gradio_client import Client
import ast
import time

HOST = 'http://localhost:7860'
client = Client(HOST)
api_name = '/submit_nochat_api'
prompt = "Who are you?"
kwargs = dict(instruction_nochat=prompt, stream_output=True)

job = client.submit(str(dict(kwargs)), api_name=api_name)

text_old = ''
while not job.done():
    outputs_list = job.communicator.job.outputs
    if outputs_list:
        res = job.communicator.job.outputs[-1]
        res_dict = ast.literal_eval(res)
        text = res_dict['response']
        new_text = text[len(text_old):]
        if new_text:
            print(new_text, end='', flush=True)
            text_old = text
        time.sleep(0.01)
# handle case if never got streaming response and already done
res_final = job.outputs()
if len(res_final) > 0:
    res = res_final[-1]
    res_dict = ast.literal_eval(res)
    text = res_dict['response']
    new_text = text[len(text_old):]
    print(new_text)
```

### Image Understanding

```python
import ast
from gradio_client import Client

# without auth:
# client = Client('http://localhost:7860')

# with auth:
client = Client('http://localhost:7860', auth=('user', 'pass'))

h2ogpt_key = 'api key here, or EMPTY if no key or do not put in kwargs'

kwargs = dict(
    visible_models='THUDM/cogvlm2-llama3-chat-19B',
    instruction_nochat="describe the imaged",
    h2ogpt_key=h2ogpt_key,
    stream_output=False,
    image_file='https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg',
    temperature=0,
    max_tokens=4000)
res = client.predict(str(dict(kwargs)), api_name='/submit_nochat_api')

response = ast.literal_eval(res)['response']
print(response)
```

WIth bytes:

```python
import ast

from gradio_client import Client

# can copy-paste these functions for own use
from src.utils import download_image
from src.vision.utils_vision import img_to_base64

# without auth:
# client = Client('http://localhost:7860')

# with auth:
client = Client('http://localhost:7860', auth=('user', 'pass'))

h2ogpt_key = 'api key here, or EMPTY if no key or do not put in kwargs'


image_url = 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'
save_dir = 'datatest'
image_file = download_image(image_url, save_dir)
image_bytes = img_to_base64(image_file)

kwargs = dict(
    visible_models='THUDM/cogvlm2-llama3-chat-19B',
    instruction_nochat="describe the imaged",
    h2ogpt_key=h2ogpt_key,
    stream_output=False,
    image_file=image_bytes,
    temperature=0,
    max_tokens=4000)
res = client.predict(str(dict(kwargs)), api_name='/submit_nochat_api')

response = ast.literal_eval(res)['response']
print(response)
```

### h2oGPT Gradio Wrapper

You can run client code with the h2oGPT wrapper class for Gradio's client, which adds extra exception handling and h2oGPT-specific calls.

For talking to just LLM, Document Q/A, summarization, and extraction, you can do:
```python
def test_readme_example(local_server):
    # self-contained example used for readme, to be copied to README_CLIENT.md if changed, setting local_server = True at first
    import os
    # The grclient.py file can be copied from h2ogpt repo and used with local gradio_client for example use
    from gradio_utils.grclient import GradioClient

    if local_server:
        client = GradioClient("http://0.0.0.0:7860")
    else:
        h2ogpt_key = os.getenv('H2OGPT_KEY') or os.getenv('H2OGPT_H2OGPT_KEY')
        if h2ogpt_key is None:
            return
        # if you have API key for public instance:
        client = GradioClient("https://gpt.h2o.ai", h2ogpt_key=h2ogpt_key)

    # LLM
    print(client.question("Who are you?"))

    url = "https://cdn.openai.com/papers/whisper.pdf"

    # Q/A
    print(client.query("What is whisper?", url=url))
    # summarization (map_reduce over all pages if top_k_docs=-1)
    print(client.summarize("What is whisper?", url=url, top_k_docs=3))
    # extraction (map per page)
    print(client.extract("Give bullet for all key points", url=url, top_k_docs=3))
test_readme_example(local_server=True)
```

#### Other API calls

For other ways to use gradio client, see example [test code](../src/client_test.py) or other tests in our [tests](https://github.com/h2oai/h2ogpt/blob/main/tests/test_client_calls.py).  E.g. `test_client_chat_stream_langchain_steps3` in [client tests](https://github.com/h2oai/h2ogpt/blob/main/tests/test_client_calls.py) uses many different API calls for docs etc.s

Note that any element in [gradio_runner.py](../src/gradio_runner.py) with `api_name` defined can be accessed via the gradio client.

#### Listing models

```python
>>> from gradio_client import Client
>>> client = Client('http://localhost:7860')
Loaded as API: http://localhost:7860/ ✔
>>> import ast
>>> res = client.predict(api_name='/model_names')
>>> {x['base_model']: x['max_seq_len'] for x in ast.literal_eval(res)}
{'h2oai/h2ogpt-4096-llama2-70b-chat': 4046, 'lmsys/vicuna-13b-v1.5-16k': 16334, 'mistralai/Mistral-7B-Instruct-v0.1': 4046, 'gpt-3.5-turbo-0613': 4046, 'gpt-3.5-turbo-16k-0613': 16335, 'gpt-4-0613': 8142, 'gpt-4-32k-0613': 32718}
```

### h2oGPT Server options for efficient Summarization and Extraction

You can specify the h2oGPT server to have `--async_output=True` and `--num_async=10` (or some optimal value) to enable full parallel summarization when the h2oGPT server uses `--inference_server` that points to Gradio Inference Server, vLLM, text-generation inference (TGI) server, or OpenAI servers to allow for high tokens/sec.

### Curl Client API

As long as objects within the `gradio_runner.py` file for a given api_name are for a function without `gr.State()` objects, then curl can work. Note that full `curl` capability is [not yet supported in Gradio](https://github.com/gradio-app/gradio/issues/4932).

For example, for a server launched as:
```bash
python generate.py --base_model=TheBloke/Llama-2-7b-Chat-GPTQ --load_gptq="model" --use_safetensors=True --prompt_type=llama2 --save_dir=fooasdf --system_prompt='auto'
```
you can use the `submit_nochat_plain_api`, which has no `state` objects, to perform chat via `curl` by entering the following command:
```bash
curl 127.0.0.1:7860/api/submit_nochat_plain_api -X POST -d '{"data": ["{\"instruction_nochat\": \"Who are you?\"}"]}' -H 'Content-Type: application/json'
```
and get back for a 7B LLaMA2-chat GPTQ model:

`{"data":["{'response': \" Hello! I'm just an AI assistant designed to provide helpful and informative responses to your questions. My purpose is to assist and provide accurate information to the best of my abilities, while adhering to ethical and moral guidelines. I am not capable of providing personal opinions or engaging in discussions that promote harmful or offensive content. My goal is to be a positive and respectful presence in your interactions with me. Is there anything else I can help you with?\", 'sources': '', 'save_dict': {'prompt': \"<s>[INST] <<SYS>>\\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\\n\\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\\n<</SYS>>\\n\\nWho are you? [/INST]\", 'output': \" Hello! I'm just an AI assistant designed to provide helpful and informative responses to your questions. My purpose is to assist and provide accurate information to the best of my abilities, while adhering to ethical and moral guidelines. I am not capable of providing personal opinions or engaging in discussions that promote harmful or offensive content. My goal is to be a positive and respectful presence in your interactions with me. Is there anything else I can help you with?\", 'base_model': 'TheBloke/Llama-2-7b-Chat-GPTQ', 'save_dir': 'fooasdf', 'where_from': 'evaluate_False', 'extra_dict': {'num_beams': 1, 'do_sample': False, 'repetition_penalty': 1.07, 'num_return_sequences': 1, 'renormalize_logits': True, 'remove_invalid_values': True, 'use_cache': True, 'eos_token_id': 2, 'bos_token_id': 1, 'num_prompt_tokens': 5, 't_generate': 9.243812322616577, 'ntokens': 120, 'tokens_persecond': 12.981605669647344}, 'error': None, 'extra': None}}"],"is_generating":true,"duration":39.33809685707092,"average_duration":39.33809685707092}`

This response contains the full dictionary of `data` from the `curl` operation as well as the data contents that are a string of a dictionary like when using the API `submit_nochat_api` for Gradio client.  This inner string of a dictionary can be parsed as a literal python string to get keys `response`, `source`, `save_dict`, where `save_dict` contains metadata about the query such as generation hyperparameters, tokens generated, etc.

