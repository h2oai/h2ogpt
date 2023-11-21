## Client APIs

A Gradio API and an OpenAI-compliant API are supported. You can also use `curl` to some extent for basic API.

### OpenAI Compliant Python Client Library

An OpenAI compliant client is available. For more information, refer to the [h2oGPT client README](../client/README.md).

### Gradio Client API

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
