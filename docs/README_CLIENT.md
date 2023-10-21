### Client APIs

A Gradio API and an OpenAI-compliant API are supported.  One can also use `curl` to some extent for basic API.

##### Gradio Client API

`generate.py` by default runs a gradio server, which also gives access to client API using gradio client.  One can use it with h2oGPT, or independently of h2oGPT repository by installing an env:
```bash
conda create -n gradioclient -y
conda activate gradioclient
conda install python=3.10 -y
pip install gradio_client
```
then running client code:
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
For other ways to use gradio client, see example [test code](../client_test.py) or other tests in our [tests](https://github.com/h2oai/h2ogpt/blob/main/tests/test_client_calls.py).  E.g. `test_client_chat_stream_langchain_steps3` etc. [tests](https://github.com/h2oai/h2ogpt/blob/main/tests/test_client_calls.py) use many different API calls for docs etc.

One can also stream the response.  Here is a complete example code of streaming to console each updated text fragment so appears to stream in console:
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


Any element in [gradio_runner.py](../gradio_runner.py) with `api_name` defined can be accessed via the gradio client.

The below is an example client code, which handles persistence of state when doing multiple queries, or avoids persistence to avoid issues when server goes up and down for a fixed client.  Choose `HOST` to be the h2oGPT server, and as gradio client use function calls `answer_question_using_context` and `summarize` that handle question-answer or summarization using LangChain backend.   One can choose h2oGPT server to have `--async_output=True` and `--num_async=10` (or some optimal value) to enable full parallel summarization when the h2oGPT server uses `--inference_server` that points to a text-generation inference server, to allow for high tokens/sec.
```python
HOST = "localhost:7860"  # choose

import ast
import os
import traceback
from enum import Enum
from typing import Union

from gradio_client.client import Job

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from gradio_client import Client


class GradioClient(Client):
    """
    Parent class of gradio client
    To handle automatically refreshing client if detect gradio server changed
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)
        self.server_hash = self.get_server_hash()

    def get_server_hash(self):
        """
        Get server hash using super without any refresh action triggered
        Returns: git hash of gradio server
        """
        return super().submit(api_name='/system_hash').result()

    def refresh_client_if_should(self):
        # get current hash in order to update api_name -> fn_index map in case gradio server changed
        # FIXME: Could add cli api as hash
        server_hash = self.get_server_hash()
        if self.server_hash != server_hash:
            self.refresh_client()
            self.server_hash = server_hash
        else:
            self.reset_session()

    def refresh_client(self):
        """
        Ensure every client call is independent
        Also ensure map between api_name and fn_index is updated in case server changed (e.g. restarted with new code)
        Returns:
        """
        # need session hash to be new every time, to avoid "generator already executing"
        self.reset_session()

        client = Client(*self.args, **self.kwargs)
        for k, v in client.__dict__.items():
            setattr(self, k, v)

    def submit(
            self,
            *args,
            api_name=None,
            fn_index=None,
            result_callbacks=None,
    ) -> Job:
        # Note predict calls submit
        try:
            self.refresh_client_if_should()
            job = super().submit(*args, api_name=api_name, fn_index=fn_index)
        except Exception as e:
            print("Hit e=%s" % str(e), flush=True)
            # force reconfig in case only that
            self.refresh_client()
            job = super().submit(*args, api_name=api_name, fn_index=fn_index)

        # see if immediately failed
        e = job.future._exception
        if e is not None:
            print("GR job failed: %s %s" % (str(e), ''.join(traceback.format_tb(e.__traceback__))), flush=True)
            # force reconfig in case only that
            self.refresh_client()
            job = super().submit(*args, api_name=api_name, fn_index=fn_index)
            e2 = job.future._exception
            if e2 is not None:
                print("GR job failed again: %s\n%s" % (str(e2), ''.join(traceback.format_tb(e2.__traceback__))),
                      flush=True)

        return job


from .settings import settings

# TODO use settings.llm_api_key for auth
client = GradioClient(settings.llm_address)


def _call_h2o_gpt_api(prompt: str) -> str:
    # don't specify prompt_type etc., use whatever endpoint setup
    kwargs = dict(
        stream_output=False,
        max_time=360,
        instruction_nochat=prompt,
    )
    return client.predict(str(kwargs), api_name='/submit_nochat_api')


prompt_template = '''
"""
{context}
"""
{question}
'''


def answer_question_using_context(question: str, context: str) -> str:
    prompt = prompt_template.format(context=context, question=question)
    answer = _call_h2o_gpt_api(prompt)
    return ast.literal_eval(answer)['response']


class LangChainAction(Enum):
    """LangChain action"""

    QUERY = "Query"
    SUMMARIZE_MAP = "Summarize"


def query(instruction: str = None,
          text: str = None,
          file: str = None,
          url: str = None,
          top_k_docs: int = 4,
          pre_prompt_query: str = None,
          prompt_query: str = None,
          asserts: bool = True) -> str:
    """
    Query using h2oGPT
    """
    return query_or_summarize(instruction=instruction,
                              text=text,
                              file=file,
                              url=url,
                              langchain_action=LangChainAction.QUERY.value,
                              top_k_docs=top_k_docs,
                              pre_prompt_query=pre_prompt_query,
                              prompt_query=prompt_query,
                              asserts=asserts)


def summarize(text: str = None,
              file: str = None,
              url: str = None,
              top_k_docs: int = 4,
              pre_prompt_summary: str = None,
              prompt_summary: str = None,
              asserts: bool = True) -> str:
    """
    Summarize using h2oGPT
    """
    return query_or_summarize(text=text,
                              file=file,
                              url=url,
                              langchain_action=LangChainAction.SUMMARIZE_MAP.value,
                              top_k_docs=top_k_docs,
                              pre_prompt_summary=pre_prompt_summary,
                              prompt_summary=prompt_summary,
                              asserts=asserts)


def query_or_summarize(instruction: str = '',
                       text: Union[list[str], str] = None,
                       file: Union[list[str], str] = None,
                       url: Union[list[str], str] = None,
                       langchain_action: str = None,
                       embed: str = True,
                       top_k_docs: int = 4,
                       pre_prompt_query: str = None,
                       prompt_query: str = None,
                       pre_prompt_summary: str = None,
                       prompt_summary: str = None,
                       asserts: bool = True) -> str:
    """
    Query or Summarize using h2oGPT
    Args:
        instruction: Query
        For query, prompt template is:
          "{pre_prompt_query}\"\"\"
            {content}
            \"\"\"\n{prompt_query}{instruction}"
         If added to summarization, prompt template is
          "{pre_prompt_summary}:\"\"\"
            {content}
            \"\"\"\n, Focusing on {instruction}, {prompt_summary}"
        text: textual content or list of such contents
        file: a local file to upload or files to upload
        url: a url to give or urls to use
        embed: whether to embed content uploaded
        langchain_action: Action to take, "Query" or "Summarize"
        top_k_docs: number of document parts.
                    When doing query, number of chunks
                    When doing summarization, not related to vectorDB chunks that are not used
                    E.g. if PDF, then number of pages
        pre_prompt_query: Prompt that comes before document part
        prompt_query: Prompt that comes after document part
        pre_prompt_summary: Prompt that comes before document part
           None makes h2oGPT internally use its defaults
           E.g. "In order to write a concise single-paragraph or bulleted list summary, pay attention to the following text"
        prompt_summary: Prompt that comes after document part
          None makes h2oGPT internally use its defaults
          E.g. "Using only the text above, write a condensed and concise summary of key results (preferably as bullet points):\n"
        i.e. for some internal document part fstring, the template looks like:
            template = "%s:
            \"\"\"
            %s
            \"\"\"\n%s" % (pre_prompt_summary, fstring, prompt_summary)
        asserts: whether to do asserts to ensure handling is correct

    Returns: summary: str

    """
    assert text or file or url, "Need to pass either text, file, or url"

    # get persistent client
    client_persist = Client(*client.args, **client.kwargs, serialize=True)

    # chunking not used here
    chunk = True
    chunk_size = 512
    # MyData specifies scratch space, only persisted for this individual client call
    langchain_mode = 'MyData'
    loaders = tuple([None, None, None, None])
    doc_options = tuple([langchain_mode, chunk, chunk_size, embed])

    if text:
        res = client_persist.predict(text,
                                     *doc_options,
                                     *loaders,
                                     api_name='/add_text')
        if asserts:
            assert res[0] is None
            assert res[1] == langchain_mode
            assert 'user_paste' in res[2]
            assert res[3] == ''
    if file:
        # upload file(s).  Can be list or single file
        # after below call, "file" replaced with remote location of file
        _, file = client_persist.predict(file, api_name='/upload_api')

        res = client_persist.predict(file,
                                     *doc_options,
                                     *loaders,
                                     api_name='/add_file_api')
        if asserts:
            assert res[0] is None
            assert res[1] == langchain_mode
            assert os.path.basename(file) in res[2]
            assert res[3] == ''
    if url:
        res = client_persist.predict(url,
                                     *doc_options,
                                     *loaders,
                                     api_name='/add_url')
        if asserts:
            assert res[0] is None
            assert res[1] == langchain_mode
            assert url in res[2]
            assert res[3] == ''

    if langchain_action == LangChainAction.SUMMARIZE_MAP.value:
        # ensure, so full asyncio mode used when gradio connected to TGI server
        stream_output = False
    else:
        # FIXME: should stream
        stream_output = False

    # ask for summary, need to use same client if using MyData
    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    kwargs = dict(instruction=instruction,
                  langchain_mode=langchain_mode,
                  langchain_action=langchain_action,  # uses full document, not vectorDB chunks
                  top_k_docs=top_k_docs,
                  stream_output=stream_output,
                  document_subset='Relevant',
                  document_choice='All',
                  max_new_tokens=256,
                  max_time=360,
                  do_sample=False,
                  pre_prompt_query=pre_prompt_query,
                  prompt_query=prompt_query,
                  pre_prompt_summary=pre_prompt_summary,
                  prompt_summary=prompt_summary,
                  )

    # get result
    res = client_persist.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    res = ast.literal_eval(res)
    response = res['response']
    sources = res['sources']

    if api_name == '/submit_nochat_api':
        scores_out = [x[0] for x in sources]
        texts_out = [x[1] for x in sources]
        if asserts and text and not file and not url:
            assert text == texts_out
            assert len(text) == len(scores_out)
    else:
        if asserts:
            # only pass back file link etc. if not nochat
            if text:
                assert 'user_paste' in sources
            if file:
                assert file in sources
            if url:
                assert url in sources

    return response
```
See tests in https://github.com/h2oai/h2ogpt/blob/main/tests/test_client_calls.py#L678-L1036 that this code is based upon.


##### Listing models

```python
>>> from gradio_client import Client
>>> client = Client('http://localhost:7860')
Loaded as API: http://localhost:7860/ ✔
>>> import ast
>>> res = client.predict(api_name='/model_names')
>>> {x['base_model']: x['max_seq_len'] for x in ast.literal_eval(res)}
{'h2oai/h2ogpt-4096-llama2-70b-chat': 4046, 'lmsys/vicuna-13b-v1.5-16k': 16334, 'mistralai/Mistral-7B-Instruct-v0.1': 4046, 'gpt-3.5-turbo-0613': 4046, 'gpt-3.5-turbo-16k-0613': 16335, 'gpt-4-0613': 8142, 'gpt-4-32k-0613': 32718}
```

##### OpenAI Python Client Library

An OpenAI compliant client is available. Refer the [README](../client/README.md)  for more details.


##### Curl Client API

As long as objects within the `gradio_runner.py` for a given api_name are for a function without `gr.State()` objects, then curl can work.  Full `curl` capability is not supported in Gradio [yet](https://github.com/gradio-app/gradio/issues/4932).

For example, for a server launched as:
```bash
python generate.py --base_model=TheBloke/Llama-2-7b-Chat-GPTQ --load_gptq="model" --use_safetensors=True --prompt_type=llama2 --save_dir=fooasdf --system_prompt='auto'
```
one can use the `submit_nochat_plain_api` that has no `state` objects to perform chat via `curl` by doing:
```bash
curl 127.0.0.1:7860/api/submit_nochat_plain_api -X POST -d '{"data": ["{\"instruction_nochat\": \"Who are you?\"}"]}' -H 'Content-Type: application/json'
```
and get back for a 7B LLaMA2-chat GPTQ model:

`{"data":["{'response': \" Hello! I'm just an AI assistant designed to provide helpful and informative responses to your questions. My purpose is to assist and provide accurate information to the best of my abilities, while adhering to ethical and moral guidelines. I am not capable of providing personal opinions or engaging in discussions that promote harmful or offensive content. My goal is to be a positive and respectful presence in your interactions with me. Is there anything else I can help you with?\", 'sources': '', 'save_dict': {'prompt': \"<s>[INST] <<SYS>>\\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\\n\\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\\n<</SYS>>\\n\\nWho are you? [/INST]\", 'output': \" Hello! I'm just an AI assistant designed to provide helpful and informative responses to your questions. My purpose is to assist and provide accurate information to the best of my abilities, while adhering to ethical and moral guidelines. I am not capable of providing personal opinions or engaging in discussions that promote harmful or offensive content. My goal is to be a positive and respectful presence in your interactions with me. Is there anything else I can help you with?\", 'base_model': 'TheBloke/Llama-2-7b-Chat-GPTQ', 'save_dir': 'fooasdf', 'where_from': 'evaluate_False', 'extra_dict': {'num_beams': 1, 'do_sample': False, 'repetition_penalty': 1.07, 'num_return_sequences': 1, 'renormalize_logits': True, 'remove_invalid_values': True, 'use_cache': True, 'eos_token_id': 2, 'bos_token_id': 1, 'num_prompt_tokens': 5, 't_generate': 9.243812322616577, 'ntokens': 120, 'tokens_persecond': 12.981605669647344}, 'error': None, 'extra': None}}"],"is_generating":true,"duration":39.33809685707092,"average_duration":39.33809685707092}`

This contains the full dictionary of `data` from `curl` operation as well is the data contents that are a string of a dictionary like when using the API `submit_nochat_api` for Gradio client.  This inner string of a dictionary can be parsed as a literal python string to get keys `response`, `source`, `save_dict`, where `save_dict` contains meta data about the query such as generation hyperparameters, tokens generated, etc.
