### Client APIs

A Gradio API and an OpenAI-compliant API are supported.

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
For other ways to use gradio client, see example [test code](../client_test.py) or other tests in our [tests](https://github.com/h2oai/h2ogpt/blob/main/tests/test_client_calls.py).

Any element in [gradio_runner.py](../gradio_runner.py) with `api_name` defined can be accessed via the gradio client.

The below is an example client code, which handles persistence of state when doing multiple queries, or avoids persistence to avoid issues when server goes up and down for a fixed client.  Choose `HOST` to be the h2oGPT server, and as gradio client use function calls `answer_question_using_context` and `summarize` that handle question-answer or summarization using LangChain backend.   One can choose h2oGPT server to have `--async_output=True` and `--num_async=10` (or some optimal value) to enable full parallel summarization when the h2oGPT server uses `--inference_server` that points to a text-generation inference server, to allow for high tokens/sec.
```python
HOST = "localhost:7860"  # choose

import ast
import os
import traceback

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



client = GradioClient(HOST)


def _call_h2o_gpt_api(prompt: str) -> str:
    # don't specify prompt_type etc., use whatever endpoint setup
    kwargs = dict(
        stream_output=False,
        max_new_tokens=768,
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


def summarize(text: str = None,
              file: str = None,
              url: str = None,
              top_k_docs: int = 4,
              pre_prompt_summary: str = '',
              prompt_summary: str = '',
              asserts: bool = True) -> str:
    """
    Summarize using h2oGPT
    Args:
        text: textual content
        file: a local file to upload
        url: a url to give
        top_k_docs: number of document parts.  E.g. if PDF, then number of pages
                    When doing summarization, not related to vectorDB chunks that are not used
        pre_prompt_summary: Prompt that comes before document part
           '' makes h2oGPT internally use its defaults
           E.g. "In order to write a concise single-paragraph or bulleted list summary, pay attention to the following text"
        prompt_summary: Prompt that comes after document part
          '' makes h2oGPT internally use its defaults
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
    # MyData specifies personal/scratch space, only persisted for this individual client call
    langchain_mode = 'MyData'

    if text:
        res = client_persist.predict(text, chunk, chunk_size, langchain_mode, api_name='/add_text')
        if asserts:
            assert res[0] is None
            assert res[1] == langchain_mode
            assert 'user_paste' in res[2]
            assert res[3] == ''
    if file:
        # upload file(s).  Can be list or single file
        # after below call, "file" replaced with remote location of file
        _, file = client_persist.predict(file, api_name='/upload_api')

        res = client_persist.predict(file, chunk, chunk_size, langchain_mode, api_name='/add_file_api')
        if asserts:
            assert res[0] is None
            assert res[1] == langchain_mode
            assert os.path.basename(file) in res[2]
            assert res[3] == ''
    if url:
        res = client_persist.predict(url, chunk, chunk_size, langchain_mode, api_name='/add_url')
        if asserts:
            assert res[0] is None
            assert res[1] == langchain_mode
            assert url in res[2]
            assert res[3] == ''

    # ask for summary, need to use same client if using MyData
    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    kwargs = dict(langchain_mode=langchain_mode,
                  langchain_action="Summarize",  # uses full document, not vectorDB chunks
                  top_k_docs=top_k_docs,
                  stream_output=False,  # ensure, so full asyncio mode used when gradio connected to TGI server
                  document_subset='Relevant',
                  document_choice='All',
                  max_new_tokens=256,
                  max_time=300,
                  do_sample=False,
                  pre_prompt_summary=pre_prompt_summary,
                  prompt_summary=prompt_summary)

    # get result
    res = client_persist.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    res = ast.literal_eval(res)
    summary = res['response']

    if asserts:
        sources = res['sources']
        if text:
            assert 'user_paste' in sources
        if file:
            assert file in sources
        if url:
            assert url in sources

    return summary
```
See tests in https://github.com/h2oai/h2ogpt/blob/main/tests/test_client_calls.py#L678-L1036 that this code is based upon.


##### OpenAI Python Client Library

An OpenAI compliant client is available. Refer the [README](../client/README.md)  for more details.

