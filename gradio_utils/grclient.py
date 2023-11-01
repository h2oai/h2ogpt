from __future__ import annotations

import difflib
import traceback
import concurrent.futures
import os
import concurrent.futures
import time
import urllib.parse
import uuid
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Generator, Any, Union, List
import ast

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from huggingface_hub import SpaceStage
from huggingface_hub.utils import (
    build_hf_headers,
)

from gradio_client import utils
from gradio_client.client import Job, DEFAULT_TEMP_DIR, Endpoint
from gradio_client import Client


def check_job(job, timeout=0.0, raise_exception=True, verbose=False):
    if timeout == 0:
        e = job.future._exception
    else:
        try:
            e = job.future.exception(timeout=timeout)
        except concurrent.futures.TimeoutError:
            # not enough time to determine
            if verbose:
                print("not enough time to determine job status: %s" % timeout)
            e = None
    if e:
        # raise before complain about empty response if some error hit
        if raise_exception:
            raise RuntimeError(e)
        else:
            return e


# Local copy of minimal version from h2oGPT server
class LangChainAction(Enum):
    """LangChain action"""

    QUERY = "Query"
    SUMMARIZE_MAP = "Summarize"
    EXTRACT = "Extract"


class GradioClient(Client):
    """
    Parent class of gradio client
    To handle automatically refreshing client if detect gradio server changed
    """

    def __init__(
            self,
            src: str,
            hf_token: str | None = None,
            max_workers: int = 40,
            serialize: bool = True,
            output_dir: str | Path | None = DEFAULT_TEMP_DIR,
            verbose: bool = True,
            h2ogpt_key: str = None,
    ):
        """
        Parameters:
            src: Either the name of the Hugging Face Space to load, (e.g. "abidlabs/whisper-large-v2") or the full URL (including "http" or "https") of the hosted Gradio app to load (e.g. "http://mydomain.com/app" or "https://bec81a83-5b5c-471e.gradio.live/").
            hf_token: The Hugging Face token to use to access private Spaces. Automatically fetched if you are logged in via the Hugging Face Hub CLI. Obtain from: https://huggingface.co/settings/token
            max_workers: The maximum number of thread workers that can be used to make requests to the remote Gradio app simultaneously.
            serialize: Whether the client should serialize the inputs and deserialize the outputs of the remote API. If set to False, the client will pass the inputs and outputs as-is, without serializing/deserializing them. E.g. you if you set this to False, you'd submit an image in base64 format instead of a filepath, and you'd get back an image in base64 format from the remote API instead of a filepath.
            output_dir: The directory to save files that are downloaded from the remote API. If None, reads from the GRADIO_TEMP_DIR environment variable. Defaults to a temporary directory on your machine.
            verbose: Whether the client should print statements to the console.
        """
        self.args = tuple([src])
        self.kwargs = dict(
            hf_token=hf_token,
            max_workers=max_workers,
            serialize=serialize,
            output_dir=output_dir,
            verbose=verbose,
            h2ogpt_key=h2ogpt_key,
        )

        self.verbose = verbose
        self.hf_token = hf_token
        self.serialize = serialize
        self.space_id = None
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.src = src
        self.config = None
        self.server_hash = None
        self.h2ogpt_key = h2ogpt_key

    def __repr__(self):
        if self.config:
            return self.view_api(print_info=False, return_format="str")
        return "Not setup for %s" % self.src

    def __str__(self):
        if self.config:
            return self.view_api(print_info=False, return_format="str")
        return "Not setup for %s" % self.src

    def setup(self):
        src = self.src

        self.headers = build_hf_headers(
            token=self.hf_token,
            library_name="gradio_client",
            library_version=utils.__version__,
        )
        if src.startswith("http://") or src.startswith("https://"):
            _src = src if src.endswith("/") else src + "/"
        else:
            _src = self._space_name_to_src(src)
            if _src is None:
                raise ValueError(
                    f"Could not find Space: {src}. If it is a private Space, please provide an hf_token."
                )
            self.space_id = src
        self.src = _src
        state = self._get_space_state()
        if state == SpaceStage.BUILDING:
            if self.verbose:
                print("Space is still building. Please wait...")
            while self._get_space_state() == SpaceStage.BUILDING:
                time.sleep(2)  # so we don't get rate limited by the API
                pass
        if state in utils.INVALID_RUNTIME:
            raise ValueError(
                f"The current space is in the invalid state: {state}. "
                "Please contact the owner to fix this."
            )
        if self.verbose:
            print(f"Loaded as API: {self.src} ✔")

        self.api_url = urllib.parse.urljoin(self.src, utils.API_URL)
        self.ws_url = urllib.parse.urljoin(
            self.src.replace("http", "ws", 1), utils.WS_URL
        )
        self.upload_url = urllib.parse.urljoin(self.src, utils.UPLOAD_URL)
        self.reset_url = urllib.parse.urljoin(self.src, utils.RESET_URL)
        self.config = self._get_config()
        self.session_hash = str(uuid.uuid4())

        self.endpoints = [
            Endpoint(self, fn_index, dependency)
            for fn_index, dependency in enumerate(self.config["dependencies"])
        ]

        # Create a pool of threads to handle the requests
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )

        # Disable telemetry by setting the env variable HF_HUB_DISABLE_TELEMETRY=1
        # threading.Thread(target=self._telemetry_thread).start()

        self.server_hash = self.get_server_hash()

        return self

    def get_server_hash(self):
        if self.config is None:
            self.setup()
        """
        Get server hash using super without any refresh action triggered
        Returns: git hash of gradio server
        """
        return super().submit(api_name="/system_hash").result()

    def refresh_client_if_should(self, persist=True):
        if self.config is None:
            self.setup()
        # get current hash in order to update api_name -> fn_index map in case gradio server changed
        # FIXME: Could add cli api as hash
        server_hash = self.get_server_hash()
        if self.server_hash != server_hash:
            # risky to persist if hash changed
            self.refresh_client(persist=False)
            self.server_hash = server_hash
        else:
            if not persist:
                self.reset_session()

    def refresh_client(self, persist=True):
        """
        Ensure every client call is independent
        Also ensure map between api_name and fn_index is updated in case server changed (e.g. restarted with new code)
        Returns:
        """
        if self.config is None:
            self.setup()
        if not persist:
            # need session hash to be new every time, to avoid "generator already executing"
            self.reset_session()

        client = Client(*self.args, **self.kwargs)
        for k, v in client.__dict__.items():
            setattr(self, k, v)

    def clone(self):
        if self.config is None:
            self.setup()
        client = GradioClient("")
        for k, v in self.__dict__.items():
            setattr(client, k, v)
        client.reset_session()
        client.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )
        client.endpoints = [
            Endpoint(client, fn_index, dependency)
            for fn_index, dependency in enumerate(client.config["dependencies"])
        ]
        return client

    def submit(
            self,
            *args,
            api_name: str | None = None,
            fn_index: int | None = None,
            result_callbacks: Callable | list[Callable] | None = None,
    ) -> Job:
        if self.config is None:
            self.setup()
        # Note predict calls submit
        try:
            self.refresh_client_if_should()
            job = super().submit(*args, api_name=api_name, fn_index=fn_index)
        except Exception as e:
            print("Hit e=%s\n\n%s" % (str(e), traceback.format_exc()), flush=True)
            # force reconfig in case only that
            self.refresh_client()
            job = super().submit(*args, api_name=api_name, fn_index=fn_index)

        # see if immediately failed
        e = check_job(job, timeout=0.01, raise_exception=False)
        if e is not None:
            print(
                "GR job failed: %s %s"
                % (str(e), "".join(traceback.format_tb(e.__traceback__))),
                flush=True,
            )
            # force reconfig in case only that
            self.refresh_client()
            job = super().submit(*args, api_name=api_name, fn_index=fn_index)
            e2 = check_job(job, timeout=0.1, raise_exception=False)
            if e2 is not None:
                print(
                    "GR job failed again: %s\n%s"
                    % (str(e2), "".join(traceback.format_tb(e2.__traceback__))),
                    flush=True,
                )

        return job

    def question(self, question, *args, **kwargs) -> str:
        kwargs["instruction"] = question
        kwargs["langchain_action"] = LangChainAction.QUERY.value
        kwargs["langchain_mode"] = 'LLM'
        ret = ''
        for response, texts_out in self.query_or_summarize_or_extract(*args, **kwargs):
            ret = response
        return ret

    def question_stream(self, question, *args, **kwargs) -> str:
        kwargs["instruction"] = question
        kwargs["langchain_action"] = LangChainAction.QUERY.value
        kwargs["langchain_mode"] = 'LLM'
        ret = yield from self.query_or_summarize_or_extract(*args, **kwargs)
        return ret

    def query(self, *args, **kwargs) -> str:
        kwargs["langchain_action"] = LangChainAction.QUERY.value
        ret = ''
        for response, texts_out in self.query_or_summarize_or_extract(*args, **kwargs):
            ret = response
        return ret

    def query_stream(self, *args, **kwargs) -> Generator[tuple[str | list[str], list[str]], None, None]:
        kwargs["langchain_action"] = LangChainAction.QUERY.value
        ret = yield from self.query_or_summarize_or_extract(*args, **kwargs)
        return ret

    def summarize(self, *args, **kwargs) -> str:
        kwargs["langchain_action"] = LangChainAction.SUMMARIZE_MAP.value
        ret = ''
        for response, texts_out in self.query_or_summarize_or_extract(*args, **kwargs):
            ret = response
        return ret

    def summarize_stream(self, *args, **kwargs) -> str:
        kwargs["langchain_action"] = LangChainAction.SUMMARIZE_MAP.value
        ret = yield from self.query_or_summarize_or_extract(*args, **kwargs)
        return ret

    def extract(self, *args, **kwargs) -> list[str]:
        kwargs["langchain_action"] = LangChainAction.EXTRACT.value
        ret = ''
        for response, texts_out in self.query_or_summarize_or_extract(*args, **kwargs):
            ret = response
        return ret

    def extract_stream(self, *args, **kwargs) -> list[str]:
        kwargs["langchain_action"] = LangChainAction.EXTRACT.value
        ret = yield from self.query_or_summarize_or_extract(*args, **kwargs)
        return ret

    def query_or_summarize_or_extract(self,
                                      instruction: str = "",
                                      text: list[str] | str | None = None,
                                      file: list[str] | str | None = None,
                                      url: list[str] | str | None = None,

                                      embed: bool = True,
                                      chunk: bool = True,
                                      chunk_size: int = 512,

                                      langchain_mode: str = None,
                                      langchain_action: str | None = None,
                                      top_k_docs: int = 10,
                                      document_choice: Union[str, List[str]] = "All",
                                      document_subset: str = "Relevant",

                                      system_prompt: str | None = None,
                                      pre_prompt_query: str | None = None,
                                      prompt_query: str | None = None,
                                      pre_prompt_summary: str | None = None,
                                      prompt_summary: str | None = None,
                                      h2ogpt_key: str = None,

                                      llm: str | int | None = None,
                                      llm_args: dict[str, Any] | None = None,
                                      stream_output: bool = False,
                                      do_sample: bool = False,
                                      max_time: int = 360,

                                      chat_conversation: list[tuple[str, str]] | None = None,
                                      text_context_list: list[str] | None = None,
                                      docs_ordering_type: str | None = None,

                                      max_input_tokens: int = -1,
                                      max_new_tokens: int = 1024,
                                      min_max_new_tokens: int = 512,

                                      docs_token_handling: str = "split_or_merge",
                                      docs_joiner: str = "\n\n",

                                      asserts: bool = False,
                                      ) -> Generator[tuple[str | list[str], list[str]], None, None]:
        """
        Query or Summarize or Extract using h2oGPT
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
            chunk: whether to chunk sources for document Q/A
            chunk_size: Size in characters of chunks

            langchain_mode: "LLM" to talk to LLM with no docs, "MyData" for personal docs, "UserData" for shared docs, etc.
            langchain_action: Action to take, "Query" or "Summarize" or "Extract"
            top_k_docs: number of document parts.
                        When doing query, number of chunks
                        When doing summarization, not related to vectorDB chunks that are not used
                        E.g. if PDF, then number of pages
            document_choice: Which documents ("All" means all) -- need to use upload_api API call to get server's name if want to select
            document_subset: Type of query, see src/gen.py
            system_prompt: pass system prompt to models that support it.
              If 'auto' or None, then use automatic version
              If '', then use no system prompt (default)
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
            h2ogpt_key: Access Key to h2oGPT server
            llm: base_model name or integer index of model_lock on h2oGPT server
                            None results in use of first (0th index) model in server
            llm_args: extra kwargs to pass on to GradioClient to h2oGPT
            stream_output: Whether to stream output
            do_sample: whether to sample
            max_time: how long to take

            chat_conversation: List of tuples for (human, bot) conversation that will be pre-appended to an (instruction, None) case for a query
            text_context_list: List of strings to add to context for non-database version of document Q/A for faster handling via API etc.
               Forces LangChain code path and uses as many entries in list as possible given max_seq_len, with first assumed to be most relevant and to go near prompt.
            docs_ordering_type: By default uses 'reverse_ucurve_sort' for optimal retrieval
            max_input_tokens: Max input tokens to place into model context for each LLM call
                                     -1 means auto, fully fill context for query, and fill by original document chunk for summarization
                                     >=0 means use that to limit context filling to that many tokens
            max_new_tokens: Maximum new tokens
            min_max_new_tokens: minimum value for max_new_tokens when auto-adjusting for content of prompt, docs, etc.

            docs_token_handling: 'chunk' means fill context with top_k_docs (limited by max_input_tokens or model_max_len) chunks for query
                                                                             or top_k_docs original document chunks summarization
                                        None or 'split_or_merge' means same as 'chunk' for query, while for summarization merges documents to fill up to max_input_tokens or model_max_len tokens
            docs_joiner: string to join lists of text when doing split_or_merge.  None means '\n\n'

            asserts: whether to do asserts to ensure handling is correct

        Returns: summary/answer: str or extraction List[str]

        """
        client = self.clone()
        h2ogpt_key = h2ogpt_key or self.h2ogpt_key
        client.h2ogpt_key = h2ogpt_key

        try:
            llm = int(llm)
        except:
            pass
        if llm != 0:
            valid_llms = [x["base_model"] for x in self.get_llms()]
            if (
                    isinstance(llm, int)
                    and llm >= len(valid_llms)
                    or isinstance(llm, str)
                    and llm not in valid_llms
            ):
                did_you_mean = ""
                if isinstance(llm, str):
                    alt = difflib.get_close_matches(llm, valid_llms, 1)
                    if alt:
                        did_you_mean = f"\nDid you mean {repr(alt[0])}?"
                raise RuntimeError(
                    f"Invalid llm: {repr(llm)}, must be either an integer between "
                    f"0 and {len(valid_llms) - 1} or one of the following values: {valid_llms}.{did_you_mean}"
                )

        # chunking not used here
        # MyData specifies scratch space, only persisted for this individual client call
        langchain_mode = langchain_mode or "MyData"
        loaders = tuple([None, None, None, None])
        doc_options = tuple([langchain_mode, chunk, chunk_size, embed])
        asserts |= bool(os.getenv("HARD_ASSERTS", False))
        if (
                text
                and isinstance(text, list)
                and not file
                and not url
                and not text_context_list
        ):
            # then can do optimized text-only path
            text_context_list = text
            text = None

        res = []
        if text:
            t0 = time.time()
            res = client.predict(
                text, *doc_options, *loaders, h2ogpt_key, api_name="/add_text"
            )
            t1 = time.time()
            print("upload text: %s" % str(timedelta(seconds=t1 - t0)), flush=True)
            if asserts:
                assert res[0] is None
                assert res[1] == langchain_mode
                assert "user_paste" in res[2]
                assert res[3] == ""
        if file:
            # upload file(s).  Can be list or single file
            # after below call, "file" replaced with remote location of file
            _, file = client.predict(file, api_name="/upload_api")

            res = client.predict(
                file, *doc_options, *loaders, h2ogpt_key, api_name="/add_file_api"
            )
            if asserts:
                assert res[0] is None
                assert res[1] == langchain_mode
                assert os.path.basename(file) in res[2]
                assert res[3] == ""
        if url:
            res = client.predict(
                url, *doc_options, *loaders, h2ogpt_key, api_name="/add_url"
            )
            if asserts:
                assert res[0] is None
                assert res[1] == langchain_mode
                assert url in res[2]
                assert res[3] == ""
                assert res[4]  # should have file name or something similar
        if res and not res[4] and "Exception" in res[2]:
            print("Exception: %s" % res[2], flush=True)

        # ask for summary, need to use same client if using MyData
        api_name = "/submit_nochat_api"  # NOTE: like submit_nochat but stable API for string dict passing
        kwargs = dict(
            instruction=instruction,
            langchain_mode=langchain_mode,
            langchain_action=langchain_action,  # uses full document, not vectorDB chunks
            top_k_docs=top_k_docs,
            stream_output=stream_output,
            document_subset=document_subset,
            document_choice=document_choice,
            max_time=max_time,
            max_new_tokens=max_new_tokens,
            min_max_new_tokens=min_max_new_tokens,
            do_sample=do_sample,
            system_prompt=system_prompt,
            pre_prompt_query=pre_prompt_query,
            prompt_query=prompt_query,
            pre_prompt_summary=pre_prompt_summary,
            prompt_summary=prompt_summary,
            h2ogpt_key=h2ogpt_key,
            visible_models=llm,
            chat_conversation=chat_conversation,
            text_context_list=text_context_list,
            docs_ordering_type=docs_ordering_type,
            max_input_tokens=max_input_tokens,
            docs_token_handling=docs_token_handling,
            docs_joiner=docs_joiner,
        )
        if llm_args:
            for key in llm_args:
                # only allow certain keys in llm_args - has to be supported by h2oGPT
                if key not in [
                    "do_sample",
                    "temperature",
                    "top_p",
                    "top_k",
                    "num_beams",  # not for streaming yet https://github.com/h2oai/h2ogpt/issues/106
                    "repetition_penalty",
                    "max_new_tokens",
                    "min_max_new_tokens",
                    "max_input_tokens",
                    "max_time",
                ]:
                    raise RuntimeError(
                        f"User error, unexpected key '{key}' in 'llm_args'"
                    )
                if key in kwargs:
                    print(
                        "overriding inference parameter %s: %s %s" % (key,
                                                                      kwargs[key],
                                                                      llm_args[key])
                    )
                else:
                    print(
                        f"using custom inference parameter %s: %s" % (key, llm_args[key])
                    )
        kwargs.update(llm_args or {})
        empty = ""
        for thing in [
            "system_prompt",
            "pre_prompt_summary",
            "prompt_summary",
            "pre_prompt_query",
            "prompt_query",
        ]:
            value = kwargs[thing]
            if value in ["", "null", None]:
                empty += f"{thing}={repr(value)}, "

        # get result
        trials = 3
        for trial in range(trials):
            try:
                if not stream_output:
                    res = client.predict(
                        str(dict(kwargs)),
                        api_name=api_name,
                    )
                    res = ast.literal_eval(res)
                    response = res["response"]
                    if langchain_action != LangChainAction.EXTRACT.value:
                        response = response.strip()
                    else:
                        response = [r.strip() for r in response]
                    sources = res["sources"]
                    scores_out = [x["score"] for x in sources]
                    texts_out = [x["content"] for x in sources]
                    if asserts:
                        if text and not file and not url:
                            assert any(
                                text[:cutoff] == texts_out for cutoff in range(len(text))
                            )
                        assert len(texts_out) == len(scores_out)

                    yield response, texts_out
                else:
                    job = client.submit(str(dict(kwargs)), api_name=api_name)
                    text0 = ""
                    response = ""
                    texts_out = []
                    while not job.done():
                        if job.communicator.job.latest_status.code.name == "FINISHED":
                            break
                        e = check_job(job, timeout=0, raise_exception=False)
                        if e is not None:
                            break
                        outputs_list = job.communicator.job.outputs
                        if outputs_list:
                            res = job.communicator.job.outputs[-1]
                            res_dict = ast.literal_eval(res)
                            response = res_dict["response"]  # keeps growing
                            sources = res_dict["sources"]
                            texts_out = [x["content"] for x in sources]
                            text_chunk = response[len(text0):]  # only keep new stuff
                            if not text_chunk:
                                time.sleep(0.001)
                                continue
                            text0 = response
                            assert text_chunk, "must yield non-empty string"
                            yield text_chunk, texts_out
                        time.sleep(
                            0.1
                        )  # let LLM deliver larger chunks, don't need to get every token output immediately

                    # Get final response (if anything left), but also get the actual references (texts_out), above is empty.
                    res_all = job.outputs()
                    if len(res_all) > 0:
                        # 0.1 slightly longer than 0.02 in open source
                        check_job(job, timeout=0.1, raise_exception=True)

                        res = res_all[-1]
                        res_dict = ast.literal_eval(res)
                        response = res_dict["response"]
                        sources = res_dict["sources"]
                        texts_out = [x["content"] for x in sources]
                        assert (
                            response.strip()
                        ), "h2ogpt.py final response 1 must not be empty"
                        yield response[len(text0):].strip(), texts_out
                    else:
                        # 1.0 slightly longer than 0.3 in open source
                        check_job(job, timeout=1.0, raise_exception=True)

                        assert (
                            response.strip()
                        ), "h2ogpt.py final response 2 must not be empty"
                        yield response[len(text0):].strip(), texts_out
                break
            except Exception as e:
                print(
                    "h2oGPT predict failed: %s %s"
                    % (str(e), "".join(traceback.format_tb(e.__traceback__))),
                    flush=True,
                )
                if trial == trials - 1:
                    raise
                else:
                    print("trying again: %s" % trial, flush=True)
                    time.sleep(1 * trial)


    def get_llms(self) -> list[dict[str, Any]]:
        return ast.literal_eval(self.predict(api_name="/model_names"))
