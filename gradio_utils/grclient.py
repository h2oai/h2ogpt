from __future__ import annotations

import difflib
import traceback
import os
import concurrent.futures
import time
import urllib.parse
import uuid
from concurrent.futures import Future
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Generator, Any, Union, List
import ast
from packaging import version

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from huggingface_hub import SpaceStage
from huggingface_hub.utils import (
    build_hf_headers,
)

from gradio_client import utils

from importlib.metadata import distribution, PackageNotFoundError

try:
    assert distribution('gradio_client') is not None
    have_gradio_client = True
    from packaging import version
    client_version = distribution('gradio_client').version
    is_gradio_client_version7plus = version.parse(client_version) >= version.parse("0.7.0")
except (PackageNotFoundError, AssertionError):
    have_gradio_client = False
    is_gradio_client_version7plus = False

from gradio_client.client import Job, DEFAULT_TEMP_DIR, Endpoint
from gradio_client import Client


def check_job(job, timeout=0.0, raise_exception=True, verbose=False):
    try:
        e = job.exception(timeout=timeout)
    except concurrent.futures.TimeoutError:
        # not enough time to determine
        if verbose:
            print("not enough time to determine job status: %s" % timeout)
        e = None
    if e:
        # raise before complain about empty response if some error hit
        if raise_exception:
            raise RuntimeError(traceback.format_exception(e))
        else:
            return e


# Local copy of minimal version from h2oGPT server
class LangChainAction(Enum):
    """LangChain action"""

    QUERY = "Query"
    SUMMARIZE_MAP = "Summarize"
    EXTRACT = "Extract"


pre_prompt_query0 = "Pay attention and remember the information below, which will help to answer the question or imperative after the context ends."
prompt_query0 = "According to only the information in the document sources provided within the context above: "

pre_prompt_summary0 = """"""
prompt_summary0 = "Using only the information in the document sources above, write a condensed and concise summary of key results (preferably as bullet points)."

pre_prompt_extraction0 = """In order to extract information, pay attention to the following text."""
prompt_extraction0 = "Using only the information in the document sources above, extract "

hyde_llm_prompt0 = "Answer this question with vibrant details in order for some NLP embedding model to use that answer as better query than original question: "


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
            serialize: bool = None,
            output_dir: str | Path | None = DEFAULT_TEMP_DIR,
            verbose: bool = False,
            auth: tuple[str, str] | None = None,
            h2ogpt_key: str = None,
            persist: bool = False,
            check_hash: bool = True,
            check_model_name: bool = False,
    ):
        """
        Parameters:
            src: Either the name of the Hugging Face Space to load, (e.g. "abidlabs/whisper-large-v2") or the full URL (including "http" or "https") of the hosted Gradio app to load (e.g. "http://mydomain.com/app" or "https://bec81a83-5b5c-471e.gradio.live/").
            hf_token: The Hugging Face token to use to access private Spaces. Automatically fetched if you are logged in via the Hugging Face Hub CLI. Obtain from: https://huggingface.co/settings/token
            max_workers: The maximum number of thread workers that can be used to make requests to the remote Gradio app simultaneously.
            serialize: Whether the client should serialize the inputs and deserialize the outputs of the remote API. If set to False, the client will pass the inputs and outputs as-is, without serializing/deserializing them. E.g. you if you set this to False, you'd submit an image in base64 format instead of a filepath, and you'd get back an image in base64 format from the remote API instead of a filepath.
            output_dir: The directory to save files that are downloaded from the remote API. If None, reads from the GRADIO_TEMP_DIR environment variable. Defaults to a temporary directory on your machine.
            verbose: Whether the client should print statements to the console.

            h2ogpt_key: h2oGPT key to gain access to the server
            persist: whether to persist the state, so repeated calls are aware of the prior user session
                     This allows the scratch MyData to be reused, etc.
                     This also maintains the chat_conversation history
            check_hash: whether to check git hash for consistency between server and client to ensure API always up to date
            check_model_name: whether to check the model name here (adds delays), or just let server fail (fater)
        """
        if serialize is None:
            # else converts inputs arbitrarily and outputs mutate
            # False keeps as-is and is normal for h2oGPT
            serialize = False
        self.args = tuple([src])
        self.kwargs = dict(
            hf_token=hf_token,
            max_workers=max_workers,
            serialize=serialize,
            output_dir=output_dir,
            verbose=verbose,
            h2ogpt_key=h2ogpt_key,
            persist=persist,
            check_hash=check_hash,
            check_model_name=check_model_name,
        )
        if is_gradio_client_version7plus:
            self.kwargs.update(dict(auth=auth))

        self.verbose = verbose
        self.hf_token = hf_token
        self.serialize = serialize
        self.space_id = None
        self.cookies: dict[str, str] = {}
        if is_gradio_client_version7plus:
            self.output_dir = (
                str(output_dir) if isinstance(output_dir, Path) else output_dir
            )
        else:
            self.output_dir = output_dir
        self.max_workers = max_workers
        self.src = src
        self.auth = auth
        self.config = None
        self.h2ogpt_key = h2ogpt_key
        self.persist = persist
        self.check_hash = check_hash
        self.check_model_name = check_model_name

        self.chat_conversation = []  # internal for persist=True
        self.server_hash = None  # internal

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
        # self.headers.pop('authorization', None)  # else get illegal Bearer for old servers
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

        if is_gradio_client_version7plus:
            if self.auth is not None:
                self._login(self.auth)

        self.api_url = urllib.parse.urljoin(self.src, utils.API_URL)
        if is_gradio_client_version7plus:
            self.sse_url = urllib.parse.urljoin(self.src, utils.SSE_URL)
            self.sse_data_url = urllib.parse.urljoin(self.src, utils.SSE_DATA_URL)
        self.ws_url = urllib.parse.urljoin(
            self.src.replace("http", "ws", 1), utils.WS_URL
        )
        self.upload_url = urllib.parse.urljoin(self.src, utils.UPLOAD_URL)
        self.reset_url = urllib.parse.urljoin(self.src, utils.RESET_URL)
        self.config = self._get_config()
        if is_gradio_client_version7plus:
            self.protocol: str = self.config.get("protocol", "ws")
            self.app_version = version.parse(self.config.get("version", "2.0"))
            self._info = self._get_api_info()
        self.session_hash = str(uuid.uuid4())

        if is_gradio_client_version7plus:
            from gradio_client.client import EndpointV3Compatibility
            endpoint_class = (
                Endpoint if self.protocol.startswith("sse") else EndpointV3Compatibility
            )
        else:
            endpoint_class = Endpoint

        if is_gradio_client_version7plus:
            self.endpoints = [
                endpoint_class(self, fn_index, dependency, self.protocol)
                for fn_index, dependency in enumerate(self.config["dependencies"])
            ]
        else:
            self.endpoints = [
                endpoint_class(self, fn_index, dependency)
                for fn_index, dependency in enumerate(self.config["dependencies"])
            ]

        # Create a pool of threads to handle the requests
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )

        # Disable telemetry by setting the env variable HF_HUB_DISABLE_TELEMETRY=1
        # threading.Thread(target=self._telemetry_thread).start()

        self.server_hash = self.get_server_hash()

        if is_gradio_client_version7plus:
            self.stream_open = False
            self.streaming_future: Future | None = None
            from gradio_client.utils import Message
            self.pending_messages_per_event: dict[str, list[Message | None]] = {}
            self.pending_event_ids: set[str] = set()

        return self

    def get_server_hash(self):
        if self.config is None:
            self.setup()
        """
        Get server hash using super without any refresh action triggered
        Returns: git hash of gradio server
        """
        if self.check_hash:
            return super().submit(api_name="/system_hash").result()
        else:
            return "GET_GITHASH"

    def refresh_client_if_should(self):
        if self.config is None:
            self.setup()
        # get current hash in order to update api_name -> fn_index map in case gradio server changed
        # FIXME: Could add cli api as hash
        server_hash = self.get_server_hash()
        if self.server_hash != server_hash:
            if self.verbose:
                print("server hash changed: %s %s" % (self.server_hash, server_hash), flush=True)
            if self.server_hash is not None and self.persist:
                if self.verbose:
                    print(
                        "Failed to persist due to server hash change, only kept chat_conversation not user session hash",
                        flush=True)
            # risky to persist if hash changed
            self.refresh_client()
            self.server_hash = server_hash

    def refresh_client(self):
        """
        Ensure every client call is independent
        Also ensure map between api_name and fn_index is updated in case server changed (e.g. restarted with new code)
        Returns:
        """
        if self.config is None:
            self.setup()

        kwargs = self.kwargs.copy()
        kwargs.pop('h2ogpt_key', None)
        kwargs.pop('persist', None)
        kwargs.pop('check_hash', None)
        kwargs.pop('check_model_name', None)
        ntrials = 3
        client = None
        for trial in range(0, ntrials + 1):
            try:
                client = Client(*self.args, **kwargs)
            except ValueError as e:
                if trial >= ntrials:
                    raise
                else:
                    if self.verbose:
                        print("Trying refresh %d/%d %s" % (trial, ntrials - 1, str(e)))
                    trial += 1
                    time.sleep(10)
        if client is None:
            raise RuntimeError("Failed to get new client")
        session_hash0 = self.session_hash if self.persist else None
        for k, v in client.__dict__.items():
            setattr(self, k, v)
        if session_hash0:
            # keep same system hash in case server API only changed and not restarted
            self.session_hash = session_hash0
        if self.verbose:
            print("Hit refresh_client(): %s %s" % (self.session_hash, session_hash0))
        # ensure server hash also updated
        self.server_hash = self.get_server_hash()

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
        # transfer internals in case used
        client.server_hash = self.server_hash
        client.chat_conversation = self.chat_conversation
        return client

    def submit(
            self,
            *args,
            api_name: str | None = None,
            fn_index: int | None = None,
            result_callbacks: Callable | list[Callable] | None = None,
            exception_handling=True,  # new_stream = True, can make False, doesn't matter.
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

        if exception_handling: # for debugging if causes issues
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

    def question(self, instruction, *args, **kwargs) -> str:
        """
        Prompt LLM (direct to LLM with instruct prompting required for instruct models) and get response
        """
        kwargs["instruction"] = kwargs.get("instruction", instruction)
        kwargs["langchain_action"] = LangChainAction.QUERY.value
        kwargs["langchain_mode"] = 'LLM'
        ret = ''
        for response, texts_out in self.query_or_summarize_or_extract(*args, **kwargs):
            ret = response
        return ret

    def question_stream(self, instruction, *args, **kwargs) -> str:
        """
        Prompt LLM (direct to LLM with instruct prompting required for instruct models) and get response
        """
        kwargs["instruction"] = kwargs.get("instruction", instruction)
        kwargs["langchain_action"] = LangChainAction.QUERY.value
        kwargs["langchain_mode"] = 'LLM'
        ret = yield from self.query_or_summarize_or_extract(*args, **kwargs)
        return ret

    def query(self, query, *args, **kwargs) -> str:
        """
        Search for documents matching a query, then ask that query to LLM with those documents
        """
        kwargs["instruction"] = kwargs.get("instruction", query)
        kwargs["langchain_action"] = LangChainAction.QUERY.value
        ret = ''
        for response, texts_out in self.query_or_summarize_or_extract(*args, **kwargs):
            ret = response
        return ret

    def query_stream(self, query, *args, **kwargs) -> Generator[tuple[str | list[str], list[str]], None, None]:
        """
        Search for documents matching a query, then ask that query to LLM with those documents
        """
        kwargs["instruction"] = kwargs.get("instruction", query)
        kwargs["langchain_action"] = LangChainAction.QUERY.value
        ret = yield from self.query_or_summarize_or_extract(*args, **kwargs)
        return ret

    def summarize(self, *args, query=None, focus=None, **kwargs) -> str:
        """
        Search for documents matching a focus, then ask a query to LLM with those documents
        If focus "" or None, no similarity search is done and all documents (up to top_k_docs) are used
        """
        kwargs["prompt_summary"] = kwargs.get("prompt_summary", query or prompt_summary0)
        kwargs["instruction"] = kwargs.get('instruction', focus)
        kwargs["langchain_action"] = LangChainAction.SUMMARIZE_MAP.value
        ret = ''
        for response, texts_out in self.query_or_summarize_or_extract(*args, **kwargs):
            ret = response
        return ret

    def summarize_stream(self, *args, query=None, focus=None, **kwargs) -> str:
        """
        Search for documents matching a focus, then ask a query to LLM with those documents
        If focus "" or None, no similarity search is done and all documents (up to top_k_docs) are used
        """
        kwargs["prompt_summary"] = kwargs.get("prompt_summary", query or prompt_summary0)
        kwargs["instruction"] = kwargs.get('instruction', focus)
        kwargs["langchain_action"] = LangChainAction.SUMMARIZE_MAP.value
        ret = yield from self.query_or_summarize_or_extract(*args, **kwargs)
        return ret

    def extract(self, *args, query=None, focus=None, **kwargs) -> list[str]:
        """
        Search for documents matching a focus, then ask a query to LLM with those documents
        If focus "" or None, no similarity search is done and all documents (up to top_k_docs) are used
        """
        kwargs["prompt_extraction"] = kwargs.get("prompt_extraction", query or prompt_extraction0)
        kwargs["instruction"] = kwargs.get('instruction', focus)
        kwargs["langchain_action"] = LangChainAction.EXTRACT.value
        ret = ''
        for response, texts_out in self.query_or_summarize_or_extract(*args, **kwargs):
            ret = response
        return ret

    def extract_stream(self, *args, query=None, focus=None, **kwargs) -> list[str]:
        """
        Search for documents matching a focus, then ask a query to LLM with those documents
        If focus "" or None, no similarity search is done and all documents (up to top_k_docs) are used
        """
        kwargs["prompt_extraction"] = kwargs.get("prompt_extraction", query or prompt_extraction0)
        kwargs["instruction"] = kwargs.get('instruction', focus)
        kwargs["langchain_action"] = LangChainAction.EXTRACT.value
        ret = yield from self.query_or_summarize_or_extract(*args, **kwargs)
        return ret

    def query_or_summarize_or_extract(self,
                                      h2ogpt_key: str = None,

                                      instruction: str = "",

                                      text: list[str] | str | None = None,
                                      file: list[str] | str | None = None,
                                      url: list[str] | str | None = None,
                                      embed: bool = True,
                                      chunk: bool = True,
                                      chunk_size: int = 512,

                                      langchain_mode: str = None,
                                      langchain_action: str | None = None,
                                      langchain_agents: List[str] = [],
                                      top_k_docs: int = 10,
                                      document_choice: Union[str, List[str]] = "All",
                                      document_subset: str = "Relevant",
                                      document_source_substrings: Union[str, List[str]] = [],
                                      document_source_substrings_op: str = 'and',
                                      document_content_substrings: Union[str, List[str]] = [],
                                      document_content_substrings_op: str = 'and',

                                      system_prompt: str | None = '',
                                      pre_prompt_query: str | None = pre_prompt_query0,
                                      prompt_query: str | None = prompt_query0,
                                      pre_prompt_summary: str | None = pre_prompt_summary0,
                                      prompt_summary: str | None = prompt_summary0,
                                      pre_prompt_extraction: str | None = pre_prompt_extraction0,
                                      prompt_extraction: str | None = prompt_extraction0,
                                      hyde_llm_prompt: str | None = hyde_llm_prompt0,

                                      model: str | int | None = None,
                                      stream_output: bool = False,
                                      do_sample: bool = False,
                                      temperature: float = 0.0,
                                      top_p: float = 0.75,
                                      top_k: int = 40,
                                      repetition_penalty: float = 1.07,
                                      penalty_alpha: float = 0.0,
                                      max_time: int = 360,
                                      max_new_tokens: int = 1024,

                                      add_search_to_context: bool = False,
                                      chat_conversation: list[tuple[str, str]] | None = None,
                                      text_context_list: list[str] | None = None,
                                      docs_ordering_type: str | None = None,
                                      min_max_new_tokens: int = 512,
                                      max_input_tokens: int = -1,
                                      max_total_input_tokens: int = -1,
                                      docs_token_handling: str = "split_or_merge",
                                      docs_joiner: str = "\n\n",
                                      hyde_level: int = 0,
                                      hyde_template: str = None,
                                      hyde_show_only_final: bool = True,
                                      doc_json_mode: bool = False,

                                      asserts: bool = False,
                                      ) -> Generator[tuple[str | list[str], list[str]], None, None]:
        """
        Query or Summarize or Extract using h2oGPT
        Args:
            instruction: Query for LLM chat.  Used for similarity search

            For query, prompt template is:
              "{pre_prompt_query}
                \"\"\"
                {content}
                \"\"\"
                {prompt_query}{instruction}"
             If added to summarization, prompt template is
              "{pre_prompt_summary}
                \"\"\"
                {content}
                \"\"\"
                Focusing on {instruction}, {prompt_summary}"
            text: textual content or list of such contents
            file: a local file to upload or files to upload
            url: a url to give or urls to use
            embed: whether to embed content uploaded

            langchain_mode: "LLM" to talk to LLM with no docs, "MyData" for personal docs, "UserData" for shared docs, etc.
            langchain_action: Action to take, "Query" or "Summarize" or "Extract"
            langchain_agents: Which agents to use, if any
            top_k_docs: number of document parts.
                        When doing query, number of chunks
                        When doing summarization, not related to vectorDB chunks that are not used
                        E.g. if PDF, then number of pages
            chunk: whether to chunk sources for document Q/A
            chunk_size: Size in characters of chunks
            document_choice: Which documents ("All" means all) -- need to use upload_api API call to get server's name if want to select
            document_subset: Type of query, see src/gen.py
            document_source_substrings: See gen.py
            document_source_substrings_op: See gen.py
            document_content_substrings: See gen.py
            document_content_substrings_op: See gen.py

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
                template = "%s
                \"\"\"
                %s
                \"\"\"
                %s" % (pre_prompt_summary, fstring, prompt_summary)
            hyde_llm_prompt: hyde prompt for first step when using LLM
            h2ogpt_key: Access Key to h2oGPT server (if not already set in client at init time)
            model: base_model name or integer index of model_lock on h2oGPT server
                            None results in use of first (0th index) model in server
                   to get list of models do client.list_models()
            pre_prompt_extraction: Same as pre_prompt_summary but for when doing extraction
            prompt_extraction: Same as prompt_summary but for when doing extraction
            do_sample: see src/gen.py
            temperature: see src/gen.py
            top_p: see src/gen.py
            top_k: see src/gen.py
            repetition_penalty: see src/gen.py
            penalty_alpha: see src/gen.py
            max_new_tokens: see src/gen.py
            min_max_new_tokens: see src/gen.py
            max_input_tokens: see src/gen.py
            max_total_input_tokens: see src/gen.py

            stream_output: Whether to stream output
            do_sample: whether to sample
            max_time: how long to take

            add_search_to_context: Whether to do web search and add results to context
            chat_conversation: List of tuples for (human, bot) conversation that will be pre-appended to an (instruction, None) case for a query
            text_context_list: List of strings to add to context for non-database version of document Q/A for faster handling via API etc.
               Forces LangChain code path and uses as many entries in list as possible given max_seq_len, with first assumed to be most relevant and to go near prompt.
            docs_ordering_type: By default uses 'reverse_ucurve_sort' for optimal retrieval
            max_input_tokens: Max input tokens to place into model context for each LLM call
                                     -1 means auto, fully fill context for query, and fill by original document chunk for summarization
                                     >=0 means use that to limit context filling to that many tokens
            max_total_input_tokens: like max_input_tokens but instead of per LLM call, applies across all LLM calls for single summarization/extraction action
            max_new_tokens: Maximum new tokens
            min_max_new_tokens: minimum value for max_new_tokens when auto-adjusting for content of prompt, docs, etc.

            docs_token_handling: 'chunk' means fill context with top_k_docs (limited by max_input_tokens or model_max_len) chunks for query
                                                                             or top_k_docs original document chunks summarization
                                        None or 'split_or_merge' means same as 'chunk' for query, while for summarization merges documents to fill up to max_input_tokens or model_max_len tokens
            docs_joiner: string to join lists of text when doing split_or_merge.  None means '\n\n'
            hyde_level: 0-3 for HYDE.
                        0 uses just query to find similarity with docs
                        1 uses query + pure LLM response to find similarity with docs
                        2: uses query + LLM response using docs to find similarity with docs
                        3+: etc.
            hyde_template: see src/gen.py
            hyde_show_only_final: see src/gen.py
            doc_json_mode: see src/gen.py

            asserts: whether to do asserts to ensure handling is correct

        Returns: summary/answer: str or extraction List[str]

        """
        if self.config is None:
            self.setup()
        if self.persist:
            client = self
        else:
            client = self.clone()
        h2ogpt_key = h2ogpt_key or self.h2ogpt_key
        client.h2ogpt_key = h2ogpt_key

        self.check_model(model)

        # chunking not used here
        # MyData specifies scratch space, only persisted for this individual client call
        langchain_mode = langchain_mode or "MyData"
        loaders = tuple([None, None, None, None, None, None])
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

        pre_prompt_summary = pre_prompt_summary \
            if langchain_action == LangChainAction.SUMMARIZE_MAP.value \
            else pre_prompt_extraction
        prompt_summary = prompt_summary \
            if langchain_action == LangChainAction.SUMMARIZE_MAP.value \
            else prompt_extraction

        kwargs = dict(
            h2ogpt_key=h2ogpt_key,

            instruction=instruction,

            langchain_mode=langchain_mode,
            langchain_action=langchain_action,  # uses full document, not vectorDB chunks
            langchain_agents=langchain_agents,
            top_k_docs=top_k_docs,
            document_choice=document_choice,
            document_subset=document_subset,
            document_source_substrings=document_source_substrings,
            document_source_substrings_op=document_source_substrings_op,
            document_content_substrings=document_content_substrings,
            document_content_substrings_op=document_content_substrings_op,

            system_prompt=system_prompt,
            pre_prompt_query=pre_prompt_query,
            prompt_query=prompt_query,
            pre_prompt_summary=pre_prompt_summary,
            prompt_summary=prompt_summary,
            hyde_llm_prompt=hyde_llm_prompt,

            visible_models=model,
            stream_output=stream_output,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            penalty_alpha=penalty_alpha,
            max_time=max_time,
            max_new_tokens=max_new_tokens,

            add_search_to_context=add_search_to_context,
            chat_conversation=chat_conversation if chat_conversation else self.chat_conversation,
            text_context_list=text_context_list,
            docs_ordering_type=docs_ordering_type,
            min_max_new_tokens=min_max_new_tokens,
            max_input_tokens=max_input_tokens,
            max_total_input_tokens=max_total_input_tokens,
            docs_token_handling=docs_token_handling,
            docs_joiner=docs_joiner,
            hyde_level=hyde_level,
            hyde_template=hyde_template,
            hyde_show_only_final=hyde_show_only_final,
            doc_json_mode=doc_json_mode,
        )

        # in case server changed, update in case clone()
        self.server_hash = client.server_hash

        # ensure can fill conversation
        self.chat_conversation.append((instruction, None))

        # get result
        trials = 3
        for trial in range(trials):
            try:
                if not stream_output:
                    res = client.predict(
                        str(dict(kwargs)),
                        api_name=api_name,
                    )
                    # in case server changed, update in case clone()
                    self.server_hash = client.server_hash
                    res = ast.literal_eval(res)
                    response = res["response"]
                    if langchain_action != LangChainAction.EXTRACT.value:
                        response = response.strip()
                    else:
                        response = [r.strip() for r in ast.literal_eval(response)]
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
                    self.chat_conversation[-1] = (instruction, response)
                else:
                    job = client.submit(str(dict(kwargs)), api_name=api_name)
                    text0 = ""
                    response = ""
                    texts_out = []
                    while not job.done():
                        e = check_job(job, timeout=0, raise_exception=False)
                        if e is not None:
                            break
                        outputs_list = job.outputs().copy()
                        if outputs_list:
                            res = outputs_list[-1]
                            res_dict = ast.literal_eval(res)
                            response = res_dict["response"]  # keeps growing
                            text_chunk = response[len(text0):]  # only keep new stuff
                            if not text_chunk:
                                time.sleep(0.001)
                                continue
                            text0 = response
                            assert text_chunk, "must yield non-empty string"
                            yield text_chunk, texts_out
                        time.sleep(0.01)

                    # Get final response (if anything left), but also get the actual references (texts_out), above is empty.
                    res_all = job.outputs().copy()
                    if len(res_all) > 0:
                        # 0.1 slightly longer than 0.02 in open source
                        check_job(job, timeout=0.1, raise_exception=True)

                        res = res_all[-1]
                        res_dict = ast.literal_eval(res)
                        response = res_dict["response"]
                        sources = res_dict["sources"]
                        texts_out = [x["content"] for x in sources]
                        yield response[len(text0):], texts_out
                        self.chat_conversation[-1] = (instruction, response[len(text0):])
                    else:
                        # 1.0 slightly longer than 0.3 in open source
                        check_job(job, timeout=1.0, raise_exception=True)
                        yield response[len(text0):], texts_out
                        self.chat_conversation[-1] = (instruction, response[len(text0):])
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
            finally:
                # in case server changed, update in case clone()
                self.server_hash = client.server_hash

    def check_model(self, model):
        if model != 0 and self.check_model_name:
            valid_llms = self.list_models()
            if (
                    isinstance(model, int)
                    and model >= len(valid_llms)
                    or isinstance(model, str)
                    and model not in valid_llms
            ):
                did_you_mean = ""
                if isinstance(model, str):
                    alt = difflib.get_close_matches(model, valid_llms, 1)
                    if alt:
                        did_you_mean = f"\nDid you mean {repr(alt[0])}?"
                raise RuntimeError(
                    f"Invalid llm: {repr(model)}, must be either an integer between "
                    f"0 and {len(valid_llms) - 1} or one of the following values: {valid_llms}.{did_you_mean}"
                )

    def get_models_full(self) -> list[dict[str, Any]]:
        """
        Full model info in list if dict
        """
        if self.config is None:
            self.setup()
        return ast.literal_eval(self.predict(api_name="/model_names"))

    def list_models(self) -> list[str]:
        """
        Model names available from endpoint
        """
        if self.config is None:
            self.setup()
        return [x['base_model'] for x in ast.literal_eval(self.predict(api_name="/model_names"))]

    def stream(self,
               client_kwargs,
               api_name='/submit_nochat_api',
               prompt='', prompter=None, sanitize_bot_response=False,
               max_time=None,
               is_public=False,
               raise_exception=True,
               verbose=False):
        strex = ''
        e = None
        res_dict = {}
        try:
            res_dict = yield from self._stream(client_kwargs,
                                               api_name=api_name,
                                               prompt=prompt,
                                               prompter=prompter,
                                               sanitize_bot_response=sanitize_bot_response,
                                               max_time=max_time,
                                               verbose=verbose)
        except Exception as e:
            strex = ''.join(traceback.format_tb(e.__traceback__))
            # check validity of final results and check for timeout
            # NOTE: server may have more before its timeout, and res_all will have more if waited a bit
            if raise_exception:
                raise

        if 'timeout' in res_dict['save_dict']['extra_dict']:
            timeout_time = res_dict['save_dict']['extra_dict']['timeout']
            raise TimeoutError("Timeout from local after %s %s" % (timeout_time, ': ' + strex if e else ''))

        # won't have sources if timed out
        if res_dict.get('sources') is None:
            # then communication terminated, keep what have, but send error
            if is_public:
                raise ValueError("Abrupt termination of communication")
            else:
                raise ValueError("Abrupt termination of communication: %s" % strex)
        return res_dict

    def _stream(self,
                client_kwargs,
                api_name='/submit_nochat_api',
                prompt='', prompter=None, sanitize_bot_response=False,
                max_time=None,
                verbose=False):
        job = self.submit(str(dict(client_kwargs)), api_name=api_name)

        text = ''
        sources = []
        save_dict = {}
        save_dict['extra_dict'] = {}
        res_dict = dict(response=text, sources=sources, save_dict=save_dict, llm_answers={},
                        response_no_refs=text, sources_str='', prompt_raw='')
        yield res_dict

        text0 = ''
        tgen0 = time.time()
        n = 0
        for res in job:
            res_dict, text0 = yield from self.yield_res(res, res_dict, prompt, prompter, sanitize_bot_response,
                                                        max_time, text0, tgen0, verbose)
            n += 1
            if 'timeout' in res_dict['save_dict']['extra_dict']:
                break
        # final res
        outputs = job.outputs().copy()
        all_n = len(outputs)
        for nn in range(n, all_n):
            res = outputs[nn]
            res_dict, text0 = yield from self.yield_res(res, res_dict, prompt, prompter, sanitize_bot_response,
                                                        max_time, text0, tgen0, verbose)
        return res_dict

    @staticmethod
    def yield_res(res, res_dict, prompt, prompter, sanitize_bot_response, max_time, text0, tgen0, verbose):
        do_yield = True
        res_dict_server = ast.literal_eval(res)
        # yield what have
        text = res_dict_server['response']
        if prompter:
            response = prompter.get_response(prompt + text, prompt=prompt,
                                             sanitize_bot_response=sanitize_bot_response)
        else:
            response = text
        text_chunk = response[len(text0):]
        if not text_chunk:
            # just need some sleep for threads to switch
            time.sleep(0.001)
            do_yield = False
        # save old
        text0 = response
        res_dict.update(res_dict_server)
        res_dict.update(dict(response=response, response_no_refs=response))

        timeout_time_other = res_dict.get('save_dict', {}).get('extra_dict', {}).get('timeout')
        if timeout_time_other:
            if verbose:
                print("Took too long for other Gradio: %s" % (time.time() - tgen0), flush=True)
            return res_dict, text0

        timeout_time = time.time() - tgen0
        if max_time is not None and timeout_time > max_time:
            if 'save_dict' not in res_dict:
                res_dict['save_dict'] = {}
            if 'extra_dict' not in res_dict['save_dict']:
                res_dict['save_dict']['extra_dict'] = {}
            res_dict['save_dict']['extra_dict']['timeout'] = timeout_time
            yield res_dict
            if verbose:
                print("Took too long for Gradio: %s" % (time.time() - tgen0), flush=True)
            return res_dict, text0
        if do_yield:
            yield res_dict
            time.sleep(0.01)
        return res_dict, text0
