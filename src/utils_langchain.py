import copy
import os
import types
import uuid
from typing import Any, Dict, List, Union, Optional, Tuple, Mapping
import time
import queue
import pathlib
from datetime import datetime

from langchain.schema import BasePromptTemplate
from langchain.chains import LLMChain
from langchain.chains import MapReduceDocumentsChain, StuffDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.summarize import map_reduce_prompt, LoadingCallable, _load_stuff_chain, _load_map_reduce_chain, \
    _load_refine_chain
from langchain.schema.language_model import BaseLanguageModel

from src.utils import hash_file, get_sha

from langchain.callbacks.base import BaseCallbackHandler, Callbacks
from langchain.schema import LLMResult
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


class StreamingGradioCallbackHandler(BaseCallbackHandler):
    """
    Similar to H2OTextIteratorStreamer that is for HF backend, but here LangChain backend
    """

    def __init__(self, timeout: Optional[float] = None, block=True, max_time=None, verbose=False):
        super().__init__()
        self.text_queue = queue.SimpleQueue()
        self.stop_signal = None
        self.do_stop = False
        self.timeout = timeout
        self.block = block
        self.max_time = max_time
        self.tgen0 = None
        self.verbose = verbose

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        self.tgen0 = time.time()
        """Run when LLM starts running. Clean the queue."""
        while not self.text_queue.empty():
            try:
                self.text_queue.get(block=False)
            except queue.Empty:
                continue

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if self.tgen0 is not None and self.max_time is not None and (time.time() - self.tgen0) > self.max_time:
            if self.verbose:
                print("Took too long in StreamingGradioCallbackHandler: %s" % (time.time() - self.tgen0), flush=True)
            self.text_queue.put(self.stop_signal)
        else:
            self.text_queue.put(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.text_queue.put(self.stop_signal)

    def on_llm_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        self.text_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                value = self.stop_signal  # value looks unused in pycharm, not true
                if self.do_stop:
                    print("hit stop", flush=True)
                    # could raise or break, maybe best to raise and make parent see if any exception in thread
                    raise StopIteration()
                    # break
                value = self.text_queue.get(block=self.block, timeout=self.timeout)
                break
            except queue.Empty:
                time.sleep(0.01)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


def _chunk_sources(sources, chunk=True, chunk_size=512, language=None, db_type=None):
    assert db_type is not None

    if not isinstance(sources, (list, tuple, types.GeneratorType)) and not callable(sources):
        # if just one document
        sources = [sources]
    if not chunk:
        [x.metadata.update(dict(chunk_id=0)) for chunk_id, x in enumerate(sources)]
        if db_type in ['chroma', 'chroma_old']:
            # make copy so can have separate summarize case
            source_chunks = [Document(page_content=x.page_content,
                                      metadata=copy.deepcopy(x.metadata) or {})
                             for x in sources]
        else:
            source_chunks = sources  # just same thing
    else:
        if language and False:
            # Bug in langchain, keep separator=True not working
            # https://github.com/hwchase17/langchain/issues/2836
            # so avoid this for now
            keep_separator = True
            separators = RecursiveCharacterTextSplitter.get_separators_for_language(language)
        else:
            separators = ["\n\n", "\n", " ", ""]
            keep_separator = False
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, keep_separator=keep_separator,
                                                  separators=separators)
        source_chunks = splitter.split_documents(sources)

        # currently in order, but when pull from db won't be, so mark order and document by hash
        [x.metadata.update(dict(chunk_id=chunk_id)) for chunk_id, x in enumerate(source_chunks)]

    if db_type in ['chroma', 'chroma_old']:
        # also keep original source for summarization and other tasks

        # assign chunk_id=-1 for original content
        # this assumes, as is currently true, that splitter makes new documents and list and metadata is deepcopy
        [x.metadata.update(dict(chunk_id=-1)) for chunk_id, x in enumerate(sources)]

        # in some cases sources is generator, so convert to list
        return list(sources) + source_chunks
    else:
        return source_chunks


def add_parser(docs1, parser):
    [x.metadata.update(dict(parser=x.metadata.get('parser', parser))) for x in docs1]


def _add_meta(docs1, file, headsize=50, filei=0, parser='NotSet'):
    if os.path.isfile(file):
        file_extension = pathlib.Path(file).suffix
        hashid = hash_file(file)
    else:
        file_extension = str(file)  # not file, just show full thing
        hashid = get_sha(file)
    doc_hash = str(uuid.uuid4())[:10]
    if not isinstance(docs1, (list, tuple, types.GeneratorType)):
        docs1 = [docs1]
    [x.metadata.update(dict(input_type=file_extension,
                            parser=x.metadata.get('parser', parser),
                            date=str(datetime.now()),
                            time=time.time(),
                            order_id=order_id,
                            hashid=hashid,
                            doc_hash=doc_hash,
                            file_id=filei,
                            head=x.page_content[:headsize].strip())) for order_id, x in enumerate(docs1)]


def fix_json_meta(docs1):
    if not isinstance(docs1, (list, tuple, types.GeneratorType)):
        docs1 = [docs1]
    # fix meta, chroma doesn't like None, only str, int, float for values
    [x.metadata.update(dict(sender_name=x.metadata.get('sender_name') or '')) for x in docs1]
    [x.metadata.update(dict(timestamp_ms=x.metadata.get('timestamp_ms') or '')) for x in docs1]


class H2OMapReduceDocumentsChain(MapReduceDocumentsChain):
    def combine_docs(
            self,
            docs: List[Document],
            token_max: Optional[int] = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> Tuple[List, dict]:
        """Combine documents in a map reduce manner.

        Combine by mapping first chain over all documents, then reducing the results.
        This reducing can be done recursively if needed (if there are many documents).
        """
        map_results = self.llm_chain.apply(
            # FYI - this is parallelized and so it is fast.
            [{self.document_variable_name: d.page_content, **kwargs} for d in docs],
            callbacks=callbacks,
        )
        question_result_key = self.llm_chain.output_key
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
            # This uses metadata from the docs, and the textual results from `results`
            for i, r in enumerate(map_results)
        ]
        extra_return_dict = {}
        if self.return_intermediate_steps:
            intermediate_steps = [r[question_result_key] for r in map_results]
            extra_return_dict["intermediate_steps"] = intermediate_steps
        result_docs_content = [x.page_content for x in result_docs]
        return result_docs_content, extra_return_dict

    async def acombine_docs(
            self,
            docs: List[Document],
            token_max: Optional[int] = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> Tuple[List, dict]:
        """Combine documents in a map reduce manner.

        Combine by mapping first chain over all documents, then reducing the results.
        This reducing can be done recursively if needed (if there are many documents).
        """
        map_results = await self.llm_chain.aapply(
            # FYI - this is parallelized and so it is fast.
            [{**{self.document_variable_name: d.page_content}, **kwargs} for d in docs],
            callbacks=callbacks,
        )
        question_result_key = self.llm_chain.output_key
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
            # This uses metadata from the docs, and the textual results from `results`
            for i, r in enumerate(map_results)
        ]
        extra_return_dict = {}
        if self.return_intermediate_steps:
            intermediate_steps = [r[question_result_key] for r in map_results]
            extra_return_dict["intermediate_steps"] = intermediate_steps
        result_docs_content = [x.page_content for x in result_docs]
        return result_docs_content, extra_return_dict

    @property
    def _chain_type(self) -> str:
        return "map_documents_chain"


def _load_map_chain(
        llm: BaseLanguageModel,
        map_prompt: BasePromptTemplate = map_reduce_prompt.PROMPT,
        combine_prompt: BasePromptTemplate = map_reduce_prompt.PROMPT,
        combine_document_variable_name: str = "text",
        map_reduce_document_variable_name: str = "text",
        collapse_prompt: Optional[BasePromptTemplate] = None,
        reduce_llm: Optional[BaseLanguageModel] = None,
        collapse_llm: Optional[BaseLanguageModel] = None,
        verbose: Optional[bool] = None,
        token_max: int = 3000,
        callbacks: Callbacks = None,
        **kwargs: Any,
) -> H2OMapReduceDocumentsChain:
    map_chain = LLMChain(
        llm=llm, prompt=map_prompt, verbose=verbose, callbacks=callbacks
    )
    _reduce_llm = reduce_llm or llm
    reduce_chain = LLMChain(
        llm=_reduce_llm, prompt=combine_prompt, verbose=verbose, callbacks=callbacks
    )
    # TODO: document prompt
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name=combine_document_variable_name,
        verbose=verbose,
        callbacks=callbacks,
    )
    if collapse_prompt is None:
        collapse_chain = None
        if collapse_llm is not None:
            raise ValueError(
                "collapse_llm provided, but collapse_prompt was not: please "
                "provide one or stop providing collapse_llm."
            )
    else:
        _collapse_llm = collapse_llm or llm
        collapse_chain = StuffDocumentsChain(
            llm_chain=LLMChain(
                llm=_collapse_llm,
                prompt=collapse_prompt,
                verbose=verbose,
                callbacks=callbacks,
            ),
            document_variable_name=combine_document_variable_name,
        )
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=collapse_chain,
        token_max=token_max,
        verbose=verbose,
        callbacks=callbacks,
    )
    return H2OMapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name=map_reduce_document_variable_name,
        verbose=verbose,
        callbacks=callbacks,
        **kwargs,
    )


def load_general_summarization_chain(
        llm: BaseLanguageModel,
        chain_type: str = "stuff",
        verbose: Optional[bool] = None,
        **kwargs: Any,
) -> BaseCombineDocumentsChain:
    """Load summarizing chain.

    Args:
        llm: Language Model to use in the chain.
        chain_type: Type of document combining chain to use. Should be one of "stuff",
            "map_reduce", and "refine".
        verbose: Whether chains should be run in verbose mode or not. Note that this
            applies to all chains that make up the final chain.

    Returns:
        A chain to use for summarizing.
    """
    loader_mapping: Mapping[str, LoadingCallable] = {
        "stuff": _load_stuff_chain,
        "map_reduce": _load_map_reduce_chain,
        "refine": _load_refine_chain,
        "map": _load_map_chain,
    }
    if chain_type not in loader_mapping:
        raise ValueError(
            f"Got unsupported chain type: {chain_type}. "
            f"Should be one of {loader_mapping.keys()}"
        )
    return loader_mapping[chain_type](llm, verbose=verbose, **kwargs)


"""Utils for interacting with the Semantic Scholar API."""
import logging
from typing import Any, Dict, Optional

from langchain_core.pydantic_v1 import BaseModel, root_validator

logger = logging.getLogger(__name__)


class H2OSemanticScholarAPIWrapper(BaseModel):
    """Wrapper around semanticscholar.org API.
    https://github.com/danielnsilva/semanticscholar

    You should have this library installed.

    `pip install semanticscholar`

    Semantic Scholar API can conduct searches and fetch document metadata
    like title, abstract, authors, etc.

    Attributes:
    top_k_results: number of the top-scored document used for the Semantic Scholar tool
    load_max_docs: a limit to the number of loaded documents

    Example:
    .. code-block:: python

    from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
    ss = SemanticScholarAPIWrapper(
        top_k_results = 3,
        load_max_docs = 3
    )
    ss.run("biases in large language models")
    """

    semanticscholar_search: Any  #: :meta private:
    top_k_results: int = 5
    S2_MAX_QUERY_LENGTH: int = 300
    load_max_docs: int = 100
    doc_content_chars_max: Optional[int] = 4000
    returned_fields = [
        "title",
        "abstract",
        "venue",
        "year",
        "paperId",
        "citationCount",
        "openAccessPdf",
        "authors",
        "externalIds",
    ]

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in environment."""
        try:
            from semanticscholar import SemanticScholar

            sch = SemanticScholar(api_key=os.getenv('S2_API_KEY'))
            values["semanticscholar_search"] = sch.search_paper
        except ImportError:
            raise ImportError(
                "Could not import Semanticscholar python package. "
                "Please install it with `pip install semanticscholar`."
            )
        return values

    def run(self, query: str) -> str:
        """Run the Semantic Scholar API."""
        results = self.semanticscholar_search(
            query, limit=self.load_max_docs, fields=self.returned_fields
        )
        documents = []
        for item in results[: self.top_k_results]:
            authors = ", ".join(
                author["name"] for author in getattr(item, "authors", [])
            )
            documents.append(
                f"Published year: {getattr(item, 'year', None)}\n"
                f"Title: {getattr(item, 'title', None)}\n"
                f"Authors: {authors}\n"
                f"Astract: {getattr(item, 'abstract', None)}\n"
            )

        if documents:
            return "\n\n".join(documents)[: self.doc_content_chars_max]
        else:
            return "No results found."
