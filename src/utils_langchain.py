import copy
import functools
import json
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
from langchain.chains.summarize import map_reduce_prompt, LoadingCallable
from langchain.chains.summarize.chain import _load_stuff_chain, _load_refine_chain
from langchain.schema.language_model import BaseLanguageModel
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_text_splitters import TextSplitter

from enums import docs_joiner_default
from utils import hash_file, get_sha, split_list, makedirs, flatten_list, get_token_count, get_docs_tokens, \
    FakeTokenizer

from langchain.callbacks.base import BaseCallbackHandler, Callbacks
from langchain.schema import LLMResult
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


class StreamingGradioCallbackHandler(BaseCallbackHandler):
    """
    Similar to H2OTextIteratorStreamer that is for HF backend, but here LangChain backend
    """

    def __init__(self, timeout: Optional[float] = None, block=True, max_time=None, verbose=False, raise_stop=True):
        super().__init__()
        self.text_queue = queue.SimpleQueue()
        self.stop_signal = None
        self.do_stop = False
        self.timeout = timeout
        self.block = block
        self.max_time = max_time
        self.tgen0 = None
        self.verbose = verbose
        self.raise_stop = raise_stop

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
        if False and \
                self.tgen0 is not None and self.max_time is not None and (time.time() - self.tgen0) > self.max_time:
            if self.verbose:
                print("Took too long in StreamingGradioCallbackHandler: %s" % (time.time() - self.tgen0), flush=True)
            self.text_queue.put(self.stop_signal)
            self.do_stop = True
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
                time.sleep(0.005)
        if value == self.stop_signal:
            if self.raise_stop:
                raise StopIteration()
            return None
        else:
            return value


class H2OCharacterTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = False,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(separators=separators, keep_separator=keep_separator, is_separator_regex=is_separator_regex,
                         **kwargs)
        self._separators = separators or ["\n\n", "\n", "  ", " ", ""]

    @classmethod
    def from_huggingface_tokenizer(cls, tokenizer: Any, **kwargs: Any) -> TextSplitter:
        def _huggingface_tokenizer_length(text: str) -> int:
            return get_token_count(text, tokenizer, add_special_tokens=False)

        return cls(length_function=_huggingface_tokenizer_length, **kwargs)


def select_docs_with_score(docs_with_score, top_k_docs, one_doc_size):
    if top_k_docs > 0:
        docs_with_score = docs_with_score[:top_k_docs]
    elif one_doc_size is not None:
        docs_with_score = [(docs_with_score[0][:one_doc_size], docs_with_score[0][1])]
    else:
        # do nothing
        pass
    return docs_with_score


def split_merge_docs(docs_with_score, tokenizer=None, max_input_tokens=None, docs_token_handling=None,
                     joiner=docs_joiner_default,
                     non_doc_prompt='',
                     do_split=True,
                     hf_embedding_model=None,
                     use_openai_embedding=False,
                     verbose=False):
    # group docs if desired/can to fill context to avoid multiple LLM calls or too large chunks
    # only do first semantic split if have GPU
    if hf_embedding_model and \
            'model' in hf_embedding_model and \
            not use_openai_embedding and \
            hasattr(hf_embedding_model['model'], 'model_kwargs'):
        do_first_semantic_split = hf_embedding_model['model'].model_kwargs.get('device') not in ['cpu']
    else:
        do_first_semantic_split = False

    # NOTE: Could use joiner=\n\n, but if PDF and continues, might want just  full continue with joiner=''
    # NOTE: assume max_input_tokens already processed if was -1 and accounts for model_max_len and is per-llm call
    if max_input_tokens is not None:
        max_input_tokens -= get_token_count(non_doc_prompt, tokenizer)

    if docs_token_handling in ['chunk']:
        return docs_with_score, 0
    elif docs_token_handling in [None, 'split_or_merge']:
        assert tokenizer
        # see if need to split
        # account for joiner tokens
        joiner_tokens = get_token_count(joiner, tokenizer)
        doc_chunk_size = max(64, min(max_input_tokens,
                                     max(64, max_input_tokens - joiner_tokens * len(docs_with_score))))

        if do_first_semantic_split and hf_embedding_model is not None and 'model' in hf_embedding_model:
            # https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/semantic-chunker/
            from langchain_experimental.text_splitter import SemanticChunker
            text_splitter0 = SemanticChunker(hf_embedding_model['model'])
        else:
            text_splitter0 = None

        # skip split if not necessary, since expensive for some reason
        text_splitter1 = H2OCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer, chunk_size=doc_chunk_size, chunk_overlap=0,
            separators=[". "], strip_whitespace=False,
        )
        text_splitter2 = H2OCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer, chunk_size=doc_chunk_size, chunk_overlap=0, strip_whitespace=False,
        )
        # https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/
        text_splitter3 = H2OCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer, chunk_size=doc_chunk_size, chunk_overlap=0, strip_whitespace=False,
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",
            ],
        )
        text_splitter4 = RecursiveCharacterTextSplitter(chunk_size=4 * doc_chunk_size, chunk_overlap=0)

        text_splitters = dict(semantic=text_splitter0, sentence=text_splitter1, normal=text_splitter2,
                              multilingual=text_splitter3, backup=text_splitter4)
        text_splitters = {k: v for k, v in text_splitters.items() if v is not None}

        did_split = False
        for splitter_type, text_splitter in text_splitters.items():
            # don't include joiner with x, because this is each part, not joined part
            tokens_before_split = [get_token_count(x, tokenizer) for x in
                                   [x[0].page_content for x in docs_with_score]]

            do_split &= any([x > max_input_tokens for x in tokens_before_split])
            if not do_split:
                break
            did_split = True

            if verbose:
                print('tokens_before_split=%s' % tokens_before_split, flush=True)

            [x[0].metadata.update(dict(docscore=x[1], doci=doci, ntokens=tokens_before_split[doci])) for doci, x in
             enumerate(docs_with_score)]
            docs = [x[0] for x in docs_with_score]
            # only split those that need to be split, else recursive splitter goes too nuts and takes too long
            docs_to_split = [x for x in docs if x.metadata['ntokens'] > doc_chunk_size]
            docs_to_not_split = [x for x in docs if x.metadata['ntokens'] <= doc_chunk_size]
            docs_split_new = flatten_list([text_splitter.split_documents([x]) for x in docs_to_split])
            docs_new = docs_to_not_split + docs_split_new
            doci_new = [x.metadata['doci'] for x in docs_new]
            # order back by doci
            docs_new = [x for _, x in sorted(zip(doci_new, docs_new), key=lambda pair: pair[0])]
            docs_with_score = [(x, x.metadata['docscore']) for x in docs_new]

            if verbose:
                # don't include joiner with x, because this is each part, not joined part
                tokens_after_split = [get_token_count(x, tokenizer) for x in
                                      [x[0].page_content for x in docs_with_score]]
                print('tokens_after_split=%s' % tokens_after_split, flush=True)

            if splitter_type == 'sentence' and len(docs_with_score) > 1:
                # puts '. ' on next end of chunk, re-attach to end of previous chunk
                docs_with_score = [
                    (Document(x[0].page_content[2 if xi > 0 else 0:] + '.', metadata=x[0].metadata), x[1]) for xi, x in
                    enumerate(docs_with_score)]

        docs_with_score_new = []
        k = 0
        while k < len(docs_with_score):
            # means use max_input_tokens to ensure model gets no more than max_input_tokens each map
            top_k_docs, one_doc_size, num_doc_tokens = \
                get_docs_tokens(tokenizer,
                                text_context_list=[x[0].page_content for x in docs_with_score[k:]],
                                max_input_tokens=max_input_tokens)
            docs_with_score1 = select_docs_with_score(docs_with_score[k:], top_k_docs, one_doc_size)
            new_score = docs_with_score1[0][1]
            new_page_content = joiner.join([x[0].page_content for x in docs_with_score1])
            new_metadata = docs_with_score1[0][0].metadata.copy()
            # keep source as single file so can look up, leave source_merged with joined version
            if len(docs_with_score1) > 1:
                [new_metadata.update({'source_merged_%s' % xi: x[0].metadata['source']}) for xi, x in
                 enumerate(docs_with_score1)]
            new_metadata['source'] = [x[0].metadata['source'] for x in docs_with_score1][0]
            doc1 = Document(page_content=new_page_content, metadata=new_metadata)
            docs_with_score_new.append((doc1, new_score))

            strict_fail = False  # don't strictly fail, sometimes can't split due to separators, so best can
            if strict_fail and did_split:
                assert one_doc_size is None or one_doc_size == 0, "Split failed: %s" % one_doc_size
            elif one_doc_size is not None:
                # chopped
                assert top_k_docs == 1
            assert top_k_docs >= 1
            k += top_k_docs

        # don't include joiner with x, because this is each part, not joined part
        tokens_after_merge = [get_token_count(x, tokenizer) for x in
                              [x[0].page_content for x in docs_with_score_new]]
        if verbose:
            print('tokens_after_merge=%s' % tokens_after_merge, flush=True)

        max_tokens_after_merge = max(tokens_after_merge) if tokens_after_merge else 0
        return docs_with_score_new, max_tokens_after_merge
    else:
        raise ValueError("No such docs_token_handling=%s" % docs_token_handling)


def _chunk_sources(sources, chunk=True, chunk_size=512, language=None, db_type=None,
                   new_splitter=True, hf_embedding_model=None, use_openai_embedding=False, verbose=False):
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
        if not new_splitter:
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0,
                                                      keep_separator=keep_separator,
                                                      separators=separators)
            source_chunks = splitter.split_documents(sources)
        else:
            try:
                tokenizer = FakeTokenizer(model_max_length=max(20, chunk_size - 50), is_super_fake=True)
                sources_with_score = [(x, 1) for x in sources]
                source_chunks_with_score, max_tokens_after_merge = \
                    split_merge_docs(sources_with_score, tokenizer=tokenizer,
                                     max_input_tokens=chunk_size, non_doc_prompt='',
                                     do_split=True,
                                     hf_embedding_model=hf_embedding_model if not use_openai_embedding else None,
                                     verbose=verbose)
                source_chunks = [x[0] for x in source_chunks_with_score]
            except BaseException as e:
                if os.getenv('HARD_ASSERTS'):
                    raise
                print("Failed to split with new method, use old method: %s" % str(e))
                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0,
                                                          keep_separator=keep_separator,
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


def _add_meta(docs1, file, headsize=50, filei=0, parser='NotSet', file_as_source=False):
    if os.path.isfile(file):
        file_extension = pathlib.Path(file).suffix
        hashid = hash_file(file)
    else:
        file_extension = str(type(file))
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
    if file_as_source:
        [x.metadata.update(dict(source=file)) for order_id, x in enumerate(docs1)]


def fix_json_meta(docs1):
    if not isinstance(docs1, (list, tuple, types.GeneratorType)):
        docs1 = [docs1]
    # fix meta, chroma doesn't like None, only str, int, float for values
    [x.metadata.update(dict(sender_name=x.metadata.get('sender_name') or '')) for x in docs1]
    [x.metadata.update(dict(timestamp_ms=x.metadata.get('timestamp_ms') or '')) for x in docs1]


class H2OMapReduceDocumentsChain(MapReduceDocumentsChain):
    allow_map_1 = True
    which = 'map'

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
        if self.which == 'map' or len(result_docs) == 1 and self.allow_map_1:
            extra_return_dict = {}
            if self.return_intermediate_steps:
                intermediate_steps = [r[question_result_key] for r in map_results]
                extra_return_dict["intermediate_steps"] = intermediate_steps
            result = [x.page_content for x in result_docs]
            if self.which == 'map_reduce':
                result = result[0]
        else:
            result, extra_return_dict = self.reduce_documents_chain.combine_docs(
                result_docs, token_max=token_max, callbacks=callbacks, **kwargs
            )
            if self.return_intermediate_steps:
                intermediate_steps = [r[question_result_key] for r in map_results]
                extra_return_dict["intermediate_steps"] = intermediate_steps
        self.terminate_callbacks()
        return result, extra_return_dict

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
        if self.which == 'map' or len(result_docs) == 1 and self.allow_map_1:
            extra_return_dict = {}
            if self.return_intermediate_steps:
                intermediate_steps = [r[question_result_key] for r in map_results]
                extra_return_dict["intermediate_steps"] = intermediate_steps
            result = [x.page_content for x in result_docs]
            if self.which == 'map_reduce':
                result = result[0]
        else:
            result, extra_return_dict = await self.reduce_documents_chain.acombine_docs(
                result_docs, token_max=token_max, callbacks=callbacks, **kwargs
            )
            if self.return_intermediate_steps:
                intermediate_steps = [r[question_result_key] for r in map_results]
                extra_return_dict["intermediate_steps"] = intermediate_steps
        self.terminate_callbacks()
        return result, extra_return_dict

    def terminate_callbacks(self):
        if self.llm_chain.llm.callbacks:
            for callback in self.llm_chain.llm.callbacks:
                if isinstance(callback, StreamingGradioCallbackHandler):
                    if not callback.raise_stop or not callback.do_stop:
                        callback.raise_stop = True
                        # callback.on_llm_end(response)
                        callback.text_queue.put(None)

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
        allow_map_1=map_prompt == combine_prompt,
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
        "map_reduce": functools.partial(_load_map_chain, which='map_reduce'),
        "refine": _load_refine_chain,
        "map": functools.partial(_load_map_chain, which='map'),
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


class H2OHuggingFaceHubEmbeddings(HuggingFaceHubEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to HuggingFaceHub's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        # replace newlines, which can negatively affect performance.
        max_tokens = 512
        # should be less than --max-client-batch-size=4096 for launching TEI
        # shoudl also be that max_tokens * 4 * max_batch_size <= 2MB
        max_batch_size = int(os.getenv('TEI_MAX_BATCH_SIZE', '1024'))
        verbose = False

        texts = [text.replace("\n", " ")[:4 * max_tokens] for text in texts]
        # don't leave empty
        texts = [text or ' ' for text in texts]
        _model_kwargs = self.model_kwargs or {}

        texts_batches = split_list(texts, max_batch_size)
        rets = []
        batchii = 0
        for ii, text_batch in enumerate(texts_batches):
            if verbose:
                print("begin batch %s for texts %s of batch size %s" % (ii, len(texts), len(text_batch)), flush=True)
            responses = self.client.post(
                json={"inputs": text_batch, "truncate": True, "parameters": _model_kwargs}, task=self.task
            )
            rets.extend(json.loads(responses.decode()))
            batchii += len(text_batch)
            if verbose:
                print("done batch %s %s %s" % (ii, len(text_batch), batchii), flush=True)
        return rets


def make_sources_file(langchain_mode, source_files_added):
    sources_dir = "sources_dir"
    sources_dir = makedirs(sources_dir, exist_ok=True, tmp_ok=True, use_base=True)
    sources_file = os.path.join(sources_dir, 'sources_%s_%s' % (langchain_mode, str(uuid.uuid4())))
    with open(sources_file, "wt", encoding="utf-8") as f:
        f.write(source_files_added)
    return sources_file
