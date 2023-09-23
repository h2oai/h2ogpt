import copy
import types
from typing import Any, Dict, List, Union, Optional
import time
import queue

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


class StreamingGradioCallbackHandler(BaseCallbackHandler):
    """
    Similar to H2OTextIteratorStreamer that is for HF backend, but here LangChain backend
    """
    def __init__(self, timeout: Optional[float] = None, block=True):
        super().__init__()
        self.text_queue = queue.SimpleQueue()
        self.stop_signal = None
        self.do_stop = False
        self.timeout = timeout
        self.block = block

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running. Clean the queue."""
        while not self.text_queue.empty():
            try:
                self.text_queue.get(block=False)
            except queue.Empty:
                continue

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
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
        if db_type == 'chroma':
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

    if db_type == 'chroma':
        # also keep original source for summarization and other tasks

        # assign chunk_id=-1 for original content
        # this assumes, as is currently true, that splitter makes new documents and list and metadata is deepcopy
        [x.metadata.update(dict(chunk_id=-1)) for chunk_id, x in enumerate(sources)]

        # in some cases sources is generator, so convert to list
        return list(sources) + source_chunks
    else:
        return source_chunks
