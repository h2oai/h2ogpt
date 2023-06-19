from typing import Any, Dict, List, Union, Optional
import time
import queue

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


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
