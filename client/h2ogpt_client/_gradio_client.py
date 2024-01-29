import asyncio
import time
from typing import Any, AsyncGenerator, Generator, List, Optional

import gradio_client  # type: ignore


class GradioClientWrapper:
    def __init__(
        self,
        src: str,
        h2ogpt_key: Optional[str] = None,
        huggingface_token: Optional[str] = None,
    ):
        self._client = gradio_client.Client(
            src=src, hf_token=huggingface_token, serialize=False, verbose=False
        )
        self.h2ogpt_key = h2ogpt_key

    def predict(self, *args, api_name: str) -> Any:
        return self._client.predict(*args, api_name=api_name)

    def predict_and_stream(self, *args, api_name: str) -> Generator[str, None, None]:
        job = self._client.submit(*args, api_name=api_name)
        while not job.done():
            outputs: List[str] = job.outputs().copy()
            if not len(outputs):
                time.sleep(0.1)
                continue
            newest_response = outputs[-1]
            yield newest_response

        e = job.exception()
        if e and isinstance(e, BaseException):
            raise RuntimeError from e

    async def submit(self, *args, api_name: str) -> Any:
        return await asyncio.wrap_future(self._client.submit(*args, api_name=api_name))

    async def submit_and_stream(
        self, *args, api_name: str
    ) -> AsyncGenerator[Any, None]:
        job = self._client.submit(*args, api_name=api_name)
        while not job.done():
            outputs: List[str] = job.outputs().copy()
            if not len(outputs):
                await asyncio.sleep(0.1)
                continue
            newest_response = outputs[-1]
            yield newest_response

        e = job.exception()
        if e and isinstance(e, BaseException):
            raise RuntimeError from e
