import asyncio
from typing import Any, Optional

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

    async def submit(self, *args, api_name: str) -> Any:
        return await asyncio.wrap_future(self._client.submit(*args, api_name=api_name))
