import asyncio
from typing import Any, Optional

import gradio_client  # type: ignore

from h2ogpt_client._completion import ChatCompletionCreator, TextCompletionCreator
from h2ogpt_client._models import Models
from h2ogpt_client._server import Server


class Client:
    """h2oGPT Client."""

    def __init__(
        self,
        src: str,
        h2ogpt_key: Optional[str] = None,
        huggingface_token: Optional[str] = None,
    ):
        """
        Creates a GPT client.
        :param src: either the full URL to the hosted h2oGPT
            (e.g. "http://0.0.0.0:7860", "https://fc752f297207f01c32.gradio.live")
            or name of the Hugging Face Space to load, (e.g. "h2oai/h2ogpt-chatbot")
        :param h2ogpt_key: access key to connect with a h2oGPT server
        :param huggingface_token: Hugging Face token to use to access private Spaces
        """
        self._client = gradio_client.Client(
            src=src, hf_token=huggingface_token, serialize=False, verbose=False
        )
        self._h2ogpt_key = h2ogpt_key
        self._text_completion = TextCompletionCreator(self)
        self._chat_completion = ChatCompletionCreator(self)
        self._models = Models(self)
        self._server = Server(self)

    @property
    def text_completion(self) -> TextCompletionCreator:
        """Text completion."""
        return self._text_completion

    @property
    def chat_completion(self) -> ChatCompletionCreator:
        """Chat completion."""
        return self._chat_completion

    @property
    def models(self) -> Models:
        """LL models."""
        return self._models

    @property
    def server(self) -> Server:
        """h2oGPT server."""
        return self._server

    def _predict(self, *args, api_name: str) -> Any:
        return self._client.submit(*args, api_name=api_name).result()

    async def _predict_async(self, *args, api_name: str) -> Any:
        return await asyncio.wrap_future(self._client.submit(*args, api_name=api_name))
