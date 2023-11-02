from typing import Optional

from h2ogpt_client._completion import ChatCompletionCreator, TextCompletionCreator
from h2ogpt_client._gradio_client import GradioClientWrapper
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
        self._client = GradioClientWrapper(src, h2ogpt_key, huggingface_token)
        self._text_completion = TextCompletionCreator(self._client)
        self._chat_completion = ChatCompletionCreator(self._client)
        self._models = Models(self._client)
        self._server = Server(self._client)

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
