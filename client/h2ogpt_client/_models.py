import ast
from typing import Any, Dict, List

from h2ogpt_client._gradio_client import GradioClientWrapper


class Model:
    """Large language model in the h2oGPT server."""

    def __init__(self, raw_info: Dict[str, Any]):
        self._name = raw_info["base_model"]
        self._raw_info = raw_info

    @property
    def name(self) -> str:
        """Name of the model."""
        return self._name

    def __repr__(self) -> str:
        return self.name.__repr__()

    def __str__(self) -> str:
        return self.name.__str__()


class Models:
    """Interact with LL Models in h2oGPT."""

    def __init__(self, client: GradioClientWrapper):
        self._client = client

    def list(self) -> List[Model]:
        """List all models available in the h2oGPT server."""
        models = ast.literal_eval(self._client.predict(api_name="/model_names"))
        return [Model(m) for m in models]
