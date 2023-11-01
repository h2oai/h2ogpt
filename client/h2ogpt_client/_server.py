from h2ogpt_client import _core


class Server:
    """h2oGPT server."""

    def __init__(self, client: "_core.Client"):
        self._client = client

    @property
    def address(self) -> str:
        """h2oGPT server address."""
        return self._client._client.src

    @property
    def hash(self) -> str:
        """h2oGPT server system hash."""
        return str(self._client._predict(api_name="/system_hash"))
