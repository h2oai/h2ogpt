from typing import Any
import os

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from gradio_client import Client


class GradioClient(Client):
    """
    Parent class of gradio client
    To handle automatically refreshing client if detect gradio server changed
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_hash = self.get_server_hash()

    def get_server_hash(self):
        """
        Get server hash using super without any refresh action triggered
        Returns: git hash of gradio server
        """
        return super().predict(api_name='/system_hash')

    def refresh_client(self):
        """
        Ensure every client call is independent
        Also ensure map between api_name and fn_index is updated in case server changed (e.g. restarted with new code)
        Returns:
        """

        self.reset_session()

        # get current hash in order to update api_name -> fn_index map in case gradio server changed
        server_hash = self.get_server_hash()
        if self.server_hash != server_hash:
            self._get_config()
            self.server_hash = server_hash

    def predict(
            self,
            *args,
            api_name: str = None,
            fn_index: int = None,
    ) -> Any:
        self.refresh_client()
        return super().predict(*args, api_name=api_name, fn_index=fn_index)
