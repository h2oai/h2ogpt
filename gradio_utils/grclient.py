import traceback
from typing import Callable
import os

from gradio_client.client import Job

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from gradio_client import Client


class GradioClient(Client):
    """
    Parent class of gradio client
    To handle automatically refreshing client if detect gradio server changed
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)
        self.server_hash = self.get_server_hash()

    def get_server_hash(self):
        """
        Get server hash using super without any refresh action triggered
        Returns: git hash of gradio server
        """
        return super().submit(api_name='/system_hash').result()

    def refresh_client_if_should(self):
        # get current hash in order to update api_name -> fn_index map in case gradio server changed
        # FIXME: Could add cli api as hash
        server_hash = self.get_server_hash()
        if self.server_hash != server_hash:
            self.refresh_client()
            self.server_hash = server_hash
        else:
            self.reset_session()

    def refresh_client(self):
        """
        Ensure every client call is independent
        Also ensure map between api_name and fn_index is updated in case server changed (e.g. restarted with new code)
        Returns:
        """
        # need session hash to be new every time, to avoid "generator already executing"
        self.reset_session()

        client = Client(*self.args, **self.kwargs)
        for k, v in client.__dict__.items():
            setattr(self, k, v)

    def submit(
        self,
        *args,
        api_name: str | None = None,
        fn_index: int | None = None,
        result_callbacks: Callable | list[Callable] | None = None,
    ) -> Job:
        # Note predict calls submit
        try:
            self.refresh_client_if_should()
            job = super().submit(*args, api_name=api_name, fn_index=fn_index)
        except Exception as e:
            print("Hit e=%s" % str(e), flush=True)
            # force reconfig in case only that
            self.refresh_client()
            job = super().submit(*args, api_name=api_name, fn_index=fn_index)

        # see if immediately failed
        e = job.future._exception
        if e is not None:
            print("GR job failed: %s %s" % (str(e), ''.join(traceback.format_tb(e.__traceback__))), flush=True)
            # force reconfig in case only that
            self.refresh_client()
            job = super().submit(*args, api_name=api_name, fn_index=fn_index)
            e2 = job.future._exception
            if e2 is not None:
                print("GR job failed again: %s\n%s" % (str(e2), ''.join(traceback.format_tb(e2.__traceback__))), flush=True)

        return job
