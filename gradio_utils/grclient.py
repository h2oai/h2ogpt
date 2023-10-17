from __future__ import annotations
import traceback
import concurrent.futures
import os
import concurrent.futures
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Callable

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from huggingface_hub import SpaceStage
from huggingface_hub.utils import (
    build_hf_headers,
)

from gradio_client import utils
from gradio_client.client import Job, DEFAULT_TEMP_DIR, Endpoint
from gradio_client import Client


class GradioClient(Client):
    """
    Parent class of gradio client
    To handle automatically refreshing client if detect gradio server changed
    """

    def __init__(
        self,
        src: str,
        hf_token: str | None = None,
        max_workers: int = 40,
        serialize: bool = True,
        output_dir: str | Path | None = DEFAULT_TEMP_DIR,
        verbose: bool = True,
    ):
        """
        Parameters:
            src: Either the name of the Hugging Face Space to load, (e.g. "abidlabs/whisper-large-v2") or the full URL (including "http" or "https") of the hosted Gradio app to load (e.g. "http://mydomain.com/app" or "https://bec81a83-5b5c-471e.gradio.live/").
            hf_token: The Hugging Face token to use to access private Spaces. Automatically fetched if you are logged in via the Hugging Face Hub CLI. Obtain from: https://huggingface.co/settings/token
            max_workers: The maximum number of thread workers that can be used to make requests to the remote Gradio app simultaneously.
            serialize: Whether the client should serialize the inputs and deserialize the outputs of the remote API. If set to False, the client will pass the inputs and outputs as-is, without serializing/deserializing them. E.g. you if you set this to False, you'd submit an image in base64 format instead of a filepath, and you'd get back an image in base64 format from the remote API instead of a filepath.
            output_dir: The directory to save files that are downloaded from the remote API. If None, reads from the GRADIO_TEMP_DIR environment variable. Defaults to a temporary directory on your machine.
            verbose: Whether the client should print statements to the console.
        """
        self.args = tuple([src])
        self.kwargs = dict(
            hf_token=hf_token,
            max_workers=max_workers,
            serialize=serialize,
            output_dir=output_dir,
            verbose=verbose,
        )

        self.verbose = verbose
        self.hf_token = hf_token
        self.serialize = serialize
        self.space_id = None
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.src = src
        self.config = None
        self.server_hash = None

    def __repr__(self):
        if self.config:
            return self.view_api(print_info=False, return_format="str")
        return "Not setup for %s" % self.src

    def __str__(self):
        if self.config:
            return self.view_api(print_info=False, return_format="str")
        return "Not setup for %s" % self.src

    def setup(self):
        src = self.src

        self.headers = build_hf_headers(
            token=self.hf_token,
            library_name="gradio_client",
            library_version=utils.__version__,
        )
        if src.startswith("http://") or src.startswith("https://"):
            _src = src if src.endswith("/") else src + "/"
        else:
            _src = self._space_name_to_src(src)
            if _src is None:
                raise ValueError(
                    f"Could not find Space: {src}. If it is a private Space, please provide an hf_token."
                )
            self.space_id = src
        self.src = _src
        state = self._get_space_state()
        if state == SpaceStage.BUILDING:
            if self.verbose:
                print("Space is still building. Please wait...")
            while self._get_space_state() == SpaceStage.BUILDING:
                time.sleep(2)  # so we don't get rate limited by the API
                pass
        if state in utils.INVALID_RUNTIME:
            raise ValueError(
                f"The current space is in the invalid state: {state}. "
                "Please contact the owner to fix this."
            )
        if self.verbose:
            print(f"Loaded as API: {self.src} ✔")

        self.api_url = urllib.parse.urljoin(self.src, utils.API_URL)
        self.ws_url = urllib.parse.urljoin(
            self.src.replace("http", "ws", 1), utils.WS_URL
        )
        self.upload_url = urllib.parse.urljoin(self.src, utils.UPLOAD_URL)
        self.reset_url = urllib.parse.urljoin(self.src, utils.RESET_URL)
        self.config = self._get_config()
        self.session_hash = str(uuid.uuid4())

        self.endpoints = [
            Endpoint(self, fn_index, dependency)
            for fn_index, dependency in enumerate(self.config["dependencies"])
        ]

        # Create a pool of threads to handle the requests
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )

        # Disable telemetry by setting the env variable HF_HUB_DISABLE_TELEMETRY=1
        # threading.Thread(target=self._telemetry_thread).start()

        self.server_hash = self.get_server_hash()

        return self

    def get_server_hash(self):
        if self.config is None:
            self.setup()
        """
        Get server hash using super without any refresh action triggered
        Returns: git hash of gradio server
        """
        return super().submit(api_name="/system_hash").result()

    def refresh_client_if_should(self, persist=True):
        if self.config is None:
            self.setup()
        # get current hash in order to update api_name -> fn_index map in case gradio server changed
        # FIXME: Could add cli api as hash
        server_hash = self.get_server_hash()
        if self.server_hash != server_hash:
            # risky to persist if hash changed
            self.refresh_client(persist=False)
            self.server_hash = server_hash
        else:
            if not persist:
                self.reset_session()

    def refresh_client(self, persist=True):
        """
        Ensure every client call is independent
        Also ensure map between api_name and fn_index is updated in case server changed (e.g. restarted with new code)
        Returns:
        """
        if self.config is None:
            self.setup()
        if not persist:
            # need session hash to be new every time, to avoid "generator already executing"
            self.reset_session()

        client = Client(*self.args, **self.kwargs)
        for k, v in client.__dict__.items():
            setattr(self, k, v)

    def clone(self):
        if self.config is None:
            self.setup()
        client = GradioClient("")
        for k, v in self.__dict__.items():
            setattr(client, k, v)
        client.reset_session()
        client.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        )
        client.endpoints = [
            Endpoint(client, fn_index, dependency)
            for fn_index, dependency in enumerate(client.config["dependencies"])
        ]
        return client

    def submit(
        self,
        *args,
        api_name: str | None = None,
        fn_index: int | None = None,
        result_callbacks: Callable | list[Callable] | None = None,
    ) -> Job:
        if self.config is None:
            self.setup()
        # Note predict calls submit
        try:
            self.refresh_client_if_should()
            job = super().submit(*args, api_name=api_name, fn_index=fn_index)
        except Exception as e:
            print("Hit e=%s\n\n%s" % (str(e), traceback.format_exc()), flush=True)
            # force reconfig in case only that
            self.refresh_client()
            job = super().submit(*args, api_name=api_name, fn_index=fn_index)

        # see if immediately failed
        e = job.future._exception
        if e is not None:
            print(
                "GR job failed: %s %s"
                % (str(e), "".join(traceback.format_tb(e.__traceback__))),
                flush=True,
            )
            # force reconfig in case only that
            self.refresh_client()
            job = super().submit(*args, api_name=api_name, fn_index=fn_index)
            e2 = job.future._exception
            if e2 is not None:
                print(
                    "GR job failed again: %s\n%s"
                    % (str(e2), "".join(traceback.format_tb(e2.__traceback__))),
                    flush=True,
                )

        return job
