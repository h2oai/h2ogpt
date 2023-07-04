import logging
import os

import pytest

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def server_url():
    server_url = os.getenv("H2OGPT_SERVER")
    if not server_url:
        LOGGER.info("Couldn't find running h2oGPT server. Hence starting a one.")
        from generate import main  # type: ignore

        main(
            base_model="h2oai/h2ogpt-oig-oasst1-512-6_9b",
            prompt_type="human_bot",
            chat=False,
            stream_output=False,
            gradio=True,
            num_beams=1,
            block_gradio_exit=False,
        )
        server_url = "http://0.0.0.0:7860"  # assume server started
        LOGGER.info(f"h2oGPT server started at '{server_url}'.")
    return server_url
