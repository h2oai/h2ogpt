import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def server_url():
    server_url = os.getenv("H2OGPT_SERVER")
    if not server_url:
        LOGGER.info("Couldn't find a running h2oGPT server. Hence starting a one.")

        generate = _import_module_from_h2ogpt("generate.py")
        generate.main(
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


@pytest.fixture(scope="module")
def eval_func_param_names():
    parameters = _import_module_from_h2ogpt("src/evaluate_params.py")
    return parameters.eval_func_param_names


def _import_module_from_h2ogpt(file_name: str) -> ModuleType:
    h2ogpt_dir = Path(__file__).parent.parent.parent
    file_path = (h2ogpt_dir / file_name).absolute()
    module_name = file_path.stem

    LOGGER.info(f"Loading module '{module_name}' from '{file_path}'.")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not spec:
        raise Exception(f"Couldn't load module '{module_name}' from '{file_path}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module
