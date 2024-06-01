import inspect
import json
import os
import subprocess
import sys
import argparse
import logging
import typing
import uuid
from threading import Thread
from typing import Union

import uvicorn
from fastapi import FastAPI

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.path.append('openai_server')


def run_server(host: str = '0.0.0.0',
               port: int = 5000,
               ssl_certfile: str = None,
               ssl_keyfile: str = None,
               gradio_prefix: str = None,
               gradio_host: str = None,
               gradio_port: str = None,
               h2ogpt_key: str = None,
               auth: Union[typing.List[typing.Tuple[str, str]], str] = None,
               auth_access: str = 'open',
               guest_name: str = '',
               # https://docs.gunicorn.org/en/stable/design.html#how-many-workers
               workers: int = 1,
               app: Union[str, FastAPI] = None,
               is_openai_server: bool = True,
               multiple_workers_gunicorn: bool = False,
               main_kwargs: str = "",  # json.dumped dict
               ):
    if workers == 0:
        workers = min(16, os.cpu_count() * 2 + 1)
    assert app is not None

    name = 'OpenAI' if is_openai_server else 'Function'

    os.environ['GRADIO_PREFIX'] = gradio_prefix or 'http'
    os.environ['GRADIO_SERVER_HOST'] = gradio_host or 'localhost'
    os.environ['GRADIO_SERVER_PORT'] = gradio_port or '7860'
    if h2ogpt_key == 'None':
        h2ogpt_key = None
    os.environ['GRADIO_H2OGPT_H2OGPT_KEY'] = h2ogpt_key or ''  # don't use H2OGPT_H2OGPT_KEY, mixes things up
    # use h2ogpt_key if no server api key, so OpenAI inherits key by default if any keys set and enforced via API for h2oGPT
    # but OpenAI key cannot be '', so dummy value is EMPTY and if EMPTY we ignore the key in authorization
    server_api_key = os.getenv('H2OGPT_OPENAI_API_KEY', os.environ['GRADIO_H2OGPT_H2OGPT_KEY']) or 'EMPTY'
    os.environ['H2OGPT_OPENAI_API_KEY'] = server_api_key

    os.environ['GRADIO_AUTH'] = str(auth)
    os.environ['GRADIO_AUTH_ACCESS'] = auth_access
    os.environ['GRADIO_GUEST_NAME'] = guest_name

    port = int(os.getenv('H2OGPT_OPENAI_PORT', port))
    ssl_certfile = os.getenv('H2OGPT_OPENAI_CERT_PATH', ssl_certfile)
    ssl_keyfile = os.getenv('H2OGPT_OPENAI_KEY_PATH', ssl_keyfile)

    prefix = 'https' if ssl_keyfile and ssl_certfile else 'http'
    try:
        from openai_server.log import logger
    except ModuleNotFoundError:
        from log import logger
    logger.info(f'{name} API URL: {prefix}://{host}:{port}')
    logger.info(f'{name} API key: {server_api_key}')

    logging.getLogger("uvicorn.error").propagate = False

    # to pass args through so app can run gen setup
    os.environ['H2OGPT_MAIN_KWARGS'] = main_kwargs

    if not isinstance(app, str):
        workers = None

    if multiple_workers_gunicorn:
        os.environ['multiple_workers_gunicorn'] = 'True'

        assert isinstance(app, str), "app must be string for gunicorn multi-worker mode."
        print(f"Multi-worker {name} Proxy gunicorn: {workers}")
        # Build gunicorn command
        command = [
            'gunicorn',
            '-w', str(workers),
            '-k', 'uvicorn.workers.UvicornWorker',
            '-b', f"{host}:{port}",
        ]
        if ssl_certfile:
            command.extend(['--certfile', ssl_certfile])
        if ssl_keyfile:
            command.extend(['--keyfile', ssl_keyfile])
        command.append('openai_server.' + app)  # This should be a string like 'server:app'

        file_prefix = "gunicorn" + '_' + name + '_' + str(uuid.uuid4()) + '_'
        file_stdout = file_prefix + 'stdout.log'
        file_stderr = file_prefix + 'stderr.log'
        f_stdout = open(file_stdout, 'wt')
        f_stderr = open(file_stderr, 'wt')
        process = subprocess.Popen(command, stdout=f_stdout, stderr=f_stderr)
        wait = False
        if wait:
            process.communicate()
    else:
        uvicorn.run(app, host=host, port=port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile,
                    workers=workers,
                    )


def run(wait=True, **kwargs):
    assert 'is_openai_server' in kwargs
    name = 'OpenAI' if kwargs['is_openai_server'] else 'Function'
    print(kwargs)

    if kwargs['workers'] > 1 or kwargs['workers'] == 0:
        if not kwargs['multiple_workers_gunicorn']:
            # popen now, so launch uvicorn with string app
            print(f"Multi-worker {name} Proxy uvicorn: {kwargs['workers']}")
            # avoid CUDA forking
            command = ['python', 'openai_server/server_start.py']
            # Convert the kwargs to command line arguments
            for key, value in kwargs.items():
                command.append(f'--{key}')  # Assume keys are formatted as expected for the script
                command.append(str(value))  # Convert all values to strings to be safe

            file_prefix = "popen" + '_' + name + '_' + str(uuid.uuid4()) + '_'
            file_stdout = file_prefix + 'stdout.log'
            file_stderr = file_prefix + 'stderr.log'
            f_stdout = open(file_stdout, 'wt')
            f_stderr = open(file_stderr, 'wt')
            process = subprocess.Popen(command, stdout=f_stdout, stderr=f_stderr)
            if wait:
                process.communicate()
        else:
            # will launch gunicorn in popen inside run_server
            run_server(**kwargs)
    elif wait:
        kwargs['multiple_workers_gunicorn'] = False  # force uvicorn since not using multiple workers
        # launch uvicorn in this thread/process
        print(f"Single-worker {name} Proxy uvicorn in this thread: {kwargs['workers']}")
        run_server(**kwargs)
    else:
        kwargs['multiple_workers_gunicorn'] = False  # force uvicorn since not using multiple workers
        # launch uvicorn in this process in new thread
        print(f"Single-worker {name} Proxy uvicorn in new thread: {kwargs['workers']}")
        Thread(target=run_server, kwargs=kwargs, daemon=True).start()


def argv_to_kwargs(argv=None):
    parser = argparse.ArgumentParser(description='Convert command line arguments to kwargs.')

    # Inspect the run_server function to get its arguments and defaults
    sig = inspect.signature(run_server)
    for name, param in sig.parameters.items():
        # Determine if the parameter has a default value
        if param.default == inspect.Parameter.empty:
            # Parameter without a default value (treat it as required positional argument)
            parser.add_argument(f'--{name}')
        else:
            # Parameter with a default value (treat it as optional argument)
            if type(param.default) is int:  # Check if the default value is an integer
                parser.add_argument(f'--{name}', type=int, default=param.default)
            elif type(param.default) is bool:  # Add support for boolean values
                parser.add_argument(f'--{name}', type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                                    default=param.default)
            else:  # Treat as string by default
                parser.add_argument(f'--{name}', type=str, default=param.default if param.default is not None else '')

    # Parse the command line arguments
    args = parser.parse_args(argv[1:] if argv else None)

    # Convert parsed arguments to a dictionary
    kwargs = vars(args)
    return kwargs


if __name__ == '__main__':
    kwargs = argv_to_kwargs(sys.argv)
    run_server(**kwargs)
