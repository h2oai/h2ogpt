import inspect
import os
import subprocess
import sys
import argparse
import logging
import typing
from threading import Thread
from typing import Union

import uvicorn
from fastapi import FastAPI

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
               ):
    if workers == 0:
        workers = min(16, os.cpu_count() * 2 + 1)
    assert app is not None

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
    from openai_server.log import logger
    logger.info(f'OpenAI API URL: {prefix}://{host}:{port}')
    logger.info(f'OpenAI API key: {server_api_key}')

    logging.getLogger("uvicorn.error").propagate = False

    if not isinstance(app, str):
        workers = None
    uvicorn.run(app, host=host, port=port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile,
                workers=workers,
                )


def run(wait=True, **kwargs):
    print(kwargs)
    if kwargs['workers'] > 1 or kwargs['workers'] == 0:
        print("Multi-worker OpenAI Proxy uvicorn: %s" % kwargs['workers'])
        # avoid CUDA forking
        command = ['python', 'openai_server/server_start.py']
        # Convert the kwargs to command line arguments
        for key, value in kwargs.items():
            command.append(f'--{key}')  # Assume keys are formatted as expected for the script
            command.append(str(value))  # Convert all values to strings to be safe

        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        for c in iter(lambda: process.stdout.read(1), b''):
            sys.stdout.write(c.decode('utf-8', errors='replace'))  # Ensure decoding from bytes to str
    elif wait:
        print("Single-worker OpenAI Proxy uvicorn in this thread: %s" % kwargs['workers'])
        run_server(**kwargs)
    else:
        print("Single-worker OpenAI Proxy uvicorn in new thread: %s" % kwargs['workers'])
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
                parser.add_argument(f'--{name}', action='store_true' if param.default is False else 'store_false')
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
