# h2oGPT Client
A Python thin-client for h2oGPT.

## Prerequisites
- Python 3.8+

If you don't have Python 3.8 in your system, you can use [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
```bash
conda create -n h2ogpt_client_build -y
conda activate h2ogpt_client_build
conda install python=3.8 -y
```

## Download Client Wheel

Install the latest nightly wheel from S3.

```bash
pip install https://s3.amazonaws.com/artifacts.h2o.ai/snapshots/ai/h2o/h2ogpt_client/latest-nightly/h2ogpt_client-0.1.0-py3-none-any.whl
```

Nightly releases can also be found [here](https://github.com/h2oai/h2ogpt/releases)

## Build Client Wheel

If want to build fresh wheel from main branch instead of getting nightly, follow these instructions.

### Setup
:information_source: [Poetry](https://python-poetry.org) is used as the build tool.
```shell
rm -rf client/.poetry/
make -C client setup
```

### Build
```shell
make -C client build
```
Distribution wheel file can be found in the `client/dist` directory.  This wheel can be installed in the primary h2oGPT environment or any other environment, e.g.
```bash
pip uninstall -y h2ogpt_client
pip install client/dist/h2ogpt_client-*-py3-none-any.whl
```

## Usage

Based upon [test code](tests/test_client.py) and test code `test_readme_example`:
```python


def test_readme_example(local_server):
    import os
    import asyncio
    from h2ogpt_client import Client

    if local_server:
        client = Client("http://0.0.0.0:7860")
    else:
        h2ogpt_key = os.getenv("H2OGPT_KEY") or os.getenv("H2OGPT_H2OGPT_KEY")
        if h2ogpt_key is None:
            return
        # if you have API key for public instance:
        client = Client("https://gpt.h2o.ai", h2ogpt_key=h2ogpt_key)

    # Text completion
    text_completion = client.text_completion.create()
    response = asyncio.run(text_completion.complete("Hello world"))
    print("asyncio text completion response: %s" % response)
    # Text completion: synchronous
    response = text_completion.complete_sync("Hello world")
    print("sync text completion response: %s" % response)

    # Chat completion
    chat_completion = client.chat_completion.create()
    reply = asyncio.run(chat_completion.chat("Hey!"))
    print("asyncio text completion user: %s gpt: %s" % (reply["user"], reply["gpt"]))
    chat_history = chat_completion.chat_history()
    print("chat_history: %s" % chat_history)
    # Chat completion: synchronous
    reply = chat_completion.chat_sync("Hey!")
    print("sync chat completion gpt: %s" % reply["gpt"])

test_readme_example(local_server=True)
```
:warning: **Note**: Client APIs are still evolving. Hence, APIs can be changed without prior warnings.

## Development Guide

### Test

In an h2oGPT environment with the client installed, can run tests that test client and server.

### Test with h2oGPT env
1. Install test dependencies of the Client into the h2oGPT Python environment.
```shell
make -C client setup_test
```
2. Run the tests with h2oGPT.
```shell
pytest client/tests/
```

#### Test with an existing h2oGPT server
If you already have a running h2oGPT server, then set the `H2OGPT_SERVER` environment variable to use it for testing.
```shell
make H2OGPT_SERVER="http://0.0.0.0:7860" -C client test
```
