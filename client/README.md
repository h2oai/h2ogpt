# h2oGPT Client
A Python thin-client for h2oGPT.

### Prerequisites
- Python 3.8+

### Setup
:information_source: [Poetry](https://python-poetry.org) is used as the build tool.

```bash
conda create -n h2ogpt_client_build -y
conda activate h2ogpt_client_build
conda install python=3.10 -y
```

```shell
make -C client setup
```

### Build
```shell
make -C client build
```
Distribution wheel file can be found in the `client/dist` directory.  This wheel can be installed in the primary h2oGPT environment or any other environment, e.g.
```bash
pip install client/dist/h2ogpt_client-*-py3-none-any.whl
```

## Usage

Make environment or use some existing environment:
```bash
conda create -n h2ogpt_client -y
conda activate h2ogpt_client
conda install python=3.10 -y
```

```python
from h2ogpt_client import Client

client = Client("http://0.0.0.0:7860")

# text completion
text_completion = client.text_completion.create()
response = await text_completion.complete("Hello world")

# chat completion
chat_completion = client.chat_completion.create()
reply = await chat_completion.chat("Hey!")
print(reply["user"])  # prints user prompt, i.e. "Hey!"
print(reply["gpt"])   # prints reply from h2oGPT
chat_history = chat_completion.chat_history()
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
