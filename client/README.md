# h2oGPT Client
A Python thin-client for h2oGPT.

## Usage
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

### Prerequisites
- Python 3.8+

### Setup
:information_source: [Poetry](https://python-poetry.org) is used as the build tool.

:warning: If you already have an activated virtual Python environment then Poetry will use that rather than creating a new one.
So executing the following command where the h2oGPT Python environment is activated will result in installation of dev, test, and runtime dependencies of the Client into the h2oGPT Python environment.
We recommend maintaining a separate Python environment for the Client.
```shell
make -C client setup
```

### Build
```shell
make -C client build
```
- Distribution wheel file can be found in the `client/dist` directory.

### Test
### Test with h2oGPT
1. Install test dependencies of the Client into the h2oGPT Python environment.
```shell
source venv/bin/activate
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
