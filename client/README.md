# h2oGPT Client
A Python thin-client for h2oGPT.

## Installation
### Prerequisites
- Python 3.8+
- [Poetry](https://python-poetry.org/docs/#installation) - A dependency management and packaging tool for Python

### Setup environment
```shell
cd client
pip install poetry==1.5.1
make setup
make lint
make build
# install (choose version if multiple builds in dist directory)
pip install dist/h2ogpt_client-*-py3-none-any.whl
# test
cd ..
pytest -s -v --forked client
```

## Usage
```python
from h2ogpt_client import Client

client = Client("http://0.0.0.0:7860")

# text completion
response = client.text_completion.create("Hello world")
response = await client.text_completion.create_async("Hello world")

# chat completion
chat_context = client.chat_completion.create()
chat = chat_context.chat("Hey!")
print(chat["user"])  # prints user prompt, i.e. "Hey!"
print(chat["gpt"])   # prints reply of the h2oGPT
chat_history = chat_context.chat_history()
```
:warning: **Note**: Client APIs are still evolving. Hence, APIs can be changed without prior warnings.

