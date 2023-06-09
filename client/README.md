# h2oGPT Client
A Python thin-client for h2oGPT.

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

## Development
### Prerequisites
- Python 3.8
- [Poetry](https://python-poetry.org/docs/#installation) - A dependency management and packaging tool for Python

### Setup environment
```shell
make setup
```

### Run lint
```shell
make lint
```

### Build
```shell
make build
```
