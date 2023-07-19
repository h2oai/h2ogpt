### Client APIs

A Gradio API and an OpenAI-compliant API are supported.

##### Gradio Client API

`generate.py` by default runs a gradio server, which also gives access to client API using gradio client.  One can use it with h2oGPT, or independently of h2oGPT repository by installing an env:
```bash
conda create -n gradioclient -y
conda activate gradioclient
conda install python=3.10 -y
pip install gradio_client
```
then running client code:
```python
from gradio_client import Client
import ast

HOST_URL = "http://localhost:7860"
client = Client(HOST_URL)

# string of dict for input
kwargs = dict(instruction_nochat='Who are you?')
res = client.predict(str(dict(kwargs)), api_name='/submit_nochat_api')

# string of dict for output
response = ast.literal_eval(res)['response']
print(response)
```
For other ways to use gradio client, see example [test code](../client_test.py) or other tests in our [tests](https://github.com/h2oai/h2ogpt/blob/main/tests/test_client_calls.py).

Any element in [gradio_runner.py](../gradio_runner.py) with `api_name` defined can be accessed via the gradio client.

##### OpenAI Python Client Library

An OpenAI compliant client is available. Refer the [README](../client/README.md)  for more details.

