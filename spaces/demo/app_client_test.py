"""
Client test.

Run server:

python app.py

Then run this client:

python app_client_test.py

NOTE: To access a private app on gradio, do:

HUGGINGFACE_TOKEN=<HUGGINGFACE_TOKEN> GRADIO_HOST="https://huggingface.co/spaces/h2oai/h2ogpt-oasst1-512-6_9b-hosted" python app_client_test.py
"""

import os
from gradio_client import Client
import markdown  # pip install markdown
from bs4 import BeautifulSoup  # pip install beautifulsoup4


hf_token = os.environ.get('HUGGINGFACE_TOKEN')
host = os.environ.get("GRADIO_HOST", "http://localhost:7860")
client = Client(host, hf_token=hf_token)


def test_app_client_basic():
    instruction = "Who are you?"
    args = [instruction]

    api_name = '/submit'
    res = client.predict(
        *tuple(args),
        api_name=api_name,
    )
    print(md_to_text(res))


def md_to_text(md):
    html = markdown.markdown(md)
    soup = BeautifulSoup(html, features='html.parser')
    return soup.get_text()


if __name__ == '__main__':
    test_app_client_basic()
