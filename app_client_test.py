"""
Client test.

Run server:

python app.py

Then run this client:

python client_test.py
"""

from gradio_client import Client
import markdown  # pip install markdown
from bs4 import BeautifulSoup  # pip install beautifulsoup4

client = Client("http://localhost:7860")


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
