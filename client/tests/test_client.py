import os

import pytest

from h2ogpt_client import Client


def create_client(server_url: str = "") -> Client:
    server_url = server_url or os.getenv("H2OGPT_SERVER", "http://0.0.0.0:7860")
    return Client(server_url)


def test_answer():
    client = create_client()
    r = client.text_completion.create("Hello world")
    assert r
    print(r)


@pytest.mark.asyncio
async def test_answer_async():
    client = create_client()
    r = await client.text_completion.create_async("Hello world")
    assert r
    print(r)
