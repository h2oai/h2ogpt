import os

import pytest

from h2ogpt_client import Client


def create_client(server_url: str = "") -> Client:
    server_url = server_url or os.getenv("H2OGPT_SERVER", "http://0.0.0.0:7860")
    return Client(server_url)


def test_text_completion():
    client = create_client()
    r = client.text_completion.create("Hello world")
    assert r
    print(r)


@pytest.mark.asyncio
async def test_text_completion_async():
    client = create_client()
    r = await client.text_completion.create_async("Hello world")
    assert r
    print(r)


def test_chat_completion():
    client = create_client()
    chat_context = client.chat_completion.create()

    chat1 = chat_context.chat("Hey!")
    assert chat1["user"] == "Hey!"
    assert chat1["gpt"]

    chat2 = chat_context.chat("How are you?")
    assert chat2["user"] == "How are you?"
    assert chat2["gpt"]

    chat3 = chat_context.chat("Have a good day")
    assert chat3["user"] == "Have a good day"
    assert chat3["gpt"]

    chat_history = chat_context.chat_history()
    assert chat_history == [chat1, chat2, chat3]
    print(chat_history)
