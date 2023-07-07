import platform

import pytest

from h2ogpt_client import Client

platform.python_version()


@pytest.fixture
def client(server_url) -> Client:
    return Client(server_url)


@pytest.mark.asyncio
async def test_text_completion(client):
    text_completion = client.text_completion.create()
    response = await text_completion.complete(prompt="Hello world")
    assert response
    print(response)


def test_text_completion_sync(client):
    text_completion = client.text_completion.create()
    response = text_completion.complete_sync(prompt="Hello world")
    assert response
    print(response)


@pytest.mark.asyncio
async def test_chat_completion(client):
    chat_completion = client.chat_completion.create()

    chat1 = await chat_completion.chat(prompt="Hey!")
    assert chat1["user"] == "Hey!"
    assert chat1["gpt"]

    chat2 = await chat_completion.chat(prompt="How are you?")
    assert chat2["user"] == "How are you?"
    assert chat2["gpt"]

    chat3 = await chat_completion.chat(prompt="Have a good day")
    assert chat3["user"] == "Have a good day"
    assert chat3["gpt"]

    chat_history = chat_completion.chat_history()
    assert chat_history == [chat1, chat2, chat3]
    print(chat_history)


def test_chat_completion_sync(client):
    chat_completion = client.chat_completion.create()

    chat1 = chat_completion.chat_sync(prompt="Hey!")
    assert chat1["user"] == "Hey!"
    assert chat1["gpt"]

    chat2 = chat_completion.chat_sync(prompt="How are you?")
    assert chat2["user"] == "How are you?"
    assert chat2["gpt"]

    chat3 = chat_completion.chat_sync(prompt="Have a good day")
    assert chat3["user"] == "Have a good day"
    assert chat3["gpt"]

    chat_history = chat_completion.chat_history()
    assert chat_history == [chat1, chat2, chat3]
    print(chat_history)


def test_parameters_order(client, eval_func_param_names):
    text_completion = client.text_completion.create()
    assert eval_func_param_names == list(text_completion._parameters.keys())
