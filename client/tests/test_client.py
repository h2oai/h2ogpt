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

    chat2 = await chat_completion.chat(prompt="What is the capital of USA?")
    assert chat2["user"] == "What is the capital of USA?"
    assert chat2["gpt"]

    chat3 = await chat_completion.chat(prompt="What is the population in there?")
    assert chat3["user"] == "What is the population in there?"
    assert chat3["gpt"]

    chat_history = chat_completion.chat_history()
    assert chat_history == [chat1, chat2, chat3]
    print(chat_history)


def test_chat_completion_sync(client):
    chat_completion = client.chat_completion.create()

    chat1 = chat_completion.chat_sync(prompt="What is UNESCO?")
    assert chat1["user"] == "What is UNESCO?"
    assert chat1["gpt"]

    chat2 = chat_completion.chat_sync(prompt="Is it a part of the UN?")
    assert chat2["user"] == "Is it a part of the UN?"
    assert chat2["gpt"]

    chat3 = chat_completion.chat_sync(prompt="Where is the headquarters?")
    assert chat3["user"] == "Where is the headquarters?"
    assert chat3["gpt"]

    chat_history = chat_completion.chat_history()
    assert chat_history == [chat1, chat2, chat3]
    print(chat_history)


def test_available_models(client):
    models = client.list_models()
    assert len(models)
    print(models)


def test_parameters_order(client, eval_func_param_names):
    text_completion = client.text_completion.create()
    assert eval_func_param_names == list(text_completion._parameters.keys())
