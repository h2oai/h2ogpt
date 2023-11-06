import platform

import pytest

from h2ogpt_client import Client

platform.python_version()


@pytest.fixture
def client(server_url, h2ogpt_key) -> Client:
    return Client(server_url, h2ogpt_key=h2ogpt_key)


def _create_text_completion(client):
    model = client.models.list()[-1]
    return client.text_completion.create(model=model)


@pytest.mark.asyncio
async def test_text_completion(client):
    text_completion = _create_text_completion(client)
    response = await text_completion.complete(prompt="Hello world")
    assert response
    print(response)


@pytest.mark.asyncio
async def test_text_completion_stream(client):
    text_completion = _create_text_completion(client)
    response = await text_completion.complete(
        prompt="Write a poem about the Amazon rainforest. End it with an emoji.",
        enable_streaming=True,
    )
    async for token in response:
        assert token
        print(token, end="")


def test_text_completion_sync(client):
    text_completion = _create_text_completion(client)
    response = text_completion.complete_sync(prompt="Hello world")
    assert response
    print(response)


def test_text_completion_sync_stream(client):
    text_completion = _create_text_completion(client)
    response = text_completion.complete_sync(
        prompt="Write a poem about the Amazon rainforest. End it with an emoji.",
        enable_streaming=True,
    )
    for token in response:
        assert token
        print(token, end="")


def _create_chat_completion(client):
    model = client.models.list()[-1]
    return client.chat_completion.create(model=model)


@pytest.mark.asyncio
async def test_chat_completion(client):
    chat_completion = _create_chat_completion(client)

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
    chat_completion = _create_chat_completion(client)

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
    models = client.models.list()
    assert len(models)
    print(models)


def test_server_properties(client, server_url):
    assert client.server.address.startswith(server_url)
    assert client.server.hash


def test_parameters_order(client, eval_func_param_names):
    text_completion = client.text_completion.create()
    assert eval_func_param_names == list(text_completion._parameters.keys())
    chat_completion = client.chat_completion.create()
    assert eval_func_param_names == list(chat_completion._parameters.keys())


@pytest.mark.parametrize("local_server", [True, False])
def test_readme_example(local_server):
    # self-contained example used for readme,
    # to be copied to client/README.md if changed, setting local_server = True at first
    import asyncio
    import os

    from h2ogpt_client import Client

    if local_server:
        client = Client("http://0.0.0.0:7860")
    else:
        h2ogpt_key = os.getenv("H2OGPT_KEY") or os.getenv("H2OGPT_H2OGPT_KEY")
        if h2ogpt_key is None:
            return
        # if you have API key for public instance:
        client = Client("https://gpt.h2o.ai", h2ogpt_key=h2ogpt_key)

    # Text completion
    text_completion = client.text_completion.create()
    response = asyncio.run(text_completion.complete("Hello world"))
    print("asyncio text completion response: %s" % response)
    # Text completion: synchronous
    response = text_completion.complete_sync("Hello world")
    print("sync text completion response: %s" % response)

    # Chat completion
    chat_completion = client.chat_completion.create()
    reply = asyncio.run(chat_completion.chat("Hey!"))
    print("asyncio text completion user: %s gpt: %s" % (reply["user"], reply["gpt"]))
    chat_history = chat_completion.chat_history()
    print("chat_history: %s" % chat_history)
    # Chat completion: synchronous
    reply = chat_completion.chat_sync("Hey!")
    print("sync chat completion gpt: %s" % reply["gpt"])
