import pytest


def launch_openai_server():
    from openai_server.server import run
    run()


def test_openai_server():
    launch_openai_server()


#repeat0 = 100  # e.g. to test concurrency
repeat0 = 1


@pytest.mark.parametrize("stream_output", [False, True])
@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("local_server", [False])  # True
@pytest.mark.parametrize("prompt", ["Who are you?", "Tell a very long kid's story about birds."])
@pytest.mark.parametrize("repeat", list(range(0, repeat0)))
def test_openai_client(stream_output, chat, local_server, prompt, repeat):
    base_model = 'openchat/openchat-3.5-1210'

    if local_server:
        from src.gen import main
        main(base_model=base_model, chat=False,
             stream_output=stream_output, gradio=True,
             num_beams=1, block_gradio_exit=False)
    else:
        # RUN something
        # e.g. CUDA_VISIBLE_DEVICES=0 python generate.py --verbose=True --score_model=None --gradio_offline_level=2 --base_model=openchat/openchat-3.5-1210 --inference_server=vllm:IP:port --max_seq_len=4096 --save_dir=duder1 --verbose --openai_server=True --concurency_count=64
        pass

    api_key = "EMPTY"
    base_url = 'http://localhost:5000/v1'
    verbose = True
    system_prompt = "You are a helpful assistant."
    chat_conversation = []
    add_chat_history_to_context = True

    client_kwargs = dict(model=base_model,
                         max_tokens=200,
                         stream=stream_output)

    from openai import OpenAI, AsyncOpenAI
    client_args = dict(base_url=base_url, api_key=api_key)
    openai_client = OpenAI(**client_args)
    async_client = AsyncOpenAI(**client_args)

    # COMPLETION

    if chat:
        client = openai_client.chat.completions
        async_client = async_client.chat.completions

        messages0 = []
        if system_prompt:
            messages0.append({"role": "system", "content": system_prompt})
        if chat_conversation and add_chat_history_to_context:
            for message1 in chat_conversation:
                if len(message1) == 2:
                    messages0.append(
                        {'role': 'user', 'content': message1[0] if message1[0] is not None else ''})
                    messages0.append(
                        {'role': 'assistant', 'content': message1[1] if message1[1] is not None else ''})
        messages0.append({'role': 'user', 'content': prompt if prompt is not None else ''})

        client_kwargs.update(dict(messages=messages0))
    else:
        client = openai_client.completions
        async_client = async_client.completions

        client_kwargs.update(dict(prompt=prompt))

    responses = client.create(**client_kwargs)

    if not stream_output:
        if chat:
            text = responses.choices[0].message.content
        else:
            text = responses.choices[0].text
        print(text)
    else:
        collected_events = []
        text = ''
        for event in responses:
            collected_events.append(event)  # save the event response
            if chat:
                delta = event.choices[0].delta.content
            else:
                delta = event.choices[0].text  # extract the text
            text += delta  # append the text
            if verbose:
                print('delta: %s' % delta)
        print(text)

    if "Who" in prompt:
        assert 'OpenAI' in text or 'chatbot' in text
    else:
        assert 'birds' in text

    # MODELS
    model_info = openai_client.models.retrieve(base_model)
    assert model_info.base_model == base_model
    model_list = openai_client.models.list()
    assert model_list.data[0] == base_model


if __name__ == '__main__':
    launch_openai_server()
