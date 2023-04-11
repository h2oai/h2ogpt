from gradio_client import Client

client = Client("http://localhost:7860")
print(client.view_api(all_endpoints=True))

instruction = "Who are you?"
iinput = ''
context = ''
stream_output = False
prompt_type = 'human_bot'
temperature = 0.1
top_p = 0.75
top_k = 40
num_beams = 1
max_new_tokens = 500
min_new_tokens = 0
early_stopping = False
max_time = 180
repetition_penalty = 1.0
num_return_sequences = 1
do_sample = True


def test_client_basic():
    res = client.predict(
        instruction,
        iinput,
        context,
        stream_output,
        prompt_type,
        temperature,
        top_p,
        top_k,
        num_beams,
        max_new_tokens,
        min_new_tokens,
        early_stopping,
        max_time,
        repetition_penalty,
        num_return_sequences,
        do_sample,
        api_name='/submit',
    )
    print(res)
    assert "I am a chatbot." in res
