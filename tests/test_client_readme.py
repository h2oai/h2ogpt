import pytest

from tests.utils import wrap_test_forked


@pytest.mark.parametrize("local_server", [False, True])
@pytest.mark.parametrize("persist", [True, False])
@wrap_test_forked
def test_readme_example(local_server, persist):
    if local_server:
        from src.gen import main
        main(base_model='llama', chat=True, gradio=True, num_beams=1, block_gradio_exit=False, verbose=True)

    # self-contained example used for readme, to be copied to README_CLIENT.md if changed, setting local_server = True at first
    import os
    # The grclient.py file can be copied from h2ogpt repo and used with local gradio_client for example use
    from gradio_utils.grclient import GradioClient

    h2ogpt_key = os.getenv('H2OGPT_KEY') or os.getenv('H2OGPT_H2OGPT_KEY')

    if local_server:
        host = "http://0.0.0.0:7860"
    else:
        host = "https://gpt.h2o.ai"

    client = GradioClient(host, h2ogpt_key=h2ogpt_key, persist=persist)

    models = client.list_models()
    print(models)

    print(client.question("Who are you?", model=models[0]))
    print(client.question("What did I just ask?", model=models[0]))
    if persist:
        assert len(client.chat_conversation) == 2
        assert client.chat_conversation[-1][1] == "You just asked: Who are you?" or \
               client.chat_conversation[-1][1] == "You just asked: \"Who are you?\""

    # LLM
    print(client.question("Who are you?", model=models[0]))

    url = "https://cdn.openai.com/papers/whisper.pdf"

    # Q/A
    print(client.query("What is whisper?", url=url, model=models[0]))
    # summarization (map_reduce over all pages if top_k_docs=-1)
    print(client.summarize(url=url, top_k_docs=3, model=models[0]))
    # extraction (map per page)
    print(client.extract(url=url, top_k_docs=3, model=models[0]))

    # summarization (map_reduce over all pages if top_k_docs=-1)
    print(client.summarize(query="List all names", url=url, top_k_docs=3, model=models[0]))
    # extraction (map per page)
    print(client.extract(query="Give valid JSON for any names.", url=url, top_k_docs=3, model=models[0]))

    if persist:
        assert len(client.chat_conversation) == 8
