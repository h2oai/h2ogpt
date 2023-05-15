from h2ogpt_client import Client


def test_answer():
    client = Client("http://0.0.0.0:7860")
    r = client.text_completion.create("Hello world")
    assert r
    print(r)
