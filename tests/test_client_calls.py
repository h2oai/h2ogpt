import pytest


def test_client1():
    from tests.utils import call_subprocess_onetask
    call_subprocess_onetask(run_client1)


def run_client1():
    from generate import main
    main(base_model='h2oai/h2ogpt-oig-oasst1-512-6.9b', prompt_type='human_bot', chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False)

    from client_test import test_client_basic
    res_dict = test_client_basic()
    assert res_dict['instruction_nochat'] == 'Who are you?'
    assert res_dict['iinput_nochat'] == ''
    assert 'I am h2oGPT' in res_dict['response']


@pytest.mark.skip(reason="Local file required")
def test_client_long():
    from tests.utils import call_subprocess_onetask
    call_subprocess_onetask(run_client_long)


def run_client_long():
    from generate import main
    main(base_model='mosaicml/mpt-7b-storywriter', prompt_type='plain', chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False)

    with open("/home/jon/Downloads/Gatsby_PDF_FullText.txt") as f:
        instruction_nochat = f.readlines()

    from client_test import run_client_basic
    res_dict = run_client_basic(instruction_nochat=instruction_nochat, prompt_type='plain')
    print(res_dict['response'])
