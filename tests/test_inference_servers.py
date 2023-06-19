import os

from client_test import run_client_many
from tests.utils import wrap_test_forked


@wrap_test_forked
def test_gradio_inference_server(prompt='Who are you?', stream_output=False, max_new_tokens=256,
                                 base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b', prompt_type='human_bot',
                                 langchain_mode='Disabled', user_path=None,
                                 visible_langchain_modes=['UserData', 'MyData'],
                                 reverse_docs=True):
    main_kwargs = dict(base_model=base_model, prompt_type=prompt_type, chat=True,
                       stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
                       max_new_tokens=max_new_tokens,
                       langchain_mode=langchain_mode, user_path=user_path,
                       visible_langchain_modes=visible_langchain_modes,
                       reverse_docs=reverse_docs)

    # inference server
    inf_port = os.environ['GRADIO_SERVER_PORT'] = "7860"
    from generate import main
    main(**main_kwargs)

    # server that consumes inference server
    client_port = os.environ['GRADIO_SERVER_PORT'] = "7861"
    from generate import main
    main(**main_kwargs, inference_server='http://127.0.0.1:%s' % inf_port)

    # client test to server that only consumes inference server
    from client_test import run_client_chat
    os.environ['HOST'] = "http://127.0.0.1:%s" % client_port
    res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''

    # will use HOST from above
    run_client_many()
