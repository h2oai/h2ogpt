def test_client1():
    from generate import main
    main(base_model='h2oai/h2ogpt-oig-oasst1-512-6.9b', prompt_type='human_bot', chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False)

    from client_test import test_client_basic
    res_dict = test_client_basic()
    assert res_dict['instruction_nochat'] == 'Who are you?'
    assert res_dict['iinput_nochat'] == ''
    assert 'I am h2oGPT' in res_dict['response']

