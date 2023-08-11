import ast
import os

import pytest

from src.enums import LangChainAction
from tests.test_inference_servers import run_h2ogpt_docker
from tests.utils import wrap_test_forked, get_inf_server, get_inf_port
from src.utils import download_simple


@pytest.mark.parametrize("backend", [
    # 'transformers',
    'tgi',
])
@pytest.mark.parametrize("base_model", ['h2oai/h2ogpt-4096-llama2-7b-chat'])
@pytest.mark.need_tokens
@wrap_test_forked
def test_perf_benchmarks(backend, base_model):
    # launch server
    if backend == 'transformers':
        from src.gen import main
        main(base_model=base_model, chat=True, gradio=True, num_beams=1, block_gradio_exit=False, verbose=True,
             use_auth_token=True,
             )
    elif backend == 'tgi':
        from tests.test_inference_servers import run_docker
        # HF inference server
        gradio_port = get_inf_port()
        inf_port = gradio_port + 1
        inference_server = 'http://127.0.0.1:%s' % inf_port
        docker_hash1 = run_docker(inf_port, base_model, low_mem_mode=True)
        import time
        time.sleep(30)
        os.system('docker logs %s | tail -10' % docker_hash1)

        # h2oGPT server
        docker_hash2 = run_h2ogpt_docker(gradio_port, base_model, inference_server=inference_server)
        time.sleep(30)  # assumes image already downloaded, else need more time
        os.system('docker logs %s | tail -10' % docker_hash2)
    else:
        raise NotImplementedError("backend %s not implemented" % backend)

    # get file for client to upload
    url = 'https://cdn.openai.com/papers/whisper.pdf'
    test_file1 = os.path.join('/tmp/', 'my_test_pdf.pdf')
    download_simple(url, dest=test_file1)

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server())

    # upload file(s).  Can be list or single file
    test_file_local, test_file_server = client.predict(test_file1, api_name='/upload_api')
    assert os.path.normpath(test_file_local) != os.path.normpath(test_file_server)

    chunk = True
    chunk_size = 512
    langchain_mode = 'MyData'
    res = client.predict(test_file_server, chunk, chunk_size, langchain_mode, api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert os.path.basename(test_file_server) in res[2]
    assert res[3] == ''

    # ask for summary, need to use same client if using MyData
    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    kwargs = dict(langchain_mode=langchain_mode,
                  langchain_action="Summarize",  # uses full document, not vectorDB chunks
                  top_k_docs=-1,  # entire pdf
                  document_subset='Relevant',
                  document_choice='All',
                  max_new_tokens=1024,
                  max_time=300,
                  do_sample=False,
                  prompt_summary='',
                  )

    import time
    t0 = time.time()
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    t1 = time.time()
    res = ast.literal_eval(res)
    response = res['response']
    sources = res['sources']
    size_summary = os.path.getsize(test_file1)
    print(response)
    print("Time to summarize %s bytes into %s bytes: %.4f" % (size_summary, len(response), t1-t0))
    assert 'my_test_pdf.pdf' in sources

    kwargs = dict(prompt_summary="Write a poem about water.")
    import time
    t0 = time.time()
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    t1 = time.time()
    res = ast.literal_eval(res)
    response = res['response']
    print(response)
    print("Time to generate %s bytes: %.4f" % (len(response), t1-t0))


    # test this version for now, until docker updated
    # version = 0

    try:
        pass
        # client test to server that only consumes inference server
        # prompt = 'Who are you?'
        # print("Starting client tests with prompt: %s using %s" % (prompt, get_inf_server()))
        # from src.client_test import run_client_chat
        # res_dict, client = run_client_chat(prompt=prompt,
        #                                    stream_output=True,
        #                                    max_new_tokens=256,
        #                                    langchain_mode='Disabled',
        #                                    langchain_action=LangChainAction.QUERY.value,
        #                                    langchain_agents=[],
        #                                    version=version)
        # assert res_dict['prompt'] == prompt
        # assert res_dict['iinput'] == ''
    finally:
        pass
        # os.system("docker stop %s" % docker_hash1)
        # os.system("docker stop %s" % docker_hash2)
