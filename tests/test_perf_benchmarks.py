import ast
import os
import subprocess
import time

import pytest

from tests.test_inference_servers import run_h2ogpt_docker
from tests.utils import wrap_test_forked, get_inf_server, get_inf_port
from src.utils import download_simple

results_file = "./benchmarks/perf.json"

@pytest.mark.skipif(not os.getenv('BENCHMARK'),
                    reason="Only for benchmarking")
@pytest.mark.parametrize("backend", [
    # 'transformers',
    # 'text-generation-inference',
    'text-generation-inference-',
])
@pytest.mark.parametrize("base_model", [
    'h2oai/h2ogpt-4096-llama2-7b-chat',
    'h2oai/h2ogpt-4096-llama2-13b-chat',
    'h2oai/h2ogpt-4096-llama2-70b-chat',
])
@pytest.mark.parametrize("task", [
    # 'summary',
    # 'generate',
    'summary_and_generate'
])
@pytest.mark.parametrize("bits", [
    16,
    8,
    4,
], ids=[
    "16-bit",
    "8-bit",
    "4-bit",
])
@pytest.mark.parametrize("ngpus", [
    0,
    1,
    2,
    4,
    8,
], ids=[
    "CPU",
    "1 GPU",
    "2 GPUs",
    "4 GPUs",
    "8 GPUs",
])
@pytest.mark.need_tokens
@wrap_test_forked
def test_perf_benchmarks(backend, base_model, task, bits, ngpus):
    reps = 3
    bench_dict = locals()
    from datetime import datetime
    import json
    import socket
    os.environ['CUDA_VISIBLE_DEVICES'] = "" if ngpus == 0 else "0" if ngpus == 1 else ",".join([str(x) for x in range(ngpus)])
    import torch
    n_gpus = torch.cuda.device_count()
    if n_gpus != ngpus:
        return
    git_sha = (
        subprocess.check_output("git rev-parse HEAD", shell=True)
        .decode("utf-8")
        .strip()
    )
    bench_dict["date"] = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    bench_dict["git_sha"] = git_sha[:8]
    bench_dict["n_gpus"] = n_gpus
    from importlib.metadata import version
    bench_dict["transformers"] = str(version('transformers'))
    bench_dict["bitsandbytes"] = str(version('bitsandbytes'))
    bench_dict["cuda"] = str(torch.version.cuda)
    bench_dict["hostname"] = str(socket.gethostname())
    gpu_list = [torch.cuda.get_device_name(i) for i in range(n_gpus)]

    # get GPU memory, assumes homogeneous system
    cmd = 'nvidia-smi -i 0 -q | grep -A 1 "FB Memory Usage" | cut -d: -f2 | tail -n 1'
    o = subprocess.check_output(cmd, shell=True, timeout=15)
    mem_gpu = o.decode("utf-8").splitlines()[0].strip() if n_gpus else 0

    bench_dict["gpus"] = "%d x %s (%s)" % (n_gpus, gpu_list[0], mem_gpu) if n_gpus else "CPU"
    assert all([x == gpu_list[0] for x in gpu_list])
    print(bench_dict)

    # launch server(s)
    docker_hash1 = None
    docker_hash2 = None
    max_new_tokens = 4096
    try:
        h2ogpt_args = dict(base_model=base_model,
             chat=True, gradio=True, num_beams=1, block_gradio_exit=False, verbose=True,
             load_half=bits == 16 and n_gpus,
             load_8bit=bits == 8,
             load_4bit=bits == 4,
             langchain_mode='MyData',
             use_auth_token=True,
             max_new_tokens=max_new_tokens,
             use_gpu_id=ngpus == 1,
             use_safetensors=True,
             score_model=None,
             )
        if backend == 'transformers':
            from src.gen import main
            main(**h2ogpt_args)
        elif backend == 'text-generation-inference':
            if bits != 16:
                return
            from tests.test_inference_servers import run_docker
            # HF inference server
            gradio_port = get_inf_port()
            inf_port = gradio_port + 1
            inference_server = 'http://127.0.0.1:%s' % inf_port
            docker_hash1 = run_docker(inf_port, base_model, low_mem_mode=False)  # don't do low-mem, since need tokens for summary
            os.system('docker logs %s | tail -10' % docker_hash1)

            # h2oGPT server
            docker_hash2 = run_h2ogpt_docker(gradio_port, base_model, inference_server=inference_server, max_new_tokens=max_new_tokens)
            time.sleep(30)  # assumes image already downloaded, else need more time
            os.system('docker logs %s | tail -10' % docker_hash2)
        elif backend == 'text-generation-inference-':
            if bits != 16:
                return
            from tests.test_inference_servers import run_docker
            # HF inference server
            gradio_port = get_inf_port()
            inf_port = gradio_port + 1
            inference_server = 'http://127.0.0.1:%s' % inf_port
            docker_hash1 = run_docker(inf_port, base_model, low_mem_mode=False)  # don't do low-mem, since need tokens for summary
            from src.gen import main
            main(**h2ogpt_args)
        else:
            raise NotImplementedError("backend %s not implemented" % backend)

        # get file for client to upload
        url = 'https://cdn.openai.com/papers/whisper.pdf'
        test_file1 = os.path.join('/tmp/', 'whisper1.pdf')
        download_simple(url, dest=test_file1)

        # PURE client code
        from gradio_client import Client
        client = Client(get_inf_server())

        if "summary" in task:
            # upload file(s).  Can be list or single file
            test_file_local, test_file_server = client.predict(test_file1, api_name='/upload_api')
            assert os.path.normpath(test_file_local) != os.path.normpath(test_file_server)

            chunk = True
            chunk_size = 512
            langchain_mode = 'MyData'
            embed = True
            loaders = tuple([None, None, None, None, None])
            extract_frames = 1
            llava_prompt = ''
            h2ogpt_key = ''
            res = client.predict(test_file_server,
                                 chunk, chunk_size, langchain_mode, embed,
                                 *loaders,
                                 extract_frames,
                                 llava_prompt,
                                 h2ogpt_key,
                                 api_name='/add_file_api')
            assert res[0] is None
            assert res[1] == langchain_mode
            # assert os.path.basename(test_file_server) in res[2]
            assert res[3] == ''

            # ask for summary, need to use same client if using MyData
            api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
            kwargs = dict(langchain_mode=langchain_mode,
                          langchain_action="Summarize",  # uses full document, not vectorDB chunks
                          top_k_docs=4,  # -1 == entire pdf
                          document_subset='Relevant',
                          document_choice='All',
                          max_new_tokens=max_new_tokens,
                          max_time=300,
                          do_sample=False,
                          prompt_summary='Summarize into single paragraph',
                          system_prompt='',
                          )

            t0 = time.time()
            for r in range(reps):
                res = client.predict(
                    str(dict(kwargs)),
                    api_name=api_name,
                )
            t1 = time.time()
            time_taken = (t1 - t0) / reps
            res = ast.literal_eval(res)
            response = res['response']
            sources = res['sources']
            size_summary = os.path.getsize(test_file1)
            # print(response)
            print("Time to summarize %s bytes into %s bytes: %.4f" % (size_summary, len(response), time_taken))
            bench_dict["summarize_input_len_bytes"] = size_summary
            bench_dict["summarize_output_len_bytes"] = len(response)
            bench_dict["summarize_time"] = time_taken
            # bench_dict["summarize_tokens_per_sec"] = res['tokens/s']
            assert 'my_test_pdf.pdf' in sources

        if "generate" in task:
            api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
            kwargs = dict(prompt_summary="Write a poem about water.")
            t0 = time.time()
            for r in range(reps):
                res = client.predict(
                    str(dict(kwargs)),
                    api_name=api_name,
                )
            t1 = time.time()
            time_taken = (t1 - t0) / reps
            res = ast.literal_eval(res)
            response = res['response']
            # print(response)
            print("Time to generate %s bytes: %.4f" % (len(response), time_taken))
            bench_dict["generate_output_len_bytes"] = len(response)
            bench_dict["generate_time"] = time_taken
            # bench_dict["generate_tokens_per_sec"] = res['tokens/s']
    except BaseException as e:
        if 'CUDA out of memory' in str(e):
            e = "OOM"
            bench_dict["exception"] = str(e)
        else:
            raise
    finally:
        if bench_dict["backend"] == "text-generation-inference-":
            # Fixup, so appears as same
            bench_dict["backend"] = "text-generation-inference"
        if 'summarize_time' in bench_dict or 'generate_time' in bench_dict or bench_dict.get('exception') == "OOM":
            with open(results_file, mode="a") as f:
                f.write(json.dumps(bench_dict) + "\n")
        if "text-generation-inference" in backend:
            if docker_hash1:
                os.system("docker stop %s" % docker_hash1)
            if docker_hash2:
                os.system("docker stop %s" % docker_hash2)


@pytest.mark.skip("run manually")
def test_plot_results():
    import pandas as pd
    import json
    res = []
    with open(results_file) as f:
        for line in f.readlines():
            entry = json.loads(line)
            res.append(entry)
    X = pd.DataFrame(res)
    X.to_csv(results_file + ".csv", index=False)

    result_cols = ['summarization time [sec]', 'generation speed [tokens/sec]']
    X[result_cols[0]] = X['summarize_time']
    X[result_cols[1]] = X['generate_output_len_bytes'] / 4 / X['generate_time']
    with open(results_file.replace(".json", ".md"), "w") as f:
        for backend in pd.unique(X['backend']):
            print("# Backend: %s" % backend, file=f)
            for base_model in pd.unique(X['base_model']):
                print("## Model: %s (%s)" % (base_model, backend), file=f)
                for n_gpus in sorted(pd.unique(X['n_gpus'])):
                    XX = X[(X['base_model'] == base_model) & (X['backend'] == backend) & (X['n_gpus'] == n_gpus)]
                    if XX.shape[0] == 0:
                        continue
                    print("### Number of GPUs: %s" % n_gpus, file=f)
                    XX.drop_duplicates(subset=['bits', 'gpus'], keep='last', inplace=True)
                    XX = XX.sort_values(['bits', result_cols[1]], ascending=[False, False])
                    XX['exception'] = XX['exception'].astype(str).replace("nan", "")
                    print(XX[['bits', 'gpus', result_cols[0], result_cols[1], 'exception']].to_markdown(index=False), file=f)
