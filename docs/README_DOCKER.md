# Run or Build h2oGPT Docker

## Setup Docker for CPU Inference

No special docker instructions are required, just follow [these instructions](https://docs.docker.com/engine/install/ubuntu/) to get docker setup at all, i.e.:
```bash
sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
apt-cache policy docker-ce
sudo apt install docker-ce
sudo systemctl status docker
```

Add your user as part of `docker` group, exit shell, login back in, and run:
```bash
newgrp docker
```
which avoids having to reboot.  Or just reboot to have docker access.  If this cannot be done without entering root access, then edit the `/etc/group` and add your user to group `docker`.

## Setup Docker for GPU Inference

Ensure docker installed and ready (requires sudo), can skip if system is already capable of running nvidia containers.  Example here is for Ubuntu, see [NVIDIA Containers](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) for more examples.
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit-base
sudo apt install -y nvidia-container-runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Confirm runs nvidia-smi from within docker without errors:
```bash
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

If running on A100's, might require [Installing Fabric Manager](INSTALL.md#install-and-run-fabric-manager-if-have-multiple-a100100s) and [Installing GPU Manager](INSTALL.md#install-nvidia-gpu-manager-if-have-multiple-a100h100s).

## Run h2oGPT using Docker

All available public h2oGPT docker images can be found in [Google Container Registry](https://console.cloud.google.com/gcr/images/vorvan/global/h2oai/h2ogpt-runtime).

Ensure image is up-to-date by running:
```bash
docker pull gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0
```

An example running h2oGPT via docker using LLaMa2 7B model is:
```bash
mkdir -p ~/.cache
mkdir -p ~/save
mkdir -p ~/user_path
mkdir -p ~/db_dir_UserData
mkdir -p ~/users
mkdir -p ~/db_nonusers
export CUDA_VISIBLE_DEVICES=0
docker run \
       --gpus all \
       --runtime=nvidia \
       --shm-size=2g \
       -p 7860:7860 \
       --rm --init \
       --network host \
       -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
       -v /etc/passwd:/etc/passwd:ro \
       -v /etc/group:/etc/group:ro \
       -u `id -u`:`id -g` \
       -v "${HOME}"/.cache:/workspace/.cache \
       -v "${HOME}"/save:/workspace/save \
       -v "${HOME}"/user_path:/workspace/user_path \
       -v "${HOME}"/db_dir_UserData:/workspace/db_dir_UserData \
       -v "${HOME}"/users:/workspace/users \
       -v "${HOME}"/db_nonusers:/workspace/db_nonusers \
       gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0 /workspace/generate.py \
          --base_model=h2oai/h2ogpt-4096-llama2-7b-chat \
          --use_safetensors=True \
          --prompt_type=llama2 \
          --save_dir='/workspace/save/' \
          --use_gpu_id=False \
          --score_model=None \
          --max_max_new_tokens=2048 \
          --max_new_tokens=1024
```
Use `docker run -d` to run in detached background. Then go to http://localhost:7860/ or http://127.0.0.1:7860/.

An example of running h2oGPT via docker using AutoGPTQ (4-bit, so using less GPU memory) with LLaMa2 7B model is:
```bash
mkdir -p $HOME/.cache
mkdir -p $HOME/save
mkdir -p ~/user_path
mkdir -p ~/db_dir_UserData
mkdir -p ~/users
mkdir -p ~/db_nonusers
export CUDA_VISIBLE_DEVICES=0
docker run \
       --gpus all \
       --runtime=nvidia \
       --shm-size=2g \
       -p 7860:7860 \
       --rm --init \
       --network host \
       -v /etc/passwd:/etc/passwd:ro \
       -v /etc/group:/etc/group:ro \
       -u `id -u`:`id -g` \
       -v "${HOME}"/.cache:/workspace/.cache \
       -v "${HOME}"/save:/workspace/save \
       -v "${HOME}"/user_path:/workspace/user_path \
       -v "${HOME}"/db_dir_UserData:/workspace/db_dir_UserData \
       -v "${HOME}"/users:/workspace/users \
       -v "${HOME}"/db_nonusers:/workspace/db_nonusers \
       -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
       gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0 /workspace/generate.py \
          --base_model=TheBloke/Llama-2-7b-Chat-GPTQ \
          --load_gptq=model \
          --use_safetensors=True \
          --prompt_type=llama2 \
          --save_dir='/workspace/save/' \
          --use_gpu_id=False \
          --score_model=None \
          --max_max_new_tokens=2048 \
          --max_new_tokens=1024
```
Use `docker run -d` to run in detached background.  Then go to http://localhost:7860/ or http://127.0.0.1:7860/.

If one needs to use a Hugging Face token to access certain Hugging Face models like Meta version of LLaMa2, can run like:
```bash
mkdir -p ~/.cache
mkdir -p ~/save
mkdir -p ~/user_path
mkdir -p ~/db_dir_UserData
mkdir -p ~/users
mkdir -p ~/db_nonusers
export CUDA_VISIBLE_DEVICES=0
docker run \
       --gpus all \
       --runtime=nvidia \
       --shm-size=2g \
       -p 7860:7860 \
       --rm --init \
       --network host \
       -v /etc/passwd:/etc/passwd:ro \
       -v /etc/group:/etc/group:ro \
       -u `id -u`:`id -g` \
       -v "${HOME}"/.cache:/workspace/.cache \
       -v "${HOME}"/save:/workspace/save \
       -v "${HOME}"/user_path:/workspace/user_path \
       -v "${HOME}"/db_dir_UserData:/workspace/db_dir_UserData \
       -v "${HOME}"/users:/workspace/users \
       -v "${HOME}"/db_nonusers:/workspace/db_nonusers \
       -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
       gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0 /workspace/generate.py \
          --base_model=h2oai/h2ogpt-4096-llama2-7b-chat \
          --prompt_type=llama2 \
          --save_dir='/workspace/save/' \
          --use_gpu_id=False \
          --score_model=None \
          --max_max_new_tokens=2048 \
          --max_new_tokens=1024
```
Use `docker run -d` to run in detached background.

For [GGML/GPT4All models](FAQ.md#adding-models), one should either download the file and map that path outsider docker to a pain told to h2oGPT for inside docker, or pass a URL that would download the model internally to docker.

See [README_GPU](README_GPU.md) for more details about what to run.

## Run h2oGPT +  vLLM or vLLM using Docker

One can run an inference server in one docker and h2oGPT in another docker.

For the vLLM server running on 2 GPUs using h2oai/h2ogpt-4096-llama2-7b-chat model, run:
```bash
docker pull gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0
unset CUDA_VISIBLE_DEVICES
mkdir -p $HOME/.cache/huggingface/hub
mkdir -p $HOME/save
docker run \
    --runtime=nvidia \
    --gpus '"device=0,1"' \
    --shm-size=10.24gb \
    -p 5000:5000 \
    --rm --init \
    --entrypoint /h2ogpt_conda/vllm_env/bin/python3.10 \
    -e NCCL_IGNORE_DISABLED_P2P=1 \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -u `id -u`:`id -g` \
    -v "${HOME}"/.cache:/workspace/.cache \
    --network host \
    gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0 -m vllm.entrypoints.openai.api_server \
        --port=5000 \
        --host=0.0.0.0 \
        --model=h2oai/h2ogpt-4096-llama2-7b-chat \
        --tokenizer=hf-internal-testing/llama-tokenizer \
        --tensor-parallel-size=2 \
        --seed 1234 \
        --trust-remote-code \
        --download-dir=/workspace/.cache/huggingface/hub &>> logs.vllm_server.txt
```
Use `docker run -d` to run in detached background.

Checks the logs `logs.vllm_server.txt` to make sure server is running.
If ones sees similar output to below, then endpoint it up & running.
```bash
INFO:     Started server process [7]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5000 (Press CTRL+C to quit
```

### Curl Test


One can also verify the endpoint by running following curl command.
```bash
curl http://localhost:5000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
    "model": "h2oai/h2ogpt-4096-llama2-7b-chat",
    "prompt": "San Francisco is a",
    "max_tokens": 7,
    "temperature": 0
    }'
```
If one sees similar output to below, then endpoint it up & running.

```json
{
    "id": "cmpl-4b9584f743ff4dc590f0c168f82b063b",
    "object": "text_completion",
    "created": 1692796549,
    "model": "h2oai/h2ogpt-4096-llama2-7b-chat",
    "choices": [
        {
            "index": 0,
            "text": "city in Northern California that is known",
            "logprobs": null,
            "finish_reason": "length"
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "total_tokens": 12,
        "completion_tokens": 7
    }
}
```

If one needs to only setup vLLM one can stop here.

### Run h2oGPT
```bash
mkdir -p ~/.cache
mkdir -p ~/save
mkdir -p ~/user_path
mkdir -p ~/db_dir_UserData
mkdir -p ~/users
mkdir -p ~/db_nonusers
docker run \
    --gpus '"device=2,3"' \
    --runtime=nvidia \
    --shm-size=2g \
    -p 7860:7860 \
    --rm --init \
    --network host \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -u `id -u`:`id -g` \
    -v "${HOME}"/.cache:/workspace/.cache \
    -v "${HOME}"/save:/workspace/save \
       -v "${HOME}"/user_path:/workspace/user_path \
       -v "${HOME}"/db_dir_UserData:/workspace/db_dir_UserData \
       -v "${HOME}"/users:/workspace/users \
       -v "${HOME}"/db_nonusers:/workspace/db_nonusers \
    gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0 /workspace/generate.py \
        --inference_server="vllm:0.0.0.0:5000" \
        --base_model=h2oai/h2ogpt-4096-llama2-7b-chat \
        --langchain_mode=UserData
```

Make sure to set `--inference_server` argument to the correct vllm endpoint.

When one is done with the docker instance, run `docker ps` and find the container ID's hash, then run `docker stop <hash>`.

Follow [README_InferenceServers.md](README_InferenceServers.md) for more information on how to setup vLLM.

## Run h2oGPT and TGI using Docker

One can run an inference server in one docker and h2oGPT in another docker.

For the TGI server run (e.g. to run on GPU 0)
```bash
export MODEL=h2oai/h2ogpt-4096-llama2-7b-chat
export CUDA_VISIBLE_DEVICES=0
docker run -d --gpus all \
       --shm-size 1g \
       --network host \
       -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
       -p 6112:80 \
       -v $HOME/.cache/huggingface/hub/:/data ghcr.io/huggingface/text-generation-inference:0.9.3 \
       --model-id $MODEL \
       --max-input-length 4096 \
       --max-total-tokens 8192 \
       --max-stop-sequences 6 &>> logs.infserver.txt
```
Each docker can run on any system where network can reach or on same system on different GPUs.  E.g. replace `--gpus all` with `--gpus '"device=0,3"'` to run on GPUs 0 and 3, and note the extra quotes, and then `unset CUDA_VISIBLE_DEVICES` and avoid passing that into the docker image.  This multi-device format is required to avoid TGI server getting confused about which GPUs are available.

One a low-memory GPU system can add other options to limit batching, e.g.:
```bash
mkdir -p $HOME/.cache/huggingface/hub/
export MODEL=h2oai/h2ogpt-4096-llama2-7b-chat
unset CUDA_VISIBLE_DEVICES
docker run -d --gpus '"device=0"' \
        --shm-size 1g \
        -p 6112:80 \
        -v $HOME/.cache/huggingface/hub/:/data ghcr.io/huggingface/text-generation-inference:0.9.3 \
        --model-id $MODEL \
        --max-input-length 1024 \
        --max-total-tokens 2048 \
        --max-batch-prefill-tokens 2048 \
        --max-batch-total-tokens 2048 \
        --max-stop-sequences 6 &>> logs.infserver.txt
```

Then wait till it comes up (e.g. check docker logs for detached container hash in logs.infserver.txt), about 30 seconds for 7B LLaMa2 on 1 GPU.  Then for h2oGPT, just run one of the commands like the above, but add e.g. `--inference_server=192.168.0.1:6112` to the docker command line.  E.g. using same export's as above, run:
```bash
export GRADIO_SERVER_PORT=7860
export CUDA_VISIBLE_DEVICES=0
mkdir -p ~/.cache
mkdir -p ~/save
mkdir -p ~/user_path
mkdir -p ~/db_dir_UserData
mkdir -p ~/users
mkdir -p ~/db_nonusers
docker run -d \
       --gpus all \
       --runtime=nvidia \
       --shm-size=2g \
       -p $GRADIO_SERVER_PORT:$GRADIO_SERVER_PORT \
       --rm --init \
       --network host \
       -v /etc/passwd:/etc/passwd:ro \
       -v /etc/group:/etc/group:ro \
       -u `id -u`:`id -g` \
       -v "${HOME}"/.cache:/workspace/.cache \
       -v "${HOME}"/save:/workspace/save \
       -v "${HOME}"/user_path:/workspace/user_path \
       -v "${HOME}"/db_dir_UserData:/workspace/db_dir_UserData \
       -v "${HOME}"/users:/workspace/users \
       -v "${HOME}"/db_nonusers:/workspace/db_nonusers \
       -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
       gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0 /workspace/generate.py \
          --base_model=$MODEL \
          --inference_server=http://localhost:6112 \
          --prompt_type=llama2 \
          --save_dir='/workspace/save/' \
          --use_gpu_id=False \
          --score_model=None \
          --max_max_new_tokens=4096 \
          --max_new_tokens=1024
```
or change `max_max_new_tokens` to `2048` for low-memory case.  Note the h2oGPT container has `--network host` with same port inside and outside so the other container on same host can see it.  Otherwise use actual IP addersses if on separate hosts.

For maximal summarization performance when connecting to TGI server, auto-detection of file changes in `--user_path` every query, and maximum document filling of context, add these options:
```
          --num_async=10 \
          --top_k_docs=-1
          --detect_user_path_changes_every_query=True
```
When one is done with the docker instance, run `docker ps` and find the container ID's hash, then run `docker stop <hash>`.

Follow [README_InferenceServers.md](README_InferenceServers.md) for similar (and more) examples of how to launch TGI server using docker.

## Make UserData db for generate.py using Docker

To make UserData db for generate.py, put pdfs, etc. into path user_path and run:
```bash
mkdir -p ~/.cache
mkdir -p ~/save
mkdir -p ~/user_path
mkdir -p ~/db_dir_UserData
docker run \
       --gpus all \
       --runtime=nvidia \
       --shm-size=2g \
       --rm --init \
       --network host \
       -v /etc/passwd:/etc/passwd:ro \
       -v /etc/group:/etc/group:ro \
       -u `id -u`:`id -g` \
       -v "${HOME}"/.cache:/workspace/.cache \
       -v "${HOME}"/save:/workspace/save \
       -v "${HOME}"/user_path:/workspace/user_path \
       -v "${HOME}"/db_dir_UserData:/workspace/db_dir_UserData \
       gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0 /workspace/src/make_db.py
```

Once db is made, can use in generate.py like:
```bash
mkdir -p ~/.cache
mkdir -p ~/save
mkdir -p ~/user_path
mkdir -p ~/db_dir_UserData
mkdir -p ~/users
mkdir -p ~/db_nonusers
export CUDA_VISIBLE_DEVICES=0
docker run \
       --gpus all \
       --runtime=nvidia \
       --shm-size=2g \
       -p 7860:7860 \
       --rm --init \
       --network host \
       -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
       -v /etc/passwd:/etc/passwd:ro \
       -v /etc/group:/etc/group:ro \
       -u `id -u`:`id -g` \
       -v "${HOME}"/.cache:/workspace/.cache \
       -v "${HOME}"/save:/workspace/save \
       -v "${HOME}"/user_path:/workspace/user_path \
       -v "${HOME}"/db_dir_UserData:/workspace/db_dir_UserData \
       -v "${HOME}"/users:/workspace/users \
       -v "${HOME}"/db_nonusers:/workspace/db_nonusers \
       gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0 /workspace/generate.py \
          --base_model=h2oai/h2ogpt-4096-llama2-7b-chat \
          --use_safetensors=True \
          --prompt_type=llama2 \
          --save_dir='/workspace/save/' \
          --use_gpu_id=False \
          --score_model=None \
          --max_max_new_tokens=2048 \
          --max_new_tokens=1024 \
          --langchain_mode=UserData
```

For a more detailed description of other parameters of the make_db script, checkout the definition in this file: https://github.com/h2oai/h2ogpt/blob/main/src/make_db.py

## Build Docker

```bash
# build image
touch build_info.txt
docker build -t h2ogpt .
```
then to run this version of the docker image, just replace `gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0` with `h2ogpt:latest` in above run command.
when any of the prebuilt dependencies are changed, e.g. duckdb or auto-gptq, you need to run `make docker_build_deps` or similar code what's in that Makefile target.

## Docker Compose Setup & Inference

1. (optional) Change desired model and weights under `environment` in the `docker-compose.yml`

2. Build and run the container

    ```bash
    docker-compose up -d --build
    ```

3. Open `https://localhost:7860` in the browser

4. See logs:

    ```bash
    docker-compose logs -f
    ```

5. Clean everything up:

    ```bash
    docker-compose down --volumes --rmi all
    ```

