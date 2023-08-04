# Run or Build h2oGPT Docker

## Setup Docker for CPU Inference

No special docker instructions are required, just follow [these instructions](https://docs.docker.com/engine/install/ubuntu/) to get docker setup at all.  Add your user as part of `docker` group, exit shell, login back in, and run:
```bash
newgrp docker
```
which avoids having to reboot.  Or just reboot to have docker access.

## Setup Docker for GPU Inference

Ensure docker installed and ready (requires sudo), can skip if system is already capable of running nvidia containers.  Example here is for Ubuntu, see [NVIDIA Containers](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) for more examples.
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit-base
sudo apt install nvidia-container-runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

If running on A100's, might require [Installing Fabric Manager](INSTALL.md#install-and-run-fabric-manager-if-have-multiple-a100100s) and [Installing GPU Manager](INSTALL.md#install-nvidia-gpu-manager-if-have-multiple-a100h100s).

## Run h2oGPT using Docker

An example of running h2oGPT via docker using AutoGPTQ LLaMa2 7B model is as follows.  First, ensure you have the latest docker image:
```bash
docker pull gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0
```

All available public h2oGPT docker images can be found in [Google Container Registry](https://console.cloud.google.com/gcr/images/vorvan/global/h2oai/h2ogpt-runtime).

then run:
```bash
docker run \
       --gpus all \
       --runtime=nvidia \
       --shm-size=2g \
       -p 7860:7860 \
       --rm --init \
       -v "${HOME}"/.cache:/workspace/.cache \
       -v "${HOME}"/save:/workspace/save \
       gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0 /workspace/generate.py \
          --base_model=TheBloke/Llama-2-7b-Chat-GPTQ \
          --load_gptq="gptq_model-4bit-128g" \
          --use_safetensors=True \
          --prompt_type=llama2 \
          --save_dir='/workspace/save/' \
          --score_model=None \
          --max_max_new_tokens=2048 \
          --max_new_tokens=1024 \
          --num_async=10 \
          --top_k_docs=-1
```
then go to http://localhost:7860/ or http://127.0.0.1:7860/.

If one needs to use a Hugging Face token to access certain Hugging Face models like Meta version of LLaMa2, can run like:
```bash
export HUGGING_FACE_HUB_TOKEN=<hf_...>
docker run \
       --gpus all \
       --runtime=nvidia \
       --shm-size=2g \
       -p 7860:7860 \
       --rm --init \
       -v "${HOME}"/.cache:/workspace/.cache \
       -v "${HOME}"/save:/workspace/save \
       -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
       gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0 /workspace/generate.py \
          --base_model=meta-llama/Llama-2-7b-chat-hf \
          --prompt_type=llama2 \
          --save_dir='/workspace/save/' \
          --score_model=None \
          --max_max_new_tokens=2048 \
          --max_new_tokens=1024 \
          --num_async=10 \
          --top_k_docs=-1 \
          --use_auth_token=$HUGGING_FACE_HUB_TOKEN
```
for some token `<hf_...>`.  See [Hugging Face User Tokens](https://huggingface.co/docs/hub/security-tokens) for more details.

See [README_GPU](README_GPU.md) for more details about what to run.

## Run h2oGPT and TGI using Docker

One can run an inference server in one docker and h2oGPT in another docker.

For the TGI server run (e.g. to run on GPU 0)
```bash
export MODEL=meta-llama/Llama-2-7b-chat-hf
export HUGGING_FACE_HUB_TOKEN=<hf_...>
export CUDA_VISIBLE_DEVICES=0
docker run -d --gpus all \
       --shm-size 1g \
       -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
       -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
       -e TRANSFORMERS_CACHE="/.cache/" \
       -p 6112:80 \
       -v $HOME/.cache:/.cache/ \
       -v $HOME/.cache/huggingface/hub/:/data ghcr.io/huggingface/text-generation-inference:0.9.3 \
       --model-id $MODEL \
       --max-input-length 4096 \
       --max-total-tokens 8192 \
       --max-stop-sequences 6 &>> logs.infserver.txt
```
Each docker can run on any system where network can reach or on same system on different GPUs.  E.g. replace `--gpus all` with `--gpus '"device=0,3"'` to run on GPUs 0 and 3, and note the extra quotes, and then `unset CUDA_VISIBLE_DEVICES` and avoid passing that into the docker image.  This multi-device format is required to avoid TGI server getting confused about which GPUs are available.

One a low-memory GPU system can add other options to limit batching, e.g.:
```bash
export MODEL=meta-llama/Llama-2-7b-chat-hf
export HUGGING_FACE_HUB_TOKEN=<hf_...>
unset CUDA_VISIBLE_DEVICES
docker run -d --gpus '"device=0"' \
        --shm-size 1g \
        -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
        -e TRANSFORMERS_CACHE="/.cache/" \
        -p 6112:80 \
        -v $HOME/.cache:/.cache/ \
        -v $HOME/.cache/huggingface/hub/:/data ghcr.io/huggingface/text-generation-inference:0.9.3 \
        --model-id $MODEL \
        --max-input-length 1024 \
        --max-total-tokens 2048 \
        --max-batch-prefill-tokens 2048 \
        --max-batch-total-tokens 2048 \
        --max-stop-sequences 6 &>> logs.infserver.txt
```
then wait till it comes up (e.g. check docker logs for detatched container hash in logs.infserver.txt), about 30 seconds for 7B LLaMa2 on 1 GPU.  Then for h2oGPT, just run one of the commands like the above, but add e.g. `--inference_server=192.168.0.1:6112` to the docker command line.  E.g. using same export's as above, run:
```bash
export GRADIO_SERVER_PORT=7860
export CUDA_VISIBLE_DEVICES=0
docker run -d \
       --gpus all \
       --runtime=nvidia \
       --shm-size=2g \
       -p $GRADIO_SERVER_PORT:7860 \
       --rm --init \
       --network host \
       -v "${HOME}"/.cache:/workspace/.cache \
       -v "${HOME}"/save:/workspace/save \
       -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
       gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0 /workspace/generate.py \
          --base_model=$MODEL \
          --inference_server=http://localhost:6112 \
          --prompt_type=llama2 \
          --save_dir='/workspace/save/' \
          --score_model=None \
          --max_max_new_tokens=4096 \
          --max_new_tokens=1024 \
          --num_async=10 \
          --top_k_docs=-1 \
          --use_auth_token="$HUGGING_FACE_HUB_TOKEN"
```
or change `max_max_new_tokens` to `2048` for low-memory case.

When one is done with the docker instance, run `docker ps` and find the container ID's hash, then run `docker stop <hash>`.

Follow [README_InferenceServers.md](README_InferenceServers.md) for similar (and more) examples of how to launch TGI server using docker.

## Build Docker

```bash
# build auto-gptq
make docker_build_deps
# build image
touch build_info.txt
docker build -t h2ogpt .
```
then to run this version of the docker image, just replace `gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0` with `h2ogpt:latest` in above run command.

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

