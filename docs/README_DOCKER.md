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

All available public h2oGPT docker images can be found in [Google Container Registry](https://console.cloud.google.com/gcr/images/vorvan/global/h2oai/h2ogpt-runtime).

Ensure image is up-to-date by running:
```bash
docker pull gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0
```

An example running h2oGPT via docker using LLaMa2 7B model is:
```bash
mkdir -p ~/.cache
mkdir -p ~/save
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
then go to http://localhost:7860/ or http://127.0.0.1:7860/.

(`mkdir -p ~/save` prior to running docker to make sure those directories exist, and are created by the local user in case dockerd was installed with root, not that this is true for any other directories you wish to mount to the container as a volume).

An example of running h2oGPT via docker using AutoGPTQ (4-bit, so using less GPU memory) with LLaMa2 7B model is:
```bash
mkdir -p ~/.cache
mkdir -p ~/save
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
       -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
       gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0 /workspace/generate.py \
          --base_model=TheBloke/Llama-2-7b-Chat-GPTQ \
          --load_gptq="gptq_model-4bit-128g" \
          --use_safetensors=True \
          --prompt_type=llama2 \
          --save_dir='/workspace/save/' \
          --use_gpu_id=False \
          --score_model=None \
          --max_max_new_tokens=2048 \
          --max_new_tokens=1024
```
then go to http://localhost:7860/ or http://127.0.0.1:7860/.

If one needs to use a Hugging Face token to access certain Hugging Face models like Meta version of LLaMa2, can run like:
```bash
mkdir -p ~/.cache
mkdir -p ~/save
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
       -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
       gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0 /workspace/generate.py \
          --base_model=h2oai/h2ogpt-4096-llama2-7b-chat \
          --prompt_type=llama2 \
          --save_dir='/workspace/save/' \
          --use_gpu_id=False \
          --score_model=None \
          --max_max_new_tokens=2048 \
          --max_new_tokens=1024 \
          --use_auth_token=$HUGGING_FACE_HUB_TOKEN
```
for some token `<hf_...>`.  See [Hugging Face User Tokens](https://huggingface.co/docs/hub/security-tokens) for more details.

For [GGML/GPT4All models](FAQ.md#adding-models), one should either download the file and map that path outsider docker to a pain told to h2oGPT for inside docker, or pass a URL that would download the model internally to docker.

See [README_GPU](README_GPU.md) for more details about what to run.

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
export MODEL=h2oai/h2ogpt-4096-llama2-7b-chat
unset CUDA_VISIBLE_DEVICES
docker run -d --gpus '"device=0"' \
        --shm-size 1g \
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
mkdir -p ~/.cache
mkdir -p ~/save
docker run -d \
       --gpus all \
       --runtime=nvidia \
       --shm-size=2g \
       -p $GRADIO_SERVER_PORT:7860 \
       --rm --init \
       --network host \
       -v /etc/passwd:/etc/passwd:ro \
       -v /etc/group:/etc/group:ro \
       -u `id -u`:`id -g` \
       -v "${HOME}"/.cache:/workspace/.cache \
       -v "${HOME}"/save:/workspace/save \
       -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
       -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
       gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0 /workspace/generate.py \
          --base_model=$MODEL \
          --inference_server=http://localhost:6112 \
          --prompt_type=llama2 \
          --save_dir='/workspace/save/' \
          --use_gpu_id=False \
          --score_model=None \
          --max_max_new_tokens=4096 \
          --max_new_tokens=1024 \
          --use_auth_token="$HUGGING_FACE_HUB_TOKEN"
```
or change `max_max_new_tokens` to `2048` for low-memory case.

For maximal summarization performance when connecting to TGI server, auto-detection of file changes in `--user_path` every query, and maximum document filling of context, add these options:
```
          --num_async=10 \
          --top_k_docs=-1
          --detect_user_path_changes_every_query=True
```
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

