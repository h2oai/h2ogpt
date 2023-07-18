### Containerized Installation for Inference on Linux GPU Servers

1. Ensure docker installed and ready (requires sudo), can skip if system is already capable of running nvidia containers.  Example here is for Ubuntu, see [NVIDIA Containers](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) for more examples.

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

2. Build the container image:

    ```bash
    docker build -t h2ogpt .
    ```

3. Run the container (you can also use `finetune.py` and all of its parameters as shown above for training):

    For the fine-tuned h2oGPT with 20 billion parameters:
    ```bash
    docker run --runtime=nvidia --shm-size=64g -p 7860:7860 \
        -v ${HOME}/.cache:/root/.cache --rm h2ogpt -it generate.py \
        --base_model=h2oai/h2ogpt-oasst1-512-20b
    ```
    
    if have a private HF token, can instead run:
    ```bash
    docker run --runtime=nvidia --shm-size=64g --entrypoint=bash -p 7860:7860 \
    -e HUGGINGFACE_API_TOKEN=<HUGGINGFACE_API_TOKEN> \
    -v ${HOME}/.cache:/root/.cache --rm h2ogpt -it \
     -c 'huggingface-cli login --token $HUGGINGFACE_API_TOKEN && python3.10 generate.py --base_model=h2oai/h2ogpt-oasst1-512-20b --use_auth_token=True'
    ```
   
    For your own fine-tuned model starting from the gpt-neox-20b foundation model for example:
    ```bash
    docker run --runtime=nvidia --shm-size=64g -p 7860:7860 \
        -v ${HOME}/.cache:/root/.cache --rm h2ogpt -it generate.py \
        --base_model=EleutherAI/gpt-neox-20b \
        --lora_weights=h2ogpt_lora_weights --prompt_type=human_bot
    ```

4. Open `https://localhost:7860` in the browser



### Run h2oGPT using Docker
__Optional: Running with a custom entrypoint__

To run with a custom entrypoint, modify the local [`run-gpt.sh`](https://github.com/h2oai/h2ogpt/blob/76947c009a82d7a4a871548e68a60ce0a28b75d1/run-gpt.sh) & mount it.

```bash
docker run \
    --runtime=nvidia --shm-size=64g \
    -e HF_MODEL=h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b \
    -p 8888:8888 -p 7860:7860 \
    --rm --init \
    -v `pwd`/h2ogpt_env:/h2ogpt_env \
    -v `pwd`/run-gpt.sh:/run-gpt.sh \
    gcr.io/vorvan/h2oai/h2ogpt-runtime:61d6aea6fff3b1190aa42eee7fa10d6c
```

### Docker Compose Setup & Inference

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

