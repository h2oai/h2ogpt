### Containerized Installation for Inference on Linux GPU Servers

1. Ensure docker installed and ready (requires sudo), can skip if system is already capable of running nvidia containers.  Example here is for Ubuntu, see [NVIDIA Containers](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for more examples.

```bash
sudo apt-get update     && sudo apt-get install -y nvidia-container-toolkit-base
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

2. Build the container image:

```bash
docker build -t h2o-llm .
```

3. Run the container (you can also use `finetune.py` and all of its parameters as shown above for training):

For the fine-tuned h2oGPT with 20 billion parameters:
```bash
docker run --runtime=nvidia --shm-size=64g -p 7860:7860 \
    -v ${HOME}/.cache:/root/.cache --rm h2o-llm -it generate.py \
    --base_model=h2oai/h2ogpt-oasst1-512-20b \
    --prompt_type=human_bot
`````

For your own fine-tuned model starting from the gpt-neox-20b foundation model for example:
```bash
docker run --runtime=nvidia --shm-size=64g -p 7860:7860 \
    -v ${HOME}/.cache:/root/.cache --rm h2o-llm -it generate.py \
    --base_model=EleutherAI/gpt-neox-20b \
    --lora_weights=h2ogpt_lora_weights --prompt_type=human_bot
```

4. Open `https://localhost:7860` in the browser

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


