# Run h2oGPT using Docker
1. Make sure Docker & Nvidia Containers are setup correctly by following instructions [here](INSTALL-DOCKER.md).

2. Specify the required model using `HF_MODEL` parameter.
    All open-source models are posted on [🤗 H2O.ai's Hugging Face page](https://huggingface.co/h2oai/).
    ```bash
    docker run \
      --runtime=nvidia --shm-size=64g \
      -e HF_MODEL=h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b \
      -p 8888:8888 -p 7860:7860 \
      --rm --init \
      -v `pwd`/h2ogpt_env:/h2ogpt_env \
      gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0
    ```
3. Navigate to http://localhost:7860/  & start using h2oGPT.

To run h2oGPT with custom entrypoint, refer [here](INSTALL-DOCKER.md).
