FROM gcr.io/vorvan/h2oai/h2ogpt-oss-wolfi-base:1 AS build-stage

USER root

ENV HOME=/workspace
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV VLLM_CACHE=/workspace/.vllm_cache
ENV TIKTOKEN_CACHE_DIR=/workspace/tiktoken_cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace

# copy code
COPY .              /workspace/

# copy build info
COPY build_info.txt /workspace/
COPY git_hash.txt   /workspace/

# copy install script
COPY linux_install_wolfi.sh    /workspace/

# run setup
RUN cd /workspace && ./linux_install_wolfi.sh

EXPOSE 8888
EXPOSE 7860
EXPOSE 5000

USER h2ogpt

ENTRYPOINT ["python3.10"]
