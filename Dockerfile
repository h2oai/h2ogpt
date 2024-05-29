# devel needed for bitsandbytes requirement of libcudart.so, otherwise runtime sufficient
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ENV PATH="/h2ogpt_conda/envs/h2ogpt/bin:${PATH}"
ARG PATH="/h2ogpt_conda/envs/h2ogpt/bin:${PATH}"

ENV HOME=/workspace
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV VLLM_CACHE=/workspace/.vllm_cache
ENV TIKTOKEN_CACHE_DIR=/workspace/tiktoken_cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace

COPY . /workspace/

COPY build_info.txt /workspace/

COPY git_hash.txt /workspace/

RUN cd /workspace && ./docker_build_script_ubuntu.sh

RUN chmod -R a+rwx /workspace

ARG user=h2ogpt
ARG group=h2ogpt
ARG uid=1000
ARG gid=1000

RUN groupadd -g ${gid} ${group} && useradd -u ${uid} -g ${group} -s /bin/bash ${user}

EXPOSE 8888
EXPOSE 7860
EXPOSE 5000

USER h2ogpt

ENTRYPOINT ["python3.10"]
