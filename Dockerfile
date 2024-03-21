# devel needed for bitsandbytes requirement of libcudart.so, otherwise runtime sufficient
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

ENV PATH="/h2ogpt_conda/bin:${PATH}"
ARG PATH="/h2ogpt_conda/bin:${PATH}"

ENV HOME=/workspace
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121 https://huggingface.github.io/autogptq-index/whl/cu121"

RUN \
    apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    software-properties-common \
    pandoc \
    vim \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    libreoffice \
    autoconf \
    libtool \
    unzip \
    xvfb \
    libxi6 \
    libgconf-2-4 \
    libu2f-udev \
    default-jdk \
    && apt-get upgrade -y

# Install conda
RUN mkdir -p /h2ogpt_conda && chmod -R a+rwx /h2ogpt_conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh && \
    bash ./Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -u -p /h2ogpt_conda && \
    conda install python=3.10 pygobject weasyprint -c conda-forge -y && \
    rm -rf ./Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

WORKDIR /workspace

COPY . /workspace/

# Track build info
RUN make build_info.txt git_hash.txt && \
    cp /workspace/build_info.txt /build_info.txt && \
    cp /workspace/git_hash.txt /git_hash.txt

# install system and python dependencies
# if building for CPU, would remove CMAKE_ARGS and avoid GPU image as base image
ENV LLAMA_CUBLAS=1
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all"
ENV FORCE_CMAKE=1
RUN ./docs/linux_install.sh && \
    rm -rf /workspace/.cache/pip

# pre-load tiktoken cache
ENV TIKTOKEN_CACHE_DIR=/workspace/tiktoken_cache
RUN ./docker_build_script_tiktoken_cache.sh

# install vllm
ENV VLLM_CACHE=/workspace/.vllm_cache
RUN ./docker_build_script_vllm.sh  \
    && rm -rf /workspace/.cache/pip

# create save directory with open permissions
RUN mkdir -p /workspace/save && \
    chmod -R a+rwx /workspace/save

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
