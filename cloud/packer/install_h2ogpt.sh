#!/bin/bash -e

export DEBIAN_FRONTEND=noninteractive
export PATH=$PATH:/home/ubuntu/.local/bin
export PATH=/h2ogpt_conda/bin:$PATH
export HOME=/workspace
export CUDA_HOME=/usr/local/cuda-12.1

sudo mkdir -p /workspace && cd /workspace
sudo chmod a+rwx .

git config --global --add safe.directory /workspace
git config --global advice.detachedHead false
git clone https://github.com/h2oai/h2ogpt.git .

if [ -z "$BRANCH_TAG" ]; then
  echo "BRANCH_TAG environment variable is not set."
  exit 1
fi

git checkout $BRANCH_TAG

ls -la

sudo apt-get update && apt-get install -y \
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
    && apt-get upgrade -y

# Install conda
sudo mkdir -p /h2ogpt_conda && chmod -R a+rwx /h2ogpt_conda
sudo wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh && \
    bash ./Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -u -p /h2ogpt_conda && \
    conda install python=3.10 pygobject weasyprint -c conda-forge -y

sudo ./docs/linux_install.sh
sudo ./docker_build_script_tiktoken_cache.sh
sudo ./docker_build_script_vllm.sh

# Track build info
sudo cd /workspace && \
    make build_info.txt git_hash.txt && \
    cp /workspace/build_info.txt /build_info.txt && \
    cp /workspace/git_hash.txt /git_hash.txt

sudo mkdir -p /workspace/save && \
    chmod -R a+rwx /workspace/save

# Cleanup
sudo rm -rf /workspace/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh && \
    rm -rf /workspace/.cache/pip
