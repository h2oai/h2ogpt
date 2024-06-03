#!/bin/bash
set -o pipefail
set -ex

export DEBIAN_FRONTEND=noninteractive
export PATH=/h2ogpt_conda/bin:$PATH
export HOME=/workspace
export CUDA_HOME=/usr/local/cuda-12.1
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121 https://huggingface.github.io/autogptq-index/whl/cu121"

# Install linux dependencies
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
    libtool

# Run upgrades
apt-get upgrade -y

# Install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir -p /h2ogpt_conda && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -u -p /h2ogpt_conda && \
    conda update -n base conda && \
    source /h2ogpt_conda/etc/profile.d/conda.sh && \
    conda create -n h2ogpt -y && \
    conda activate h2ogpt && \
    conda install python=3.10 pygobject weasyprint -c conda-forge -y && \
    echo "h2oGPT conda env: $CONDA_DEFAULT_ENV"

# if building for CPU, would remove CMAKE_ARGS and avoid GPU image as base image
# Choose llama_cpp_python ARGS for your system according to [llama_cpp_python backend documentation](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends), e.g. for CUDA:
export LLAMA_CUBLAS=1
export CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all"
export FORCE_CMAKE=1

bash docs/linux_install.sh

chmod -R a+rwx /h2ogpt_conda

# setup tiktoken cache
export TIKTOKEN_CACHE_DIR=/workspace/tiktoken_cache
python3.10 -c "
import tiktoken
from tiktoken_ext import openai_public
# FakeTokenizer etc. needs tiktoken for general tasks
for enc in openai_public.ENCODING_CONSTRUCTORS:
    encoding = tiktoken.get_encoding(enc)
model_encodings = [
    'gpt-4',
    'gpt-4-0314',
    'gpt-4-32k',
    'gpt-4-32k-0314',
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-16k',
    'gpt-3.5-turbo-0301',
    'text-ada-001',
    'ada',
    'text-babbage-001',
    'babbage',
    'text-curie-001',
    'curie',
    'davinci',
    'text-davinci-003',
    'text-davinci-002',
    'code-davinci-002',
    'code-davinci-001',
    'code-cushman-002',
    'code-cushman-001'
]
for enc in model_encodings:
    encoding = tiktoken.encoding_for_model(enc)
print('Done!')
"

############################################################
# vllm server
export VLLM_CACHE=/workspace/.vllm_cache
conda create -n vllm -y
source /h2ogpt_conda/etc/profile.d/conda.sh
conda activate vllm
conda install python=3.10 -y
echo "vLLM conda env: $CONDA_DEFAULT_ENV"

# gputil is for rayWorker in vllm to run as non-root
# below required outside docker:
# apt-get install libnccl2
python -m pip install vllm==0.4.0.post1
python -m pip install gputil==1.4.0 hf_transfer==0.1.6
python -m pip install flash-attn==2.5.6 --no-build-isolation --no-deps --no-cache-dir

# pip install hf_transfer
# pip install tiktoken accelerate flash_attn
mkdir $VLLM_CACHE
chmod -R a+rwx /h2ogpt_conda

# Make sure old python location works in case using scripts from old documentation
mkdir -p /h2ogpt_conda/vllm_env/bin/
ln -s /h2ogpt_conda/envs/vllm/bin/python3.10 /h2ogpt_conda/vllm_env/bin/python3.10

# Track build info
cp /workspace/build_info.txt /build_info.txt
cp /workspace/git_hash.txt /git_hash.txt

mkdir -p /workspace/save
chmod -R a+rwx /workspace/save

# Cleanup
rm -rf /workspace/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
rm -rf /workspace/.cache/pip
rm -rf /h2ogpt_conda/pkgs
rm -rf /workspace/spaces
rm -rf /workspace/benchmarks
rm -rf /workspace/data
rm -rf /workspace/cloud
rm -rf /workspace/docs
rm -rf /workspace/helm
rm -rf /workspace/notebooks
rm -rf /workspace/papers

# Hotswap vulnerable dependencies
wget https://s3.amazonaws.com/artifacts.h2o.ai/deps/h2ogpt/ubuntu20.04/apparmor_4.0.0~alpha2-0ubuntu5_amd64.deb
wget https://s3.amazonaws.com/artifacts.h2o.ai/deps/h2ogpt/ubuntu20.04/libapparmor1_4.0.0~alpha2-0ubuntu5_amd64.deb
dpkg -i libapparmor1_4.0.0~alpha2-0ubuntu5_amd64.deb
dpkg -i apparmor_4.0.0~alpha2-0ubuntu5_amd64.deb
rm -rf libapparmor1_4*.deb apparmor_4*.deb

wget https://s3.amazonaws.com/artifacts.h2o.ai/deps/h2ogpt/ubuntu20.04/libarchive13_3.6.2-1ubuntu1_amd64.deb
dpkg -i libarchive13_3.6.2-1ubuntu1_amd64.deb
rm -rf libarchive13_3.6.2-1ubuntu1_amd64.deb
