#!/bin/bash
set -o pipefail
set -ex

export DEBIAN_FRONTEND=noninteractive
export PATH=/h2ogpt_conda/bin:$PATH
export HOME=/workspace
export CUDA_HOME=/usr/local/cuda-11.8

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
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh && \
    mkdir -p /h2ogpt_conda && \
    bash ./Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -u -p /h2ogpt_conda && \
    conda install python=3.10 pygobject weasyprint -c conda-forge -y

# Install base python dependencies
python3.10 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.txt --extra-index-url https://download.pytorch.org/whl/cu118
python3.10 -m pip install -r reqs_optional/requirements_optional_gpt4all.txt --extra-index-url https://download.pytorch.org/whl/cu118
python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt --extra-index-url https://download.pytorch.org/whl/cu118
python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.urls.txt --extra-index-url https://download.pytorch.org/whl/cu118

python3.10 -m pip install -r reqs_optional/requirements_optional_doctr.txt --extra-index-url https://download.pytorch.org/whl/cu118
# go back to older onnx so Tesseract OCR still works
python3.10 -m pip install onnxruntime==1.15.0 onnxruntime-gpu==1.15.0 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    python3.10 -m pip uninstall -y weasyprint && \
    python3.10 -m pip install weasyprint
chmod -R a+rwx /h2ogpt_conda

# Install prebuilt dependencies
for i in 1 2 3 4; do python3.10 -m nltk.downloader all && break || sleep 1; done  # retry as frequently fails with github downloading issues
python3.10 -m pip install https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.4.2/auto_gptq-0.4.2+cu118-cp310-cp310-linux_x86_64.whl
python3.10 -m pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.1.73+cu118-cp310-cp310-linux_x86_64.whl
python3.10 -m pip install https://github.com/jllllll/exllama/releases/download/0.0.13/exllama-0.0.13+cu118-cp310-cp310-linux_x86_64.whl --no-cache-dir
playwright install --with-deps

# Uninstall duckdb and use own so can control thread count per db
python3.10 -m pip uninstall -y pyduckdb duckdb && \
python3.10 -m pip install https://s3.amazonaws.com/artifacts.h2o.ai/deps/h2ogpt/duckdb-0.8.2.dev4026%2Bgdcd8c1ffc5-cp310-cp310-linux_x86_64.whl --no-cache-dir --force-reinstall

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

# Install vllm
# gputil is for rayWorker in vllm to run as non-root
export VLLM_CACHE=/workspace/.vllm_cache
cd /h2ogpt_conda && python -m venv vllm_env --system-site-packages
sp=`python3.10 -c 'import site; print(site.getsitepackages()[0])'` && \
    sed -i 's/posthog\.capture/return\n            posthog.capture/' $sp/chromadb/telemetry/posthog.py && \
    cd $sp && \
    rm -rf openai_vllm* && \
    cp -a openai openai_vllm && \
    file0=`ls|grep openai|grep dist-info` && \
    file1=`echo $file0|sed 's/openai-/openai_vllm-/g'` && \
    cp -a $file0 $file1 && \
    find openai_vllm -name '*.py' | xargs sed -i 's/from openai /from openai_vllm /g' && \
    find openai_vllm -name '*.py' | xargs sed -i 's/openai\./openai_vllm./g' && \
    find openai_vllm -name '*.py' | xargs sed -i 's/from openai\./from openai_vllm./g' && \
    find openai_vllm -name '*.py' | xargs sed -i 's/import openai/import openai_vllm/g' && \
    find openai_vllm -name '*.py' | xargs sed -i 's/OpenAI/vLLM/g' && \
    cd /h2ogpt_conda && \
    python -m venv vllm_env --system-site-packages && \
    /h2ogpt_conda/vllm_env/bin/python -m pip install vllm ray pandas gputil==1.4.0 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    mkdir $VLLM_CACHE
chmod -R a+rwx /h2ogpt_conda

# Make sure old python location works in case using scripts from old documentation
mkdir -p /h2ogpt_conda/envs/vllm/bin && \
    ln -s /h2ogpt_conda/vllm_env/bin/python3.10 /h2ogpt_conda/envs/vllm/bin/python3.10

# Track build info
cd /workspace && make build_info.txt
cp /workspace/build_info.txt /build_info.txt

mkdir -p /workspace/save
chmod -R a+rwx /workspace/save

# Cleanup
rm -rf /workspace/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
rm -rf /workspace/.cache/pip
