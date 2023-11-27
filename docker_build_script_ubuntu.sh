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

# Selenium
apt install -y unzip xvfb libxi6 libgconf-2-4
apt install -y default-jdk
curl -sS -o - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add
bash -c "echo 'deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main' >> /etc/apt/sources.list.d/google-chrome.list"
apt -y update
apt -y install google-chrome-stable  # e.g. Google Chrome 114.0.5735.198
google-chrome --version  # e.g. Google Chrome 114.0.5735.198
# visit https://chromedriver.chromium.org/downloads and download matching version
# E.g.
wget https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_linux64.zip
unzip chromedriver_linux64.zip
sudo mv chromedriver /usr/bin/chromedriver
sudo chown root:root /usr/bin/chromedriver
sudo chmod +x /usr/bin/chromedriver

# Install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh && \
    mkdir -p /h2ogpt_conda && \
    bash ./Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -u -p /h2ogpt_conda && \
    conda install python=3.10 pygobject weasyprint -c conda-forge -y

export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu118"

# Install base python dependencies
pip install -r requirements.txt
pip install -r reqs_optional/requirements_optional_langchain.txt
pip install -r reqs_optional/requirements_optional_gpt4all.txt
# for commercial purposes remove the below line
pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt
pip install -r reqs_optional/requirements_optional_langchain.urls.txt

pip install -r reqs_optional/requirements_optional_doctr.txt
# go back to older onnx so Tesseract OCR still works
pip install onnxruntime==1.15.0 onnxruntime-gpu==1.15.0
# need python weasyprint, not just conda library
pip install weasyprint==60.1

# STT from microphone
pip install pydub==0.25.1 librosa==0.10.1 ffmpeg==1.4 yt_dlp==2023.10.13 wavio==0.0.8
# For STT below may also be required
apt-get install -y ffmpeg
# for TTS:
pip install torchaudio soundfile==0.12.1
# for Coqui XTTS (ensure CUDA_HOME set and consistent with added postfix for extra-index):
pip install TTS deepspeed noisereduce pydantic==1.10.13 emoji ffmpeg-python==0.2.0 trainer pysbd coqpit
# undo excessive TTS constraint on transformers that seems to have no impact on h2oGPT usage
pip install transformers==4.35.0
# for Coqui XTTS language helpers (specific versions probably not required)
pip install cutlet==0.3.0 langid==1.1.6 g2pkk==0.1.2 jamo==0.4.1 gruut[de,es,fr]==2.2.3 jieba==0.42.1

chmod -R a+rwx /h2ogpt_conda

# Install prebuilt dependencies
for i in 1 2 3 4; do python3.10 -m nltk.downloader all && break || sleep 1; done  # retry as frequently fails with github downloading issues
pip install https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.4.2/auto_gptq-0.4.2+cu118-cp310-cp310-linux_x86_64.whl
pip install optimum==1.14.1

pip uninstall -y llama-cpp-python-cuda
pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.2.19+cu118-cp310-cp310-manylinux_2_31_x86_64.whl
pip uninstall -y llama-cpp-python
pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/cpu/llama_cpp_python-0.2.19+cpuavx2-cp310-cp310-manylinux_2_31_x86_64.whl

pip install autoawq==0.1.7
pip install sacrebleu==2.3.1 --upgrade

pip install attention_sinks --no-deps

pip install https://github.com/jllllll/exllama/releases/download/0.0.18/exllama-0.0.18+cu118-cp310-cp310-linux_x86_64.whl --no-cache-dir
pip install flash-attn==2.3.4 --no-build-isolation
playwright install --with-deps

# Uninstall duckdb and use own so can control thread count per db
pip uninstall -y pyduckdb duckdb && \
pip install https://s3.amazonaws.com/artifacts.h2o.ai/deps/h2ogpt/duckdb-0.8.2.dev4026%2Bgdcd8c1ffc5-cp310-cp310-linux_x86_64.whl --no-cache-dir --force-reinstall

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

# Install vllm client
pip install https://h2o-release.s3.amazonaws.com/h2ogpt/openvllm-0.28.1-py3-none-any.whl

# some fixes to bad packages
sp=`python3.10 -c 'import site; print(site.getsitepackages()[0])'`
sed -i 's/posthog\.capture/return\n            posthog.capture/' $sp/chromadb/telemetry/posthog.py
cd $sp
sed -i  's/with HiddenPrints():/if True:/g' langchain/utilities/serpapi.py

# vllm server
export VLLM_CACHE=/workspace/.vllm_cache
cd /h2ogpt_conda
python -m venv vllm_env --system-site-packages
# gputil is for rayWorker in vllm to run as non-root
/h2ogpt_conda/vllm_env/bin/python -m pip install https://github.com/vllm-project/vllm/releases/download/v0.2.2/vllm-0.2.2+cu118-cp310-cp310-manylinux1_x86_64.whl
/h2ogpt_conda/vllm_env/bin/python -m pip install ray pandas gputil==1.4.0
mkdir $VLLM_CACHE
chmod -R a+rwx /h2ogpt_conda

# Make sure old python location works in case using scripts from old documentation
mkdir -p /h2ogpt_conda/envs/vllm/bin
ln -s /h2ogpt_conda/vllm_env/bin/python3.10 /h2ogpt_conda/envs/vllm/bin/python3.10

# Track build info
cd /workspace && make build_info.txt
cp /workspace/build_info.txt /build_info.txt

mkdir -p /workspace/save
chmod -R a+rwx /workspace/save

# Cleanup
rm -rf /workspace/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
rm -rf /workspace/.cache/pip
