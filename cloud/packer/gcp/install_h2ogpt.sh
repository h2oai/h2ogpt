#!/bin/bash -e

sleep 180

sudo DEBIAN_FRONTEND=noninteractive apt-get -y update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
  git \
  software-properties-common \
  pandoc \
  curl \
  apt-utils \
  make \
  build-essential \
  wget

MAX_GCC_VERSION=11
sudo DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install gcc-$MAX_GCC_VERSION g++-$MAX_GCC_VERSION

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-$MAX_GCC_VERSION 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-$MAX_GCC_VERSION 100
sudo update-alternatives --set gcc /usr/bin/gcc-$MAX_GCC_VERSION
sudo update-alternatives --set g++ /usr/bin/g++-$MAX_GCC_VERSION

export CONDA_DIR=/opt/conda
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -O ~/miniconda.sh
sudo bash ~/miniconda.sh -b -p /opt/conda
export PATH=$CONDA_DIR/bin:$PATH
sudo chown -R ubuntu:ubuntu /opt/conda

conda update -n base -c defaults conda
conda install python=3.10 -c conda-forge -y
conda install cudatoolkit-dev -c conda-forge -y
export CUDA_HOME=$CONDA_PREFIX 

git clone https://github.com/h2oai/h2ogpt.git
cd h2ogpt || exit

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r reqs_optional/requirements_optional_langchain.txt
pip install -r reqs_optional/requirements_optional_gpt4all.txt
pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt
pip install -r reqs_optional/requirements_optional_langchain.urls.txt

sudo DEBIAN_FRONTEND=noninteractive apt-get install -y libmagic-dev poppler-utils tesseract-ocr libtesseract-dev libreoffice

python -m nltk.downloader all

pip install https://s3.amazonaws.com/artifacts.h2o.ai/deps/h2ogpt/auto_gptq-0.3.0-cp310-cp310-linux_x86_64.whl --use-deprecated=legacy-resolver
pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.1.73+cu117-cp310-cp310-linux_x86_64.whl
pip install https://github.com/jllllll/exllama/releases/download/0.0.8/exllama-0.0.8+cu118-cp310-cp310-linux_x86_64.whl --no-cache-dir

sp=$(python3.10 -c 'import site; print(site.getsitepackages()[0])')
sed -i 's/posthog\.capture/return\n            posthog.capture/' "$sp"/chromadb/telemetry/posthog.py
sed -i 's/# n_gpu_layers=20/n_gpu_layers=20/g' .env_gpt4all

export TRANSFORMERS_CACHE=~/.cache/huggingface/hub/
