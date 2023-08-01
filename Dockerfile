# devel needed for bitsandbytes requirement of libcudart.so, otherwise runtime sufficient
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    software-properties-common \
    pandoc \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.10 python3-dev libpython3.10-dev

WORKDIR /workspace

COPY requirements.txt requirements.txt
COPY reqs_optional reqs_optional

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3.10 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
RUN python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.txt
RUN python3.10 -m pip install -r reqs_optional/requirements_optional_gpt4all.txt
RUN python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt
RUN python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.urls.txt

RUN apt-get install -y libmagic-dev poppler-utils tesseract-ocr libtesseract-dev libreoffice

RUN python3.10 -m nltk.downloader all

RUN export CUDA_HOME=/usr/local/cuda-11.7/ && GITHUB_ACTIONS=true python3.10 -m pip install auto-gptq --no-cache-dir

RUN python3.10 -m pip install https://github.com/jllllll/exllama/releases/download/0.0.8/exllama-0.0.8+cu118-cp310-cp310-linux_x86_64.whl --no-cache-dir

RUN sp=`python3.10 -c 'import site; print(site.getsitepackages()[0])'` && sed -i 's/posthog\.capture/return\n            posthog.capture/' $sp/chromadb/telemetry/posthog.py

RUN python3.10 -m pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.1.73+cu117-cp310-cp310-linux_x86_64.whl

RUN sed -i 's/# n_gpu_layers=20/n_gpu_layers=20/g' .env_gpt4all

EXPOSE 8888
EXPOSE 7860

ENV TRANSFORMERS_CACHE=/workspace/.cache

COPY . .
COPY build_info.txt /build_info.txt

ENTRYPOINT ["python3.10"]
