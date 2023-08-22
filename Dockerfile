# devel needed for bitsandbytes requirement of libcudart.so, otherwise runtime sufficient
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    software-properties-common \
    pandoc

ENV PATH="/h2ogpt_conda/bin:${PATH}"
ARG PATH="/h2ogpt_conda/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh && \
    mkdir -p h2ogpt_conda && \
    bash ./Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -u -p /h2ogpt_conda && \
    conda install python=3.10 -c conda-forge -y

WORKDIR /workspace

COPY requirements.txt requirements.txt
COPY reqs_optional reqs_optional

RUN python3.10 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
RUN python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.txt
RUN python3.10 -m pip install -r reqs_optional/requirements_optional_gpt4all.txt
RUN python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt
RUN python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.urls.txt

RUN apt-get install -y libmagic-dev poppler-utils tesseract-ocr libtesseract-dev libreoffice

RUN python3.10 -m nltk.downloader all

ENV CUDA_HOME=/usr/local/cuda-11.8

# Install prebuilt dependencies

RUN python3.10 -m pip install https://s3.amazonaws.com/artifacts.h2o.ai/deps/h2ogpt/auto_gptq-0.3.0-cp310-cp310-linux_x86_64.whl --use-deprecated=legacy-resolver
RUN python3.10 -m pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.1.73+cu118-cp310-cp310-linux_x86_64.whl
RUN python3.10 -m pip install https://github.com/jllllll/exllama/releases/download/0.0.8/exllama-0.0.8+cu118-cp310-cp310-linux_x86_64.whl --no-cache-dir

COPY . .

ENV VLLM_CACHE=/workspace/.vllm_cache

RUN sp=`python3.10 -c 'import site; print(site.getsitepackages()[0])'` && \
    sed -i 's/posthog\.capture/return\n            posthog.capture/' $sp/chromadb/telemetry/posthog.py && \
    cd $sp && \
    rm -rf openai_vllm* && \
    cp -a openai openai_vllm && \
    cp -a openai-0.27.8.dist-info openai_vllm-0.27.8.dist-info && \
    find openai_vllm -name '*.py' | xargs sed -i 's/from openai /from openai_vllm /g' && \
    find openai_vllm -name '*.py' | xargs sed -i 's/openai\./openai_vllm./g' && \
    find openai_vllm -name '*.py' | xargs sed -i 's/from openai\./from openai_vllm./g' && \
    find openai_vllm -name '*.py' | xargs sed -i 's/import openai/import openai_vllm/g' && \
    conda create -n vllm python=3.10 -y && \
    /h2ogpt_conda/envs/vllm/bin/python3.10 -m pip install vllm ray pandas && \
    mkdir ${VLLM_CACHE}

EXPOSE 8888
EXPOSE 7860
EXPOSE 5000

ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub/

COPY build_info.txt* /build_info.txt
RUN touch /build_info.txt

ARG user=h2ogpt
ARG group=h2ogpt
ARG uid=1000
ARG gid=1000

RUN groupadd -g ${gid} ${group} && useradd -u ${uid} -g ${group} -s /bin/bash ${user}
RUN chmod -R a+rwx /workspace
RUN chmod -R a+rwx /h2ogpt_conda

USER h2ogpt

ENTRYPOINT ["python3.10"]
