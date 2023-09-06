# devel needed for bitsandbytes requirement of libcudart.so, otherwise runtime sufficient
FROM h2ogpt-base as stage0

ARG DEBIAN_FRONTEND=noninteractive

ENV PATH="/h2ogpt_conda/bin:${PATH}"
ARG PATH="/h2ogpt_conda/bin:${PATH}"

COPY . .

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh && \
    mkdir -p h2ogpt_conda && \
    bash ./Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -u -p /h2ogpt_conda && \
    conda install python=3.10 pygobject weasyprint -c conda-forge -y

RUN python3.10 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
RUN python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.txt --extra-index-url https://download.pytorch.org/whl/cu118
RUN python3.10 -m pip install -r reqs_optional/requirements_optional_gpt4all.txt --extra-index-url https://download.pytorch.org/whl/cu118
RUN python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt --extra-index-url https://download.pytorch.org/whl/cu118
RUN python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.urls.txt --extra-index-url https://download.pytorch.org/whl/cu118

RUN python3.10 -m pip install -r reqs_optional/requirements_optional_doctr.txt --extra-index-url https://download.pytorch.org/whl/cu118
# go back to older onnx so Tesseract OCR still works
RUN python3.10 -m pip install onnxruntime==1.15.0 onnxruntime-gpu==1.15.0 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    python3.10 -m pip uninstall -y weasyprint && \
    python3.10 -m pip install weasyprint

# Install prebuilt dependencies

RUN python3.10 -m nltk.downloader all
RUN python3.10 -m pip install https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.4.2/auto_gptq-0.4.2+cu118-cp310-cp310-linux_x86_64.whl
RUN python3.10 -m pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.1.73+cu118-cp310-cp310-linux_x86_64.whl
RUN python3.10 -m pip install https://github.com/jllllll/exllama/releases/download/0.0.13/exllama-0.0.13+cu118-cp310-cp310-linux_x86_64.whl --no-cache-dir
RUN playwright install --with-deps

# Install vllm
ENV VLLM_CACHE=/workspace/.vllm_cache

RUN cd /h2ogpt_conda && python -m venv vllm_env --system-site-packages
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
    cd /h2ogpt_conda && \
    python -m venv vllm_env --system-site-packages && \
    /h2ogpt_conda/vllm_env/bin/python -m pip install vllm ray pandas --extra-index-url https://download.pytorch.org/whl/cu118 && \
    mkdir ${VLLM_CACHE}

RUN chmod -R a+rwx /h2ogpt_conda

FROM h2ogpt-base as stage1

ENV VLLM_CACHE=/workspace/.vllm_cache
ENV PATH="/h2ogpt_conda/bin:${PATH}"
ARG PATH="/h2ogpt_conda/bin:${PATH}"

COPY . .
COPY --from=stage0 /h2ogpt_conda/ /h2ogpt_conda/

COPY build_info.txt* /build_info.txt
RUN touch /build_info.txt

USER h2ogpt

ENTRYPOINT ["python3.10"]
