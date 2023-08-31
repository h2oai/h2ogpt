#!/bin/bash -e

cd h2ogpt
source venv/bin/activate
export PATH=$PATH:/home/ubuntu/.local/bin
python3.10 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.txt
python3.10 -m pip install -r reqs_optional/requirements_optional_gpt4all.txt
python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt
python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.urls.txt

sudo DEBIAN_FRONTEND=noninteractive apt-get install -y libmagic-dev poppler-utils tesseract-ocr libtesseract-dev libreoffice

python3.10 -m nltk.downloader all
python3.10 -m pip install https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.4.1/auto_gptq-0.4.1+cu118-cp310-cp310-linux_x86_64.whl
python3.10 -m pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.1.73+cu118-cp310-cp310-linux_x86_64.whl
python3.10 -m pip install https://github.com/jllllll/exllama/releases/download/0.0.8/exllama-0.0.8+cu118-cp310-cp310-linux_x86_64.whl --no-cache-dir

sp=$(python3.10 -c 'import site; print(site.getsitepackages()[0])')
sed -i 's/posthog\.capture/return\n            posthog.capture/' "$sp"/chromadb/telemetry/posthog.py
cd $sp
rm -rf openai_vllm*
cp -a openai openai_vllm
cp -a openai-0.27.8.dist-info openai_vllm-0.27.8.dist-info
find openai_vllm -name '*.py' | xargs sed -i 's/from openai /from openai_vllm /g'
find openai_vllm -name '*.py' | xargs sed -i 's/openai\./openai_vllm./g'
find openai_vllm -name '*.py' | xargs sed -i 's/from openai\./from openai_vllm./g'
find openai_vllm -name '*.py' | xargs sed -i 's/import openai/import openai_vllm/g'

sudo echo "export TRANSFORMERS_CACHE=~/.cache/huggingface/hub/" >> ~/.bashrc
source ~/.bashrc

deactivate
cd $HOME
sudo echo "export PATH=$PATH:/home/ubuntu/.local/bin" >> ~/.bashrc
export PATH=$PATH:/home/ubuntu/.local/bin 

virtualenv -p /usr/bin/python3.10 vllm
vllm/bin/python3.10 -m pip install vllm ray pandas
