#!/bin/bash -e

cd h2ogpt
source venv/bin/activate

python3.10 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.txt
python3.10 -m pip install -r reqs_optional/requirements_optional_gpt4all.txt
python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt
python3.10 -m pip install -r reqs_optional/requirements_optional_langchain.urls.txt

sudo DEBIAN_FRONTEND=noninteractive apt-get install -y libmagic-dev poppler-utils tesseract-ocr libtesseract-dev libreoffice

python3.10 -m nltk.downloader all

python3.10 -m pip install https://s3.amazonaws.com/artifacts.h2o.ai/deps/h2ogpt/auto_gptq-0.3.0-cp310-cp310-linux_x86_64.whl --use-deprecated=legacy-resolver
python3.10 -m pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.1.73+cu117-cp310-cp310-linux_x86_64.whl
python3.10 -m pip install https://github.com/jllllll/exllama/releases/download/0.0.8/exllama-0.0.8+cu118-cp310-cp310-linux_x86_64.whl --no-cache-dir

sp=$(python3.10 -c 'import site; print(site.getsitepackages()[0])')
sed -i 's/posthog\.capture/return\n            posthog.capture/' "$sp"/chromadb/telemetry/posthog.py
sed -i 's/# n_gpu_layers=20/n_gpu_layers=20/g' .env_gpt4all

sudo echo "export TRANSFORMERS_CACHE=~/.cache/huggingface/hub/" >> ~/.bashrc
source ~/.bashrc
wget --quiet https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin

cd /etc/systemd/system
sudo chown -R ubuntu:ubuntu .

printf """
[Unit]
Description=h2oGPT Service
After=network.target
[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/h2ogpt
ExecStart=/home/ubuntu/h2ogpt/venv/bin/python3.10 generate.py --base_model='llama' --prompt_type=llama2
Restart=always
[Install]
WantedBy=multi-user.target
""" >> h2ogpt.service

sudo systemctl daemon-reload
sudo systemctl enable h2ogpt.service
sudo systemctl start h2ogpt.service
