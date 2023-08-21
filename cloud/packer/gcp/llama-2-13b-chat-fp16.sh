#!/bin/bash -e

sp=$(/home/ubuntu/h2ogpt/venv/bin/python3.10 -c 'import site; print(site.getsitepackages()[0])')
cd $sp
rm -rf openai_vllm*
cp -a openai openai_vllm
cp -a openai-0.27.8.dist-info openai_vllm-0.27.8.dist-info
find openai_vllm -name '*.py' | xargs sed -i 's/from openai /from openai_vllm /g'
find openai_vllm -name '*.py' | xargs sed -i 's/openai\./openai_vllm./g'
find openai_vllm -name '*.py' | xargs sed -i 's/from openai\./from openai_vllm./g'
find openai_vllm -name '*.py' | xargs sed -i 's/import openai/import openai_vllm/g'

cd /home/ubuntu
/home/ubuntu/.local/bin/virtualenv -p python3.10 vllm

CUDA_HOME=/usr/local/cuda-11.7 /home/ubuntu/vllm/bin/python3.10 -m pip install vllm ray pandas

export NCCL_IGNORE_DISABLED_P2P=1
export CUDA_VISIBLE_DEVICES=0

sudo echo "export NCCL_IGNORE_DISABLED_P2P=1" >> ~/.bashrc
sudo echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc

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
ExecStart=/home/ubuntu/vllm/bin/python3.10 -m vllm.entrypoints.openai.api_server --port=5000 --host=0.0.0.0 --model TheBloke/Llama-2-13B-Chat-fp16 --tokenizer=hf-internal-testing/llama-tokenizer --tensor-parallel-size=4 --seed 1234
Restart=always
[Install]
WantedBy=multi-user.target
""" >> h2ogpt.service

sudo systemctl daemon-reload
sudo systemctl enable h2ogpt.service
sudo systemctl start h2ogpt.service
