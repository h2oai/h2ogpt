#!/bin/bash -e

source vllm/bin/activate

export NCCL_IGNORE_DISABLED_P2P=1
export CUDA_VISIBLE_DEVICES=$(echo $(seq -s, 0 $(($(nvidia-smi -L | wc -l) - 1))))

sudo echo "export NCCL_IGNORE_DISABLED_P2P=1" >> ~/.bashrc
sudo echo "export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> ~/.bashrc

cd /etc/systemd/system
sudo chown -R ubuntu:ubuntu .
printf """
[Unit]
Description=vLLM Server
After=network.target
[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
ExecStart=/home/ubuntu/vllm/bin/python3.10 -m vllm.entrypoints.openai.api_server --port=5000 --host=0.0.0.0 --model h2oai/h2ogpt-4096-llama2-13b-chat --tokenizer=hf-internal-testing/llama-tokenizer --tensor-parallel-size=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l) --seed 1234
Restart=always
[Install]
WantedBy=multi-user.target
""" >> vllm.service

sudo systemctl daemon-reload
sudo systemctl enable vllm.service
sudo systemctl start vllm.service

deactivate

cd $HOME/h2ogpt
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=$(echo $(seq -s, 0 $(($(nvidia-smi -L | wc -l) - 1))))
sudo echo "export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> ~/.bashrc

cd /etc/systemd/system

printf """
[Unit]
Description=h2oGPT Server
After=network.target
[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/h2ogpt
ExecStart=/home/ubuntu/h2ogpt/venv/bin/python3.10 /home/ubuntu/h2ogpt/generate.py --inference_server="vllm:0.0.0.0:5000" --base_model=h2oai/h2ogpt-4096-llama2-13b-chat --langchain_mode=UserData
Restart=always
[Install]
WantedBy=multi-user.target
""" >> h2ogpt.service

sudo systemctl daemon-reload
sudo systemctl enable h2ogpt.service
sudo systemctl start h2ogpt.service
