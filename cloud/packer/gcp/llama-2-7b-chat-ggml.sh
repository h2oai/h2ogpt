#!/bin/bash -e

wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin

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
ExecStart=/home/ubuntu/h2ogpt/venv/bin/python3.10 generate.py --base_model='llama' --prompt_type=llama2 --score_model=None --langchain_mode='UserData' --user_path=user_path
Restart=always
[Install]
WantedBy=multi-user.target
""" >> h2ogpt.service

sudo systemctl daemon-reload
sudo systemctl enable h2ogpt.service
sudo systemctl start h2ogpt.service
