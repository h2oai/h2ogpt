#!/bin/bash -e

sudo systemctl daemon-reload
sudo systemctl enable h2ogpt_nginx.service
sudo systemctl enable vllm.service
sudo systemctl enable h2ogpt.service

cd "$HOME"
# sudo rm -rf "$HOME"/.cache/huggingface/hub/
sudo DEBIAN_FRONTEND=noninteractive apt-get -y autoremove
sudo DEBIAN_FRONTEND=noninteractive apt-get -y clean
