#!/bin/bash -e


cd /etc/nginx/conf.d
sudo chown -R ubuntu:ubuntu .
cd $HOME
printf """
server {
    listen 80;
    listen [::]:80;
    server_name <|_SUBST_PUBLIC_IP|>;  # Change this to your domain name

    location / {  # Change this if you'd like to server your Gradio app on a different path
        proxy_pass http://0.0.0.0:7860/; # Change this if your Gradio app will be running on a different port
        proxy_redirect off;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection \"upgrade\";
        proxy_set_header Host \$host;
    }
}
""" > temp.conf

printf """
ip=\$(dig +short myip.opendns.com @resolver1.opendns.com)
sed \"s/<|_SUBST_PUBLIC_IP|>;/\$ip;/g\" /home/ubuntu/temp.conf  > /etc/nginx/conf.d/h2ogpt.conf
sudo systemctl restart nginx.service
""" > run_nginx.sh

sudo chmod u+x run_nginx.sh

cd /etc/systemd/system
sudo chown -R ubuntu:ubuntu .
printf """
[Unit]
Description=h2oGPT Nginx Server
StartLimitIntervalSec=300
StartLimitBurst=5
After=network.target
[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu
ExecStart=bash /home/ubuntu/run_nginx.sh
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
""" > h2ogpt_nginx.service

sudo systemctl daemon-reload
sudo systemctl enable h2ogpt_nginx.service

cd $HOME
printf """
tps=\$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l | awk '{if (\$1 > 1) print int(\$1/2); else print 1}')
NCCL_IGNORE_DISABLED_P2P=1 CUDA_VISIBLE_DEVICES=\$(seq -s, 0 \$((\$(nvidia-smi -L | wc -l) > 1 ? \$(nvidia-smi -L | wc -l) / 2 - 1 : 0))) /home/ubuntu/vllm/bin/python3.10 -m vllm.entrypoints.openai.api_server \
    --port=5000 \
    --host=0.0.0.0 \
    --model h2oai/h2ogpt-4096-llama2-13b-chat \
    --tokenizer=hf-internal-testing/llama-tokenizer \
    --tensor-parallel-size=\$tps --seed 1234
""" > run_vllm.sh
sudo chmod u+x run_vllm.sh

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
ExecStart=bash /home/ubuntu/run_vllm.sh
Restart=always
[Install]
WantedBy=multi-user.target
""" > vllm.service

sudo systemctl daemon-reload
sudo systemctl enable vllm.service

cd $HOME/h2ogpt

printf """
CUDA_VISIBLE_DEVICES=\$(echo \$(seq -s, \$((\$(nvidia-smi -L | wc -l) / 2)) \$((\$(nvidia-smi -L | wc -l) - 1)))) /home/ubuntu/h2ogpt/venv/bin/python3.10 /home/ubuntu/h2ogpt/generate.py --inference_server="vllm:0.0.0.0:5000" --base_model=h2oai/h2ogpt-4096-llama2-13b-chat --langchain_mode=UserData
""" > run_h2ogpt.sh
sudo chmod u+x run_h2ogpt.sh

cd /etc/systemd/system

printf """
[Unit]
Description=h2oGPT Server
After=network.target
[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/h2ogpt
ExecStart=bash /home/ubuntu/h2ogpt/run_h2ogpt.sh
[Install]
WantedBy=multi-user.target
""" > h2ogpt.service

sudo systemctl daemon-reload
sudo systemctl enable h2ogpt.service

cd $HOME
sudo rm -rf $HOME/.cache/huggingface/hub/
sudo DEBIAN_FRONTEND=noninteractive apt-get -y autoremove
sudo DEBIAN_FRONTEND=noninteractive apt-get -y clean
sudo rm -rf *.deb
