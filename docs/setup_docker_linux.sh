# BUILD
sudo apt-get update
sudo apt install software-properties-common
sudo apt-get install build-essential

# DRIVER + toolkit
sudo apt-get update
sudo apt-get -y install nvidia-headless-535-server nvidia-fabricmanager-535 nvidia-utils-535-server

# wget wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
# sudo sh cuda_11.8.0_520.61.05_linux.run

wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64/" >> ~/.bashrc
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
echo "export PATH=\$PATH:/usr/local/cuda/bin/" >> ~/.bashrc
echo "sudo nvidia-smi -pm 1" >> ~/.bashrc

# reboot after driver installed if installed driver, else no need if just cuda toolkit added, then just logout and log back in or do: source ~/.bashrc

# DOCKER
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
apt-cache policy docker-ce
sudo apt install -y docker-ce
sudo systemctl status docker

sudo usermod -aG docker $USER

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit-base
sudo apt install -y nvidia-container-runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
docker pull gcr.io/vorvan/h2oai/h2ogpt-runtime:0.1.0

# no need to reboot

# if /  is too small, can move docker to new location, change /etc/docker/daemon.json to have:
#
#{
#    "runtimes": {
#        "nvidia": {
#            "args": [],
#            "path": "nvidia-container-runtime"
#        }
#    },
#    "data-root": "/ephemeral/docker-data"
#}
# then run:
# docker image prune -a
# sudo systemctl restart docker
