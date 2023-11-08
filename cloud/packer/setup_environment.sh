#!/bin/bash -e

sudo DEBIAN_FRONTEND=noninteractive apt-get -y update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
  git \
  software-properties-common \
  pandoc \
  curl \
  apt-utils \
  make \
  build-essential \
  wget \
  gnupg2 \
  ca-certificates \
  lsb-release \
  ubuntu-keyring

curl https://nginx.org/keys/nginx_signing.key | gpg --dearmor | sudo tee /usr/share/keyrings/nginx-archive-keyring.gpg >/dev/null
gpg --dry-run --quiet --no-keyring --import --import-options import-show /usr/share/keyrings/nginx-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/nginx-archive-keyring.gpg] http://nginx.org/packages/ubuntu `lsb_release -cs` nginx" sudo tee /etc/apt/sources.list.d/nginx.list
echo -e "Package: *\nPin: origin nginx.org\nPin: release o=nginx\nPin-Priority: 900\n" sudo tee /etc/apt/preferences.d/99nginx

sudo DEBIAN_FRONTEND=noninteractive apt -y update
sudo DEBIAN_FRONTEND=noninteractive apt -y install nginx

MAX_GCC_VERSION=11
sudo DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install gcc-$MAX_GCC_VERSION g++-$MAX_GCC_VERSION

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-$MAX_GCC_VERSION 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-$MAX_GCC_VERSION 100
sudo update-alternatives --set gcc /usr/bin/gcc-$MAX_GCC_VERSION
sudo update-alternatives --set g++ /usr/bin/g++-$MAX_GCC_VERSION

wget --quiet https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget --quiet https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo DEBIAN_FRONTEND=noninteractive apt-get -y update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install cuda
sudo rm -rf "*.deb"

sudo echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64/" >> ~/.bashrc
sudo echo "export CUDA_HOME=/usr/local/cuda-11.8" >> ~/.bashrc
sudo echo "export PATH=$PATH:/h2ogpt_conda/bin:/usr/local/cuda-11.8/bin/" >> ~/.bashrc
