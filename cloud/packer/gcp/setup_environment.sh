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
  wget

MAX_GCC_VERSION=11
sudo DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install gcc-$MAX_GCC_VERSION g++-$MAX_GCC_VERSION

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-$MAX_GCC_VERSION 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-$MAX_GCC_VERSION 100
sudo update-alternatives --set gcc /usr/bin/gcc-$MAX_GCC_VERSION
sudo update-alternatives --set g++ /usr/bin/g++-$MAX_GCC_VERSION

sudo DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:deadsnakes/ppa
sudo DEBIAN_FRONTEND=noninteractive apt-get -y update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install python3.10 python3.10-dev python3.10-distutils
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

sudo echo "export PATH=$PATH:/home/ubuntu/.local/bin" >> ~/.bashrc
export PATH=$PATH:/home/ubuntu/.local/bin

git clone https://github.com/h2oai/h2ogpt.git
cd h2ogpt

python3.10 -m pip install virtualenv
virtualenv -p python3.10 venv
source venv/bin/activate

wget --quiet https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.0-515.43.04-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo DEBIAN_FRONTEND=noninteractive apt-get -y update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y install cuda

sudo echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/" >> ~/.bashrc
sudo echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
sudo echo "export PATH=$PATH:/usr/local/cuda/bin/" >> ~/.bashrc
