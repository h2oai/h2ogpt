### Setup

Follow these instructions to get a working Python environment on a Linux system.

#### Install Python environment

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
bash ./Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
# follow license agreement and add to bash if required
source ~/.bashrc
conda create -n h2ollm -y
conda activate h2ollm
conda install mamba -n base -c conda-forge
conda install python=3.10 -y
conda update -n base -c defaults conda
```

#### Install Python packages

```bash
git clone https://github.com/h2oai/h2o-llm.git
cd h2o-llm
pip install -r requirements.txt
```

#### Install CUDA 12.1 [install cuda coolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)

E.g. for Ubuntu 20.04, select Ubuntu, Version 20.04, Installer Type "deb (local)", and you should get the following commands:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

Then set the system up to use the freshly installed CUDA location:
```bash
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/" >> ~/.bashrc
echo "CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
echo "export PATH=$PATH:/usr/local/cuda/bin/" >> ~/.bashrc
echo "conda activate h2ollm" >> ~/.bashrc
source ~/.bashrc
conda activate h2ollm
```

Then reboot the machine, to get everything sync'ed up on restart.
```bash
sudo reboot
```

#### Compile bitsandbytes for fast 8-bit training [howto src](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md)

```bash
git clone http://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes
git checkout 7c651012fce87881bb4e194a26af25790cadea4f
CUDA_VERSION=121 make cuda12x
CUDA_VERSION=121 python setup.py install
cd ..
```

Now you're ready to go back to data prep and fine-tuning!