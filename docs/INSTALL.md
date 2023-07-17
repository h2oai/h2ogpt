## h2oGPT Installation Help

Follow these instructions to get a working Python environment on a Linux system.

### Install Python environment

Download Miniconda, for [Linux](https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh) or MACOS [Miniconda](https://docs.conda.io/en/latest/miniconda.html#macos-installers) or Windows [Miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe).  Then, install conda and setup environment:
```bash
bash ./Miniconda3-py310_23.1.0-1-Linux-x86_64.sh  # for linux x86-64
# follow license agreement and add to bash if required
```
Enter new shell and should also see `(base)` in prompt.  Then, create new env:
```bash
conda create -n h2ogpt -y
conda activate h2ogpt
conda install -y mamba -c conda-forge  # for speed
mamba install python=3.10 -c conda-forge -y
conda update -n base -c defaults conda -y
```
You should see `(h2ogpt)` in shell prompt.  Test your python:
```bash
python --version
```
should say 3.10.xx and:
```bash
python -c "import os, sys ; print('hello world')"
```
should print `hello world`.  Then clone:
```bash
git clone https://github.com/h2oai/h2ogpt.git
cd h2ogpt
```
Then go back to [README](../README.md) for package installation and use of `generate.py`.

### Installing CUDA Toolkit

E.g. CUDA 12.1 [install cuda coolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)

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
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64/" >> ~/.bashrc
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
echo "export PATH=\$PATH:/usr/local/cuda/bin/" >> ~/.bashrc
source ~/.bashrc
conda activate h2ogpt
```

Then reboot the machine, to get everything sync'ed up on restart.
```bash
sudo reboot
```

### Compile bitsandbytes

For fast 4-bit and 8-bit training, one needs bitsandbytes.  [Compiling bitsandbytes](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) is only required if you have different CUDA than built into bitsandbytes pypi package,
which includes CUDA 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 12.0, 12.1.  Here we compile for 12.1 as example.
```bash
git clone http://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes
git checkout 7c651012fce87881bb4e194a26af25790cadea4f
CUDA_VERSION=121 make cuda12x
CUDA_VERSION=121 python setup.py install
cd ..
```

### Install nvidia GPU manager if have multiple A100/H100s.
```bash
sudo apt-key del 7fa2af80
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y datacenter-gpu-manager
sudo apt-get install -y libnvidia-nscq-530
sudo systemctl --now enable nvidia-dcgm
dcgmi discovery -l
```
See [GPU Manager](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/getting-started.html)

### Install and run Fabric Manager if have multiple A100/100s

```bash
sudo apt-get install cuda-drivers-fabricmanager
sudo systemctl start nvidia-fabricmanager
sudo systemctl status nvidia-fabricmanager
```
See [Fabric Manager](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html)

Once have installed and reboot system, just do:

```bash
sudo systemctl --now enable nvidia-dcgm
dcgmi discovery -l
sudo systemctl start nvidia-fabricmanager
sudo systemctl status nvidia-fabricmanager
```

### Tensorboard (optional) to inspect training

```bash
tensorboard --logdir=runs/
```

### Flash Attention

Update: this is not needed anymore, see https://github.com/h2oai/h2ogpt/issues/128

To use flash attention with LLaMa, need cuda 11.7 so flash attention module compiles against torch.

E.g. for Ubuntu, one goes to [cuda toolkit](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local), then:
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo bash ./cuda_11.7.0_515.43.04_linux.run
```
Then No for symlink change, say continue (not abort), accept license, keep only toolkit selected, select install.

If cuda 11.7 is not your base installation, then when doing pip install -r requirements.txt do instead:
```bash
CUDA_HOME=/usr/local/cuda-11.8 pip install -r reqs_optional/requirements_optional_flashattention.txt
```
