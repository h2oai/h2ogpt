## h2oGPT Installation Help

The following sections describe how to get a working Python environment on a Linux system.

### Install for A100+

E.g. for Ubuntu 20.04, install driver if you haven't already done so:

```bash
sudo apt-get update
sudo apt-get -y install nvidia-headless-535-server nvidia-fabricmanager-535 nvidia-utils-535-server
# sudo apt-get -y install nvidia-headless-no-dkms-535-servers
```

Note that if you run the preceding commands, you don't need to use the NVIDIA developer downloads in the following sections.

### Install CUDA Toolkit

If happy with above drivers, then just get run local file for [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local):
```bash
wget wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```
only choose to install toolkit and do not replace existing `/usr/local/cuda` link if you already have one.

If instead, you want full deb CUDA [install cuda coolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local).  Pick deb local, e.g. for Ubuntu:
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
```

Then reboot the machine, to get everything sync'ed up on restart.
```bash
sudo reboot
```

### Compile bitsandbytes

For fast 4-bit and 8-bit training, you need to use [bitsandbytes](https://github.com/TimDettmers/bitsandbytes/tree/main#readme). Note that [compiling bitsandbytes](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) is only required if you have a different CUDA version from the ones built into the [bitsandbytes PyPI package](https://pypi.org/project/bitsandbytes/),
which includes CUDA 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 12.0, and 12.1. In the following example, bitsandbytes is compiled for CUDA 12.1:
```bash
git clone http://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes
git checkout 7c651012fce87881bb4e194a26af25790cadea4f
CUDA_VERSION=121 make cuda12x
CUDA_VERSION=121 python setup.py install
cd ..
```

### Install NVIDIA GPU Manager on systems with multiple A100 or H100 GPUs

To install NVIDIA GPU Manager, run the following:

```bash
sudo apt-key del 7fa2af80
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y datacenter-gpu-manager
# if use 535 drivers, then use 535 below
sudo apt-get install -y libnvidia-nscq-535
sudo systemctl --now enable nvidia-dcgm
dcgmi discovery -l
```
For more information, see the official [GPU Manager user guide](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/getting-started.html).

### Install and run NVIDIA Fabric Manager on systems with multiple A100 or H100 GPUs

To install the CUDA drivers for NVIDIA Fabric Manager, run the following:

```bash
sudo apt-get install -y cuda-drivers-fabricmanager
```

Once you've installed Fabric Manager and rebooted your system, run the following to start the NVIDIA Fabric Manager service:

```bash
sudo systemctl --now enable nvidia-dcgm
dcgmi discovery -l
sudo systemctl start nvidia-fabricmanager
sudo systemctl status nvidia-fabricmanager
```

For more information, see the official [Fabric Manager user guide](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html).

### Optional: Use TensorBoard to inspect training

You can use [TensorBoard](https://www.tensorflow.org/tensorboard/get_started) to inspect the training process. To launch TensorBoard and instruct it to read event files from the `runs/` directory, use the following command:

```bash
tensorboard --logdir=runs/
```

For more information, see [TensorBoard usage](https://github.com/tensorflow/tensorboard/blob/master/README.md#usage).

### Flash Attention

**Update:** Flash attention specifics are no longer needed. For more information, see https://github.com/h2oai/h2ogpt/issues/128.

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
