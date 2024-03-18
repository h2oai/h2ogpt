#!/bin/bash
set -o pipefail
set -ex

apt-get clean all 2>&1
apt-get update 2>&1
apt-get -y full-upgrade 2>&1
apt-get -y dist-upgrade 2>&1
apt-get -y autoremove 2>&1
apt-get clean all 2>&1

# Check if the h2ogpt directory already exists
if [ -d "h2ogpt" ]; then
    echo "h2ogpt directory exists. Updating the repository."
    cd h2ogpt
    git stash 2>&1
    git pull 2>&1
else
    echo "h2ogpt directory does not exist. Cloning the repository."
    git clone https://github.com/h2oai/h2ogpt.git
    cd h2ogpt
fi

if ! command -v conda &> /dev/null; then
    echo "Conda not found, installing Miniconda."
    wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
    bash ./Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -u
    source ~/miniconda3/bin/activate
    conda init bash
    conda deactivate
else
    echo "Conda is already installed."
fi

conda remove -n h2ogpt --all -y
conda update conda -y
conda create -n h2ogpt -y
conda activate h2ogpt
conda install python=3.10 -c conda-forge -y

export CUDA_HOME=/usr/local/cuda-12.1
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
export LLAMA_CUBLAS=1
export CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all"
export FORCE_CMAKE=1

set +x
GPLOK=1 bash docs/linux_install.sh

echo -e "\n\n\n\t\tFINISHED - Right\n\n\n";
