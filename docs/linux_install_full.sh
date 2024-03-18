#!/bin/bash
set -o pipefail
set -ex

echo -e "\n\n\n\t\tSTART\n\n\n";

# ensure not in h2ogpt repo folder
cd $HOME

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
    source ~/miniconda3/bin/activate
    conda init bash
    conda deactivate
fi

if [ "$CONDA_DEFAULT_ENV" = "h2ogpt" ]; then
    echo "Deactivating the h2ogpt Conda environment."
    conda deactivate
else
    echo "The h2ogpt Conda environment is not currently activated."
fi

echo "Installing fresh h2oGPT env."
if conda env list | grep -q 'h2ogpt'; then
    conda remove -n h2ogpt --all -y
else
    echo "h2ogpt environment does not exist."
fi
conda update conda -y
conda create -n h2ogpt -y
conda activate h2ogpt
conda install python=3.10 -c conda-forge -y

export CUDA_HOME=/usr/local/cuda-12.1
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
export LLAMA_CUBLAS=1
export CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all"
export FORCE_CMAKE=1

echo "Installing fresh h2oGPT"
set +x
export GPLOK=1
curl -fsSL https://h2o-release.s3.amazonaws.com/h2ogpt/linux_install.sh | bash

echo -e "\n\n\n\t\t h2oGPT installation FINISHED\n\n\n";
