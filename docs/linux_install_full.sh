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
else
    echo "Conda is already installed."
fi

# if there no session created or instanced, it will be
source ~/miniconda3/bin/activate
conda init bash
# it does not matter where it is, it will be utterly deactivate.
while [ $CONDA_SHLVL -gt 0 ]; do conda deactivate; done
# Now we can stay on base conda session

echo "Installing fresh h2oGPT env."
if conda env list | grep -w 'h2ogpt'; then
    conda remove -n h2ogpt --all -y
else
    echo "h2ogpt environment does not exist."
fi
conda update conda -y
conda create -n h2ogpt -y
conda activate h2ogpt
conda install python=3.10 -c conda-forge -y

if ! command -v nvcc &> /dev/null; then
    echo -e "\n\n\tThere is no NVIDIA CUDA Compiler Driver NVCC installed\n\n"
    exit 1
fi
nvccVersion=$(nvcc --version | grep -i release | sed -e 's/^.*release *\(\([[:digit:]]\|\.\)\+\).*$/\1/i')
if [ ! "$nvccVersion" == "12.1" ] && [ ! "$nvccVersion" == "11.8" ]; then
    echo -e "\n\n\tWARNING - There can be problems if you do not have installed NVIDIA CUDA Compiler Driver NVCC like 12.1 or 11.8"
    echo -e "\tYour current version is: ${nvccVersion}"
    echo -en "\t"; read -r -p "Are you sure to continue [y|Y]: " response < /dev/tty
    if [ ! "${response^^}" == "Y" ]; then
        echo -e "\n\n\tExit without installing\n\n"
        exit 1
    fi
fi

export CUDA_HOME=/usr/local/cuda-12.1
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
export GGML_CUDA=1
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=all"
export FORCE_CMAKE=1
# Overwriting in case of CUDA 11.8
if [ "${nvccVersion}" == "11.8" ]; then
	export CUDA_HOME=/usr/local/cuda-11.8
	export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu118 https://huggingface.github.io/autogptq-index/whl/cu118"
fi

# get patches
curl -O  https://h2o-release.s3.amazonaws.com/h2ogpt/run_patches.sh
curl -O https://h2o-release.s3.amazonaws.com/h2ogpt/trans.patch
curl -O https://h2o-release.s3.amazonaws.com/h2ogpt/xtt.patch
curl -O https://h2o-release.s3.amazonaws.com/h2ogpt/trans2.patch
curl -O https://h2o-release.s3.amazonaws.com/h2ogpt/google.patch
mkdir -p docs
alias cp='cp'
cp run_patches.sh trans.patch xtt.patch trans2.patch google.patch docs/

echo "Installing fresh h2oGPT"
set +x
export GPLOK=1
curl -fsSL https://h2o-release.s3.amazonaws.com/h2ogpt/linux_install.sh | bash


echo -e "\n\n\n\t\t h2oGPT installation FINISHED\n\n\n";
