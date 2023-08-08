#!/bin/bash

# install instructions
# to install follow these steps:
# wget https://raw.githubusercontent.com/Royce-Geospatial-Consultants/h2ogpt_rg/main/royce_install_script.sh
# chmod +x royce_install_script.sh
# less install.sh # this will allow you to check the script before running it to make sure it is not compromised
# sudo ./royce_install_script.sh # this will run the script in the codespace.

# Update and install git
sudo apt-get update -y
sudo apt-get install git -y

# Print git version
git --version

# Clone repository
git clone https://github.com/Royce-Geospatial-Consultants/h2ogpt_rg.git
cd h2ogpt_rg

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
echo "yes" | ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Verify the installation by checking if 'conda' command is available
source $HOME/miniconda3/etc/profile.d/conda.sh
if conda --version >/dev/null 2>&1; then
  echo "Miniconda installed successfully."
else
  echo "Miniconda installation failed. Please manually install or consult the documentation."
  exit 1
fi


# Create conda environment
conda create --name h2ogpt_rg python=3.10 -y
conda activate h2ogpt_rg

# Print Python version
python --version

# Print a hello message
python -c "import os, sys ; print('hello world')"

# Install other dependencies
conda install cudatoolkit-dev -c conda-forge -y
export CUDA_HOME=$CONDA_PREFIX
pip uninstall -y pandoc pypandoc pypandoc-binary
pip install -r requirements.txt --extra-index https://download.pytorch.org/whl/cu117

# Additional installation
pip install -r reqs_optional/requirements_optional_langchain.txt
pip install -r reqs_optional/requirements_optional_gpt4all.txt
pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt
pip install -r reqs_optional/requirements_optional_langchain.urls.txt
sudo apt-get install -y libmagic-dev poppler-utils tesseract-ocr libtesseract-dev libreoffice
python -m nltk.downloader all

# More pip installations
pip uninstall -y auto-gptq
pip install https://s3.amazonaws.com/artifacts.h2o.ai/deps/h2ogpt/auto_gptq-0.3.0-cp310-cp310-linux_x86_64.whl --use-deprecated=legacy-resolver
pip uninstall -y exllama
pip install https://github.com/jllllll/exllama/releases/download/0.0.8/exllama-0.0.8+cu118-cp310-cp310-linux_x86_64.whl --no-cache-dir

# Modify Python package
sp=`python -c 'import site; print(site.getsitepackages()[0])'`
sed -i 's/posthog\.capture/return\n            posthog.capture/' $sp/chromadb/telemetry/posthog.py

# GPU handling and further pip installations
pip uninstall -y llama-cpp-python
pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.1.73+cu117-cp310-cp310-linux_x86_64.whl

# GPU handling requires more detailed instructions and is commented for now
# Please see the original text for manual handling

# Test CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Print instructions for the first session
echo "UI using GPU with at least 24GB with streaming:"
echo "python generate.py --base_model=h2oai/h2ogpt-oasst1-512-12b --load_8bit=True --score_model=None --langchain_mode='UserData' --user_path=user_path"
