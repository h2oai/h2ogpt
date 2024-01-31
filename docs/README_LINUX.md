# Linux

This page describes how to manually install and run h2oGPT on Linux. Note that the following instructions are for Ubuntu x86_64. (The steps in the following subsection can be adapted to other Linux distributions by substituting `apt-get` with the appropriate package management command.)

- [Install](#install)
- [Run](#run)

## Install

* Set up a Python 3.10 environment. We recommend using [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/).

  Download [Miniconda for Linux](https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh).  After downloading, run:
  ```bash
  bash ./Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
  # follow license agreement and add to bash if required
  ```
  Open a new shell and look for `(base)` in the prompt to confirm that Miniconda is properly installed, then create a new env:
  ```bash
  conda create -n h2ogpt -y
  conda activate h2ogpt
  conda install python=3.10 -c conda-forge -y
  ```
  You should see `(h2ogpt)` in the shell prompt.
  
  Alternatively, on newer Ubuntu systems, you can set up a Python 3.10 environment by doing the following:
  ```bash
  sudo apt-get update
  sudo apt-get install -y build-essential gcc python3.10-dev
  virtualenv -p python3 h2ogpt
  source h2ogpt/bin/activate
  ```
  
* Check your python version with the following command:
  ```bash
  python --version
  ```
  The return should say 3.10.xx, and:
  ```bash
  python -c "import os, sys ; print('hello world')"
  ```
  should print `hello world`.  Then clone:
  ```bash
  git clone https://github.com/h2oai/h2ogpt.git
  cd h2ogpt
  ```
  On some systems, `pip` still refers back to the system one, then one can use `python -m pip` or `pip3` instead of `pip` or try `python3` instead of `python`.

* For GPU: Install CUDA ToolKit with ability to compile using nvcc for some packages like llama-cpp-python, AutoGPTQ, exllama, flash attention, TTS use of deepspeed, by going to [CUDA Toolkit](INSTALL.md#install-cuda-toolkit).  E.g. [CUDA 12.1 Toolkit](https://developer.nvidia.com/cuda-12-1-1-download-archive).  In order to avoid removing the original CUDA toolkit/driver you have, on NVIDIA's website, use the `runfile (local)` installer, and choose to not install driver or overwrite `/usr/local/cuda` link and just install the toolkit, and rely upon the `CUDA_HOME` env to point to the desired CUDA version.  Then do:
  ```bash
  export CUDA_HOME=/usr/local/cuda-12.1
  ```

* Place the `CUDA_HOME` export into your `~/.bashrc` or before starting h2oGPT for TTS's use of deepspeed to work.
  
* Prepare to install dependencies:
   ```bash
   export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
   ```
  Choose cu118+ for A100/H100+.  Or for CPU set
   ```bash
   export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
   ```

* Run (`bash docs/linux_install.sh`)[linux_install.sh] for full normal document Q/A installation.  To allow all (GPL too) packages, run:
    ```bash
    GPLOK=1 bash docs/linux_install.sh
    ```
One can pick and choose different optional things to install instead by commenting them out in the shell script, or edit the script if any issues.  See script for notes about installation.

---

## Run

See the [FAQ](FAQ.md#adding-models) for many ways to run models.  The following are some other examples.

Note that models are stored in `/home/$USER/.cache/` for chroma, huggingface, selenium, torch, weaviate, etc. directories.

* Check that can see CUDA from Torch:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
    should print True.

* Place all documents in `user_path` or upload in UI ([Help with UI](README_ui.md)).

  UI using GPU with at least 24GB with streaming:
  ```bash
  python generate.py --base_model=h2oai/h2ogpt-4096-llama2-13b-chat --load_8bit=True  --score_model=None --langchain_mode='UserData' --user_path=user_path
  ```
  Same with a smaller model without quantization:
  ```bash
  python generate.py --base_model=h2oai/h2ogpt-4096-llama2-7b-chat --score_model=None --langchain_mode='UserData' --user_path=user_path
  ```
  UI using LLaMa.cpp LLaMa2 model:
  ```bash
  python generate.py --base_model='llama' --prompt_type=llama2 --score_model=None --langchain_mode='UserData' --user_path=user_path --model_path_llama=https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf?download=true --max_seq_len=4096
  ```
  which works on CPU or GPU (assuming llama cpp python package compiled against CUDA or Metal).

  If using OpenAI for the LLM is ok, but you want documents to be parsed and embedded locally, then do:
  ```bash
  OPENAI_API_KEY=<key> python generate.py  --inference_server=openai_chat --base_model=gpt-3.5-turbo --score_model=None
  ```
  where `<key>` should be replaced by your OpenAI key that probably starts with `sk-`.  OpenAI is **not** recommended for private document question-answer, but it can be a good reference for testing purposes or when privacy is not required.  
  Perhaps you want better image caption performance and focus local GPU on that, then do:
  ```bash
  OPENAI_API_KEY=<key> python generate.py  --inference_server=openai_chat --base_model=gpt-3.5-turbo --score_model=None --captions_model=Salesforce/blip2-flan-t5-xl
  ```
  For Azure OpenAI:
  ```bash
   OPENAI_API_KEY=<key> python generate.py --inference_server="openai_azure_chat:<deployment_name>:<base_url>:<api_version>" --base_model=gpt-3.5-turbo --h2ocolors=False --langchain_mode=UserData
   ```
  where the entry `<deployment_name>` is required for Azure, others are optional and can be filled with string `None` or have empty input between `:`.  Azure OpenAI is a bit safer for private access to Azure-based docs.
  
  Add `--share=True` to make gradio server visible via sharable URL.
 
  If you see an error about protobuf, try:
  ```bash
  pip install protobuf==3.20.0
  ```

See [CPU](README_CPU.md) and [GPU](README_GPU.md) for some other general aspects about using h2oGPT on CPU or GPU, such as which models to try.

#### Google Colab

* A Google Colab version of a 3B GPU model is at:

  [![](https://colab.research.google.com/assets/colab-badge.svg) h2oGPT GPU](https://colab.research.google.com/drive/143-KFHs2iCqXTQLI2pFCDiR69z0dR8iE?usp=sharing)

  A local copy of that GPU Google Colab is [h2oGPT_GPU.ipynb](h2oGPT_GPU.ipynb).

* A Google Colab version of a 7B LLaMa CPU model is at:

  [![](https://colab.research.google.com/assets/colab-badge.svg) h2oGPT CPU](https://colab.research.google.com/drive/13RiBdAFZ6xqDwDKfW6BG_-tXfXiqPNQe?usp=sharing)

  A local copy of that CPU Google Colab is [h2oGPT_CPU.ipynb](h2oGPT_CPU.ipynb).
