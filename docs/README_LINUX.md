# Linux

This page describes how to manually install and run h2oGPT on Linux. Note that the following instructions are for Ubuntu x86_64. (The steps in the following subsection can be adapted to other Linux distributions by substituting `apt-get` with the appropriate package management command.)

- [Install](#install)
- [Run](#run)

## Quick Install

Ensure cuda toolkit is installed, e.g. for CUDA 12.1 on Ubuntu 22:
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run
```
One only needs to install the toolkit, and one does not have to overwrite the symlink.  Then run:
```bash
curl -fsSL https://h2o-release.s3.amazonaws.com/h2ogpt/linux_install_full.sh | bash
```
and enter the sudo password when required. Once install done, do:
```bash
conda activate h2ogpt
```

To avoid periodically entering the sudo password (default 5 minute timeout), then extend the sudo timeout by running:
```bash
sudo visudo
```
and adding:
```
Defaults        timestamp_timeout=60
```
after the `Defaults env_reset` line.  Then run:
```bash
sudo bash
exit
```
So allow your user session to run sudo for 60 minutes. Then the script will not ask for sudo password during its run.

## Install

* Set up a Python 3.10 environment. We recommend using [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/).

  Download Miniconda for Linux and install:
  ```bash
  wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
  bash ./Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -p $HOME/miniconda3

  # Manually adding Conda init to .bashrc
  echo '### Conda init ###' >> $HOME/.bashrc
  echo 'source $HOME/miniconda3/etc/profile.d/conda.sh' >> $HOME/.bashrc
  echo 'conda activate' >> $HOME/.bashrc
  source $HOME/.bashrc

  # install h2ogpt env

  # Run below if have existing h2ogpt env
  # conda remove -n h2ogpt --all -y

  conda update conda -y
  conda create -n h2ogpt -y
  conda activate h2ogpt
  conda install python=3.10 -c conda-forge -y
  ```
  You should see `(h2ogpt)` in the shell prompt.  If do not want conda in your `~/.bashrc`, then add to different shell script to `source` before starting h2oGPT.

* Check your python version with the following command:
  ```bash
  python --version
  python -c "import os, sys ; print('hello world')"
  ```
  The return should say 3.10.xx, and print `hello world`.

* Clone h2oGPT:
  ```bash
  git clone https://github.com/h2oai/h2ogpt.git
  cd h2ogpt
  ```
  On some systems, `pip` still refers back to the system one, then one can use `python -m pip` or `pip3` instead of `pip` or try `python3` instead of `python`.

* For GPU: Install CUDA ToolKit with ability to compile using nvcc for some packages like llama-cpp-python, AutoGPTQ, exllama, flash attention, TTS use of deepspeed, by going to [CUDA Toolkit](INSTALL.md#install-cuda-toolkit).  E.g. [CUDA 12.1 Toolkit](https://developer.nvidia.com/cuda-12-1-1-download-archive).  In order to avoid removing the original CUDA toolkit/driver you have, on NVIDIA's website, use the `runfile (local)` installer, and choose to not install driver or overwrite `/usr/local/cuda` link and just install the toolkit, and rely upon the `CUDA_HOME` env to point to the desired CUDA version.  E.g. for CUDA 12.1 do:
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run
```
* Then do:
  ```bash
  echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> $HOME/.bashrc
  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64' >> $HOME/.bashrc
  echo 'export PATH=$PATH:$CUDA_HOME/bin' >> $HOME/.bashrc
  ```
  If you do not want these in your `~/.bashrc`, then add to different shell script to `source` before starting h2oGPT (e.g. for TTS's use of deepspeed to work).
  
* Prepare to install dependencies for CUDA 12.1:
   ```bash
   export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121 https://huggingface.github.io/autogptq-index/whl/cu121"
   ```
  or for CUDA 11.8:
   ```bash
   export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu118 https://huggingface.github.io/autogptq-index/whl/cu118"
   ```
  For some packages, this requires changing cu118 in reqs_optional/requirements*.txt if built for cu118 specifically. 
  Choose cu121+ for A100/H100+.  Or for CPU set
   ```bash
   export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
   ```

* Choose llama_cpp_python ARGS for your system according to [llama_cpp_python backend documentation](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends), e.g. for CUDA:
   ```bash
   export LLAMA_CUBLAS=1
   export CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all"
   export FORCE_CMAKE=1
   ```
  Note for some reason things will fail with llama_cpp_python if don't add all cuda arches, and building with all those arches does take some time.
* Run (`bash docs/linux_install.sh`)[linux_install.sh] for full normal document Q/A installation.  To allow all (GPL too) packages, run:
    ```bash
    GPLOK=1 bash docs/linux_install.sh
    ```
One can pick and choose different optional things to install instead by commenting them out in the shell script, or edit the script if any issues.  See script for notes about installation.

---

## Run

For information on how to run h2oGPT offline, see [Offline](README_offline.md#tldr).

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

#### Issues

## Old Ubuntu 18

* If your Ubuntu etc. is very out of date (E.g. Ubuntu 18), you can run the below, but it might lead to system issues.  If you already have Ubuntu 20, 22, do **not** run these.
```bash
apt-get clean all
apt-get update
apt-get -y full-upgrade
apt-get -y dist-upgrade
apt-get -y autoremove
apt-get clean all
```

## undefined symbols

If see:
```text
  File "/home/jon/h2ogpt/src/gen.py", line 2289, in get_config
    model = AutoModel.from_config(
  File "/home/jon/miniconda3/envs/h2ogpt/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 434, in from_config
    model_class = _get_model_class(config, cls._model_mapping)
  File "/home/jon/miniconda3/envs/h2ogpt/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 381, in _get_model_class
    supported_models = model_mapping[type(config)]
  File "/home/jon/miniconda3/envs/h2ogpt/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 732, in __getitem__
    return self._load_attr_from_module(model_type, model_name)
  File "/home/jon/miniconda3/envs/h2ogpt/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 746, in _load_attr_from_module
    return getattribute_from_module(self._modules[module_name], attr)
  File "/home/jon/miniconda3/envs/h2ogpt/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py", line 690, in getattribute_from_module
    if hasattr(module, attr):
  File "/home/jon/miniconda3/envs/h2ogpt/lib/python3.10/site-packages/transformers/utils/import_utils.py", line 1380, in __getattr__
    module = self._get_module(self._class_to_module[name])
  File "/home/jon/miniconda3/envs/h2ogpt/lib/python3.10/site-packages/transformers/utils/import_utils.py", line 1392, in _get_module
    raise RuntimeError(
RuntimeError: Failed to import transformers.models.mistral.modeling_mistral because of the following error (look up to see its traceback):
/home/jon/miniconda3/envs/h2ogpt/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops5zeros4callEN3c108ArrayRefINS2_6SymIntEEENS2_8optionalINS2_10ScalarTypeEEENS6_INS2_6LayoutEEENS6_INS2_6DeviceEEENS6_IbEE
```

Ensure your `CUDA_HOME` env is set to the same as you installed h2oGPT with, e.g.
```bash
export CUDA_HOME=/usr/local/cuda-12.1

Then run in the `h2ogpt` conda env:
```bash
# https://github.com/h2oai/h2ogpt/issues/1483
pip uninstall flash_attn autoawq autoawq-kernels -y && pip install flash_attn autoawq autoawq-kernels --no-cache-dir
```

```
