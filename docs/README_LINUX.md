# Linux

These instructions are for Ubuntu x86_64 (other linux would be similar with different command instead of apt-get).

## Install:

* First one needs a Python 3.10 environment.  We recommend using Miniconda.

  Download [MiniConda for Linux](https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh).  After downloading, run:
  ```bash
  bash ./Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
  # follow license agreement and add to bash if required
  ```
  Enter new shell and should also see `(base)` in prompt.  Then, create new env:
  ```bash
  conda create -n h2ogpt -y
  conda activate h2ogpt
  mamba install python=3.10 -c conda-forge -y
  ```
  You should see `(h2ogpt)` in shell prompt.
  
  Alternatively, on newer Ubuntu systems you can get Python 3.10 environment setup by doing:
  ```bash
  sudo apt-get install -y build-essential gcc python3.10-dev
  virtualenv -p python3 h2ogpt
  source h2ogpt/bin/activate
  ```
  
* Test your python:
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
  On some systems, `pip` still refers back to the system one, then one can use `python -m pip` or `pip3` instead of `pip` or try `python3` instead of `python`.

* For GPU: Install CUDA ToolKit with ability to compile using nvcc for some packages like llama-cpp-python, AutoGPTQ, exllama, and flash attention:
  ```bash
  conda install cudatoolkit-dev -c conda-forge -y
  export CUDA_HOME=$CONDA_PREFIX 
  ```
  which gives CUDA 11.7, or if you prefer follow [CUDA Toolkit](INSTALL.md#installing-cuda-toolkit), then do:
  ```bash
  export CUDA_HOME=/usr/local/cuda-11.7
  ```
  If you do not plan to use one of those packages, you can just use the non-dev version:
  ```bash
  conda install cudatoolkit=11.7 -c conda-forge -y
  export CUDA_HOME=$CONDA_PREFIX 
  ```
  
* Install dependencies:
    ```bash
    git clone https://github.com/h2oai/h2ogpt.git
    cd h2ogpt
    # fix any bad env
    pip uninstall -y pandoc pypandoc pypandoc-binary
    # broad support, but no training-time or data creation dependencies
    
    # CPU only:
    pip install -r requirements.txt --extra-index https://download.pytorch.org/whl/cpu
    
    # GPU only:
    pip install -r requirements.txt --extra-index https://download.pytorch.org/whl/cu117
    ```
* Install document question-answer dependencies:
    ```bash
    # Required for Doc Q/A: LangChain:
    pip install -r reqs_optional/requirements_optional_langchain.txt
    # Required for CPU: LLaMa/GPT4All:
    pip install -r reqs_optional/requirements_optional_gpt4all.txt
    # Optional: PyMuPDF/ArXiv:
    pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt
    # Optional: Selenium/PlayWright:
    pip install -r reqs_optional/requirements_optional_langchain.urls.txt
    # Optional: support docx, pptx, ArXiv, etc. required by some python packages
    sudo apt-get install -y libmagic-dev poppler-utils tesseract-ocr libtesseract-dev libreoffice
    # Optional: for supporting unstructured package
    python -m nltk.downloader all
* GPU Optional: For AutoGPTQ support on x86_64 linux
    ```bash
    pip uninstall -y auto-gptq ; GITHUB_ACTIONS=true pip install auto-gptq --no-cache-dir
    ```
   We recommend to install like the above in order to avoid warnings and inefficient memory usage. If one has trouble installing AutoGPTQ, can try:
   ```bash
   pip install https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.3.0/auto_gptq-0.3.0+cu117-cp310-cp310-linux_x86_64.whl
   ```
    However, if one sees `CUDA extension not installed` in output after loading model, one needs to compile it, else will use double memory and be slower on GPU.
    See [AutoGPTQ](README_GPU.md#autogptq) about running AutoGPT models.
* GPU Optional: For exllama support on x86_64 linux
    ```bash
    pip uninstall -y exllama ; pip install https://github.com/jllllll/exllama/releases/download/0.0.8/exllama-0.0.8+cu118-cp310-cp310-linux_x86_64.whl --no-cache-dir
    ```
    See [exllama](README_GPU.md#exllama) about running exllama models.

* To avoid unauthorized telemetry, which document options still do not disable, run:
    ```bash
    sp=`python -c 'import site; print(site.getsitepackages()[0])'`
    sed -i 's/posthog\.capture/return\n            posthog.capture/' $sp/chromadb/telemetry/posthog.py
    ```
* GPU Optional: Compile llama-cpp-python with CUDA support:
  ```bash
  pip uninstall -y llama-cpp-python
  export LLAMA_CUBLAS=1
  export CMAKE_ARGS=-DLLAMA_CUBLAS=on
  export FORCE_CMAKE=1
  CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.68 --no-cache-dir --verbose
  ```
   and uncomment `# n_gpu_layers=20` in `.env_gpt4all`.  One can try also `40` instead of `20`.  If one sees `/usr/bin/nvcc` mentioned in errors, that file needs to be removed as would likely conflict with version installed for conda.  
   Note that once `llama-cpp-python` is compiled to support CUDA, it no longer works for CPU mode,
   so one would have to reinstall it without the above options to recovers CPU mode or have a separate h2oGPT env for CPU mode.

---

## Run

* Check that can see CUDA from Torch:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```
    should print True.

* Place all documents in `user_path` or upload in UI ([Help with UI](README_ui.md)).

  UI using GPU with at least 24GB with streaming:
  ```bash
  python generate.py --base_model=h2oai/h2ogpt-oasst1-512-12b --load_8bit=True  --score_model=None --langchain_mode='UserData' --user_path=user_path
  ```
  UI using LLaMa.cpp model:
  ```bash
  wget https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GGML/resolve/main/WizardLM-7B-uncensored.ggmlv3.q8_0.bin
  python generate.py --base_model='llama' --prompt_type=wizard2 --score_model=None --langchain_mode='UserData' --user_path=user_path
  ```
  which works on CPU or GPU (assuming llama cpp python package compiled against CUDA or Metal).

  If using OpenAI for the LLM is ok, but you want documents to be parsed and embedded locally, then do:
  ```bash
  python generate.py  --inference_server=openai_chat --base_model=gpt-3.5-turbo --score_model=None
  ```
  and perhaps you want better image caption performance and focus local GPU on that, then do:
  ```bash
  python generate.py  --inference_server=openai_chat --base_model=gpt-3.5-turbo --score_model=None --captions_model=Salesforce/blip2-flan-t5-xl
  ```
  
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
