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
  conda install python=3.10 -c conda-forge -y
  ```
  You should see `(h2ogpt)` in shell prompt.
  
  Alternatively, on newer Ubuntu systems you can get Python 3.10 environment setup by doing:
  ```bash
  sudo apt-get update
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
  This is also required for A100/H100+ and use CUDA 11.8+.

  If you do not plan to use one of those packages, you can just use the non-dev version:
  ```bash
  conda install cudatoolkit=11.7 -c conda-forge -y
  export CUDA_HOME=$CONDA_PREFIX 
  ```
  Choose cu118 for A100/H100+.
  
* Install dependencies:
    ```bash
    # fix any bad env
    pip uninstall -y pandoc pypandoc pypandoc-binary flash-attn
    # broad support, but no training-time or data creation dependencies
    
    # CPU only:
    pip install -r requirements.txt --extra-index https://download.pytorch.org/whl/cpu
    
    # GPU only:
    pip install -r requirements.txt --extra-index https://download.pytorch.org/whl/cu117
    ```
    Choose cu118 for A100/H100+.
* Install document question-answer dependencies:
    ```bash
    # May be required for jq package:
    sudo apt-get -y install autoconf libtool
    # Required for Doc Q/A: LangChain:
    pip install -r reqs_optional/requirements_optional_langchain.txt --extra-index https://download.pytorch.org/whl/cu117
    # Required for CPU: LLaMa/GPT4All:
    pip install -r reqs_optional/requirements_optional_gpt4all.txt --extra-index https://download.pytorch.org/whl/cu117
    # Optional: PyMuPDF/ArXiv:
    pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt --extra-index https://download.pytorch.org/whl/cu117
    # Optional: Selenium/PlayWright:
    pip install -r reqs_optional/requirements_optional_langchain.urls.txt --extra-index https://download.pytorch.org/whl/cu117
    # Optional: support docx, pptx, ArXiv, etc. required by some python packages
    sudo apt-get install -y libmagic-dev poppler-utils tesseract-ocr libtesseract-dev libreoffice
    # Improved OCR with DocTR:
    conda install -y -c conda-forge pygobject
    pip install -r reqs_optional/requirements_optional_doctr.txt --extra-index https://download.pytorch.org/whl/cu117
    # go back to older onnx so Tesseract OCR still works
    pip install onnxruntime==1.15.0 onnxruntime-gpu==1.15.0 --extra-index https://download.pytorch.org/whl/cu117
    # Optional: for supporting unstructured package
    python -m nltk.downloader all
    # Optional but required for PlayWright
    playwright install --with-deps
* GPU Optional: For AutoGPTQ support on x86_64 linux
    ```bash
    pip uninstall -y auto-gptq ; pip install https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.4.2/auto_gptq-0.4.2+cu118-cp310-cp310-linux_x86_64.whl
    # in-transformers support of AutoGPTQ
    pip install git+https://github.com/huggingface/optimum.git
    ```
    This avoids issues with missing cuda extensions etc.  if this does not apply to your system, run:
    ```bash
    pip uninstall -y auto-gptq ; GITHUB_ACTIONS=true pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/ --no-cache-dir
    ```
    If one sees `CUDA extension not installed` in output after loading model, one needs to compile AutoGPTQ, else will use double memory and be slower on GPU.
    See [AutoGPTQ](README_GPU.md#autogptq) about running AutoGPT models.
* GPU Optional: For AutoAWQ support on x86_64 linux
    ```bash
    pip uninstall -y autoawq ; pip install autoawq
    ```
    If this has issues, you need to build:
    ```bash
    pip uninstall -y autoawq
    git clone https://github.com/casper-hansen/AutoAWQ
    cd AutoAWQ
    pip install .
    ```
* GPU Optional: For exllama support on x86_64 linux
    ```bash
    pip uninstall -y exllama ; pip install https://github.com/jllllll/exllama/releases/download/0.0.13/exllama-0.0.13+cu118-cp310-cp310-linux_x86_64.whl --no-cache-dir
    ```
    See [exllama](README_GPU.md#exllama) about running exllama models.

* GPU Optional: Support LLaMa.cpp with CUDA:
  * Download/Install [CUDA llama-cpp-python wheel](https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels), E.g.:
    ```bash
    pip uninstall -y llama-cpp-python llama-cpp-python-cuda
    # GGMLv3 ONLY:
    pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.1.73+cu117-cp310-cp310-linux_x86_64.whl
    # GGUF ONLY for GPU:
    pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.1.83+cu117-cp310-cp310-linux_x86_64.whl
    # GGUF ONLY for CPU (AVX2):
    pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/cpu/llama_cpp_python-0.1.83+cpuavx2-cp310-cp310-linux_x86_64.whl
    ```
     For CPU, ensure to run with `CUDA_VISIBLE_DEVICES=` in case torch with CUDA installed.
     ```bash
      CUDA_VISIBLE_DEVICES= python generate.py --base_model=llama --prompt_type=mistral --model_path_llama=https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf --max_seq_len=4096 --score_model=None
     ```
  * If any issues, then must compile llama-cpp-python with CUDA support:
   ```bash
    pip uninstall -y llama-cpp-python llama-cpp-python-cuda
    export LLAMA_CUBLAS=1
    export CMAKE_ARGS=-DLLAMA_CUBLAS=on
    export FORCE_CMAKE=1
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.73 --no-cache-dir --verbose
   ```
  * By default, we set `n_gpu_layers` to large value, so llama.cpp offloads all layers for maximum GPU performance.  You can control this by passing `--llamacpp_dict="{'n_gpu_layers':20}"` for value 20, or setting in UI.  For highest performance, offload *all* layers.
    That is, one gets maximum performance if one sees in startup of h2oGPT all layers offloaded:
      ```text
    llama_model_load_internal: offloaded 35/35 layers to GPU
    ```
  but this requires sufficient GPU memory.  Reduce if you have low memory GPU, say 15.
  * Pass to `generate.py` the option `--max_seq_len=2048` or some other number if you want model have controlled smaller context, else default (relatively large) value is used that will be slower on CPU.
  * For LLaMa2, can set `max_tokens` to a larger value for longer output.
  * If one sees `/usr/bin/nvcc` mentioned in errors, that file needs to be removed as would likely conflict with version installed for conda.  
  * Note that once `llama-cpp-python` is compiled to support CUDA, it no longer works for CPU mode, so one would have to reinstall it without the above options to recovers CPU mode or have a separate h2oGPT env for CPU mode.

* Control Core Count for chroma < 0.4 using chromamigdb package:
    * Duckdb used by Chroma < 0.4 uses DuckDB 0.8.1 that has no control over number of threads per database, `import duckdb` leads to all virtual cores as threads and each db consumes another number of threads equal to virtual cores.  To prevent this, one can rebuild duckdb using [this modification](https://github.com/h2oai/duckdb/commit/dcd8c1ffc53dd020623630efb99ba6a3a4cbc5ad) or one can try to use the prebuild wheel for x86_64 built on Ubuntu 20.
        ```bash
        pip install https://h2o-release.s3.amazonaws.com/h2ogpt/duckdb-0.8.2.dev4025%2Bg9698e9e6a8.d20230907-cp310-cp310-linux_x86_64.whl --no-cache-dir --force-reinstall --no-deps
      ```

### Compile Install Issues
  * `/usr/local/cuda/include/crt/host_config.h:132:2: error: #error -- unsupported GNU version! gcc versions later than 11 are not supported!`
    * gcc > 11 is not currently supported by nvcc.  Install GCC with a maximum version:
    ```
    MAX_GCC_VERSION=11
    sudo apt install gcc-$MAX_GCC_VERSION g++-$MAX_GCC_VERSION
    sudo update-alternatives --config gcc
    # pick version 11
    sudo update-alternatives --config g++
    # pick version 11
    ```

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
  python generate.py --base_model=h2oai/h2ogpt-4096-llama2-13b-chat --load_8bit=True  --score_model=None --langchain_mode='UserData' --user_path=user_path
  ```
  Same with a smaller model without quantization:
  ```bash
  python generate.py --base_model=h2oai/h2ogpt-4096-llama2-7b-chat --score_model=None --langchain_mode='UserData' --user_path=user_path
  ```
  UI using LLaMa.cpp LLaMa2 model:
  ```bash
  python generate.py --base_model='llama' --prompt_type=llama2 --score_model=None --langchain_mode='UserData' --user_path=user_path --model_path_llama=https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin --max_seq_len=4096
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
