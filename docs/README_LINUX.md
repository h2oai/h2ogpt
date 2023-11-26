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

* For GPU: Install CUDA ToolKit with ability to compile using nvcc for some packages like llama-cpp-python, AutoGPTQ, exllama, flash attention, TTS use of deepspeed, by going to [CUDA Toolkit](INSTALL.md#install-cuda-toolkit).  In order to avoid removing the original CUDA toolkit/driver you have, on NVIDIA's website, use the `runfile (local)` installer, and choose to not install driver or overwrite `/usr/local/cuda` link and just install the toolkit, and rely upon the `CUDA_HOME` env to point to the desired CUDA version.  Then do:
  ```bash
  export CUDA_HOME=/usr/local/cuda-11.8
  ```
  Or if you do not plan to use packages like deepspeed in coqui's TTS or build other packages (i.e. only use binaries), you can just use the non-dev version from conda if preferred:
  ```bash
  conda install cudatoolkit=11.8 -c conda-forge -y
  export CUDA_HOME=$CONDA_PREFIX 
  ```
  Do not install `cudatoolkit-dev` as it only goes up to cuda 11.7 that is no longer supported.

* Place the `CUDA_HOME` export into your `~/.bashrc` or before starting h2oGPT for TTS's use of deepspeed to work.
  
* Prepare to install dependencies:
   ```bash
   export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu118"
   ```
  Choose cu118+ for A100/H100+.  Or for CPU set
   ```bash
   export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
   ```
* Optional: For document Q/A and use of DocTR.  Install before other pips to avoid long conflict checks.
   ```bash
   conda install weasyprint pygobject -c conda-forge -y
   ```
   Avoids library mismatch.
* Install primary dependencies
    ```bash
    # fix any bad env
    pip uninstall -y pandoc pypandoc pypandoc-binary flash-attn
    # broad support, but no training-time or data creation dependencies
    
    pip install -r requirements.txt
    ```
* Optional: Install document question-answer dependencies:
    ```bash
    # May be required for jq package:
    sudo apt-get -y install autoconf libtool
    # Required for Doc Q/A: LangChain:
    pip install -r reqs_optional/requirements_optional_langchain.txt
    # Required for CPU: LLaMa/GPT4All:
    pip install -r reqs_optional/requirements_optional_gpt4all.txt
    # Optional: PyMuPDF/ArXiv:
    #   Note!! that pymupdf is AGPL, requiring any source code be made available, but it's like GPL and too strong a constraint for general commercial use.
    pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt
    # Optional: Selenium/PlayWright:
    pip install -r reqs_optional/requirements_optional_langchain.urls.txt
    # Optional: support docx, pptx, ArXiv, etc. required by some python packages
    sudo apt-get install -y libmagic-dev poppler-utils tesseract-ocr libtesseract-dev libreoffice
    # Optional: For DocTR
      pip install -r reqs_optional/requirements_optional_doctr.txt
      # For DocTR: go back to older onnx so Tesseract OCR still works
      pip install onnxruntime==1.15.0
      # GPU only:
      pip install onnxruntime-gpu==1.15.0
    # Optional: for supporting unstructured package
    python -m nltk.downloader all
    # Optional: Required for PlayWright
    playwright install --with-deps
    # Audio transcription from Youtube videos and local mp3 files:
      pip install pydub==0.25.1 librosa==0.10.1 ffmpeg==1.4 yt_dlp==2023.10.13 wavio==0.0.8
    # STT from microphone (may not be required if ffmpeg installed above)
    sudo apt-get install ffmpeg
    # for any TTS:
      pip install torchaudio soundfile==0.12.1
    # GPU Only: for Coqui XTTS (ensure CUDA_HOME set and consistent with added postfix for extra-index):
      # pydantic can't be >=2.0
      # relaxed versions to avoid conflicts
      pip install TTS deepspeed noisereduce pydantic==1.10.13 emoji ffmpeg-python==0.2.0 trainer pysbd coqpit
      # for Coqui XTTS language helpers (specific versions probably not required)
      pip install cutlet==0.3.0 langid==1.1.6 g2pkk==0.1.2 jamo==0.4.1 gruut[de,es,fr]==2.2.3 jieba==0.42.1
    ```
* STT and TTS Notes:
  * STT: Ensure microphone is on and in browser go to http://localhost:7860 instead of http://0.0.0.0:7860 for microphone to be possible to allow in browser.
  * TTS: For XTT models, ensure `CUDA_HOME` is set correctly, because deepspeed compiles at runtime using torch and nvcc.  Those must match CUDA version.  E.g. if used `--extra-index https://download.pytorch.org/whl/cu118`, then must have ENV `CUDA_HOME=/usr/local/cuda-11.7` or ENV from conda must be that version.  Since conda only has up to cuda 11.7 for dev toolkit, but H100+ need cuda 11.8, for those cases one should download the toolkit from NVIDIA.
* HNSW issue:
    In some cases old chroma migration package will install old hnswlib and that may cause issues when making a database, then do:
   ```bash
   pip uninstall hnswlib==0.7.0
   # restore correct version
   pip install chroma-hnswlib==0.7.3
   ```
* Selenium needs to have chrome installed, e.g. on Ubuntu:
    ```bash
    sudo bash
    apt install -y unzip xvfb libxi6 libgconf-2-4
    apt install -y default-jdk
    curl -sS -o - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add
    bash -c "echo 'deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main' >> /etc/apt/sources.list.d/google-chrome.list"
    apt -y update
    apt -y install google-chrome-stable  # e.g. Google Chrome 114.0.5735.198
    google-chrome --version  # e.g. Google Chrome 114.0.5735.198
    # visit https://chromedriver.chromium.org/downloads and download matching version
    # E.g.
    wget https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_linux64.zip
    unzip chromedriver_linux64.zip
    sudo mv chromedriver /usr/bin/chromedriver
    sudo chown root:root /usr/bin/chromedriver
    sudo chmod +x /usr/bin/chromedriver
    ```
* GPU Optional: For AutoGPTQ support on x86_64 linux
    ```bash
    pip uninstall -y auto-gptq ; pip install https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.4.2/auto_gptq-0.4.2+cu118-cp310-cp310-linux_x86_64.whl
    # in-transformers support of AutoGPTQ, requires also auto-gptq above to be installed since used internally by transformers/optimum
    pip install optimum==1.14.1
    ```
    This avoids issues with missing cuda extensions etc.  if this does not apply to your system, run:
    ```bash
    pip uninstall -y auto-gptq ; GITHUB_ACTIONS=true pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/ --no-cache-dir
    ```
    If one sees `CUDA extension not installed` in output after loading model, one needs to compile AutoGPTQ, else will use double memory and be slower on GPU.
    See [AutoGPTQ](README_GPU.md#autogptq) about running AutoGPT models.
* GPU Optional: For AutoAWQ support on x86_64 linux
    ```bash
    pip uninstall -y autoawq ; pip install autoawq==0.1.7
    # fix version since don't need lm-eval to have its version of 1.5.0
    pip install sacrebleu==2.3.1 --upgrade
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
    pip uninstall -y exllama ; pip install https://github.com/jllllll/exllama/releases/download/0.0.18/exllama-0.0.18+cu118-cp310-cp310-linux_x86_64.whl --no-cache-dir
    ```
    See [exllama](README_GPU.md#exllama) about running exllama models.

* GPU Optional: Support LLaMa.cpp with CUDA:
  * Download/Install from [CUDA llama-cpp-python wheel](https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels) or [https://github.com/abetlen/llama-cpp-python/releases](https://github.com/abetlen/llama-cpp-python/releases), E.g.:
    * GGUF ONLY for CUDA GPU (keeping CPU package in place to support CPU + GPU at same time):
      ```bash
      pip uninstall -y llama-cpp-python-cuda
      pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.2.18+cu118-cp310-cp310-manylinux_2_31_x86_64.whl
      ```
    * GGUF ONLY for CPU-AVX (can be used with -cuda one above)
      ```bash
      pip uninstall -y llama-cpp-python
      pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/cpu/llama_cpp_python-0.2.18+cpuavx2-cp310-cp310-manylinux_2_31_x86_64.whl
      ```
      For CPU, ensure to run with `CUDA_VISIBLE_DEVICES=` in case torch with CUDA installed.
  * If any issues, then must compile llama-cpp-python with CUDA support:
   ```bash
    pip uninstall -y llama-cpp-python llama-cpp-python-cuda
    export LLAMA_CUBLAS=1
    export CMAKE_ARGS=-DLLAMA_CUBLAS=on
    export FORCE_CMAKE=1
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.18 --no-cache-dir --verbose
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
* GPU Optional: Support attention sinks for infinite generation
    ```bash
    pip install attention_sinks --no-deps
    # --no-deps is to avoid going back to old transformers/tokenizers https://github.com/tomaarsen/attention_sinks/issues/26
  ```
* GPU Optional: Support amazon/MistralLite with flash attention 2
   ```bash
    pip install flash-attn==2.3.4 --no-build-isolation
  ```
* Control Core Count for chroma < 0.4 using chromamigdb package:
    * Duckdb used by Chroma < 0.4 uses DuckDB 0.8.1 that has no control over number of threads per database, `import duckdb` leads to all virtual cores as threads and each db consumes another number of threads equal to virtual cores.  To prevent this, one can rebuild duckdb using [this modification](https://github.com/h2oai/duckdb/commit/dcd8c1ffc53dd020623630efb99ba6a3a4cbc5ad) or one can try to use the prebuild wheel for x86_64 built on Ubuntu 20.
        ```bash
        pip install https://h2o-release.s3.amazonaws.com/h2ogpt/duckdb-0.8.2.dev4025%2Bg9698e9e6a8.d20230907-cp310-cp310-linux_x86_64.whl --no-cache-dir --force-reinstall --no-deps
      ```
* SERP for search:
  ```bash
  pip install -r reqs_optional/requirements_optional_agents.txt
  ```
  For more info see [SERP Docs](README_SerpAPI.md).
* Deal with not-thread-safe things in LangChain:
    ```bash
  sp=`python3.10 -c 'import site; print(site.getsitepackages()[0])'`
  cd $sp
  sed -i  's/with HiddenPrints():/if True:/g' langchain/utilities/serpapi.py
    ```
* vLLM support
   ```bash
   pip install https://h2o-release.s3.amazonaws.com/h2ogpt/openvllm-0.28.1-py3-none-any.whl
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

See [FAQ](FAQ.md#adding-models) for many ways to run models.  The below are some other examples.

Note models are stored in `/home/$USER/.cache/` for chroma, huggingface, selenium, torch, weaviate, etc. directories.

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
  python generate.py --base_model='llama' --prompt_type=llama2 --score_model=None --langchain_mode='UserData' --user_path=user_path --model_path_llama=https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf --max_seq_len=4096
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
