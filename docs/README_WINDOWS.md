# Windows 10/11

If using GPU on Windows 10/11 Pro 64-bit, we recommend using [Windows installers](../README.md#windows-1011-64-bit-with-full-document-qa-capability).

For newer builds of windows versions of 10/11.

## Installation
* Download Visual Studio 2022: [Download Link](https://visualstudio.microsoft.com/vs/community/)
  * Run Installer, click ok to run, click Continue
  * Click on `Individual Components`
  * Search for these in the search bar and click on them:
     * `Windows 11 SDK` (e.g. 10.0.22000.0)
     * `C++ Universal Windows Platform support` (e.g. for v143 build tools)
     * `MSVC VS 2022 C++ x64/x86 build tools` (latest)
     * `C++ CMake tools for Windows`
     * ![vs2022small.png](vs2022small.png)
  * Click Install, and follow through installation, and do not need to launch VS 2022 at end.
* Download the MinGW installer: [MiniGW](https://sourceforge.net/projects/mingw/)
  * Run Installer, Click Install, Continue, Install/Run to launch installation manager.
  * Select packages to install:
     * minigw32-base
     * mingw32-gcc-g++
     * ![minigw32small.png](minigw32small.png)
  * Go to installation tab, then apply changes.
* Download and install [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html)
* Run Miniconda shell (not powershell!) as Administrator
* Run: `set path=%path%;c:\MinGW\msys\1.0\bin\` to get C++ in path.  In some cases it may be instead correct to use `set path=%path%;c:\MinGW\bin\`
* Download latest nvidia driver for windows if one has old drivers before CUDA 11.7 supported
* Confirm can run `nvidia-smi` and see driver version
* Setup Conda Environment:
    * ![minicondashellsmall.png](minicondashellsmall.png)
   ```bash
    conda create -n h2ogpt -y
    conda activate h2ogpt
    conda install python=3.10 -c conda-forge -y
    python --version  # should say python 3.10.xx
    python -c "import os, sys ; print('hello world')"  # should print "hello world"
    ```
* GPU Only: Install CUDA
   ```bash
    conda install cudatoolkit=11.7 -c conda-forge -y
    set CUDA_HOME=$CONDA_PREFIX
    ```
* Install Git:
   ```bash
    conda install -c conda-forge git
    ```
* Install h2oGPT:
   ```bash
    git clone https://github.com/h2oai/h2ogpt.git
    cd h2ogpt
    ```
* Install primary dependencies.
  * Remove any bad dependencies that existed (required for new transformers it seems):
      ```bash
      pip uninstall flash-attn
       ```
  * For CPU Only:
      ```bash
      pip install -r requirements.txt --extra-index https://download.pytorch.org/whl/cpu
       ```
  * For GPU:
      ```bash
      pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
       ```
    In some cases this may lead to the message `No GPU` and in which case you can run next something like:
      ```bash
      pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
       ```
 * Optional: for bitsandbytes 4-bit and 8-bit:
   ```bash
   pip uninstall bitsandbytes -y
   pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
   ```
* Install document question-answer dependencies:
   ```bash
    # Required for Doc Q/A: LangChain:
    pip install -r reqs_optional/requirements_optional_langchain.txt --extra-index-url https://download.pytorch.org/whl/cu117
    # Required for CPU: LLaMa/GPT4All:
    pip install -r reqs_optional/requirements_optional_gpt4all.txt --extra-index-url https://download.pytorch.org/whl/cu117
    # Optional: PyMuPDF/ArXiv:
    pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt --extra-index-url https://download.pytorch.org/whl/cu117
    # Optional: Selenium/PlayWright:
    pip install -r reqs_optional/requirements_optional_langchain.urls.txt --extra-index-url https://download.pytorch.org/whl/cu117
    # Optional: for supporting unstructured package
    python -m nltk.downloader all
    # Optional but required for PlayWright
    playwright install --with-deps
    # Note: for Selenium, we match versions of playwright so above installer will add chrome version needed
* GPU Optional: For optional AutoGPTQ support:
   ```bash
    pip uninstall -y auto-gptq
    pip install https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.4.2/auto_gptq-0.4.2+cu118-cp310-cp310-win_amd64.whl
   ```
* GPU Optional: For optional AutoAWQ support:
   ```bash
    pip uninstall -y autoawq
    pip install autoawq
   ```
* GPU Optional: For optional exllama support:
    ```bash
    pip uninstall -y exllama
    pip install https://github.com/jllllll/exllama/releases/download/0.0.13/exllama-0.0.13+cu118-cp310-cp310-win_amd64.whl --no-cache-dir
    ```
* GPU Optional: Support LLaMa.cpp with CUDA via llama-cpp-python:
  * Download/Install [CUDA llama-cpp-python wheel](https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels), or choose link and run pip directly.  E.g.:
    ```bash
      pip uninstall -y llama-cpp-python llama_cpp_python_cuda
      # GGUF ONLY for GPU:
      pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.2.10+cu118-cp310-cp310-win_amd64.whl
      # GGUF ONLY for CPU for AVX2:
      pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/cpu/llama_cpp_python-0.2.9+cpuavx2-cp310-cp310-win_amd64.whl
      # GGMLv3 ONLY for GPU (no longer recommended):
      pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.1.73+cu117-cp310-cp310-win_amd64.whl
    ```
    See [https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases](https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases) for other releases, try to stick to same version.
  * If any issues, then must compile llama-cpp-python with CUDA support:
    ```bash
    pip uninstall -y llama-cpp-python
    set LLAMA_CUBLAS=1
    set CMAKE_ARGS=-DLLAMA_CUBLAS=on
    set FORCE_CMAKE=1
    pip install llama-cpp-python==0.2.11 --no-cache-dir --verbose
    ```
  * By default, we set `n_gpu_layers` to large value, so llama.cpp offloads all layers for maximum GPU performance.  You can control this by passing `--llamacpp_dict="{'n_gpu_layers':20}"` for value 20, or setting in UI.  For highest performance, offload *all* layers.
    That is, one gets maximum performance if one sees in startup of h2oGPT all layers offloaded:
      ```text
    llama_model_load_internal: offloaded 35/35 layers to GPU
    ```
  but this requires sufficient GPU memory.  Reduce if you have low memory GPU, say 15.
  * Pass to `generate.py` the option `--max_seq_len=2048` or some other number if you want model have controlled smaller context, else default (relatively large) value is used that will be slower on CPU.
  * If one sees `/usr/bin/nvcc` mentioned in errors, that file needs to be removed as would likely conflict with version installed for conda.
  * Note that once `llama-cpp-python` is compiled to support CUDA, it no longer works for CPU mode, so one would have to reinstall it without the above options to recovers CPU mode or have a separate h2oGPT env for CPU mode.
* For supporting Word and Excel documents, if you don't have Word/Excel already, then download and install libreoffice: https://www.libreoffice.org/download/download-libreoffice/ .
* To support OCR, download and install [tesseract](https://github.com/UB-Mannheim/tesseract/wiki), see also: [Tesseract Documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html).  Please add the installation directories to your PATH.
* vLLM support

    Run windows equivalent of this sequence from Bash/Linux (can use bash shell in windows):
    ```bash
    cd $HOME/miniconda3/envs/h2ogpt/lib/python3.10/site-packages/
    rm -rf openvllm* openai_vllm*
    cp -a openai openvllm
    file0=`ls|grep openai|grep dist-info`
    file1=`echo $file0|sed 's/openai-/openvllm-/g'`
    cp -a $file0 $file1
    find openvllm -name '*.py' | xargs sed -i 's/from openai /from openvllm /g'
    find openvllm -name '*.py' | xargs sed -i 's/openai\./openvllm./g'
    find openvllm -name '*.py' | xargs sed -i 's/from openai\./from openvllm./g'
    find openvllm -name '*.py' | xargs sed -i 's/import openai/import openvllm/g'
    find openvllm -name '*.py' | xargs sed -i 's/OpenAI/vLLM/g'
    ```
---

## Run
* For document Q/A with UI using LLaMa.cpp-based model on CPU or GPU:

  * Choose some GGUF model by [TheBloke](https://huggingface.co/TheBloke), then do:
       ```bash
       python generate.py --base_model='llama' --prompt_type=llama2 --score_model=None --langchain_mode='UserData' --user_path=user_path --model_path_llama=https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf --max_seq_len=4096
       ```
    Choose some other `model_path_llama` from TheBloke if desired, e.g. 13B.  If no model passed, the 7B LLaMa-2 GGUF is used.
    For an absolute windows path, change to `--user_path=C:\Users\YourUsername\h2ogpt` or something similar for some user `YourUsername`.
      If llama-cpp-python was compiled with CUDA support, you should see in the output:
    ```text
      Device 0: NVIDIA GeForce RTX 3090 Ti
    ```
  * Go to `http://127.0.0.1:7860` (ignore message above).  Add `--share=True` to get sharable secure link.
  * To just chat with LLM, click `Resources` and click `LLM` in Collections, or start without `--langchain_mode=UserData`.
  * In `nvidia-smi` or some other GPU monitor program you should see `python.exe` using GPUs in `C` (Compute) mode and using GPU resources.
  * If you have multiple GPUs, best to specify to use the fasted GPU by doing (e.g. if device 0 is fastest and largest memory GPU):
    ```bash
    set CUDA_VISIBLE_DEVICES=0
    ```
  * On an i9 with 3090Ti, one gets about 5 tokens/second.

  * ![llamasmall.jpg](llamasmall.jpg)

  * For LLaMa2 70B model, launch as
    ```bash
    python generate.py --base_model=llama --model_path_llama=https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf n_gqa=8
    ```
* To use Hugging Face type models (faster on GPU than LLaMa.cpp if one has a powerful GPU with enough memory):
   ```bash
   python generate.py --base_model=h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3 --langchain_mode=UserData --score_model=None
   ```
  * On an i9 with 3090Ti, one gets about 9 tokens/second.
* To use Hugging Face type models in 8-bit do:
   ```bash
   python generate.py --base_model=h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3 --langchain_mode=UserData --score_model=None --load_8bit=True
   ```
  When running windows on GPUs with bitsandbytes in 8-bit you should see something like the below in output:
  ```bash
  bin C:\Users\pseud\.conda\envs\h2ogpt\lib\site-packages\bitsandbytes\libbitsandbytes_cuda117.dll
  ```
  * On an i9 with 3090Ti, one gets about 5 tokens/second, so about half 16-bit speed.
  * You can confirm GPU use via `nvidia-smi` showing GPU memory consumed is less than 16-bit, at about 9.2GB when in use.  Also try 13B models in 8-bit for similar memory usage.
  * Note 8-bit inference is about twice slower than 16-bit inference, and the only use of 8-bit is to keep memory profile low.
  * Bitsandbytes can be uninstalled (`pip uninstall bitsandbytes`) and still h2oGPT can be used if one does not pass `--load_8bit=True`.
* To use Hugging Face type models in 4-bit do:
   ```bash
   python generate.py --base_model=h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3 --langchain_mode=UserData --score_model=None --load_4bit=True
   ```
  * On an i9 with 3090Ti, one gets about 4 tokens/second, so still about half 16-bit speed.  Memory use is about 6.6GB.

See [CPU](README_CPU.md) and [GPU](README_GPU.md) for some other general aspects about using h2oGPT on CPU or GPU, such as which models to try, quantization, etc.

## Issues
* SSL Certification failure when connecting to Hugging Face.
  * Your org may be blocking HF
  * Try: https://stackoverflow.com/a/75111104
  * Or try: https://github.com/huggingface/transformers/issues/17611#issuecomment-1619582900
  * Try using proxy.
* If you see import problems, then try setting `PYTHONPATH` in a `.bat` file:
  ```shell
  SET PYTHONPATH=.:src:$PYTHONPATH
  python generate.py ...
  ```
  for some options ...
* For easier handling of command line operations, consider using bash in windows with [coreutils](https://github.com/git-for-windows/git/releases/download/v2.41.0.windows.3/Git-2.41.0.3-64-bit.exe).
