# Windows 10/11

## Installation
* Install Visual Studio 2022 (requires newer windows versions of 10/11) with following selected:
   * Windows 11 SDK
   * C++ Universal Windows Platform support for development
   * MSVC VS 2022 C++ x64/x86 build tools
   * C++ CMake tools for Windows
* Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/) and select, go to installation tab, then apply changes:
   * minigw32-base
   * mingw32-gcc-g++
* Download and install [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) and Run Miniconda shell (not power shell) as administrator
* Run: `set path=%path%;c:\MinGW\msys\1.0\bin\` to get C++ in path
* Download latest nvidia driver for windows
* Confirm can run `nvidia-smi` and see driver version
* Setup Conda Environment:
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
    export CUDA_HOME=$CONDA_PREFIX
    ```
* Install h2oGPT:
   ```bash
    git clone https://github.com/h2oai/h2ogpt.git
    cd h2ogpt
    ```
* Install primary dependencies.

  * For CPU Only:
      ```bash
      pip install -r requirements.txt --extra-index https://download.pytorch.org/whl/cpu
       ```
  * For GPU:
      ```bash
      pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
       ```
    Optional: for bitsandbytes 4-bit and 8-bit:
       ```bash
       pip uninstall bitsandbytes -y
       pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.40.1.post1-py3-none-win_amd64.whl
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
    # Optional: for supporting unstructured package
    python -m nltk.downloader all
   ```
* GPU Only: For optional AutoGPTQ support:
   ```bash
    pip uninstall -y auto-gptq
    pip install https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.3.0/auto_gptq-0.3.0+cu118-cp310-cp310-win_amd64.whl
* GPU Only: For optional exllama support:
   ```bash
    pip uninstall -y exllama
    pip install https://github.com/jllllll/exllama/releases/download/0.0.8/exllama-0.0.8+cu118-cp310-cp310-win_amd64.whl --no-cache-dir
    ```
* GPU Only: For optional llama-cpp-python CUDA support:
  ```bash
  pip uninstall -y llama-cpp-python
  set LLAMA_CUBLAS=1
  set CMAKE_ARGS=-DLLAMA_CUBLAS=on
  set FORCE_CMAKE=1
  pip install llama-cpp-python==0.1.68 --no-cache-dir --verbose
  ```
   and uncomment `# n_gpu_layers=20` in `.env_gpt4all`.  One can try also `40` instead of `20`.  If one sees `/usr/bin/nvcc` mentioned in errors, that file needs to be removed as would likely conflict with version installed for conda.  
   Note that once `llama-cpp-python` is compiled to support CUDA, it no longer works for CPU mode,
   so one would have to reinstall it without the above options to recovers CPU mode or have a separate h2oGPT env for CPU mode.
* For supporting Word and Excel documents, if you don't have Word/Excel already, then download and install libreoffice: https://www.libreoffice.org/download/download-libreoffice/ .
* To support OCR, download and install [tesseract](https://github.com/UB-Mannheim/tesseract/wiki), see also: [Tesseract Documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html).  Please add the installation directories to your PATH.

---

## Run
   For document Q/A with UI using CPU:
   ```bash
   python generate.py --base_model='llama' --prompt_type=wizard2 --score_model=None --langchain_mode='UserData' --user_path=user_path
   ```
   For document Q/A with UI using GPU:
   ```bash
   python generate.py --base_model=h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b --langchain_mode=UserData --score_model=None
   ```
For the above, ignore the CLI output saying `0.0.0.0`, and instead point browser at http://localhost:7860 (for windows/mac) or the public live URL printed by the server (disable shared link with `--share=False`).

See [CPU](README_CPU.md) and [GPU](README_GPU.md) for some other general aspects about using h2oGPT on CPU or GPU, such as which models to try.

---

## Issues

When running windows on GPUs with bitsandbytes you should see something like:
```bash
(h2ogpt) c:\Users\pseud\h2ogpt>python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b --load_8bit=True
bin C:\Users\pseud\.conda\envs\h2ogpt\lib\site-packages\bitsandbytes\libbitsandbytes_cuda118.dll
Using Model h2oai/h2ogpt-oig-oasst1-512-6_9b
device_map: {'': 0}
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:06<00:00,  2.16s/it]
device_map: {'': 1}
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://f8fa95f123416c72dc.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades (NEW!), check out Spaces: https://huggingface.co/spaces
```
where bitsandbytes cuda118 was used because conda cuda toolkit is cuda 11.7.  You can confirm GPU use via `nvidia-smi` showing GPU memory consumed.

Note 8-bit inference is about twice slower than 16-bit inference, and the only use of 8-bit is to keep memory profile low.

Bitsandbytes can be uninstalled (`pip uninstall bitsandbytes`) and still h2oGPT can be used if one does not pass `--load_8bit=True`.
