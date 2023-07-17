### Windows 10/11

Follow these steps:
1. Install Visual Studio 2022 (requires newer windows versions of 10/11) with following selected:
   * Windows 11 SDK
   * C++ Universal Windows Platform support for development
   * MSVC VS 2022 C++ x64/x86 build tools
   * C++ CMake tools for Windows
2. Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/) and select, go to installation tab, then apply changes:
   * minigw32-base
   * mingw32-gcc-g++
3. Download and install [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) and Run Miniconda shell (not power shell) as administrator
4. Run: `set path=%path%;c:\MinGW\msys\1.0\bin\` to get C++ in path
5. Download latest nvidia driver for windows
6. Confirm can run `nvidia-smi` and see driver version
7. Run: `wsl --install`
8. Setup Conda Environment:
   ```bash
    conda create -n h2ogpt -y
    conda activate h2ogpt
    conda install python=3.10 -c conda-forge -y
    conda install cudatoolkit -c conda-forge -y  # required for bitsandbytes
    python --version  # should say python 3.10.xx
    python -c "import os, sys ; print('hello world')"  # should print "hello world"
    git clone https://github.com/h2oai/h2ogpt.git
    cd h2ogpt
    ```
9. Install dependencies.

    For CPU:
    ```bash
   pip install -r requirements.txt
    ```
   For GPU:
    ```bash
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
    ```
10. For GPU, install bitsandbytes 4-bit and 8-bit:
    ```bash
    pip uninstall bitsandbytes
    pip install https://github.com/jllllll/bitsandbytes-windows-webui/raw/main/bitsandbytes-0.39.0-py3-none-any.whl
    ```
    unless you have compute capability <7.0, then your GPU only supports 8-bit (not 4-bit) and you should install older bitsandbytes:
    ```bash
    pip uninstall bitsandbytes
    pip install https://github.com/jllllll/bitsandbytes-windows-webui/raw/main/bitsandbytes-0.38.1-py3-none-any.whl
    ```
11. Install optional document Q/A dependencies
    ```bash
    pip install -r reqs_optional/requirements_optional_langchain.txt
    pip install -r reqs_optional/requirements_optional_gpt4all.txt
    pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt
    pip install -r reqs_optional/requirements_optional_langchain.urls.txt
    ```
    Optional dependencies for supporting unstructured package
    ```bash
    python -m nltk.downloader all
    ```
    For supporting Word and Excel documents download and install libreoffice: https://www.libreoffice.org/download/download-libreoffice/ . To support OCR, downnload and install [tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
12. Install optional AutoGPTQ dependency:
    ```bash
    pip install -r https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.2.2/auto_gptq-0.2.2+cu118-cp310-cp310-win_amd64.whl
    ```
13. Run h2oGPT for chat only:
    ```bash
    python generate.py --base_model=h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b --score_model=None
    ```
    For document Q/A with UI using CPU:
    ```bash
    python generate.py --base_model='llama' --prompt_type=wizard2 --score_model=None --langchain_mode='UserData' --user_path=user_path
    ```
    For document Q/A with UI using GPU:
    ```bash
    python generate.py --base_model=h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b --langchain_mode=MyData --score_model=None
    ```
For the above, ignore the CLI output saying `0.0.0.0`, and instead point browser at http://localhost:7860 (for windows/mac) or the public live URL printed by the server (disable shared link with `--share=False`).

See [CPU](README_CPU.md) and [GPU](README_GPU.md) for some other general aspects about using h2oGPT on CPU or GPU, such as which models to try.

---

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
where bitsandbytes cuda118 was used because conda cuda toolkit is cuda 11.8.  You can confirm GPU use via `nvidia-smi` showing GPU memory consumed.

Note 8-bit inference is about twice slower than 16-bit inference, and the only use of 8-bit is to keep memory profile low.

Bitsandbytes can be uninstalled (`pip uninstall bitsandbytes`) and still h2oGPT can be used if one does not pass `--load_8bit=True`.
