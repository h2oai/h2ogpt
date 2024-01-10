# Windows 10/11

If using GPU on Windows 10/11 Pro 64-bit, we recommend using [Windows installers](../README.md#windows-1011-64-bit-with-full-document-qa-capability).  This excludes DocTR and PlayWright support. 

For newer builds of windows versions of 10/11.

## Install
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
* Download latest nvidia driver for windows if one has old drivers before CUDA 11.8 supported
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
    conda install cudatoolkit=11.8 -c conda-forge -y
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
* Prepare to install dependencies:
   ```cmdline
   set PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118 https://huggingface.github.io/autogptq-index/whl/cu118/
   ```
  Choose cu118+ for A100/H100+.  Or for CPU set
   ```cmdline
   set PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
   ```
* Run [`docs\windows_install.bat](windows_install.bat) for full normal document Q/A installation.  To allow all (GPL too) packages, run:
    ```cmdline
    set GPLOK=1
    docs\windows_install.bat
    ```
One can pick and choose different optional things to install instead by commenting them out in the shell script, or edit the script if any issues.  See script for notes about installation.

See [`docs\windows_install.bat](windows_install.bat) for additional installation instructions for:
 * Microsoft Word/Excel support
 * Tesseract OCR support

Note models are stored in `C:\Users\<user>\.cache\` for chroma, huggingface, selenium, torch, weaviate, etc. directories.  For an absolute windows path, choose `--user_path=C:\Users\YourUsername\h2ogpt` or something similar for some user `YourUsername`.  If the model is using the GPU, in `nvidia-smi` or some other GPU monitor program you should see `python.exe` using GPUs in `C` (Compute) mode and using GPU resources.  Use `set CUDA_VISIBLE_DEVICES=0` to pick first model, since llama.cpp models cannot choose which GPU otherwise.

See [FAQ](FAQ.md#adding-models) for how to run various models.  See [CPU](README_CPU.md) and [GPU](README_GPU.md) for some other general aspects about using h2oGPT on CPU or GPU, such as which models to try, quantization, etc.

## Possible Issues
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
