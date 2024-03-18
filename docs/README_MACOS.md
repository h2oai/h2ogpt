# macOS

Supports CPU and MPS (Metal M1/M2).

- [Install](#install)
- [Run](#run)

## Install
* Download and Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html#macos-installers) for Python 3.10.
* Run Miniconda
* Setup environment with Conda Rust:
    ```bash
    conda create -n h2ogpt python=3.10 rust
    conda activate h2ogpt
    ```
* Install dependencies:
    ```bash
    git clone https://github.com/h2oai/h2ogpt.git
    cd h2ogpt

    # fix any bad env
    pip uninstall -y pandoc pypandoc pypandoc-binary
    pip install --upgrade pip
    python -m pip install --upgrade setuptools
    
    # Install Torch:
    pip install -r requirements.txt --extra-index https://download.pytorch.org/whl/cpu -c reqs_optional/reqs_constraints.txt
    ```
* Install document question-answer dependencies:
    ```bash
    # Required for Doc Q/A: LangChain:
    pip install -r reqs_optional/requirements_optional_langchain.txt -c reqs_optional/reqs_constraints.txt
  
    # Required for CPU: LLaMa/GPT4All:
    pip uninstall -y llama-cpp-python llama-cpp-python-cuda
    export CMAKE_ARGS=-DLLAMA_METAL=on
    export FORCE_CMAKE=1
    pip install -r reqs_optional/requirements_optional_llamacpp_gpt4all.txt -c reqs_optional/reqs_constraints.txt --no-cache-dir

    pip install librosa -c reqs_optional/reqs_constraints.txt
    # Optional: PyMuPDF/ArXiv:
    pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt -c reqs_optional/reqs_constraints.txt
    # Optional: Selenium/PlayWright:
    pip install -r reqs_optional/requirements_optional_langchain.urls.txt -c reqs_optional/reqs_constraints.txt
    # Optional: DocTR OCR:
    conda install weasyprint pygobject -c conda-forge -y
    pip install -r reqs_optional/requirements_optional_doctr.txt -c reqs_optional/reqs_constraints.txt
    # Optional: for supporting unstructured package
    python -m nltk.downloader all
  ```
* For supporting Word and Excel documents, download libreoffice: https://www.libreoffice.org/download/download-libreoffice/ .
* To support OCR, install [Tesseract Documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html):
    ```bash
    brew install libmagic
    brew link libmagic
    brew install poppler
    brew install tesseract
    brew install tesseract-lang
    brew install rubberband
    brew install pygobject3 gtk4
    brew install libjpeg
    brew install libpng
    brew install wget
    ```

See [FAQ](FAQ.md#adding-models) for how to run various models.  See [CPU](README_CPU.md) and [GPU](README_GPU.md) for some other general aspects about using h2oGPT on CPU or GPU, such as which models to try.

## Run 

For information on how to run h2oGPT offline, see [Offline](README_offline.md#tldr).

In your terminal, run:
```bash
python generate.py --base_model=TheBloke/zephyr-7B-beta-GGUF --prompt_type=zephyr --max_seq_len=4096
```
Or you can run it from a file called `run.sh` that would contain following text:
```bash
#!/bin/bash
python generate.py --base_model=TheBloke/zephyr-7B-beta-GGUF --prompt_type=zephyr --max_seq_len=4096
```
and run `sh run.sh` from the terminal placed in the parent folder of `run.sh`

---

## Issues
* Metal M1/M2 Only:
   Verify whether torch uses MPS, run below python script:
     ```python
      import torch
      if torch.backends.mps.is_available():
          mps_device = torch.device("mps")
          x = torch.ones(1, device=mps_device)
          print (x)
      else:
          print ("MPS device not found.")
     ```
  Output
     ```bash
     tensor([1.], device='mps:0')
     ```
* If you see `ld: library not found for -lSystem` then ensure you do below and then retry from scratch to do `pip install` commands:
    ```bash
    export LDFLAGS=-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib`
    ```
* If conda Rust has issus, you can download and install [Native Rust]((https://www.geeksforgeeks.org/how-to-install-rust-in-macos/):
    ```bash
    curl –proto ‘=https’ –tlsv1.2 -sSf https://sh.rustup.rs | sh
    # enter new shell and test:
    rustc --version
    ```
* When running a Mac with Intel hardware (not M1), you may run into
    ```text
    _clang: error: the clang compiler does not support '-march=native'_
    ```
    during pip install.  If so, set your archflags during pip install. E.g.
    ```bash
    ARCHFLAGS="-arch x86_64" pip install -r requirements.txt -c reqs_optional/reqs_constraints.txt
    ```
* If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.
* If you see the error `TypeError: Trying to convert BFloat16 to the MPS backend but it does not have support for that dtype.`:
  ```bash
  pip install -U torch==2.3.0.dev20240229 --extra-index https://download.pytorch.org/whl/nightly/cpu --pre
  pip install -U torchvision==0.18.0.dev20240229 --extra-index https://download.pytorch.org/whl/nightly/cpu --pre
  ```
  * Support for BFloat16 is added to MacOS from Sonama (14.0)
  * Based on that a fix is added to torch with PR https://github.com/pytorch/pytorch/pull/119641 and it is still only available in nighlty. Expecting the feature with release `torch-2.3.0`
  * **Note:** Updating torch to nighlty is experimental and it might break features in h2ogpt
