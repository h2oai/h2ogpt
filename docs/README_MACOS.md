# MACOS

Supports CPU and MPS (Metal M1/M2).

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
    # Optional: for supporting unstructured package
    python -m nltk.downloader all
* For supporting Word and Excel documents, download libreoffice: https://www.libreoffice.org/download/download-libreoffice/ .
* To support OCR, install [Tesseract Documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html):
    ```bash
    brew install libmagic
    brew link libmagic
    brew install poppler
    brew install tesseract --all-languages
    ```
* Metal M1/M2 Only: Install newer Torch for GPU support:
   ```bash
   pip uninstall -y torch
   pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu
   ```
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
* Metal M1/M2 Only:  Install and setup GPU-specific dependencies to support LLaMa.cpp on GPU:
    ```bash
    pip uninstall llama-cpp-python -y
    CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir
    ```
    - In `.env_gpt4all`, uncomment line with `n_gpu_layers=20`
    - In `.env_gpt4all`, optionally change model name: `model_path_llama=WizardLM-7B-uncensored.ggmlv3.q8_0.bin`
    - **Note** Only supports v3 ggml 4 bit quantized models for MPS, so use llama models ends with `ggmlv3` & `q4_x.bin`.

---

## Run

* To run LLaMa.cpp model in CPU or GPU mode:
    ```bash
    python generate.py --base_model='llama' --prompt_type=wizard2 --score_model=None --langchain_mode='UserData' --user_path=user_path
    ```
Ignore CLI output showing `0.0.0.0`, and instead go to http://localhost:7860 or the public live URL printed by the server (disable shared link with `--share=False`).

* Full Hugging Face Model (recommended for M1/M2 only):
    ```bash
    python generate.py --base_model=h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b --score_model=None --langchain_mode='UserData' --user_path=user_path
    ```

* CLI mode:
    ```bash
    python generate.py --base_model='llama' --prompt_type=wizard2 --score_model=None --langchain_mode='UserData' --user_path=user_path --cli==True
    ```

See [CPU](README_CPU.md) and [GPU](README_GPU.md) for some other general aspects about using h2oGPT on CPU or GPU, such as which models to try.

---

## Issues

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
    ARCHFLAGS="-arch x86_64" pip install -r requirements.txt
    ```

* If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.
