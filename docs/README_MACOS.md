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
    
    # Install Torch:
    pip install -r requirements.txt --extra-index https://download.pytorch.org/whl/cpu
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
    brew install tesseract
    brew install tesseract-lang
    ```
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
* Metal M1/M2 Only:  Install and setup GPU-specific dependencies to support LLaMa.cpp on GPU:
    ```bash
    pip uninstall llama-cpp-python -y
    CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install -U llama-cpp-python==0.1.78 --no-cache-dir
    ```
    - Pass difference value of `--model_path_llama` if download a different GGML v3 model from TheBloke, or pass URL/path in UI. The default model can be [downloaded here](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin) and placed in repo folder or give this URL.
    - **Note** Only supports v3 ggml 4 bit quantized models for MPS, so use llama models ends with `ggmlv3` & `q4_x.bin`.

* GGUF and GGML:
  See [https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases](https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases) for other releases, try to stick to same version.  One roughly follows:
  ```
  pip uninstall -y llama-cpp-python llama-cpp-python-cuda
  # GGUF:
  pip install https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/metal/llama_cpp_python-0.2.10-cp310-cp310-macosx_11_0_arm64.whl
  ```

---

## Run

* To run LLaMa.cpp model in CPU or GPU mode (NOTE: if you haven't compiled llama-cpp-python for M1/M2 as mentioned above you can simply run without `--llamacpp_dict` arg, which will run on CPU):
    
    * CPU Mode: To run in CPU mode, specify the `'n_gpu_layers':0` in `--llamacpp_dict` arg
      ```bash
      python generate.py --base_model='llama' --prompt_type=llama2 --score_model=None --langchain_mode='UserData' --user_path=user_path --model_path_llama=https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin --max_seq_len=4096 --llamacpp_dict="{'n_gpu_layers':0,'n_batch':128}"
      ```
    * GPU Mode: To run in GPU mode, specify the number of gpus needed to be used `'n_gpu_layers: 2'` in `--llamacpp_dict` arg, by default it is set to higher value to use all the available gpus
       ```bash
      python generate.py --base_model='llama' --prompt_type=llama2 --score_model=None --langchain_mode='UserData' --user_path=user_path --model_path_llama=https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin --max_seq_len=4096
      ```
Ignore CLI output showing `0.0.0.0`, and instead go to http://localhost:7860 or the public live URL printed by the server (disable shared link with `--share=False`).

* Full Hugging Face Model -- slower than GGML in general:
    ```bash
    python generate.py --base_model=h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b --score_model=None --langchain_mode='UserData' --user_path=user_path
    ```

* CLI mode:
    ```bash
    python generate.py --base_model='llama' --prompt_type=llama2 --score_model=None --langchain_mode='UserData' --user_path=user_path --cli=True --model_path_llama=https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin --max_seq_len=4096
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
