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
    pip install -r requirements.txt --extra-index https://download.pytorch.org/whl/cu118
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
    # Optional: For AutoGPTQ support on x86_64 linux
    pip uninstall -y auto-gptq ; CUDA_HOME=/usr/local/cuda-11.8  GITHUB_ACTIONS=true pip install auto-gptq --no-cache-dir
    # Optional: For exllama support on x86_64 linux
    pip uninstall -y exllama ; pip install https://github.com/jllllll/exllama/releases/download/0.0.8/exllama-0.0.8+cu118-cp310-cp310-linux_x86_64.whl --no-cache-dir
    ```
    See [AutoGPTQ](docs/README_GPU.md#autogptq) for more details for AutoGPTQ and other GPU installation aspects.
    
    See [exllama](docs/README_GPU.md#) for more details for AutoGPTQ and other GPU installation aspects.

* To avoid unauthorized telemetry, which document options still do not disable, run:
    ```bash
    sp=`python -c 'import site; print(site.getsitepackages()[0])'`
    sed -i 's/posthog\.capture/return\n            posthog.capture/' $sp/chromadb/telemetry/posthog.py
    ```

---

## Run

Place all documents in `user_path` or upload in UI ([Help with UI](docs/README_ui.md)).

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

Add `--share=True` to make gradio server visible via sharable URL.  If you see an error about protobuf, try:
```bash
pip install protobuf==3.20.0
```

See [CPU](README_CPU.md) and [GPU](README_GPU.md) for some other general aspects about using h2oGPT on CPU or GPU, such as which models to try.