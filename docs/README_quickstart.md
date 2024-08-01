# Quick Start

## Install

To quickly try out h2oGPT with limited document Q/A capability, create a fresh Python 3.10 environment and run:
* CPU or MAC (M1/M2):
   ```bash
   # for windows/mac use "set" or relevant environment setting mechanism
   export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
   ```
* Linux/Windows CPU/CUDA/ROC:
   ```bash
   # for windows/mac use "set" or relevant environment setting mechanism
   export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121 https://huggingface.github.io/autogptq-index/whl/cu121"
   # for cu118 use export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu118 https://huggingface.github.io/autogptq-index/whl/cu118"
   ```
Then choose your llama_cpp_python options, by changing `CMAKE_ARGS` to whichever system you have according to [llama_cpp_python backend documentation](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends).
E.g. CUDA on Linux:
```bash
export LLAMA_CUBLAS=1
export CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all"
export FORCE_CMAKE=1
```
Note for some reason things will fail with llama_cpp_python if don't add all cuda arches, and building with all those arches does take some time.
Windows CUDA:
```cmdline
set CMAKE_ARGS=-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all
set LLAMA_CUBLAS=1
set FORCE_CMAKE=1
```
Note for some reason things will fail with llama_cpp_python if don't add all cuda arches, and building with all those arches does take some time.
Metal M1/M2:
```bash
export CMAKE_ARGS="-DLLAMA_METAL=on"
export FORCE_CMAKE=1
```
Run PyPI install:
```bash
pip install h2ogpt
```
or manually install
```bash
   ```bash
   git clone https://github.com/h2oai/h2ogpt.git
   cd h2ogpt
   pip install -r requirements.txt
   pip install -r reqs_optional/requirements_optional_langchain.txt

   pip uninstall llama_cpp_python llama_cpp_python_cuda -y
   pip install -r reqs_optional/requirements_optional_llamacpp_gpt4all.txt --no-cache-dir

   pip install -r reqs_optional/requirements_optional_langchain.urls.txt
   # GPL, only run next line if that is ok:
   pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt
```

## Chat with h2oGPT

```bash
   # choose up to 32768 if have enough GPU memory:
   python generate.py --base_model=TheBloke/Mistral-7B-Instruct-v0.2-GGUF --prompt_type=mistral --max_seq_len=4096
   ```
Next, go to your browser by visiting [http://127.0.0.1:7860](http://127.0.0.1:7860) or [http://localhost:7860](http://localhost:7860).  Choose 13B for a better model than 7B.

#### Chat template based GGUF models

For newer chat template models, a `--prompt_type` is not required on CLI, but for GGUF files one should pass the HF tokenizer so it knows the chat template, e.g. for LLaMa-3:
```bash
python generate.py --base_model=llama --model_path_llama=https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf?download=true --tokenizer_base_model=meta-llama/Meta-Llama-3-8B-Instruct --max_seq_len=8192
```
Or for Phi:
```bash
python generate.py  --tokenizer_base_model=microsoft/Phi-3-mini-4k-instruct --base_model=llama --llama_cpp_model=https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf --max_seq_len=4096 
```
the `--llama_cpp_path` could be a local path as well if you already downloaded it, or we will also check the `llamacpp_path` for the file.

See [Offline](docs/README_offline.md#tldr) for how to run h2oGPT offline.

---

Note that for all platforms, some packages such as DocTR, Unstructured, Florence-2, Stable Diffusion, etc. download models at runtime that appear to delay operations in the UI. The progress appears in the console logs.
