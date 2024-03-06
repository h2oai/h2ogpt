# Python Wheel

### Building wheel for your platform

```bash
git clone https://github.com/h2oai/h2ogpt.git
cd h2ogpt
python setup.py bdist_wheel
```
Note that Coqui TTS is not installed due to issues with librosa.  Use one-click, docker, or manual install scripts to get Coqui TTS.  Also, AMD ROC and others are supported, but need manual edits to the `reqs_optional/requirements_optional_llamacpp_gpt4all.txt` file to select it and comment out others.

Install in fresh env, avoiding being inside h2ogpt directory or a directory where it is a sub directory.  For CUDA GPU do:
```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121 https://huggingface.github.io/autogptq-index/whl/cu121"
set CMAKE_ARGS=-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all
set LLAMA_CUBLAS=1
set FORCE_CMAKE=1
```
for the cmake args, choose e llama_cpp_python ARGS for your system according to [llama_cpp_python backend documentation](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends).  Note for some reason things will fail with llama_cpp_python if don't add all cuda arches, and building with all those arches does take some time.
Then pip install:
```bash
pip install <h2ogpt_path>/dist/h2ogpt-0.1.0-py3-none-any.whl[cuda]
pip install flash-attn==2.4.2
```
and pick your CUDA version, where `<h2ogpt_path>` is the relative path to the h2ogpt repo where the wheel was built. Replace `0.1.0` with actual version built if more than one.

For non CUDA cases, e.g. CPU, Metal M1/M2 do:
```bash
pip install <h2ogpt_path>/dist/h2ogpt-0.1.0-py3-none-any.whl[cpu]
```

A wheel online is provided for this and can be installed as follows:
First, if using conda, DocTR can be enabled using above installation if first doing:
```bash
conda install weasyprint pygobject -c conda-forge -y
```
second run:
```bash
export CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all"
export CUDA_HOME=/usr/local/cuda-12.1
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121 https://huggingface.github.io/autogptq-index/whl/cu121"
pip install h2ogpt==0.2.0[cuda] --index-url https://downloads.h2ogpt.h2o.ai --extra-index-url https://pypi.org/simple --no-cache
pip install flash-attn==2.4.2
```
for CUDA support.  If conda and those packages weren't installed, this would exclude some DocTR support that is provided otherwise also by  docker, one-click installer for windows and mac, or manual windows/linux installers.

## Checks
Once the wheel is built, if you do:
```bash
python -m pip check
```
and you should see:
```text
No broken requirements found.
```

## PyPI

For PyPI, we use a more limited set of packages built like:
```bash
PYPI=1 python setup.py bdist_wheel
```
which can be installed with basic CUDA support like:
```bash
# For other GPUs etc. see: https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends
# required for PyPi wheels that do not allow URLs, so uses generic llama_cpp_python package:
export CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all"
export CUDA_HOME=/usr/local/cuda-12.1
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121 https://huggingface.github.io/autogptq-index/whl/cu121"
# below [cuda] assumes CUDA 12.1 for some packages like AutoAWQ etc.
pip install h2ogpt[cuda]
pip install flash-attn==2.4.2
```

## Run

To run h2oGPT, do, e.g.
```bash
CUDA_VISIBLE_DEVICES=0 python -m h2ogpt.generate --base_model=llama
```
or inside python:
```python
from h2ogpt.generate import main
main(base_model='llama')
```
See `src/gen.py` for all documented options one can pass to `main()`.  E.g. to start LLaMa7B:
```python
from h2ogpt.generate import main
main(base_model='meta-llama/Llama-2-7b-chat-hf',
          prompt_type='llama2',
          save_dir='save_gpt7',
          score_model=None,
          max_max_new_tokens=2048,
          max_new_tokens=1024,
          num_async=10,
          top_k_docs=-1)
```

