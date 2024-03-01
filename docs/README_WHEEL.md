# Python Wheel

### Building wheel for your platform

```bash
git clone https://github.com/h2oai/h2ogpt.git
cd h2ogpt
python setup.py bdist_wheel
```
Note that Coqui TTS is not installed due to issues with librosa.  Use one-click, docker, or manual install scripts to get Coqui TTS.  Also, AMD ROC and others are supported, but need manual edits to the `reqs_optional/requirements_optional_gpt4all.txt` file to select it and comment out others.

Install in fresh env, avoiding being inside h2ogpt directory or a directory where it is a sub directory.  For CUDA GPU do:
```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121 https://huggingface.github.io/autogptq-index/whl/cu121"
pip install <h2ogpt_path>/dist/h2ogpt-0.1.0-py3-none-any.whl[gpu]
pip install flash-attn==2.4.2
```
and pick your CUDA version, where `<h2ogpt_path>` is the relative path to the h2ogpt repo where the wheel was built. Replace `0.1.0` with actual version built if more than one.

For non CUDA cases, e.g. CPU, Metal M1/M2 do:
```bash
pip install <h2ogpt_path>/dist/h2ogpt-0.1.0-py3-none-any.whl[cpu]
```

## Checks
Once the wheel is built, if you do:
```bash
python -m pip check
```
and you should see:
```text
No broken requirements found.
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

