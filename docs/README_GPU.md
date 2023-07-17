### GPU

GPU via CUDA is supported via Hugging Face type models and LLaMa.cpp models.

#### Google Colab

A Google Colab version of a 3B GPU model is at:

[![](https://colab.research.google.com/assets/colab-badge.svg) h2oGPT GPU](https://colab.research.google.com/drive/143-KFHs2iCqXTQLI2pFCDiR69z0dR8iE?usp=sharing)

A local copy of that GPU Google Colab is [h2oGPT_GPU.ipynb](h2oGPT_GPU.ipynb).

---

#### GPU (CUDA)

For help installing cuda toolkit, see [CUDA Toolkit](INSTALL.md#installing-cuda-toolkit).

```bash
git clone https://github.com/h2oai/h2ogpt.git
cd h2ogpt
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r reqs_optional/requirements_optional_langchain.txt
pip install -r reqs_optional/requirements_optional_gpt4all.txt
pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt
pip install -r reqs_optional/requirements_optional_langchain.urls.txt
# Optional: support docx, pptx, ArXiv, etc.
sudo apt-get install -y libmagic-dev poppler-utils tesseract-ocr libreoffice
# Optional: for supporting unstructured package
python -m nltk.downloader all
```
then check that can see CUDA from Torch:
```python
import torch
print(torch.cuda.is_available())
```
should print True.

To support [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) models, run:
```bash
pip install auto-gptq[triton]
```
although to avoid building the package you can run the [specific version](https://github.com/PanQiWei/AutoGPTQ/releases), e.g.
```bash
pip install https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.2.2/auto_gptq-0.2.2+cu118-cp310-cp310-linux_x86_64.whl
```
However, if one sees issues like `CUDA extension not installed.` mentioned during loading of model, need to recompile,
because, otherwise, the generation will be much slower even if uses GPU.  If you have CUDA 11.7 installed from NVIDIA, run:
```bash
pip uninstall -y auto-gptq ; CUDA_HOME=/usr/local/cuda-11.8 GITHUB_ACTIONS=true pip install auto-gptq --no-cache-dir
```
or use cuda-11.8 if one has that installed, etc.  If one used conda cudatoolkit:
```bash
conda install -c conda-forge cudatoolkit-dev
```
then use that location instead:
```bash
pip uninstall -y auto-gptq ; CUDA_HOME=$CONDA_PREFIX GITHUB_ACTIONS=true pip install auto-gptq --no-cache-dir
```

To run in ChatBot mode, do:
```bash
python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b --load_8bit=True
```
Then point browser at http://0.0.0.0:7860 (linux) or http://localhost:7860 (windows/mac) or the public live URL printed by the server (disable shared link with `--share=False`).  For 4-bit or 8-bit support, older GPUs may require older bitsandbytes installed as `pip uninstall bitsandbytes -y ; pip install bitsandbytes==0.38.1`.  For production uses, we recommend at least the 12B model, ran as:
```bash
python generate.py --base_model=h2oai/h2ogpt-oasst1-512-12b --load_8bit=True
```
and one can use `--h2ocolors=False` to get soft blue-gray colors instead of H2O.ai colors.  [Here](FAQ.md#what-envs-can-i-pass-to-control-h2ogpt) is a list of environment variables that can control some things in `generate.py`.

Note if you download the model yourself and point `--base_model` to that location, you'll need to specify the prompt_type as well by running:
```bash
python generate.py --base_model=<user path> --load_8bit=True --prompt_type=human_bot
```
for some user path `<user path>` and the `prompt_type` must match the model or a new version created in `prompter.py` or added in UI/CLI via `prompt_dict`.

For quickly using a private document collection for Q/A, place documents (PDFs, text, etc.) into a folder called `user_path` and run
```bash
python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b  --load_8bit=True --langchain_mode=UserData --user_path=user_path
```
For more details about document Q/A, see [LangChain Readme](README_LangChain.md).

For 4-bit support, when running generate pass `--load_4bit=True`, which is only supported for certain [architectures](https://github.com/huggingface/peft#models-support-matrix) like GPT-NeoX-20B, GPT-J, LLaMa, etc.

Any other instruct-tuned base models can be used, including non-h2oGPT ones.  [Larger models require more GPU memory](FAQ.md#larger-models-require-more-gpu-memory).

---

#### AutoGPTQ

An example with AutoGPTQ is:
```bash
python generate.py --base_model=TheBloke/Nous-Hermes-13B-GPTQ --score_model=None --load_gptq=nous-hermes-13b-GPTQ-4bit-128g.no-act.order --use_safetensors=True --prompt_type=instruct --langchain_mode=MyData
```
This will use about 9800MB.  You can also add `--hf_embedding_model=sentence-transformers/all-MiniLM-L6-v2` to save some memory on embedding to reach 9340MB.

---

#### GPU with LLaMa

* Install langchain, and GPT4All, and python LLaMa dependencies:
```bash
pip install -r reqs_optional/requirements_optional_langchain.txt
pip install -r reqs_optional/requirements_optional_gpt4all.txt
```
then compile llama-cpp-python with CUDA support:
```bash
conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit  # maybe optional
pip uninstall -y llama-cpp-python
export LLAMA_CUBLAS=1
export CMAKE_ARGS=-DLLAMA_CUBLAS=on
export FORCE_CMAKE=1
export CUDA_HOME=$HOME/miniconda3/envs/h2ogpt
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.68 --no-cache-dir --verbose
```
and uncomment `# n_gpu_layers=20` in `.env_gpt4all`, one can try also `40` instead of `20`.  If one sees `/usr/bin/nvcc` mentioned in errors, that file needs to be removed as would likely conflict with version installed for conda.  Then run:
```bash
python generate.py --base_model='llama' --prompt_type=wizard2 --score_model=None --langchain_mode='UserData' --user_path=user_path
```
when loading you should see something like:
```text
Using Model llama
Prep: persist_directory=db_dir_UserData exists, user_path=user_path passed, adding any changed or new documents
load INSTRUCTOR_Transformer
max_seq_length  512
0it [00:00, ?it/s]
0it [00:00, ?it/s]
Loaded 0 sources for potentially adding to UserData
ggml_init_cublas: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090 Ti
  Device 1: NVIDIA GeForce RTX 2080
llama.cpp: loading model from WizardLM-7B-uncensored.ggmlv3.q8_0.bin
llama_model_load_internal: format     = ggjt v3 (latest)
llama_model_load_internal: n_vocab    = 32001
llama_model_load_internal: n_ctx      = 1792
llama_model_load_internal: n_embd     = 4096
llama_model_load_internal: n_mult     = 256
llama_model_load_internal: n_head     = 32
llama_model_load_internal: n_layer    = 32
llama_model_load_internal: n_rot      = 128
llama_model_load_internal: ftype      = 7 (mostly Q8_0)
llama_model_load_internal: n_ff       = 11008
llama_model_load_internal: model size = 7B
llama_model_load_internal: ggml ctx size =    0.08 MB
llama_model_load_internal: using CUDA for GPU acceleration
ggml_cuda_set_main_device: using device 0 (NVIDIA GeForce RTX 3090 Ti) as main device
llama_model_load_internal: mem required  = 4518.85 MB (+ 1026.00 MB per state)
llama_model_load_internal: allocating batch_size x (512 kB + n_ctx x 128 B) = 368 MB VRAM for the scratch buffer
llama_model_load_internal: offloading 20 repeating layers to GPU
llama_model_load_internal: offloaded 20/35 layers to GPU
llama_model_load_internal: total VRAM used: 4470 MB
llama_new_context_with_model: kv self size  =  896.00 MB
AVX = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | VSX = 0 | 
Model {'base_model': 'llama', 'tokenizer_base_model': '', 'lora_weights': '', 'inference_server': '', 'prompt_type': 'wizard2', 'prompt_dict': {'promptA': 'Below is an instruction that describes a task. Write a response that appropriately completes the request.', 'promptB': 'Below is an instruction that describes a task. Write a response that appropriately completes the request.', 'PreInstruct': '\n### Instruction:\n', 'PreInput': None, 'PreResponse': '\n### Response:\n', 'terminate_response': ['\n### Response:\n'], 'chat_sep': '\n', 'chat_turn_sep': '\n', 'humanstr': '\n### Instruction:\n', 'botstr': '\n### Response:\n', 'generates_leading_space': False}}
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://1ccb24d03273a3d085.gradio.live
```
and GPU usage when using.  Note that once `llama-cpp-python` is compiled to support CUDA, it no longer works for CPU mode,
so one would have to reinstall it without the above options to recovers CPU mode or have a separate h2oGPT env for CPU mode.
