# GPU Details

Hugging Face type models and LLaMa.cpp models are supported via CUDA on linux and via MPS on MACOS. 

To run in ChatBot mode using bitsandbytes in 8-bit, do:
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

##### AutoGPTQ

To support [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) models, run:
```bash
pip install auto-gptq[triton]
```
although to avoid building the package you can run the [specific version](https://github.com/PanQiWei/AutoGPTQ/releases), e.g.
```bash
pip install https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.3.0/auto_gptq-0.3.0+cu118-cp310-cp310-linux_x86_64.whl
```
However, if one sees issues like `CUDA extension not installed.` mentioned during loading of model, need to recompile,
because, otherwise, the generation will be much slower even if uses GPU.  If you have CUDA 11.8 installed from NVIDIA, run:
```bash
pip uninstall -y auto-gptq ; CUDA_HOME=/usr/local/cuda-11.8 GITHUB_ACTIONS=true pip install auto-gptq --no-cache-dir
```
If one used conda cudatoolkit:
```bash
conda install -c conda-forge cudatoolkit-dev
```
then use that location instead:
```bash
pip uninstall -y auto-gptq ; CUDA_HOME=$CONDA_PREFIX GITHUB_ACTIONS=true pip install auto-gptq --no-cache-dir
```
An example with AutoGPTQ is:
```bash
python generate.py --base_model=TheBloke/Nous-Hermes-13B-GPTQ --score_model=None --load_gptq=nous-hermes-13b-GPTQ-4bit-128g.no-act.order --use_safetensors=True --prompt_type=instruct --langchain_mode=UserData
```
This will use about 9800MB.  You can also add `--hf_embedding_model=sentence-transformers/all-MiniLM-L6-v2` to save some memory on embedding to reach 9340MB.

For LLaMa2 70B model quantized in 4-bit AutoGPTQ, can run:
```bash
CUDA_VISIBLE_DEVICES=0 python generate.py --base_model=Llama-2-70B-chat-GPTQ --load_gptq="gptq_model-4bit--1g" --use_safetensors=True --prompt_type=llama2 --save_dir='70bgptq4bit`
```
which gives about 12 tokens/sec.  For 7b run:
```bash
python generate.py --base_model=TheBloke/Llama-2-7b-Chat-GPTQ --load_gptq="gptq_model-4bit-128g" --use_safetensors=True --prompt_type=llama2 --save_dir='7bgptq4bit`
```
For full 16-bit with 16k context across all GPUs:
```bash
pip install transformers==4.31.0  # breaks load_in_8bit=True in some cases (https://github.com/huggingface/transformers/issues/25026)
python generate.py --base_model=meta-llama/Llama-2-70b-chat-hf --prompt_type=llama2 --rope_scaling="{'type': 'linear', 'factor': 4}" --use_gpu_id=False --save_dir=savemeta70b
```
and running on 4xA6000 gives about 4tokens/sec consuming about 35GB per GPU of 4 GPUs when idle.
Currently, Hugging Face transformers does not support GPTQ directly except in text-generation-inference (TGI) server, but TGI does not support RoPE scaling.  Also, vLLM supports LLaMa2 and AutoGPTQ but not RoPE scaling.  Only exllama supports AutoGPTQ with RoPE scaling.

##### exllama

Currently, only [exllama](https://github.com/turboderp/exllama) supports AutoGPTQ with RoPE scaling.  To install run:
```bash
pip uninstall -y exllama ; pip install https://github.com/jllllll/exllama/releases/download/0.0.7/exllama-0.0.7+cu118-cp310-cp310-linux_x86_64.whl
```
And then run with RoPE scaling the LLaMa-2 7B model for 16k context:
```bash
python generate.py --base_model=TheBloke/Llama-2-7b-Chat-GPTQ --load_gptq="gptq_model-4bit-128g" --use_safetensors=True --prompt_type=llama2 --save_dir='7bgptq4bit' --load_exllama=True --revision=gptq-4bit-32g-actorder_True --rope_scaling="{'alpha_value':4}"
```
which shows how to control `alpha_value` and the `revision` for a given model on [TheBloke/Llama-2-7b-Chat-GPTQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ).  Be careful as setting `alpha_value` higher consumes substantially more GPU memory.  Also, some models have wrong config values for `max_position_embeddings` or `max_sequence_length`, and we try to fix those for LLaMa2 if `llama-2` appears in the lower-case version of the model name.
Another type of model is
```bash
python generate.py --base_model=TheBloke/Nous-Hermes-Llama2-GPTQ --load_gptq="gptq_model-4bit-128g" --use_safetensors=True --prompt_type=wizard2 --save_dir='7bgptq4bit' --load_exllama=True --revision=gptq-4bit-32g-actorder_True --rope_scaling="{'alpha_value':4}"
```
and note the different `prompt_type`.

For LLaMa.cpp on GPU run:
```bash
python generate.py --base_model='llama' --prompt_type=wizard2 --score_model=None --langchain_mode='UserData' --user_path=user_path
```
and ensure output shows:
```text
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
```
