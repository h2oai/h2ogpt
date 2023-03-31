## H2O LLM Prototyping Playground

Goal is to create 100% permissive MIT/ApacheV2 LLM model that is useful for ChatGPT usecases.

Training code is based on [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/), but all models will be fully open source.

No OpenAI-based Alpaca fine-tuning data will be left.

Final result will be committed to [H2OGPT](https://github.com/h2oai/h2ogpt/).


### Setup

1. Install python environment

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
bash ./Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
# follow license agreement and add to bash if required
source ~/.bashrc
# For more control: Copy block it added to .bashrc, put into ~/.bashrc.conda, then source ~/.bashrc.conda
conda create -n h2ollm
conda activate h2ollm
conda install mamba -n base -c conda-forge
conda install python=3.10 -y
conda update -n base -c defaults conda
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Install full cuda toolkit, e.g. cuda 12.1 for Ubuntu 22.04 [install cuda coolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) and [CUDNN8](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.8.1/local_installers/12.0/cudnn-local-repo-ubuntu2204-8.8.1.3_1.0-1_amd64.deb/) then reboot.

When it comes time to apt-get install after dpkg, ignore nvidia instructions about specific version, too hard to do, just do:
```bash
sudo apt-get install libcudnn8 libcudnn8-dev libcudnn8-samples
```

4. Ensure cuda in path:

```bash
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/" >> ~/.bashrc
echo "CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
echo "export PATH=$PATH:/usr/local/cuda/bin/" >> ~/.bashrc
source ~/.bashrc  # or source ~/.bashrc.conda
conda activate h2ollm
```

5. Compile bitsandbytes [howto src](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md)

E.g. for CUDA 12.1 (for CUDA 11.7, use `CUDA_VERSION=117 make cuda11x` etc.)
```bash
pip uninstall bitsandbytes || true
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes
CUDA_VERSION=121 make cuda12x
CUDA_VERSION=121 python setup.py install
cd ..
```

Fine-tune on single GPU on single node:
```
torchrun finetune.py --base_model='EleutherAI/gpt-j-6B' --data_path=alpaca_data_cleaned.json 
```
this will download the model, load the data, and generate an output directory lora-alpaca.

Fine-tune using 2 nodes with 2 GPUs each:
```
WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1" torchrun --nnodes=2 --master_addr="10.10.10.2" --node_rank=0 --nproc_per_node=2 --master_port=1234 finetune.py --data_path=alpaca_data_cleaned.json --run_id=0 --base_model='EleutherAI/gpt-j-6B'

WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1" torchrun --nnodes=2 --master_addr="10.10.10.2" --node_rank=1 --nproc_per_node=2 --master_port=1234 finetune.py --data_path=alpaca_data_cleaned.json --run_id=0 --base_model='EleutherAI/gpt-j-6B'
```

Fine-tune using 2 24GB GPUs to split up a 30B model:
```
WORLD_SIZE=2 python finetune.py --data_path=alpaca_data_cleaned.json --base_model="decapoda-research/llama-30b-hf" --ddp=False
```

Fine-tune previously saved model (running `export_hf_checkpoint.py`):
```
WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1" torchrun --nnodes=2 --master_addr="10.10.10.2" --node_rank=0 --nproc_per_node=2 --master_port=1234 finetune.py --num_epochs=2 --micro_batch_size=8 --data_path=alpaca_data_cleaned.json --run_id=3 --base_model='gpt-j-6B.DAIdocs' --tokenizer_base_model='EleutherAI/gpt-j-6B' --output_dir=lora_6B.DAIdocs &> 3.node0.log

WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1" torchrun --nnodes=2 --master_addr="10.10.10.2" --node_rank=1 --nproc_per_node=2 --master_port=1234 finetune.py --num_epochs=2 --micro_batch_size=8 --data_path=alpaca_data_cleaned.json --run_id=3 --base_model='gpt-j-6B.DAIdocs' --tokenizer_base_model='EleutherAI/gpt-j-6B' --output_dir=lora_6B.DAIdocs &> 3.node1.log
```

Generate on single GPU on single node:
```
torchrun generate.py --base_model='EleutherAI/gpt-j-6B' --lora_weights=lora-alpaca
```
this will download the foundation model, our fine-tuned lora_weights, and open up a GUI with text generation input/output.


In case you get peer to peer related errors, set this env var:
```
export NCCL_P2P_LEVEL=LOC
```


### Docker Setup & Inference

1. Build the container image:

```bash
docker build -t h2o-llm .
```

2. Run the container (you can also use `finetune.py` and all of its parameters as shown above for training):

```bash
docker run --runtime=nvidia --shm-size=64g -p 7860:7860 -v ${HOME}/.cache:/root/.cache --rm h2o-llm -it generate.py \
    --load_8bit=True --base_model='EleutherAI/gpt-neox-20b' --prompt_type=human_bot
```

3. Open `https://localhost:7860` in the browser

### Docker Compose Setup & Inference

1. (optional) Change desired model and weights under `environment` in the `docker-compose.yml`

2. Build and run the container

```bash
docker-compose up -d --build
```

3. Open `https://localhost:7860` in the browser

4. See logs:

```bash
docker-compose logs -f
```

5. Clean everything up:

```bash
docker-compose down --volumes --rmi all
```


### Plan
Open source instruct model for demoable usecases.
1. Base: Start with fully open source apache 2.0 models EleutherAI--gpt-j-6B, EleutherAI--gpt-neox-20b, 
GPT-NeoXT-Chat-Base-20B, etc. 
2. Construct Prompt: Setup prompt engineering on 6B-20B as-is to convert a sentence into question/answer or command/response format 
3. Open-Source Instruct Data: Convert wiki data into instruct form
4. Fine-tune: LORA fine-tune 6B and 20B using DAI docs
5. Open Data & Model: Submit DAI docs model huggingface
6. Use toolformer approach for external APIs

### Goals
1. Demonstrate fine-tuning working on some existing corpus
2. Demonstrate efficiency of LORA for fast and low-memory fine-tuning


### Code to consider including
[shawwn/llama](https://github.com/shawwn/llama/commit/40d99d329a5e38d85904d3a6519c54e6dd6ee9e1)<br />
[llama PRs](https://github.com/facebookresearch/llama/pulls)<br />
[text-generation-webui](https://github.com/oobabooga/text-generation-webui)<br />
[minimal-llama](https://github.com/zphang/minimal-llama/)<br />
[finetune GPT-NeoX](https://nn.labml.ai/neox/samples/finetune.html)<br />
[GPTQ-for_LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/compare/cuda...Digitous:GPTQ-for-GPT-NeoX:main)<br />
[OpenChatKit on multi-GPU](https://github.com/togethercomputer/OpenChatKit/issues/20)<br />
[Non-Causal LLM](https://huggingface.co/docs/transformers/main/en/model_doc/gptj#transformers.GPTJForSequenceClassification)<br />
[OpenChatKit_Offload](https://github.com/togethercomputer/OpenChatKit/commit/148b5745a57a6059231178c41859ecb09164c157)<br />

### Help

[FAQs](FAQ.md)

### More links, context, competitors, models, datasets

[Links](LINKS.md)
