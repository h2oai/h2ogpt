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

4. Ensure cuda in path:

```bash
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/" >> ~/.bashrc
echo "CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
echo "export PATH=$PATH:/usr/local/cuda/bin/" >> ~/.bashrc
source ~/.bashrc  # or source ~/.bashrc.conda
conda activate h2ollm
```

5. [compile bitesandbytes](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md)

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
WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1" torchrun --nnodes=2 --master_addr="10.10.10.2" --node_rank=0 --nproc_per_node=2 --master_port=1234 finetune.py --llama_type=False --data_path=alpaca_data_cleaned.json --run_id=0 --base_model='EleutherAI/gpt-j-6B'

WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1" torchrun --nnodes=2 --master_addr="10.10.10.2" --node_rank=1 --nproc_per_node=2 --master_port=1234 finetune.py --llama_type=False --data_path=alpaca_data_cleaned.json --run_id=0 --base_model='EleutherAI/gpt-j-6B'
```

Fine-tune using 2 24GB GPUs to split up a 30B model:
```
WORLD_SIZE=2 python finetune.py --data_path=alpaca_data_cleaned.json --base_model="decapoda-research/llama-30b-hf" --llama_type=True --ddp=False
```

Fine-tune previously saved model (running `export_hf_checkpoint.py`):
```
WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1" torchrun --nnodes=2 --master_addr="10.10.10.2" --node_rank=0 --nproc_per_node=2 --master_port=1234 finetune.py --num_epochs=2 --micro_batch_size=8 --llama_type=False --data_path=alpaca_data_cleaned.json --run_id=3 --base_model='gpt-j-6B.DAIdocs' --tokenizer_base_model='EleutherAI/gpt-j-6B' --output_dir=lora_6B.DAIdocs &> 3.node0.log

WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1" torchrun --nnodes=2 --master_addr="10.10.10.2" --node_rank=1 --nproc_per_node=2 --master_port=1234 finetune.py --num_epochs=2 --micro_batch_size=8 --llama_type=False --data_path=alpaca_data_cleaned.json --run_id=3 --base_model='gpt-j-6B.DAIdocs' --tokenizer_base_model='EleutherAI/gpt-j-6B' --output_dir=lora_6B.DAIdocs &> 3.node1.log
```

Generate on single GPU on single node:
```
torchrun generate.py --base_model='EleutherAI/gpt-j-6B' --lora_weights=lora-alpaca
```
this will download the foundation model, our fine-tuned lora_weights, and open up a GUI with text generation input/output.


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
[minimal-llama](https://github.com/zphang/minimal-llama/)

### Some open source models:
[GPT-NeoXT-Chat-Base-20B](https://huggingface.co/togethercomputer/GPT-NeoXT-Chat-Base-20B/tree/main)<br />
[Pythia-6.9B](https://huggingface.co/EleutherAI/pythia-6.9b)<br />
[Pythia-12B](https://huggingface.co/EleutherAI/neox-ckpt-pythia-12b)<br />
[Flan-T5-XXL](https://huggingface.co/google/flan-t5-xxl)<br />
[GPT-J-Moderation-6B](https://huggingface.co/togethercomputer/GPT-JT-Moderation-6B)
[OIG safety models](https://laion.ai/blog/oig-dataset/#safety-models)

### Some create commons models that would be interesting to use:
[Galactica-120B](https://huggingface.co/facebook/galactica-120b)<br />
[LLaMa-small-pt](https://huggingface.co/decapoda-research/llama-smallint-pt)<br />
[LLaMa-64b-4bit](https://huggingface.co/maderix/llama-65b-4bit/tree/main)<br />

### Papers/Repos
[LLaMa](https://arxiv.org/abs/2302.13971)<br />
[GLM-130B](https://github.com/THUDM/GLM-130B)<br />
[RWKV RNN](https://github.com/BlinkDL/RWKV-LM)<br />
[Toolformer](https://arxiv.org/abs/2302.04761)<br />
[GPTQ](https://github.com/qwopqwop200/GPTQ-for-LLaMa)<br />
[Retro](https://www.deepmind.com/publications/improving-language-models-by-retrieving-from-trillions-of-tokens)<br />
[Clinical_outperforms](https://arxiv.org/abs/2302.08091)<br />
[Chain-Of-Thought](https://github.com/amazon-science/mm-cot)

### Other projects:
[alpaca-lora](https://github.com/h2oai/alpaca-lora)<br />
[alpaca.http](https://github.com/Nuked88/alpaca.http)<br />
[langchain](https://python.langchain.com/en/latest/)<br />
[cohere](https://cohere.io/)<br />
[coherefinetune](https://docs.cohere.ai/reference/finetune)<br />
[langchain+pinecone](https://www.youtube.com/watch?v=nMniwlGyX-c)<br />
[chatgpt-retrieval-pllugin](https://github.com/openai/chatgpt-retrieval-plugin)<br />
[subtl.ai docs search on private docs](https://www.subtl.ai/)<br />
[gertel](https://gretel.ai/)<br />
[alpaca_lora_4bit](https://github.com/johnsmith0031/alpaca_lora_4bit)<br />
[alpaca_lora_4bit_readme](https://github.com/s4rduk4r/alpaca_lora_4bit_readme)<br />
[code alpaca](https://github.com/sahil280114/codealpaca)<br />
[serge](https://github.com/nsarrazin/serge)<br />
[BlinkDL](https://huggingface.co/spaces/BlinkDL/ChatRWKV-gradio)<br />
[MosaicCM](https://github.com/mosaicml/examples#large-language-models-llms)<br />
[OpenAI Plugins](https://openai.com/blog/chatgpt-plugins)<br />
[GPT3.5-Turbo-PGVector](https://github.com/gannonh/gpt3.5-turbo-pgvector)<br />
[DocsBotAI](https://docsbot.ai/)<br />
[Perplexity](https://www.perplexity.ai/)<br />
[VoiceFlow](https://www.voiceflow.com/)<br />
[LLaMa-Adapter](https://github.com/ZrrSkywalker/LLaMA-Adapter)<br />
[llama-index](https://github.com/jerryjliu/llama_index)<br />
[minimal-llama](https://github.com/zphang/minimal-llama/)<br />
[llama.cpp](https://github.com/ggerganov/llama.cpp)<br />
[lamma.cpp more](https://til.simonwillison.net/llms/llama-7b-m2)<br />

### Apache2/etc. Data
[OIG 43M instructions](https://laion.ai/blog/oig-dataset/) [direct HF link](https://huggingface.co/datasets/laion/OIG)

### non-commerical Data
[GPT-3 based Alpaca Cleaned](https://github.com/gururise/AlpacaDataCleaned)

### Throttle GPUs in case of reset/reboot

```bash
(alpaca) jon@gpu:/data/jon/alpaca-lora$ sudo nvidia-smi -pl 250
Power limit for GPU 00000000:3B:00.0 was set to 250.00 W from 300.00 W.
Power limit for GPU 00000000:5E:00.0 was set to 250.00 W from 300.00 W.
Power limit for GPU 00000000:86:00.0 was set to 250.00 W from 300.00 W.
Power limit for GPU 00000000:AF:00.0 was set to 250.00 W from 300.00 W.
All done.
```

### Wiki

```python
>>> from datasets import load_dataset
>>> wk = load_dataset("wikipedia", "20220301.en")
>>> wk
DatasetDict({
    train: Dataset({
        features: ['id', 'url', 'title', 'text'],
        num_rows: 6458670
    })
})
>>> sentences = ".".join(wk['train'][0]['text'].split('.')[0:2])
'Anarchism is a political philosophy and movement that is sceptical of authority and rejects all involuntary, coercive forms of hierarchy. Anarchism calls for the abolition of the state, which it holds to be unnecessary, undesirable, and harmful'
>>> 
```

### Fine-Tune

TODO: replace `alpaca_data_cleaned.json` with open-source/bootstrapped approach


