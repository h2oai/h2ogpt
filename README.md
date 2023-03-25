## H2O LLM Prototyping Playground

Goal is to create 100% permissive MIT/ApacheV2 LLM model that is useful for ChatGPT usecases.

Training code is based on [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/), but all models will be fully open source.

No OpenAI-based Alpaca fine-tuning data will be left.

Final result will be committed to [H2OGPT](https://github.com/h2oai/h2ogpt/).


### Setup

1. Install python environment

```bash
conda create -n h2ollm
conda activate h2ollm
conda install python=3.10 -y
conda update -n base -c defaults conda
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Install full cuda toolkit, e.g. cuda 12.1 for Ubuntu 22.04 [install cuda coolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)

4. Ensure cuda in path:

```bash
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/" >> ~/.bashrc
echo "CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
echo "export PATH=$PATH:/usr/local/cuda/bin/" >> ~/.bashrc
source ~/.bashrc
```

5. If don't have cuda 11.7 or other specific versions of libraries that bitsandbytes comes with, then must [compile bitesandbytes](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md)

E.g. for CUDA 12.1:
```bash
CUDA_VERSION=121 python setup.py install
```

### Plan
Frst truly open source instruct model.
1. Base: Start with fully open source apache 2.0 models EleutherAI--gpt-j-6B and EleutherAI--gpt-neox-20b
2. Construct Prompt: Setup prompt engineering on 6B as-is to convert a sentence into question/answer or command/response format 
3. Open-Source Instruct Data: Convert wiki data into instruct form
4. Fine-tune: LORA fine-tune 6B and 20B using the open-source instruct data
5. Open Data & Model: Submit instruct 6B and 20B on huggingface as first apache 2 model

### Goals
1. Publish on hugging face first fully open-source Apache v2 instruct dataset
2. Publish on hugging face first fully open-source Apache v2 instruct models
3. Demonstrate efficiency of LORA for fast and low-memory fine-tuning

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


Single-node:
```
torchrun finetune.py --base_model='EleutherAI/gpt-j-6B' --data_path=alpaca_data_cleaned.json --run_id=1 --batch_size=128 --micro_batch_size=16 
```
