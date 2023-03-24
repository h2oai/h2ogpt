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

i.e.
```bash
CUDA_VERSION=121 python setup.py install
```


### Fine-Tune

TODO: replace `alpaca_data_cleaned.json` with open-source/bootstrapped approach


Single-node:
```
torchrun finetune.py --base_model='EleutherAI/gpt-j-6B' --data_path=alpaca_data_cleaned.json --run_id=1 --batch_size=128 --micro_batch_size=16 
```
