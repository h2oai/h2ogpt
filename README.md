## H2O LLM Prototyping Playground

Goal is to create 100% permissive MIT/ApacheV2 LLM model that is useful for ChatGPT usecases.

Training code is based on [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/), but all models will be fully open source.

No OpenAI-based Alpaca fine-tuning data will be left.

Final result will be committed to [H2OGPT](https://github.com/h2oai/h2ogpt/).


### Install

```py
virtualenv -p python3.10 env
pip install -r requirements.txt
```

### Fine-Tune

TODO: replace `alpaca_data_cleaned.json` with open-source/bootstrapped approach


Single-node:
```
torchrun finetune.py --base_model='EleutherAI/gpt-j-6B' --data_path=alpaca_data_cleaned.json --run_id=1 --batch_size=128 --micro_batch_size=16 
```
