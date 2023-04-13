## h2oGPT - The world's best open source GPT

### Why H2O.ai?

Our [Makers](https://h2o.ai/company/team/) at [H2O.ai](https://h2o.ai) have built several world-class Machine Learning, Deep Learning and AI platforms:
  - #1 open-source machine learning platform for the enterprise [H2O-3](https://github.com/h2oai/h2o-3)
  - The world's best AutoML (Automatic Machine Learning) with [H2O Driverless AI](https://h2o.ai/platform/ai-cloud/make/h2o-driverless-ai/)
  - No-Code Deep Learning with [H2O Hydrogen Torch](https://h2o.ai/platform/ai-cloud/make/hydrogen-torch/)
  - Document Processing with Deep Learning in [Document AI](https://h2o.ai/platform/ai-cloud/make/document-ai/)

We also built platforms for deployment and monitoring, and for data governance:
  - [H2O MLOps](https://h2o.ai/platform/ai-cloud/operate/h2o-mlops/) to deploy models at scale
  - Low-Code AI App Development Frameworks [Wave](https://wave.h2o.ai/) and [Nitro](https://nitro.h2o.ai/)
  - [H2O Feature Store](https://h2o.ai/platform/ai-cloud/make/feature-store/) in collaboration with AT&T

Many of our customers are creating models and deploying them at enterprise-wide at scale in the [H2O AI Cloud](https://h2o.ai/platform/ai-cloud/):
  - Multi-Cloud or on Premises
  - [Managed Cloud (SaaS)](https://h2o.ai/platform/ai-cloud/managed)
  - [Hybrid Cloud](https://h2o.ai/platform/ai-cloud/hybrid)
  - [AI Appstore](https://docs.h2o.ai/h2o-ai-cloud/)

We are proud to have over 20 (of the world’s 300+) total [Kaggle Grandmasters](https://h2o.ai/company/team/kaggle-grandmasters/) call H2O home, including three Kaggle Grandmasters who have made it to world #1.

Our goal is to create Apache-2.0 licensed LLM models that are useful for ChatGPT use cases.

### Goals

1. Curate high-quality open-source instruct data for fine-tuning
2. Develop robust and performant training/inference code using best practices
3. Create state-of-the-art open-source fine-tuned LLM models
4. Democratize knowledge about GPT models
5. Get community contributions

### Plan

1. Start with fully open source Apache 2.0 models EleutherAI--gpt-j-6B, EleutherAI--gpt-neox-20b,
   GPT-NeoXT-Chat-Base-20B, etc.
2. Open-Source Instruct Data: Collect and curate high quality data with human <-> bot interaction
3. Prompt Engineering: Convert plain text into question/answer or command/response format
4. Fine-tune LLM models using 8-bit and LoRA for speed and memory efficiency
5. Create GUIs, validation tools, APIs for integration with other software
6. Submit data and model to HuggingFace with 100% permissive license
7. Collect feedback from community


Original training code is based on [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/).

All training data will be based on open-source permissive data. No Alpaca, no LLama, no OpenAI.

All models will be published to HuggingFace.

### Installation for Developers

Follow the [installation instructions](INSTALL.md) to create a development environment for training and generation.

#### Create instruct dataset

```bash
pytest create_data.py::test_get_OIG_data
pytest create_data.py::test_merge_shuffle_OIG_data
```

#### Perform fine-tuning on your data

Fine-tune on a single node with NVIDIA GPUs A6000/A6000Ada/A100/H100, needs 48GB of GPU memory per GPU for default settings.
For GPUs with 24GB of memory, need to set `--micro_batch_size=1` and `--batch_size=$NGPUS` below.
```
export NGPUS=`nvidia-smi -L | wc -l`
torchrun --nproc_per_node=$NGPUS finetune.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --data_path=merged_shuffled_OIG_87f6a1e788.json --prompt_type=plain --output_dir=my_finetuned_weights
```
this will download the model, load the data, and generate an output directory `my_finetuned_weights` containing the fine-tuned state.


#### Start ChatBot

Start a chatbot, also requires 48GB GPU. Likely run out of memory on 24GB GPUs, but can work with lower values for `--chat_history`.
```
torchrun generate.py --load_8bit=True --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --lora_weights=my_finetuned_weights --prompt_type=human_bot
```
this will download the foundation model, our fine-tuned lora_weights, and open up a GUI with text generation input/output.


In case you get peer to peer related errors on non-homogeneous GPU systems, set this env var:
```
export NCCL_P2P_LEVEL=LOC
```


### Containerized Installation for Inference on Linux GPU Servers

1. Build the container image:

```bash
docker build -t h2o-llm .
```

2. Run the container (you can also use `finetune.py` and all of its parameters as shown above for training):

```bash
docker run --runtime=nvidia --shm-size=64g -p 7860:7860 -v ${HOME}/.cache:/root/.cache --rm h2o-llm -it generate.py \
    --load_8bit=True --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --lora_weights=my_finetuned_weights --prompt_type=human_bot
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


### Code to consider including
[flan-alpaca](https://github.com/declare-lab/flan-alpaca)<br />
[text-generation-webui](https://github.com/oobabooga/text-generation-webui)<br />
[minimal-llama](https://github.com/zphang/minimal-llama/)<br />
[finetune GPT-NeoX](https://nn.labml.ai/neox/samples/finetune.html)<br />
[GPTQ-for_LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/compare/cuda...Digitous:GPTQ-for-GPT-NeoX:main)<br />
[OpenChatKit on multi-GPU](https://github.com/togethercomputer/OpenChatKit/issues/20)<br />
[Non-Causal LLM](https://huggingface.co/docs/transformers/main/en/model_doc/gptj#transformers.GPTJForSequenceClassification)<br />
[OpenChatKit_Offload](https://github.com/togethercomputer/OpenChatKit/commit/148b5745a57a6059231178c41859ecb09164c157)<br />
[Flan-alpaca](https://github.com/declare-lab/flan-alpaca/blob/main/training.py)<br />

### Help

[FAQs](FAQ.md)

### More links, context, competitors, models, datasets

[Links](LINKS.md)
