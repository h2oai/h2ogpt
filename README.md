## h2oGPT - The world's best open source GPT

Our goal is to make the world's best open source GPT!



https://user-images.githubusercontent.com/6147661/232924684-6c0e2dfb-2f24-4098-848a-c3e4396f29f6.mov

### Current state

- Open-source repository with **fully permissive, commercially usable code, data and models**
- Code for preparing **large open-source datasets** as instruction datasets for fine-tuning of large language models (LLMs), including prompt engineering
- Code for **fine-tuning large language models** (currently up to 20B parameters) on commodity hardware and enterprise GPU servers (single or multi node)
- Code to **run a chatbot** on a GPU server, with shareable end-point with Python client API
- Code to evaluate and compare the **performance** of fine-tuned LLMs

All open-source datasets and models are posted on [H2O.ai's Hugging Face page](https://huggingface.co/h2oai/).

Also check out [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio) for our no-code LLM fine-tuning framework!

### Roadmap items

- Integration of code and resulting LLMs with downstream applications and low/no-code platforms
- Complement h2oGPT chatbot with search and other APIs
- High-performance distributed training of larger models on trillion tokens
- Improve code completion, reasoning, mathematics, factual correctness, hallucinations and avoid repetitions

### Chat with h2oGPT

Try h2oGPT now on [Hugging Face Spaces](https://huggingface.co/spaces/h2oai/h2ogpt-oasst1-256-6.9b-hosted)!

```bash
git clone https://github.com/h2oai/h2ogpt.git
cd h2ogpt
pip install -r requirements.txt
python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-256-6.9b
```
and then use browser at http://0.0.0.0:7860 or the public live URL printed by the server.

For help installing a Python 3.10 environment or CUDA toolkit, see the [installation instructions](INSTALL.md)

You can also use [Docker](INSTALL-DOCKER.md#containerized-installation-for-inference-on-linux-gpu-servers) for inference.

#### Larger models require more GPU memory

Depending on available GPU memory, you can load differently sized models. For multiple GPUs, automatic sharding can be enabled with `--infer_devices=False`, but that is disabled by default since cuda:x cuda:y mismatches can occur.

For GPUs with at least 24GB of memory, we recommend:
```bash
python generate.py --base_model=h2oai/h2ogpt-oasst1-512-12b
```
For GPUs with at least 48GB of memory, we recommend:
```bash
python generate.py --base_model=h2oai/h2ogpt-oasst1-512-20b
```
The numbers `256` and `512` in the model names indicate the cutoff lengths (in tokens) used for fine-tuning. Shorter values generally result in faster training and more focus on the last part of the provided input text (consisting of prompt and answer).

More information about the models can be found on [H2O.ai's Hugging Face page](https://huggingface.co/h2oai/).

### Why H2O.ai?

Our [Makers](https://h2o.ai/company/team/) at [H2O.ai](https://h2o.ai) have built several world-class Machine Learning, Deep Learning and AI platforms:
  - #1 open-source machine learning platform for the enterprise [H2O-3](https://github.com/h2oai/h2o-3)
  - The world's best AutoML (Automatic Machine Learning) with [H2O Driverless AI](https://h2o.ai/platform/ai-cloud/make/h2o-driverless-ai/)
  - No-Code Deep Learning with [H2O Hydrogen Torch](https://h2o.ai/platform/ai-cloud/make/hydrogen-torch/)
  - Document Processing with Deep Learning in [Document AI](https://h2o.ai/platform/ai-cloud/make/document-ai/)

We also built platforms for deployment and monitoring, and for data wrangling and governance:
  - [H2O MLOps](https://h2o.ai/platform/ai-cloud/operate/h2o-mlops/) to deploy and monitor models at scale
  - [H2O Feature Store](https://h2o.ai/platform/ai-cloud/make/feature-store/) in collaboration with AT&T
  - Open-source Low-Code AI App Development Frameworks [Wave](https://wave.h2o.ai/) and [Nitro](https://nitro.h2o.ai/)
  - Open-source Python [datatable](https://github.com/h2oai/datatable/) (the engine for H2O Driverless AI feature engineering)

Many of our customers are creating models and deploying them enterprise-wide and at scale in the [H2O AI Cloud](https://h2o.ai/platform/ai-cloud/):
  - Multi-Cloud or on Premises
  - [Managed Cloud (SaaS)](https://h2o.ai/platform/ai-cloud/managed)
  - [Hybrid Cloud](https://h2o.ai/platform/ai-cloud/hybrid)
  - [AI Appstore](https://docs.h2o.ai/h2o-ai-cloud/)

We are proud to have over 25 (of the world's 280) [Kaggle Grandmasters](https://h2o.ai/company/team/kaggle-grandmasters/) call H2O home, including three Kaggle Grandmasters who have made it to world #1.

### Development

- Follow the [installation instructions](INSTALL.md) to create a development environment for training and generation.
- Follow the [fine-tuning instructions](FINETUNE.md) to fine-tune any LLM models on your data.
- Follow the [Docker instructions](INSTALL-DOCKER.md) to create a container for deployment.

### Help

[FAQs](FAQ.md)

### More links, context, competitors, models, datasets

[Links](LINKS.md)

Original training code is based on [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/).
