## h2oGPT - The world's best open source GPT

Our goal is to make the world's best open source GPT!

### Current state

1. Open-source repository with **fully permissive, commercially usable code, data and models**
2. Code for preparing **large open-source datasets** as instruction datasets for fine-tuning of large language models (LLMs), including prompt engineering
3. Code for **fine-tuning large language models** (currently up to 20B parameters) on commodity hardware and enterprise GPU servers (single or multi node)
4. Code to **run a chatbot** on a GPU server, with shareable APIs
5. Code to evaluate and compare the **performance** of fine-tuned LLMs

Best of all, you don't need to worry about contamination with non-permissive licenses like Alpaca, LLama, Vicuna, ShareGPT, Dolly, OpenAssistant's models, etc.

All open-source datasets and models are posted on [H2O.ai's HuggingFace page](https://huggingface.co/h2oai/).

### Roadmap items

1. Integration of code and resulting LLMs with downstream applications and low/no-code platforms
2. High-performance distributed training (using DeepSpeed or similar)
3. Pre-training of foundation models on many billions of tokens
4. State-of-the-art LLM. Improve code completion, reasoning, factual correctness, reduced repetitions, hallucinations and other common issues that most LLMs share.

### Development

Follow the [installation instructions](INSTALL.md) to create a development environment for training and generation.
Follow the [Docker instructions](INSTALL-DOCKER.md) to create a container for deployment.
Follow the [fine-tuning instructions](FINETUNE.md) to fine-tune any LLM models on your data.

### Chat with h2oGPT

To start an h2oGPT chatbot on a 24GB GPU (3090/4090/A6000/A100/H100), run this command:
```bash
torchrun generate.py --load_8bit=True --base_model=h2oai/h2ogpt-oasst1-512-12b --prompt_type=human_bot  # needs 24GB GPU
```

Depending on available GPU memory, you can load differently sized models, it will automatically shard the model across multiple GPUs if needed.
```bash
torchrun generate.py --load_8bit=True --base_model=h2oai/h2ogpt-oig-oasst1-256-6.9b --prompt_type=human_bot  # needs 12GB GPU memory
torchrun generate.py --load_8bit=True --base_model=h2oai/h2ogpt-oasst1-512-12b --prompt_type=human_bot  # needs 24GB GPU memory
torchrun generate.py --load_8bit=True --base_model=h2oai/h2ogpt-oasst1-256-20b --prompt_type=human_bot  # needs 48GB GPU memory
torchrun generate.py --load_8bit=True --base_model=h2oai/h2ogpt-oig-oasst1-256-6.9b --prompt_type=human_bot  # needs 12GB GPU memory
torchrun generate.py --load_8bit=True --base_model=h2oai/h2ogpt-oasst1-512-12b --prompt_type=human_bot  # needs 24GB GPU memory
torchrun generate.py --load_8bit=True --base_model=h2oai/h2ogpt-oasst1-256-20b --prompt_type=human_bot  # needs 48GB GPU memory
```
You can also use [Docker](INSTALL-DOCKER.md#containerized-installation-for-inference-on-linux-gpu-servers) to start an h2oGPT chatbot:
This will download the h2oGPT model and open up a GUI with text generation input/output.

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


### Help

[FAQs](FAQ.md)

### More links, context, competitors, models, datasets

[Links](LINKS.md)

Original training code is based on [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/).
