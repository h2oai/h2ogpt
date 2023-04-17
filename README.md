## h2oGPT - The world's best open source GPT

Come join the movement to make the world's best open source GPT led by H2O.ai!

### Goals

1. Curate high-quality open-source instruct data for fine-tuning
2. Develop robust and performant training/inference code using best practices
3. Create state-of-the-art open-source fine-tuned LLM models
4. Democratize knowledge about the creation and usage of LLM models
5. Get community contributions

### Plan

1. Start with fully open source Apache 2.0 models EleutherAI--gpt-j-6B, EleutherAI--gpt-neox-20b,
   GPT-NeoXT-Chat-Base-20B, etc.
2. Open-Source Instruct Data: Collect and curate high quality data with instruction and response interaction sequences
3. Prompt Engineering: Convert plain text into question/answer or command/response format
4. Fine-tune LLM models using 8-bit and LoRA for speed and memory efficiency
5. Create GUIs, validation tools, APIs for integration with other software
6. Submit data and models to HuggingFace with 100% permissive license
7. Collect feedback from community

### What this is

Open-source repository with commercially usable code (Apache v2) to:
1. Generate datasets for fine-tuning LLMs (including prompt engineering)
2. Fine-tune LLMs on commodity hardware and enterprise GPU servers (single or multi node)
3. Run a chatbot on a single GPU server
4. Evaluate the performance of fine-tuned LLMs
5. Create an API for such a chat bot

### What this isn't

1. Pre-training of foundation models on hundreds of billions of tokens or more
2. High-performance distributed training (DeepSpeed etc. are more suitable)
3. State-of-the-art LLM. It's not great at code completion, reasoning, factual correctness. It suffers from hallucinations and other common issues that all LLMs share.
4. End-user applications using LLMs


Original training code is based on [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/).

All training data will be based on open-source permissive data. No Alpaca, no LLama, no OpenAI, no ShareGPT.

All datasets and models will be published to [H2O.ai's HuggingFace page](https://huggingface.co/h2oai/) and H2O.ai's S3 buckets.

### Native Installation (Recommended for Developers)

Follow the [installation instructions](INSTALL.md) to create a development environment for training and generation.

### Containerized Installation using Docker (Recommended for Chatbot Demos)

Follow the [Docker instructions](INSTALL-DOCKER.md) to create a container for deployment.

### Fine-tuning (see how h2oGPT was created)

Follow the [fine-tuning instructions](FINETUNE.md) to fine-tune any LLM models on your data. We prefer to use truly open-source models.

### Chat with h2oGPT

Start an h2oGPT chatbot like this:
```
torchrun generate.py --load_8bit=True --base_model=h2oai/h2ogpt-oasst1-512-12b --prompt_type=human_bot  # needs 24GB GPU

# torchrun generate.py --load_8bit=True --base_model=h2oai/h2ogpt-oig-oasst1-256-6.9b --prompt_type=human_bot  # needs 8GB GPU
# torchrun generate.py --load_8bit=True --base_model=h2oai/h2ogpt-oig-oasst1-256-12b --prompt_type=human_bot  # needs 16GB GPU
# torchrun generate.py --load_8bit=True --base_model=h2oai/h2ogpt-oasst1-256-20b --prompt_type=human_bot  # needs 48GB GPU
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

