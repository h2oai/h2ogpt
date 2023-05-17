## h2oGPT - The world's best open source GPT

Our goal is to make the world's best open source GPT!

### Try h2oGPT now 

Live hosted instances:
- [![img-small.png](img-small.png) h2oGPT 12B](https://gpt.h2o.ai/)
- [🤗 h2oGPT 12B #1](https://huggingface.co/spaces/h2oai/h2ogpt-chatbot)
- [🤗 h2oGPT 12B #2](https://huggingface.co/spaces/h2oai/h2ogpt-chatbot2)
- [![img-small.png](img-small.png) h2oGPT (research) 30B](http://gpt2.h2o.ai)
- [![img-small.png](img-small.png) Original LangChain-enabled h2oGPT (temporary link) 12B](https://9b1c74d9de90a71538.gradio.live/)
- [![img-small.png](img-small.png) Latest LangChain-enabled h2oGPT (temporary link) 12B](https://f559f52f1375366f3c.gradio.live/)

For questions, discussing, or just hanging out, come and join our <a href="https://discord.gg/WKhYMWcVbq"><b>Discord</b></a>!

### Apache V2 ChatBot with LangChain Integration

- [**LangChain**](README_LangChain.md) equipped Chatbot integration and streaming
- **Persistent** database using Chroma or in-memory with FAISS
- **Original** content url links and scores to rank content against query
- **Private** offline database of any documents ([PDFs and more](README_LangChain.md#supported-datatypes))
- **Upload** documents via chatbot into shared space or only allow scratch space
- **Control** data sources and the context provided to LLM
- **Efficient** use of context using instruct-tuned LLMs (no need for many examples)
- **API** for client-server control

<img src="langchain.png" alt="VectorDB" title="VectorDB via LangChain">

### Apache V2 Data Preparation code, Training code, and Models

- **Variety** of models (h2oGPT, WizardLM, Vicuna, OpenAssistant, etc.) supported
- **Fully Commercially** Apache V2 code, data and models
- **High-Quality** data cleaning of large open-source instruction datasets
- **LORA** (low-rank approximation) efficient 8-bit and 16-bit fine-tuning and generation
- **Large** (up to 65B parameters) models built on commodity or enterprise GPUs (single or multi node)
- **Evaluate** performance using RLHF-based reward models

https://user-images.githubusercontent.com/6147661/232924684-6c0e2dfb-2f24-4098-848a-c3e4396f29f6.mov

All open-source datasets and models are posted on [🤗 H2O.ai's Hugging Face page](https://huggingface.co/h2oai/).

Also check out [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio) for our no-code LLM fine-tuning framework!

### General Roadmap items

- Integration of code and resulting LLMs with downstream applications and low/no-code platforms
- Complement h2oGPT chatbot with search and other APIs
- High-performance distributed training of larger models on trillion tokens
- Enhance the model's code completion, reasoning, and mathematical capabilities, ensure factual correctness, minimize hallucinations, and avoid repetitive output

### ChatBot and LangChain Roadmap items

- Support URLs including within chat itself
- Ability to save chats and start new chats
- Add other tools like search

### Getting Started

```bash
git clone https://github.com/h2oai/h2ogpt.git
cd h2ogpt
pip install -r requirements.txt
python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6.9b
```
Then point browser at http://0.0.0.0:7860 or the public live URL printed by the server (disable shared link with `--share=False`).

For quickly using a private document collection for Q/A, place documents (PDFs, text, etc.) into a folder called `user_path` and run
```bash
pip install -r requirements_optional_langchain.txt
python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6.9b --langchain_mode=UserData --user_path=user_path
```
Any other instruct-tuned base models can be used, including non-h2oGPT ones.  For more ways to ingest on CLI and contro see [LangChain Readme](README_LangChain.md)

For help installing a Python 3.10 environment or CUDA toolkit or installing flash attention support, see the [installation instructions](INSTALL.md)

You can also use [Docker](INSTALL-DOCKER.md#containerized-installation-for-inference-on-linux-gpu-servers) for inference.

#### Larger models require more GPU memory

Depending on available GPU memory, you can load differently sized models. For multiple GPUs, automatic sharding can be enabled with `--infer_devices=False`, but this is disabled by default since cuda:x cuda:y mismatches can occur.

For GPUs with at least 24GB of memory, we recommend:
```bash
python generate.py --base_model=h2oai/h2ogpt-oasst1-512-12b
```
For GPUs with at least 48GB of memory, we recommend:
```bash
python generate.py --base_model=h2oai/h2ogpt-oasst1-512-20b
```
The number `512` in the model names indicates the cutoff lengths (in tokens) used for fine-tuning. Shorter values generally result in faster training and more focus on the last part of the provided input text (consisting of prompt and answer).

More information about the models can be found on [H2O.ai's Hugging Face page](https://huggingface.co/h2oai/).

### Development

- To create a development environment for training and generation, follow the [installation instructions](INSTALL.md).
- To fine-tune any LLM models on your data, follow the [fine-tuning instructions](FINETUNE.md).
- To create a container for deployment, follow the [Docker instructions](INSTALL-DOCKER.md).

### Help

[FAQs](FAQ.md)

### More links, context, competitors, models, datasets

[Links](LINKS.md)

### Acknowledgements

* Some training code was based upon March 24 version of [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/).
* Used high-quality created data by [OpenAssistant](https://open-assistant.io/).
* Used base models by [EleutherAI](https://www.eleuther.ai/).
* Used OIG data created by [LAION](https://laion.ai/blog/oig-dataset/).

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

### Disclaimer

Please read this disclaimer carefully before using the large language model provided in this repository. Your use of the model signifies your agreement to the following terms and conditions.

- Biases and Offensiveness: The large language model is trained on a diverse range of internet text data, which may contain biased, racist, offensive, or otherwise inappropriate content. By using this model, you acknowledge and accept that the generated content may sometimes exhibit biases or produce content that is offensive or inappropriate. The developers of this repository do not endorse, support, or promote any such content or viewpoints.
- Limitations: The large language model is an AI-based tool and not a human. It may produce incorrect, nonsensical, or irrelevant responses. It is the user's responsibility to critically evaluate the generated content and use it at their discretion.
- Use at Your Own Risk: Users of this large language model must assume full responsibility for any consequences that may arise from their use of the tool. The developers and contributors of this repository shall not be held liable for any damages, losses, or harm resulting from the use or misuse of the provided model.
- Ethical Considerations: Users are encouraged to use the large language model responsibly and ethically. By using this model, you agree not to use it for purposes that promote hate speech, discrimination, harassment, or any form of illegal or harmful activities.
- Reporting Issues: If you encounter any biased, offensive, or otherwise inappropriate content generated by the large language model, please report it to the repository maintainers through the provided channels. Your feedback will help improve the model and mitigate potential issues.
- Changes to this Disclaimer: The developers of this repository reserve the right to modify or update this disclaimer at any time without prior notice. It is the user's responsibility to periodically review the disclaimer to stay informed about any changes.

By using the large language model provided in this repository, you agree to accept and comply with the terms and conditions outlined in this disclaimer. If you do not agree with any part of this disclaimer, you should refrain from using the model and any content generated by it.
