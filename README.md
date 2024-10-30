# h2oGPT

Turn ★ into ⭐ (top-right corner) if you like the project!

Query and summarize your documents or just chat with local private GPT LLMs using h2oGPT, an Apache V2 open-source project.

Check out a long CoT Open-o1 open 🍓strawberry🍓 project: https://github.com/pseudotensor/open-strawberry

## Live Demo

[![img-small.png](docs/img-small.png) Gradio Demo](https://gpt.h2o.ai/)

[![img-small.png](docs/img-small.png) OpenWebUI Demo](https://gpt-docs.h2o.ai/)

## Video Demo

https://github.com/h2oai/h2ogpt/assets/2249614/2f805035-2c85-42fb-807f-fd0bca79abc6

[![img-small.png](docs/img-small.png) YouTube 4K Video](https://www.youtube.com/watch?v=_iktbj4obAI)

## Features

- **Private** offline database of any documents [(PDFs, Excel, Word, Images, Video Frames, YouTube, Audio, Code, Text, MarkDown, etc.)](docs/README_LangChain.md#supported-datatypes)
  - **Persistent** database (Chroma, Weaviate, or in-memory FAISS) using accurate embeddings (instructor-large, all-MiniLM-L6-v2, etc.)
  - **Efficient** use of context using instruct-tuned LLMs (no need for LangChain's few-shot approach)
  - **Parallel** summarization and extraction, reaching an output of 80 tokens per second with the 13B LLaMa2 model
  - **HYDE** (Hypothetical Document Embeddings) for enhanced retrieval based upon LLM responses
  - **Semantic Chunking** for better document splitting (requires GPU)
- **Variety** of models supported (LLaMa2, Mistral, Falcon, Vicuna, WizardLM.  With AutoGPTQ, 4-bit/8-bit, LORA, etc.)
  - **GPU** support from HF and LLaMa.cpp GGML models, and **CPU** support using HF, LLaMa.cpp, and GPT4ALL models
  - **Attention Sinks** for [arbitrarily long](https://github.com/tomaarsen/attention_sinks) generation (LLaMa-2, Mistral, MPT, Pythia, Falcon, etc.)
- **Gradio UI** or CLI with streaming of all models
  - **Upload** and **View** documents through the UI (control multiple collaborative or personal collections)
  - **Vision Models** LLaVa, Claude-3, Gemini-Pro-Vision, GPT-4-Vision
  - **Image Generation** Stable Diffusion (sdxl-turbo, sdxl, SD3), PlaygroundAI (playv2), and Flux
  - **Voice STT** using Whisper with streaming audio conversion
  - **Voice TTS** using MIT-Licensed Microsoft Speech T5 with multiple voices and Streaming audio conversion
  - **Voice TTS** using MPL2-Licensed TTS including Voice Cloning and Streaming audio conversion
  - **AI Assistant Voice Control Mode** for hands-free control of h2oGPT chat
  - **Bake-off** UI mode against many models at the same time
  - **Easy Download** of model artifacts and control over models like LLaMa.cpp through the UI
  - **Authentication** in the UI by user/password via Native or Google OAuth
  - **State Preservation** in the UI by user/password
- **Open Web UI** with h2oGPT as backend via OpenAI Proxy
  - See [Start-up Docs](docs/FAQ.md#open-web-ui).
  - Chat completion with streaming
  - Document Q/A using h2oGPT ingestion with advanced OCR from DocTR
  - Vision models
  - Audio Transcription (STT)
  - Audio Generation (TTS)
  - Image generation
  - Authentication
  - State preservation
- **Linux, Docker, macOS, and Windows** support
- **Inference Servers** [support](docs/README_InferenceServers.md) for oLLaMa, HF TGI server, vLLM, Gradio, ExLLaMa, Replicate, Together.ai, OpenAI, Azure OpenAI, Anthropic, MistralAI, Google, and Groq
- **OpenAI compliant**
  - Server Proxy [API](docs/README_CLIENT.md) (h2oGPT acts as drop-in-replacement to OpenAI server)
  - Chat and Text Completions (streaming and non-streaming)
  - Audio Transcription (STT)
  - Audio Generation (TTS)
  - Image Generation
  - Embedding
  - Function tool calling w/auto tool selection
  - AutoGen Code Execution Agent
- **JSON Mode**
  - Strict schema control for vLLM via its use of outlines
  - Strict schema control for OpenAI, Anthropic, Google Gemini, MistralAI models
  - JSON mode for some older OpenAI or Gemini models with schema control if model is smart enough (e.g. gemini 1.5 flash)
  - Any model via code block extraction
- **Web-Search** integration with Chat and Document Q/A
- **Agents** for Search, Document Q/A, Python Code, CSV frames
  - High quality Agents via OpenAI proxy server on separate port
  - Code-first agent that generates plots, researches, evaluates images via vision model, etc. (client code openai_server/openai_client.py).
  - No UI for this, just API
- **Evaluate** performance using reward models
- **Quality** maintained with over 1000 unit and integration tests taking over 24 GPU-hours

## Get Started

[![GitHub license](https://img.shields.io/github/license/NVIDIA/nvidia-docker?style=flat-square)](LICENSE)
[![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)](https://github.com/h2oai/h2ogpt/blob/main/docs/README_LINUX.md)
[![macOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0)](https://github.com/h2oai/h2ogpt/blob/main/docs/README_MACOS.md)
[![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://github.com/h2oai/h2ogpt/blob/main/docs/README_WINDOWS.md)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://github.com/h2oai/h2ogpt/blob/main/docs/README_DOCKER.md)

### Install h2oGPT

Docker is recommended for Linux, Windows, and MAC for full capabilities.  Linux Script also has full capability, while Windows and MAC scripts have less capabilities than using Docker.

* [Docker Build and Run Docs (Linux, Windows, MAC)](docs/README_DOCKER.md)
* [Linux Install and Run Docs](docs/README_LINUX.md)
* [Windows 10/11 Installation Script](docs/README_WINDOWS.md)
* [MAC Install and Run Docs](docs/README_MACOS.md)
* [Quick Start on any Platform](docs/README_quickstart.md)

---

### Collab Demos
- [![](https://colab.research.google.com/assets/colab-badge.svg) h2oGPT CPU](https://colab.research.google.com/drive/13RiBdAFZ6xqDwDKfW6BG_-tXfXiqPNQe?usp=sharing)
- [![](https://colab.research.google.com/assets/colab-badge.svg) h2oGPT GPU](https://colab.research.google.com/drive/143-KFHs2iCqXTQLI2pFCDiR69z0dR8iE?usp=sharing)

### Resources
- [FAQs](docs/FAQ.md)
- [README for LangChain](docs/README_LangChain.md)
- [Discord](https://discord.gg/WKhYMWcVbq)
- [Models (LLaMa-2, Falcon 40, etc.) at 🤗](https://huggingface.co/h2oai/)
- [YouTube: 100% Offline ChatGPT Alternative?](https://www.youtube.com/watch?v=Coj72EzmX20)
- [YouTube: Ultimate Open-Source LLM Showdown (6 Models Tested) - Surprising Results!](https://www.youtube.com/watch?v=FTm5C_vV_EY)
- [YouTube: Blazing Fast Falcon 40b 🚀 Uncensored, Open-Source, Fully Hosted, Chat With Your Docs](https://www.youtube.com/watch?v=H8Dx-iUY49s)
- [Technical Paper: https://arxiv.org/pdf/2306.08161.pdf](https://arxiv.org/pdf/2306.08161.pdf)

### Docs Guide
<!--  cat README.md | ./gh-md-toc  -  But Help is heavily processed -->
* [Get Started](#get-started)
   * [Linux (CPU or CUDA)](docs/README_LINUX.md)
   * [macOS (CPU or M1/M2)](docs/README_MACOS.md)
   * [Windows 10/11 (CPU or CUDA)](docs/README_WINDOWS.md)
   * [GPU (CUDA, AutoGPTQ, exllama) Running Details](docs/README_GPU.md)
   * [CPU Running Details](docs/README_CPU.md)
   * [CLI chat](docs/README_CLI.md)
   * [Gradio UI](docs/README_ui.md)
   * [Client API (Gradio, OpenAI-Compliant)](docs/README_CLIENT.md)
   * [Inference Servers (oLLaMa, HF TGI server, vLLM, Groq, Anthropic, Google, Mistral, Gradio, ExLLaMa, Replicate, OpenAI, Azure OpenAI)](docs/README_InferenceServers.md)
   * [Build Python Wheel](docs/README_WHEEL.md)
   * [Offline Installation](docs/README_offline.md)
   * [Low Memory](docs/FAQ.md#low-memory-mode)
   * [Docker](docs/README_DOCKER.md)
* [LangChain Document Support](docs/README_LangChain.md)
* [Compare to PrivateGPT et al.](docs/README_LangChain.md#what-is-h2ogpts-langchain-integration-like)
* [Roadmap](#roadmap)
* [Development](#development)
* [Help](#help)
   * [LangChain file types supported](docs/README_LangChain.md#supported-datatypes)
   * [CLI Database control](docs/README_LangChain.md#database-creation)
   * [FAQ](docs/FAQ.md)
     * [Model Usage Notes](docs/FAQ.md#model-usage-notes)
     * [Adding LLM Models (including using GGUF and Attention Sinks)](docs/FAQ.md#adding-models)
     * [Adding Embedding Models](docs/FAQ.md#add-new-embedding-model)
     * [Adding Prompts](docs/FAQ.md#adding-prompt-templates)
     * [In-Context Learning](docs/FAQ.md#in-context-learning-via-prompt-engineering)
     * [Multiple GPUs](docs/FAQ.md#multiple-gpus)
     * [Low-Memory Usage](docs/FAQ.md#low-memory-mode)
     * [Environment Variables](docs/FAQ.md#what-envs-can-i-pass-to-control-h2ogpt)
     * [HTTPS access for server and client](docs/FAQ.md#https-access-for-server-and-client)
   * [Useful Links](docs/LINKS.md)
   * [Fine-Tuning](docs/FINETUNE.md)
   * [Triton](docs/TRITON.md)
   * [Commercial viability](docs/FAQ.md#commercial-viability)
* [Acknowledgments](#Acknowledgments)
* [Why H2O.ai?](#why-h2oai)
* [Disclaimer](#disclaimer)

### Development

- To create a development environment for training and generation, follow the [installation instructions](docs/INSTALL.md).
- To fine-tune any LLM models on your data, follow the [fine-tuning instructions](docs/FINETUNE.md).
- To run h2oGPT tests:
    ```bash
    pip install requirements-parser pytest-instafail pytest-random-order playsound==1.3.0
    conda install -c conda-forge gst-python -y
    sudo apt-get install gstreamer-1.0
    pip install pygame
    GPT_H2O_AI=0 CONCURRENCY_COUNT=1 pytest --instafail -s -v tests
    # for openai server test on already-running local server
    pytest -s -v -n 4 openai_server/test_openai_server.py::test_openai_client
    ```
  or tweak/run `tests/test4gpus.sh` to run tests in parallel.

### Acknowledgments

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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=h2oai/h2ogpt&type=Timeline)](https://star-history.com/#h2oai/h2ogpt&Timeline)
