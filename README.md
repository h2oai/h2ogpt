# h2oGPT

Turn ★ into ⭐ (top-right corner) if you like the project!

Query and summarize your documents or just chat with local private GPT LLMs using h2oGPT, an Apache V2 open-source project.

- **Private** offline database of any documents [(PDFs, Excel, Word, Images, Youtube, Audio, Code, Text, MarkDown, etc.)](docs/README_LangChain.md#supported-datatypes)
  - **Persistent** database (Chroma, Weaviate, or in-memory FAISS) using accurate embeddings (instructor-large, all-MiniLM-L6-v2, etc.)
  - **Efficient** use of context using instruct-tuned LLMs (no need for LangChain's few-shot approach)
  - **Parallel** summarization and extraction, reaching an output of 80 tokens per second with the 13B LLaMa2 model
  - **HYDE** (Hypothetical Document Embeddings) for enhanced retrieval based upon LLM responses
- **Variety** of models supported (LLaMa2, Mistral, Falcon, Vicuna, WizardLM.  With AutoGPTQ, 4-bit/8-bit, LORA, etc.)
  - **GPU** support from HF and LLaMa.cpp GGML models, and **CPU** support using HF, LLaMa.cpp, and GPT4ALL models
  - **Attention Sinks** for [arbitrarily long](https://github.com/tomaarsen/attention_sinks) generation (LLaMa-2, Mistral, MPT, Pythia, Falcon, etc.)
- **UI** or CLI with streaming of all models
  - **Upload** and **View** documents through the UI (control multiple collaborative or personal collections)
  - **Voice STT** using Whisper with streaming audio conversion
  - **Voice TTS** using MIT-Licensed Microsoft Speech T5 with multiple voices and Streaming audio conversion
  - **Voice TTS** using MPL2-Licensed TTS including Voice Cloning and Streaming audio conversion
  - **AI Assistant Voice Control Mode** to hands-free control h2oGPT chat
  - **Bake-off** UI mode against many models at same time
  - **Easy Download** of model artifacts and control over models like LLaMa.cpp through the UI
  - **Authentication** in the UI by user/password
  - **State Preservation** in the UI by user/password
- **Linux, Docker, macOS, and Windows** support
  - [**Easy Windows Installer**](#windows-1011-64-bit-with-full-document-qa-capability) for Windows 10 64-bit
- **Inference Servers** support (HF TGI server, vLLM, Gradio, ExLLaMa, Replicate, OpenAI, Azure OpenAI)
- **OpenAI-compliant Python client API** for client-server control
- **Web-Search** integration with Chat and Document Q/A
- **Agents** for Search, Document Q/A, Python Code, CSV frames (Experimental, best with OpenAI currently)
- **Evaluate** performance using reward models
- **Quality** maintained with over 1000 unit and integration tests taking over 4 GPU-hours

### Getting Started

[![GitHub license](https://img.shields.io/github/license/NVIDIA/nvidia-docker?style=flat-square)](https://raw.githubusercontent.com/h2oai/h2ogpt/main/LICENSE)
[![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)](https://github.com/h2oai/h2ogpt/blob/main/docs/README_LINUX.md)
[![macOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0)](https://github.com/h2oai/h2ogpt/blob/main/docs/README_MACOS.md)
[![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)](https://github.com/h2oai/h2ogpt/blob/main/docs/README_WINDOWS.md)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://github.com/h2oai/h2ogpt/blob/main/docs/README_DOCKER.md)

To quickly try out h2oGPT on CPU with limited document Q/A capability using LLaMa2 7B Chat, create a fresh Python 3.10 environment and run:
```bash
git clone https://github.com/h2oai/h2ogpt.git
cd h2ogpt
pip install -r requirements.txt
pip install -r reqs_optional/requirements_optional_langchain.txt
pip install -r reqs_optional/requirements_optional_gpt4all.txt
python generate.py --base_model=llama --prompt_type=llama2 --model_path_llama=https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf --max_seq_len=4096
```
then go to your browser by visiting [http://127.0.0.1:7860](http://127.0.0.1:7860) or [http://localhost:7860](http://localhost:7860).  Choose 13B for a better model than 7B.
If you encounter issues with `llama-cpp-python` or other packages that try to compile and fail, try binary wheels for your platform as linked in the detailed instructions below.

We recommend quantized models for most small-GPU systems, e.g. [LLaMa-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf) for 9GB+ GPU memory or larger models like [LLaMa-2-13B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-13b-chat.Q6_K.gguf) if you have 16GB+ GPU memory.

---

#### Windows 10/11 64-bit with full document Q/A capability
  * One-click Installers

    Nov 05, 2023:
    * [h2oGPT GPU-CUDA Installer](https://h2o-release.s3.amazonaws.com/h2ogpt/Nov2023/h2oGPT_0.0.1_gpu.exe) (1.9GB file)
    * [h2oGPT CPU Installer](https://h2o-release.s3.amazonaws.com/h2ogpt/Nov2023/h2oGPT_0.0.1_cpu.exe) (850MB file)

    Oct 06, 2023:
    * [h2oGPT GPU-CUDA Installer](https://h2o-release.s3.amazonaws.com/h2ogpt/Oct2023/h2oGPT_0.0.1_gpu.exe) (1.9GB file)
    * [h2oGPT CPU Installer](https://h2o-release.s3.amazonaws.com/h2ogpt/Oct2023/h2oGPT_0.0.1_cpu.exe) (800MB file)

    Aug 19, 2023:
    * [h2oGPT GPU-CUDA Installer](https://h2o-release.s3.amazonaws.com/h2ogpt/h2oGPT_0.0.1_gpu.exe) (1.8GB file)
    * [h2oGPT CPU Installer](https://h2o-release.s3.amazonaws.com/h2ogpt/h2oGPT_0.0.1_cpu.exe) (755MB file)

    The installers include all dependencies for document Q/A except for models (LLM, embedding, reward), which you can download through the UI.  After installation, go to start and run h2oGPT, and a web browser will open for h2oGPT.  To use LLaMa model, go to Models tab, select `llama` base model, then click load to download from preset URL.  Then use as normal.  To terminate the app, in task manager kill the `Python` process named `pythonw.exe` as will also show up in `nvidia-smi` if using GPUs.  Set environment variables (in system properties->advanced->environment variables) to control things:
      * `n_jobs`: number of cores for various tasks
      * `OMP_NUM_THREADS` thread count for LLaMa
      * `CUDA_VISIBLE_DEVICES` which GPUs are used.  Recommend set to single fast GPU, e.g. `CUDA_VISIBLE_DEVICES=0` if have multiple GPUs.  Note that UI cannot control which GPUs (or CPU mode) for LLaMa models.
      * Any CLI argument from `python generate.py --help` with environment variable set as `h2ogpt_x`, e.g. `h2ogpt_h2ocolors` to `False`.
      * Set env `h2ogpt_server_name` to actual IP address for LAN to see app, e.g. `h2ogpt_server_name` to `192.168.1.172` and allow access through firewall if have Windows Defender activated.
  * [Windows 10/11 Manual Install and Run Docs](docs/README_WINDOWS.md)

---

#### Linux (CPU/CUDA) with full document Q/A capability
  * [Docker Build and Run Docs](docs/README_DOCKER.md)
  * [Linux Manual Install and Run Docs](docs/README_LINUX.md)

---

#### macOS (CPU/M1/M2) with full document Q/A capability
* One-click Installers (Experimental and subject to changes)

  Nov 08, 2023
  - [h2ogpt-osx-m1-cpu](https://h2o-release.s3.amazonaws.com/h2ogpt/Nov2023/h2ogpt-osx-m1-cpu)
  - [h2ogpt-osx-m1-gpu](https://h2o-release.s3.amazonaws.com/h2ogpt/Nov2023/h2ogpt-osx-m1-gpu)
  
  Download the runnable file and open it from finder, it will take few minutes to unpack and run the application.
  These one-click installers are experimental, report any issues with steps to reproduce at https://github.com/h2oai/h2ogpt/issues.

  **Note:** The app bundle is unsigned, if there are any issues in running the app
  ```bash
  $ xattr -dr com.apple.quarantine {file-path}/h2ogpt-osx-m1-gpu
  $ chmod +x {file-path}/h2ogpt-osx-m1-gpu
  ```
* [macOS Manual Install and Run Docs](docs/README_MACOS.md)

---

#### Example Models
* [Highest accuracy and speed](https://huggingface.co/h2oai/h2ogpt-4096-llama2-70b-chat) on 16-bit with TGI/vLLM using ~48GB/GPU when in use (4xA100 high concurrency, 2xA100 for low concurrency)
* [Middle-range accuracy](https://huggingface.co/h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2) on 16-bit with TGI/vLLM using ~45GB/GPU when in use (2xA100)
* [Small memory profile with ok accuracy](https://huggingface.co/TheBloke/Llama-2-13B-Chat-GGUF) 16GB GPU if full GPU offloading
* [Balanced accuracy and size](https://huggingface.co/h2oai/h2ogpt-4096-llama2-13b-chat) on 16-bit with TGI/vLLM using ~45GB/GPU when in use (1xA100)
* [Smallest or CPU friendly](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF) 32GB system ram or 9GB GPU if full GPU offloading
* [Best for 4*A10G using g5.12xlarge](https://huggingface.co/TheBloke/Llama-2-70B-chat-AWQ) AWQ LLaMa 70B using 4*A10G using vLLM

**GPU** mode requires CUDA support via torch and transformers. A 7B/13B model in 16-bit uses 14GB/26GB of GPU memory to store the weights (2 bytes per weight). Compression such as 4-bit precision (bitsandbytes, AWQ, GPTQ, etc.) can further reduce memory requirements down to less than 6GB when asking a question about your documents. (For more information, see [low-memory mode](docs/FAQ.md#low-memory-mode).)

**CPU** mode uses GPT4ALL and LLaMa.cpp, e.g. gpt4all-j, requiring about 14GB of system RAM in typical use.

---

### Live Demos
- [![img-small.png](docs/img-small.png) Live h2oGPT Document Q/A Demo](https://gpt.h2o.ai/)
- [🤗 Live h2oGPT Chat Demo 1](https://huggingface.co/spaces/h2oai/h2ogpt-chatbot)
- [🤗 Live h2oGPT Chat Demo 2](https://huggingface.co/spaces/h2oai/h2ogpt-chatbot2)
- [![](https://colab.research.google.com/assets/colab-badge.svg) h2oGPT CPU](https://colab.research.google.com/drive/13RiBdAFZ6xqDwDKfW6BG_-tXfXiqPNQe?usp=sharing)
- [![](https://colab.research.google.com/assets/colab-badge.svg) h2oGPT GPU](https://colab.research.google.com/drive/143-KFHs2iCqXTQLI2pFCDiR69z0dR8iE?usp=sharing)

### Inference Benchmarks for Summarization & Generation

* [Benchmark results for Llama2](https://github.com/h2oai/h2ogpt/blob/main/benchmarks/perf.md)
* [pytest to create benchmark results](https://github.com/h2oai/h2ogpt/blob/main/tests/test_perf_benchmarks.py)
* [Raw benchmark results (JSON)](https://github.com/h2oai/h2ogpt/blob/main/benchmarks/perf.json)

### Resources
- [Discord](https://discord.gg/WKhYMWcVbq)
- [Models (LLaMa-2, Falcon 40, etc.) at 🤗](https://huggingface.co/h2oai/)
- [YouTube: 100% Offline ChatGPT Alternative?](https://www.youtube.com/watch?v=Coj72EzmX20)
- [YouTube: Ultimate Open-Source LLM Showdown (6 Models Tested) - Surprising Results!](https://www.youtube.com/watch?v=FTm5C_vV_EY)
- [YouTube: Blazing Fast Falcon 40b 🚀 Uncensored, Open-Source, Fully Hosted, Chat With Your Docs](https://www.youtube.com/watch?v=H8Dx-iUY49s)
- [Technical Paper: https://arxiv.org/pdf/2306.08161.pdf](https://arxiv.org/pdf/2306.08161.pdf)

### Partners

- [Live Leaderboard](https://evalgpt.ai/) for GPT-4 Elo Evaluation of Instruct/Chat models with [h2o-LLM-eval](https://github.com/h2oai/h2o-LLM-eval).
- Advanced fine-tuning with [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio)

### Video Demo

https://github.com/h2oai/h2ogpt/assets/2249614/2f805035-2c85-42fb-807f-fd0bca79abc6

YouTube 4K version: https://www.youtube.com/watch?v=_iktbj4obAI

### Docs Guide
<!--  cat README.md | ./gh-md-toc  -  But Help is heavily processed -->
* [Getting Started](#getting-started)
   * [Linux (CPU or CUDA)](docs/README_LINUX.md)
   * [macOS (CPU or M1/M2)](docs/README_MACOS.md)
   * [Windows 10/11 (CPU or CUDA)](docs/README_WINDOWS.md)
   * [GPU (CUDA, AutoGPTQ, exllama) Running Details](docs/README_GPU.md)
   * [CPU Running Details](docs/README_CPU.md)
   * [CLI chat](docs/README_CLI.md)
   * [Gradio UI](docs/README_ui.md)
   * [Client API (Gradio, OpenAI-Compliant)](docs/README_CLIENT.md)
   * [Inference Servers (HF TGI server, vLLM, Gradio, ExLLaMa, Replicate, OpenAI, Azure OpenAI)](docs/README_InferenceServers.md)
   * [Python Wheel](docs/README_WHEEL.md)
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
* [Acknowledgements](#acknowledgements)
* [Why H2O.ai?](#why-h2oai)
* [Disclaimer](#disclaimer)

### Experimental features

These are not part of normal installation instructions and are experimental.

* [Agents](docs/README_Agents.md) -- in Alpha testing.  Optimal for OpenAI, but that also fails sometimes.

### Roadmap

- Integration of code and resulting LLMs with downstream applications and low/no-code platforms
- Complement h2oGPT chatbot with other APIs like [ToolBench](https://github.com/OpenBMB/ToolBench)
- Enhance the model's code completion, reasoning, and mathematical capabilities, ensure factual correctness, minimize hallucinations, and avoid repetitive output
- Add better agents for SQL and CSV question/answer

### Development

- To create a development environment for training and generation, follow the [installation instructions](docs/INSTALL.md).
- To fine-tune any LLM models on your data, follow the [fine-tuning instructions](docs/FINETUNE.md).
- To run h2oGPT tests:
    ```bash
    wget https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf
    pip install requirements-parser pytest-instafail
    pip install playsound
    pytest --instafail -s -v tests
    # for client tests
    make -C client setup
    make -C client build
    pytest --instafail -s -v client/tests
    ```
  or tweak/run `tests/test4gpus.sh` to run tests in parallel.

### Help

- [FAQs](docs/FAQ.md)

- [README for LangChain](docs/README_LangChain.md)

- Useful [links](docs/LINKS.md) for additional context and information on competitors, models, and datasets

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

Please carefully review this disclaimer before utilizing the extensive language model accessible in this repository. Your usage of the model indicates your consent and agreement to the terms and conditions outlined below:

- Biases and Offensiveness: The extensive language model has been trained on a diverse array of internet text data, potentially containing biased, racist, offensive, or otherwise inappropriate content. By utilizing this model, you acknowledge and recognize that the generated content might occasionally exhibit biases or produce material that is offensive or unsuitable. The developers of this repository do not endorse, support, or advocate for any such content or perspectives.

- Limitations: The extensive language model is an AI-based tool and not a human. It has the potential to generate incorrect, nonsensical, or irrelevant responses. It remains the user's responsibility to critically evaluate the generated content and utilize it at their discretion.

- Use at Your Own Risk: Users utilizing this extensive language model must assume full responsibility for any potential consequences arising from its use. The developers and contributors of this repository disclaim any liability for damages, losses, or harm resulting from the usage or misapplication of the provided model.

- Ethical Considerations: Users are encouraged to employ the extensive language model responsibly and ethically. By utilizing this model, you agree not to use it for purposes that promote hate speech, discrimination, harassment, or any form of illegal or harmful activities.

- Reporting Issues: If you encounter biased, offensive, or otherwise inappropriate content generated by the extensive language model, please report it to the repository maintainers through the provided channels. Your feedback will aid in enhancing the model and addressing potential issues.

- Changes to this Disclaimer: The developers of this repository retain the right to modify or update this disclaimer at any time without prior notice. It is the user's responsibility to periodically review the disclaimer to stay informed about any alterations.

- By using the large language model provided in this repository, you agree to accept and comply with the terms and conditions outlined in this disclaimer. If you do not agree with any part of this disclaimer, you should refrain from using the model and any content generated by it.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=h2oai/h2ogpt&type=Timeline)](https://star-history.com/#h2oai/h2ogpt&Timeline)
