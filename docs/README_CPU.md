### CPU

#### Google Colab

A Google Colab version of a 7B LLaMa CPU model is at:

[![](https://colab.research.google.com/assets/colab-badge.svg) h2oGPT CPU](https://colab.research.google.com/drive/13RiBdAFZ6xqDwDKfW6BG_-tXfXiqPNQe?usp=sharing)

A local copy of that CPU Google Colab is [h2oGPT_CPU.ipynb](h2oGPT_CPU.ipynb).

---

#### Local

CPU support is obtained after installing two optional requirements.txt files.  This does not preclude GPU support, just adds CPU support:

* Install base, langchain, and GPT4All, and python LLaMa dependencies:
```bash
git clone https://github.com/h2oai/h2ogpt.git
cd h2ogpt
for fil in requirements.txt reqs_optional/requirements_optional_langchain.txt reqs_optional/requirements_optional_gpt4all.txt reqs_optional/requirements_optional_langchain.gpllike.txt reqs_optional/requirements_optional_langchain.urls.txt ; do pip install -r $fil --extra-index https://download.pytorch.org/whl/cpu ; done
# Optional: support docx, pptx, ArXiv, etc.
sudo apt-get install -y libmagic-dev poppler-utils tesseract-ocr libreoffice
# Optional: for supporting unstructured package
python -m nltk.downloader all
```
See [GPT4All](https://github.com/nomic-ai/gpt4all) for details on installation instructions if any issues encountered.

* Change `.env_gpt4all` model name if desired.
```.env_gpt4all
model_path_llama=WizardLM-7B-uncensored.ggmlv3.q8_0.bin
model_path_gptj=ggml-gpt4all-j-v1.3-groovy.bin
model_name_gpt4all_llama=ggml-wizardLM-7B.q4_2.bin
```
For `gptj` and `gpt4all_llama`, you can choose a different model than our default choice by going to GPT4All Model explorer [GPT4All-J compatible model](https://gpt4all.io/index.html). One does not need to download manually, the gp4all package will download at runtime and put it into `.cache` like Hugging Face would.  However, `gpjt` model often gives [no output](FAQ.md#gpt4all-not-producing-output), even outside h2oGPT.

So, for chatting, a better instruct fine-tuned LLaMa-based model for llama.cpp can be downloaded from [TheBloke](https://huggingface.co/TheBloke).  For example, [13B WizardLM Quantized](https://huggingface.co/TheBloke/wizardLM-13B-1.0-GGML) or [7B WizardLM Quantized](https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GGML).  TheBloke has a variety of model types, quantization bit depths, and memory consumption.  Choose what is best for your system's specs.  However, be aware that LLaMa-based models are not [commercially viable](FAQ.md#commercial-viability).

For 7B case, download [WizardLM-7B-uncensored.ggmlv3.q8_0.bin](https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GGML/resolve/main/WizardLM-7B-uncensored.ggmlv3.q8_0.bin) into local path:
```bash
wget https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GGML/resolve/main/WizardLM-7B-uncensored.ggmlv3.q8_0.bin
```
Then one sets `model_path_llama` in `.env_gpt4all`, which is currently the default.

* Run generate.py

For LangChain support using documents in `user_path` folder, run h2oGPT like:
```bash
python generate.py --base_model='llama' --prompt_type=wizard2 --score_model=None --langchain_mode='UserData' --user_path=user_path
```
See [LangChain Readme](README_LangChain.md) for more details.
For no langchain support (still uses LangChain package as model wrapper), run as:
```bash
python generate.py --base_model='llama' --prompt_type=wizard2 --score_model=None
```

When using `llama.cpp` based CPU models, for computers with low system RAM or slow CPUs, we recommend adding to `.env_gpt4all`:
```.env_gpt4all
use_mlock=False
n_ctx=1024
```
where `use_mlock=True` is default to avoid slowness and `n_ctx=2048` is default for large context handling.  For computers with plenty of system RAM, we recommend adding to `.env_gpt4all`:
```.env_gpt4all
n_batch=1024
```
for faster handling.  On some systems this has no strong effect, but on others may increase speed quite a bit.

Also, for slow and low-memory systems, we recommend using a smaller embedding by using with `generate.py`:
```bash
python generate.py ... --hf_embedding_model=sentence-transformers/all-MiniLM-L6-v2
```
where `...` means any other options one should add like `--base_model` etc.  This simpler embedding is about half the size as default `instruct-large` and so uses less disk, CPU memory, and GPU memory if using GPUs.

See also [Low Memory](FAQ.md#low-memory-mode) for more information about low-memory recommendations.

