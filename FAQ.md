### What are the different prompt types? How does prompt engineering work for h2oGPT?

In general, all LLMs use strings as inputs for training/fine-tuning and generation/inference.
To manage a variety of possible language task types, we divide any such string into three parts:

- Instruction
- Input
- Response

Each of these three parts can be empty or non-empty strings, such as titles or newlines. In the end, all
these prompt parts are concatenated into one string. The magic is in the content of those sub-strings. This is called **prompt engineering**.

#### Summarization

For training a summarization task, we concatenate these three parts together:

- Instruction = `<INSTRUCTION>`
- Input = `'## Main Text\n\n'` + `<INPUT>`
- Response = `'\n\n## Summary\n\n'` + `<OUTPUT>`

For each training record, we take `<INPUT>` and `<OUTPUT>` from the summarization dataset (typically two fields/columns), place them into the appropriate position, and turn that record into
one long string that the model can be trained with: `'## Main Text\n\nLarge Language Models are Useful.\n\n## Summary\n\nLLMs rock.'`

At inference time, we will take the `<INPUT>` only and stop right after `'\n\n## Summary\n\n'` and the model will generate the summary
as the continuation of the prompt.


#### ChatBot

For a conversational chatbot use case, we use the following 3 parts:

- Instruction = `<INSTRUCTION>`
- Input = `'<human>: '` + `<INPUT>`
- Response = `'<bot>: '` + `<OUTPUT>`

And a training string could look like this: `'<human>: hi, how are you?<bot>: Hi, I am doing great. How can I help you?'`.
At inference time, the model input would be like this: `'<human>: Tell me a joke about snow flakes.<bot>: '`, and the model would generate the bot part.


### How to prepare data?

Training data (in `JSON` format) must contain at least one column that maps to `instruction`, `input` or `output`.
Their content will be placed into the `<INSTRUCTION>`, `<INPUT>` and `<OUTPUT>` placeholders mentioned above.
The chosen `prompt_type` will fill in the strings in between to form the actual input into the model.
Any missing columns will lead to empty strings. Optional `--data_col_dict={'A': 'input', 'B': 'output'}` argument can
be used to map different column names into the required ones.

#### Examples

Below are examples of training records in `JSON` format.

- `human_bot` prompt type
```json
{
  "input": "Who are you?",
  "output": "My name is h2oGPT.",
  "prompt_type": "human_bot"
}
```

- `plain` version of `human_bot`, useful for longer conversations
```json
{
  "input": "<human>: Who are you?\n<bot>: My name is h2oGPT.\n<human>: Can you write a poem about horses?\n<bot>: Yes, of course. Here it goes...",
  "prompt_type": "plain"
}
```

- `summarize` prompt type
```json
{
  "instruction": "",
  "input": "Long long long text.",
  "output": "text.",
  "prompt_type": "summarize"
}
```

### Context length

Note that the total length of the text (i.e., input and output) the LLM can handle is limited by the so-called *context length*. For our current models, the context length is 2048 tokens. Longer context lengths are computationally more expensive due to the interactions between all tokens in the sequence.
A context length of 2048 means that for an input of, for example, 1900 tokens, the model will be able to create no more than 148 new tokens as part of the output.

For fine-tuning, if the average length of inputs is less than the context length, one can provide a `cutoff_len` of less than the context length to truncate inputs to this amount of tokens. For most instruction-type datasets, a cutoff length of 512 seems reasonable and provides nice memory and time savings.
For example, the `h2oai/h2ogpt-oasst1-512-20b` model was trained with a cutoff length of 512.

### Tokens

Here are some example tokens (from a total of ~50k), each of which is assigned a number:
```text
"osed": 1700,
"ised": 1701,
"================": 1702,
"ED": 1703,
"sec": 1704,
"Ġcome": 1705,
"34": 1706,
"ĠThere": 1707,
"Ġlight": 1708,
"Ġassoci": 1709,
"gram": 1710,
"Ġold": 1711,
"Ġ{#": 1712,
```
The model is trained with these specific numbers, so the tokenizer must be kept the same for training and inference/generation.
The input format doesn't change whether the model is in pretraining, fine-tuning or inference mode, but the text itself can change slightly for better results, and that's called prompt engineering.

### Why does h2oGPT say it was trained by OpenAI or Open Assistant?

![](https://user-images.githubusercontent.com/6147661/233486736-812d7b95-8c2f-438e-be76-ec4845c28a33.png)

As explained on the [model card](https://huggingface.co/h2oai/h2ogpt-oasst1-512-20b) h2oGPT is a fine-tuned version
of [GPT-NeoX-20b](https://huggingface.co/EleutherAI/gpt-neox-20b), which was trained on the [Pile](https://pile.eleuther.ai/)
and on the [h2oai/openassistant_oasst1](https://huggingface.co/datasets/h2oai/openassistant_oasst1).
These datasets contain training data created by OpenAI (from the GPT-2 days) and by Open Assistant, which injected the above
answer and similar answers. In other words, they "contaminated" the training data with their desired outputs for the model (i.e., personality).
Most of the knowledge of the model is from pre-training on the billions of tokens, the fine-tuning only turns that language
model into a chatbot by returning short answers for short questions, or in other words, pre-training creates language
understanding and some knowledge, while fine-tuning injects style. Certain simple personality traits can be modified by fine-tuning however, and we are working on giving h2oGPT proper personality: https://github.com/h2oai/h2ogpt/issues/73

#### Update
We continued fine-tuning with the [h2oai/h2oai/openassistant_oasst1_h2ogpt](https://huggingface.co/datasets/h2oai/openassistant_oasst1_h2ogpt) dataset, which includes some personalization, and the effects are noticeable.

![image_480](https://user-images.githubusercontent.com/6147661/236146853-31ac322a-7191-43a6-a50a-649e9912d55a.png)

### Is h2oGPT multi-lingual?

Yes. Try it in your preferred language.

### What does 512 mean in model name?

The number `512` in the model names indicates the cutoff lengths (in tokens) used for fine-tuning. Shorter values generally result in faster training and more focus on the last part of the provided input text (consisting of prompt and answer).

### Throttle GPUs in case of reset/reboot

```bash
(h2ogpt) jon@gpu:~$ sudo nvidia-smi -pl 250
Power limit for GPU 00000000:3B:00.0 was set to 250.00 W from 300.00 W.
Power limit for GPU 00000000:5E:00.0 was set to 250.00 W from 300.00 W.
Power limit for GPU 00000000:86:00.0 was set to 250.00 W from 300.00 W.
Power limit for GPU 00000000:AF:00.0 was set to 250.00 W from 300.00 W.
All done.
```

### Use Wiki Data

```python
>>> from datasets import load_dataset
>>> wk = load_dataset("wikipedia", "20220301.en")
>>> wk
DatasetDict({
    train: Dataset({
        features: ['id', 'url', 'title', 'text'],
        num_rows: 6458670
    })
})
>>> sentences = ".".join(wk['train'][0]['text'].split('.')[0:2])
'Anarchism is a political philosophy and movement that is sceptical of authority and rejects all involuntary, coercive forms of hierarchy. Anarchism calls for the abolition of the state, which it holds to be unnecessary, undesirable, and harmful'
>>> 
```


### Heterogeneous GPU systems

In case you get peer to peer related errors on non-homogeneous GPU systems, set this env var:
```
export NCCL_P2P_LEVEL=LOC
```

### Offline Mode:

1) Download model and tokenizer of choice

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = 'h2oai/h2ogpt-oasst1-512-12b'
model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(model_name)
```

2) Download reward model, unless pass `--score_model='None'` to `generate.py`
```python
# and reward model
reward_model = 'OpenAssistant/reward-model-deberta-v3-large-v2'
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained(reward_model)
model.save_pretrained(reward_model)
tokenizer = AutoTokenizer.from_pretrained(reward_model)
tokenizer.save_pretrained(reward_model)
```

3) For LangChain support, download embedding model:
```python
hf_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = 'cpu'
from langchain.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name=hf_embedding_model, model_kwargs=model_kwargs)
```

4) Gradio uses Cloudfare scripts, download from Cloudfare:
```
iframeResizer.contentWindow.min.js
index-8bb1e421.js
``` 
place them into python environment at:
```
site-packages/gradio/templates/cdn/assets
site-packages/gradio/templates/frontend/assets
```

5) For jupyterhub dashboard,  modify `index-8bb1e421.js` to remove or hardcode port number into urls where `/port/7860` is located.  One may have to modify:
```
templates/cdn/index.html
templates/frontend/index.html
templates/frontend/share.html
```
 
6) Run generate with transformers in [Offline Mode](https://huggingface.co/docs/transformers/installation#offline-mode)

```bash
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python generate.py --base_model='h2oai/h2ogpt-oasst1-512-12b'
```

### LangChain Usage:

See [tests/test_langchain_simple.py](tests/test_langchain_simple.py)


### MACOS

* Install [Rust](https://www.geeksforgeeks.org/how-to-install-rust-in-macos/)
```bash
curl –proto ‘=https’ –tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Enter new shell and test: `rustc --version`

* Mac Running Intel
When running a Mac with Intel hardware (not M1), you may run into _clang: error: the clang compiler does not support '-march=native'_ during pip install.
If so set your archflags during pip install. eg: _ARCHFLAGS="-arch x86_64" pip3 install -r requirements.txt_

### C++ Compiler
If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.

### For Windows 10/11
To install a C++ compiler on Windows 10/11, follow these steps:

1. Install Visual Studio 2022.
2. Make sure the following components are selected:
   * Universal Windows Platform development
   * C++ CMake tools for Windows
3. Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/).
4. Run the installer and select the `gcc` component.

###  ENV installation

* Install, e.g. for MACOS: [Miniconda](https://docs.conda.io/en/latest/miniconda.html#macos-installers)

* Enter new shell and should also see `(base)` in prompt

* Create new env:
```bash
conda create -n h2ogpt -y
conda activate h2ogpt
conda install -y mamba -c conda-forge  # for speed
mamba install python=3.10 -c conda-forge -y
```
Should see `(h2ogpt)` in shell prompt.

* Test python:
```bash
python --version
```
should say 3.10.xx
```bash
python -c 'import os, sys ; print("hello world")'
```
should print `hello world`.

* Clone and pip install as usual:
```
bash
git clone https://github.com/h2oai/h2ogpt.git
cd h2ogpt
pip install -r requirements.txt
```

* For non-cuda support, edit requirements_optional_langchain.txt and switch to `faiss_cpu`.

* Install langchain dependencies if want to use langchain:
```bash
pip install -r requirements_optional_langchain.txt
```
and fill `user_path` path with documents to be scanned recursively.

* Run:
```bash
python generate.py --load_8bit=True --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b --langchain_mode=MyData --user_path=user_path --score_model=None
```
It will download the model, which takes about 15 minutes per 3 pytorch bin files if have 10MB/s download.
One can choose any huggingface model, just pass the name after `--base_model=`, but a prompt_type is required if we don't already have support.
E.g. for vicuna models, a typical prompt_type is used and we support that already automatically for specific models,
but if you pass `--prompt_type=instruct_vicuna` with any other vicuna model, we'll use it assuming that is the correct prompt type.
See models that are currently supported in this automatic way, and the same dictionary shows which prompt types are supported: [prompter](prompter.py).

* Potential Errors:
```
ValueError: The current `device_map` had weights offloaded to the disk. Please provide an `offload_folder` for them. Alternatively, make sure you have `safetensors` installed if the model you are using offers
the weights in this format.
```
If you see this error, then you either have insufficient GPU memory or insufficient CPU memory.  E.g. for 6.9B model one needs minimum of 27GB free memory.

### Larger models require more GPU memory

Depending on available GPU memory, you can load differently sized models. For multiple GPUs, automatic sharding can be enabled with `--infer_devices=False`, but this is disabled by default since cuda:x cuda:y mismatches can occur.

For GPUs with at least 24GB of memory, we recommend:
```bash
python generate.py --base_model=h2oai/h2ogpt-oasst1-512-12b --load_8bit=True
```
or
```bash
python generate.py --base_model=h2oai/h2ogpt-oasst1-512-20b --load_8bit=True
```
For GPUs with at least 48GB of memory, we recommend:
```bash
python generate.py --base_model=h2oai/h2ogpt-oasst1-512-20b --load_8bit=True
```
etc.

### CPU with no AVX2

For GPT4All based models, require AVX2, unless one recompiles that project on your system.  Until then, use llama.cpp models instead,
e.g. by compiling the llama model on your system by following the [instructions](https://github.com/ggerganov/llama.cpp#description),
then adding an entry in the .env file like:
```.env_gpt4all
# model path and model_kwargs
model_path_llama=./models/7B/ggml-model-q4_0.bin
```
or wherever you placed the model.

### I get the error: `The model 'OptimizedModule' is not supported for . Supported models are ...`

Ignore this warning.

### What ENVs can I pass to control h2oGPT?

   - `SAVE_DIR`: Local directory to save logs to,
   - `ADMIN_PASS`: Password to acces system info, logs, or push to aws s3 bucket,
   - `AWS_BUCKET`: AWS bucket name to push logs to when have admin access,
   - `AWS_SERVER_PUBLIC_KEY`: AWS public key for pushing logs to when have admin access,
   - `AWS_SERVER_SECRET_KEY`: AWS secret key for pushing logs to when have admin access,
   - `HUGGINGFACE_API_TOKEN`: Read or write HF token for accessing private models,
   - `LANGCHAIN_MODE`: LangChain mode, overrides CLI,
   - `SCORE_MODEL`: HF model to use for scoring prompt-response pairs, `None` for no scoring of responses,
   - `HEIGHT`: Height of Chat window,
   - `allow_upload_to_user_data`: Whether to allow uploading to Shared UserData,
   - `allow_upload_to_my_data`: Whether to allow uploading to Scratch MyData,
   - `HEIGHT`: Height of Chat window,
   - `HUGGINGFACE_SPACES`: Whether on public A10G 24GB HF spaces, sets some low-GPU-memory defaults for public access to avoid GPU memory abuse by model switching, etc.
   - `HF_HOSTNAME`: Name of HF spaces for purpose of naming log files,
   - `GPT_H2O_AI`: Whether on public 48GB+ GPU instance, sets some defaults for public access to avoid GPU memory abuse by model switching, etc.,
   - `CONCURRENCY_COUNT`: Number of concurrency users to gradio server (1 is fastest since LLMs tend to consume all GPU cores, but 2-4 is best to avoid any single user waiting too long to get response)
   - `API_OPEN`: Whether API access is visible,
   - `ALLOW_API`: Whether to allow API access,
   - `CUDA_VISIBLE_DEVICES`: Standard list of CUDA devices to make visible.

These can be usful on HuggingFace spaces, where one sets secret tokens because CLI options cannot be used.

### GPT4All not producing output.

Please contact GPT4All team.  Even a basic test can give empty result.
```python
>>> from gpt4all import GPT4All as GPT4AllModel
>>> m = GPT4AllModel('ggml-gpt4all-j-v1.3-groovy.bin')
Found model file.
gptj_model_load: loading model from '/home/jon/.cache/gpt4all/ggml-gpt4all-j-v1.3-groovy.bin' - please wait ...
gptj_model_load: n_vocab = 50400
gptj_model_load: n_ctx   = 2048
gptj_model_load: n_embd  = 4096
gptj_model_load: n_head  = 16
gptj_model_load: n_layer = 28
gptj_model_load: n_rot   = 64
gptj_model_load: f16     = 2
gptj_model_load: ggml ctx size = 5401.45 MB
gptj_model_load: kv self size  =  896.00 MB
gptj_model_load: ................................... done
gptj_model_load: model size =  3609.38 MB / num tensors = 285
>>> m.generate('Was Avogadro a  professor at the University of Turin?')

''
>>>
```
Also, the model tends to not do well when input has new lines, spaces or `<br>` work better.
This does not seem to be an issue with h2oGPT.

### Commercial viability

Open-source means the models are not proprietary and are available to download.  In addition, the license for all of our non-research models is Apache V2, which is a fully permissive license.  Some licenses for other open-source models are not fully permissive, such as StabilityAI's models that are CC-BY-SA that require derivatives to be shared too.

We post models and license and data origin details on our huggingface page: https://huggingface.co/h2oai (all models, except research ones, are fully permissive).  The foundational models we fine-tuned on, e.g. Pythia 6.9B, Pythia 12B, NeoX 20B, or Open-LLaMa checkpoints are fully commercially viable.  These foundational models are also listed on the huggingface page for each fine-tuned model.  Full training logs, source data, etc. are all provided for all models.  [GPT4All](https://github.com/nomic-ai/gpt4all) GPT_J is commercially viable, but other models may not be.  Any Meta based [LLaMa](https://github.com/facebookresearch/llama) based models are not commercially viable.

Data used to fine-tune are provided on the huggingface pages for each model.  Data for foundational models are provided on their huggingface pages.  Any models trained on GPT3.5 data like ShareGPT, Vicuna, Alpaca, etc. are not commercially viable due to ToS violations w.r.t. building competitive models.  Any research-based h2oGPT models based upon Meta's weights for LLaMa are not commercially viable.

Overall, we have done a significant amount of due diligence regarding data and model licenses to carefully select only fully permissive data and models for our models we license as Apache V2.  Outside our models, some "open-source" models like Vicuna, Koala, WizardLM, etc. are based upon Meta's weights for LLaMa, which is not commercially usable due to ToS violations w.r.t. non-competitive clauses well as research-only clauses.  Such models tend to also use data from GPT3.5 (ChatGPT), which is also not commercially usable due to ToS violations w.r.t. non-competitive clauses.  E.g. Alpaca data, ShareGPT data, WizardLM data, etc. all fall under that category. All open-source foundational models consume data from the internet, including the Pile or C4 (web crawl) that may contain objectionable material.  Future licenses w.r.t. new web crawls may change, but it is our understanding that existing data crawls would not be affected by any new licenses.  However, some web crawl data may contain pirated books.
