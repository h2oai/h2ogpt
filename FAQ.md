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
python generate.py --load_8bit=True --base_model=h2oai/h2ogpt-oig-oasst1-512-6.9b --langchain_mode=MyData --user_path=user_path --score_model=None
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

### CPU

* Install LangChain dependencies (currently required):
```bash
pip install pip --upgrade -y
make
pip install -r requirements_optional_langchain.txt -c req_constraints.txt
```

* Install LLaMa/GPT4All dependencies
```bash
pip install pip --upgrade -y
make
pip install -r requirements_optional_gpt4all.txt -c req_constraints.txt
```
See [GPT4All](https://github.com/nomic-ai/gpt4all) for details on installation instructions if any issues encountered.

* Download [ggml-gpt4all-j-v1.3-groovy.bin](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin)
If you prefer a different [GPT4All-J compatible model](https://gpt4all.io/index.html) (see Model Explorer), can use that instead. Download from the Model Explorer, and place into repo folder.

* Fill `.env_gpt4all` with at least model path, if did not copy to repo folder, change name of model to the one you have chosen from Model Explorer if was not the default.
```.env_gpt4all
# model path and model_kwargs
model_path_gptj=ggml-gpt4all-j-v1.3-groovy.bin
```
See [llama.cpp](https://github.com/ggerganov/llama.cpp) for instructions on getting model for `--base_model=llama` case.

For LangChain support using documents in `user_path` folder, run h2oGPT like:
```bash
python generate.py --base_model=gptj --score_model=None --langchain_mode='UserData' --user_path=user_path
```
For no langchain support, run as:
```bash
python generate.py --base_model=gptj --score_model=None
```

### I get the error: `The model 'OptimizedModule' is not supported for . Supported models are ...`

Ignore this warning.
