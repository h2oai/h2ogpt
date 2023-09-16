## Frequently asked questions

### Migration from Chroma < 0.4 to > 0.4

#### Option 1: Use old Chroma for old DBs

Do nothing as user.  h2oGPT will by default not migrate for old databases.  This is the default way handled internally by requirements added in `requirements_optional_langchain.txt` by adding special wheels for old versions of chromadb and hnswlib, handling migration better than chromadb itself.

#### Option 2: Automatically Migrate

h2oGPT by default does not migrate automatically with `--auto_migrate_db=False` for `generate.py`.  One can set this to `True` for auto-migration, which may time some time for larger databases.  This will occur on-demand when accessing a database.  This takes about 0.03s per chunk.

#### Option 3: Manually Migrate

One can set that to False and manually migrate databases by doing the following.

* Install and run migration tool
```
pip install chroma-migrate
chroma-migrate
```
* Choose DuckDB
* Choose "Files I can use ..."
* Choose your collection path, e.g. `db_dir_UserData` for collection name `UserData`

### Adding Models

One can choose any Hugging Face model or quantized GGML model file in h2oGPT.

Hugging Face models are passed via `--base_model` in all cases, with an extra `--load_gptq` for GPTQ models, e.g., by [TheBloke](https://huggingface.co/TheBloke).  Hugging Face models are automatically downloaded to the Hugging Face .cache folder (in home folder).

GGML v3 quantized models are supported, and [TheBloke](https://huggingface.co/TheBloke) also has many of those, e.g.
```bash
python generate.py --base_model=llama --model_path_llama=llama-2-7b-chat.ggmlv3.q8_0.bin --max_seq_len=4096
```
For GGML models, always good to pass `--max_seq_len` directly.  When passing the filename like above, we assume one has previously downloaded the model to the local path, but if one passes a URL, then we download the file for you.
You can also pass a URL for automatic downloading (which will not re-download if file already exists):
```bash
python generate.py --base_model=llama --model_path_llama=https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin --max_seq_len=4096
```
for any TheBloke GGML v3 models.

GPT4All models are supported, which are automatically downloaded to a GPT4All cache folder (in the home folder).  E.g.
```bash
python generate.py --base_model=gptj --model_name_gptj=ggml-gpt4all-j-v1.3-groovy.bin
```
for GPTJ models (also downloaded automatically):
```bash
python generate.py --base_model=gpt4all_llama --model_name_gpt4all_llama=ggml-wizardLM-7B.q4_2.bin
```
for GPT4All LLaMa models.

See [README_CPU.md](README_CPU.md) and [README_GPU.md](README_GPU.md) for more information on controlling these parameters.

### Adding Prompt Templates

After specifying a model, one needs to consider if an existing `prompt_type` will work or a new one is required.  E.g. for Vicuna models, a well-defined `prompt_type` is used which we support automatically for specific model names.  If the model is in `prompter.py` as associated with some `prompt_type` name, then we added it already.  See models that are currently supported in this automatic way in [prompter.py](../src/prompter.py) and [enums.py](../src/enums.py).

If we do not list the model in `prompter.py`, then if you find a `prompt_type` by name that works for your new model, you can pass `--prompt_type=<NAME>` for some prompt_type `<NAME>`, and we will use that for the new model.

However, in some cases, you need to add a new prompt structure because the model does not conform at all (or exactly enough) to the template given in, e.g., the Hugging Face model card or elsewhere.  In that case you have two options:

* **Option 1**: Use custom prompt

    In CLI you can pass `--prompt_type=custom --prompt_dict="{....}"` for some dict {....}.  The dictionary doesn't need to contain all the things mentioned below, but should contain primary ones.

    You can also choose `prompt_type=custom` in expert settings and change `prompt_dict` in the UI under `Models tab`.  Not all of these dictionary keys need to be set:
    ```
    promptA
    promptB
    PreInstruct
    PreInput
    PreResponse
    terminate_response
    chat_sep
    chat_turn_sep
    humanstr
    botstr
    ```
    i.e. see how consumed:  https://github.com/h2oai/h2ogpt/blob/a51576cd174e9fda61f00c3889a26888a604172c/src/prompter.py#L130-L142

    The ones that are most crucial are:
    ```
    PreInstruct
    PreResponse
    humanstr
    botstr
    ```
    and often `humanstr` just equals `PreInstruct` and `botstr` just equals `PreResponse`.

    If so, then really only have to set 2 things.

* **Option 2**: Tweak or Edit code

   You can change the code itself if that seems easier than using CLI or UI.  For that case you'd do:

   1) In `prompter.py`, add new key (`prompt_type` name) and value (model name) into `prompt_type_to_model_name`
   2) In `enums.py`, add a new name and value for the new `prompt_type`
   3) In `prompter.py`, add new block in `get_prompt()`

    A simple example to follow is vicuna11, with this block:
    ```
    elif prompt_type in [PromptType.vicuna11.value, str(PromptType.vicuna11.value),
                         PromptType.vicuna11.name]:
        preprompt = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. """ if not (
                chat and reduced) else ''
        start = ''
        promptB = promptA = '%s%s' % (preprompt, start)
        eos = '</s>'
        PreInstruct = """USER: """
        PreInput = None
        PreResponse = """ASSISTANT:"""
        terminate_response = [PreResponse]
        chat_sep = ' '
        chat_turn_sep = eos
        humanstr = PreInstruct
        botstr = PreResponse

        if making_context:
            # when making context, want it to appear as-if LLM generated, which starts with space after :
            PreResponse = PreResponse + ' '
        else:
            # normally LLM adds space after this, because was how trained.
            # if add space here, non-unique tokenization will often make LLM produce wrong output
            PreResponse = PreResponse
    ```
    You can start by changing each thing that appears in the model card that tells about the prompting.  You can always ask for help in a GitHub issue or Discord.

In either case, if the model card doesn't have that information, you'll need to ask around.  Sometimes, prompt information will be in their pipeline file or in a GitHub repository associated with the model with training of inference code.  Or sometimes the model builds upon another, and you should look at the original model card.  You can also  ask in the community section on Hugging Face for that model card.

### Add new Embedding Model

The option `--use_openai_embedding` set to `True` or `False` controls whether use OpenAI embedding, `--hf_embedding_model` set to some HuggingFace model name sets that as embedding model if not using OpenAI.  The setting `--migrate_embedding_model` as `True` or `False` chooses whether to migrate to new chosen embeddings or stick with existing/original embedding for a given database.  The option `--cut_distance` as float chooses the distance above which to avoid using document sources.  The default is 1.64, tuned for  Mini and instructor-large.  One can pass `--cut_distance=100000` to avoid any filter.  E.g.
```bash
python generate.py --base_model=h2oai/h2ogpt-4096-llama2-13b-chat  --score_model=None --langchain_mode='UserData' --user_path=user_path --use_auth_token=True --hf_embedding_model=BAAI/bge-large-en --cut_distance=1000000
```

### In-Context learning via Prompt Engineering

For arbitrary tasks, good to use uncensored models like [Falcon 40 GM](https://huggingface.co/h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2).  If censored is ok, then [LLama-2 Chat](https://huggingface.co/h2oai/h2ogpt-4096-llama2-70b-chat) are ok. Choose model size according to your system specs.

For the UI, CLI, or EVAL this means editing the `System Pre-Context` text box in expert settings.  When starting h2oGPT, one can pass `--context` or `--iinput` for setting a fixed default context choice and fixed default instruction choice.  Note if no context is passed but `--chat_context=True`, then that function sets the context.

Or for API, passing `context` variable.  This can be filled with arbitrary things, including actual conversations to prime the model, although if a conversation then need to put in prompts like:
```python
from gradio_client import Client
import ast

HOST_URL = "http://localhost:7860"
client = Client(HOST_URL)

# string of dict for input
prompt = 'Who are you?'
# falcon, but falcon7B is not good at this:
#context = """<|answer|>I am a pixie filled with fairy dust<|endoftext|><|prompt|>What kind of pixie are you?<|endoftext|><|answer|>Magical<|endoftext|>"""
# LLama2 7B handles this well:
context = """[/INST] I am a pixie filled with fairy dust </s><s>[INST] What kind of pixie are you? [/INST] Magical"""
kwargs = dict(instruction_nochat=prompt, context=context)
res = client.predict(str(dict(kwargs)), api_name='/submit_nochat_api')

# string of dict for output
response = ast.literal_eval(res)['response']
print(response)
```
See for example: https://github.com/h2oai/h2ogpt/blob/d3334233ca6de6a778707feadcadfef4249240ad/tests/test_prompter.py#L47 .

Note that even if the prompting is not perfect or matches the model, smarter models will still do quite well, as long as you give their answers as part of context.

### Token access to Hugging Face models:

Related to transformers.  There are two independent ways to do this (choose one):
* Use ENV:
    ```
    export HUGGING_FACE_HUB_TOKEN=<token goes here>
    ```
    token starts with `hf_` usually.  Then start h2oGPT like normal.
  See [Hugging Face ENV documentation](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables) for other environment variables.
* Use cli tool:
    ```bash
    huggingface-cli login
    ```
    in repo.  Then add to generate.py:
    ```
    python generate.py --use_auth_token=True ...
    ```
  See [Hugging Face Access Tokens](https://huggingface.co/docs/hub/security-tokens) for more details.

### Low-memory mode

For GPU case, a reasonable model for low memory is to run:
```bash
python generate.py --base_model=h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3 --hf_embedding_model=sentence-transformers/all-MiniLM-L6-v2 --score_model=None --load_8bit=True --langchain_mode='UserData'
```
which uses good but smaller base model, embedding model, and no response score model to save GPU memory.  If you can do 4-bit, then do:
```bash
python generate.py --base_model=h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3 --hf_embedding_model=sentence-transformers/all-MiniLM-L6-v2 --score_model=None --load_4bit=True --langchain_mode='UserData'
```
This uses 5800MB to startup, then soon drops to 5075MB after torch cache is cleared. Asking a simple question uses up to 6050MB. Adding a document uses no more new GPU memory.  Asking a question uses up to 6312MB for a few chunks (default), then drops back down to 5600MB.

On CPU case, a good model that's still low memory is to run:
```bash
python generate.py --base_model='llama' --prompt_type=llama2 --hf_embedding_model=sentence-transformers/all-MiniLM-L6-v2 --langchain_mode=UserData --user_path=user_path
```

Ensure to vary `n_gpu_layers` at CLI or in UI to smaller values to reduce offloading for smaller GPU memory boards.

### ValueError: ...offload....

```
The current `device_map` had weights offloaded to the disk. Please provide an `offload_folder` for them. Alternatively, make sure you have `safetensors` installed if the model you are using offers
the weights in this format.
```

If you see this error, then you either have insufficient GPU memory or insufficient CPU memory.  E.g. for 6.9B model one needs minimum of 27GB free memory.

### TypeError: Chroma.init() got an unexpected keyword argument 'anonymized_telemetry'

Please check your version of langchain vs. the one in requirements.txt.  Somehow the wrong version is installed.  Try to install the correct one.

### bitsandbytes CUDA error
  ```text
  CUDA Setup failed despite GPU being available. Please run the following command to get more information:
  E               
  E                       python -m bitsandbytes
  E               
  E                       Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
  E                       to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
  E                       and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues
  ```

Ensure you have cuda version supported by bitsandbytes, e.g. in Ubuntu:
```text
sudo update-alternatives --display cuda
sudo update-alternatives --config cuda
```
and ensure you choose CUDA 12.1 if using bitsandbytes 0.39.0 since that is last version it supports.  Or upgrade bitsandbytes if that works.  Or uninstall bitsandbytes to remove 4-bit and 8-bit support, but that will also avoid the error. 

### Multiple GPUs

Automatic sharding can be enabled with `--use_gpu_id=False`.  This is disabled by default, as in rare cases torch hits a bug with `cuda:x cuda:y mismatch`.  E.g. to use GPU IDs 0 and 3, one can run:
```bash
export HUGGING_FACE_HUB_TOKEN=<hf_...>
exoprt CUDA_VISIBLE_DEVICES="0,3"
export GRADIO_SERVER_PORT=7860
python generate.py \
          --base_model=meta-llama/Llama-2-7b-chat-hf \
          --prompt_type=llama2 \
          --max_max_new_tokens=4096 \
          --max_new_tokens=1024 \
          --use_gpu_id=False \
          --save_dir=save7b \
          --score_model=None \
          --use_auth_token="$HUGGING_FACE_HUB_TOKEN"
```
where `use_auth_token` has been set as required for LLaMa2.

### Larger models require more GPU memory

Depending on available GPU memory, you can load differently sized models. For multiple GPUs, automatic sharding can be enabled with `--use_gpu_id=False`, but this is disabled by default since cuda:x cuda:y mismatches can occur.

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

### CPU with no AVX2 or using LLaMa.cpp

For GPT4All based models, require AVX2, unless one recompiles that project on your system.  Until then, use llama.cpp models instead.

So we recommend downloading models from [TheBloke](https://huggingface.co/TheBloke) that are version 3 quantized ggml files to work with latest llama.cpp.  See main [README.md](README_CPU.md).

The following example is for the base LLaMa model, not instruct-tuned, so it is not recommended for chatting.  It just gives an example of how to quantize if you are an expert.

Compile the llama model on your system by following the [instructions](https://github.com/ggerganov/llama.cpp#build) and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), e.g. for Linux:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make clean
make LLAMA_OPENBLAS=1
```
on CPU, or for GPU:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make clean
make LLAMA_CUBLAS=1
```
etc. following different [scenarios](https://github.com/ggerganov/llama.cpp#build).

Then:
```bash
# obtain the original LLaMA model weights and place them in ./models, i.e. models should contain:
# 65B 30B 13B 7B tokenizer_checklist.chk tokenizer.model

# install Python dependencies
conda create -n llamacpp -y
conda activate llamacpp
conda install python=3.10 -y
pip install -r requirements.txt

# convert the 7B model to ggml FP16 format
python convert.py models/7B/

# quantize the model to 4-bits (using q4_0 method)
./quantize ./models/7B/ggml-model-f16.bin ./models/7B/ggml-model-q4_0.bin q4_0

# test by running the inference
./main -m ./models/7B/ggml-model-q4_0.bin -n 128
```
then pass run like (assumes version 3 quantization):
```bash
python generate.py --base_model=llama --model_path_llama=./models/7B/ggml-model-q4_0.bin
```
or wherever you placed the model with the path pointing to wherever the files are located (e.g. link from h2oGPT repo to llama.cpp repo folder), e.g.
```bash
cd ~/h2ogpt/
ln -s ~/llama.cpp/models/* .
```
then run h2oGPT like:
```bash
python generate.py --base_model='llama' --langchain_mode=UserData --user_path=user_path
```

### Is this really a GGML file? Or Using version 2 quantization files from GPT4All that are LLaMa based

If hit error:
```text
Found model file.
llama.cpp: loading model from ./models/7B/ggml-model-q4_0.bin
error loading model: unknown (magic, version) combination: 67676a74, 00000003; is this really a GGML file?
llama_init_from_file: failed to load model
LLAMA ERROR: failed to load model from ./models/7B/ggml-model-q4_0.bin
```
then note that llama.cpp upgraded to version 3, and we use llama-cpp-python version that supports only that latest version 3.  GPT4All does not support version 3 yet.  If you want to support older version 2 llama quantized models, then do:
```bash
pip install --force-reinstall --ignore-installed --no-cache-dir llama-cpp-python==0.1.73
```
to go back to the prior version.  Or specify the model using GPT4All, run:
```bash
python generate.py --base_model=gpt4all_llama  --model_path_gpt4all_llama=./models/7B/ggml-model-q4_0.bin
```
assuming that file is from version 2 quantization.

### not enough memory: you tried to allocate 590938112 bytes.

    If one sees: 
    ```
    RuntimeError: [enforce fail at ..\c10\core\impl\alloc_cpu.cpp:72] data. DefaultCPUAllocator: not enough memory: you tried to allocate 590938112 bytes.
    ```
    then probably CPU has insufficient memory to handle the model.  Try GGML.

### WARNING: failed to allocate 258.00 MB of pinned memory: out of memory

    If you see:
    ```
    Warning: failed to VirtualLock 17825792-byte buffer (after previously locking 1407303680 bytes): The paging file is too small for this operation to complete.
    
    WARNING: failed to allocate 258.00 MB of pinned memory: out of memory
    Traceback (most recent call last):
    ```
    then you have insufficient pinned memory on your GPU.  You can disable pinning by setting this env before launching h2oGPT:
* Linux:
    ```
    export GGML_CUDA_NO_PINNED=1
    ```
* Windows:
    ```
    setenv GGML_CUDA_NO_PINNED=1
    ```


### I get the error: `The model 'OptimizedModule' is not supported for . Supported models are ...`

This warning can be safely ignored.

### What ENVs can I pass to control h2oGPT?

   - `SAVE_DIR`: Local directory to save logs to,
   - `ADMIN_PASS`: Password to access system info, logs, or push to aws s3 bucket,
   - `AWS_BUCKET`: AWS bucket name to push logs to when have admin access,
   - `AWS_SERVER_PUBLIC_KEY`: AWS public key for pushing logs to when have admin access,
   - `AWS_SERVER_SECRET_KEY`: AWS secret key for pushing logs to when have admin access,
   - `HUGGING_FACE_HUB_TOKEN`: Read or write HF token for accessing private models,
   - `LANGCHAIN_MODE`: LangChain mode, overrides CLI,
   - `SCORE_MODEL`: HF model to use for scoring prompt-response pairs, `None` for no scoring of responses,
   - `HEIGHT`: Height of Chat window,
   - `allow_upload_to_user_data`: Whether to allow uploading to Shared UserData,
   - `allow_upload_to_my_data`: Whether to allow uploading to Personal MyData,
   - `HEIGHT`: Height of Chat window,
   - `HUGGINGFACE_SPACES`: Whether on public A10G 24GB HF spaces, sets some low-GPU-memory defaults for public access to avoid GPU memory abuse by model switching, etc.
   - `HF_HOSTNAME`: Name of HF spaces for purpose of naming log files,
   - `GPT_H2O_AI`: Whether on public 48GB+ GPU instance, sets some defaults for public access to avoid GPU memory abuse by model switching, etc.,
   - `CONCURRENCY_COUNT`: Number of concurrency users to gradio server (1 is fastest since LLMs tend to consume all GPU cores, but 2-4 is best to avoid any single user waiting too long to get response)
   - `API_OPEN`: Whether API access is visible,
   - `ALLOW_API`: Whether to allow API access,
   - `CUDA_VISIBLE_DEVICES`: Standard list of CUDA devices to make visible.
   - `PING_GPU`: ping GPU every few minutes for full GPU memory usage by torch, useful for debugging OOMs or memory leaks
   - `GET_GITHASH`: get git hash on startup for system info.  Avoided normally as can fail with extra messages in output for CLI mode
   - `H2OGPT_BASE_PATH`: Choose base folder for all files except personal/scratch files
These can be useful on HuggingFace spaces, where one sets secret tokens because CLI options cannot be used.

> **_NOTE:_**  Scripts can accept different environment variables to control query arguments. For instance, if a Python script takes an argument like `--load_8bit=True`, the corresponding ENV variable would follow this format: `H2OGPT_LOAD_8BIT=True` (regardless of capitalization). It is important to ensure that the environment variable is assigned the exact value that would have been used for the script's query argument.

### How to run functions in src from Python interpreter

E.g.
```python
import sys
sys.path.append('src')
from src.gpt_langchain import get_supported_types
non_image_types, image_types, video_types = get_supported_types()
print(non_image_types)
print(image_types)
for x in image_types:
    print('   - `.%s` : %s Image (optional),' % (x.lower(), x.upper()))
# unused in h2oGPT:
print(video_types)
```

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

### AMD support

Untested AMD support: Download and install [bitsandbytes on AMD](https://github.com/arlo-phoenix/bitsandbytes-rocm-5.6)

#### Disclaimers

Disclaimers and a ToS link are displayed to protect the app creators.

### What are the different prompt types? How does prompt engineering work for h2oGPT?

In general, all LLMs use strings as inputs for training/fine-tuning and generation/inference.
To manage a variety of possible language task types, we divide any such string into the following three parts:

- Instruction
- Input
- Response

Each of these three parts can be empty or non-empty strings, such as titles or newlines. In the end, all of these prompt parts are concatenated into one string. The magic is in the content of those substrings. This is called **prompt engineering**.

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

For a conversational chatbot use case, we use the following three parts:

- Instruction = `<INSTRUCTION>`
- Input = `'<human>: '` + `<INPUT>`
- Response = `'<bot>: '` + `<OUTPUT>`

And a training string could look like this: `'<human>: hi, how are you?<bot>: Hi, I am doing great. How can I help you?'`.
At inference time, the model input would be like this: `'<human>: Tell me a joke about snow flakes.<bot>: '`, and the model would generate the bot part.


### How should training data be prepared?

Training data (in `JSON` format) must contain at least one column that maps to `instruction`, `input` or `output`.
Their content will be placed into the `<INSTRUCTION>`, `<INPUT>`, and `<OUTPUT>` placeholders mentioned above.
The chosen `prompt_type` will fill in the strings in between to form the actual input into the model.
Any missing columns will lead to empty strings. Optional `--data_col_dict={'A': 'input', 'B': 'output'}` argument can
be used to map different column names into the required ones.

#### Examples

The following are examples of training records in `JSON` format.

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

Note that the total length of the text (that is, the input and output) the LLM can handle is limited by the so-called *context length*. For our current models, the context length is 2048 tokens. Longer context lengths are computationally more expensive due to the interactions between all tokens in the sequence.
A context length of 2048 means that for an input of, for example, 1900 tokens, the model will be able to create no more than 148 new tokens as part of the output.

For fine-tuning, if the average length of inputs is less than the context length, one can provide a `cutoff_len` of less than the context length to truncate inputs to this amount of tokens. For most instruction-type datasets, a cutoff length of 512 seems reasonable and provides nice memory and time savings.
For example, the `h2oai/h2ogpt-oasst1-512-20b` model was trained with a cutoff length of 512.

### Tokens

The following are some example tokens (from a total of ~50k), each of which is assigned a number:
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
The input format doesn't change whether the model is in pretraining, fine-tuning, or inference mode, but the text itself can change slightly for better results, and that's called prompt engineering.

### Is h2oGPT multilingual?

Yes. Try it in your preferred language.

### What does 512 mean in the model name?

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



### Heterogeneous GPU systems

In case you get peer-to-peer related errors on non-homogeneous GPU systems, set this env var:
```
export NCCL_P2P_LEVEL=LOC
```


### Use Wiki data

The following example demonstrates how to use Wiki data:

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

### Centos with llama-cpp-python

This may help to get llama-cpp-python to install

```bash
# remove old gcc
yum remove gcc yum remove gdb
# install scl-utils
sudo yum install scl-utils sudo yum install centos-release-scl
# find devtoolset-11
yum list all --enablerepo='centos-sclo-rh' | grep "devtoolset"
# install devtoolset-11-toolchain
yum install -y devtoolset-11-toolchain
# add gcc 11 to PATH by adding following script to /etc/profile
PATH=$PATH::/opt/rh/devtoolset-11/root/usr/bin export PATH sudo scl enable devtoolset-11 bash
# show gcc version and gcc11 is installed successfully.
gcc --version
export FORCE_CMAKE=1
export CMAKE_ARGS=-DLLAMA_OPENBLAS=on
pip install llama-cpp-python --no-cache-dir
```
