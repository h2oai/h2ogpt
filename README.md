## H2O LLM Prototyping Playground

Goal is to create 100% permissive MIT/ApacheV2 LLM model that is useful for ChatGPT usecases.

Training code is based on [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/), but all models will be fully open source.

No OpenAI-based Alpaca fine-tuning data will be left.

Final result will be committed to [H2OGPT](https://github.com/h2oai/h2ogpt/).


### Installation

Follow the [instructions](INSTALL.md) to create an environment.

#### Create instruct dataset

```bash
pytest create_data.py::test_get_OIG_data
pytest create_data.py::test_merge_shuffle_OIG_data
```

#### Perform fine-tuning on your data

Fine-tune on a single node with NVIDIA GPUs A6000/A6000Ada/A100/H100, needs 48GB of GPU memory per GPU.
```
export NGPUS=`nvidia-smi -L | wc -l`
torchrun --nproc_per_node=$NGPUS finetune.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --data_path=merged_shuffled_OIG_87f6a1e788.json --prompt_type=plain --output_dir=my_finetuned_weights
```
this will download the model, load the data, and generate an output directory `my_finetuned_weights` containing the fine-tuned state.


#### Start ChatBot

Start a chatbot, also requires 48GB GPU.
```
torchrun generate.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --lora_weights=my_finetuned_weights --prompt_type=human_bot
```
this will download the foundation model, our fine-tuned lora_weights, and open up a GUI with text generation input/output.


In case you get peer to peer related errors on non-homogeneous GPU systems, set this env var:
```
export NCCL_P2P_LEVEL=LOC
```


### Docker Setup & Inference

1. Build the container image:

```bash
docker build -t h2o-llm .
```

2. Run the container (you can also use `finetune.py` and all of its parameters as shown above for training):

```bash
docker run --runtime=nvidia --shm-size=64g -p 7860:7860 -v ${HOME}/.cache:/root/.cache --rm h2o-llm -it generate.py \
    --load_8bit=True --base_model='EleutherAI/gpt-neox-20b' --prompt_type=human_bot
```

3. Open `https://localhost:7860` in the browser

### Docker Compose Setup & Inference

1. (optional) Change desired model and weights under `environment` in the `docker-compose.yml`

2. Build and run the container

```bash
docker-compose up -d --build
```

3. Open `https://localhost:7860` in the browser

4. See logs:

```bash
docker-compose logs -f
```

5. Clean everything up:

```bash
docker-compose down --volumes --rmi all
```


### Tensorboard

```bash
tensorboard --logdir=runs/
```

### Plan
Open source instruct model for demoable usecases.
1. Base: Start with fully open source apache 2.0 models EleutherAI--gpt-j-6B, EleutherAI--gpt-neox-20b, 
GPT-NeoXT-Chat-Base-20B, etc. 
2. Construct Prompt: Setup prompt engineering on 6B-20B as-is to convert a sentence into question/answer or command/response format 
3. Open-Source Instruct Data: Convert wiki data into instruct form
4. Fine-tune: LORA fine-tune 6B and 20B using DAI docs
5. Open Data & Model: Submit DAI docs model huggingface
6. Use toolformer approach for external APIs

### Goals
1. Demonstrate fine-tuning working on some existing corpus
2. Demonstrate efficiency of LORA for fast and low-memory fine-tuning


### Code to consider including
[flan-alpaca](https://github.com/declare-lab/flan-alpaca)<br />
[text-generation-webui](https://github.com/oobabooga/text-generation-webui)<br />
[minimal-llama](https://github.com/zphang/minimal-llama/)<br />
[finetune GPT-NeoX](https://nn.labml.ai/neox/samples/finetune.html)<br />
[GPTQ-for_LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/compare/cuda...Digitous:GPTQ-for-GPT-NeoX:main)<br />
[OpenChatKit on multi-GPU](https://github.com/togethercomputer/OpenChatKit/issues/20)<br />
[Non-Causal LLM](https://huggingface.co/docs/transformers/main/en/model_doc/gptj#transformers.GPTJForSequenceClassification)<br />
[OpenChatKit_Offload](https://github.com/togethercomputer/OpenChatKit/commit/148b5745a57a6059231178c41859ecb09164c157)<br />
[Flan-alpaca](https://github.com/declare-lab/flan-alpaca/blob/main/training.py)<br />

### Help

[FAQs](FAQ.md)

### More links, context, competitors, models, datasets

[Links](LINKS.md)
