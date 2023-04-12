## H2O LLM Prototyping Playground

Goal is to create 100% permissive MIT/ApacheV2 LLM model that is useful for ChatGPT usecases.

Training code is based on [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/), but all models will be fully open source.

No OpenAI-based Alpaca fine-tuning data will be left.

Final result will be committed to [H2OGPT](https://github.com/h2oai/h2ogpt/).


### Setup

#### Install Python environment

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
bash ./Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
# follow license agreement and add to bash if required
source ~/.bashrc
conda create -n h2ollm -y
conda activate h2ollm
conda install mamba -n base -c conda-forge
conda install python=3.10 -y
conda update -n base -c defaults conda
```

#### Install Python packages

```bash
git clone https://github.com/h2oai/h2o-llm.git
cd h2o-llm
pip install -r requirements.txt
```

#### Install CUDA 12.1 [install cuda coolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)

E.g. for Ubuntu 20.04, select Ubuntu, Version 20.04, Installer Type "deb (local)", and you should get the following commands:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

Then set the system up to use the freshly installed CUDA location:
```bash
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/" >> ~/.bashrc
echo "CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
echo "export PATH=$PATH:/usr/local/cuda/bin/" >> ~/.bashrc
echo "conda activate h2ollm" >> ~/.bashrc
source ~/.bashrc
conda activate h2ollm
```

Then reboot the machine, to get everything sync'ed up on restart.
```bash
sudo reboot
```

#### Compile bitsandbytes for fast 8-bit training [howto src](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md)

```bash
git clone http://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes
git checkout 7c651012fce87881bb4e194a26af25790cadea4f
CUDA_VERSION=121 make cuda12x
CUDA_VERSION=121 python setup.py install
cd ..
```

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
