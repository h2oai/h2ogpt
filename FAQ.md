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

### Install GPT-NEOX
```bash
source ~/.bashrc.mamba
mamba create -n gptneox
conda activate gptneox
mamba install python=3.8 -y
mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
cd gpt-neox/
pip install -r requirements/requirements.txt
mamba install cudatoolkit-dev=11.7 cudatoolkit=11.7 -c conda-forge -c nvidia -y
unset CUDA_HOME
python ./megatron/fused_kernels/setup.py install
pip install -r ./requirements/requirements-flashattention.txt
cd ..
git clone https://github.com/EleutherAI/DeeperSpeed.git
cd DeeperSpeed
./install.sh
python prepare_data.py -d ./data
wget --cut-dirs=5 -nH -r --no-parent --reject "index.html*" https://the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/ -P 20B_checkpoints
```
Now can train, fine-tune, inference with Flash attention by changing the config file for neox to specify attention_type to flash.
```git
diff --git a/configs/20B.yml b/configs/20B.yml
index 6595919..52dfbfb 100644
--- a/configs/20B.yml
+++ b/configs/20B.yml
@@ -14,12 +14,13 @@
   # parallelism settings ( you will want to change these based on your cluster setup, ideally scheduling pipeline stages
   # across the node boundaries )
   "pipe-parallel-size": 4,
-  "model-parallel-size": 2,
+  "model-parallel-size": 1,
 
   # model settings
   "num-layers": 44,
   "hidden-size": 6144,
   "num-attention-heads": 64,
+  "attention_config": [[["flash"], 44]],
   "seq-length": 2048,
   "max-position-embeddings": 2048,
   "norm": "layernorm",
```
The change to model parallel size is to use one pipeline per GPU, required to satisfy deep.py
Run generation like:
```bash
./deepy.py generate.py ./configs/20B.yml
```

### Use fast attention with LLaMa from Vicunda/FastChat  repo:
[Special transformers hash](https://github.com/lm-sys/FastChat#install)<br />
[Patch1](https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py)<br />
[Patch2](https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_mem.py#L5)<br />


In case you get peer to peer related errors on non-homogeneous GPU systems, set this env var:
```
export NCCL_P2P_LEVEL=LOC
```
