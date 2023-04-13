## Fine-tuning

Make sure you have followed the [native installation instructions](INSTALL.md).

### Create instruct dataset

#### Simple OIG based data
By running the following commands, we are assembling some of the instruct-type datasets
that are [publicly available with Apache V2 license](https://huggingface.co/datasets/laion/OIG).
```bash
pytest create_data.py::test_get_OIG_data
pytest create_data.py::test_merge_shuffle_OIG_data
```
This will create a file called `merged_shuffled_OIG_87f6a1e788.json`. This is not cleaned up yet, but it's
fully open-source and 135MB of instruct data, which will work for v1.

### Perform fine-tuning on your data

Fine-tune on a single node with NVIDIA GPUs A6000/A6000Ada/A100/H100, needs 48GB of GPU memory per GPU for default settings.
For GPUs with 24GB of memory, need to set `--micro_batch_size=1` and `--batch_size=$NGPUS` below.
```
export NGPUS=`nvidia-smi -L | wc -l`
torchrun --nproc_per_node=$NGPUS finetune.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --data_path=merged_shuffled_OIG_87f6a1e788.json --prompt_type=plain --output_dir=my_finetuned_weights
```
This will download the model, load the data, and generate an output directory `my_finetuned_weights` containing the fine-tuned state.


### Start your own fine-tuned chat bot

Start a chatbot, also requires 48GB GPU. Likely run out of memory on 24GB GPUs, but can work with lower values for `--chat_history`.
```
torchrun generate.py --load_8bit=True --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --lora_weights=my_finetuned_weights --prompt_type=human_bot
```
This will download the foundation model, our fine-tuned lora_weights, and open up a GUI with text generation input/output.
