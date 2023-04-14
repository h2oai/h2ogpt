## Fine-tuning

Make sure you have followed the [native installation instructions](INSTALL.md).

### Create instruct dataset

Below are some of our scripts to help with assembling and cleaning instruct-type datasets that are
[publicly available with permissive licenses](https://huggingface.co/datasets/laion/OIG).

#### High-quality OIG based instruct data

For a higher quality dataset, run the following commands:
```bash
pytest -s create_data.py::test_download_useful_data_as_parquet  # downloads ~ 11GB of open-source permissive data
pytest -s create_data.py::test_assemble_and_detox               # ~ 20 minutes, 5.9M clean conversations
pytest -s create_data.py::test_chop_by_lengths                  # ~ 5 minutes, 5.0M clean and long enough conversations
pytest -s create_data.py::test_grade                            # ~ 3 hours, keeps only high quality data
pytest -s create_data.py::test_finalize_to_json
```
This will take several hours and produce a file called `h2ogpt-oig-instruct-cleaned.json` (XX MB) with XXk human <-> bot interactions.
Note: This dataset is cleaned up, but might still contain undesired words and concepts.

### Perform fine-tuning on high-quality instruct data

Fine-tune on a single node with NVIDIA GPUs A6000/A6000Ada/A100/H100, needs 48GB of GPU memory per GPU for default settings.
For GPUs with 24GB of memory, need to set `--micro_batch_size=1`, `--batch_size=$NGPUS` and `--cutoff_len=256` below.
```
export NGPUS=`nvidia-smi -L | wc -l`
torchrun --nproc_per_node=$NGPUS finetune.py --base_model=EleutherAI/gpt-neox-20b --data_path=h2oai/h2ogpt-oig-instruct-cleaned --prompt_type=plain --output_dir=h2ogpt_lora_weights
```
This will download the model, load the data, and generate an output directory `h2ogpt_lora_weights` containing the fine-tuned state.


### Start your own fine-tuned chat bot

Start a chatbot, also requires 48GB GPU. Likely run out of memory on 24GB GPUs, but can work with lower values for `--chat_history`.
```
torchrun generate.py --load_8bit=True --base_model=EleutherAI/gpt-neox-20b --lora_weights=h2ogpt_lora_weights --prompt_type=human_bot
```
This will download the foundation model, our fine-tuned lora_weights, and open up a GUI with text generation input/output.
