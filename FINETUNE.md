## Fine-tuning

Make sure you have followed the [native installation instructions](INSTALL.md).

### Create instruct dataset

Below are some of our scripts to help with assembling and cleaning instruct-type datasets that are
[publicly available with permissive licenses](https://huggingface.co/datasets/laion/OIG).

#### Simple OIG based data

To reproduce a reasonably good dataset, run the following commands:
```bash
pytest create_data.py::test_get_small_sample_oig_data
pytest create_data.py::test_merge_shuffle_small_sample_oig_data
```
This creates a file called `merged_shuffled_OIG_87f6a1e788.json` (136 MB) with 240k human <-> bot interactions.
Note: The dataset contains profanities, and is not cleaned up. Use this only for quick explorations.

#### Higher-quality OIG based data

For a higher quality dataset, run the following commands:
```bash
pytest create_data.py::test_useful_oig_data_as_parquet
pytest create_data.py::test_basic_cleaning
pytest create_data.py::test_grade_final
pytest create_data.py::test_grade_final_parquet_to_json
```
This will take about one hour on A6000 Ada, and produce a file called `df_final_graded_full.json` (XX MB) with XXk human <-> bot interactions.
Note: This dataset is cleaned up, but might still contain undesired words and concepts.

### Perform fine-tuning on your data

Fine-tune on a single node with NVIDIA GPUs A6000/A6000Ada/A100/H100, needs 48GB of GPU memory per GPU for default settings.
For GPUs with 24GB of memory, need to set `--micro_batch_size=1` and `--batch_size=$NGPUS` below.
```
export NGPUS=`nvidia-smi -L | wc -l`
torchrun --nproc_per_node=$NGPUS finetune.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --data_path=df_final_graded_full.json --prompt_type=plain --output_dir=h2ogpt_lora_weights
```
This will download the model, load the data, and generate an output directory `h2ogpt_lora_weights` containing the fine-tuned state.


### Start your own fine-tuned chat bot

Start a chatbot, also requires 48GB GPU. Likely run out of memory on 24GB GPUs, but can work with lower values for `--chat_history`.
```
torchrun generate.py --load_8bit=True --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --lora_weights=h2ogpt_lora_weights --prompt_type=human_bot
```
This will download the foundation model, our fine-tuned lora_weights, and open up a GUI with text generation input/output.
