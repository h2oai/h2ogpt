## Fine-tuning

Make sure you have followed the [native installation instructions](INSTALL.md).


### Fine-tuning vs Pre-training

- Pre-training (typically on TBs of data) gives the LLM the ability to master one or many languages. Pre-training usually takes weeks or months on dozens or hundreds of GPUs. The most common concern is underfitting and cost.
- Fine-tuning (typically on MBs or GBs of data) makes a model more familiar with a specific style of prompting, which generally leads to improved outcomes for this one specific case. The most common concern is overfitting. Fine-tuning usually takes hours or days on a few GPUs.


### Dataset format

In general, LLMs take plain text (ordered list of tokens, explained in the [FAQ](FAQ.md)) as input and generate plain text as output.
For example, for pretraining this text is perfectly usable:
```text
and suddenly all the players raised their hands and shouted
```
as the model will learn to say `suddenly` after `and` and it will learn to say `players` after `and suddenly all the` etc., as 
part of the overall language training on hundreds of billions of tokens. Imagine that this is not a very efficient way to learn a language, but it works.

For fine-tuning, when we only present a small set of high-quality data to the model, the creation of good input/output pairs is the *labeling* work one has to do.

For example, for fine-tuning, one could create such a dataset entry:
```text
Instruction: Summarize.
Input: This is a very very very long paragraph saying nothing much.
Output: Nothing was said.
```
This text is better suited to teach the model to summarize. During inference, one would present the model with the following text and it would provide the summary as the continuation of the input, since it is already familiar with this prompting technique:
```text
Instruction: Summarize.
Input: TEXT TO SUMMARIZE
Output:
```

For a chatbot, one could fine-tune the model by providing data examples like this:
```text
<human>: Hi, who are you?
<bot>: I'm h2oGPT.
<human>: Who trained you?
<bot>: I was trained by H2O.ai, the visionary leader in democratizing AI.
```

and during inference, one would present the following to the LLM, for it to respond as the `<bot>`:
```text
<human>: USER INPUT FROM CHAT APPLICATION
<bot>:
```

More details about the exact dataset specs can be found in our [FAQ](FAQ.md).

### Create instruct dataset

Below are some of our scripts to help with assembling and cleaning instruct-type datasets that are
[publicly available with permissive licenses](https://huggingface.co/datasets/laion/OIG).

#### High-quality OIG based instruct data

For a higher quality dataset, run the following commands:
```bash
pytest -s create_data.py::test_download_useful_data_as_parquet  # downloads ~ 4.2GB of open-source permissive data
pytest -s create_data.py::test_assemble_and_detox               # ~ 3 minutes, 4.1M clean conversations
pytest -s create_data.py::test_chop_by_lengths                  # ~ 2 minutes, 2.8M clean and long enough conversations
pytest -s create_data.py::test_grade                            # ~ 3 hours, keeps only high quality data
pytest -s create_data.py::test_finalize_to_json
```
This will take several hours and produce a file called [h2ogpt-oig-oasst1-instruct-cleaned-v2.json](https://huggingface.co/datasets/h2oai/h2ogpt-oig-oasst1-instruct-cleaned-v2) (575 MB) with 350k human <-> bot interactions.

Note: This dataset is cleaned up, but might still contain undesired words and concepts.

### Install training specific dependencies

```bash
pip install -r reqs_optional/requirements_optional_training.txt
```

### Perform fine-tuning on high-quality instruct data

Fine-tune on a single node with NVIDIA GPUs A6000/A6000Ada/A100/H100, needs 48GB of GPU memory per GPU for default settings (fast 16-bit training).
For larger models or GPUs with less memory, need to set a combination of `--train_4bit=True` (or `--train_8bit=True`) and `--micro_batch_size=1`, `--batch_size=$NGPUS` and `--cutoff_len=256` below, or use smaller models like `h2oai/h2ogpt-oasst1-512-12b`.
```
export NGPUS=`nvidia-smi -L | wc -l`
torchrun --nproc_per_node=$NGPUS finetune.py --base_model=h2oai/h2ogpt-oasst1-512-20b --data_path=h2oai/h2ogpt-oig-oasst1-instruct-cleaned-v2 --output_dir=h2ogpt_lora_weights
```
This will download the model, load the data, and generate an output directory `h2ogpt_lora_weights` containing the fine-tuned state.


### Start your own fine-tuned chatbot

Start a chatbot, also requires 48GB GPU. Use `--load_4bit=True` instead for 24GB GPUs.
```
torchrun generate.py --load_8bit=True --base_model=h2oai/h2ogpt-oasst1-512-20b --lora_weights=h2ogpt_lora_weights --prompt_type=human_bot
```
This will download the foundation model, our fine-tuned lora_weights, and open up a GUI with text generation input/output.
