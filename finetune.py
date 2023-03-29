import os
import pathlib
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Union
import fire
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
import transformers
import torch.distributed as dist

from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


def log(*args, **kwargs):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args, **kwargs)


try:
    import neptune
    from transformers.integrations import NeptuneCallback

    neptune_run = neptune.init_run(
        source_files=[],
    )
    log("Connected to Neptune.")
except ImportError:
    neptune_run = None
    log("Please pip install neptune for tracking.")
except neptune.exceptions.NeptuneMissingApiTokenException:
    neptune_run = None
    os.environ["NEPTUNE_MODE"] = 'debug'
    log("No neptune configured, set NEPTUNE_API_TOKEN env var.")


def train(
        save_code: bool = False,
        run_id: int = random.randint(0, 2 ** 31),

        # base_model: str = 'togethercomputer/GPT-NeoXT-Chat-Base-20B',
        # base_model: str = 'EleutherAI/gpt-neox-20b',
        # base_model: str = 'decapoda-research/llama-7b-hf',
        # base_model: str = 'decapoda-research/llama-13b-hf',
        # base_model: str = 'decapoda-research/llama-30b-hf',
        base_model: str = 'EleutherAI/gpt-j-6B',

        # only needed if base_model is self-exported HF state without tokenizer
        tokenizer_base_model: str = None,
        # tokenizer_base_model: str = 'EleutherAI/gpt-neox-20b',

        data_path: str = "./alpaca_data_cleaned.json",
        # data_path: str = "./dai_docs.train.json",
        prompt_type: Union[str, int] = "llama",  # "plain", "llama", "quality", "human_bot", "dai_faq"

        valid_path: str = None,
        # valid_path: str = "./dai_docs.valid.json",

        # data_mix_in_path: str = "laion/OIG",  # way too big, medium quality
        data_mix_in_path: str = "0-hero/OIG-small-chip2",  # high quality, 50 MB, good enough for now
        data_mix_in_factor: float = 1.0,  # >1: more mix-in data, <1: more of data_path data
        data_mix_in_col_dict: dict = {'user': 'instruction', 'chip2': 'output'},
        data_mix_in_prompt_type: str = "llama",  # just instruction->output, same as llama

        output_dir: str = None,

        # LoRA checkpoint continuation
        lora_weights: str = "",

        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # if True, faster, but produces an odd training loss curve
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        # torch training params
        ddp: bool = True,  # set to False if OOM with True, for multi-GPU model parallelism
):
    prompt_type = str(prompt_type)  # migration from integers
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    rank = int(os.getenv("RANK", 0))
    print(f"local_rank: {local_rank}")
    print(f"global rank: {rank}")
    gpus = max(world_size, torch.cuda.device_count())
    if world_size > 1:
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    if output_dir is None:
        output_dir = f"{base_model.split('/')[-1]}.{data_path.replace('/', '')}.{num_epochs}_epochs.{get_githash() or 'nogit'}.{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        # time-based output dir
        if world_size > 1:
            # make sure all workers have same output_dir, otherwise final state is corrupted.
            pickleable = [output_dir]
            dist.broadcast_object_list(pickleable, 0)
            output_dir = pickleable[0]
            del pickleable
    if save_code:
        copy_code(run_id)
    if tokenizer_base_model is None:
        tokenizer_base_model = base_model
    if lora_target_modules is None:
        if "gpt-neox" in base_model.lower():
            lora_target_modules = ["query_key_value"]
        else:
            lora_target_modules = ["q_proj", "v_proj"]
    llama_type = "llama" in base_model.lower()
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"

    locals_dict = locals()
    locals_print = '\n'.join(['%s: %s' % (k, v) for k, v in locals_dict.items()])
    log(f"Training model with params:\n{locals_print}")

    max_memory = None
    if gpus > 1:
        if ddp:
            log("Distributed: data parallel")
            device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
            gradient_accumulation_steps = gradient_accumulation_steps // world_size
        else:
            free_in_GB = int(min(torch.cuda.mem_get_info()) / 1024 ** 3)
            max_memory = f"{free_in_GB - 2}GB"
            max_memory = {i: max_memory for i in range(gpus)}
            log("world_size: %d" % world_size)
            log("num_gpus: %d" % gpus)
            log("max mem: %s" % max_memory)

    model_loader, tokenizer_loader = get_loaders(llama_type=llama_type)

    model = model_loader.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map=device_map,
        max_memory=max_memory,
    )
    if gpus > 1:
        if not ddp:
            log("model parallel")
            model.is_parallelizable = True
            model.model_parallel = True

    tokenizer = tokenizer_loader.from_pretrained(tokenizer_base_model)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt, _, _ = generate_prompt(data_point, prompt_type)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt, _, _ = generate_prompt({**data_point, "output": ""}, prompt_type)
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    if lora_weights:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
    else:
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = False  # So the trainer won't try loading its state
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            log(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            log(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if valid_path:
        data = load_dataset("json", data_files={"train": data_path, "valid": valid_path})
    else:
        data = load_dataset("json", data_files={"train": data_path})

    valid_data = None
    train_data_mix_in = None
    valid_data_mix_in = None

    if data_mix_in_path:
        # get mix-in training/validation data - to keep model "sane"
        num_rows = data["train"].num_rows
        log("Loading mix-in dataset: %s" % data_mix_in_path)
        data_mix_in = load_dataset(data_mix_in_path)["train"]  # can be large
        data_mix_in = data_mix_in.rename_columns(data_mix_in_col_dict or {})

        # only get as much as we need to balance
        train_size = int(num_rows * data_mix_in_factor)
        valid_size = val_set_size or 0
        mixin_small = data_mix_in.train_test_split(
            test_size=train_size + valid_size,
            shuffle=True, seed=np.random.randint(10000),
        )["test"]
        if valid_size:
            mixin_train_test = mixin_small.train_test_split(
                test_size=valid_size, shuffle=False,
            )
            train_data_mix_in = mixin_train_test["train"]
            valid_data_mix_in = mixin_train_test["test"]
        else:
            train_data_mix_in = mixin_small

        if "prompt_type" not in train_data_mix_in.column_names:
            train_data_mix_in = train_data_mix_in.add_column(
                "prompt_type",
                [data_mix_in_prompt_type] * train_data_mix_in.num_rows,
            )
            log("Added prompt type %s to mix-in training data" % data_mix_in_prompt_type)
        if valid_data_mix_in and "prompt_type" not in valid_data_mix_in.column_names:
            valid_data_mix_in = valid_data_mix_in.add_column(
                "prompt_type",
                [data_mix_in_prompt_type] * valid_data_mix_in.num_rows,
            )
            log("Added prompt type %s to mix-in validation data" % data_mix_in_prompt_type)
        log("Created mix-in data:\nTrain %s\nValid %s" % (train_data_mix_in, valid_data_mix_in))

    # get our own training/validation data - for fine-tuning
    if val_set_size > 0 and not valid_path and not data_mix_in_path:
        # create valid split from train
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"]
        valid_data = train_val["test"]
    else:
        train_data = data["train"]
        if valid_path:
            # use given valid split, has priority over data_mix_in_path
            valid_data = data["valid"]
    if "prompt_type" not in train_data.column_names:
        train_data = train_data.add_column(
            "prompt_type",
            [prompt_type] * train_data.num_rows,
        )
        log("Added prompt type %s to training data" % data_mix_in_prompt_type)
    if valid_data and "prompt_type" not in valid_data.column_names:
        valid_data = valid_data.add_column(
            "prompt_type",
            [prompt_type] * valid_data.num_rows,
        )
        log("Added prompt type %s to validation data" % data_mix_in_prompt_type)

    assert train_data is not None

    # shuffle and tokenize data
    if train_data_mix_in:
        train_data = concatenate_datasets([train_data, train_data_mix_in])
    train_data = train_data.shuffle().map(generate_and_tokenize_prompt)

    if valid_data and valid_data_mix_in:
        valid_data = concatenate_datasets([valid_data, valid_data_mix_in])
    elif valid_data_mix_in:
        valid_data = valid_data_mix_in

    if valid_data:
        valid_data = valid_data.shuffle().map(generate_and_tokenize_prompt)
        val_set_size = len(valid_data)
    else:
        val_set_size = 0
    log("Final fine-tuning data:\nTrain %s\nValid %s" % (train_data, valid_data))

    if neptune_run:
        neptune_callback = NeptuneCallback(run=neptune_run)
        callbacks = [neptune_callback]
    else:
        callbacks = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=100 if val_set_size > 0 else None,
            save_steps=100,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=callbacks,
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if gpus > 1 and not ddp:
        assert trainer.is_model_parallel
    else:
        assert not trainer.is_model_parallel
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    log("\n If there's a warning about missing keys above, please disregard :)")


def get_loaders(llama_type):
    if llama_type:
        assert (
                "LlamaTokenizer" in transformers._import_structure["models.llama"]
        ), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
        from transformers import LlamaForCausalLM, LlamaTokenizer

        model_loader = LlamaForCausalLM
        tokenizer_loader = LlamaTokenizer
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_loader = AutoModelForCausalLM
        tokenizer_loader = AutoTokenizer
    return model_loader, tokenizer_loader


def get_githash():
    try:
        githash = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE).stdout.decode('utf-8')[0:-1]
    except:
        githash = ''
    return githash


def copy_code(run_id):
    """
    copy code to track changes
    :param run_id:
    :return:
    """
    rnd_num = str(random.randint(0, 2 ** 31))
    run_id = 'run_' + str(run_id)
    os.makedirs(run_id, exist_ok=True)
    me_full = os.path.join(pathlib.Path(__file__).parent.resolve(), __file__)
    me_file = os.path.basename(__file__)
    new_me = os.path.join(run_id, me_file + '_' + get_githash())
    if os.path.isfile(new_me):
        new_me = os.path.join(run_id, me_file + '_' + get_githash() + '_' + rnd_num)
        shutil.copy(me_full, new_me)
    else:
        shutil.copy(me_full, new_me)


def get_prompt(prompt_type):
    if prompt_type in [-1, "-1", "plain"]:
        promptA = promptB = PreInstruct = PreInput = PreResponse = ''
        terminate_response = []
    elif prompt_type in [0, "0", "llama"]:
        promptA = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n'
        promptB = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n'

        PreInstruct = """
### Instruction:
"""

        PreInput = """
### Input:
"""

        PreResponse = """
### Response:
"""
        terminate_response = None
    elif prompt_type in [1, "1", "quality"]:
        promptA = 'Write a detailed high-quality, accurate, fair, Response with about 100 words by following the Instruction as applied on the Input.\n'
        promptB = 'Write a detailed high-quality, accurate, fair, Response with about 100 words by following the Instruction.\n'

        PreInstruct = """
### Instruction:
"""

        PreInput = """
### Input:
"""

        PreResponse = """
### Response:
"""
        terminate_response = None
    elif prompt_type in [2, "2", "human_bot"]:
        cur_date = time.strftime('%Y-%m-%d')
        cur_time = time.strftime('%H:%M:%S %p %Z')

        PRE_PROMPT = """\
Current Date: {}
Current Time: {}

"""

        preprompt = PRE_PROMPT.format(cur_date, cur_time)
        start = '<human>:'
        promptB = promptA = '%s%s ' % (preprompt, start)

        PreInstruct = ""

        PreInput = None

        PreResponse = "<bot>:"

        terminate_response = [start, PreResponse]
    elif prompt_type in [3, "3", "dai_faq"]:
        promptA = ''
        promptB = 'Answer the following Driverless AI question.\n'

        PreInstruct = """
### Driverless AI frequently asked question:
"""

        PreInput = None

        PreResponse = """
### Driverless AI documentation answer:
"""
        terminate_response = ['\n\n']
    else:
        raise RuntimeError("No such prompt_type=%s" % prompt_type)

    return promptA, promptB, PreInstruct, PreInput, PreResponse, terminate_response


def generate_prompt(data_point, prompt_type):
    instruction = data_point.get('instruction')
    input = data_point.get('input')
    output = data_point.get('output')
    prompt_type = data_point.get('prompt_type', prompt_type)
    promptA, promptB, PreInstruct, PreInput, PreResponse, terminate_response = get_prompt(prompt_type)

    prompt = ''

    if input and promptA:
        prompt += f"""{promptA}"""
    elif promptB:
        prompt += f"""{promptB}"""

    if instruction and PreInstruct is not None and input and PreInput is not None:
        prompt += f"""{PreInstruct}{instruction}{PreInput}{input}
"""
    elif instruction and input and PreInstruct is None and PreInput is not None:
        prompt += f"""{PreInput}{instruction}
{input}
"""
    elif input and instruction and PreInput is None and PreInstruct is not None:
        prompt += f"""{PreInstruct}{instruction}
{input}
"""
    elif instruction and PreInstruct is not None:
        prompt += f"""{PreInstruct}{instruction}
"""
    elif input and PreInput is not None:
        prompt += f"""{PreInput}{input}
"""
    elif input and instruction and PreInput is not None:
        prompt += f"""{PreInput}{instruction}{input}
"""
    elif input and instruction and PreInstruct is not None:
        prompt += f"""{PreInstruct}{instruction}{input}
"""
    elif input and instruction:
        prompt += f"""{PreInput}{instruction}{input}
"""
    elif input:
        prompt += f"""{input}
"""
    elif instruction:
        prompt += f"""{instruction}
"""

    if PreResponse is not None:
        prompt += f"""{PreResponse}"""
        pre_response = PreResponse.strip()
    else:
        pre_response = ''

    if output:
        prompt += f"""{output}"""

    return prompt, pre_response, terminate_response


example_data_point0 = dict(instruction="Summarize",
                           input="Ducks eat seeds by the lake, then swim in the lake where fish eat small animals.",
                           output="Ducks eat and swim at the lake.")

example_data_point1 = dict(instruction="Who is smarter, Einstein or Newton?",
                           output="Einstein.")

example_data_point2 = dict(input="Who is smarter, Einstein or Newton?",
                           output="Einstein.")

example_data_points = [example_data_point0, example_data_point1, example_data_point2]


def test_train_prompt(prompt_type=0, data_point=0):
    example_data_point = example_data_points[data_point]
    return generate_prompt(example_data_point, prompt_type)


def test_debug():
    fire.Fire(train)


if __name__ == "__main__":
    if os.environ.get("LOCAL_RANK") is None:
        # then not using torchrun, so can't do distributed, ensure CVD set
        assert os.environ.get("CUDA_VISIBLE_DEVICES") is not None, "Run python script using: torchrun finetune.py OR set CUDA_VISIBLE_DEVICES to single GPU"

    CONFIG = "NCCL_P2P_LEVEL=LOC WORLD_SIZE=5 torchrun --nnodes=5 --master_addr=10.10.10.2 --master_port=1111 --nproc_per_node=1"
    CMD = "finetune.py --data_path=config.json --num_epochs=1 --base_model=decapoda-research/llama-13b-hf"
    log(f"""
    Example runs on 4 GPUs:
    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port=1234 finetune.py --base_model='decapoda-research/llama-7b-hf' --output_dir='lora_alpaca_7B' --data_path=alpaca_data_cleaned.json --run_id=0 &> 0.log
    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port=1234 finetune.py --base_model='decapoda-research/llama-30b-hf' --output_dir='lora_alpaca_30B' --data_path=alpaca_data_cleaned.json --batch_size=16 --micro_batch_size=1 --run_id=1 --save_code=True &> 1.log
    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port=1234 finetune.py --base_model='EleutherAI/gpt-j-6B' --output_dir='lora_alpaca_6B' --data_path=alpaca_data_cleaned.json --run_id=2 &> 2.log

    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port=1234 finetune.py --base_model='EleutherAI/gpt-neox-20b' --output_dir='lora_alpaca_20B' --data_path=alpaca_data_cleaned.json --lora_target_modules='["query_key_value"]' --run_id=8 --batch_size=16 --micro_batch_size=4 &> 8.log

    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port=1234 finetune.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --output_dir='lora_20B_daifaq' --data_path=dai_faq.json --lora_target_modules='["query_key_value"]' --prompt_type=3 --run_id=13 --batch_size=16 --micro_batch_size=4 --num_epochs=100 --val_set_size=0 data_mix_in_path='' &> 13.log

    Example run on 3 nodes with 1 to 2 GPU each (we'll consider SLURM etc.)

    rippa> CUDA_VISIBLE_DEVICES=0 {CONFIG} --node_rank=0 {CMD} &>log.rank.0
    rippa> CUDA_VISIBLE_DEVICES=1 {CONFIG} --node_rank=1 {CMD} &>log.rank.1
    ova> CUDA_VISIBLE_DEVICES=0 {CONFIG} --node_rank=2 {CMD} &>log.rank.2
    ova> CUDA_VISIBLE_DEVICES=1 {CONFIG} --node_rank=3 {CMD} &>log.rank.3
    timemachine> CUDA_VISIBLE_DEVICES=0 {CONFIG} --node_rank=4 {CMD} &>log.rank.4
    """, flush=True)
    fire.Fire(train)
