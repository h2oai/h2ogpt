import os
import sys
from functools import partial
from typing import List, Union
import fire
import numpy as np

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if os.path.dirname('src') not in sys.path:
    sys.path.append('src')

from src.loaders import get_loaders, get_tokenizer
from src.prompter import generate_prompt, prompt_types, PromptType
from src.utils import get_githash, copy_code
import torch


def log(*args, **kwargs):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        if 'flush' not in kwargs:
            kwargs['flush'] = True
        print(*args, **kwargs)


# supported by huggingface evaluate
supported_metrics = ['bleu', 'rouge', 'sacrebleu', 'meteor']


def train(
        save_code: bool = False,
        run_id: int = None,

        base_model: str = 'h2oai/h2ogpt-oig-oasst1-512-6_9b',
        # base_model: str = 'h2oai/h2ogpt-oasst1-512-12b',
        # base_model: str = 'h2oai/h2ogpt-oasst1-512-20b',
        # base_model: str = 'EleutherAI/gpt-neox-20b',
        # base_model: str = 'EleutherAI/pythia-12b-deduped',
        # base_model: str = 'togethercomputer/GPT-NeoXT-Chat-Base-20B',
        # base_model: str = 'decapoda-research/llama-7b-hf',
        # base_model: str = 'decapoda-research/llama-13b-hf',
        # base_model: str = 'decapoda-research/llama-30b-hf',
        # base_model: str = 'EleutherAI/gpt-j-6B',

        # only needed if base_model is self-exported HF state without tokenizer
        tokenizer_base_model: str = None,
        # tokenizer_base_model: str = 'EleutherAI/gpt-neox-20b',

        data_path: str = "h2oai/openassistant_oasst1_h2ogpt",
        data_col_dict: dict = None,
        # data_path: str = "./dai_docs.train.json",
        prompt_type: Union[str, int] = "plain",  # "plain", "instruct", "quality", "human_bot", "dai_faq"

        valid_path: str = None,
        # valid_path: str = "./dai_docs.valid.json",

        # data_mix_in_path: str = "laion/OIG",  # way too big, medium quality
        data_mix_in_path: str = "0-hero/OIG-small-chip2",  # high quality, 50 MB, good enough for now
        data_mix_in_factor: float = 0.0,  # >1: more mix-in data, <1: more of data_path data
        data_mix_in_col_dict: dict = {'user': 'instruction', 'chip2': 'output'},
        data_mix_in_prompt_type: str = "instruct",  # just instruction->output, same as instruct

        output_dir: str = None,

        # LoRA checkpoint continuation
        lora_weights: str = "",

        # batching training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        gradient_checkpointing=False,  # unnecessary with gradient accumulation enabled
        fp16=True,
        train_8bit=False,
        train_4bit=False,

        # general training hyperparams
        num_epochs: float = 1,
        learning_rate: float = 3e-4,

        # validation settings
        val_set_size: int = None,
        val_metrics: List[str] = [],
        eval_steps: int = None,  # to control eval steps via steps
        eval_epochs: float = None,  # to control eval steps via epochs

        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        llama_type: bool = None,
        llama_flash_attn: bool = False,

        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # if True, faster, but produces an odd training loss curve
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        cutoff_len: int = 512,  # larger values use more memory
        drop_truncations: bool = False,  # if True, drop any truncated long sequences

        # torch training params
        ddp: bool = True,  # set to False if OOM with True, for multi-GPU model parallelism
        local_files_only: bool = False,  # else will download new versions, normally unwanted
        resume_download: bool = True,
        use_auth_token: Union[str, bool] = False,  # True requires CLI did huggingface-cli login before running
        warmup_steps: int = 100,
        logging_steps: int = 1,
        save_steps: int = None,  # must be round multiple of eval_steps
        save_total_limit: int = 3,
        add_eos_token: bool = False,
):
    if llama_flash_attn:
        # Need to call this before importing transformers.
        from src.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    # allow set token directly
    use_auth_token = os.environ.get("HUGGINGFACE_API_TOKEN", use_auth_token)

    prompt_type = str(prompt_type)  # migration from integers
    assert prompt_type in prompt_types

    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    rank = int(os.getenv("RANK", 0))
    print(f"local_rank: {local_rank}")
    print(f"global rank: {rank}")

    gpus = max(world_size, torch.cuda.device_count())
    run_id = run_id or 0
    if not data_path:
        raise ValueError("No data_path provided")
    if not output_dir:
        output_dir = f"{base_model.split('/')[-1]}.{data_path.replace('/', '')}.{num_epochs}_epochs.{get_githash() or 'nogit'}.{run_id}"
        if os.path.exists(output_dir) and not resume_from_checkpoint:
            raise FileExistsError(
                f"output_dir {output_dir} based on run_id {run_id} already exists. Please pick a different run_id.")
    else:
        if os.path.exists(output_dir) and not resume_from_checkpoint:
            raise FileExistsError(
                f"output_dir {output_dir} already exists. Please pick a different output_dir, or specify a run_id instead.")
    device_map = "auto"

    if save_code:
        copy_code(run_id)
    if tokenizer_base_model is None:
        tokenizer_base_model = base_model
    if llama_type is None:
        llama_type = "llama" in base_model.lower()
    if llama_type and llama_flash_attn:
        import pkg_resources
        try:
            pkg_resources.get_distribution('flash_attn')
            can_do_flash_attn = True
        except (pkg_resources.DistributionNotFound, pkg_resources.ContextualVersionConflict):
            can_do_flash_attn = False

        if not can_do_flash_attn:
            raise RuntimeError("""Flash attention not installed.
            NOTE: for current pytorch 2.0, flash attention requires installing cuda 11.7 via https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local and then when running, to avoid installing driver, docs, samples, just install toolkit.  Then when pip installing flash attention do:

            CUDA_HOME=/usr/local/cuda-11.8 pip install flash-attn""")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    assert gradient_accumulation_steps >= world_size, "must increase batch_size for multi-GPU"

    device_map = "auto"

    locals_dict = locals()
    locals_print = '\n'.join(['%s: %s' % (k, v) for k, v in locals_dict.items()])
    log(f"Training model with params:\n{locals_print}")
    log("Command: %s\nHash: %s" % (str(' '.join(sys.argv)), get_githash()))

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

    model_loader, tokenizer_loader = get_loaders(model_name=base_model, reward_type=False, llama_type=llama_type)

    model = model_loader(
        base_model,
        load_in_8bit=train_8bit,
        load_in_4bit=train_4bit,
        device_map=device_map,
        torch_dtype=torch.float16,
        max_memory=max_memory,
        local_files_only=local_files_only,
        trust_remote_code=True,
        resume_download=resume_download,
        use_auth_token=use_auth_token,
    )
    if gpus > 1:
        if not ddp:
            log("model parallel")
            model.is_parallelizable = True
            model.model_parallel = True

    tokenizer = get_tokenizer(tokenizer_loader, tokenizer_base_model, local_files_only, resume_download, use_auth_token)

    if train_8bit or train_4bit:
        from peft import (
            prepare_model_for_kbit_training,
        )

        model = prepare_model_for_kbit_training(model)

    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    try:
        from peft import utils
        lora_mappings = utils.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.copy()
    except AttributeError:
        from peft import mapping
        lora_mappings = mapping.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.copy()
    lora_mappings['distilgpt2'] = ["c_attn"]

    if lora_weights:

        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map=device_map,
            local_files_only=local_files_only,
            resume_download=resume_download,
            use_auth_token=use_auth_token,
        )
    elif lora_r > 0:
        if lora_target_modules is None:
            base_model_lower = base_model.lower()
            if base_model_lower in lora_mappings:
                lora_target_modules_cand = [lora_mappings[base_model_lower]]
            else:
                lora_target_modules_cand = [["query_key_value"], ["q_proj", "v_proj"]]
        else:
            lora_target_modules_cand = [lora_target_modules]

        for lora_target_modules in lora_target_modules_cand:
            try:
                config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, config)
                break
            except ValueError as e:
                if "Target modules" in str(e) and "not found" in str(e):
                    continue
                else:
                    raise
        from peft import PeftModel
        assert isinstance(model, PeftModel), "LoRA failed. Please provide --lora_target_modules explicitly."
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
            set_peft_model_state_dict(model, adapters_weights)
        else:
            log(f"Checkpoint {checkpoint_name} not found")

    print(model)
    try:
        # only for PeftModel
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    except:
        pass

    metrics = {}
    for name in supported_metrics:
        if name in val_metrics:
            import evaluate  # Causes hang for 'python generate.py' on dual 4090 if imported early, 100% reproducible
            metrics[name] = evaluate.load(name)
    log("Using Validation Metrics: %s" % str(list(metrics.keys())))
    log("Supported Metrics: %s" % supported_metrics)

    if val_set_size is None:
        if len(metrics) == 0:
            val_set_size = 1000
        else:
            val_set_size = 100
        log("Auto set val_set_size %s" % val_set_size)
    elif val_set_size < 1.0 and val_set_size != 0:
        raise RuntimeError("Fractional validation size not supported.")

    from datasets import load_dataset, concatenate_datasets
    if valid_path:
        data = load_dataset("json", data_files={"train": data_path, "valid": valid_path})
    else:
        if "json" in data_path:
            data = load_dataset("json", data_files={"train": data_path})
        else:
            data = load_dataset(data_path)
            data = data.rename_columns(data_col_dict or {})

    valid_data = None
    train_data_mix_in = None
    valid_data_mix_in = None

    if data_mix_in_path and data_mix_in_factor > 0:
        # get mix-in training/validation data - to keep model "sane"
        num_rows = data["train"].num_rows
        log("Loading mix-in dataset: %s" % data_mix_in_path)
        if "json" in data_mix_in_path:
            data_mix_in = load_dataset("json", data_files={"train": data_mix_in_path})["train"]
        else:
            data_mix_in = load_dataset(data_mix_in_path)["train"]  # can be large
        data_mix_in = data_mix_in.rename_columns(data_mix_in_col_dict or {})
        mix_in_rows = int(num_rows * data_mix_in_factor)

        if mix_in_rows > data_mix_in.num_rows:
            # duplicate rows if mix-in is smaller than required
            log("Duplicating mixin to compensate for its size for training size and mixin fraction")
            data_mix_in = concatenate_datasets([data_mix_in] * int(np.ceil(mix_in_rows / data_mix_in.num_rows)))

        # only get as much as we need to balance
        valid_size = min(data_mix_in.num_rows // 2, val_set_size or 0)
        train_size = max(1, min(data_mix_in.num_rows - valid_size, mix_in_rows))
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
        log("Added prompt type %s to training data" % prompt_type)
    if valid_data and "prompt_type" not in valid_data.column_names:
        valid_data = valid_data.add_column(
            "prompt_type",
            [prompt_type] * valid_data.num_rows,
        )
        log("Added prompt type %s to validation data" % prompt_type)

    assert train_data is not None

    generate_and_tokenize_prompt_fun = partial(generate_and_tokenize_prompt, prompt_type=prompt_type,
                                               train_on_inputs=train_on_inputs, add_eos_token=add_eos_token,
                                               cutoff_len=cutoff_len, tokenizer=tokenizer)

    # shuffle and tokenize data
    if train_data_mix_in:
        train_data = concatenate_datasets([train_data, train_data_mix_in])
    log("Tokenizing %s training rows" % train_data.num_rows)
    train_data = train_data.shuffle().map(generate_and_tokenize_prompt_fun,
                                          num_proc=os.cpu_count() // torch.cuda.device_count())
    if drop_truncations:
        log("avoid keeping truncated cases to avoid contaminating model with truncation cases.  Original size: %s" % train_data.num_rows)
        prune_long_sequences_func = partial(prune_long_sequences, cutoff_len=cutoff_len)
        train_data = train_data.filter(prune_long_sequences_func, num_proc=os.cpu_count() // torch.cuda.device_count())
        log("avoid keeping truncated cases to avoid contaminating model with truncation cases.  New size: %s" % train_data.num_rows)
    train_set_size = len(train_data)

    if valid_data and valid_data_mix_in:
        valid_data = concatenate_datasets([valid_data, valid_data_mix_in])
    elif valid_data_mix_in:
        valid_data = valid_data_mix_in

    if valid_data:
        log("Tokenizing %s validation rows" % valid_data.num_rows)
        valid_data = valid_data.shuffle().map(generate_and_tokenize_prompt_fun,
                                              num_proc=os.cpu_count() // torch.cuda.device_count())
        val_set_size = len(valid_data)
    else:
        val_set_size = 0
    log("Final fine-tuning data:\nTrain %s\nValid %s" % (train_data, valid_data))
    sample_row_dict = train_data[:1]
    del sample_row_dict['input_ids']
    del sample_row_dict['attention_mask']
    del sample_row_dict['labels']
    log("Sample input: %s" % sample_row_dict)

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

    if neptune_run:
        neptune_callback = NeptuneCallback(run=neptune_run)
        callbacks = [neptune_callback]
    else:
        from transformers.integrations import TensorBoardCallback, is_tensorboard_available
        if is_tensorboard_available:
            # tensorboard --logdir=runs/
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter()
            callbacks = [TensorBoardCallback(tb_writer=tb_writer)]
        else:
            callbacks = []

    expected_steps = (train_set_size * num_epochs) // batch_size
    if eval_steps is None and eval_epochs is None:
        # 20 evaluations for a run
        eval_steps = max(1, int(expected_steps / 20))
        log("Auto set eval_steps to %s out of %s total training steps" % (eval_steps, expected_steps))
    elif eval_steps is None and eval_epochs is not None:
        eval_steps = max(1, int(expected_steps * eval_epochs / num_epochs))
        log("Auto converted eval_epochs=%s to eval_steps %s"
            " out of %s total training steps" % (eval_epochs, eval_steps, expected_steps))
    if save_steps is None:
        save_steps = eval_steps
        log("Auto step save_steps to %s" % save_steps)
    elif save_steps > eval_steps:
        # save steps must be round multiple of eval_steps
        save_steps0 = save_steps
        save_steps = max(1, (save_steps // eval_steps)) * eval_steps
        if save_steps0 != save_steps:
            log("Auto converted save_steps from %s to %s" % (save_steps0, save_steps))

    def compute_metrics(eval_preds):
        # e.g. see: https://huggingface.co/docs/transformers/v4.25.1/en/tasks/translation#evaluate
        inputs = eval_preds.inputs
        label_ids = eval_preds.label_ids
        predictions = eval_preds.predictions

        # inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        # decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        # decoded_inputs = [pred.strip() for pred in decoded_inputs]

        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        # tokenizer behavior like generate time
        decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True)
        decoded_labels = [pred.strip() for pred in decoded_labels]

        predictions = np.argmax(predictions, -1)
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        # tokenizer behavior like generate time
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=True)
        decoded_predictions = [pred.strip() for pred in decoded_predictions]

        result = {}
        for metric in metrics.values():
            result1 = metric.compute(predictions=decoded_predictions, references=decoded_labels)
            # get rid of lists, for precision etc., for now
            numeric_results = {k: v for k, v in result1.items() if isinstance(v, (int, float))}
            result.update(numeric_results)
        return result

    # the callback that computes metrics of interest
    if val_metrics:
        trainer_kwargs = dict(compute_metrics=compute_metrics)
    else:
        trainer_kwargs = dict()

    import transformers
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=valid_data,
        # FIXME: might need Seq2SeqTrainingArguments for some models
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=1,
            eval_accumulation_steps=10,
            # predict_with_generate=True,  # SEQ2SEQ only
            include_inputs_for_metrics=True,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            gradient_checkpointing=gradient_checkpointing,
            fp16=fp16,
            # cosnider 8-bit adam: https://huggingface.co/docs/transformers/v4.18.0/en/performance#8bit-adam
            optim="adamw_torch",  # consider "adafactor" to save memory
            logging_steps=logging_steps,
            logging_strategy="steps",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if val_set_size > 0 else None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            # fsdp="shard_grad_op auto_wrap" if gpus > 1 and not ddp else None,
            # fsdp_min_num_params=20000 if gpus > 1 and not ddp else None,
            report_to='tensorboard' if not neptune_run else 'neptune',
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=callbacks,
        **trainer_kwargs,
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
        # WIP (not generally replacing layers until pytorch 2.1)
        if not llama_flash_attn:
            torch.backends.cuda.enable_flash_sdp(True)

    if gpus > 1 and not ddp:
        assert trainer.is_model_parallel
    else:
        assert not trainer.is_model_parallel
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    log("\n If there's a warning about missing keys above, please disregard :)")


def tokenize(prompt, tokenizer, cutoff_len, add_eos_token=False):
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


def prune_long_sequences(data_point, cutoff_len=None):
    """
    Prune if too long for tokenizer, so truncation doesn't lead training to learn from truncated language
    :param data_point:
    :param cutoff_len:
    :return:
    """
    assert cutoff_len is not None
    return len(data_point['input_ids']) < cutoff_len


def generate_and_tokenize_prompt(data_point, prompt_type=None, train_on_inputs=False, add_eos_token=False,
                                 cutoff_len=None, tokenizer=None):
    assert prompt_type is not None
    assert cutoff_len is not None
    assert tokenizer is not None
    prompt_dict = ''  # only for custom prompt_type
    assert prompt_type != PromptType.custom.name, "custom not setup for finetune"
    full_prompt, _, _, _, _ = generate_prompt(data_point, prompt_type, prompt_dict, False, False, False)
    tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len, add_eos_token=add_eos_token)
    if not train_on_inputs:
        user_prompt, _, _, _, _ = generate_prompt({**data_point, "output": ""}, prompt_type, prompt_dict, False, False, False)
        tokenized_user_prompt = tokenize(user_prompt, tokenizer, cutoff_len, add_eos_token=add_eos_token)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if add_eos_token:
            user_prompt_len -= 1

        # ignore_index=-100 ensures torch/tf don't include padding token id in CrossEntropyLoss
        tokenized_full_prompt["labels"] = [
                                              -100
                                          ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                user_prompt_len:
                                                                ]  # could be sped up, probably
    return tokenized_full_prompt


def test_debug():
    fire.Fire(train)


def entrypoint_main():
    CONFIG = "NCCL_P2P_LEVEL=LOC WORLD_SIZE=5 torchrun --nnodes=5 --master_addr=10.10.10.2 --master_port=1111 --nproc_per_node=1"
    CMD = "finetune.py --data_path=config.json --num_epochs=1 --base_model=decapoda-research/llama-13b-hf"
    log(f"""
    Example runs on 4 GPUs:
    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 finetune.py --base_model='decapoda-research/llama-7b-hf' --data_path=data/config.json --run_id=0 &> 0.log
    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 finetune.py --base_model='decapoda-research/llama-30b-hf' --data_path=data/config.json --batch_size=16 --micro_batch_size=1 --run_id=1 --save_code=True &> 1.log
    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 finetune.py --base_model='EleutherAI/gpt-j-6B' --data_path=data/config.json --run_id=2 &> 2.log
    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 finetune.py --base_model='EleutherAI/gpt-neox-20b' --data_path=data/config.json --run_id=8 --batch_size=16 --micro_batch_size=4 &> 8.log
    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 finetune.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --data_path=data/config.json --prompt_type='dai_faq' --run_id=13 --batch_size=16 --micro_batch_size=4 --num_epochs=100 --val_set_size=0 data_mix_in_path='' &> 13.log
    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 finetune.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --data_path=data/config.json --run_id=28 --batch_size=16 --micro_batch_size=4 --num_epochs=8 --val_set_size=0 --data_mix_in_factor=0.1 --data_mix_in_prompt_type='human_bot' --save_code=True --cutoff_len=512  &> 28.log

    All metrics:
    CUDA_VISIBLE_DEVICES= finetune.py --data_mix_in_factor=0 --eval_steps=100 --warmup_steps=2 --val_set_size=100 --val_metrics="['bleu', 'rouge', 'sacrebleu', 'meteor']"

    # Fine-tune 20B on 24GB GPUs across 3 nodes with 3+2+2 GPUs
    rippa>
NCCL_P2P_LEVEL=LOC WORLD_SIZE=7 CUDA_VISIBLE_DEVICES="0,1,2" torchrun --node_rank 0 --nproc_per_node=3 --master_port=1234 --nnodes=3 --master_addr=10.10.10.2 finetune.py --data_path=merged_shuffled_OIG_87f6a1e788.json --micro_batch_size=1 --batch_size=7 --cutoff_len=512 --run_id=17 &>log.17.rank0
    ova>
NCCL_P2P_LEVEL=LOC WORLD_SIZE=7 CUDA_VISIBLE_DEVICES="0,1" torchrun --node_rank 1 --nproc_per_node=2 --master_port=1234 --nnodes=3 --master_addr=10.10.10.2 finetune.py --data_path=merged_shuffled_OIG_87f6a1e788.json --micro_batch_size=1 --batch_size=7 --cutoff_len=512 --run_id=17 &>log.17.rank1
    timemachine>
NCCL_P2P_LEVEL=LOC WORLD_SIZE=7 CUDA_VISIBLE_DEVICES="0,1" torchrun --node_rank 2 --nproc_per_node=2 --master_port=1234 --nnodes=3 --master_addr=10.10.10.2 finetune.py --data_path=merged_shuffled_OIG_87f6a1e788.json --micro_batch_size=1 --batch_size=7 --cutoff_len=512 --run_id=17 &>log.17.rank2

    """, flush=True)

    if os.environ.get("LOCAL_RANK") is None:
        # then not using torchrun, so can't do distributed, ensure CVD set
        assert os.environ.get(
            "CUDA_VISIBLE_DEVICES") is not None, "Run python script using: torchrun finetune.py OR set CUDA_VISIBLE_DEVICES to single GPU"

    fire.Fire(train)


if __name__ == "__main__":
    entrypoint_main()
