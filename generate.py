import ast
import functools
import glob
import queue
import shutil
import sys
import os
import time
import traceback
import typing
from datetime import datetime
import psutil
from auto_gptq import AutoGPTQForCausalLM

from utils import set_seed, clear_torch_cache, save_generate_output, NullContext, wrapped_partial, EThread, get_githash, \
    import_matplotlib

import_matplotlib()
from matplotlib import pyplot as plt

SEED = 1236
set_seed(SEED)

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
from typing import Union
import numpy as np
import pandas as pd

import fire
import torch
from peft import PeftModel
from transformers import GenerationConfig, AutoModel, TextIteratorStreamer
from accelerate import init_empty_weights, infer_auto_device_map

from prompter import Prompter, inv_prompt_type_to_model_lower, generate_prompt
from finetune import get_loaders, example_data_points
from stopping import get_stopping

eval_extra_columns = ['prompt', 'response', 'score']

langchain_modes = ['Disabled', 'ChatLLM', 'LLM', 'All', 'wiki', 'wiki_full', 'UserData', 'MyData', 'github h2oGPT',
                   'DriverlessAI docs']

scratch_base_dir = '/tmp/'


def main(
        load_8bit: bool = False,
        load_half: bool = True,
        infer_devices: bool = True,
        base_model: str = '',
        quant_model: str = '',
        tokenizer_base_model: str = '',
        lora_weights: str = "",
        gpu_id: int = 0,

        prompt_type: Union[int, str] = None,
        # input to generation
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        num_beams: int = None,
        repetition_penalty: float = None,
        num_return_sequences: int = None,
        do_sample: bool = None,
        max_new_tokens: int = None,
        min_new_tokens: int = None,
        early_stopping: Union[bool, str] = None,
        max_time: float = None,

        debug: bool = False,
        save_dir: str = None,
        share: bool = True,
        local_files_only: bool = False,
        resume_download: bool = True,
        use_auth_token: Union[str, bool] = False,
        trust_remote_code: Union[str, bool] = True,

        src_lang: str = "English",
        tgt_lang: str = "Russian",

        gradio: bool = True,
        gradio_avoid_processing_markdown: bool = False,
        chat: bool = True,
        chat_context: bool = False,
        stream_output: bool = True,
        show_examples: bool = None,
        verbose: bool = False,
        h2ocolors: bool = True,
        height: int = 400,
        show_lora: bool = True,
        login_mode_if_model0: bool = False,
        block_gradio_exit: bool = True,
        concurrency_count: int = 1,
        api_open: bool = False,
        allow_api: bool = True,
        input_lines: int = 1,
        auth: typing.List[typing.Tuple[str, str]] = None,

        sanitize_user_prompt: bool = True,
        sanitize_bot_response: bool = True,

        extra_model_options: typing.List[str] = [],
        extra_lora_options: typing.List[str] = [],

        score_model: str = 'OpenAssistant/reward-model-deberta-v3-large-v2',
        auto_score: bool = True,

        eval_sharegpt_prompts_only: int = 0,
        eval_sharegpt_prompts_only_seed: int = 1234,
        eval_sharegpt_as_output: bool = False,

        langchain_mode: str = 'Disabled',
        visible_langchain_modes: list = ['UserData', 'MyData'],
        user_path: str = None,
        load_db_if_exists: bool = True,
        keep_sources_in_context: bool = False,
        db_type: str = 'chroma',
        use_openai_embedding: bool = False,
        allow_upload_to_user_data: bool = True,
        allow_upload_to_my_data: bool = True,
):
    """

    :param load_8bit: load model in 8-bit using bitsandbytes
    :param load_half: load model in float16
    :param infer_devices: whether to control devices with gpu_id.  If False, then spread across GPUs
    :param base_model: model HF-type name
    :param tokenizer_base_model: tokenizer HF-type name
    :param lora_weights: LORA weights path/HF link
    :param gpu_id: if infer_devices, then use gpu_id for cuda device ID, or auto mode if gpu_id != -1
    :param prompt_type: type of prompt, usually matched to fine-tuned model or plain for foundational model
    :param temperature: generation temperature
    :param top_p: generation top_p
    :param top_k: generation top_k
    :param num_beams: generation number of beams
    :param repetition_penalty: generation repetition penalty
    :param num_return_sequences: generation number of sequences (1 forced for chat)
    :param do_sample: generation sample
    :param max_new_tokens: generation max new tokens
    :param min_new_tokens: generation min tokens
    :param early_stopping: generation early stopping
    :param max_time: maximum time to allow for generation
    :param debug: enable debug mode
    :param save_dir: directory chat data is saved to
    :param share: whether to share the gradio app with sharable URL
    :param local_files_only: whether to only use local files instead of doing to HF for models
    :param resume_download: whether to resume downloads from HF for models
    :param use_auth_token: whether to use HF auth token (requires CLI did huggingface-cli login before)
    :param trust_remote_code: whether to use trust any code needed for HF model
    :param src_lang: source languages to include if doing translation (None = all)
    :param tgt_lang: target languages to include if doing translation (None = all)
    :param gradio: whether to enable gradio, or to enable benchmark mode
    :param gradio_avoid_processing_markdown:
    :param chat: whether to enable chat mode with chat history
    :param chat_context: whether to use extra helpful context if human_bot
    :param stream_output: whether to stream output from generate
    :param show_examples: whether to show clickable examples in gradio
    :param verbose: whether to show verbose prints
    :param h2ocolors: whether to use H2O.ai theme
    :param height: height of chat window
    :param show_lora: whether to show LORA options in UI (expert so can be hard to understand)
    :param login_mode_if_model0: set to True to load --base_model after client logs in, to be able to free GPU memory when model is swapped
    :param block_gradio_exit: whether to block gradio exit (used for testing)
    :param concurrency_count: gradio concurrency count (1 is optimal for LLMs)
    :param api_open: If False, don't let API calls skip gradio queue
    :param allow_api: whether to allow API calls at all to gradio server
    :param input_lines: how many input lines to show for chat box (>1 forces shift-enter for submit, else enter is submit)
    :param auth: gradio auth for launcher in form [(user1, pass1), (user2, pass2), ...]
                 e.g. --auth=[('jon','password')] with no spaces
    :param sanitize_user_prompt: whether to remove profanity from user input
    :param sanitize_bot_response: whether to remove profanity and repeat lines from bot output
    :param extra_model_options: extra models to show in list in gradio
    :param extra_lora_options: extra LORA to show in list in gradio
    :param score_model: which model to score responses (None means no scoring)
    :param auto_score: whether to automatically score responses
    :param eval_sharegpt_prompts_only: for no gradio benchmark, if using ShareGPT prompts for eval
    :param eval_sharegpt_prompts_only_seed: for no gradio benchmark, if seed for ShareGPT sampling
    :param eval_sharegpt_as_output: for no gradio benchmark, whether to test ShareGPT output itself
    :param langchain_mode: Data source to include.  Choose "UserData" to only consume files from make_db.py.
           WARNING: wiki_full requires extra data processing via read_wiki_full.py and requires really good workstation to generate db, unless already present.
    :param user_path: user path to glob from to generate db for vector search, for 'UserData' langchain mode
    :param visible_langchain_modes: dbs to generate at launch to be ready for LLM
           Can be up to ['wiki', 'wiki_full', 'UserData', 'MyData', 'github h2oGPT', 'DriverlessAI docs']
           But wiki_full is expensive and requires preparation
           To allow scratch space only live in session, add 'MyData' to list
           Default: If only want to consume local files, e.g. prepared by make_db.py, only include ['UserData']
           FIXME: Avoid 'All' for now, not implemented
    :param load_db_if_exists: Whether to load chroma db if exists or re-generate db
    :param keep_sources_in_context: Whether to keep url sources in context, not helpful usually
    :param db_type: 'faiss' for in-memory or 'chroma' for persisted on disk
    :param use_openai_embedding: Whether to use OpenAI embeddings for vector db
    :param allow_upload_to_user_data: Whether to allow file uploads to update shared vector db
    :param allow_upload_to_my_data: Whether to allow file uploads to update scratch vector db
    :return:
    """
    is_hf = bool(os.getenv("HUGGINGFACE_SPACES"))
    is_gpth2oai = bool(os.getenv("GPT_H2O_AI"))
    is_public = is_hf or is_gpth2oai  # multi-user case with fixed model and disclaimer
    is_low_mem = is_hf  # assumes run on 24GB consumer GPU
    admin_pass = os.getenv("ADMIN_PASS")
    # will sometimes appear in UI or sometimes actual generation, but maybe better than empty result
    # but becomes unrecoverable sometimes if raise, so just be silent for now
    raise_generate_gpu_exceptions = True

    # allow set token directly
    use_auth_token = os.environ.get("HUGGINGFACE_API_TOKEN", use_auth_token)
    allow_upload_to_user_data = bool(os.environ.get("allow_upload_to_user_data", allow_upload_to_user_data))
    allow_upload_to_my_data = bool(os.environ.get("allow_upload_to_my_data", allow_upload_to_my_data))
    height = os.environ.get("HEIGHT", height)

    # allow enabling langchain via ENV
    # FIRST PLACE where LangChain referenced, but no imports related to it
    langchain_mode = os.environ.get("LANGCHAIN_MODE", langchain_mode)
    assert langchain_mode in langchain_modes, "Invalid langchain_mode %s" % langchain_mode
    visible_langchain_modes = ast.literal_eval(os.environ.get("visible_langchain_modes", str(visible_langchain_modes)))
    if langchain_mode not in visible_langchain_modes and langchain_mode in langchain_modes:
        visible_langchain_modes += [langchain_mode]

    if is_public:
        allow_upload_to_user_data = False
        input_lines = 1  # ensure set, for ease of use
        temperature = 0.2 if temperature is None else temperature
        top_p = 0.85 if top_p is None else top_p
        top_k = 70 if top_k is None else top_k
        if is_hf:
            do_sample = True if do_sample is None else do_sample
        else:
            # by default don't sample, too chatty
            do_sample = False if do_sample is None else do_sample

        if is_low_mem:
            if not base_model:
                base_model = 'h2oai/h2ogpt-oasst1-512-12b'
                # don't set load_8bit if passed base_model, doesn't always work so can't just override
                load_8bit = True
        else:
            base_model = 'h2oai/h2ogpt-oasst1-512-20b' if not base_model else base_model
    if is_low_mem:
        load_8bit = True
    if is_hf:
        # must override share if in spaces
        share = False
    save_dir = os.getenv('SAVE_DIR', save_dir)
    score_model = os.getenv('SCORE_MODEL', score_model)
    if score_model == 'None':
        score_model = ''
    concurrency_count = int(os.getenv('CONCURRENCY_COUNT', concurrency_count))
    api_open = bool(int(os.getenv('API_OPEN', api_open)))
    allow_api = bool(int(os.getenv('ALLOW_API', allow_api)))

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available else 0
    if n_gpus == 0:
        gpu_id = None
        load_8bit = False
        load_half = False
        infer_devices = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = False
        torch.set_default_dtype(torch.float32)
        if psutil.virtual_memory().available < 94 * 1024 ** 3:
            # 12B uses ~94GB
            # 6.9B uses ~47GB
            base_model = 'h2oai/h2ogpt-oig-oasst1-512-6.9b' if not base_model else base_model

    # get defaults
    model_lower = base_model.lower()
    if not gradio:
        # force, else not single response like want to look at
        stream_output = False
        # else prompt removal can mess up output
        chat = False

    placeholder_instruction, placeholder_input, \
        stream_output, show_examples, \
        prompt_type, temperature, top_p, top_k, num_beams, \
        max_new_tokens, min_new_tokens, early_stopping, max_time, \
        repetition_penalty, num_return_sequences, \
        do_sample, \
        src_lang, tgt_lang, \
        examples, \
        task_info = \
        get_generate_params(model_lower, chat,
                            stream_output, show_examples,
                            prompt_type, temperature, top_p, top_k, num_beams,
                            max_new_tokens, min_new_tokens, early_stopping, max_time,
                            repetition_penalty, num_return_sequences,
                            do_sample,
                            )

    locals_dict = locals()
    locals_print = '\n'.join(['%s: %s' % (k, v) for k, v in locals_dict.items()])
    print(f"Generating model with params:\n{locals_print}", flush=True)
    print("Command: %s\nHash: %s" % (str(' '.join(sys.argv)), get_githash()), flush=True)

    if langchain_mode != "Disabled":
        # SECOND PLACE where LangChain referenced, but all imports are kept local so not required
        from gpt_langchain import prep_langchain, get_some_dbs_from_hf
        if is_hf:
            get_some_dbs_from_hf()
        dbs = {}
        for langchain_mode1 in visible_langchain_modes:
            if langchain_mode1 in ['MyData']:
                # don't use what is on disk, remove it instead
                for gpath1 in glob.glob(os.path.join(scratch_base_dir, 'db_dir_%s*' % langchain_mode1)):
                    if os.path.isdir(gpath1):
                        print("Removing old MyData: %s" % gpath1, flush=True)
                        shutil.rmtree(gpath1)
                continue
            if langchain_mode1 in ['All']:
                # FIXME: All should be avoided until scans over each db, shouldn't be separate db
                continue
            persist_directory1 = 'db_dir_%s' % langchain_mode1  # single place, no special names for each case
            db = prep_langchain(persist_directory1, load_db_if_exists, db_type, use_openai_embedding, langchain_mode1, user_path)
            dbs[langchain_mode1] = db
        # remove None db's so can just rely upon k in dbs for if hav db
        dbs = {k: v for k, v in dbs.items() if v is not None}
    else:
        dbs = {}

    if not gradio:
        if eval_sharegpt_prompts_only > 0:
            # override default examples with shareGPT ones for human-level eval purposes only
            eval_filename = 'ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json'
            if not os.path.isfile(eval_filename):
                os.system(
                    'wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/%s' % eval_filename)
            import json
            data = json.load(open(eval_filename, 'rt'))
            # focus on data that starts with human, else likely chopped from other data
            turn_start = 0  # odd in general
            data = [x for x in data if len(x['conversations']) > turn_start + 1 and
                    x['conversations'][turn_start]['from'] == 'human' and
                    x['conversations'][turn_start + 1]['from'] == 'gpt']
            np.random.seed(eval_sharegpt_prompts_only_seed)
            example1 = examples[-1]  # pick reference example
            examples = []
            responses = []
            for i in list(np.random.randint(0, len(data), size=eval_sharegpt_prompts_only)):
                assert data[i]['conversations'][turn_start]['from'] == 'human'
                instruction = data[i]['conversations'][turn_start]['value']
                assert data[i]['conversations'][turn_start + 1]['from'] == 'gpt'
                output = data[i]['conversations'][turn_start + 1]['value']
                examplenew = example1.copy()
                assert not chat, "No gradio must use chat=False, uses nochat instruct"
                examplenew[eval_func_param_names.index('instruction_nochat')] = instruction
                examplenew[eval_func_param_names.index('iinput_nochat')] = ''  # no input
                examplenew[eval_func_param_names.index('context')] = get_context(chat_context, prompt_type)
                examples.append(examplenew)
                responses.append(output)

        num_examples = len(examples)
        scoring_path = 'scoring'
        os.makedirs(scoring_path, exist_ok=True)
        if eval_sharegpt_as_output:
            used_base_model = 'gpt35'
            used_lora_weights = ''
        else:
            used_base_model = str(base_model.split('/')[-1])
            used_lora_weights = str(lora_weights.split('/')[-1])
        eval_filename = "df_scores_%s_%s_%s_%s_%s_%s.parquet" % (num_examples, eval_sharegpt_prompts_only,
                                                                 eval_sharegpt_prompts_only_seed,
                                                                 eval_sharegpt_as_output,
                                                                 used_base_model,
                                                                 used_lora_weights)
        eval_filename = os.path.join(scoring_path, eval_filename)

        # torch.device("cuda") leads to cuda:x cuda:y mismatches for multi-GPU consistently
        device = 'cpu' if n_gpus == 0 else 'cuda'
        context_class = NullContext if n_gpus > 1 or n_gpus == 0 else torch.device

        with context_class(device):
            # ensure was set right above before examples generated
            assert not stream_output, "stream_output=True does not make sense with example loop"
            import time
            from functools import partial

            # get score model
            smodel, stokenizer, sdevice = get_score_model(**locals())

            if not eval_sharegpt_as_output:
                model, tokenizer, device = get_model(**locals())
                model_state = [model, tokenizer, device, base_model]
                kwargs_evaluate = {k: v for k, v in locals().items() if k in inputs_kwargs_list}
                my_db_state = [None]
                fun = partial(evaluate, model_state, my_db_state, debug=debug, save_dir=save_dir, is_low_mem=is_low_mem,
                              raise_generate_gpu_exceptions=raise_generate_gpu_exceptions,
                              chat_context=chat_context,
                              concurrency_count=concurrency_count,
                              lora_weights=lora_weights)
            else:
                assert eval_sharegpt_prompts_only > 0

                def get_response(*args, exi=0):
                    # assumes same ordering of examples and responses
                    yield responses[exi]

                fun = get_response
            t0 = time.time()
            score_dump = []

            for exi, ex in enumerate(examples):
                instruction = ex[eval_func_param_names.index('instruction_nochat')]
                iinput = ex[eval_func_param_names.index('iinput_nochat')]
                context = ex[eval_func_param_names.index('context')]
                clear_torch_cache()
                print("")
                print("START" + "=" * 100)
                print("Question: %s %s" % (instruction, ('input=%s' % iinput if iinput else '')))
                print("-" * 105)
                # fun yields as generator, so have to iterate over it
                # Also means likely do NOT want --stream_output=True, else would show all generations
                gener = fun(*tuple(ex), exi=exi) if eval_sharegpt_as_output else fun(*tuple(ex))
                for res in gener:
                    print(res)
                    if smodel:
                        score_with_prompt = False
                        if score_with_prompt:
                            data_point = dict(instruction=instruction, input=iinput, context=context)
                            prompter = Prompter(prompt_type, debug=debug, chat=chat, stream_output=stream_output)
                            prompt = prompter.generate_prompt(data_point)
                        else:
                            # just raw input and output
                            if eval_sharegpt_prompts_only > 0:
                                # only our own examples have this filled at moment
                                assert iinput in [None, ''], iinput  # should be no iinput
                            if not (chat_context and prompt_type == 'human_bot'):
                                assert context in [None, ''], context  # should be no context
                            prompt = instruction
                        cutoff_len = 768 if is_low_mem else 2048
                        inputs = stokenizer(prompt, res,
                                            return_tensors="pt",
                                            truncation=True,
                                            max_length=cutoff_len)
                        try:
                            score = torch.sigmoid(smodel(**inputs).logits[0].float()).cpu().detach().numpy()[0]
                        except torch.cuda.OutOfMemoryError as e:
                            print("GPU OOM 1: question: %s answer: %s exception: %s" % (prompt, res, str(e)),
                                  flush=True)
                            traceback.print_exc()
                            score = 0.0
                            clear_torch_cache()
                        except (Exception, RuntimeError) as e:
                            if 'Expected all tensors to be on the same device' in str(e) or \
                                    'expected scalar type Half but found Float' in str(e) or \
                                    'probability tensor contains either' in str(e) or \
                                    'cublasLt ran into an error!' in str(e):
                                print("GPU error: question: %s answer: %s exception: %s" % (prompt, res, str(e)),
                                      flush=True)
                                traceback.print_exc()
                                score = 0.0
                                clear_torch_cache()
                            else:
                                raise
                        print("SCORE %s: %s" % (exi, score), flush=True)
                        score_dump.append(ex + [prompt, res, score])
                        # dump every score in case abort
                        df_scores = pd.DataFrame(score_dump,
                                                 columns=eval_func_param_names + eval_extra_columns)
                        df_scores.to_parquet(eval_filename, index=False)
                        # plot histogram so far
                        plt.figure(figsize=(10, 10))
                        plt.hist(df_scores['score'], bins=20)
                        score_avg = np.mean(df_scores['score'])
                        score_median = np.median(df_scores['score'])
                        plt.title("Score avg: %s median: %s" % (score_avg, score_median))
                        plt.savefig(eval_filename.replace('.parquet', '.png'))
                        plt.close()

                print("END" + "=" * 102)
                print("")
                t2 = time.time()
                print("Time taken so far: %.4f about %.4g per example" % (t2 - t0, (t2 - t0) / (1 + exi)))
            t1 = time.time()
            print("Total time taken: %.4f about %.4g per example" % (t1 - t0, (t1 - t0) / num_examples))
        return eval_filename

    if gradio:
        # imported here so don't require gradio to run generate
        from gradio_runner import go_gradio

        # get default model
        all_kwargs = locals().copy()
        if all_kwargs.get('base_model') and not all_kwargs['login_mode_if_model0']:
            model0, tokenizer0, device = get_model(**all_kwargs)
        else:
            # if empty model, then don't load anything, just get gradio up
            model0, tokenizer0, device = None, None, None
        model_state0 = [model0, tokenizer0, device, all_kwargs['base_model']]

        # get score model
        smodel, stokenizer, sdevice = get_score_model(**all_kwargs)
        score_model_state0 = [smodel, stokenizer, sdevice, score_model]

        go_gradio(**locals())


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device


def get_non_lora_model(base_model, model_loader, load_half, model_kwargs, reward_type,
                       gpu_id=0,
                       use_auth_token=False,
                       trust_remote_code=True,
                       triton_attn=False,
                       long_sequence=True,
                       ):
    """
    Ensure model gets on correct device
    :param base_model:
    :param model_loader:
    :param load_half:
    :param model_kwargs:
    :param reward_type:
    :param gpu_id:
    :param use_auth_token:
    :param trust_remote_code:
    :param triton_attn:
    :param long_sequence:
    :return:
    """
    with init_empty_weights():
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(base_model, use_auth_token=use_auth_token,
                                            trust_remote_code=trust_remote_code)
        if triton_attn and 'mpt-' in base_model.lower():
            config.attn_config['attn_impl'] = 'triton'
        if long_sequence:
            if 'mpt-7b-storywriter' in base_model.lower():
                config.update({"max_seq_len": 83968})
            if 'mosaicml/mpt-7b-chat' in base_model.lower():
                config.update({"max_seq_len": 4096})
        if issubclass(config.__class__, tuple(AutoModel._model_mapping.keys())):
            model = AutoModel.from_config(
                config,
            )
        else:
            # can't infer
            model = None

    if model is not None:
        # NOTE: Can specify max_memory={0: max_mem, 1: max_mem}, to shard model
        # NOTE: Some models require avoiding sharding some layers,
        # then would pass no_split_module_classes and give list of those layers.
        device_map = infer_auto_device_map(
            model,
            dtype=torch.float16 if load_half else torch.float32,
        )
        if hasattr(model, 'model'):
            device_map_model = infer_auto_device_map(
                model.model,
                dtype=torch.float16 if load_half else torch.float32,
            )
            device_map.update(device_map_model)
    else:
        device_map = "auto"

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available else 0

    if n_gpus > 0:
        if gpu_id >= 0:
            # FIXME: If really distributes model, tend to get things like: ValueError: gpt_neox.embed_in.weight doesn't have any device set.
            # So avoid for now, just put on first GPU, unless score_model, put on last
            if reward_type:
                device_map = {'': n_gpus - 1}
            else:
                device_map = {'': min(n_gpus - 1, gpu_id)}
        if gpu_id == -1:
            device_map = {'': 'cuda'}
    else:
        device_map = {'': 'cpu'}
        model_kwargs['load_in_8bit'] = False
    print('device_map: %s' % device_map, flush=True)

    load_in_8bit = model_kwargs.get('load_in_8bit', False)
    model_kwargs['device_map'] = device_map

    if load_in_8bit or not load_half:
        model = model_loader.from_pretrained(
            base_model,
            config=config,
            **model_kwargs,
        )
    else:
        model = model_loader.from_pretrained(
            base_model,
            config=config,
            **model_kwargs,
        ).half()
    return model


def get_model(
        load_8bit: bool = False,
        load_half: bool = True,
        infer_devices: bool = True,
        base_model: str = '',
        quant_model: str = '',
        tokenizer_base_model: str = '',
        lora_weights: str = "",
        gpu_id: int = 0,

        reward_type: bool = None,
        local_files_only: bool = False,
        resume_download: bool = True,
        use_auth_token: Union[str, bool] = False,
        trust_remote_code: bool = True,
        compile: bool = True,
        **kwargs,
):
    """

    :param load_8bit: load model in 8-bit, not supported by all models
    :param load_half: load model in 16-bit
    :param infer_devices: Use torch infer of optimal placement of layers on devices (for non-lora case)
           For non-LORA case, False will spread shards across multiple GPUs, but this can lead to cuda:x cuda:y mismatches
           So it is not the default
    :param base_model: name/path of base model
    :param tokenizer_base_model: name/path of tokenizer
    :param lora_weights: name/path
    :param gpu_id: which GPU (0..n_gpus-1) or allow all GPUs if relevant (-1)
    :param reward_type: reward type model for sequence classification
    :param local_files_only: use local files instead of from HF
    :param resume_download: resume downloads from HF
    :param use_auth_token: assumes user did on CLI `huggingface-cli login` to access private repo
    :param trust_remote_code: trust code needed by model
    :param compile: whether to compile torch model
    :param kwargs:
    :return:
    """
    print("Get %s model" % base_model, flush=True)
    if lora_weights is not None and lora_weights.strip():
        print("Get %s lora weights" % lora_weights, flush=True)
    device = get_device()

    if 'gpt2' in base_model.lower():
        # RuntimeError: where expected condition to be a boolean tensor, but got a tensor with dtype Half
        load_8bit = False

    assert base_model.strip(), (
        "Please choose a base model with --base_model (CLI) or in Models Tab (gradio)"
    )

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(base_model, use_auth_token=use_auth_token,
                                        trust_remote_code=trust_remote_code)
    llama_type_from_config = 'llama' in str(config).lower()
    llama_type_from_name = "llama" in base_model.lower()
    llama_type = llama_type_from_config or llama_type_from_name
    if llama_type:
        print("Detected as llama type from"
              " config (%s) or name (%s)" % (llama_type_from_config, llama_type_from_name), flush=True)

    model_loader, tokenizer_loader = get_loaders(llama_type=llama_type, model_name=base_model, reward_type=reward_type)
    if not tokenizer_base_model:
        tokenizer_base_model = base_model

    if tokenizer_loader is not None and not isinstance(tokenizer_loader, str):
        tokenizer = tokenizer_loader.from_pretrained(tokenizer_base_model,
                                                     local_files_only=local_files_only,
                                                     resume_download=resume_download,
                                                     use_auth_token=use_auth_token,
                                                     trust_remote_code=trust_remote_code,
                                                     )
    else:
        tokenizer = tokenizer_loader

    if isinstance(tokenizer, str):
        # already a pipeline, tokenizer_loader is string for task
        model = model_loader(tokenizer,
                             model=base_model,
                             device=0 if device == "cuda" else -1,
                             torch_dtype=torch.float16 if device == 'cuda' else torch.float32)
    else:
        assert device in ["cuda", "cpu"], "Unsupported device %s" % device
        model_kwargs = dict(local_files_only=local_files_only,
                            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                            resume_download=resume_download,
                            use_auth_token=use_auth_token,
                            trust_remote_code=trust_remote_code,
                            )
        if 'mbart-' not in base_model.lower() and 'mpt-' not in base_model.lower():
            model_kwargs.update(dict(load_in_8bit=load_8bit,
                                     device_map={"": 0} if load_8bit and device == 'cuda' else "auto",
                                     ))
        if 'mpt-' in base_model.lower() and gpu_id >= 0:
            model_kwargs.update(dict(device_map={"": gpu_id} if device == 'cuda' else "cpu"))

        if 'OpenAssistant/reward-model'.lower() in base_model.lower():
            # could put on other GPUs
            model_kwargs['device_map'] = {"": 0} if device == 'cuda' else {"": 'cpu'}
            model_kwargs.pop('torch_dtype', None)

        if not lora_weights:
            if quant_model:
                model = AutoGPTQForCausalLM.from_quantized(quant_model, use_triton=False, device=device)
            else:
                with torch.device(device):
                    if infer_devices:
                        model = get_non_lora_model(base_model, model_loader, load_half, model_kwargs, reward_type,
                                                   gpu_id=gpu_id,
                                                   use_auth_token=use_auth_token,
                                                   trust_remote_code=trust_remote_code,
                                                   )
                    else:
                        if load_half and not load_8bit:
                            model = model_loader.from_pretrained(
                                base_model,
                                **model_kwargs).half()
                        else:
                            model = model_loader.from_pretrained(
                                base_model,
                                **model_kwargs)
        elif load_8bit:
            model = model_loader.from_pretrained(
                base_model,
                **model_kwargs
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                local_files_only=local_files_only,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
                trust_remote_code=trust_remote_code,
                device_map={"": 0} if device == 'cuda' else {"": 'cpu'},  # seems to be required
            )
        else:
            with torch.device(device):
                model = model_loader.from_pretrained(
                    base_model,
                    **model_kwargs
                )
                model = PeftModel.from_pretrained(
                    model,
                    lora_weights,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                    local_files_only=local_files_only,
                    resume_download=resume_download,
                    use_auth_token=use_auth_token,
                    trust_remote_code=trust_remote_code,
                    device_map="auto",
                )
                if load_half:
                    model.half()

    # unwind broken decapoda-research config
    if llama_type:
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    if 'gpt2' in base_model.lower():
        # add special tokens that otherwise all share the same id
        tokenizer.add_special_tokens({'bos_token': '<bos>',
                                      'eos_token': '<eos>',
                                      'pad_token': '<pad>'})

    if not isinstance(tokenizer, str):
        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32" and compile:
            model = torch.compile(model)

    return model, tokenizer, device


def get_score_model(**kwargs):
    # score model
    if kwargs.get('score_model') is not None and kwargs.get('score_model').strip():
        score_all_kwargs = kwargs.copy()
        score_all_kwargs['load_8bit'] = False
        score_all_kwargs['load_half'] = False
        score_all_kwargs['base_model'] = kwargs.get('score_model').strip()
        score_all_kwargs['tokenizer_base_model'] = ''
        score_all_kwargs['lora_weights'] = ''
        score_all_kwargs['llama_type'] = False
        score_all_kwargs['compile'] = False
        score_all_kwargs['quant_model'] = ''
        smodel, stokenizer, sdevice = get_model(**score_all_kwargs)
    else:
        smodel, stokenizer, sdevice = None, None, None
    return smodel, stokenizer, sdevice


eval_func_param_names = ['instruction',
                         'iinput',
                         'context',
                         'stream_output',
                         'prompt_type',
                         'temperature',
                         'top_p',
                         'top_k',
                         'num_beams',
                         'max_new_tokens',
                         'min_new_tokens',
                         'early_stopping',
                         'max_time',
                         'repetition_penalty',
                         'num_return_sequences',
                         'do_sample',
                         'chat',
                         'instruction_nochat',
                         'iinput_nochat',
                         'langchain_mode',
                         ]

inputs_kwargs_list = ['debug', 'save_dir', 'sanitize_bot_response', 'model_state0', 'is_low_mem',
                      'raise_generate_gpu_exceptions', 'chat_context', 'concurrency_count', 'lora_weights',
                      'load_db_if_exists', 'dbs', 'user_path']


def evaluate(
        model_state,
        my_db_state,
        # START NOTE: Examples must have same order of parameters
        instruction,
        iinput,
        context,
        stream_output,
        prompt_type,
        temperature,
        top_p,
        top_k,
        num_beams,
        max_new_tokens,
        min_new_tokens,
        early_stopping,
        max_time,
        repetition_penalty,
        num_return_sequences,
        do_sample,
        chat,
        instruction_nochat,
        iinput_nochat,
        langchain_mode,
        # END NOTE: Examples must have same order of parameters
        src_lang=None,
        tgt_lang=None,
        debug=False,
        concurrency_count=None,
        save_dir=None,
        sanitize_bot_response=True,
        model_state0=None,
        is_low_mem=None,
        raise_generate_gpu_exceptions=None,
        chat_context=None,
        lora_weights=None,
        load_db_if_exists=True,
        dbs=None,
        user_path=None,
):
    # ensure passed these
    assert concurrency_count is not None
    assert is_low_mem is not None
    assert raise_generate_gpu_exceptions is not None
    assert chat_context is not None

    if debug:
        locals_dict = locals().copy()
        locals_dict.pop('model_state', None)
        locals_dict.pop('model_state0', None)
        print(locals_dict)

    no_model_msg = "Please choose a base model with --base_model (CLI) or in Models Tab (gradio).\nThen start New Conversation"

    if model_state0 is None:
        # e.g. for no gradio case, set dummy value, else should be set
        model_state0 = [None, None, None, None]

    if model_state is not None and len(model_state) == 4 and not isinstance(model_state[0], str):
        # try to free-up original model (i.e. list was passed as reference)
        if model_state0 is not None and model_state0[0] is not None:
            model_state0[0].cpu()
            model_state0[0] = None
        # try to free-up original tokenizer (i.e. list was passed as reference)
        if model_state0 is not None and model_state0[1] is not None:
            model_state0[1] = None
        clear_torch_cache()
        model, tokenizer, device, base_model = model_state
    elif model_state0 is not None and len(model_state0) == 4 and model_state0[0] is not None:
        assert isinstance(model_state[0], str)
        model, tokenizer, device, base_model = model_state0
    else:
        raise AssertionError(no_model_msg)

    if base_model is None:
        raise AssertionError(no_model_msg)

    assert base_model.strip(), no_model_msg
    assert model, "Model is missing"
    assert tokenizer, "Tokenizer is missing"

    # choose chat or non-chat mode
    if not chat:
        instruction = instruction_nochat
        iinput = iinput_nochat

    if not context:
        # get hidden context if have one
        context = get_context(chat_context, prompt_type)

    prompter = Prompter(prompt_type, debug=debug, chat=chat, stream_output=stream_output)
    data_point = dict(context=context, instruction=instruction, input=iinput)
    prompt = prompter.generate_prompt(data_point)

    # THIRD PLACE where LangChain referenced, but imports only occur if enabled and have db to use
    assert langchain_mode in langchain_modes, "Invalid langchain_mode %s" % langchain_mode
    if langchain_mode in ['MyData'] and my_db_state is not None and len(my_db_state) > 0 and my_db_state[0] is not None:
        db1 = my_db_state[0]
    elif dbs is not None and langchain_mode in dbs:
        db1 = dbs[langchain_mode]
    else:
        db1 = None
    if langchain_mode not in [False, 'Disabled', 'ChatLLM', 'LLM'] and db1 is not None:
        query = instruction if not iinput else "%s\n%s" % (instruction, iinput)
        from gpt_langchain import run_qa_db, get_db_kwargs
        langchain_kwargs = get_db_kwargs(langchain_mode)
        outr = ""
        # use smaller cut_distanct for wiki_full since so many matches could be obtained, and often irrelevant unless close
        for r in run_qa_db(query=query,
                           model_name=base_model, model=model, tokenizer=tokenizer,
                           stream_output=stream_output, prompter=prompter,
                           do_yield=True,
                           load_db_if_exists=load_db_if_exists,
                           db=db1,
                           user_path=user_path,
                           max_new_tokens=max_new_tokens,
                           cut_distanct=1.1 if langchain_mode in ['wiki_full'] else 1.8,
                           **langchain_kwargs):
            outr = r  # doesn't accumualte, new answer every yield, so only save that full answer
            yield r
        if save_dir:
            save_generate_output(output=outr, base_model=base_model, save_dir=save_dir)
            print('Post-Generate Langchain: %s decoded_output: %s' % (str(datetime.now()), len(outr) if outr else -1),
                  flush=True)
        if outr:
            return

    if isinstance(tokenizer, str):
        # pipeline
        if tokenizer == "summarization":
            key = 'summary_text'
        else:
            raise RuntimeError("No such task type %s" % tokenizer)
        # NOTE: uses max_length only
        yield model(prompt, max_length=max_new_tokens)[0][key]

    if 'mbart-' in base_model.lower():
        assert src_lang is not None
        tokenizer.src_lang = languages_covered()[src_lang]

    if chat:
        # override, ignore user change
        num_return_sequences = 1
    stopping_criteria = get_stopping(prompt_type, tokenizer, device)
    _, _, max_length_tokenize, max_prompt_length = get_cutoffs(is_low_mem)
    prompt = prompt[-max_prompt_length:]
    inputs = tokenizer(prompt,
                       return_tensors="pt",
                       truncation=True,
                       max_length=max_length_tokenize)
    if inputs['input_ids'].shape[1] >= max_length_tokenize - 1:
        print("Cutting off input: %s %s" % (inputs['input_ids'].shape[1], max_length_tokenize), flush=True)
    if debug and len(inputs["input_ids"]) > 0:
        print('input_ids length', len(inputs["input_ids"][0]), flush=True)
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=top_k,
        num_beams=num_beams,
        do_sample=do_sample,
        repetition_penalty=float(repetition_penalty),
        num_return_sequences=num_return_sequences,
        renormalize_logits=True,
        remove_invalid_values=True,
    )

    gen_kwargs = dict(input_ids=input_ids,
                      generation_config=generation_config,
                      return_dict_in_generate=True,
                      output_scores=True,
                      max_new_tokens=max_new_tokens,  # prompt + new
                      min_new_tokens=min_new_tokens,  # prompt + new
                      early_stopping=early_stopping,  # False, True, "never"
                      max_time=max_time,
                      stopping_criteria=stopping_criteria,
                      )
    if 'gpt2' in base_model.lower():
        gen_kwargs.update(dict(bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.eos_token_id))
    elif 'mbart-' in base_model.lower():
        assert tgt_lang is not None
        tgt_lang = languages_covered()[tgt_lang]
        gen_kwargs.update(dict(forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]))
    else:
        gen_kwargs.update(dict(pad_token_id=tokenizer.eos_token_id))

    decoder_kwargs = dict(skip_special_tokens=True,
                          clean_up_tokenization_spaces=True)

    decoder = functools.partial(tokenizer.decode,
                                **decoder_kwargs
                                )
    decoder_raw_kwargs = dict(skip_special_tokens=False,
                              clean_up_tokenization_spaces=True)

    decoder_raw = functools.partial(tokenizer.decode,
                                    **decoder_raw_kwargs
                                    )

    with torch.no_grad():
        context_class_cast = NullContext if device == 'cpu' or lora_weights else torch.autocast
        with context_class_cast(device):
            # protection for gradio not keeping track of closed users,
            # else hit bitsandbytes lack of thread safety:
            # https://github.com/h2oai/h2ogpt/issues/104
            # but only makes sense if concurrency_count == 1
            context_class = NullContext  # if concurrency_count > 1 else filelock.FileLock
            print('Pre-Generate: %s' % str(datetime.now()), flush=True)
            decoded_output = None
            with context_class("generate.lock"):
                print('Generate: %s' % str(datetime.now()), flush=True)
                # decoded tokenized prompt can deviate from prompt due to special characters
                inputs_decoded = decoder(input_ids[0])
                inputs_decoded_raw = decoder_raw(input_ids[0])
                if inputs_decoded == prompt:
                    # normal
                    pass
                elif inputs_decoded.lstrip() == prompt.lstrip():
                    # sometimes extra space in front, make prompt same for prompt removal
                    prompt = inputs_decoded
                elif inputs_decoded_raw == prompt:
                    # some models specify special tokens that are part of normal prompt, so can't skip them
                    inputs_decoded = prompt = inputs_decoded_raw
                    decoder = decoder_raw
                    decoder_kwargs = decoder_raw_kwargs
                elif inputs_decoded_raw.replace("<unk> ", "").replace("<unk>", "").replace('\n', ' ').replace(' ',
                                                                                                              '') == prompt.replace(
                    '\n', ' ').replace(' ', ''):
                    inputs_decoded = prompt = inputs_decoded_raw
                    decoder = decoder_raw
                    decoder_kwargs = decoder_raw_kwargs
                else:
                    print("WARNING: Special characters in prompt", flush=True)
                if stream_output:
                    skip_prompt = False
                    streamer = H2OTextIteratorStreamer(tokenizer, skip_prompt=skip_prompt, block=False,
                                                       **decoder_kwargs)
                    gen_kwargs.update(dict(streamer=streamer))
                    target = wrapped_partial(generate_with_exceptions, model.generate,
                                             prompt=prompt, inputs_decoded=inputs_decoded,
                                             raise_generate_gpu_exceptions=raise_generate_gpu_exceptions,
                                             **gen_kwargs)
                    bucket = queue.Queue()
                    thread = EThread(target=target, streamer=streamer, bucket=bucket)
                    thread.start()
                    outputs = ""
                    try:
                        for new_text in streamer:
                            if bucket.qsize() > 0 or thread.exc:
                                thread.join()
                            outputs += new_text
                            yield prompter.get_response(outputs, prompt=inputs_decoded,
                                                        sanitize_bot_response=sanitize_bot_response)
                    except BaseException:
                        # if any exception, raise that exception if was from thread, first
                        if thread.exc:
                            raise thread.exc
                        raise
                    finally:
                        # in case no exception and didn't join with thread yet, then join
                        if not thread.exc:
                            thread.join()
                    # in case raise StopIteration or broke queue loop in streamer, but still have exception
                    if thread.exc:
                        raise thread.exc
                    decoded_output = outputs
                else:
                    outputs = model.generate(**gen_kwargs)
                    outputs = [decoder(s) for s in outputs.sequences]
                    yield prompter.get_response(outputs, prompt=inputs_decoded,
                                                sanitize_bot_response=sanitize_bot_response)
                    if outputs and len(outputs) >= 1:
                        decoded_output = prompt + outputs[0]
                if save_dir and decoded_output:
                    save_generate_output(output=decoded_output, base_model=base_model, save_dir=save_dir)
            print('Post-Generate: %s decoded_output: %s' % (
                str(datetime.now()), len(decoded_output) if decoded_output else -1), flush=True)


def get_cutoffs(is_low_mem, for_context=False):
    # help to avoid errors like:
    # RuntimeError: The size of tensor a (2048) must match the size of tensor b (2049) at non-singleton dimension 3
    # RuntimeError: expected scalar type Half but found Float
    # with - 256
    max_length_tokenize = 768 - 256 if is_low_mem else 2048 - 256
    cutoff_len = max_length_tokenize * 4  # if reaches limit, then can't generate new tokens
    output_smallest = 30 * 4
    max_prompt_length = cutoff_len - output_smallest

    if for_context:
        # then lower even more to avoid later chop, since just estimate tokens in context bot
        max_prompt_length = max(64, int(max_prompt_length * 0.8))

    return cutoff_len, output_smallest, max_length_tokenize, max_prompt_length


class H2OTextIteratorStreamer(TextIteratorStreamer):
    """
    normally, timeout required for now to handle exceptions, else get()
    but with H2O version of TextIteratorStreamer, loop over block to handle
    """

    def __init__(self, tokenizer, skip_prompt: bool = False, timeout: typing.Optional[float] = None,
                 block=True, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = queue.Queue()
        self.stop_signal = None
        self.do_stop = False
        self.timeout = timeout
        self.block = block

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                value = self.stop_signal  # value looks unused in pycharm, not true
                if self.do_stop:
                    print("hit stop", flush=True)
                    # could raise or break, maybe best to raise and make parent see if any exception in thread
                    raise StopIteration()
                    # break
                value = self.text_queue.get(block=self.block, timeout=self.timeout)
                break
            except queue.Empty:
                time.sleep(0.01)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


def generate_with_exceptions(func, *args, prompt='', inputs_decoded='', raise_generate_gpu_exceptions=True, **kwargs):
    try:
        func(*args, **kwargs)
    except torch.cuda.OutOfMemoryError as e:
        print("GPU OOM 2: prompt: %s inputs_decoded: %s exception: %s" % (prompt, inputs_decoded, str(e)),
              flush=True)
        if 'input_ids' in kwargs:
            if kwargs['input_ids'] is not None:
                kwargs['input_ids'].cpu()
            kwargs['input_ids'] = None
        traceback.print_exc()
        clear_torch_cache()
        return
    except (Exception, RuntimeError) as e:
        if 'Expected all tensors to be on the same device' in str(e) or \
                'expected scalar type Half but found Float' in str(e) or \
                'probability tensor contains either' in str(e) or \
                'cublasLt ran into an error!' in str(e) or \
                'mat1 and mat2 shapes cannot be multiplied' in str(e):
            print(
                "GPU Error: prompt: %s inputs_decoded: %s exception: %s" % (prompt, inputs_decoded, str(e)),
                flush=True)
            traceback.print_exc()
            clear_torch_cache()
            if raise_generate_gpu_exceptions:
                raise
            return
        else:
            clear_torch_cache()
            if raise_generate_gpu_exceptions:
                raise


def get_generate_params(model_lower, chat,
                        stream_output, show_examples,
                        prompt_type, temperature, top_p, top_k, num_beams,
                        max_new_tokens, min_new_tokens, early_stopping, max_time,
                        repetition_penalty, num_return_sequences,
                        do_sample):
    use_defaults = False
    use_default_examples = True
    examples = []
    task_info = f"{prompt_type}"
    if model_lower:
        print(f"Using Model {model_lower}", flush=True)
    else:
        print("No model defined yet", flush=True)

    min_new_tokens = min_new_tokens if min_new_tokens is not None else 0
    early_stopping = early_stopping if early_stopping is not None else False
    max_time_defaults = 60 * 3
    max_time = max_time if max_time is not None else max_time_defaults

    if not prompt_type and model_lower in inv_prompt_type_to_model_lower:
        prompt_type = inv_prompt_type_to_model_lower[model_lower]
        print("Auto-selecting prompt_type=%s for %s" % (prompt_type, model_lower), flush=True)

    # examples at first don't include chat, instruction_nochat, iinput_nochat, added at end
    if show_examples is None:
        if chat:
            show_examples = False
        else:
            show_examples = True

    summarize_example1 = """Jeff: Can I train a ? Transformers model on Amazon SageMaker? 
Philipp: Sure you can use the new Hugging Face Deep Learning Container. 
Jeff: ok.
Jeff: and how can I get started? 
Jeff: where can I find documentation? 
Philipp: ok, ok you can find everything here. https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face"""

    if 'bart-large-cnn-samsum' in model_lower or 'flan-t5-base-samsum' in model_lower:
        placeholder_instruction = summarize_example1
        placeholder_input = ""
        use_defaults = True
        use_default_examples = False
        examples += [
            [placeholder_instruction, "", "", stream_output, 'plain', 1.0, 1.0, 50, 1, 128, 0, False, max_time_defaults,
             1.0, 1,
             False]]
        task_info = "Summarization"
    elif 't5-' in model_lower or 't5' == model_lower or 'flan-' in model_lower:
        placeholder_instruction = "The square root of x is the cube root of y. What is y to the power of 2, if x = 4?"
        placeholder_input = ""
        use_defaults = True
        use_default_examples = True
        task_info = "Multi-Task: Q/A, translation, Chain-of-Thought, Logical Reasoning, Summarization, etc.  Best to use task prefix as trained on, e.g. `translate English to German: ` (space after colon)"
    elif 'mbart-' in model_lower:
        placeholder_instruction = "The girl has long hair."
        placeholder_input = ""
        use_defaults = True
        use_default_examples = False
        examples += [
            [placeholder_instruction, "", "", stream_output, 'plain', 1.0, 1.0, 50, 1, 128, 0, False, max_time_defaults,
             1.0, 1,
             False]]
    elif 'gpt2' in model_lower:
        placeholder_instruction = "The sky is"
        placeholder_input = ""
        prompt_type = prompt_type or 'plain'
        use_default_examples = True  # some will be odd "continuations" but can be ok
        examples += [
            [placeholder_instruction, "", "", stream_output, 'plain', 1.0, 1.0, 50, 1, 128, 0, False, max_time_defaults,
             1.0, 1,
             False]]
        task_info = "Auto-complete phrase, code, etc."
        use_defaults = True
    else:
        if chat:
            placeholder_instruction = "Enter a question or imperative."
        else:
            placeholder_instruction = "Give detailed answer for whether Einstein or Newton is smarter."
        placeholder_input = ""
        if model_lower:
            # default is plain, because might relly upon trust_remote_code to handle prompting
            prompt_type = prompt_type or 'plain'
        else:
            prompt_type = ''
        examples += [[summarize_example1, 'Summarize' if prompt_type not in ['plain', 'instruct_simple'] else '', "",
                      stream_output, prompt_type or 'plain', 0.1, 0.75, 40, 4, 256, 0, False, max_time_defaults, 1.0, 1,
                      False]]
        task_info = "No task"
        if prompt_type == 'instruct':
            task_info = "Answer question or follow imperative as instruction with optionally input."
        elif prompt_type == 'plain':
            task_info = "Auto-complete phrase, code, etc."
        elif prompt_type == 'human_bot':
            if chat:
                task_info = "Chat (Shift-Enter to give question/imperative, input concatenated with instruction)"
            else:
                task_info = "Ask question/imperative (input concatenated with instruction)"

    # revert to plain if still nothing
    prompt_type = prompt_type or 'plain'
    if use_defaults:
        temperature = 1.0 if temperature is None else temperature
        top_p = 1.0 if top_p is None else top_p
        top_k = 40 if top_k is None else top_k
        num_beams = num_beams or 1
        max_new_tokens = max_new_tokens or 128
        repetition_penalty = repetition_penalty or 1.07
        num_return_sequences = min(num_beams, num_return_sequences or 1)
        do_sample = False if do_sample is None else do_sample
    else:
        temperature = 0.1 if temperature is None else temperature
        top_p = 0.75 if top_p is None else top_p
        top_k = 40 if top_k is None else top_k
        if chat:
            num_beams = num_beams or 1
        else:
            num_beams = num_beams or 4
        max_new_tokens = max_new_tokens or 256
        repetition_penalty = repetition_penalty or 1.07
        num_return_sequences = min(num_beams, num_return_sequences or 1)
        do_sample = False if do_sample is None else do_sample
    # doesn't include chat, instruction_nochat, iinput_nochat, added later
    params_list = ["", stream_output, prompt_type, temperature, top_p, top_k, num_beams, max_new_tokens, min_new_tokens,
                   early_stopping, max_time, repetition_penalty, num_return_sequences, do_sample]

    if use_default_examples:
        examples += [
            ["Translate English to French", "Good morning"] + params_list,
            ["Give detailed answer for whether Einstein or Newton is smarter.", ''] + params_list,
            ["Explain in detailed list, all the best practices for coding in python.", ''] + params_list,
            [
                "Create a markdown table with 3 rows for the primary colors, and 2 columns, with color name and hex codes.",
                ''] + params_list,
            ['Translate to German:  My name is Arthur', ''] + params_list,
            ["Please answer to the following question. Who is going to be the next Ballon d'or?", ''] + params_list,
            ['Can Geoffrey Hinton have a conversation with George Washington? Give the rationale before answering.',
             ''] + params_list,
            ['Please answer the following question. What is the boiling point of Nitrogen?', ''] + params_list,
            ['Answer the following yes/no question. Can you write a whole Haiku in a single tweet?', ''] + params_list,
            ["Simplify the following expression: (False or False and True). Explain your answer.", ''] + params_list,
            [
                "Premise: At my age you will probably have learnt one lesson. Hypothesis:  It's not certain how many lessons you'll learn by your thirties. Does the premise entail the hypothesis?",
                ''] + params_list,
            ['The square root of x is the cube root of y. What is y to the power of 2, if x = 4?', ''] + params_list,
            [
                'Answer the following question by reasoning step by step.  The cafeteria had 23 apples. If they used 20 for lunch, and bought 6 more, how many apple do they have?',
                ''] + params_list,
            ["""def area_of_rectangle(a: float, b: float):
    \"\"\"Return the area of the rectangle.\"\"\"""", ''] + params_list,
            ["""# a function in native python:
def mean(a):
    return sum(a)/len(a)

# the same function using numpy:
import numpy as np
def mean(a):""", ''] + params_list,
            ["""X = np.random.randn(100, 100)
y = np.random.randint(0, 1, 100)

# fit random forest classifier with 20 estimators""", ''] + params_list,
        ]

    src_lang = "English"
    tgt_lang = "Russian"

    # move to correct position
    for example in examples:
        example += [chat, '', '', 'Disabled']
        # adjust examples if non-chat mode
        if not chat:
            example[eval_func_param_names.index('instruction_nochat')] = example[
                eval_func_param_names.index('instruction')]
            example[eval_func_param_names.index('instruction')] = ''

            example[eval_func_param_names.index('iinput_nochat')] = example[eval_func_param_names.index('iinput')]
            example[eval_func_param_names.index('iinput')] = ''
        assert len(example) == len(eval_func_param_names), "Wrong example: %s %s" % (len(example), len(eval_func_param_names))

    return placeholder_instruction, placeholder_input, \
        stream_output, show_examples, \
        prompt_type, temperature, top_p, top_k, num_beams, \
        max_new_tokens, min_new_tokens, early_stopping, max_time, \
        repetition_penalty, num_return_sequences, \
        do_sample, \
        src_lang, tgt_lang, \
        examples, \
        task_info


def languages_covered():
    # https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt#languages-covered
    covered = """Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE), Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)"""
    covered = covered.split(', ')
    covered = {x.split(' ')[0]: x.split(' ')[1].replace(')', '').replace('(', '') for x in covered}
    return covered


def get_context(chat_context, prompt_type):
    if chat_context and prompt_type == 'human_bot':
        context0 = """<bot>: I am an intelligent, helpful, truthful, and fair assistant named h2oGPT, who will give accurate, balanced, and reliable responses.  I will not respond with I don't know or I don't understand.
<human>: I am a human person seeking useful assistance and request all questions be answered completely, and typically expect detailed responses.  Give answers in numbered list format if several distinct but related items are being listed."""
    else:
        context0 = ''
    return context0


def test_test_prompt(prompt_type='instruct', data_point=0):
    example_data_point = example_data_points[data_point]
    example_data_point.pop('output', None)
    return generate_prompt(example_data_point, prompt_type, False, False)


def score_qa(smodel, stokenizer, max_length_tokenize, question, answer, cutoff_len):
    question = question[-cutoff_len:]
    answer = answer[-cutoff_len:]

    inputs = stokenizer(question, answer,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length_tokenize).to(smodel.device)
    try:
        score = torch.sigmoid(smodel(**inputs).logits[0]).cpu().detach().numpy()[0]
    except torch.cuda.OutOfMemoryError as e:
        print("GPU OOM 3: question: %s answer: %s exception: %s" % (question, answer, str(e)), flush=True)
        del inputs
        traceback.print_exc()
        clear_torch_cache()
        return 'Response Score: GPU OOM'
    except (Exception, RuntimeError) as e:
        if 'Expected all tensors to be on the same device' in str(e) or \
                'expected scalar type Half but found Float' in str(e) or \
                'probability tensor contains either' in str(e) or \
                'cublasLt ran into an error!' in str(e):
            print("GPU Error: question: %s answer: %s exception: %s" % (question, answer, str(e)),
                  flush=True)
            traceback.print_exc()
            clear_torch_cache()
            return 'Response Score: GPU Error'
        else:
            raise
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    return score


if __name__ == "__main__":
    """
    Examples:

    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port=1234 generate.py --base_model='EleutherAI/gpt-j-6B' --lora_weights=lora-alpaca_6B
    python generate.py --base_model='EleutherAI/gpt-j-6B' --lora_weights='lora-alpaca_6B'
    python generate.py --base_model='EleutherAI/gpt-neox-20b' --lora_weights='lora-alpaca_20B'
    
    # generate without lora weights, no prompt
    python generate.py --base_model='EleutherAI/gpt-neox-20b' --prompt_type='plain'
    python generate.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --prompt_type='dai_faq'

    python generate.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --prompt_type='dai_faq' --lora_weights='lora_20B_daifaq'
    # OpenChatKit settings:
    python generate.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --prompt_type='human_bot --debug=True --num_beams=1 --temperature=0.6 --top_k=40 --top_p=1.0

    python generate.py --base_model='distilgpt2' --prompt_type='plain' --debug=True --num_beams=1 --temperature=0.6 --top_k=40 --top_p=1.0 --share=False
    python generate.py --base_model='t5-large' --prompt_type='simple_instruct'
    python generate.py --base_model='philschmid/bart-large-cnn-samsum'
    python generate.py --base_model='philschmid/flan-t5-base-samsum'
    python generate.py --base_model='facebook/mbart-large-50-many-to-many-mmt'

    python generate.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --prompt_type='human_bot' --lora_weights='GPT-NeoXT-Chat-Base-20B.merged.json.8_epochs.57b2892c53df5b8cefac45f84d019cace803ef26.28'

    must have 4*48GB GPU and run without 8bit in order for sharding to work with infer_devices=False
    can also pass --prompt_type='human_bot' and model can somewhat handle instructions without being instruct tuned
    python generate.py --base_model=decapoda-research/llama-65b-hf --load_8bit=False --infer_devices=False --prompt_type='human_bot'

    python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6.9b
    """
    fire.Fire(main)


import pytest

@pytest.mark.parametrize(
    "base_model",
    [
        "h2oai/h2ogpt-oig-oasst1-512-6.9b",
        "h2oai/h2ogpt-oig-oasst1-512-12b",
        "h2oai/h2ogpt-oig-oasst1-512-20b",
        "h2oai/h2ogpt-oasst1-512-12b",
        "h2oai/h2ogpt-oasst1-512-20b",
        "h2oai/h2ogpt-gm-oasst1-en-1024-20b",
        "databricks/dolly-v2-12b",
        "h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b-preview-300bt-v2",
        "ehartford/WizardLM-7B-Uncensored",
        "ehartford/WizardLM-13B-Uncensored",
        "AlekseyKorshuk/vicuna-7b",
        "TheBloke/stable-vicuna-13B-HF",
        "decapoda-research/llama-7b-hf",
        "decapoda-research/llama-13b-hf",
        "decapoda-research/llama-30b-hf",
        "junelee/wizard-vicuna-13b",
    ]
)
def test_score_eval(base_model):
    main(
        base_model=base_model,
        chat=False,
        stream_output=False,
        gradio=False,
        eval_sharegpt_prompts_only=500,
        eval_sharegpt_as_output=False,
        num_beams=2,
        infer_devices=False,
    )
