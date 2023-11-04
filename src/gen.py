import ast
import copy
import functools
import inspect
import queue
import sys
import os
import time
import traceback
import typing
import warnings
from datetime import datetime
import requests
from requests import ConnectTimeout, JSONDecodeError
from urllib3.exceptions import ConnectTimeoutError, MaxRetryError, ConnectionError
from requests.exceptions import ConnectionError as ConnectionError2
from requests.exceptions import ReadTimeout as ReadTimeout2

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# more is not useful typically, don't let these go beyond limits and eat up resources
max_cores = max(1, os.cpu_count() // 2)
if os.getenv('NUMEXPR_MAX_THREADS') is None:
    os.environ['NUMEXPR_MAX_THREADS'] = str(min(8, max_cores))
if os.getenv('NUMEXPR_NUM_THREADS') is None:
    os.environ['NUMEXPR_NUM_THREADS'] = str(min(8, max_cores))
if os.getenv('OMP_NUM_THREADS') is None:
    os.environ['OMP_NUM_THREADS'] = str(min(8, max_cores))
if os.getenv('OPENBLAS_NUM_THREADS') is None:
    os.environ['OPENBLAS_NUM_THREADS'] = str(min(8, max_cores))
if os.getenv('DUCKDB_NUM_THREADS') is None:
    os.environ['DUCKDB_NUM_THREADS'] = str(min(4, max_cores))
if os.getenv('RAYON_RS_NUM_CPUS') is None:
    os.environ['RAYON_RS_NUM_CPUS'] = str(min(8, max_cores))
if os.getenv('RAYON_NUM_THREADS') is None:
    os.environ['RAYON_NUM_THREADS'] = str(min(8, max_cores))

import numpy as np
from evaluate_params import eval_func_param_names, no_default_param_names, input_args_list
from enums import DocumentSubset, LangChainMode, no_lora_str, model_token_mapping, no_model_str, \
    LangChainAction, LangChainAgent, DocumentChoice, LangChainTypes, super_source_prefix, \
    super_source_postfix, t5_type, get_langchain_prompts, gr_to_lg, invalid_key_msg, docs_joiner_default, \
    docs_ordering_types_default, docs_token_handling_default, max_input_tokens_public, max_total_input_tokens_public, \
    max_top_k_docs_public, max_top_k_docs_default, max_total_input_tokens_public_api, max_top_k_docs_public_api, \
    max_input_tokens_public_api
from loaders import get_loaders
from utils import set_seed, clear_torch_cache, NullContext, wrapped_partial, EThread, get_githash, \
    import_matplotlib, get_device, makedirs, get_kwargs, start_faulthandler, get_hf_server, FakeTokenizer, \
    have_langchain, set_openai, cuda_vis_check, H2O_Fire, lg_to_gr, str_to_list, str_to_dict, get_token_count, url_alive

start_faulthandler()
import_matplotlib()

SEED = 1236
set_seed(SEED)

from typing import Union

import torch
from transformers import GenerationConfig, AutoModel, TextIteratorStreamer

from prompter import Prompter, inv_prompt_type_to_model_lower, non_hf_types, PromptType, get_prompt, generate_prompt, \
    openai_gpts
from stopping import get_stopping

langchain_actions = [x.value for x in list(LangChainAction)]

langchain_agents_list = [x.value for x in list(LangChainAgent)]


def switch_a_roo_llama(base_model, model_path_llama, load_gptq, load_awq, n_gqa):
    is_gguf = 'GGUF'.lower() in base_model.lower()
    is_ggml = 'GGML'.lower() in base_model.lower()
    postfix = '-GGUF' if is_gguf else '-GGML'
    file_postfix = postfix.lower().replace('-', '.')
    model_split = base_model.split('TheBloke/')
    if base_model.lower().startswith('TheBloke'.lower()) and (is_gguf or is_ggml) and len(model_split) == 2:
        # auto-switch-a-roo to support GGUF/GGML put into base model in UI
        just_model_split = model_split[1].split(postfix)
        if postfix.lower() in base_model.lower() and \
                file_postfix not in base_model and \
                len(just_model_split) == 2:
            just_model = just_model_split[0]
            lower_model = just_model.lower()
            base_model0 = 'https://huggingface.co/%s/resolve/main/%s.Q5_K_M%s' % (base_model, lower_model, file_postfix)
            if url_alive(base_model0):
                base_model = base_model0
        model_path_llama = base_model
        base_model = 'llama'
    if (base_model.endswith('.gguf') or base_model.endswith('.ggml')) and os.path.isfile(base_model):
        # then file but still either gguf or ggml
        model_path_llama = base_model
        base_model = 'llama'

    # some auto things for TheBloke models:
    if 'TheBloke' in base_model and '-GPTQ' in base_model:
        load_gptq = load_gptq or 'model'
    elif 'TheBloke' in base_model and '-AWQ' in base_model:
        load_awq = load_awq or 'model'
    elif '2-70B-GGUF' in model_path_llama:
        n_gqa = n_gqa or 8

    return base_model, model_path_llama, load_gptq, load_awq, n_gqa


def main(
        load_8bit: bool = False,
        load_4bit: bool = False,
        low_bit_mode: int = 1,
        load_half: bool = None,
        load_gptq: str = '',
        use_autogptq: bool = False,
        load_awq: str = '',
        load_exllama: bool = False,
        use_safetensors: bool = False,
        revision: str = None,
        use_gpu_id: bool = True,
        base_model: str = '',
        tokenizer_base_model: str = '',
        lora_weights: str = "",
        gpu_id: int = 0,
        compile_model: bool = None,
        use_cache: bool = None,
        inference_server: str = "",
        prompt_type: Union[int, str] = None,
        prompt_dict: typing.Dict = None,
        system_prompt: str = '',

        # llama and gpt4all settings
        llamacpp_dict: typing.Dict = dict(n_gpu_layers=100, use_mlock=True, n_batch=1024, n_gqa=0),
        model_path_llama: str = 'https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf',
        model_name_gptj: str = 'ggml-gpt4all-j-v1.3-groovy.bin',
        model_name_gpt4all_llama: str = 'ggml-wizardLM-7B.q4_2.bin',
        model_name_exllama_if_no_config: str = 'TheBloke/Nous-Hermes-Llama2-GPTQ',
        exllama_dict: typing.Dict = dict(),
        gptq_dict: typing.Dict = dict(),
        attention_sinks: bool = False,
        sink_dict: typing.Dict = dict(),
        truncation_generation: bool = False,
        hf_model_dict: typing.Dict = dict(),

        model_lock: typing.List[typing.Dict[str, str]] = None,
        model_lock_columns: int = None,
        fail_if_cannot_connect: bool = False,

        # input to generation
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        penalty_alpha: float = None,
        num_beams: int = None,
        repetition_penalty: float = None,
        num_return_sequences: int = None,
        do_sample: bool = None,
        max_new_tokens: int = None,
        min_new_tokens: int = None,
        early_stopping: Union[bool, str] = None,
        max_time: float = None,

        memory_restriction_level: int = None,
        debug: bool = False,
        save_dir: str = None,
        local_files_only: bool = False,
        resume_download: bool = True,
        use_auth_token: Union[str, bool] = False,
        trust_remote_code: Union[str, bool] = True,
        rope_scaling: dict = None,
        max_seq_len: int = None,
        offload_folder: str = "offline_folder",

        src_lang: str = "English",
        tgt_lang: str = "Russian",

        prepare_offline_level: int = 0,
        cli: bool = False,
        cli_loop: bool = True,
        gradio: bool = True,
        gradio_offline_level: int = 0,
        server_name: str = "0.0.0.0",
        share: bool = False,
        open_browser: bool = False,
        root_path: str = "",
        ssl_verify: bool = True,
        ssl_keyfile: str | None = None,
        ssl_certfile: str | None = None,
        ssl_keyfile_password: str | None = None,

        chat: bool = True,
        chat_conversation: typing.List[typing.Tuple[str, str]] = None,
        text_context_list: typing.List[str] = None,
        stream_output: bool = True,
        async_output: bool = True,
        num_async: int = 3,
        show_examples: bool = None,
        verbose: bool = False,
        h2ocolors: bool = True,
        dark: bool = False,  # light tends to be best
        height: int = 600,
        render_markdown: bool = True,
        show_lora: bool = True,
        show_llama: bool = True,
        show_gpt4all: bool = False,
        login_mode_if_model0: bool = False,
        block_gradio_exit: bool = True,
        concurrency_count: int = 1,
        api_open: bool = False,
        allow_api: bool = True,
        input_lines: int = 1,
        gradio_size: str = None,
        show_copy_button: bool = True,
        large_file_count_mode: bool = False,
        pre_load_embedding_model: bool = True,

        auth: Union[typing.List[typing.Tuple[str, str]], str] = None,
        auth_filename: str = None,
        auth_access: str = 'open',
        auth_freeze: bool = False,
        auth_message: str = None,
        guest_name: str = "guest",
        enforce_h2ogpt_api_key: bool = None,
        enforce_h2ogpt_ui_key: bool = None,
        h2ogpt_api_keys: Union[list, str] = [],
        h2ogpt_key: str = None,

        max_max_time=None,
        max_max_new_tokens=None,

        visible_models: list = None,
        visible_visible_models: bool = True,
        visible_submit_buttons: bool = True,
        visible_side_bar: bool = True,
        visible_doc_track: bool = True,
        visible_chat_tab: bool = True,
        visible_doc_selection_tab: bool = True,
        visible_doc_view_tab: bool = True,
        visible_chat_history_tab: bool = True,
        visible_expert_tab: bool = True,
        visible_models_tab: bool = True,
        visible_system_tab: bool = True,
        visible_tos_tab: bool = False,
        visible_login_tab: bool = True,
        visible_hosts_tab: bool = False,
        chat_tables: bool = False,
        visible_h2ogpt_header: bool = True,
        visible_all_prompter_models: bool = False,
        enable_add_models_to_list_ui: bool = False,
        max_raw_chunks: int = None,

        sanitize_user_prompt: bool = False,
        sanitize_bot_response: bool = False,

        extra_model_options: typing.List[str] = [],
        extra_lora_options: typing.List[str] = [],
        extra_server_options: typing.List[str] = [],

        score_model: str = 'auto',

        eval_filename: str = None,
        eval_prompts_only_num: int = 0,
        eval_prompts_only_seed: int = 1234,
        eval_as_output: bool = False,

        langchain_mode: str = None,
        user_path: str = None,
        langchain_modes: list = [LangChainMode.USER_DATA.value, LangChainMode.MY_DATA.value, LangChainMode.LLM.value,
                                 LangChainMode.DISABLED.value],
        langchain_mode_paths: dict = {LangChainMode.USER_DATA.value: None},
        langchain_mode_types: dict = {LangChainMode.USER_DATA.value: LangChainTypes.SHARED.value},
        detect_user_path_changes_every_query: bool = False,

        langchain_action: str = LangChainAction.QUERY.value,
        langchain_agents: list = [],
        force_langchain_evaluate: bool = False,

        visible_langchain_actions: list = [LangChainAction.QUERY.value, LangChainAction.SUMMARIZE_MAP.value,
                                           LangChainAction.EXTRACT.value],
        visible_langchain_agents: list = langchain_agents_list.copy(),

        document_subset: str = DocumentSubset.Relevant.name,
        document_choice: list = [DocumentChoice.ALL.value],

        use_llm_if_no_docs: bool = True,
        load_db_if_exists: bool = True,
        keep_sources_in_context: bool = False,
        db_type: str = 'chroma',
        use_openai_embedding: bool = False,
        use_openai_model: bool = False,
        hf_embedding_model: str = None,
        migrate_embedding_model: str = False,
        auto_migrate_db: bool = False,
        cut_distance: float = 1.64,
        answer_with_sources: bool = True,
        append_sources_to_answer: bool = True,
        show_accordions: bool = True,
        top_k_docs_max_show: int = 10,
        show_link_in_sources: bool = True,
        pre_prompt_query: str = None,
        prompt_query: str = None,
        pre_prompt_summary: str = None,
        prompt_summary: str = None,
        add_chat_history_to_context: bool = True,
        add_search_to_context: bool = False,
        context: str = '',
        iinput: str = '',
        allow_upload_to_user_data: bool = True,
        reload_langchain_state: bool = True,
        allow_upload_to_my_data: bool = True,
        enable_url_upload: bool = True,
        enable_text_upload: bool = True,
        enable_sources_list: bool = True,
        chunk: bool = True,
        chunk_size: int = 512,
        top_k_docs: int = None,
        docs_ordering_type: str = docs_ordering_types_default,
        min_max_new_tokens=256,
        max_input_tokens=None,
        max_total_input_tokens=None,
        docs_token_handling: str = docs_token_handling_default,
        docs_joiner: str = docs_joiner_default,
        hyde_level: int = 0,
        hyde_template: str = None,
        doc_json_mode: bool = False,

        auto_reduce_chunks: bool = True,
        max_chunks: int = 100,
        headsize: int = 50,
        n_jobs: int = -1,
        n_gpus: int = None,

        # urls
        use_unstructured=True,
        use_playwright=False,
        use_selenium=False,

        # pdfs
        use_pymupdf='auto',
        use_unstructured_pdf='auto',
        use_pypdf='auto',
        enable_pdf_ocr='auto',
        enable_pdf_doctr='auto',
        try_pdf_as_html='auto',

        # images
        enable_ocr=False,
        enable_doctr=True,
        enable_pix2struct=False,
        enable_captions=True,

        pre_load_caption_model: bool = False,
        caption_gpu: bool = True,
        caption_gpu_id: Union[int, str] = 'auto',
        captions_model: str = "Salesforce/blip-image-captioning-base",
        doctr_gpu: bool = True,
        doctr_gpu_id: Union[int, str] = 'auto',

        # json
        jq_schema='.[]',

        max_quality: bool = False,

        enable_heap_analytics: bool = True,
        heap_app_id: str = "1680123994",
):
    """

    :param load_8bit: load model in 8-bit using bitsandbytes
    :param load_4bit: load model in 4-bit using bitsandbytes
    :param low_bit_mode: 0: no quantization config 1: change compute 2: nf4 3: double quant 4: 2 and 3
           See: https://huggingface.co/docs/transformers/main_classes/quantization
           If using older bitsandbytes or transformers, 0 is required
    :param load_half: load model in float16 (None means auto, which means True unless t5 based model)
                      otherwise specify bool
    :param load_gptq: to load model with GPTQ, put model_basename here, e.g. 'model' for TheBloke models
    :param use_autogptq: whether to use AutoGPTQ (True) or HF Transformers (False)
           Some models are only supported by one or the other
    :param load_awq: load model with AWQ, e.g. 'model' for TheBloke models
    :param load_exllama: whether to use exllama (only applicable to LLaMa1/2 models with 16-bit or GPTQ
    :param use_safetensors: to use safetensors version (assumes file/HF points to safe tensors version)
    :param revision: Which HF revision to use
    :param use_gpu_id: whether to control devices with gpu_id.  If False, then spread across GPUs
    :param base_model: model HF-type name.  If use --base_model to preload model, cannot unload in gradio in models tab
    :param tokenizer_base_model: tokenizer HF-type name.  Usually not required, inferred from base_model.
    :param lora_weights: LORA weights path/HF link
    :param gpu_id: if use_gpu_id, then use gpu_id for cuda device ID, or auto mode if gpu_id != -1
    :param compile_model Whether to compile the model
    :param use_cache: Whether to use caching in model (some models fail when multiple threads use)
    :param inference_server: Consume base_model as type of model at this address
                             Address can be text-generation-server hosting that base_model
                             e.g. python generate.py --inference_server="http://192.168.1.46:6112" --base_model=h2oai/h2ogpt-oasst1-512-12b

                             Or Address can be "openai_chat" or "openai" for OpenAI API
                             Or Address can be "openai_azure_chat" or "openai_azure" for Azure OpenAI API
                             e.g. python generate.py --inference_server="openai_chat" --base_model=gpt-3.5-turbo
                             e.g. python generate.py --inference_server="openai" --base_model=text-davinci-003
                             e.g. python generate.py --inference_server="openai_azure_chat:<deployment_name>:<baseurl>:<api_version>:<model_version>" --base_model=gpt-3.5-turbo
                             e.g. python generate.py --inference_server="openai_azure:<deployment_name>:<baseurl>:<api_version>:<model_version>" --base_model=text-davinci-003
                             Optionals (Replace with None or just leave empty but keep :)
                                 <deployment_name> of some deployment name
                                 <baseurl>: e.g. "<endpoint>.openai.azure.com" for some <endpoint> without https://
                                 <api_version> of some api, e.g. 2023-05-15
                                 <model_version> e.g. 0613

                             Or Address can be for vLLM:
                              Use: "vllm:IP:port" for OpenAI-compliant vLLM endpoint
                              Use: "vllm_chat:IP:port" for OpenAI-Chat-compliant vLLM endpoint

                              Use: "vllm:http://IP:port/v1" for OpenAI-compliant vLLM endpoint
                              Use: "vllm_chat:http://IP:port/v1" for OpenAI-Chat-compliant vLLM endpoint

                              Use: "vllm:https://IP/v1" for OpenAI-compliant vLLM endpoint
                              Use: "vllm_chat:https://IP/v1" for OpenAI-Chat-compliant vLLM endpoint

                             Or Address can be replicate:
                             Use:
                              --inference_server=replicate:<model name string> will use a Replicate server, requiring a Replicate key.
                              e.g. <model name string> looks like "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"

                             Or Address can be for AWS SageMaker:
                              Use: "sagemaker_chat:<endpoint name>" for chat models that AWS sets up as dialog
                              Use: "sagemaker:<endpoint name>" for foundation models that AWS only text as inputs

    :param prompt_type: type of prompt, usually matched to fine-tuned model or plain for foundational model
    :param prompt_dict: If prompt_type=custom, then expects (some) items returned by get_prompt(..., return_dict=True)
    :param system_prompt: Universal system prompt to use if model supports, like LLaMa2, regardless of prompt_type definition.
           Useful for langchain case to control behavior, or OpenAI and Replicate.
           If None, 'None', or 'auto', then for LLaMa or other models that internally have system_prompt, will use default for each model
           If '', then no system prompt (no empty template given to model either, just no system part added at all)
           If some string not in ['None', 'auto'], then use that as system prompt
           Default is '', no system_prompt, because often it hurts performance/accuracy

    :param llamacpp_dict:
           n_gpu_layers: for llama.cpp based models, number of GPU layers to offload (default is all by using large value)
           use_mlock: when using `llama.cpp` based CPU models, for computers with low system RAM or slow CPUs, recommended False
           n_batch: Can make smaller to 128 for slower low-memory CPU systems
           n_gqa: Required to be 8 for LLaMa 70B
           ... etc. anything that could be passed to llama.cpp or GPT4All models
           e.g. python generate.py --base_model='llama' --prompt_type=llama2 --score_model=None --langchain_mode='UserData' --user_path=user_path --llamacpp_dict="{'n_gpu_layers':25,'n_batch':128}"
    :param model_path_llama: model path or URL (for auto-download)
    :param model_name_gptj: model path or URL (for auto-download)
    :param model_name_gpt4all_llama: model path or URL (for auto-download)
    :param model_name_exllama_if_no_config: exllama model's full path for model, tokenizer, generator for use when no HuggingFace config
    :param exllama_dict for setting various things for Exllama class
           E.g. compress_pos_emb,
                set_auto_map,
                gpu_peer_fix,
                alpha_value,
                matmul_recons_thd,
                fused_mlp_thd
                sdp_thd
                fused_attn
                matmul_fused_remap
                rmsnorm_no_half2
                rope_no_half2
                matmul_no_half2
                silu_no_half2
                concurrent_streams
           E.g. to set memory to be split across 2 GPUs, use --exllama_dict="{'set_auto_map':20,20}"
    :param gptq_dict: Choices for AutoGPTQ, e.g. one can change defaults to these non-defaults:
         inject_fused_attention=False
         disable_exllama=True
         use_triton=True
    :param attention_sinks: Whether to enable attention sinks. Requires in local repo:
         git clone https://github.com/tomaarsen/attention_sinks.git
    :param sink_dict: dict of options for attention sinks
    :param hf_model_dict: dict of options for HF models using transformers

    :param truncation_generation: Whether (for torch) to terminate generation once reach context length of model.
            For some models, perplexity becomes critically large beyond context
            For other models like Mistral, one can generate beyond max_seq_len set to 4096 or 8192 without issue, since based upon 32k embeddings
            codellama can also generate beyond its 16k context length
            So default is off, but for simpler/older models True may be wise to avoid bad generations

    :param model_lock: Lock models to specific combinations, for ease of use and extending to many models
           Only used if gradio = True
           List of dicts, each dict has base_model, tokenizer_base_model, lora_weights, inference_server, prompt_type, and prompt_dict
           If all models have same prompt_type, and prompt_dict, can still specify that once in CLI outside model_lock as default for dict
           Can specify model_lock instead of those items on CLI
           As with CLI itself, base_model can infer prompt_type and prompt_dict if in prompter.py.
             Also, tokenizer_base_model and lora_weights are optional.
             Also, inference_server is optional if loading model from local system.
           All models provided will automatically appear in compare model mode
           Model loading-unloading and related choices will be disabled.  Model/lora/server adding will be disabled
    :param model_lock_columns: How many columns to show if locking models (and so showing all at once)
           If None, then defaults to up to 3
           if -1, then all goes into 1 row
           Maximum value is 4 due to non-dynamic gradio rendering elements
    :param fail_if_cannot_connect: if doing model locking (e.g. with many models), fail if True.  Otherwise ignore.
           Useful when many endpoints and want to just see what works, but still have to wait for timeout.

    :param temperature: generation temperature
    :param top_p: generation top_p
    :param top_k: generation top_k
    :param penalty_alpha: penalty_alpha>0 and top_k>1 enables contrastive search (not all models support)
    :param num_beams: generation number of beams
    :param repetition_penalty: generation repetition penalty
    :param num_return_sequences: generation number of sequences (1 forced for chat)
    :param do_sample: generation sample
    :param max_new_tokens: generation max new tokens
    :param min_new_tokens: generation min tokens
    :param early_stopping: generation early stopping
    :param max_time: maximum time to allow for generation
    :param memory_restriction_level: 0 = no restriction to tokens or model, 1 = some restrictions on token 2 = HF like restriction 3 = very low memory case
    :param debug: enable debug mode
    :param save_dir: directory chat data is saved to
    :param local_files_only: whether to only use local files instead of doing to HF for models
    :param resume_download: whether to resume downloads from HF for models
    :param use_auth_token: whether to use HF auth token (requires CLI did huggingface-cli login before)
    :param trust_remote_code: whether to use trust any code needed for HF model
    :param rope_scaling:
           For HF transformers model: scaling for rope-based models.
           For long context models that have been tuned for a specific size, you have to only use that specific size by setting the `--rope_scaling` exactly correctly
            e.g. --rope_scaling="{'type':'dynamic', 'factor':4}"
            e.g. --rope_scaling="{'type':'linear', 'factor':4}"
            e.g. python generate.py --rope_scaling="{'type':'linear','factor':4}" --base_model=lmsys/vicuna-13b-v1.5-16k --hf_embedding_model=sentence-transformers/all-MiniLM-L6-v2 --load_8bit=True --langchain_mode=UserData --user_path=user_path --prompt_type=vicuna11 --h2ocolors=False
           For exllama model: --rope_scaling="{'alpha_value':4}" .  This automatically scales max_seq_len for exllama
    :param max_seq_len: Manually set maximum sequence length for the LLM
    :param offload_folder: path for spilling model onto disk
    :param src_lang: source languages to include if doing translation (None = all)
    :param tgt_lang: target languages to include if doing translation (None = all)

    :param prepare_offline_level:
           Whether to just prepare for offline use, do not go into cli, eval, or gradio run modes
           0 : no prep
           1: prepare just h2oGPT with exact same setup as passed to CLI and ensure all artifacts for h2oGPT alone added to ~/.cache/
           2: prepare h2oGPT + all inference servers so h2oGPT+inference servers can use the ~/.cache/
    :param cli: whether to use CLI (non-gradio) interface.
    :param cli_loop: whether to loop for CLI (False usually only for testing)
    :param gradio: whether to enable gradio, or to enable benchmark mode
    :param gradio_offline_level: > 0, then change fonts so full offline
           == 1 means backend won't need internet for fonts, but front-end UI might if font not cached
           == 2 means backend and frontend don't need internet to download any fonts.
           Note: Some things always disabled include HF telemetry, gradio telemetry, chromadb posthog that involve uploading.
           This option further disables google fonts for downloading, which is less intrusive than uploading,
           but still required in air-gapped case.  The fonts don't look as nice as google fonts, but ensure full offline behavior.
           Also set --share=False to avoid sharing a gradio live link.
    :param server_name: IP to use.  In linux 0.0.0.0 is good choice so exposed to outside host, else for only local use 127.0.0.1.
                        For windows/MAC 0.0.0.0 or 127.0.0.1 will work, but may need to specify actual LAN IP address for other LAN clients to see.
    :param share: whether to share the gradio app with sharable URL
    :param open_browser: whether to automatically open browser tab with gradio UI
    :param root_path: The root path (or "mount point") of the application,
           if it's not served from the root ("/") of the domain. Often used when the application is behind a reverse proxy
           that forwards requests to the application. For example, if the application is served at "https://example.com/myapp",
           the `root_path` should be set to "/myapp".
    :param ssl_verify: passed go gradio launch
    :param ssl_keyfile: passed go gradio launch
    :param ssl_certfile: passed go gradio launch
    :param ssl_keyfile_password: passed go gradio launch

    :param chat: whether to enable chat mode with chat history
    :param chat_conversation: list of tuples of (human, bot) conversation pre-appended to existing chat when using instruct/chat models
           Requires also add_chat_history_to_context = True
           It does *not* require chat=True, so works with nochat_api etc.
    :param text_context_list: List of strings to add to context for non-database version of document Q/A for faster handling via API etc.
           Forces LangChain code path and uses as many entries in list as possible given max_seq_len, with first assumed to be most relevant and to go near prompt.
    :param stream_output: whether to stream output
    :param async_output: Whether to do asyncio handling
           For summarization
           Applicable to HF TGI server
           Only if stream_output=False in CLI, UI, or API
    :param num_async: Number of simultaneously allowed asyncio calls to make for async_output
           Too many will overload inference server, too few will be too slow
    :param show_examples: whether to show clickable examples in gradio
    :param verbose: whether to show verbose prints
    :param h2ocolors: whether to use H2O.ai theme
    :param dark: whether to use dark mode for UI by default (still controlled in UI)
    :param height: height of chat window
    :param render_markdown: Whether to render markdown in chatbot UI.  In some cases this distorts the rendering.
           https://github.com/gradio-app/gradio/issues/4344#issuecomment-1771963021
    :param show_lora: whether to show LORA options in UI (expert so can be hard to understand)
    :param show_llama: whether to show LLaMa.cpp/GPT4All options in UI (only likely useful if have weak GPUs)
    :param show_gpt4all: whether to show GPT4All models in UI (not often useful, llama.cpp models best)
    :param login_mode_if_model0: set to True to load --base_model after client logs in, to be able to free GPU memory when model is swapped
    :param block_gradio_exit: whether to block gradio exit (used for testing)
    :param concurrency_count: gradio concurrency count (1 is optimal for LLMs)
    :param api_open: If False, don't let API calls skip gradio queue
    :param allow_api: whether to allow API calls at all to gradio server
    :param input_lines: how many input lines to show for chat box (>1 forces shift-enter for submit, else enter is submit)
    :param gradio_size: Overall size of text and spaces: "xsmall", "small", "medium", "large".
           Small useful for many chatbots in model_lock mode
    :param show_copy_button: Whether to show copy button for chatbots
    :param large_file_count_mode: Whether to force manual update to UI of drop-downs, good idea if millions of chunks or documents
    :param pre_load_embedding_model: Whether to preload embedding model for shared use across DBs and users (multi-thread safe only)

    :param auth: gradio auth for launcher in form [(user1, pass1), (user2, pass2), ...]
                 e.g. --auth=[('jon','password')] with no spaces
                 e.g. --auth="[('jon', 'password)())(')]" so any special characters can be used
                 e.g. --auth=auth.json to specify persisted state file with name auth.json (auth_filename then not required)
                 e.g. --auth='' will use default auth.json as file name for persisted state file (auth_filename then not required)
                 e.g. --auth=None will use no auth, but still keep track of auth state, just not from logins
    :param auth_filename:
         Set auth filename, used only if --auth= was passed list of user/passwords
    :param auth_access:
         'open': Allow new users to be added
         'closed': Stick to existing users
    :param auth_freeze: whether freeze authentication based upon current file, no longer update file
    :param auth_message: Message to show if having users login, fixed if passed, else dynamic internally
    :param guest_name: guess name if using auth and have open access.
           If '', then no guest allowed even if open access, then all databases for each user always persisted
    :param enforce_h2ogpt_api_key: Whether to enforce h2oGPT token usage for API
    :param enforce_h2ogpt_ui_key: Whether to enforce h2oGPT token usage for UI (same keys as API assumed)
    :param h2ogpt_api_keys: list of tokens allowed for API access or file accessed on demand for json of list of keys
    :param h2ogpt_key: E.g. can be set when accessing gradio h2oGPT server from local gradio h2oGPT server that acts as client to that inference server

    :param max_max_time: Maximum max_time for gradio slider
    :param max_max_new_tokens: Maximum max_new_tokens for gradio slider
    :param min_max_new_tokens: Minimum of max_new_tokens, when auto-scaling down to handle more docs/prompt, but still let generation have some tokens
    :param max_input_tokens: Max input tokens to place into model context for each LLM call
                             -1 means auto, fully fill context for query, and fill by original document chunk for summarization
                             >=0 means use that to limit context filling to that many tokens
    :param max_total_input_tokens: like max_input_tokens but instead of per LLM call, applies across all LLM calls for single summarization/extraction action
    :param docs_token_handling: 'chunk' means fill context with top_k_docs (limited by max_input_tokens or model_max_len) chunks for query
                                                                     or top_k_docs original document chunks summarization
                                None or 'split_or_merge' means same as 'chunk' for query, while for summarization merges documents to fill up to max_input_tokens or model_max_len tokens

    :param docs_joiner: string to join lists of text when doing split_or_merge.  None means '\n\n'
    :param hyde_level: HYDE level for HYDE approach (https://arxiv.org/abs/2212.10496)
                 0: No HYDE
                 1: Use non-document-based LLM response and original query for embedding query
                 2: Use document-based LLM response and original query for embedding query
                 3+: Continue iterations of embedding prior answer and getting new response
    :param hyde_template:
                 None, 'None', 'auto' uses internal value and enable
                 '{query}' is minimal template one can pass
    :param visible_models: Which models in model_lock list to show by default
           Takes integers of position in model_lock (model_states) list or strings of base_model names
           Ignored if model_lock not used
           For nochat API, this is single item within a list for model by name or by index in model_lock
                                If None, then just use first model in model_lock list
                                If model_lock not set, use model selected by CLI --base_model etc.
           Note that unlike h2ogpt_key, this visible_models only applies to this running h2oGPT server,
              and the value is not used to access the inference server.
              If need a visible_models for an inference server, then use --model_lock and group together.

    :param visible_visible_models: Whether visible models drop-down is visible in UI
    :param visible_submit_buttons: whether submit buttons are visible when UI first comes up
    :param visible_side_bar: whether left side bar is visible when UI first comes up
    :param visible_doc_track: whether left side bar's document tracking is visible when UI first comes up
    :param visible_chat_tab: "" for chat tab
    :param visible_doc_selection_tab:  "" for doc selection tab
    :param visible_doc_view_tab: "" for doc view tab
    :param visible_chat_history_tab: "" for chat history tab
    :param visible_expert_tab: "" for expert tab
    :param visible_models_tab: "" for models tab
    :param visible_system_tab: "" for system tab
    :param visible_tos_tab: "" for ToS tab
    :param visible_login_tab: "" for Login tab (needed for persistence or to enter key for UI access to models and ingestion)
    :param visible_hosts_tab: "" for hosts tab
    :param chat_tables: Just show Chat as block without tab (useful if want only chat view)
    :param visible_h2ogpt_header: Whether github stars, URL, logo, and QR code are visible
    :param visible_all_prompter_models: Whether to show all prompt_type_to_model_name items or just curated ones
    :param max_raw_chunks: Maximum number of chunks to show in UI when asking for raw DB text from documents/collection

    :param sanitize_user_prompt: whether to remove profanity from user input (slows down input processing)
      Requires optional packages:
      pip install alt-profanity-check==1.2.2 better-profanity==0.7.0
    :param sanitize_bot_response: whether to remove profanity and repeat lines from bot output (about 2x slower generation for long streaming cases due to better_profanity being slow)
    :param extra_model_options: extra models to show in list in gradio
    :param extra_lora_options: extra LORA to show in list in gradio
    :param extra_server_options: extra servers to show in list in gradio
    :param score_model: which model to score responses
           None: no response scoring
           'auto': auto mode, '' (no model) for CPU or 1 GPU, 'OpenAssistant/reward-model-deberta-v3-large-v2' for >=2 GPUs,
            because on CPU takes too much compute just for scoring response
    :param eval_filename: json file to use for evaluation, if None is sharegpt
    :param eval_prompts_only_num: for no gradio benchmark, if using eval_filename prompts for eval instead of examples
    :param eval_prompts_only_seed: for no gradio benchmark, seed for eval_filename sampling
    :param eval_as_output: for no gradio benchmark, whether to test eval_filename output itself

    :param langchain_mode: Data source to include.  Choose "UserData" to only consume files from make_db.py.
           None: auto mode, check if langchain package exists, at least do LLM if so, else Disabled
           If not passed, then chosen to be first langchain_modes, else langchain_mode->Disabled is set if no langchain_modes either
           WARNING: wiki_full requires extra data processing via read_wiki_full.py and requires really good workstation to generate db, unless already present.
    :param user_path: user path to glob from to generate db for vector search, for 'UserData' langchain mode.
           If already have db, any new/changed files are added automatically if path set, does not have to be same path used for prior db sources
    :param langchain_modes: dbs to generate at launch to be ready for LLM
           Apart from additional user-defined collections, can include ['wiki', 'wiki_full', 'UserData', 'MyData', 'github h2oGPT', 'DriverlessAI docs']
             But wiki_full is expensive and requires preparation
           To allow personal space only live in session, add 'MyData' to list
           Default: If only want to consume local files, e.g. prepared by make_db.py, only include ['UserData']
           If have own user modes, need to add these here or add in UI.
    :param langchain_mode_paths: dict of langchain_mode keys and disk path values to use for source of documents
           E.g. "{'UserData2': 'userpath2'}"
           A disk path be None, e.g. --langchain_mode_paths="{'UserData2': None}" even if existing DB, to avoid new documents being added from that path, source links that are on disk still work.
           If `--user_path` was passed, that path is used for 'UserData' instead of the value in this dict
    :param langchain_mode_types: dict of langchain_mode keys and database types
           E.g. python generate.py --base_model=llama --langchain_modes=['TestData'] --langchain_mode_types="{'TestData':'shared'}"
           The type is attempted to be inferred if directory already exists, then don't have to pass this
    :param detect_user_path_changes_every_query: whether to detect if any files changed or added every similarity search (by file hashes).
           Expensive for large number of files, so not done by default.  By default only detect changes during db loading.

    :param langchain_action: Mode langchain operations in on documents.
            Query: Make query of document(s)
            Summarize or Summarize_map_reduce: Summarize document(s) via map_reduce
            Summarize_all: Summarize document(s) using entire document at once
            Summarize_refine: Summarize document(s) using entire document, and try to refine before returning summary
            Extract: Extract information from document(s) via map (no reduce)
    :param langchain_agents: Which agents to use
            'search': Use Web Search as context for LLM response, e.g. SERP if have SERPAPI_API_KEY in env
    :param force_langchain_evaluate: Whether to force langchain LLM use even if not doing langchain, mostly for testing.

    :param visible_langchain_actions: Which actions to allow
    :param visible_langchain_agents: Which agents to allow

    :param document_subset: Default document choice when taking subset of collection
    :param document_choice: Chosen document(s) by internal name, 'All' means use all docs

    :param use_llm_if_no_docs: Whether to use LLM even if no documents, when langchain_mode=UserData or MyData or custom
    :param load_db_if_exists: Whether to load chroma db if exists or re-generate db
    :param keep_sources_in_context: Whether to keep url sources in context, not helpful usually
    :param db_type: 'faiss' for in-memory
                    'chroma' (for chroma >= 0.4)
                    'chroma_old' (for chroma < 0.4) -- recommended for large collections
                    'weaviate' for persisted on disk
    :param use_openai_embedding: Whether to use OpenAI embeddings for vector db
    :param use_openai_model: Whether to use OpenAI model for use with vector db
    :param hf_embedding_model: Which HF embedding model to use for vector db
           Default is instructor-large with 768 parameters per embedding if have GPUs, else all-MiniLM-L6-v2 if no GPUs
           Can also choose simpler model with 384 parameters per embedding: "sentence-transformers/all-MiniLM-L6-v2"
           Can also choose even better embedding with 1024 parameters: 'hkunlp/instructor-xl'
           We support automatically changing of embeddings for chroma, with a backup of db made if this is done
    :param migrate_embedding_model: whether to use hf_embedding_model embedding even if database already had an embedding set.
           used to migrate all embeddings to a new one, but will take time to re-embed.
           Default (False) is to use the prior embedding for existing databases, and only use hf_embedding_model for new databases
           If had old database without embedding saved, then hf_embedding_model is also used.
    :param auto_migrate_db: whether to automatically migrate any chroma<0.4 database from duckdb -> sqlite version
    :param cut_distance: Distance to cut off references with larger distances when showing references.
           1.64 is good to avoid dropping references for all-MiniLM-L6-v2, but instructor-large will always show excessive references.
           For all-MiniLM-L6-v2, a value of 1.5 can push out even more references, or a large value of 100 can avoid any loss of references.
    :param answer_with_sources: Whether to determine (and return) sources
    :param append_sources_to_answer: Whether to place source information in chat response (ignored by LLM).  Always disabled for API.
    :param show_accordions: whether to show accordion for document references in chatbot UI
    :param top_k_docs_max_show: Max number of docs to show in UI for sources
           If web search is enabled, then this is modified to be max(top_k_docs_max_show, number of links used in search)
    :param show_link_in_sources: Whether to show URL link to source document in references
    :param pre_prompt_query: prompt before documents to query, if None then use internal defaults
    :param prompt_query: prompt after documents to query, if None then use internal defaults
    :param pre_prompt_summary: prompt before documents to summarize/extract from, if None then use internal defaults
    :param prompt_summary: prompt after documents to summarize/extract from, if None then use internal defaults
           For summarize/extract, normal to have empty query (nothing added in ask anything in UI or empty string in API)
           If pass query, template is "Focusing on %s, %s" % (query, prompt_summary)
           If pass query and iinput, template is "Focusing on %s, %s, %s" % (query, iinput, prompt_summary)
    :param doc_json_mode: Use system prompting approach with JSON input and output, e.g. for codellama or GPT-4
    :param add_chat_history_to_context: Include chat context when performing action
           Not supported yet for openai_chat when using document collection instead of LLM
           Also not supported when using CLI mode
    :param add_search_to_context: Include web search in context as augmented prompt
    :param context: Default context to use (for system pre-context in gradio UI)
           context comes before chat_conversation and any document Q/A from text_context_list
    :param iinput: Default input for instruction-based prompts
    :param allow_upload_to_user_data: Whether to allow file uploads to update shared vector db (UserData or custom user dbs)
           Ensure pass user_path for the files uploaded to be moved to this location for linking.
    :param reload_langchain_state: Whether to reload langchain_modes.pkl file that contains any new user collections.
    :param allow_upload_to_my_data: Whether to allow file uploads to update personal vector db
    :param enable_url_upload: Whether to allow upload from URL
    :param enable_text_upload: Whether to allow upload of text
    :param enable_sources_list: Whether to allow list (or download for non-shared db) of list of sources for chosen db
    :param chunk: Whether to chunk data (True unless know data is already optimally chunked)
    :param chunk_size: Size of chunks, with typically top-4 passed to LLM, so needs to be in context length
    :param top_k_docs: For langchain_action query: number of chunks to give LLM
                       -1 : auto-fills context up to max_seq_len
                       For langchain_action summarize/extract: number of document parts, like pages for PDF.
                       There's no such thing as chunks for summarization.
                       -1 : auto-fills context up to max_seq_len
    :param docs_ordering_type:
        Type of ordering of docs.
        'best_first': Order by score so score is worst match near prompt
        'best_near_prompt' or 'reverse_sort' : reverse docs order so most relevant is closest to question.
           Best choice for sufficiently smart model, and truncation occurs for oldest context, so best then too.
           But smaller 6_9 models fail to use newest context and can get stuck on old information.
        '' or None (i.e. default) or 'reverse_ucurve_sort' : Sort so most relevant is either near start or near end
           Best to avoid "lost in middle" as well as avoid hallucinating off starting content that LLM focuses on alot.
    :param auto_reduce_chunks: Whether to automatically reduce top_k_docs to fit context given prompt
    :param max_chunks: If top_k_docs=-1, maximum number of chunks to allow
    :param headsize: Maximum number of characters for head of document document for UI to show
    :param n_jobs: Number of processors to use when consuming documents (-1 = all, is default)
    :param n_gpus: Number of GPUs (None = autodetect)

    :param use_unstructured: Enable unstructured URL loader
    :param use_playwright: Enable PlayWright URL loader
    :param use_selenium: Enable Selenium URL loader

    :param use_pymupdf: enable PyMUPDF 'auto' means use first, use others if they are 'auto' if no result
    :param use_unstructured_pdf: enable Unstructured PDF loader, 'auto' means use if pymupdf fails to get doc result
    :param use_pypdf: enable PyPDF loader 'auto' means use if unstructured fails to get doc result
    :param enable_pdf_ocr: 'auto' means only use OCR if normal text extraction fails.  Useful for pure image-based PDFs with text.
                                  if enable_pdf_doctr == 'on' then don't do.
                            'on' means always do OCR as additional parsing of same documents
                            'off' means don't do OCR (e.g. because it's slow even if 'auto' only would trigger if nothing else worked)
    :param enable_pdf_doctr: Whether to support doctr on pdfs, 'auto' means use do if failed to get doc result so far
    :param try_pdf_as_html: Try "PDF" as if HTML file, in case web link has .pdf extension but really is just HTML

    :param enable_ocr: Whether to support OCR on images
    :param enable_doctr: Whether to support doctr on images (using OCR better than enable_ocr=True)
    :param enable_pix2struct: Whether to support pix2struct on images for captions
    :param enable_captions: Whether to support captions using BLIP for image files as documents,
           then preloads that model if pre_load_caption_model=True

    :param pre_load_caption_model: Whether to preload caption model (True), or load after forking parallel doc loader (False)
           parallel loading disabled if preload and have images, to prevent deadlocking on cuda context
           Recommended if using larger caption model or doing production serving with many users to avoid GPU OOM if many would use model at same time
           Also applies to DocTR
    :param captions_model: Which model to use for captions.
           captions_model: str = "Salesforce/blip-image-captioning-base",  # continue capable
           captions_model: str = "Salesforce/blip2-flan-t5-xl",   # question/answer capable, 16GB state
           captions_model: str = "Salesforce/blip2-flan-t5-xxl",  # question/answer capable, 60GB state
           Note: opt-based blip2 are not permissive license due to opt and Meta license restrictions
           Disabled for CPU since BLIP requires CUDA
    :param caption_gpu: If support caption, then use GPU if exists
    :param caption_gpu_id: Which GPU id to use, if 'auto' then select 0

    :param doctr_gpu: If support doctr, then use GPU if exists
    :param doctr_gpu_id: Which GPU id to use, if 'auto' then select 0

    :param jq_schema: control json loader
           By default '.[]' ingests everything in brute-force way, but better to match your schema
           See: https://python.langchain.com/docs/modules/data_connection/document_loaders/json#using-jsonloader

    :param max_quality: Choose maximum quality ingestion with all available parsers
           Pro: Catches document when some default parsers would fail
           Pro: Enables DocTR that has much better OCR than Tesseract
           Con: Fills DB with results from all parsers, so similarity search gives redundant results

    :param enable_heap_analytics: Toggle telemetry.
    :param heap_app_id: App ID for Heap, change to your ID.
    :return:
    """
    if base_model is None:
        base_model = ''
    if tokenizer_base_model is None:
        tokenizer_base_model = ''
    if lora_weights is None:
        lora_weights = ''
    if inference_server is None:
        inference_server = ''

    # listen to env if set
    model_lock = os.getenv('model_lock', str(model_lock))
    model_lock = ast.literal_eval(model_lock)

    chat_conversation = str_to_list(chat_conversation)
    text_context_list = str_to_list(text_context_list)
    llamacpp_dict = str_to_dict(llamacpp_dict)

    # switch-a-roo on base_model so can pass GGUF/GGML as base model
    base_model, model_path_llama, load_gptq, load_awq, llamacpp_dict['n_gqa'] = \
        switch_a_roo_llama(base_model, model_path_llama, load_gptq, load_awq, llamacpp_dict.get('n_gqa', 0))

    # add others to single dict
    llamacpp_dict['model_path_llama'] = model_path_llama
    llamacpp_dict['model_name_gptj'] = model_name_gptj
    llamacpp_dict['model_name_gpt4all_llama'] = model_name_gpt4all_llama
    llamacpp_dict['model_name_exllama_if_no_config'] = model_name_exllama_if_no_config
    # if user overrides but doesn't set these:
    if 'n_batch' not in llamacpp_dict:
        llamacpp_dict['n_batch'] = 128
    if 'n_gpu_layers' not in llamacpp_dict:
        llamacpp_dict['n_gpu_layers'] = 100
    if 'n_gqa' not in llamacpp_dict:
        llamacpp_dict['n_gqa'] = 0

    exllama_dict = str_to_dict(exllama_dict)
    gptq_dict = str_to_dict(gptq_dict)
    sink_dict = str_to_dict(sink_dict)
    hf_model_dict = str_to_dict(hf_model_dict)

    if os.environ.get('SERPAPI_API_KEY') is None and LangChainAgent.SEARCH.value in visible_langchain_agents:
        visible_langchain_agents.remove(LangChainAgent.SEARCH.value)

    if model_lock:
        assert gradio, "model_lock only supported for gradio=True"
        assert not cli, "model_lock only supported for cli=False"
        assert not (not cli and not gradio), "model_lock only supported for eval (cli=gradio=False)"
        assert not base_model, "Don't specify model_lock and base_model"
        assert not tokenizer_base_model, "Don't specify model_lock and tokenizer_base_model"
        assert not lora_weights, "Don't specify model_lock and lora_weights"
        assert not inference_server, "Don't specify model_lock and inference_server"
        # assert not prompt_type, "Don't specify model_lock and prompt_type"
        # assert not prompt_dict, "Don't specify model_lock and prompt_dict"

    n_jobs = int(os.getenv('n_jobs', str(n_jobs)))
    is_hf = bool(int(os.getenv("HUGGINGFACE_SPACES", '0')))
    is_gpth2oai = bool(int(os.getenv("GPT_H2O_AI", '0')))
    is_public = is_hf or is_gpth2oai  # multi-user case with fixed model and disclaimer
    if enforce_h2ogpt_ui_key is None:
        # nominally allow UI access public or not
        enforce_h2ogpt_ui_key = False
    if is_public:
        visible_tos_tab = visible_hosts_tab = True
        if enforce_h2ogpt_api_key is None:
            enforce_h2ogpt_api_key = True
    else:
        if enforce_h2ogpt_api_key is None:
            enforce_h2ogpt_api_key = False
    if isinstance(h2ogpt_api_keys, str) and not os.path.isfile(h2ogpt_api_keys):
        h2ogpt_api_keys = str_to_list(h2ogpt_api_keys)
    if memory_restriction_level is None:
        memory_restriction_level = 2 if is_hf else 0  # 2 assumes run on 24GB consumer GPU
    else:
        assert 0 <= memory_restriction_level <= 3, "Bad memory_restriction_level=%s" % memory_restriction_level
    if n_jobs == -1:
        # if -1, assume hypercores, don't use, force user to pass n_jobs to be specific if not standard cores
        n_jobs = max(1, os.cpu_count() // 2)
    if is_public and os.getenv('n_jobs') is None:
        n_jobs = min(n_jobs, max(1, min(os.cpu_count() // 2, 8)))
    admin_pass = os.getenv("ADMIN_PASS")
    # will sometimes appear in UI or sometimes actual generation, but maybe better than empty result
    # but becomes unrecoverable sometimes if raise, so just be silent for now
    raise_generate_gpu_exceptions = True

    rope_scaling = str_to_dict(rope_scaling)

    if isinstance(auth, str):
        if auth.strip().startswith('['):
            auth = str_to_list(auth)
    if isinstance(auth, str) and auth:
        auth_filename = auth
    if not auth_filename:
        auth_filename = "auth.json"
    assert isinstance(auth, (str, list, tuple, type(None))), "Unknown type %s for auth=%s" % (type(auth), auth)

    # allow set token directly
    use_auth_token = os.environ.get("HUGGING_FACE_HUB_TOKEN", use_auth_token)
    allow_upload_to_user_data = bool(
        int(os.environ.get("allow_upload_to_user_data", str(int(allow_upload_to_user_data)))))
    allow_upload_to_my_data = bool(int(os.environ.get("allow_upload_to_my_data", str(int(allow_upload_to_my_data)))))
    height = int(os.environ.get("HEIGHT", height))
    h2ocolors = bool(int(os.getenv('h2ocolors', h2ocolors)))

    # allow enabling langchain via ENV
    # FIRST PLACE where LangChain referenced, but no imports related to it
    langchain_modes = ast.literal_eval(os.environ.get("langchain_modes", str(langchain_modes)))
    if not isinstance(langchain_modes, list):
        langchain_modes = []
    # always allow DISABLED
    if LangChainMode.DISABLED.value not in langchain_modes:
        langchain_modes.append(LangChainMode.DISABLED.value)
    if not have_langchain:
        # only allow disabled, not even LLM that is langchain related
        langchain_mode = LangChainMode.DISABLED.value
        langchain_modes = [langchain_mode]

    # update
    langchain_mode_paths = str_to_dict(langchain_mode_paths)
    langchain_mode_types = str_to_dict(langchain_mode_types)
    for lmode in [LangChainMode.GITHUB_H2OGPT.value,
                  LangChainMode.H2O_DAI_DOCS.value,
                  LangChainMode.WIKI.value,
                  LangChainMode.WIKI_FULL.value,
                  ]:
        if lmode not in langchain_mode_types:
            langchain_mode_types[lmode] = 'shared'
    if lmode not in langchain_mode_paths:
        langchain_mode_types[lmode] = ''
    if user_path:
        user_path = makedirs(user_path, use_base=True)
        langchain_mode_paths['UserData'] = user_path
        langchain_mode_paths['UserData'] = LangChainTypes.SHARED.value

    if is_public:
        allow_upload_to_user_data = False
        if LangChainMode.USER_DATA.value in langchain_modes:
            langchain_modes.remove(LangChainMode.USER_DATA.value)
    if max_raw_chunks is None:
        max_raw_chunks = 30 if is_public else 1000000

    # in-place, for non-scratch dbs
    if allow_upload_to_user_data:
        # always listen to CLI-passed user_path if passed
        if user_path:
            langchain_mode_paths['UserData'] = user_path

    assert langchain_action in langchain_actions, "Invalid langchain_action %s not in %s" % (
        langchain_action, langchain_actions)
    assert len(
        set(langchain_agents).difference(langchain_agents_list)) == 0, "Invalid langchain_agents %s" % langchain_agents

    # auto-set langchain_mode
    langchain_mode = os.environ.get("LANGCHAIN_MODE", langchain_mode)
    if have_langchain and langchain_mode is None:
        # start in chat mode, in case just want to chat and don't want to get "No documents to query" by default.
        if LangChainMode.LLM.value in langchain_modes:
            langchain_mode = LangChainMode.LLM.value
        elif len(langchain_modes) >= 1:
            # infer even if don't pass which langchain_mode, just langchain_modes.
            langchain_mode = langchain_modes[0]
        if allow_upload_to_user_data and not is_public and langchain_mode_paths['UserData']:
            if verbose:
                print("Auto set langchain_mode=%s.  Could use UserData instead." % langchain_mode, flush=True)
        elif allow_upload_to_my_data:
            if verbose:
                print("Auto set langchain_mode=%s.  Could use MyData instead."
                      "  To allow UserData to pull files from disk,"
                      " set user_path or langchain_mode_paths, and ensure allow_upload_to_user_data=True" % langchain_mode,
                      flush=True)
        else:
            raise RuntimeError("Please pass --langchain_mode=<chosen mode> out of %s" % langchain_modes)
    if not have_langchain and langchain_mode not in [None, LangChainMode.DISABLED.value, LangChainMode.LLM.value]:
        raise RuntimeError("Asked for LangChain mode but langchain python package cannot be found.")
    if langchain_mode is None:
        # if not set yet, disable
        langchain_mode = LangChainMode.DISABLED.value
        print("Auto set langchain_mode=%s  Have langchain package: %s" % (langchain_mode, have_langchain), flush=True)
    # go ahead and add
    if langchain_mode not in langchain_modes:
        langchain_modes.append(langchain_mode)

    if is_public:
        # See also get_minmax_top_k_docs()
        # as another restriction apart from top_k_docs and when using long context models
        # model will limit more if required
        max_input_tokens = max_input_tokens_public if max_input_tokens is None else max_input_tokens
        max_total_input_tokens = max_total_input_tokens_public if max_total_input_tokens is None else max_total_input_tokens
        allow_upload_to_user_data = False
        input_lines = 1  # ensure set, for ease of use
        temperature = 0.2 if temperature is None else temperature
        top_p = 0.85 if top_p is None else top_p
        top_k = 70 if top_k is None else top_k
        penalty_alpha = 0.0 if penalty_alpha is None else penalty_alpha
        if is_hf:
            do_sample = True if do_sample is None else do_sample
            top_k_docs = 3 if top_k_docs is None else top_k_docs
        else:
            # by default don't sample, too chatty
            do_sample = False if do_sample is None else do_sample
            # now 10 since also limiting total tokens, in case some pages (for summarization) are small
            top_k_docs = max_top_k_docs_public if top_k_docs is None else top_k_docs

        if memory_restriction_level == 2:
            if not base_model and not inference_server and not model_lock:
                base_model = 'h2oai/h2ogpt-oasst1-512-12b'
                # don't set load_8bit if passed base_model, doesn't always work so can't just override
                load_8bit = True
                load_4bit = False  # FIXME - consider using 4-bit instead of 8-bit
        elif not inference_server:
            top_k_docs = max_top_k_docs_public if top_k_docs is None else top_k_docs
    if memory_restriction_level >= 2:
        load_8bit = True
        load_4bit = False  # FIXME - consider using 4-bit instead of 8-bit
        if hf_embedding_model is None:
            hf_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        top_k_docs = 3 if top_k_docs is None else top_k_docs
    if top_k_docs is None:
        top_k_docs = max_top_k_docs_default
    if max_input_tokens is None:
        max_input_tokens = -1
    if max_total_input_tokens is None:
        max_total_input_tokens = -1
    if is_public:
        if not max_time:
            max_time = 60 * 2
        if not max_max_time:
            max_max_time = max_time
        if not max_new_tokens:
            max_new_tokens = 256
        if not max_max_new_tokens:
            max_max_new_tokens = 512
    else:
        if not max_max_time:
            max_max_time = 60 * 20
        if not max_max_new_tokens:
            max_max_new_tokens = 1024
    if is_hf:
        # must override share if in spaces
        share = False
        if not max_time:
            max_time = 60 * 1
        if not max_max_time:
            max_max_time = max_time
        # HF accounted for later in get_max_max_new_tokens()
    save_dir = os.getenv('SAVE_DIR', save_dir)
    save_dir = makedirs(save_dir, exist_ok=True, tmp_ok=True, use_base=True)
    score_model = os.getenv('SCORE_MODEL', score_model)
    if str(score_model) == 'None':
        score_model = ''
    concurrency_count = int(os.getenv('CONCURRENCY_COUNT', concurrency_count))
    api_open = bool(int(os.getenv('API_OPEN', str(int(api_open)))))
    allow_api = bool(int(os.getenv('ALLOW_API', str(int(allow_api)))))

    n_gpus1 = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_gpus1, gpu_ids = cuda_vis_check(n_gpus1)
    if n_gpus is None:
        n_gpus = n_gpus1

    if load_half is None and t5_type(base_model):
        load_half = False
        print("load_half=%s auto-set for %s to avoid bad generation" % (load_half, base_model), flush=True)

    if n_gpus == 0 or get_device(n_gpus=n_gpus) == "mps":
        # No CUDA GPUs usable

        if get_device(n_gpus=n_gpus) != "mps":
            print("No GPUs detected", flush=True)

        enable_captions = False
        gpu_id = None
        load_8bit = False
        load_4bit = False
        low_bit_mode = 1
        if load_half is None:
            # wouldn't work if specified True, but respect
            load_half = False
        load_gptq = ''
        load_awq = ''
        load_exllama = False
        use_gpu_id = False
        if get_device(n_gpus=n_gpus) == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = False
            torch.set_default_dtype(torch.float32)
        if is_public and not inference_server and not model_lock:
            # 12B uses ~94GB
            # 6.9B uses ~47GB
            base_model = 'h2oai/h2ogpt-oig-oasst1-512-6_9b' if not base_model else base_model
        if hf_embedding_model is None:
            # if no GPUs, use simpler embedding model to avoid cost in time
            hf_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        if score_model == 'auto':
            score_model = ''
    else:
        if load_half is None:
            load_half = True
        # CUDA GPUs visible
        if score_model == 'auto':
            if n_gpus >= 2:
                # will by default place scoring model on last GPU
                score_model = 'OpenAssistant/reward-model-deberta-v3-large-v2'
            else:
                score_model = ''
        if hf_embedding_model is None:
            # if still None, then set default
            hf_embedding_model = 'hkunlp/instructor-large'

    # get defaults
    if base_model:
        model_lower = base_model.lower()
    elif model_lock:
        # have 0th model be thought of as normal model
        assert len(model_lock) > 0 and model_lock[0]['base_model'], "model_lock: %s" % model_lock
        model_lower = model_lock[0]['base_model'].lower()
    else:
        model_lower = ''
    if not gradio:
        # force, else not single response like want to look at
        stream_output = False
        # else prompt removal can mess up output
        chat = False
    # hard-coded defaults
    first_para = False
    text_limit = None

    if compile_model is None:
        # too avoid noisy CLI
        compile_model = not cli

    if offload_folder:
        offload_folder = makedirs(offload_folder, exist_ok=True, tmp_ok=True, use_base=True)

    # defaults
    caption_loader = None
    doctr_loader = None
    pix2struct_loader = None

    image_loaders_options0, image_loaders_options, \
        pdf_loaders_options0, pdf_loaders_options, \
        url_loaders_options0, url_loaders_options = lg_to_gr(**locals())
    jq_schema0 = jq_schema
    # transcribe
    image_loaders = image_loaders_options0
    pdf_loaders = pdf_loaders_options0
    url_loaders = url_loaders_options0

    placeholder_instruction, placeholder_input, \
        stream_output, show_examples, \
        prompt_type, prompt_dict, \
        temperature, top_p, top_k, penalty_alpha, num_beams, \
        max_new_tokens, min_new_tokens, early_stopping, max_time, \
        repetition_penalty, num_return_sequences, \
        do_sample, \
        src_lang, tgt_lang, \
        examples, \
        task_info = \
        get_generate_params(model_lower,
                            chat,
                            stream_output, show_examples,
                            prompt_type, prompt_dict,
                            system_prompt,
                            pre_prompt_query, prompt_query,
                            pre_prompt_summary, prompt_summary,
                            temperature, top_p, top_k, penalty_alpha, num_beams,
                            max_new_tokens, min_new_tokens, early_stopping, max_time,
                            repetition_penalty, num_return_sequences,
                            do_sample,
                            top_k_docs,
                            chunk,
                            chunk_size,
                            image_loaders,
                            pdf_loaders,
                            url_loaders,
                            jq_schema,
                            docs_ordering_type,
                            min_max_new_tokens,
                            max_input_tokens,
                            max_total_input_tokens,
                            docs_token_handling,
                            docs_joiner,
                            hyde_level,
                            hyde_template,
                            doc_json_mode,
                            verbose,
                            )

    git_hash = get_githash() if is_public or os.getenv('GET_GITHASH') else "GET_GITHASH"
    locals_dict = locals()
    locals_print = '\n'.join(['%s: %s' % (k, v) for k, v in locals_dict.items()])
    if verbose:
        print(f"Generating model with params:\n{locals_print}", flush=True)
        print("Command: %s\nHash: %s" % (str(' '.join(sys.argv)), git_hash), flush=True)

    if langchain_mode != LangChainMode.DISABLED.value:
        # SECOND PLACE where LangChain referenced, but all imports are kept local so not required
        from gpt_langchain import prep_langchain, get_some_dbs_from_hf, get_persist_directory
        if is_hf:
            get_some_dbs_from_hf()
        dbs = {}
        for langchain_mode1 in langchain_modes:
            langchain_type = langchain_mode_types.get(langchain_mode1, LangChainTypes.EITHER.value)
            if langchain_type == LangChainTypes.PERSONAL.value:
                # shouldn't prepare per-user databases here
                continue
            persist_directory1, langchain_type = get_persist_directory(langchain_mode1, langchain_type=langchain_type)
            langchain_mode_types[langchain_mode1] = langchain_type
            if langchain_type == LangChainTypes.PERSONAL.value:
                # shouldn't prepare per-user databases here
                continue
            try:
                db = prep_langchain(persist_directory1,
                                    load_db_if_exists,
                                    db_type, use_openai_embedding,
                                    langchain_mode1, langchain_mode_paths, langchain_mode_types,
                                    hf_embedding_model,
                                    migrate_embedding_model,
                                    auto_migrate_db,
                                    kwargs_make_db=locals(),
                                    verbose=verbose)
            finally:
                # in case updated embeddings or created new embeddings
                clear_torch_cache()
            dbs[langchain_mode1] = db
        # remove None db's so can just rely upon k in dbs for if hav db
        dbs = {k: v for k, v in dbs.items() if v is not None}
    else:
        dbs = {}
        # import control
        if os.environ.get("TEST_LANGCHAIN_IMPORT"):
            assert 'gpt_langchain' not in sys.modules, "Dev bug, import of langchain when should not have"
            assert 'langchain' not in sys.modules, "Dev bug, import of langchain when should not have"

    if attention_sinks:
        if use_cache is False:
            raise ValueError("attention sinks requires use_cache=True")
        else:
            use_cache = True
    # never truncate if using attention sinks
    truncation_generation = truncation_generation and not attention_sinks

    other_model_state_defaults = dict(load_8bit=load_8bit, load_4bit=load_4bit, low_bit_mode=low_bit_mode,
                                      load_half=load_half,
                                      load_gptq=load_gptq, load_awq=load_awq, load_exllama=load_exllama,
                                      use_safetensors=use_safetensors,
                                      revision=revision, use_gpu_id=use_gpu_id, gpu_id=gpu_id,
                                      compile_model=compile_model,
                                      use_cache=use_cache,
                                      llamacpp_dict=llamacpp_dict, model_path_llama=model_path_llama,
                                      model_name_gptj=model_name_gptj,
                                      model_name_gpt4all_llama=model_name_gpt4all_llama,
                                      model_name_exllama_if_no_config=model_name_exllama_if_no_config,
                                      rope_scaling=rope_scaling,
                                      max_seq_len=max_seq_len,
                                      exllama_dict=exllama_dict,
                                      gptq_dict=gptq_dict,
                                      attention_sinks=attention_sinks,
                                      sink_dict=sink_dict,
                                      truncation_generation=truncation_generation,
                                      hf_model_dict=hf_model_dict,
                                      )
    model_state_none = dict(model=None, tokenizer=None, device=None,
                            base_model=None, tokenizer_base_model=None, lora_weights=None,
                            inference_server=None, prompt_type=None, prompt_dict=None,
                            visible_models=None, h2ogpt_key=None,
                            )
    model_state_none.update(other_model_state_defaults)
    my_db_state0 = {LangChainMode.MY_DATA.value: [None, None, None]}
    selection_docs_state0 = dict(langchain_modes=langchain_modes,
                                 langchain_mode_paths=langchain_mode_paths,
                                 langchain_mode_types=langchain_mode_types)
    selection_docs_state = copy.deepcopy(selection_docs_state0)

    if cli or not gradio:
        # initial state for query prompt
        model_name = base_model
        pre_prompt_query, prompt_query, pre_prompt_summary, prompt_summary = \
            get_langchain_prompts(pre_prompt_query, prompt_query,
                                  pre_prompt_summary, prompt_summary,
                                  model_name, inference_server,
                                  model_path_llama,
                                  doc_json_mode)

    if cli:
        from cli import run_cli
        return run_cli(**get_kwargs(run_cli, exclude_names=['model_state0'], **locals()))
    elif not gradio:
        from eval import run_eval
        return run_eval(**get_kwargs(run_eval, exclude_names=['model_state0'], **locals()))
    elif gradio or prepare_offline_level > 0:
        # imported here so don't require gradio to run generate
        from gradio_runner import go_gradio

        # get default model
        model_states = []
        model_list = [dict(base_model=base_model, tokenizer_base_model=tokenizer_base_model, lora_weights=lora_weights,
                           inference_server=inference_server, prompt_type=prompt_type, prompt_dict=prompt_dict,
                           visible_models=None, h2ogpt_key=None)]
        model_list[0].update(other_model_state_defaults)
        # FIXME: hyper per model, not about model loading
        # for k in gen_hyper:
        #     model_list[k] = locals()[k]

        model_list0 = copy.deepcopy(model_list)  # just strings, safe to deepcopy
        model_state0 = model_state_none.copy()
        assert len(model_state_none) == len(model_state0)
        if model_lock:
            model_list = model_lock
        # do reverse, so first is default base_model etc., so some logic works in go_gradio() more easily
        for model_dict in reversed(model_list):
            # handle defaults user didn't have to pass
            # special defaults, ignore defaults for these if not specifically set, replace with ''
            model_dict['base_model'] = model_dict.get('base_model', '')
            model_dict['tokenizer_base_model'] = model_dict.get('tokenizer_base_model', '')
            model_dict['lora_weights'] = model_dict.get('lora_weights', '')
            model_dict['inference_server'] = model_dict.get('inference_server', '')
            if prepare_offline_level >= 2:
                if 'openai' not in model_dict['inference_server'] and 'replicate' not in model_dict['inference_server']:
                    # assume want locally, but OpenAI and replicate are never local for model part
                    model_dict['inference_server'] = ''
            prompt_type_infer = not model_dict.get('prompt_type')
            model_dict['prompt_type'] = model_dict.get('prompt_type',
                                                       model_list0[0]['prompt_type'])  # don't use mutated value
            # rest of generic defaults
            for k in model_list0[0]:
                if k not in model_dict:
                    model_dict[k] = model_list0[0][k]

            model_dict['llamacpp_dict'] = model_dict.get('llamacpp_dict', {})
            model_dict['base_model'], model_dict['model_path_llama'], \
                model_dict['load_gptq'], \
                model_dict['load_awq'], \
                model_dict['llamacpp_dict']['n_gqa'] = \
                switch_a_roo_llama(model_dict['base_model'],
                                   model_dict['model_path_llama'],
                                   model_dict['load_gptq'],
                                   model_dict['load_awq'],
                                   model_dict.get('llamacpp_dict', {}).get('n_gqa', 0))

            # begin prompt adjustments
            # get query prompt for (say) last base model if using model lock
            pre_prompt_query1, prompt_query1, pre_prompt_summary1, prompt_summary1 = (
                get_langchain_prompts(pre_prompt_query, prompt_query,
                                      pre_prompt_summary, prompt_summary,
                                      model_dict['base_model'],
                                      model_dict['inference_server'],
                                      model_dict['model_path_llama'],
                                      doc_json_mode))
            # if mixed setup, choose non-empty so best models best
            # FIXME: Make per model dict passed through to evaluate
            pre_prompt_query = pre_prompt_query or pre_prompt_query1
            prompt_query = prompt_query or prompt_query1
            pre_prompt_summary = pre_prompt_summary or pre_prompt_summary1
            prompt_summary = prompt_summary or prompt_summary1

            # try to infer, ignore empty initial state leading to get_generate_params -> 'plain'
            if prompt_type_infer:
                model_lower1 = model_dict['base_model'].lower()
                model_path_llama1 = model_dict.get('model_path_llama', '').lower()
                model_dict['llamacpp_dict'] = model_dict.get('llamacpp_dict', {}) or {}
                llamacpp_dict1 = model_dict.get('llamacpp_dict', {}) or {}
                load_gptq1 = model_dict.get('load_gptq', '')
                load_awq1 = model_dict.get('load_awq', '')
                model_lower10 = model_lower1
                get_prompt_kwargs = dict(chat=False, context='', reduced=False,
                                         making_context=False,
                                         return_dict=True,
                                         system_prompt=system_prompt)
                if model_lower1 in inv_prompt_type_to_model_lower:
                    model_dict['prompt_type'] = inv_prompt_type_to_model_lower[model_lower1]
                    model_dict['prompt_dict'], error0 = get_prompt(model_dict['prompt_type'], '',
                                                                   **get_prompt_kwargs)
                elif model_lower10 in inv_prompt_type_to_model_lower:
                    model_dict['prompt_type'] = inv_prompt_type_to_model_lower[model_lower10]
                    model_dict['prompt_dict'], error0 = get_prompt(model_dict['prompt_type'], '',
                                                                   **get_prompt_kwargs)
                else:
                    model_dict['prompt_dict'] = prompt_dict
            else:
                model_dict['prompt_dict'] = prompt_dict
            model_dict['prompt_dict'] = model_dict.get('prompt_dict', model_dict['prompt_dict'])
            # end prompt adjustments
            all_kwargs = locals().copy()
            all_kwargs.update(model_dict)
            if model_dict['base_model'] and not login_mode_if_model0:
                model0, tokenizer0, device = get_model_retry(reward_type=False,
                                                             **get_kwargs(get_model, exclude_names=['reward_type'],
                                                                          **all_kwargs))
                # update model state
                if hasattr(tokenizer0, 'model_max_length'):
                    model_dict['max_seq_len'] = tokenizer0.model_max_length
            else:
                # if empty model, then don't load anything, just get gradio up
                model0, tokenizer0, device = None, None, None
            if model0 is None:
                if fail_if_cannot_connect:
                    raise RuntimeError("Could not connect, see logs")
                # skip
                if isinstance(model_lock, list):
                    model_lock.remove(model_dict)
                continue
            model_state_trial = dict(model=model0, tokenizer=tokenizer0, device=device)
            model_state_trial.update(model_dict)
            diff_keys = set(list(model_state_none.keys())).symmetric_difference(model_state_trial.keys())
            assert len(model_state_none) == len(model_state_trial), diff_keys
            print("Model %s" % model_dict, flush=True)
            if model_lock:
                # last in iteration will be first
                model_states.insert(0, model_state_trial)
                # fill model_state0 so go_gradio() easier, manage model_states separately
                model_state0 = model_state_trial.copy()
            else:
                model_state0 = model_state_trial.copy()
            assert len(model_state_none) == len(model_state0)

        visible_models = str_to_list(visible_models, allow_none=True)  # None means first model
        all_possible_visible_models = [
            x.get('base_model', xi) if x.get('base_model', '') != 'llama' or
                                       not x.get('model_path_llama', '') else x.get(
                'model_path_llama', '') for xi, x in enumerate(model_states)]
        visible_models_state0 = [x for xi, x in enumerate(all_possible_visible_models) if
                                 visible_models is None or
                                 x in visible_models or
                                 xi in visible_models]

        # update to be consistent with what is passed from CLI and model chose
        # do after go over all models if multi-model, so don't contaminate
        # This is just so UI shows reasonable correct value, not 2048 dummy value
        if len(model_states) >= 1:
            max_seq_len = model_states[0]['tokenizer'].model_max_length
        elif model_state0 is not None and \
                'tokenizer' in model_state0 and \
                hasattr(model_state0['tokenizer'], 'model_max_length'):
            max_seq_len = model_state0['tokenizer'].model_max_length

        # get score model
        all_kwargs = locals().copy()
        smodel, stokenizer, sdevice = get_score_model(reward_type=True,
                                                      **get_kwargs(get_score_model, exclude_names=['reward_type'],
                                                                   **all_kwargs))
        score_model_state0 = dict(model=smodel, tokenizer=stokenizer, device=sdevice,
                                  base_model=score_model, tokenizer_base_model='', lora_weights='',
                                  inference_server='', prompt_type='', prompt_dict='',
                                  visible_models=None, h2ogpt_key=None)

        if enable_captions:
            if pre_load_caption_model:
                from image_captions import H2OImageCaptionLoader
                caption_loader = H2OImageCaptionLoader(caption_gpu=caption_gpu, gpu_id=caption_gpu_id).load_model()
            else:
                caption_loader = 'gpu' if n_gpus > 0 and caption_gpu else 'cpu'
        else:
            caption_loader = False

        if pre_load_embedding_model and \
                langchain_mode != LangChainMode.DISABLED.value and \
                not use_openai_embedding:
            from src.gpt_langchain import get_embedding
            hf_embedding_model = dict(name=hf_embedding_model,
                                      model=get_embedding(use_openai_embedding, hf_embedding_model=hf_embedding_model,
                                                          preload=True))

        if enable_doctr or enable_pdf_ocr in [True, 'auto', 'on']:
            if pre_load_caption_model:
                from image_doctr import H2OOCRLoader
                doctr_loader = H2OOCRLoader(layout_aware=True, gpu_id=doctr_gpu_id)
            else:
                doctr_loader = 'gpu' if n_gpus > 0 and caption_gpu else 'cpu'
        else:
            doctr_loader = False

        # assume gradio needs everything
        go_gradio(**locals())


def get_config(base_model,
               use_auth_token=False,
               trust_remote_code=True,
               offload_folder=None,
               revision=None,
               rope_scaling=None,
               triton_attn=False,
               long_sequence=True,
               return_model=False,
               raise_exception=False,
               max_seq_len=None,
               verbose=False,
               ):
    from accelerate import init_empty_weights
    with init_empty_weights():
        from transformers import AutoConfig
        try:
            if rope_scaling:
                rope_kwargs = dict(rope_scaling=rope_scaling)
            else:
                rope_kwargs = {}
            config = AutoConfig.from_pretrained(base_model, token=use_auth_token,
                                                trust_remote_code=trust_remote_code,
                                                offload_folder=offload_folder,
                                                revision=revision,
                                                **rope_kwargs)
        except OSError as e:
            if raise_exception:
                raise
            if 'not a local folder and is not a valid model identifier listed on' in str(
                    e) or '404 Client Error' in str(e) or "couldn't connect" in str(e):
                # e.g. llama, gpjt, etc.
                # e.g. HF TGI but not model on HF or private etc.
                if max_seq_len is None and base_model.lower() in non_hf_types:
                    print("Could not determine --max_seq_len, setting to 2048.  Pass if not correct", flush=True)
                    max_seq_len = 2048
                # HF TGI server only should really require prompt_type, not HF model state
                return None, None, max_seq_len
            else:
                raise
        if triton_attn and 'mpt-' in base_model.lower():
            config.attn_config['attn_impl'] = 'triton'
        if long_sequence:
            if 'mpt-7b-storywriter' in base_model.lower():
                config.update({"max_seq_len": 83968})
            if 'mosaicml/mpt-7b-chat' in base_model.lower():
                config.update({"max_seq_len": 4096})
            if 'mpt-30b' in base_model.lower():
                config.update({"max_seq_len": 2 * 8192})
        if return_model and \
                issubclass(config.__class__, tuple(AutoModel._model_mapping.keys())):
            model = AutoModel.from_config(
                config,
                trust_remote_code=trust_remote_code,
            )
        else:
            # can't infer
            model = None
    if 'falcon' in base_model.lower():
        config.use_cache = False

    # allow override
    if max_seq_len is not None:
        print("Overriding max_seq_len -> %d" % max_seq_len, flush=True)
    else:
        if hasattr(config, 'max_seq_len'):
            max_seq_len = int(config.max_seq_len)
        # Note https://huggingface.co/lmsys/vicuna-13b-v1.5-16k/blob/main/config.json has below, but here just want base size before rope
        # elif hasattr(config, 'max_sequence_length'):
        #    max_seq_len = int(config.max_sequence_length)
        elif hasattr(config, 'max_position_embeddings') and isinstance(config.max_position_embeddings, int):
            # help automatically limit inputs to generate
            max_seq_len = config.max_position_embeddings
            if verbose:
                print("Used max_position_embeddings=%s as base model (pre-rope) max_seq_len."
                      "  If not desired, pass --max_seq_len and set to some integer value." % config.max_position_embeddings,
                      flush=True)
        elif hasattr(config, 'n_ctx'):
            # e.g. gpt2
            max_seq_len = int(config.n_ctx)
        else:
            print("Could not determine --max_seq_len, setting to 2048.  Pass if not correct", flush=True)
            max_seq_len = 2048
            # FIXME:
            # raise RuntimeError("Could not determine max_seq_len,"
            #                   " please pass --max_seq_len and set to some value, e.g. 2048.")

        # listen to model if sets this and user passed nothing
        if not rope_scaling and hasattr(config, 'rope_scaling'):
            rope_scaling = config.rope_scaling

        if rope_scaling:
            if rope_scaling.get('factor'):
                # HF transformers
                max_seq_len *= rope_scaling.get('factor')
            elif rope_scaling.get('alpha_value'):
                # exllama
                # Note: exllama's own tokenizer has this set correctly in loaders.py, this config will be unused
                max_seq_len *= rope_scaling.get('alpha_value')
            max_seq_len = int(max_seq_len)
            print("Automatically setting max_seq_len=%d for RoPE scaling for %s" % (max_seq_len, base_model),
                  flush=True)

    return config, model, max_seq_len


def get_non_lora_model(base_model, model_loader, load_half,
                       load_gptq,
                       use_autogptq,
                       load_awq,
                       load_exllama,
                       use_safetensors,
                       revision,
                       model_kwargs, reward_type,
                       config, model,
                       gpu_id=0,
                       ):
    """
    Ensure model gets on correct device
    """

    if model is not None:
        # NOTE: Can specify max_memory={0: max_mem, 1: max_mem}, to shard model
        # NOTE: Some models require avoiding sharding some layers,
        # then would pass no_split_module_classes and give list of those layers.
        from accelerate import infer_auto_device_map
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

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_gpus, gpu_ids = cuda_vis_check(n_gpus)

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
        model_kwargs['load_in_4bit'] = False
    print('device_map: %s' % device_map, flush=True)

    load_in_8bit = model_kwargs.get('load_in_8bit', False)
    load_in_4bit = model_kwargs.get('load_in_4bit', False)
    model_kwargs['device_map'] = device_map
    model_kwargs['use_safetensors'] = use_safetensors
    model_kwargs['revision'] = revision
    pop_unused_model_kwargs(model_kwargs)

    if load_exllama:
        model = model_loader
    elif load_gptq and use_autogptq:
        model_kwargs.pop('torch_dtype', None)
        loader_kwargs = dict(model_name_or_path=base_model,
                             model_basename=load_gptq,
                             **model_kwargs)
        model = model_loader(**loader_kwargs)
    elif load_awq:
        allowed_dict = dict(max_new_tokens=None,
                            trust_remote_code=True, fuse_layers=True,
                            batch_size=1, safetensors=False,
                            max_memory=None, offload_folder=None)
        for k in model_kwargs.copy():
            if k not in allowed_dict:
                model_kwargs.pop(k)
        if load_awq.endswith('.pt'):
            args = tuple([base_model, load_awq])
        else:
            args = tuple([base_model])
        model = model_loader(
            *args,
            safetensors=use_safetensors,
            **model_kwargs,
        )
    elif load_in_8bit or load_in_4bit or not load_half:
        model = model_loader(
            base_model,
            config=config,
            **model_kwargs,
        )
    else:
        model = model_loader(
            base_model,
            config=config,
            **model_kwargs,
        )
        if not getattr(model, "is_quantized", False):
            model = model.half()
    return model


def get_client_from_inference_server(inference_server, base_model=None, raise_connection_exception=False):
    inference_server, headers = get_hf_server(inference_server)
    # preload client since slow for gradio case especially
    from gradio_utils.grclient import GradioClient
    gr_client = None
    hf_client = None
    if headers is None:
        try:
            print("GR Client Begin: %s %s" % (inference_server, base_model), flush=True)
            # first do sanity check if alive, else gradio client takes too long by default
            requests.get(inference_server, timeout=int(os.getenv('REQUEST_TIMEOUT', '30')))
            gr_client = GradioClient(inference_server).setup()
            print("GR Client End: %s" % inference_server, flush=True)
        except (OSError, ValueError) as e:
            # Occurs when wrong endpoint and should have been HF client, so don't hard raise, just move to HF
            gr_client = None
            print("GR Client Failed %s %s: %s" % (inference_server, base_model, str(e)), flush=True)
        except (ConnectTimeoutError, ConnectTimeout, MaxRetryError, ConnectionError, ConnectionError2,
                JSONDecodeError, ReadTimeout2, KeyError) as e:
            t, v, tb = sys.exc_info()
            ex = ''.join(traceback.format_exception(t, v, tb))
            print("GR Client Failed %s %s: %s" % (inference_server, base_model, str(ex)), flush=True)
            if raise_connection_exception:
                raise

    if gr_client is None:
        res = None
        from text_generation import Client as HFClient
        print("HF Client Begin: %s %s" % (inference_server, base_model))
        try:
            hf_client = HFClient(inference_server, headers=headers, timeout=int(os.getenv('REQUEST_TIMEOUT', '30')))
            # quick check valid TGI endpoint
            res = hf_client.generate('What?', max_new_tokens=1)
            hf_client = HFClient(inference_server, headers=headers, timeout=300)
        except (ConnectTimeoutError, ConnectTimeout, MaxRetryError, ConnectionError, ConnectionError2,
                JSONDecodeError, ReadTimeout2, KeyError) as e:
            hf_client = None
            t, v, tb = sys.exc_info()
            ex = ''.join(traceback.format_exception(t, v, tb))
            print("HF Client Failed %s %s: %s" % (inference_server, base_model, str(ex)))
            if raise_connection_exception:
                raise
        print("HF Client End: %s %s : %s" % (inference_server, base_model, res))
    return inference_server, gr_client, hf_client


def get_model_retry(**kwargs):
    model1, tokenizer1, device1 = None, None, None
    trials = 4
    for trial in range(trials):
        try:
            model1, tokenizer1, device1 = get_model(**kwargs)
            break
        except Exception as e:
            stre = str(e)
            if 'Exllama kernel does not support' in stre:
                # help user a bit
                kwargs['gptq_dict'].update(
                    {'inject_fused_attention': False, 'disable_exllama': True})
            if 'Could not find model' in stre or \
                    'safetensors' in stre or \
                    'not appear to have a file named pytorch_model.bin' in stre:
                kwargs['use_safetensors'] = True
            clear_torch_cache()
            if trial >= trials - 1:
                raise
    return model1, tokenizer1, device1


def get_model(
        load_8bit: bool = False,
        load_4bit: bool = False,
        low_bit_mode: int = 1,
        load_half: bool = True,
        load_gptq: str = '',
        use_autogptq: bool = False,
        load_awq: str = '',
        load_exllama: bool = False,
        use_safetensors: bool = False,
        revision: str = None,
        use_gpu_id: bool = True,
        base_model: str = '',
        inference_server: str = "",
        tokenizer_base_model: str = '',
        lora_weights: str = "",
        gpu_id: int = 0,
        n_jobs=None,
        n_gpus=None,

        reward_type: bool = None,
        local_files_only: bool = False,
        resume_download: bool = True,
        use_auth_token: Union[str, bool] = False,
        trust_remote_code: bool = True,
        offload_folder: str = None,
        rope_scaling: dict = None,
        max_seq_len: int = None,
        compile_model: bool = True,
        llamacpp_dict=None,
        exllama_dict=None,
        gptq_dict=None,
        attention_sinks=None,
        sink_dict=None,
        truncation_generation=None,
        hf_model_dict={},

        verbose: bool = False,
):
    """

    :param load_8bit: load model in 8-bit, not supported by all models
    :param load_4bit: load model in 4-bit, not supported by all models
    :param low_bit_mode: See gen.py
    :param load_half: load model in 16-bit
    :param load_gptq: GPTQ model_basename
    :param use_autogptq: Use AutoGPTQ (True) or HF transformers (False)
    :param load_awq: AWQ model_basename
    :param load_exllama: whether to use exllama
    :param use_safetensors: use safetensors file
    :param revision:
    :param use_gpu_id: Use torch infer of optimal placement of layers on devices (for non-lora case)
           For non-LORA case, False will spread shards across multiple GPUs, but this can lead to cuda:x cuda:y mismatches
           So it is not the default
    :param base_model: name/path of base model
    :param inference_server: whether base_model is hosted locally ('') or via http (url)
    :param tokenizer_base_model: name/path of tokenizer
    :param lora_weights: name/path
    :param gpu_id: which GPU (0..n_gpus-1) or allow all GPUs if relevant (-1)
    :param n_jobs: number of cores to use (e.g. for llama CPU model)
    :param n_gpus: number of GPUs (-1 for all)
    :param reward_type: reward type model for sequence classification
    :param local_files_only: use local files instead of from HF
    :param resume_download: resume downloads from HF
    :param use_auth_token: assumes user did on CLI `huggingface-cli login` to access private repo
    :param trust_remote_code: trust code needed by model
    :param offload_folder: offload folder
    :param rope_scaling: scaling for rope-based models, e.g. "{'type':'dynamic', 'factor':4}"
    :param max_seq_len: override for maximum sequence length for model
    :param max_seq_len: if set, use as max_seq_len for model
    :param compile_model: whether to compile torch model
    :param llamacpp_dict: dict of llama.cpp and GPT4All model options
    :param exllama_dict: dict of exllama options
    :param gptq_dict: dict of AutoGPTQ options
    :param attention_sinks: whether to use attention_sinks package
    :param sink_dict: dict of attention sinks options
    :param truncation_generation: whether to truncate generation in torch case to max_seq_len
    :param hf_model_dict
    :param verbose:
    :return:
    """
    print("Starting get_model: %s %s" % (base_model, inference_server), flush=True)

    triton_attn = False
    long_sequence = True
    config_kwargs = dict(use_auth_token=use_auth_token,
                         trust_remote_code=trust_remote_code,
                         offload_folder=offload_folder,
                         rope_scaling=rope_scaling,
                         triton_attn=triton_attn,
                         long_sequence=long_sequence,
                         revision=revision,
                         max_seq_len=max_seq_len,
                         verbose=verbose)
    if base_model == 'llama':
        # in case max_seq_len = None, try to auto-set
        config = None
    else:
        config, _, max_seq_len = get_config(base_model, **config_kwargs, raise_exception=False)

    if base_model in non_hf_types:
        assert config is None, "Expected config None for %s" % base_model

    llama_type_from_config = 'llama' in str(config).lower()
    llama_type_from_name = "llama" in base_model.lower()
    llama_type = llama_type_from_config or llama_type_from_name
    if "xgen" in base_model.lower() or 'llama2' in base_model.lower() or 'llama-2' in base_model.lower():
        llama_type = False
    if llama_type:
        if verbose:
            print("Detected as llama type from"
                  " config (%s) or name (%s)" % (llama_type_from_config, llama_type_from_name), flush=True)

    model_name_exllama_if_no_config = '' if not llamacpp_dict else llamacpp_dict.get('model_name_exllama_if_no_config',
                                                                                     '')
    model_loader, tokenizer_loader, conditional_type = (
        get_loaders(model_name=base_model, reward_type=reward_type, llama_type=llama_type,
                    load_gptq=load_gptq,
                    use_autogptq=use_autogptq,
                    load_awq=load_awq, load_exllama=load_exllama,
                    config=config,
                    rope_scaling=rope_scaling, max_seq_len=max_seq_len,
                    model_name_exllama_if_no_config=model_name_exllama_if_no_config,
                    exllama_dict=exllama_dict, gptq_dict=gptq_dict,
                    attention_sinks=attention_sinks, sink_dict=sink_dict,
                    truncation_generation=truncation_generation,
                    hf_model_dict=hf_model_dict))

    tokenizer_kwargs = dict(local_files_only=local_files_only,
                            resume_download=resume_download,
                            token=use_auth_token,
                            trust_remote_code=trust_remote_code,
                            offload_folder=offload_folder,
                            revision=revision,
                            padding_side='left',
                            config=config,
                            )
    if not tokenizer_base_model:
        tokenizer_base_model = base_model

    if load_exllama:
        tokenizer = tokenizer_loader
    elif config is not None and tokenizer_loader is not None and not isinstance(tokenizer_loader, str):
        if load_exllama:
            tokenizer = tokenizer_loader
        else:
            tokenizer = tokenizer_loader.from_pretrained(tokenizer_base_model, **tokenizer_kwargs)
            # sets raw (no cushion) limit
            # If using RoPE with scaling, then for non-exllama models (e.g. HF models),
            #  then config -> tokenizer will set model_max_length correctly
            set_model_max_len(max_seq_len, tokenizer, verbose=False)
            # if using fake tokenizer, not really accurate when lots of numbers, give a bit of buffer, else get:
            # Generation Failed: Input validation error: `inputs` must have less than 2048 tokens. Given: 2233
            tokenizer.model_max_length = int(tokenizer.model_max_length - 50)
    else:
        tokenizer = None

    if isinstance(inference_server, str) and inference_server.startswith("http"):
        inference_server, gr_client, hf_client = get_client_from_inference_server(inference_server,
                                                                                  base_model=base_model)
        client = gr_client or hf_client
        # Don't return None, None for model, tokenizer so triggers
        if tokenizer is None:
            # FIXME: Could use only tokenizer from llama etc. but hard to detatch from model, just use fake for now
            if os.getenv("HARD_ASSERTS") and base_model not in non_hf_types:
                raise RuntimeError("Unexpected tokenizer=None")
            tokenizer = FakeTokenizer()
        return client, tokenizer, 'http'

    if base_model in openai_gpts and not inference_server:
        raise ValueError("Must select inference server when choosing OpenAI models")

    if isinstance(inference_server, str) and (
            inference_server.startswith('openai') or
            inference_server.startswith('vllm') or
            inference_server.startswith('replicate') or
            inference_server.startswith('sagemaker')
    ):
        if inference_server.startswith('openai'):
            assert os.getenv('OPENAI_API_KEY'), "Set environment for OPENAI_API_KEY"
            # Don't return None, None for model, tokenizer so triggers
            # include small token cushion
            if base_model in model_token_mapping:
                max_seq_len = model_token_mapping[base_model]
            else:
                raise ValueError("Invalid base_model=%s for inference_server=%s" % (base_model, inference_server))
        if inference_server.startswith('replicate'):
            assert len(inference_server.split(':')) >= 3, "Expected replicate:model string, got %s" % inference_server
            assert os.getenv('REPLICATE_API_TOKEN'), "Set environment for REPLICATE_API_TOKEN"
            assert max_seq_len is not None, "Please pass --max_seq_len=<max_seq_len> for replicate models."
            try:
                import replicate as replicate_python
            except ImportError:
                raise ImportError(
                    "Could not import replicate python package. "
                    "Please install it with `pip install replicate`."
                )
        if inference_server.startswith('sagemaker'):
            assert len(
                inference_server.split(
                    ':')) >= 3, "Expected sagemaker_chat:<endpoint name>:<region>, got %s" % inference_server
            assert os.getenv('AWS_ACCESS_KEY_ID'), "Set environment for AWS_ACCESS_KEY_ID"
            assert os.getenv('AWS_SECRET_ACCESS_KEY'), "Set environment for AWS_SECRET_ACCESS_KEY"
        # Don't return None, None for model, tokenizer so triggers
        # include small token cushion
        if inference_server.startswith('openai') or tokenizer is None:
            # don't use fake (tiktoken) tokenizer for vLLM//replicate if know actual model with actual tokenizer
            assert max_seq_len is not None, "Please pass --max_seq_len=<max_seq_len> for unknown or non-HF model %s" % base_model
            tokenizer = FakeTokenizer(model_max_length=max_seq_len - 50, is_openai=True)
        return inference_server, tokenizer, inference_server
    assert not inference_server, "Malformed inference_server=%s" % inference_server
    if base_model in non_hf_types:
        from gpt4all_llm import get_model_tokenizer_gpt4all
        model, tokenizer, device = get_model_tokenizer_gpt4all(base_model,
                                                               n_jobs=n_jobs,
                                                               gpu_id=gpu_id,
                                                               n_gpus=n_gpus,
                                                               max_seq_len=max_seq_len,
                                                               llamacpp_dict=llamacpp_dict)
        return model, tokenizer, device
    if load_exllama:
        return model_loader, tokenizer, 'cuda' if n_gpus != 0 else 'cpu'

    # get local torch-HF model
    return get_hf_model(load_8bit=load_8bit,
                        load_4bit=load_4bit,
                        low_bit_mode=low_bit_mode,
                        load_half=load_half,
                        load_gptq=load_gptq,
                        use_autogptq=use_autogptq,
                        load_awq=load_awq,
                        use_safetensors=use_safetensors,
                        revision=revision,
                        use_gpu_id=use_gpu_id,
                        base_model=base_model,
                        tokenizer_base_model=tokenizer_base_model,
                        lora_weights=lora_weights,
                        gpu_id=gpu_id,
                        n_gpus=n_gpus,

                        reward_type=reward_type,
                        local_files_only=local_files_only,
                        resume_download=resume_download,
                        use_auth_token=use_auth_token,
                        trust_remote_code=trust_remote_code,
                        offload_folder=offload_folder,
                        rope_scaling=rope_scaling,
                        compile_model=compile_model,

                        llama_type=llama_type,
                        config_kwargs=config_kwargs,
                        tokenizer_kwargs=tokenizer_kwargs,
                        gptq_dict=gptq_dict,
                        attention_sinks=attention_sinks,
                        sink_dict=sink_dict,
                        truncation_generation=truncation_generation,
                        hf_model_dict=hf_model_dict,

                        verbose=verbose)


def get_hf_model(load_8bit: bool = False,
                 load_4bit: bool = False,
                 low_bit_mode: int = 1,
                 load_half: bool = True,
                 load_gptq: str = '',
                 use_autogptq: bool = False,
                 load_awq: str = '',
                 use_safetensors: bool = False,
                 revision: str = None,
                 use_gpu_id: bool = True,
                 base_model: str = '',
                 tokenizer_base_model: str = '',
                 lora_weights: str = "",
                 gpu_id: int = 0,
                 n_gpus: int = None,

                 reward_type: bool = None,
                 local_files_only: bool = False,
                 resume_download: bool = True,
                 use_auth_token: Union[str, bool] = False,
                 trust_remote_code: bool = True,
                 offload_folder: str = None,
                 rope_scaling: dict = None,
                 compile_model: bool = True,

                 llama_type: bool = False,
                 config_kwargs=None,
                 tokenizer_kwargs=None,
                 gptq_dict=None,
                 attention_sinks=None,
                 sink_dict=None,
                 truncation_generation=None,
                 hf_model_dict=None,

                 verbose: bool = False,
                 ):
    assert config_kwargs is not None
    assert tokenizer_kwargs is not None

    load_exllama = False  # Never should be in HF code for exllama
    exllama_dict = {}

    if lora_weights is not None and lora_weights.strip():
        if verbose:
            print("Get %s lora weights" % lora_weights, flush=True)
    device = get_device(n_gpus=n_gpus)

    if 'gpt2' in base_model.lower():
        # RuntimeError: where expected condition to be a boolean tensor, but got a tensor with dtype Half
        load_8bit = False
        load_4bit = False

    assert base_model.strip(), (
        "Please choose a base model with --base_model (CLI) or load one from Models Tab (gradio)"
    )

    model_loader, tokenizer_loader, conditional_type = (
        get_loaders(model_name=base_model, reward_type=reward_type, llama_type=llama_type,
                    load_gptq=load_gptq,
                    use_autogptq=use_autogptq,
                    load_awq=load_awq, load_exllama=load_exllama,
                    exllama_dict=exllama_dict, gptq_dict=gptq_dict,
                    attention_sinks=attention_sinks, sink_dict=sink_dict,
                    truncation_generation=truncation_generation,
                    hf_model_dict=hf_model_dict))

    config, _, max_seq_len = get_config(base_model, return_model=False, raise_exception=True, **config_kwargs)

    if tokenizer_loader is not None and not isinstance(tokenizer_loader, str):
        if load_exllama:
            tokenizer = tokenizer_loader
        else:
            tokenizer = tokenizer_loader.from_pretrained(tokenizer_base_model,
                                                         **tokenizer_kwargs)
    else:
        tokenizer = tokenizer_loader

    if isinstance(tokenizer, str):
        # already a pipeline, tokenizer_loader is string for task
        model = model_loader(tokenizer,
                             model=base_model,
                             device=0 if device == "cuda" else -1,
                             torch_dtype=torch.float16 if device == 'cuda' else torch.float32)
    else:
        assert device in ["cuda", "cpu", "mps"], "Unsupported device %s" % device
        model_kwargs = dict(local_files_only=local_files_only,
                            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                            resume_download=resume_download,
                            token=use_auth_token,
                            trust_remote_code=trust_remote_code,
                            offload_folder=offload_folder,
                            revision=revision,
                            # rope_scaling=rope_scaling,  # only put into config
                            )
        if 'mbart-' not in base_model.lower() and 'mpt-' not in base_model.lower():
            if use_gpu_id and gpu_id is not None and gpu_id >= 0 and device == 'cuda':
                device_map = {"": gpu_id}
            else:
                device_map = "auto"
            model_kwargs.update(dict(load_in_8bit=load_8bit,
                                     load_in_4bit=load_4bit,
                                     device_map=device_map,
                                     ))
        if 'mpt-' in base_model.lower() and gpu_id is not None and gpu_id >= 0:
            # MPT doesn't support spreading over GPUs
            model_kwargs.update(dict(device_map={"": gpu_id} if device == 'cuda' else "cpu"))

        if 'OpenAssistant/reward-model'.lower() in base_model.lower():
            # FIXME: could put on other GPUs
            model_kwargs['device_map'] = {"": 0} if device == 'cuda' else {"": 'cpu'}
            model_kwargs.pop('torch_dtype', None)
        pop_unused_model_kwargs(model_kwargs)

        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        n_gpus, gpu_ids = cuda_vis_check(n_gpus)
        if n_gpus != 0 and not (load_gptq and not use_autogptq):
            if low_bit_mode == 1:
                from transformers import BitsAndBytesConfig
                model_kwargs['quantization_config'] = BitsAndBytesConfig(bnb_4bit_compute_dtype=torch.bfloat16,
                                                                         load_in_4bit=load_4bit,
                                                                         load_in_8bit=load_8bit,
                                                                         )
            elif low_bit_mode == 2:
                from transformers import BitsAndBytesConfig
                model_kwargs['quantization_config'] = BitsAndBytesConfig(bnb_4bit_quant_type="nf4",
                                                                         load_in_4bit=load_4bit,
                                                                         load_in_8bit=load_8bit,
                                                                         )
            elif low_bit_mode == 3:
                from transformers import BitsAndBytesConfig
                model_kwargs['quantization_config'] = BitsAndBytesConfig(bnb_4bit_use_double_quant=True,
                                                                         load_in_4bit=load_4bit,
                                                                         load_in_8bit=load_8bit,
                                                                         )
            elif low_bit_mode == 4:
                from transformers import BitsAndBytesConfig
                model_kwargs['quantization_config'] = BitsAndBytesConfig(bnb_4bit_use_double_quant=True,
                                                                         bnb_4bit_quant_type="nf4",
                                                                         load_in_4bit=load_4bit,
                                                                         load_in_8bit=load_8bit,
                                                                         )

        if not lora_weights:
            # torch.device context uses twice memory for AutoGPTQ
            context = NullContext if (load_gptq and use_autogptq or load_awq) else torch.device
            with context(device):

                if use_gpu_id:
                    config, model, max_seq_len = get_config(base_model,
                                                            return_model=True, raise_exception=True, **config_kwargs)
                    model = get_non_lora_model(base_model, model_loader, load_half,
                                               load_gptq,
                                               use_autogptq,
                                               load_awq,
                                               load_exllama,
                                               use_safetensors,
                                               revision,
                                               model_kwargs, reward_type,
                                               config, model,
                                               gpu_id=gpu_id,
                                               )
                else:
                    model_kwargs['use_safetensors'] = use_safetensors
                    model_kwargs['revision'] = revision
                    config, _, max_seq_len = get_config(base_model, **config_kwargs)
                    if load_half and not (load_8bit or load_4bit or load_gptq and use_autogptq or load_awq):
                        model = model_loader(
                            base_model,
                            config=config,
                            **model_kwargs)
                        if not getattr(model, "is_quantized", False):
                            model = model.half()
                    else:
                        if load_gptq and use_autogptq:
                            model_kwargs.pop('torch_dtype', None)
                            model = model_loader(
                                model_name_or_path=base_model,
                                model_basename=load_gptq,
                                **model_kwargs,
                            )
                        elif load_awq:
                            allowed_dict = dict(max_new_tokens=None,
                                                trust_remote_code=True, fuse_layers=True,
                                                batch_size=1, safetensors=False,
                                                max_memory=None, offload_folder=None)
                            for k in model_kwargs.copy():
                                if k not in allowed_dict:
                                    model_kwargs.pop(k)
                            if load_awq.endswith('.pt'):
                                args = tuple([base_model, load_awq])
                            else:
                                args = tuple([base_model])
                            model = model_loader(
                                *args,
                                safetensors=use_safetensors,
                                **model_kwargs,
                            )
                        else:
                            model = model_loader(
                                base_model,
                                config=config,
                                **model_kwargs)
        elif load_8bit or load_4bit:
            config, _, max_seq_len = get_config(base_model, **config_kwargs)
            model = model_loader(
                base_model,
                config=config,
                **model_kwargs
            )
            from peft import PeftModel  # loads cuda, so avoid in global scope
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                local_files_only=local_files_only,
                resume_download=resume_download,
                token=use_auth_token,
                trust_remote_code=trust_remote_code,
                offload_folder=offload_folder,
                rope_scaling=rope_scaling,
                revision=revision,
                device_map={"": 0} if device == 'cuda' else {"": 'cpu'},  # seems to be required
            )
        else:
            with torch.device(device):
                config, _, max_seq_len = get_config(base_model, raise_exception=True, **config_kwargs)
                model = model_loader(
                    base_model,
                    config=config,
                    **model_kwargs
                )
                from peft import PeftModel  # loads cuda, so avoid in global scope
                model = PeftModel.from_pretrained(
                    model,
                    lora_weights,
                    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                    local_files_only=local_files_only,
                    resume_download=resume_download,
                    token=use_auth_token,
                    trust_remote_code=trust_remote_code,
                    offload_folder=offload_folder,
                    rope_scaling=rope_scaling,
                    device_map="auto",
                )
                if load_half and not (load_gptq and use_autogptq or load_awq):
                    if not getattr(model, "is_quantized", False):
                        model = model.half()

    # for LlamaAWQForCausalLM
    # https://github.com/casper-hansen/AutoAWQ/issues/107
    # unwind broken decapoda-research config
    if llama_type and hasattr(model, 'config'):
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    if 'gpt2' in base_model.lower():
        # add special tokens that otherwise all share the same id
        tokenizer.add_special_tokens({'bos_token': '<bos>',
                                      'eos_token': '<eos>',
                                      'pad_token': '<pad>'})

    if not isinstance(tokenizer, str) and hasattr(model, 'eval'):
        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32" and compile_model:
            model = torch.compile(model)

    set_model_max_len(max_seq_len, tokenizer, verbose=False, reward_type=reward_type)

    # tell if conditional type
    model.conditional_type = conditional_type
    tokenizer.conditional_type = conditional_type

    # https://github.com/PanQiWei/AutoGPTQ/issues/323
    if load_gptq and not use_autogptq:
        from auto_gptq import exllama_set_max_input_length
        try:
            model = exllama_set_max_input_length(model, tokenizer.model_max_length)
        except Exception as e:
            # HF transformers AutoGPTQ use is NOT user friendly
            if 'The method exllama_set_max_input_length ' in str(e):
                pass
            else:
                raise

    return model, tokenizer, device


def set_model_max_len(max_seq_len, tokenizer, verbose=False, reward_type=False):
    if reward_type:
        # limit deberta, else uses too much memory and not worth response score
        tokenizer.model_max_length = 512
        return

    tokenizer.model_max_length = int(max_seq_len)
    if verbose:
        print("model_max_length=%s" % tokenizer.model_max_length, flush=True)
    # for bug in HF transformers
    if tokenizer.model_max_length > 100000000:
        tokenizer.model_max_length = 2048


def pop_unused_model_kwargs(model_kwargs):
    """
    in-place pop unused kwargs that are not dependency-upgrade friendly
    no point passing in False, is default, and helps avoid needing to update requirements for new deps
    :param model_kwargs:
    :return:
    """
    check_list = ['load_in_8bit', 'load_in_4bit']
    for k in check_list:
        if k in model_kwargs and not model_kwargs[k]:
            model_kwargs.pop(k)


def get_score_model(score_model: str = None,
                    load_8bit: bool = False,
                    load_4bit: bool = False,
                    low_bit_mode=1,
                    load_half: bool = True,
                    load_gptq: str = '',
                    use_autogptq: bool = False,
                    load_awq: str = '',
                    load_exllama: bool = False,
                    use_gpu_id: bool = True,
                    base_model: str = '',
                    inference_server: str = '',
                    tokenizer_base_model: str = '',
                    lora_weights: str = "",
                    gpu_id: int = 0,
                    n_jobs=None,
                    n_gpus=None,

                    reward_type: bool = None,
                    local_files_only: bool = False,
                    resume_download: bool = True,
                    use_auth_token: Union[str, bool] = False,
                    trust_remote_code: bool = True,
                    offload_folder: str = None,
                    rope_scaling: dict = None,
                    compile_model: bool = True,
                    llamacpp_dict: typing.Dict = None,
                    exllama_dict: typing.Dict = None,
                    gptq_dict: typing.Dict = None,
                    attention_sinks: bool = False,
                    sink_dict: typing.Dict = None,
                    truncation_generation: bool = False,
                    hf_model_dict: typing.Dict = None,

                    verbose: bool = False,
                    ):
    if score_model is not None and score_model.strip():
        load_8bit = False
        load_4bit = False
        low_bit_mode = 1
        load_half = False
        load_gptq = ''
        use_autogptq = False
        load_awq = ''
        load_exllama = False
        use_safetensors = False
        revision = None
        base_model = score_model.strip()
        tokenizer_base_model = ''
        lora_weights = ''
        inference_server = ''
        llama_type = False
        max_seq_len = None
        rope_scaling = {}
        compile_model = False
        llamacpp_dict = {}
        exllama_dict = {}
        gptq_dict = {}
        attention_sinks = False
        sink_dict = {}
        truncation_generation = False
        hf_model_dict = {}
        smodel, stokenizer, sdevice = get_model(reward_type=True,
                                                **get_kwargs(get_model, exclude_names=['reward_type'], **locals()))
    else:
        smodel, stokenizer, sdevice = None, None, None
    return smodel, stokenizer, sdevice


def evaluate_fake(*args, **kwargs):
    yield dict(response=invalid_key_msg, sources='', save_dict=dict(), llm_answers={})
    return


def evaluate(
        model_state,
        my_db_state,
        selection_docs_state,
        requests_state,
        # START NOTE: Examples must have same order of parameters
        instruction,
        iinput,
        context,
        stream_output,
        prompt_type,
        prompt_dict,
        temperature,
        top_p,
        top_k,
        penalty_alpha,
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
        add_chat_history_to_context,
        langchain_action,
        langchain_agents,
        top_k_docs,
        chunk,
        chunk_size,
        document_subset,
        document_choice,

        pre_prompt_query,
        prompt_query,
        pre_prompt_summary,
        prompt_summary,
        system_prompt,

        image_loaders,
        pdf_loaders,
        url_loaders,
        jq_schema,
        visible_models,
        h2ogpt_key,
        add_search_to_context,

        chat_conversation,
        text_context_list,
        docs_ordering_type,
        min_max_new_tokens,
        max_input_tokens,
        max_total_input_tokens,
        docs_token_handling,
        docs_joiner,
        hyde_level,
        hyde_template,
        doc_json_mode,

        # END NOTE: Examples must have same order of parameters
        captions_model=None,
        caption_loader=None,
        doctr_loader=None,
        pix2struct_loader=None,
        async_output=None,
        num_async=None,
        src_lang=None,
        tgt_lang=None,
        debug=False,
        concurrency_count=None,
        save_dir=None,
        sanitize_bot_response=False,
        model_state0=None,
        memory_restriction_level=None,
        max_max_new_tokens=None,
        is_public=None,
        from_ui=True,
        max_max_time=None,
        raise_generate_gpu_exceptions=None,
        lora_weights=None,
        use_llm_if_no_docs=True,
        load_db_if_exists=True,
        dbs=None,
        detect_user_path_changes_every_query=None,
        use_openai_embedding=None,
        use_openai_model=None,
        hf_embedding_model=None,
        migrate_embedding_model=None,
        auto_migrate_db=None,
        cut_distance=None,
        db_type=None,
        n_jobs=None,
        first_para=None,
        text_limit=None,
        show_accordions=None,
        top_k_docs_max_show=None,
        show_link_in_sources=None,
        verbose=False,
        gradio=True,
        cli=False,
        use_cache=None,
        auto_reduce_chunks=None,
        max_chunks=None,
        headsize=None,
        model_lock=None,
        force_langchain_evaluate=None,
        model_state_none=None,
        llamacpp_dict=None,
        exllama_dict=None,
        gptq_dict=None,
        attention_sinks=None,
        sink_dict=None,
        truncation_generation=None,
        hf_model_dict=None,

        load_exllama=None,
        answer_with_sources=None,
        append_sources_to_answer=None,
        image_loaders_options0=None,
        pdf_loaders_options0=None,
        url_loaders_options0=None,
        jq_schema0=None,
        keep_sources_in_context=None,
):
    # ensure passed these
    assert concurrency_count is not None
    assert memory_restriction_level is not None
    assert raise_generate_gpu_exceptions is not None
    assert use_openai_embedding is not None
    assert use_openai_model is not None
    assert hf_embedding_model is not None
    assert migrate_embedding_model is not None
    assert auto_migrate_db is not None
    assert db_type is not None
    assert top_k_docs is not None and isinstance(top_k_docs, int)
    assert chunk is not None and isinstance(chunk, bool)
    assert chunk_size is not None and isinstance(chunk_size, int)
    assert n_jobs is not None
    assert first_para is not None
    assert isinstance(add_chat_history_to_context, bool)
    assert isinstance(add_search_to_context, bool)
    assert load_exllama is not None
    # for lazy client (even chat client)
    if image_loaders is None:
        image_loaders = image_loaders_options0
    if pdf_loaders is None:
        pdf_loaders = pdf_loaders_options0
    if url_loaders is None:
        url_loaders = url_loaders_options0
    if jq_schema is None:
        jq_schema = jq_schema0
    if isinstance(langchain_agents, str):
        if langchain_agents.strip().startswith('['):
            # already list, but as string
            langchain_agents = str_to_list(langchain_agents)
        else:
            # just 1 item and make list
            langchain_agents = [langchain_agents]
    chat_conversation = str_to_list(chat_conversation)
    text_context_list = str_to_list(text_context_list)

    langchain_modes = selection_docs_state['langchain_modes']
    langchain_mode_paths = selection_docs_state['langchain_mode_paths']
    langchain_mode_types = selection_docs_state['langchain_mode_types']

    if debug:
        locals_dict = locals().copy()
        locals_dict.pop('model_state', None)
        locals_dict.pop('model_state0', None)
        locals_dict.pop('model_states', None)
        print(locals_dict)

    no_model_msg = "Please choose a base model with --base_model (CLI) or load in Models Tab (gradio).\n" \
                   "Then start New Conversation"

    if model_state is None:
        model_state = model_state_none.copy()
    if model_state0 is None:
        # e.g. for no gradio case, set dummy value, else should be set
        model_state0 = model_state_none.copy()

    # model_state['model] is only 'model' if should use model_state0
    # model could also be None
    have_model_lock = model_lock is not None
    have_fresh_model = model_state['model'] not in [None, 'model', no_model_str]
    # for gradio UI control, expect model_state and model_state0 to match, so if have_model_lock=True, then should have_fresh_model=True
    # but gradio API control will only use nochat api etc. and won't use fresh model, so can't assert in general
    # if have_model_lock:
    #    assert have_fresh_model, "Expected model_state and model_state0 to match if have_model_lock"
    have_cli_model = model_state0['model'] not in [None, 'model', no_model_str]

    if have_fresh_model:
        # USE FRESH MODEL
        if not have_model_lock:
            # model_state0 is just one of model_state if model_lock, so don't nuke
            # try to free-up original model (i.e. list was passed as reference)
            if model_state0['model'] and hasattr(model_state0['model'], 'cpu'):
                model_state0['model'].cpu()
                model_state0['model'] = None
            # try to free-up original tokenizer (i.e. list was passed as reference)
            if model_state0['tokenizer']:
                model_state0['tokenizer'] = None
            clear_torch_cache()
        chosen_model_state = model_state
    elif have_cli_model:
        # USE MODEL SETUP AT CLI
        assert isinstance(model_state['model'], (type(None), str))  # expect no fresh model
        chosen_model_state = model_state0
    else:
        raise AssertionError(no_model_msg)
    # get variables
    model = chosen_model_state['model']
    tokenizer = chosen_model_state['tokenizer']
    device = chosen_model_state['device']
    base_model = chosen_model_state['base_model']
    tokenizer_base_model = chosen_model_state['tokenizer_base_model']
    lora_weights = chosen_model_state['lora_weights']
    inference_server = chosen_model_state['inference_server']
    visible_models = chosen_model_state['visible_models']
    # use overall key if have, so key for this gradio and any inner gradio
    if chosen_model_state['h2ogpt_key'] is not None:
        h2ogpt_key = chosen_model_state['h2ogpt_key']
    # prefer use input from API over model state
    prompt_type = prompt_type or chosen_model_state['prompt_type']
    prompt_dict = prompt_dict or chosen_model_state['prompt_dict']

    if base_model is None:
        raise AssertionError(no_model_msg)

    assert base_model.strip(), no_model_msg
    assert model, "Model is missing"
    assert tokenizer, "Tokenizer is missing"

    # choose chat or non-chat mode
    if not chat:
        instruction = instruction_nochat
        iinput = iinput_nochat

    # avoid instruction in chat_conversation itself, since always used as additional context to prompt in what follows
    if isinstance(chat_conversation, list) and \
            len(chat_conversation) > 0 and \
            len(chat_conversation[-1]) == 2 and \
            chat_conversation[-1][0] == instruction and \
            chat_conversation[-1][1] in [None, '']:
        chat_conversation = chat_conversation[:-1]
    if not add_chat_history_to_context:
        # make it easy to ignore without needing add_chat_history_to_context
        # some langchain or unit test may need to then handle more general case
        chat_conversation = []

    # in some cases, like lean nochat API, don't want to force sending prompt_type, allow default choice
    # This doesn't do switch-a-roo, assume already done
    model_lower = base_model.lower()
    if not prompt_type and model_lower in inv_prompt_type_to_model_lower and prompt_type != 'custom':
        prompt_type = inv_prompt_type_to_model_lower[model_lower]
        if verbose:
            print("Auto-selecting prompt_type=%s for %s" % (prompt_type, model_lower), flush=True)
    assert prompt_type is not None, "prompt_type was None"

    # Control generation hyperparameters
    # adjust for bad inputs, e.g. in case also come from API that doesn't get constrained by gradio sliders
    # below is for TGI server, not required for HF transformers
    # limits are chosen similar to gradio_runner.py sliders/numbers
    top_p = min(max(1e-3, top_p), 1.0 - 1e-3)
    top_k = min(max(1, int(top_k)), 100)
    penalty_alpha = min(2.0, max(0.0, penalty_alpha))
    temperature = min(max(0.01, temperature), 2.0)
    max_input_tokens = int(max_input_tokens) if max_input_tokens is not None else -1
    max_total_input_tokens = int(max_total_input_tokens) if max_total_input_tokens is not None else -1
    # FIXME: https://github.com/h2oai/h2ogpt/issues/106
    num_beams = 1 if stream_output else num_beams  # See max_beams in gradio_runner
    if model_lower == 'distilgpt2':
        # always truncate for certain models that totally fail otherwise
        truncation_generation = True
    max_max_new_tokens = get_max_max_new_tokens(chosen_model_state,
                                                memory_restriction_level=memory_restriction_level,
                                                max_new_tokens=max_new_tokens,
                                                attention_sinks=attention_sinks,
                                                max_max_new_tokens=max_max_new_tokens,
                                                truncation_generation=truncation_generation)
    if min_max_new_tokens is None:
        # default for nochat api
        min_max_new_tokens = 256
    if max_input_tokens is None:
        max_input_tokens = -1
    if max_total_input_tokens is None:
        max_total_input_tokens = -1
    if docs_ordering_type is None:
        docs_ordering_type = docs_ordering_types_default
    if docs_token_handling is None:
        docs_token_handling = docs_token_handling_default
    if docs_joiner is None:
        docs_joiner = docs_joiner_default
    model_max_length = get_model_max_length(chosen_model_state)
    max_new_tokens = min(max(1, int(max_new_tokens)), max_max_new_tokens)
    min_new_tokens = min(max(0, int(min_new_tokens)), max_new_tokens)
    max_time = min(max(0, max_time), max_max_time)
    repetition_penalty = min(max(0.01, repetition_penalty), 3.0)
    num_return_sequences = 1 if chat else min(max(1, int(num_return_sequences)), 10)
    min_top_k_docs, max_top_k_docs, label_top_k_docs = get_minmax_top_k_docs(is_public, from_ui)
    # limit total tokens processed, e.g. for summarization, if public instance
    if is_public:
        # control API too for public case
        if from_ui:
            max_input_tokens = max_input_tokens_public
        else:
            max_input_tokens = max_input_tokens_public_api

        if from_ui:
            max_total_input_tokens = min(max_total_input_tokens, max_total_input_tokens_public)
        else:
            max_total_input_tokens = min(max_total_input_tokens, max_total_input_tokens_public_api)
    top_k_docs = min(max(min_top_k_docs, int(top_k_docs)), max_top_k_docs)
    chunk_size = min(max(128, int(chunk_size)), 2048)
    if not context:
        context = ''

    # NOTE!!!!!!!!!!  Choice of developer.  But only possible to force stream if num_beams=1
    # stream if can, so can control task iteration and time of iteration
    # not required, but helpful for max_time control etc.
    stream_output0 = stream_output
    stream_output = gradio and num_beams == 1

    # get prompter
    prompter = Prompter(prompt_type, prompt_dict, debug=debug, chat=chat, stream_output=stream_output,
                        system_prompt=system_prompt)

    # THIRD PLACE where LangChain referenced, but imports only occur if enabled and have db to use
    assert langchain_mode in langchain_modes, "Invalid langchain_mode %s not in %s" % (langchain_mode, langchain_modes)
    assert langchain_action in langchain_actions, "Invalid langchain_action %s not in %s" % (
        langchain_action, langchain_actions)
    assert len(
        set(langchain_agents).difference(langchain_agents_list)) == 0, "Invalid langchain_agents %s" % langchain_agents

    # get db, but also fill db state so return already has my_db_state and dbs filled so faster next query
    if langchain_mode != LangChainMode.DISABLED.value:
        from src.gpt_langchain import get_any_db
        db = get_any_db(my_db_state, langchain_mode, langchain_mode_paths, langchain_mode_types,
                        dbs=dbs,
                        load_db_if_exists=load_db_if_exists,
                        db_type=db_type,
                        use_openai_embedding=use_openai_embedding,
                        hf_embedding_model=hf_embedding_model,
                        migrate_embedding_model=migrate_embedding_model,
                        auto_migrate_db=auto_migrate_db,
                        for_sources_list=True,
                        verbose=verbose,
                        n_jobs=n_jobs,
                        )
    else:
        db = None

    t_generate = time.time()
    langchain_only_model = base_model in non_hf_types or \
                           load_exllama or \
                           inference_server.startswith('replicate') or \
                           inference_server.startswith('sagemaker') or \
                           inference_server.startswith('openai_azure_chat') or \
                           inference_server.startswith('openai_azure')
    do_langchain_path = langchain_mode not in [False, 'Disabled', 'LLM'] or \
                        langchain_only_model or \
                        force_langchain_evaluate or \
                        len(text_context_list) > 0

    if len(langchain_agents) > 0:
        do_langchain_path = True
    if add_search_to_context:
        # easier to manage prompt etc. by doing full langchain path
        do_langchain_path = True

    if do_langchain_path:
        text = ''
        sources = ''
        response = ''
        # use smaller cut_distance for wiki_full since so many matches could be obtained, and often irrelevant unless close
        from gpt_langchain import run_qa_db
        gen_hyper_langchain = dict(do_sample=do_sample,
                                   temperature=temperature,
                                   repetition_penalty=repetition_penalty,
                                   top_p=top_p,
                                   top_k=top_k,
                                   penalty_alpha=penalty_alpha,
                                   num_beams=num_beams,
                                   min_new_tokens=min_new_tokens,
                                   max_new_tokens=max_new_tokens,
                                   early_stopping=early_stopping,
                                   max_time=max_time,
                                   num_return_sequences=num_return_sequences,
                                   )
        loaders_dict, captions_model = gr_to_lg(image_loaders,
                                                pdf_loaders,
                                                url_loaders,
                                                captions_model=captions_model,
                                                )
        loaders_dict.update(dict(captions_model=captions_model,
                                 caption_loader=caption_loader,
                                 doctr_loader=doctr_loader,
                                 pix2struct_loader=pix2struct_loader,
                                 jq_schema=jq_schema,
                                 ))
        data_point = dict(context=context, instruction=instruction, input=iinput)
        # no longer stuff chat history directly into context this early
        prompt_basic = prompter.generate_prompt(data_point, context_from_history=False)
        prompt = prompt_basic
        num_prompt_tokens = 0
        llm_answers = {}
        for r in run_qa_db(
                inference_server=inference_server,
                model_name=base_model, model=model, tokenizer=tokenizer,
                langchain_only_model=langchain_only_model,
                async_output=async_output,
                num_async=num_async,
                prompter=prompter,
                use_llm_if_no_docs=use_llm_if_no_docs,
                load_db_if_exists=load_db_if_exists,
                db=db,
                langchain_mode_paths=langchain_mode_paths,
                langchain_mode_types=langchain_mode_types,
                detect_user_path_changes_every_query=detect_user_path_changes_every_query,
                cut_distance=1.1 if langchain_mode in ['wiki_full'] else cut_distance,
                answer_with_sources=answer_with_sources,
                append_sources_to_answer=append_sources_to_answer,
                add_chat_history_to_context=add_chat_history_to_context,
                add_search_to_context=add_search_to_context,
                keep_sources_in_context=keep_sources_in_context,
                memory_restriction_level=memory_restriction_level,
                system_prompt=system_prompt,
                use_openai_embedding=use_openai_embedding,
                use_openai_model=use_openai_model,
                hf_embedding_model=hf_embedding_model,
                migrate_embedding_model=migrate_embedding_model,
                auto_migrate_db=auto_migrate_db,
                first_para=first_para,
                text_limit=text_limit,
                show_accordions=show_accordions,
                top_k_docs_max_show=top_k_docs_max_show,
                show_link_in_sources=show_link_in_sources,

                # evaluate args items
                query=instruction,
                iinput=iinput,
                context=context,
                stream_output0=stream_output0,
                stream_output=stream_output,
                chunk=chunk,
                chunk_size=chunk_size,

                **loaders_dict,

                langchain_mode=langchain_mode,
                langchain_action=langchain_action,
                langchain_agents=langchain_agents,
                document_subset=document_subset,
                document_choice=document_choice,
                top_k_docs=top_k_docs,
                prompt_type=prompt_type,
                prompt_dict=prompt_dict,
                pre_prompt_query=pre_prompt_query,
                prompt_query=prompt_query,
                pre_prompt_summary=pre_prompt_summary,
                prompt_summary=prompt_summary,
                text_context_list=text_context_list,
                chat_conversation=chat_conversation,
                visible_models=visible_models,
                h2ogpt_key=h2ogpt_key,
                docs_ordering_type=docs_ordering_type,
                min_max_new_tokens=min_max_new_tokens,
                max_input_tokens=max_input_tokens,
                max_total_input_tokens=max_total_input_tokens,
                docs_token_handling=docs_token_handling,
                docs_joiner=docs_joiner,
                hyde_level=hyde_level,
                hyde_template=hyde_template,
                doc_json_mode=doc_json_mode,

                **gen_hyper_langchain,

                db_type=db_type,
                n_jobs=n_jobs,
                verbose=verbose,
                cli=cli,
                sanitize_bot_response=sanitize_bot_response,

                lora_weights=lora_weights,
                llamacpp_dict=llamacpp_dict,
                exllama_dict=exllama_dict,
                gptq_dict=gptq_dict,
                attention_sinks=attention_sinks,
                sink_dict=sink_dict,
                truncation_generation=truncation_generation,
                hf_model_dict=hf_model_dict,

                auto_reduce_chunks=auto_reduce_chunks,
                max_chunks=max_chunks,
                headsize=headsize,
        ):
            # doesn't accumulate, new answer every yield, so only save that full answer
            response = r['response']
            sources = r['sources']
            prompt = r['prompt']
            num_prompt_tokens = r['num_prompt_tokens']
            llm_answers = r['llm_answers']
            yield dict(response=response, sources=sources, save_dict=dict(), llm_answers=llm_answers)
        if save_dir:
            # estimate using tiktoken
            extra_dict = gen_hyper_langchain.copy()
            extra_dict.update(prompt_type=prompt_type,
                              inference_server=inference_server,
                              langchain_mode=langchain_mode,
                              langchain_action=langchain_action,
                              langchain_agents=langchain_agents,
                              document_subset=document_subset,
                              document_choice=document_choice,
                              chat_conversation=chat_conversation,
                              add_search_to_context=add_search_to_context,
                              num_prompt_tokens=num_prompt_tokens,
                              instruction=instruction,
                              iinput=iinput,
                              context=context,
                              t_generate=time.time() - t_generate,
                              ntokens=None,
                              tokens_persecond=None,
                              )
            save_dict = dict(prompt=prompt,
                             output=response, base_model=base_model, save_dir=save_dir,
                             where_from='run_qa_db',
                             extra_dict=extra_dict)
            yield dict(response=response, sources=sources, save_dict=save_dict, llm_answers=llm_answers)
            if verbose:
                print(
                    'Post-Generate Langchain: %s decoded_output: %s' %
                    (str(datetime.now()), len(response) if response else -1),
                    flush=True)
        if response or sources or langchain_only_model:
            # if got no response (e.g. not showing sources and got no sources,
            # so nothing to give to LLM), then slip through and ask LLM
            # Or if llama/gptj, then just return since they had no response and can't go down below code path
            # don't clear torch cache here, delays multi-generation, and bot(), all_bot(), and evaluate_nochat() do it
            return

    # NOT LANGCHAIN PATH, raw LLM
    # restrict instruction + , typically what has large input
    from gradio_utils.grclient import GradioClient
    gradio_server = inference_server.startswith('http') and isinstance(model, GradioClient)

    prompt, \
        instruction, iinput, context, \
        num_prompt_tokens, max_new_tokens, num_prompt_tokens0, num_prompt_tokens_actual, \
        chat_index, external_handle_chat_conversation, \
        top_k_docs_trial, one_doc_size, truncation_generation = \
        get_limited_prompt(instruction,
                           iinput,
                           tokenizer,
                           prompter=prompter,
                           inference_server=inference_server,
                           # prompt_type=prompt_type,
                           # prompt_dict=prompt_dict,
                           # chat=chat,
                           max_new_tokens=max_new_tokens,
                           # system_prompt=system_prompt,
                           context=context,
                           chat_conversation=chat_conversation,
                           keep_sources_in_context=keep_sources_in_context,
                           model_max_length=model_max_length,
                           memory_restriction_level=memory_restriction_level,
                           langchain_mode=langchain_mode,
                           add_chat_history_to_context=add_chat_history_to_context,
                           min_max_new_tokens=min_max_new_tokens,
                           max_input_tokens=max_input_tokens,
                           max_total_input_tokens=max_total_input_tokens,
                           truncation_generation=truncation_generation,
                           gradio_server=gradio_server,
                           )

    if inference_server.startswith('vllm') or \
            inference_server.startswith('openai') or \
            inference_server.startswith('http'):
        if inference_server.startswith('vllm') or inference_server.startswith('openai'):
            assert not inference_server.startswith('openai_azure_chat'), "Not fo Azure, use langchain path"
            assert not inference_server.startswith('openai_azure'), "Not for Azure, use langchain path"
            openai, inf_type, deployment_name, base_url, api_version, api_key = set_openai(inference_server)
            where_from = inf_type

            terminate_response = prompter.terminate_response or []
            stop_sequences = list(set(terminate_response + [prompter.PreResponse]))
            stop_sequences = [x for x in stop_sequences if x]
            # OpenAI will complain if ask for too many new tokens, takes it as min in some sense, wrongly so.
            max_new_tokens_openai = min(max_new_tokens, model_max_length - num_prompt_tokens)
            gen_server_kwargs = dict(temperature=temperature if do_sample else 0,
                                     max_tokens=max_new_tokens_openai,
                                     top_p=top_p if do_sample else 1,
                                     frequency_penalty=0,
                                     n=num_return_sequences,
                                     presence_penalty=1.07 - repetition_penalty + 0.6,  # so good default
                                     )
            if inf_type == 'vllm' or inference_server == 'openai':
                responses = openai.Completion.create(
                    model=base_model,
                    prompt=prompt,
                    **gen_server_kwargs,
                    stop=stop_sequences,
                    stream=stream_output,
                    request_timeout=max_time,
                )
                text = ''
                sources = ''
                response = ''
                if not stream_output:
                    text = responses['choices'][0]['text']
                    response = prompter.get_response(prompt + text, prompt=prompt,
                                                     sanitize_bot_response=sanitize_bot_response)
                    yield dict(response=response, sources=sources, save_dict=dict(), llm_answers={})
                else:
                    collected_events = []
                    tgen0 = time.time()
                    for event in responses:
                        collected_events.append(event)  # save the event response
                        event_text = event['choices'][0]['text']  # extract the text
                        text += event_text  # append the text
                        response = prompter.get_response(prompt + text, prompt=prompt,
                                                         sanitize_bot_response=sanitize_bot_response)
                        yield dict(response=response, sources=sources, save_dict=dict(), llm_answers={})
                        if time.time() - tgen0 > max_time:
                            if verbose:
                                print("Took too long for OpenAI or VLLM: %s" % (time.time() - tgen0), flush=True)
                            break
            elif inf_type == 'vllm_chat' or inference_server == 'openai_chat':
                if system_prompt in [None, 'None', 'auto']:
                    openai_system_prompt = "You are a helpful assistant."
                else:
                    openai_system_prompt = system_prompt
                messages0 = []
                if openai_system_prompt:
                    messages0.append({"role": "system", "content": openai_system_prompt})
                if chat_conversation and add_chat_history_to_context:
                    assert external_handle_chat_conversation, "Should be handling only externally"
                    # chat_index handles token counting issues
                    for message1 in chat_conversation[chat_index:]:
                        if len(message1) == 2:
                            messages0.append(
                                {'role': 'user', 'content': message1[0] if message1[0] is not None else ''})
                            messages0.append(
                                {'role': 'assistant', 'content': message1[1] if message1[1] is not None else ''})
                messages0.append({'role': 'user', 'content': prompt if prompt is not None else ''})
                responses = openai.ChatCompletion.create(
                    model=base_model,
                    messages=messages0,
                    stream=stream_output,
                    **gen_server_kwargs,
                    request_timeout=max_time,
                )
                text = ""
                sources = ''
                response = ""
                if not stream_output:
                    text = responses["choices"][0]["message"]["content"]
                    response = prompter.get_response(prompt + text, prompt=prompt,
                                                     sanitize_bot_response=sanitize_bot_response)
                    yield dict(response=response, sources=sources, save_dict=dict(), llm_answers={})
                else:
                    tgen0 = time.time()
                    for chunk in responses:
                        delta = chunk["choices"][0]["delta"]
                        if 'content' in delta:
                            text += delta['content']
                            response = prompter.get_response(prompt + text, prompt=prompt,
                                                             sanitize_bot_response=sanitize_bot_response)
                            yield dict(response=response, sources=sources, save_dict=dict(), llm_answers={})
                        if time.time() - tgen0 > max_time:
                            if verbose:
                                print("Took too long for OpenAI or VLLM Chat: %s" % (time.time() - tgen0), flush=True)
                            break
            else:
                raise RuntimeError("No such OpenAI mode: %s" % inference_server)
        elif inference_server.startswith('http'):
            inference_server, headers = get_hf_server(inference_server)
            from text_generation import Client as HFClient
            if isinstance(model, GradioClient):
                gr_client = model.clone()
                hf_client = None
            elif isinstance(model, HFClient):
                gr_client = None
                hf_client = model
            else:
                inference_server, gr_client, hf_client = get_client_from_inference_server(inference_server,
                                                                                          base_model=base_model)

            # quick sanity check to avoid long timeouts, just see if can reach server
            requests.get(inference_server, timeout=int(os.getenv('REQUEST_TIMEOUT_FAST', '10')))

            if gr_client is not None:
                # Note: h2oGPT gradio server could handle input token size issues for prompt,
                # but best to handle here so send less data to server

                chat_client = False
                where_from = "gr_client"
                client_langchain_mode = 'Disabled'
                client_add_chat_history_to_context = True
                client_add_search_to_context = False
                client_langchain_action = LangChainAction.QUERY.value
                client_langchain_agents = []
                gen_server_kwargs = dict(temperature=temperature,
                                         top_p=top_p,
                                         top_k=top_k,
                                         penalty_alpha=penalty_alpha,
                                         num_beams=num_beams,
                                         max_new_tokens=max_new_tokens,
                                         min_new_tokens=min_new_tokens,
                                         early_stopping=early_stopping,
                                         max_time=max_time,
                                         repetition_penalty=repetition_penalty,
                                         num_return_sequences=num_return_sequences,
                                         do_sample=do_sample,
                                         chat=chat_client,
                                         )
                # account for gradio into gradio that handles prompting, avoid duplicating prompter prompt injection
                if prompt_type in [None, '', PromptType.plain.name, PromptType.plain.value,
                                   str(PromptType.plain.value)]:
                    # if our prompt is plain, assume either correct or gradio server knows different prompt type,
                    # so pass empty prompt_Type
                    gr_prompt_type = ''
                    gr_prompt_dict = ''
                    gr_prompt = prompt  # already prepared prompt
                    gr_context = ''
                    gr_iinput = ''
                else:
                    # if already have prompt_type that is not plain, None, or '', then already applied some prompting
                    #  But assume server can handle prompting, and need to avoid double-up.
                    #  Also assume server can do better job of using stopping.py to stop early, so avoid local prompting, let server handle
                    #  So avoid "prompt" and let gradio server reconstruct from prompt_type we passed
                    # Note it's ok that prompter.get_response() has prompt+text, prompt=prompt passed,
                    #  because just means extra processing and removal of prompt, but that has no human-bot prompting doesn't matter
                    #  since those won't appear
                    gr_context = context
                    gr_prompt = instruction
                    gr_iinput = iinput
                    gr_prompt_type = prompt_type
                    gr_prompt_dict = prompt_dict
                client_kwargs = dict(instruction=gr_prompt if chat_client else '',  # only for chat=True
                                     iinput=gr_iinput,  # only for chat=True
                                     context=gr_context,
                                     # streaming output is supported, loops over and outputs each generation in streaming mode
                                     # but leave stream_output=False for simple input/output mode
                                     stream_output=stream_output,

                                     **gen_server_kwargs,

                                     prompt_type=gr_prompt_type,
                                     prompt_dict=gr_prompt_dict,

                                     instruction_nochat=gr_prompt if not chat_client else '',
                                     iinput_nochat=gr_iinput,  # only for chat=False
                                     langchain_mode=client_langchain_mode,
                                     add_chat_history_to_context=client_add_chat_history_to_context,
                                     langchain_action=client_langchain_action,
                                     langchain_agents=client_langchain_agents,
                                     top_k_docs=top_k_docs,
                                     chunk=chunk,
                                     chunk_size=chunk_size,
                                     document_subset=DocumentSubset.Relevant.name,
                                     document_choice=[DocumentChoice.ALL.value],
                                     pre_prompt_query=pre_prompt_query,
                                     prompt_query=prompt_query,
                                     pre_prompt_summary=pre_prompt_summary,
                                     prompt_summary=prompt_summary,
                                     system_prompt=system_prompt,
                                     image_loaders=image_loaders,
                                     pdf_loaders=pdf_loaders,
                                     url_loaders=url_loaders,
                                     jq_schema=jq_schema,
                                     visible_models=visible_models,
                                     h2ogpt_key=h2ogpt_key,
                                     add_search_to_context=client_add_search_to_context,
                                     docs_ordering_type=docs_ordering_type,
                                     min_max_new_tokens=min_max_new_tokens,
                                     max_input_tokens=max_input_tokens,
                                     max_total_input_tokens=max_total_input_tokens,
                                     docs_token_handling=docs_token_handling,
                                     docs_joiner=docs_joiner,
                                     hyde_level=hyde_level,
                                     hyde_template=hyde_template,
                                     doc_json_mode=doc_json_mode,
                                     )
                api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
                response = ''
                text = ''
                sources = ''
                strex = ''
                if not stream_output:
                    res = gr_client.predict(str(dict(client_kwargs)), api_name=api_name)
                    res_dict = ast.literal_eval(res)
                    text = res_dict['response']
                    sources = res_dict['sources']
                    response = prompter.get_response(prompt + text, prompt=prompt,
                                                     sanitize_bot_response=sanitize_bot_response)
                    yield dict(response=response, sources=sources, save_dict=dict(), llm_answers={})
                else:
                    from gradio_utils.grclient import check_job
                    job = gr_client.submit(str(dict(client_kwargs)), api_name=api_name)
                    res_dict = dict(response=text, sources=sources, save_dict=dict(), llm_answers={})
                    text0 = ''
                    tgen0 = time.time()
                    while not job.done():
                        if job.communicator.job.latest_status.code.name == 'FINISHED':
                            break
                        e = check_job(job, timeout=0, raise_exception=False)
                        if e is not None:
                            break
                        outputs_list = job.communicator.job.outputs
                        if outputs_list:
                            res = job.communicator.job.outputs[-1]
                            res_dict = ast.literal_eval(res)
                            text = res_dict['response']
                            sources = res_dict['sources']
                            if gr_prompt_type == 'plain':
                                # then gradio server passes back full prompt + text
                                prompt_and_text = text
                            else:
                                prompt_and_text = prompt + text
                            response = prompter.get_response(prompt_and_text, prompt=prompt,
                                                             sanitize_bot_response=sanitize_bot_response)
                            text_chunk = response[len(text0):]
                            if not text_chunk:
                                # just need some sleep for threads to switch
                                time.sleep(0.001)
                                continue
                            # save old
                            text0 = response
                            yield dict(response=response, sources=sources, save_dict=dict(), llm_answers={})
                            if time.time() - tgen0 > max_time:
                                if verbose:
                                    print("Took too long for Gradio: %s" % (time.time() - tgen0), flush=True)
                                break
                        time.sleep(0.01)
                    # ensure get last output to avoid race
                    res_all = job.outputs()
                    if len(res_all) > 0:
                        # don't raise unless nochat API for now
                        e = check_job(job, timeout=0.02, raise_exception=not chat)
                        if e is not None:
                            strex = ''.join(traceback.format_tb(e.__traceback__))

                        res = res_all[-1]
                        res_dict = ast.literal_eval(res)
                        text = res_dict['response']
                        sources = res_dict['sources']
                    else:
                        # if got no answer at all, probably something bad, always raise exception
                        # UI will still put exception in Chat History under chat exceptions
                        e = check_job(job, timeout=0.3, raise_exception=True)
                        # go with old text if last call didn't work
                        if e is not None:
                            stre = str(e)
                            strex = ''.join(traceback.format_tb(e.__traceback__))
                        else:
                            stre = ''
                            strex = ''

                        print("Bad final response: %s %s %s %s %s: %s %s" % (base_model, inference_server,
                                                                             res_all, prompt, text, stre, strex),
                              flush=True)
                    if gr_prompt_type == 'plain':
                        # then gradio server passes back full prompt + text
                        prompt_and_text = text
                    else:
                        prompt_and_text = prompt + text
                    response = prompter.get_response(prompt_and_text, prompt=prompt,
                                                     sanitize_bot_response=sanitize_bot_response)
                    yield dict(response=response, sources=sources, save_dict=dict(), error=strex, llm_answers={})
            elif hf_client:
                # HF inference server needs control over input tokens
                where_from = "hf_client"
                response = ''
                extra = ''
                sources = ''

                # prompt must include all human-bot like tokens, already added by prompt
                # https://github.com/huggingface/text-generation-inference/tree/main/clients/python#types
                terminate_response = prompter.terminate_response or []
                stop_sequences = list(set(terminate_response + [prompter.PreResponse]))
                stop_sequences = [x for x in stop_sequences if x]
                gen_server_kwargs = dict(do_sample=do_sample,
                                         max_new_tokens=max_new_tokens,
                                         # best_of=None,
                                         repetition_penalty=repetition_penalty,
                                         return_full_text=False,
                                         seed=SEED,
                                         stop_sequences=stop_sequences,
                                         temperature=temperature,
                                         top_k=top_k,
                                         top_p=top_p,
                                         # truncate=False,  # behaves oddly
                                         # typical_p=top_p,
                                         # watermark=False,
                                         # decoder_input_details=False,
                                         )
                # work-around for timeout at constructor time, will be issue if multi-threading,
                # so just do something reasonable or max_time if larger
                # lower bound because client is re-used if multi-threading
                hf_client.timeout = max(300, max_time)
                if not stream_output:
                    text = hf_client.generate(prompt, **gen_server_kwargs).generated_text
                    response = prompter.get_response(prompt + text, prompt=prompt,
                                                     sanitize_bot_response=sanitize_bot_response)
                    yield dict(response=response, sources=sources, save_dict=dict(), llm_answers={})
                else:
                    tgen0 = time.time()
                    text = ""
                    for responses in hf_client.generate_stream(prompt, **gen_server_kwargs):
                        if not responses.token.special:
                            # stop_sequences
                            text_chunk = responses.token.text
                            text += text_chunk
                            response = prompter.get_response(prompt + text, prompt=prompt,
                                                             sanitize_bot_response=sanitize_bot_response)
                            sources = ''
                            yield dict(response=response, sources=sources, save_dict=dict(), llm_answers={})
                        if time.time() - tgen0 > max_time:
                            if verbose:
                                print("Took too long for TGI: %s" % (time.time() - tgen0), flush=True)
                            break
            else:
                raise RuntimeError("Failed to get client: %s" % inference_server)
        else:
            raise RuntimeError("No such inference_server  %s" % inference_server)

        if save_dir and text:
            # save prompt + new text
            extra_dict = gen_server_kwargs.copy()
            extra_dict.update(dict(inference_server=inference_server, num_prompt_tokens=num_prompt_tokens,
                                   t_generate=time.time() - t_generate,
                                   ntokens=None,
                                   tokens_persecond=None,
                                   ))
            save_dict = dict(prompt=prompt, output=text, base_model=base_model, save_dir=save_dir,
                             where_from=where_from, extra_dict=extra_dict)
            yield dict(response=response, sources=sources, save_dict=save_dict, llm_answers={})
        return
    else:
        assert not inference_server, "inference_server=%s not supported" % inference_server

    if isinstance(tokenizer, str):
        # pipeline
        if tokenizer == "summarization":
            key = 'summary_text'
        else:
            raise RuntimeError("No such task type %s" % tokenizer)
        # NOTE: uses max_length only
        sources = ''
        yield dict(response=model(prompt, max_length=max_new_tokens)[0][key], sources=sources, save_dict=dict(),
                   llm_answers={})

    if 'mbart-' in base_model.lower():
        assert src_lang is not None
        tokenizer.src_lang = languages_covered()[src_lang]

    stopping_criteria = get_stopping(prompt_type, prompt_dict, tokenizer, device, base_model,
                                     model_max_length=model_max_length,
                                     prompter=prompter,
                                     truncation_generation=truncation_generation)

    inputs = tokenizer(prompt, return_tensors="pt")
    if debug and len(inputs["input_ids"]) > 0:
        print('input_ids length', len(inputs["input_ids"][0]), flush=True)
    input_ids = inputs["input_ids"].to(device)
    # CRITICAL LIMIT else will fail
    max_max_tokens = int(tokenizer.model_max_length)
    max_input_tokens_default = max(0, int(max_max_tokens - min_new_tokens))
    if max_input_tokens >= 0:
        max_input_tokens = min(max_input_tokens_default, max_input_tokens)
    else:
        max_input_tokens = max_input_tokens_default
    # NOTE: Don't limit up front due to max_new_tokens, let go up to max or reach max_max_tokens in stopping.py
    assert isinstance(max_input_tokens, int), "Bad type for max_input_tokens=%s %s" % (
        max_input_tokens, type(max_input_tokens))
    input_ids = input_ids[:, -max_input_tokens:]
    # required for falcon if multiple threads or asyncio accesses to model during generation
    if use_cache is None:
        use_cache = False if 'falcon' in base_model else True
    if attention_sinks:
        assert use_cache, "attention sinks requires use_cache=True"
    bad_word_ids = [tokenizer.eos_token_id]
    gen_config_kwargs = dict(num_beams=num_beams,
                             do_sample=do_sample,
                             repetition_penalty=float(repetition_penalty),
                             num_return_sequences=num_return_sequences,
                             renormalize_logits=True,
                             remove_invalid_values=True,
                             use_cache=use_cache,
                             max_new_tokens=max_new_tokens,  # unsure if required here
                             )
    if do_sample:
        gen_config_kwargs.update(dict(temperature=float(temperature),
                                      top_p=float(top_p),
                                      top_k=top_k))
    if penalty_alpha > 0:
        gen_config_kwargs.update(dict(penalty_alpha=penalty_alpha))
    if True:
        # unclear impact, some odd things going on inside
        # leads to:
        # The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
        # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
        # or leads to:
        # Using cls_token, but it is not set yet.
        # Using mask_token, but it is not set yet.
        # Using pad_token, but it is not set yet.
        # Using sep_token, but it is not set yet.
        token_ids = ['eos_token_id', 'pad_token_id', 'bos_token_id', 'cls_token_id', 'sep_token_id']
        for token_id in token_ids:
            if hasattr(tokenizer, token_id) and getattr(tokenizer, token_id) is not None:
                gen_config_kwargs.update({token_id: getattr(tokenizer, token_id)})
    generation_config = GenerationConfig(**gen_config_kwargs)

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
        token_ids = ['eos_token_id', 'bos_token_id', 'pad_token_id']
        for token_id in token_ids:
            if hasattr(tokenizer, token_id) and getattr(tokenizer, token_id) is not None:
                gen_kwargs.update({token_id: getattr(tokenizer, token_id)})

    decoder_kwargs = dict(skip_special_tokens=True,
                          clean_up_tokenization_spaces=True)

    decoder = functools.partial(tokenizer.decode,
                                **decoder_kwargs
                                )
    with torch.no_grad():
        have_lora_weights = lora_weights not in [no_lora_str, '', None]
        context_class_cast = NullContext if device == 'cpu' or have_lora_weights or device == 'mps' else torch.autocast
        if t5_type(base_model):
            # issues when casting to float16, can mess up t5 model, e.g. only when not streaming, or other odd behaviors
            context_class_cast = NullContext
        with context_class_cast(device):
            # protection for gradio not keeping track of closed users,
            # else hit bitsandbytes lack of thread safety:
            # https://github.com/h2oai/h2ogpt/issues/104
            # but only makes sense if concurrency_count == 1
            context_class = NullContext  # if concurrency_count > 1 else filelock.FileLock
            if verbose:
                print('Pre-Generate: %s' % str(datetime.now()), flush=True)
            decoded_output = None
            response = ''
            with context_class("generate.lock"):
                if verbose:
                    print('Generate: %s' % str(datetime.now()), flush=True)
                always_use_streaming_method = True  # to deal with complex parsing of prompt vs. generation due to odd tokenizing
                if stream_output or always_use_streaming_method:
                    skip_prompt = True  # True means first output excludes prompt
                    streamer = H2OTextIteratorStreamer(tokenizer, skip_prompt=skip_prompt, block=False,
                                                       **decoder_kwargs)
                    gen_kwargs.update(dict(streamer=streamer))
                    target = wrapped_partial(generate_with_exceptions, model.generate,
                                             raise_generate_gpu_exceptions=raise_generate_gpu_exceptions,
                                             **gen_kwargs)
                    bucket = queue.Queue()
                    thread = EThread(target=target, streamer=streamer, bucket=bucket)
                    thread.start()
                    ret = dict(response='', sources='', save_dict=dict(), llm_answers={})
                    outputs = ""
                    sources = ''
                    tgen0 = time.time()
                    try:
                        for new_text in streamer:
                            if bucket.qsize() > 0 or thread.exc:
                                thread.join()
                            outputs += new_text
                            response = prompter.get_response(outputs, prompt=None,
                                                             only_new_text=True,
                                                             sanitize_bot_response=sanitize_bot_response)
                            ret = dict(response=response, sources=sources, save_dict=dict(), llm_answers={})
                            if stream_output:
                                yield ret
                            if time.time() - tgen0 > max_time:
                                if verbose:
                                    print("Took too long for Torch: %s" % (time.time() - tgen0), flush=True)
                                break
                        # yield if anything left over as can happen (FIXME: Understand better)
                        yield ret
                    except BaseException:
                        # if any exception, raise that exception if was from thread, first
                        if thread.exc:
                            raise thread.exc
                        raise
                    finally:
                        # don't clear torch cache here, delays multi-generation, and bot(), all_bot(), and evaluate_nochat() do it
                        # in case no exception and didn't join with thread yet, then join
                        if not thread.exc:
                            thread.join()
                    # in case raise StopIteration or broke queue loop in streamer, but still have exception
                    if thread.exc:
                        raise thread.exc
                    decoded_output = outputs
                    ntokens = len(outputs) // 4  # hack for now
                else:
                    # below length removal doesn't work in general, because encoding does not match internal of model generation
                    input_ids_len = gen_kwargs['input_ids'][0].shape[0]
                    try:
                        outputs = model.generate(**gen_kwargs)
                    finally:
                        pass
                        # don't clear torch cache here, delays multi-generation, and bot(), all_bot(), and evaluate_nochat() do it
                    # skip first IDs
                    ntokens = sum([len(s) - input_ids_len for s in outputs.sequences]) if save_dir else -1
                    outputs = [decoder(s[input_ids_len:]) for s in outputs.sequences]
                    sources = ''
                    response = prompter.get_response(outputs, prompt=None,
                                                     only_new_text=True,
                                                     sanitize_bot_response=sanitize_bot_response)
                    yield dict(response=response, sources=sources, save_dict=dict(), llm_answers={})
                    if outputs and len(outputs) >= 1:
                        decoded_output = prompt + outputs[0]
                if save_dir and decoded_output:
                    extra_dict = gen_config_kwargs.copy()
                    extra_dict.update(dict(num_prompt_tokens=num_prompt_tokens,
                                           t_generate=time.time() - t_generate,
                                           ntokens=ntokens,
                                           tokens_persecond=ntokens / (time.time() - t_generate),
                                           ))
                    save_dict = dict(prompt=prompt, output=decoded_output, base_model=base_model, save_dir=save_dir,
                                     where_from="evaluate_%s" % str(stream_output),
                                     extra_dict=extra_dict)
                    yield dict(response=response, sources=sources, save_dict=save_dict, llm_answers={})
            if verbose:
                print('Post-Generate: %s decoded_output: %s' % (
                    str(datetime.now()), len(decoded_output) if decoded_output else -1), flush=True)


inputs_list_names = list(inspect.signature(evaluate).parameters)
state_names = input_args_list.copy()  # doesn't have to be the same, but state_names must match evaluate() and how filled then
inputs_kwargs_list = [x for x in inputs_list_names if x not in eval_func_param_names + state_names]


def get_cutoffs(memory_restriction_level, for_context=False, model_max_length=2048, min_max_new_tokens=256):
    # help to avoid errors like:
    # RuntimeError: The size of tensor a (2048) must match the size of tensor b (2049) at non-singleton dimension 3
    # RuntimeError: expected scalar type Half but found Float
    # with - 256
    if memory_restriction_level > 0:
        max_length_tokenize = 768 - 256 if memory_restriction_level <= 2 else 512 - 256
    else:
        # at least give room for 1 paragraph output
        max_length_tokenize = model_max_length - min_max_new_tokens
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
                    self.clear_queue()
                    self.do_stop = False
                    raise StopIteration()
                    # break
                value = self.text_queue.get(block=self.block, timeout=self.timeout)
                break
            except queue.Empty:
                time.sleep(0.01)
        if value == self.stop_signal:
            self.clear_queue()
            self.do_stop = False
            raise StopIteration()
        else:
            return value

    def clear_queue(self):
        # make sure streamer is reusable after stop hit
        with self.text_queue.mutex:
            self.text_queue.queue.clear()

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        # same as base class, except remove hack w.r.t. text.rfind(" ") that ruins LLaMa2
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len:]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len:]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        elif len(text) > 0 and text[-1] == '�':
            printable_text = text[self.print_len: text.rfind(" ") + 1]
            self.print_len += len(printable_text)
        else:
            printable_text = text[self.print_len:]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)


def generate_with_exceptions(func, *args, raise_generate_gpu_exceptions=True, **kwargs):
    try:
        func(*args, **kwargs)
    except torch.cuda.OutOfMemoryError as e:
        print("GPU OOM 2: exception: %s" % str(e),
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
                "GPU Error: exception: %s" % str(e),
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


def get_generate_params(model_lower,
                        chat,
                        stream_output, show_examples,
                        prompt_type, prompt_dict,
                        system_prompt,
                        pre_prompt_query, prompt_query,
                        pre_prompt_summary, prompt_summary,
                        temperature, top_p, top_k, penalty_alpha, num_beams,
                        max_new_tokens, min_new_tokens, early_stopping, max_time,
                        repetition_penalty, num_return_sequences,
                        do_sample,
                        top_k_docs, chunk, chunk_size,
                        image_loaders,
                        pdf_loaders,
                        url_loaders,
                        jq_schema,
                        docs_ordering_type,
                        min_max_new_tokens,
                        max_input_tokens,
                        max_total_input_tokens,
                        docs_token_handling,
                        docs_joiner,
                        hyde_level,
                        hyde_template,
                        doc_json_mode,
                        verbose,
                        ):
    use_defaults = False
    use_default_examples = True
    examples = []
    task_info = 'LLM'
    if model_lower:
        print(f"Using Model {model_lower}", flush=True)
    else:
        if verbose:
            print("No model defined yet", flush=True)

    min_new_tokens = min_new_tokens if min_new_tokens is not None else 0
    early_stopping = early_stopping if early_stopping is not None else False
    max_time_defaults = 60 * 10
    max_time = max_time if max_time is not None else max_time_defaults

    if not prompt_type and model_lower in inv_prompt_type_to_model_lower and prompt_type != 'custom':
        prompt_type = inv_prompt_type_to_model_lower[model_lower]
        if verbose:
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

    use_placeholder_instruction_as_example = False
    if 'bart-large-cnn-samsum' in model_lower or 'flan-t5-base-samsum' in model_lower:
        placeholder_instruction = summarize_example1
        placeholder_input = ""
        use_defaults = True
        use_default_examples = False
        use_placeholder_instruction_as_example = True
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
        use_placeholder_instruction_as_example = True
    elif 'gpt2' in model_lower:
        placeholder_instruction = "The sky is"
        placeholder_input = ""
        prompt_type = prompt_type or 'plain'
        use_default_examples = True  # some will be odd "continuations" but can be ok
        use_placeholder_instruction_as_example = True
        task_info = "Auto-complete phrase, code, etc."
        use_defaults = True
    else:
        if chat:
            placeholder_instruction = ""
        else:
            placeholder_instruction = "Give detailed answer for whether Einstein or Newton is smarter."
        placeholder_input = ""
        if not prompt_type and model_lower in inv_prompt_type_to_model_lower and prompt_type != 'custom':
            prompt_type = inv_prompt_type_to_model_lower[model_lower]
        elif model_lower:
            # default is plain, because might rely upon trust_remote_code to handle prompting
            prompt_type = prompt_type or 'plain'
        else:
            prompt_type = ''
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
        penalty_alpha = 0 if penalty_alpha is None else penalty_alpha
        num_beams = num_beams or 1
        max_new_tokens = max_new_tokens or 512
        repetition_penalty = repetition_penalty or 1.07
        num_return_sequences = min(num_beams, num_return_sequences or 1)
        do_sample = False if do_sample is None else do_sample
    else:
        temperature = 0.1 if temperature is None else temperature
        top_p = 0.75 if top_p is None else top_p
        top_k = 40 if top_k is None else top_k
        penalty_alpha = 0 if penalty_alpha is None else penalty_alpha
        num_beams = num_beams or 1
        max_new_tokens = max_new_tokens or 1024
        repetition_penalty = repetition_penalty or 1.07
        num_return_sequences = min(num_beams, num_return_sequences or 1)
        do_sample = False if do_sample is None else do_sample
    # doesn't include chat, instruction_nochat, iinput_nochat, added later
    params_list = ["",
                   stream_output,
                   prompt_type, prompt_dict,
                   temperature, top_p, top_k, penalty_alpha, num_beams,
                   max_new_tokens, min_new_tokens,
                   early_stopping, max_time, repetition_penalty, num_return_sequences, do_sample]

    if use_placeholder_instruction_as_example:
        examples += [[placeholder_instruction, ''] + params_list]

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
    # add summary example
    examples += [
        [summarize_example1, 'Summarize' if prompt_type not in ['plain', 'instruct_simple'] else ''] + params_list]

    src_lang = "English"
    tgt_lang = "Russian"

    # move to correct position
    for example in examples:
        example += [chat, '', '', LangChainMode.DISABLED.value, True,
                    LangChainAction.QUERY.value, [],
                    top_k_docs, chunk, chunk_size, DocumentSubset.Relevant.name, [],
                    pre_prompt_query, prompt_query,
                    pre_prompt_summary, prompt_summary,
                    system_prompt,
                    image_loaders,
                    pdf_loaders,
                    url_loaders,
                    jq_schema,
                    None,
                    None,
                    False,
                    None,
                    None,
                    docs_ordering_type,
                    min_max_new_tokens,
                    max_input_tokens,
                    max_total_input_tokens,
                    docs_token_handling,
                    docs_joiner,
                    hyde_level,
                    hyde_template,
                    doc_json_mode,
                    ]
        # adjust examples if non-chat mode
        if not chat:
            example[eval_func_param_names.index('instruction_nochat')] = example[
                eval_func_param_names.index('instruction')]
            example[eval_func_param_names.index('instruction')] = ''

            example[eval_func_param_names.index('iinput_nochat')] = example[eval_func_param_names.index('iinput')]
            example[eval_func_param_names.index('iinput')] = ''
        assert len(example) == len(eval_func_param_names), "Wrong example: %s %s" % (
            len(example), len(eval_func_param_names))

    if prompt_type == PromptType.custom.name and not prompt_dict:
        raise ValueError("Unexpected to get non-empty prompt_dict=%s for prompt_type=%s" % (prompt_dict, prompt_type))

    # get prompt_dict from prompt_type, so user can see in UI etc., or for custom do nothing except check format
    prompt_dict, error0 = get_prompt(prompt_type, prompt_dict,
                                     chat=False, context='', reduced=False, making_context=False, return_dict=True,
                                     system_prompt=system_prompt)
    if error0:
        raise RuntimeError("Prompt wrong: %s" % error0)

    return placeholder_instruction, placeholder_input, \
        stream_output, show_examples, \
        prompt_type, prompt_dict, \
        temperature, top_p, top_k, penalty_alpha, num_beams, \
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


def score_qa(smodel, stokenizer, max_length_tokenize, question, answer, cutoff_len):
    question = question[-cutoff_len:]
    answer = answer[-cutoff_len:]

    inputs = stokenizer(question, answer,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length_tokenize).to(smodel.device)
    try:
        score = torch.sigmoid(smodel(**inputs.to(smodel.device)).logits[0].float()).cpu().detach().numpy()[0]
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
                'cublasLt ran into an error!' in str(e) or \
                'device-side assert triggered' in str(e):
            print("GPU Error: question: %s answer: %s exception: %s" % (question, answer, str(e)),
                  flush=True)
            traceback.print_exc()
            clear_torch_cache()
            return 'Response Score: GPU Error'
        else:
            raise
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    return score


def check_locals(**kwargs):
    # ensure everything in evaluate is here
    can_skip_because_locally_generated = no_default_param_names + [
        # get_model:
        'reward_type'
    ]
    for k in eval_func_param_names:
        if k in can_skip_because_locally_generated:
            continue
        assert k in kwargs, "Missing %s" % k
    for k in inputs_kwargs_list:
        if k in can_skip_because_locally_generated:
            continue
        assert k in kwargs, "Missing %s" % k

    for k in list(inspect.signature(get_model).parameters):
        if k in can_skip_because_locally_generated:
            continue
        assert k in kwargs, "Missing %s" % k


def get_model_max_length(model_state):
    if not isinstance(model_state['tokenizer'], (str, type(None))):
        return model_state['tokenizer'].model_max_length
    else:
        return 2048


def get_model_max_length_from_tokenizer(tokenizer):
    if hasattr(tokenizer, 'model_max_length'):
        return int(tokenizer.model_max_length)
    else:
        return 2048


def get_max_max_new_tokens(model_state, **kwargs):
    if not isinstance(model_state['tokenizer'], (str, type(None))) or not kwargs.get('truncation_generation', False):
        if hasattr(model_state['tokenizer'], 'model_max_length'):
            max_max_new_tokens = model_state['tokenizer'].model_max_length
        else:
            # e.g. fast up, no model
            max_max_new_tokens = None
    else:
        max_max_new_tokens = None

    if kwargs['max_max_new_tokens'] is not None and max_max_new_tokens is not None:
        return min(max_max_new_tokens, kwargs['max_max_new_tokens'])
    elif kwargs['max_max_new_tokens'] is not None:
        return kwargs['max_max_new_tokens']
    elif kwargs['memory_restriction_level'] == 1:
        return 768
    elif kwargs['memory_restriction_level'] == 2:
        return 512
    elif kwargs['memory_restriction_level'] >= 3:
        return 256
    else:
        # FIXME: Need to update after new model loaded, so user can control with slider
        return 2048


def get_minmax_top_k_docs(is_public, from_ui):
    label_top_k_docs = "Number of document chunks (query) or pages/parts (summarize)"
    if is_public:
        min_top_k_docs = 1
        if from_ui:
            max_top_k_docs = max_top_k_docs_public
        else:
            max_top_k_docs = max_top_k_docs_public_api
    else:
        min_top_k_docs = -1
        max_top_k_docs = 1000
        label_top_k_docs = label_top_k_docs + " (-1 = auto fill model context, all pages/docs for summarize)"
    return min_top_k_docs, max_top_k_docs, label_top_k_docs


def merge_chat_conversation_history(chat_conversation1, history):
    # chat_conversation and history ordered so largest index of list is most recent
    if chat_conversation1:
        chat_conversation1 = str_to_list(chat_conversation1)
        for conv1 in chat_conversation1:
            assert isinstance(conv1, (list, tuple))
            assert len(conv1) == 2

    if isinstance(history, list):
        # make copy so only local change
        if chat_conversation1:
            # so priority will be newest that comes from actual chat history from UI, then chat_conversation
            history = chat_conversation1 + history.copy()
    elif chat_conversation1:
        history = chat_conversation1
    else:
        history = []
    return history


def history_to_context(history, langchain_mode=None,
                       add_chat_history_to_context=None,
                       prompt_type=None, prompt_dict=None, chat=None, model_max_length=None,
                       memory_restriction_level=None, keep_sources_in_context=None,
                       system_prompt=None, chat_conversation=None,
                       min_max_new_tokens=256):
    """
    consumes all history up to (but not including) latest history item that is presumed to be an [instruction, None] pair
    :param history:
    :param langchain_mode:
    :param add_chat_history_to_context:
    :param prompt_type:
    :param prompt_dict:
    :param chat:
    :param model_max_length:
    :param memory_restriction_level:
    :param keep_sources_in_context:
    :param system_prompt:
    :param chat_conversation:
    :param min_max_new_tokens:
    :return:
    """
    history = merge_chat_conversation_history(chat_conversation, history)

    if len(history) >= 1 and len(history[-1]) >= 2 and not history[-1][1]:
        len_history = len(history) - 1
    else:
        # full history
        len_history = len(history)

    # ensure output will be unique to models
    _, _, _, max_prompt_length = get_cutoffs(memory_restriction_level,
                                             for_context=True, model_max_length=model_max_length,
                                             min_max_new_tokens=min_max_new_tokens)
    context1 = ''
    if max_prompt_length is not None and add_chat_history_to_context:
        context1 = ''
        # - 1 below because current instruction already in history from user()
        for histi in range(0, len_history):
            data_point = dict(instruction=history[histi][0], input='', output=history[histi][1])
            prompt, pre_response, terminate_response, chat_sep, chat_turn_sep = \
                generate_prompt(data_point,
                                prompt_type,
                                prompt_dict,
                                chat,
                                reduced=True,
                                making_context=True,
                                system_prompt=system_prompt,
                                histi=histi)
            # md -> back to text, maybe not super important if model trained enough
            if not keep_sources_in_context and langchain_mode != 'Disabled' and prompt.find(super_source_prefix) >= 0:
                # FIXME: This is relatively slow even for small amount of text, like 0.3s each history item
                import re
                prompt = re.sub(f'{re.escape(super_source_prefix)}.*?{re.escape(super_source_postfix)}', '', prompt,
                                flags=re.DOTALL)
                if prompt.endswith('\n<p>'):
                    prompt = prompt[:-4]
            prompt = prompt.replace('<br>', chat_turn_sep)
            if not prompt.endswith(chat_turn_sep):
                prompt += chat_turn_sep
            # most recent first, add older if can
            # only include desired chat history
            if len(prompt + context1) > max_prompt_length:
                break
            context1 += prompt

        _, pre_response, terminate_response, chat_sep, chat_turn_sep = \
            generate_prompt({}, prompt_type, prompt_dict,
                            chat, reduced=True,
                            making_context=True,
                            system_prompt=system_prompt,
                            histi=-1)
        if context1 and not context1.endswith(chat_turn_sep):
            context1 += chat_turn_sep  # ensure if terminates abruptly, then human continues on next line
    return context1


def get_relaxed_max_new_tokens(prompt, tokenizer=None, max_new_tokens=None, max_new_tokens0=None):
    # check if can relax max_new_tokens for this specific prompt
    if max_new_tokens0 is not None and \
            hasattr(tokenizer, 'model_max_len') and \
            isinstance(tokenizer.model_max_len, (float, int)):
        max_new_tokens = int(tokenizer.model_max_length) - get_token_count(prompt, tokenizer)
        if max_new_tokens is not None:
            return min(max_new_tokens0, max_new_tokens)
        else:
            return max_new_tokens0
    return max_new_tokens


def get_limited_prompt(instruction,
                       iinput,
                       tokenizer,
                       estimated_instruction=None,
                       prompter=None,
                       inference_server=None,
                       prompt_type=None, prompt_dict=None, chat=False, max_new_tokens=None,
                       system_prompt='',
                       context='', chat_conversation=None, text_context_list=None,
                       keep_sources_in_context=False,
                       model_max_length=None, memory_restriction_level=0,
                       langchain_mode=None, add_chat_history_to_context=True,
                       verbose=False,
                       doc_importance=0.5,
                       min_max_new_tokens=256,
                       max_input_tokens=-1,
                       max_total_input_tokens=-1,
                       truncation_generation=False,
                       gradio_server=False,
                       ):
    if gradio_server or not inference_server:
        # can listen to truncation_generation
        pass
    else:
        # these don't support allowing going beyond total context
        truncation_generation = True

    # for templates, use estimated for counting, but adjust instruction as output
    if estimated_instruction is None:
        estimated_instruction = instruction

    if max_input_tokens >= 0:
        # max_input_tokens is used to runtime (via client/UI) to control actual filling of context
        max_input_tokens = min(model_max_length - min_max_new_tokens, max_input_tokens)
    else:
        max_input_tokens = model_max_length - min_max_new_tokens

    if prompter:
        prompt_type = prompter.prompt_type
        prompt_dict = prompter.prompt_dict
        chat = prompter.chat
        stream_output = prompter.stream_output
        system_prompt = prompter.system_prompt

    generate_prompt_type = prompt_type
    external_handle_chat_conversation = False
    if inference_server and any(
            inference_server.startswith(x) for x in ['openai_chat', 'openai_azure_chat', 'vllm_chat']):
        # Chat APIs do not take prompting
        # Replicate does not need prompting if no chat history, but in general can take prompting
        # if using prompter, prompter.system_prompt will already be filled with automatic (e.g. from llama-2),
        # so if replicate final prompt with system prompt still correct because only access prompter.system_prompt that was already set
        # below already true for openai,
        # but not vllm by default as that can be any model and handled by FastChat API inside vLLM itself
        generate_prompt_type = 'plain'
        # Chat APIs don't handle chat history via single prompt, but in messages, assumed to be handled outside this function
        chat_conversation = []
        external_handle_chat_conversation = True

    # merge handles if chat_conversation is None
    history = []
    history = merge_chat_conversation_history(chat_conversation, history)
    history_to_context_func = functools.partial(history_to_context,
                                                langchain_mode=langchain_mode,
                                                add_chat_history_to_context=add_chat_history_to_context,
                                                prompt_type=generate_prompt_type,
                                                prompt_dict=prompt_dict,
                                                chat=chat,
                                                model_max_length=max_input_tokens,
                                                memory_restriction_level=memory_restriction_level,
                                                keep_sources_in_context=keep_sources_in_context,
                                                system_prompt=system_prompt,
                                                min_max_new_tokens=min_max_new_tokens)
    context2 = history_to_context_func(history)
    context1 = context
    if context1 is None:
        context1 = ''

    # get how many more tokens in templated instruction, somewhat of estimate at fine level
    num_instruction_tokens = get_token_count(instruction, tokenizer)
    num_estimated_instruction_tokens = get_token_count(estimated_instruction, tokenizer)
    delta_instruction = max(0, num_estimated_instruction_tokens - num_instruction_tokens)

    # get estimated templated instruction tokens for counting purposes
    from h2oai_pipeline import H2OTextGenerationPipeline
    estimated_instruction, num_estimated_instruction_tokens = H2OTextGenerationPipeline.limit_prompt(
        estimated_instruction, tokenizer,
        max_prompt_length=max_input_tokens)
    data_point_just_instruction = dict(context='', instruction=estimated_instruction, input='')
    prompt_just_estimated_instruction = prompter.generate_prompt(data_point_just_instruction)
    num_instruction_tokens = get_token_count(prompt_just_estimated_instruction, tokenizer)

    # get actual instruction, limited by template limitation
    instruction, _ = H2OTextGenerationPipeline.limit_prompt(instruction, tokenizer,
                                                            max_prompt_length=max_input_tokens - delta_instruction)

    context1, num_context1_tokens = H2OTextGenerationPipeline.limit_prompt(context1, tokenizer,
                                                                           max_prompt_length=max_input_tokens)
    context2, num_context2_tokens = H2OTextGenerationPipeline.limit_prompt(context2, tokenizer,
                                                                           max_prompt_length=max_input_tokens)
    iinput, num_iinput_tokens = H2OTextGenerationPipeline.limit_prompt(iinput, tokenizer,
                                                                       max_prompt_length=max_input_tokens)
    if text_context_list is None:
        text_context_list = []
    num_doc_tokens = sum([get_token_count(x + docs_joiner_default, tokenizer) for x in text_context_list])

    num_prompt_tokens0 = (num_instruction_tokens or 0) + \
                         (num_context1_tokens or 0) + \
                         (num_context2_tokens or 0) + \
                         (num_iinput_tokens or 0) + \
                         (num_doc_tokens or 0)

    # go down to no less than 256, about 1 paragraph
    # use max_new_tokens before use num_prompt_tokens0 else would be negative or ~0
    min_max_new_tokens = min(min_max_new_tokens, max_new_tokens)
    # by default assume can handle all chat and docs
    chat_index = 0

    # allowed residual is either half of what is allowed if doc exceeds half, or is rest of what doc didn't consume
    num_non_doc_tokens = num_prompt_tokens0 - num_doc_tokens
    # to doc first then non-doc, shouldn't matter much either way
    doc_max_length = max(max_input_tokens - num_non_doc_tokens, int(doc_importance * max_input_tokens))
    top_k_docs, one_doc_size, num_doc_tokens = get_docs_tokens(tokenizer, text_context_list=text_context_list,
                                                               max_input_tokens=doc_max_length)
    non_doc_max_length = max(max_input_tokens - num_doc_tokens, int((1.0 - doc_importance) * max_input_tokens))

    if num_non_doc_tokens > non_doc_max_length:
        # need to limit in some way, keep portion of history but all of context and instruction
        # 1) drop iinput (unusual to include anyways)
        # 2) reduce history
        # 3) reduce context1
        # 4) limit instruction so will fit
        diff1 = non_doc_max_length - (
                num_instruction_tokens + num_context1_tokens + num_context2_tokens)
        diff2 = non_doc_max_length - (num_instruction_tokens + num_context1_tokens)
        diff3 = non_doc_max_length - num_instruction_tokens
        diff4 = non_doc_max_length
        if diff1 > 0:
            # then should be able to do #1
            iinput = ''
            num_iinput_tokens = 0
        elif diff2 > 0 > diff1:
            # then may be able to do #1 + #2
            iinput = ''
            num_iinput_tokens = 0
            chat_index_final = len(history)
            for chat_index in range(len(history)):
                # NOTE: history and chat_conversation are older for first entries
                # FIXME: This is a slow for many short conversations
                context2 = history_to_context_func(history[chat_index:])
                num_context2_tokens = get_token_count(context2, tokenizer)
                diff1 = non_doc_max_length - (
                        num_instruction_tokens + num_context1_tokens + num_context2_tokens)
                if diff1 > 0:
                    chat_index_final = chat_index
                    if verbose:
                        print("chat_conversation used %d out of %d" % (chat_index, len(history)), flush=True)
                    break
            chat_index = chat_index_final  # i.e. if chat_index == len(history), then nothing can be consumed
        elif diff3 > 0 > diff2:
            # then may be able to do #1 + #2 + #3
            iinput = ''
            num_iinput_tokens = 0
            context2 = ''
            num_context2_tokens = 0
            context1, num_context1_tokens = H2OTextGenerationPipeline.limit_prompt(context1, tokenizer,
                                                                                   max_prompt_length=diff3)
            if num_context1_tokens <= diff3:
                pass
            else:
                print("failed to reduce", flush=True)
        else:
            # then must be able to do #1 + #2 + #3 + #4
            iinput = ''
            num_iinput_tokens = 0
            context2 = ''
            num_context2_tokens = 0
            context1 = ''
            num_context1_tokens = 0
            # diff4 accounts for real prompting for instruction
            # FIXME: history_to_context could include instruction, in case system prompt long, we overcount and could have more free tokens

            max_prompt_length = max(0, diff4 - delta_instruction)
            instruction, _ = H2OTextGenerationPipeline.limit_prompt(instruction, tokenizer,
                                                                    max_prompt_length=max_prompt_length)
            # get actual instruction tokens
            data_point_just_instruction = dict(context='', instruction=instruction, input='')
            prompt_just_instruction = prompter.generate_prompt(data_point_just_instruction)
            num_instruction_tokens = get_token_count(prompt_just_instruction, tokenizer) + delta_instruction

    # update full context
    context = context1 + context2
    # update token counts (docs + non-docs, all tokens)
    num_prompt_tokens = (num_instruction_tokens or 0) + \
                        (num_context1_tokens or 0) + \
                        (num_context2_tokens or 0) + \
                        (num_iinput_tokens or 0) + \
                        (num_doc_tokens or 0)

    # update max_new_tokens
    # limit so max_new_tokens = prompt + new < max
    # otherwise model can fail etc. e.g. for distilgpt2 asking for 1024 tokens is enough to fail if prompt=1 token
    if truncation_generation:
        max_new_tokens = min(max_new_tokens, model_max_length - num_prompt_tokens)

        if os.getenv('HARD_ASSERTS'):
            if max_new_tokens < min_max_new_tokens:
                raise ValueError("Invalid max_new_tokens=%s" % max_new_tokens)

    if prompter is None:
        # get prompter
        debug = False
        stream_output = False  # doesn't matter
        prompter = Prompter(prompt_type, prompt_dict, debug=debug, chat=chat, stream_output=stream_output,
                            system_prompt=system_prompt)
        if prompt_type != generate_prompt_type:
            # override just this attribute, keep system_prompt etc. from original prompt_type
            prompter.prompt_type = generate_prompt_type

    data_point = dict(context=context, instruction=instruction, input=iinput)
    # handle promptA/promptB addition if really from history.
    # if not from history, then reduced=False inside correct
    # if mixed, then no specific correct thing to do, so treat like history and promptA/B will come first still
    context_from_history = len(history) > 0 and len(context1) > 0
    prompt = prompter.generate_prompt(data_point, context_from_history=context_from_history)
    num_prompt_tokens_actual = get_token_count(prompt, tokenizer)

    return prompt, \
        instruction, iinput, context, \
        num_prompt_tokens, max_new_tokens, num_prompt_tokens0, num_prompt_tokens_actual, \
        chat_index, external_handle_chat_conversation, \
        top_k_docs, one_doc_size, truncation_generation


def get_docs_tokens(tokenizer, text_context_list=[], max_input_tokens=None):
    """
    max_input_tokens: Over all LLM calls, upper limit of total token count,
                      or single LLM call if want to know what docs fit into single call
    """
    if text_context_list is None or len(text_context_list) == 0:
        return 0, None, 0
    assert max_input_tokens is not None, "Must set max_input_tokens"
    tokens = [get_token_count(x + docs_joiner_default, tokenizer) for x in text_context_list]
    tokens_cumsum = np.cumsum(tokens)
    where_res = np.where(tokens_cumsum < max_input_tokens)[0]
    # if below condition fails, then keep top_k_docs=-1 and trigger special handling next
    if where_res.shape[0] > 0:
        top_k_docs = 1 + where_res[-1]
        one_doc_size = None
        num_doc_tokens = tokens_cumsum[top_k_docs - 1]  # by index
    else:
        # if here, means 0 and just do best with 1 doc
        top_k_docs = 1
        text_context_list = text_context_list[:top_k_docs]
        # critical protection
        from src.h2oai_pipeline import H2OTextGenerationPipeline
        doc_content = text_context_list[0]
        doc_content, new_tokens0 = H2OTextGenerationPipeline.limit_prompt(doc_content,
                                                                          tokenizer,
                                                                          max_prompt_length=max_input_tokens)
        text_context_list[0] = doc_content
        one_doc_size = len(doc_content)
        num_doc_tokens = get_token_count(doc_content + docs_joiner_default, tokenizer)
        print("Unexpected large chunks and can't add to context, will add 1 anyways.  Tokens %s -> %s" % (
            tokens[0], new_tokens0), flush=True)
    return top_k_docs, one_doc_size, num_doc_tokens


def entrypoint_main():
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

    must have 4*48GB GPU and run without 8bit in order for sharding to work with use_gpu_id=False
    can also pass --prompt_type='human_bot' and model can somewhat handle instructions without being instruct tuned
    python generate.py --base_model=decapoda-research/llama-65b-hf --load_8bit=False --use_gpu_id=False --prompt_type='human_bot'

    python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b
    """
    H2O_Fire(main)


if __name__ == "__main__":
    entrypoint_main()
