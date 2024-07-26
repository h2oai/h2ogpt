import ast
import copy
import json
import os
import sys
import time
import traceback
import typing
from functools import lru_cache
from typing import Union

import httpx
import pydantic_core
import requests
from requests import ConnectTimeout, JSONDecodeError
from urllib3.exceptions import ConnectTimeoutError, MaxRetryError, ConnectionError
from requests.exceptions import ConnectionError as ConnectionError2
from requests.exceptions import ReadTimeout as ReadTimeout2

import torch
from transformers import AutoModel, AutoTokenizer

from enums import is_gradio_vision_model, anthropic_mapping, groq_mapping, google_mapping, mistralai_mapping, \
    model_token_mapping, model_token_mapping_outputs, anthropic_mapping_outputs, google_mapping_outputs, \
    mistralai_mapping_outputs, groq_mapping_outputs, model_state_none0, other_model_state_defaults0, \
    is_json_model, is_vision_model, images_num_max_dict, llamacpp_inner_dict_keys, unknown_prompt_type
from evaluate_params import eval_func_param_names
from prompter import anthropic_gpts, openai_gpts, google_gpts, mistralai_gpts, groq_gpts, non_hf_types, \
    prompt_type_to_model_name, get_prompt, model_name_to_prompt_type
from src.prompter_utils import has_chat_template, get_chat_template, base64_decode_jinja_template
from utils import url_alive, cuda_vis_check, get_hf_server, is_gradio_version4, clear_torch_cache, set_openai, \
    FakeTokenizer, get_device, NullContext, get_kwargs, is_json_vllm

from loaders import get_loaders


def switch_a_roo_llama(base_model, model_path_llama, load_gptq, load_awq, n_gqa, llamacpp_path):
    # from TheBloke HF link
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
            download_postfix = '?download=true'
            base_model0 = 'https://huggingface.co/%s/resolve/main/%s.Q5_K_M%s%s' % (
                base_model, lower_model, file_postfix, download_postfix)
            if url_alive(base_model0):
                base_model = base_model0
        model_path_llama = base_model
        base_model = 'llama'
    elif (base_model.lower().startswith('https://huggingface.co/TheBloke'.lower()) or
          base_model.lower().startswith('http://huggingface.co/TheBloke'.lower())) \
            and (is_gguf or is_ggml) and len(model_split) == 2:
        # auto-switch-a-roo to support GGUF/GGML put into base model in UI
        just_model_split = model_split[1].split(postfix)
        if postfix.lower() in base_model.lower() and \
                file_postfix not in base_model and \
                len(just_model_split) == 2:
            just_model = just_model_split[0]
            lower_model = just_model.lower()
            download_postfix = '?download=true'
            base_model0 = '%s/resolve/main/%s.Q5_K_M%s%s' % (
                base_model, lower_model, file_postfix, download_postfix)
            if url_alive(base_model0):
                base_model = base_model0
        model_path_llama = base_model
        base_model = 'llama'
    elif base_model.endswith('.gguf') or base_model.endswith('.ggml') or base_model.endswith(
            '.gguf?download=true') or base_model.endswith('.ggml?download=true'):
        # from resolved url
        if base_model.lower().startswith(
                'https://huggingface.co/') and 'resolve/main/' in base_model.lower() and url_alive(base_model):
            model_path_llama = base_model
            base_model = 'llama'
        # from file
        elif os.path.isfile(base_model):
            # then file but still either gguf or ggml
            model_path_llama = base_model
            base_model = 'llama'
        elif os.path.isfile(os.path.join(llamacpp_path, base_model)):
            # then file but still either gguf or ggml
            model_path_llama = os.path.join(llamacpp_path, base_model)
            base_model = 'llama'

    # some auto things for TheBloke models:
    if 'TheBloke' in base_model and '-GPTQ' in base_model:
        load_gptq = load_gptq or 'model'
    elif 'TheBloke' in base_model and '-AWQ' in base_model:
        load_awq = load_awq or 'model'
    elif model_path_llama and '2-70B-GGUF' in model_path_llama:
        n_gqa = n_gqa or 8
    if not model_path_llama:
        model_path_llama = ''

    return base_model, model_path_llama, load_gptq, load_awq, n_gqa


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
            if base_model in anthropic_gpts + openai_gpts + google_gpts + mistralai_gpts + groq_gpts + non_hf_types:
                return None, None, max_seq_len
            if 'not a local folder and is not a valid model identifier listed on' in str(
                    e) or '404 Client Error' in str(e) or "couldn't connect" in str(e) or \
                    'OSError: You are trying to access a gated repo.' in str(e) or \
                    'Repository Not Found for url' in str(e) or \
                    'does not appear to have a file' in str(e) or \
                    'ncorrect path_or_model_id' in str(e):
                # e.g. llama, gpjt, etc.
                # e.g. HF TGI but not model on HF or private etc.
                if max_seq_len is None and base_model.lower() in non_hf_types:
                    max_seq_len = 4096
                    print(f"Could not determine --max_seq_len, setting to {max_seq_len}.  Pass if not correct",
                          flush=True)
                # HF TGI server only should really require prompt_type, not HF model state
                print("Not using tokenizer from HuggingFace:\n\n", flush=True)
                traceback.print_exc()
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
            try:
                model = AutoModel.from_config(
                    config,
                    trust_remote_code=trust_remote_code,
                )
            except Exception as e:
                if 'has no attribute' in str(e):
                    # half-baked hack to transformers by Cohere
                    model = None
                else:
                    raise
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
        elif hasattr(config, 'text_config') and hasattr(config.text_config, 'max_position_embeddings') and isinstance(
                config.text_config.max_position_embeddings, int):
            # help automatically limit inputs to generate
            if 'idefics' in base_model:
                # max_seq_len = 8192
                max_seq_len = 4096  # safer
            else:
                max_seq_len = config.text_config.max_position_embeddings
            if verbose:
                print("Used max_position_embeddings=%s as base model (pre-rope) max_seq_len."
                      "  If not desired, pass --max_seq_len and set to some integer value." % config.text_config.max_position_embeddings,
                      flush=True)
        elif hasattr(config, 'n_ctx'):
            # e.g. gpt2
            max_seq_len = int(config.n_ctx)
        else:
            max_seq_len = 4096
            print(f"Could not determine --max_seq_len, setting to {max_seq_len}.  Pass if not correct", flush=True)

        # listen to model if sets this and user passed nothing
        if not rope_scaling and hasattr(config, 'rope_scaling'):
            rope_scaling = config.rope_scaling

        if rope_scaling:
            set_by_rope = False
            if rope_scaling.get('factor') and rope_scaling.get('original_max_position_embeddings') and \
                    hasattr(config, 'max_position_embeddings') and \
                    isinstance(config.max_position_embeddings, int):
                # HF transformers new way
                max_seq_len = config.max_position_embeddings
                set_by_rope = True
            elif rope_scaling.get('factor') and hasattr(config, 'max_position_embeddings') and \
                    isinstance(config.max_position_embeddings, int):
                # HF transformers old way
                max_seq_len = config.max_position_embeddings * rope_scaling.get('factor')
                set_by_rope = True
            elif rope_scaling.get('alpha_value') and hasattr(config, 'max_position_embeddings') and \
                    isinstance(config.max_position_embeddings, int):
                # exllama
                # Note: exllama's own tokenizer has this set correctly in loaders.py, this config will be unused
                max_seq_len = config.max_position_embeddings * rope_scaling.get('alpha_value')
                set_by_rope = True
            max_seq_len = int(max_seq_len)
            if set_by_rope:
                print("Automatically setting max_seq_len=%d for RoPE scaling for %s" % (max_seq_len, base_model),
                      flush=True)
            else:
                print("Did NOT automatically set max_seq_len=%d for RoPE scaling for %s, \
                please set max_seq_len if not correct considering RoPE: %s" % (max_seq_len, base_model, rope_scaling),
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
        model_kwargs['use_flash_attention_2'] = False
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
                            batch_size=1, use_safetensors=False,
                            max_memory=None, offload_folder=None)
        for k in model_kwargs.copy():
            if k not in allowed_dict:
                model_kwargs.pop(k)
        if load_awq.endswith('.pt'):
            args = tuple([base_model, load_awq])
        else:
            args = tuple([base_model])
        model_kwargs['use_safetensors'] = use_safetensors
        model = model_loader(
            *args,
            **model_kwargs,
        )
    elif load_in_8bit or load_in_4bit or not load_half:
        if model_kwargs.get('quantization_config'):
            model_kwargs.pop('load_in_8bit', None)
            model_kwargs.pop('load_in_4bit', None)
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


def get_client_from_inference_server(inference_server, base_model=None, raise_connection_exception=False,
                                     verbose=False):
    inference_server, headers, username, password = get_hf_server(inference_server)
    gr_client = None
    hf_client = None

    gradio_auth = dict(auth=(username, password) if username and username else None)

    if base_model and is_gradio_vision_model(base_model):
        from gradio_utils.grclient import GradioClient
        gr_client = GradioClient(inference_server, check_hash=False, verbose=verbose, serialize=is_gradio_version4,
                                 **gradio_auth)
        gr_client.setup()
    elif headers is None:
        try:
            # preload client since slow for gradio case especially
            from gradio_utils.grclient import GradioClient
            print("GR Client Begin: %s %s" % (inference_server, base_model), flush=True)
            # first do sanity check if alive, else gradio client takes too long by default
            requests.get(inference_server, timeout=int(os.getenv('REQUEST_TIMEOUT', '30')))
            gr_client = GradioClient(inference_server, verbose=verbose, **gradio_auth).setup()
            print("GR Client End: %s" % inference_server, flush=True)
        except (OSError, ValueError) as e:
            # Occurs when wrong endpoint and should have been HF client, so don't hard raise, just move to HF
            gr_client = None
            print("GR Client Failed %s %s: %s" % (inference_server, base_model, str(e)), flush=True)
        except (ConnectTimeoutError, ConnectTimeout, MaxRetryError, ConnectionError, ConnectionError2,
                JSONDecodeError, ReadTimeout2, KeyError, httpx.LocalProtocolError) as e:
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
                    'Could not a find model' in stre or \
                    'safetensors' in stre or \
                    'not appear to have a file named pytorch_model.bin' in stre:
                kwargs['use_safetensors'] = not kwargs.get('use_safetensors', True)
            if 'current architecture does not support Flash Attention 2' in stre:
                kwargs['use_flash_attention_2'] = False
            clear_torch_cache()
            if trial >= trials - 1:
                raise
    return model1, tokenizer1, device1


def get_root_url(url):
    from urllib.parse import urlparse

    # Parse the URL to extract its components
    parsed_url = urlparse(url)

    # Extracted parts: scheme, hostname, and port
    scheme = parsed_url.scheme
    hostname = parsed_url.hostname
    port = parsed_url.port  # Will be None if the port is not explicitly specified in the URL

    # Conditionally add the port to the reassembled URL only if it was explicitly specified
    if port:
        reassembled_url = f"{scheme}://{hostname}:{port}/"
    else:
        reassembled_url = f"{scheme}://{hostname}/"

    # For displaying as separate parts
    http_part = scheme
    ip_part = hostname
    port_part = port if port else "Not specified"  # Display 'Not specified' or similar if there's no port

    # Output the reassembled URL
    return reassembled_url


def get_inf_models(inference_server, verbose=False):
    models = []
    if inference_server.startswith('google'):
        import google.generativeai as genai
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                name_split = m.name.split('models/')
                if len(name_split) >= 2:
                    name = name_split[1]
                    models.append(name)
    elif inference_server.startswith('mistralai'):
        from mistralai.client import MistralClient
        from mistralai.async_client import MistralAsyncClient

        api_key = os.environ["MISTRAL_API_KEY"]
        assert api_key, "Missing MistralAI API key"
        client = MistralClient(api_key=api_key)

        try:
            list_models_response = client.list_models()
            models.extend([x.id for x in dict(list_models_response)['data']])
        except pydantic_core.ValidationError as e:
            print("mistrail ai issue: %s" % str(e))
            # https://github.com/mistralai/client-python/issues/83
    elif inference_server.startswith('openai') or \
            inference_server.startswith('vllm') or \
            inference_server.startswith('sglang'):
        openai_client, openai_async_client, \
            inf_type, deployment_type, base_url, api_version, api_key = \
            set_openai(inference_server)
        # List models
        try:
            models.extend([x.id for x in openai_client.models.list()])
        except Exception as e:
            print("Can't get OpenAI/vLLM model list, trying ollama: %s" % str(e))
            # in case ollama
            import requests
            root_url = get_root_url(base_url)
            if not root_url.endswith('/'):
                root_url += '/'
            import json
            response = json.loads(requests.get("%sapi/tags" % root_url).text)
            # Print the response content
            if 'models' in response:
                models.extend([x['name'] for x in response['models']])
    elif inference_server.startswith('replicate'):
        pass
    elif inference_server.startswith('sagemaker'):
        pass
    elif inference_server.startswith('anthropic'):
        models.extend(list(anthropic_mapping.keys()))
    elif inference_server.startswith('groq'):
        models.extend(list(groq_mapping.keys()))
    elif inference_server.startswith('http'):
        inference_server, gr_client, hf_client = get_client_from_inference_server(inference_server, verbose=verbose)
        if gr_client is not None:
            res = gr_client.predict(api_name='/model_names')
            models.extend({x['base_model']: x['max_seq_len'] for x in ast.literal_eval(res)})

    return models


def get_model(
        load_8bit: bool = False,
        load_4bit: bool = False,
        low_bit_mode: int = 1,
        load_half: bool = True,
        use_flash_attention_2: bool = True,
        load_gptq: str = '',
        use_autogptq: bool = False,
        load_awq: str = '',
        load_exllama: bool = False,
        use_safetensors: bool = False,
        revision: str = None,
        use_gpu_id: bool = True,
        base_model: str = '',
        inference_server: str = "",
        regenerate_clients: bool = True,
        regenerate_gradio_clients: bool = False,
        tokenizer_base_model: str = '',
        lora_weights: str = "",
        gpu_id: int = 0,
        n_jobs=None,
        n_gpus=None,

        reward_type: bool = None,
        local_files_only: bool = False,
        resume_download: bool = True,
        use_auth_token: Union[str, bool] = None,
        trust_remote_code: bool = True,
        offload_folder: str = None,
        rope_scaling: dict = None,
        max_seq_len: int = None,
        max_output_seq_len: int = None,
        compile_model: bool = False,
        llamacpp_path=None,
        llamacpp_dict=None,
        exllama_dict=None,
        gptq_dict=None,
        hf_model_dict={},
        force_seq2seq_type=False,
        force_t5_type=False,

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
    :param max_output_seq_len:
    :param compile_model: whether to compile torch model
    :param llamacpp_path: Path to download llama.cpp and GPT4All models to
    :param llamacpp_dict: dict of llama.cpp and GPT4All model options
    :param exllama_dict: dict of exllama options
    :param gptq_dict: dict of AutoGPTQ options
    :param attention_sinks: whether to use attention_sinks
    :param sink_dict: dict of attention sinks options
    :param truncation_generation: whether to truncate generation in torch case to max_seq_len
    :param hf_model_dict
    :param verbose:
    :return:
    """
    print("Starting get_model: %s %s" % (base_model, inference_server), flush=True)
    model = None
    if use_auth_token is None:
        use_auth_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

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
    if os.getenv("listen_llama") is None:
        # only old models need this, avoid unless override with ENV
        llama_type = False
    if llama_type:
        if verbose:
            print("Detected as llama type from"
                  " config (%s) or name (%s)" % (llama_type_from_config, llama_type_from_name), flush=True)

    model_name_exllama_if_no_config = '' if not llamacpp_dict else llamacpp_dict.get('model_name_exllama_if_no_config',
                                                                                     '')
    loader_kwargs = dict(model_name=base_model, reward_type=reward_type, llama_type=llama_type,
                         load_gptq=load_gptq,
                         use_autogptq=use_autogptq,
                         load_awq=load_awq, load_exllama=load_exllama,
                         config=config,
                         rope_scaling=rope_scaling, max_seq_len=max_seq_len,
                         model_name_exllama_if_no_config=model_name_exllama_if_no_config,
                         exllama_dict=exllama_dict, gptq_dict=gptq_dict,
                         hf_model_dict=hf_model_dict,
                         force_seq2seq_type=force_seq2seq_type,
                         force_t5_type=force_t5_type,
                         )
    model_loader, tokenizer_loader, conditional_type = get_loaders(**loader_kwargs)

    if not tokenizer_base_model:
        tokenizer_base_model = base_model
        config_tokenizer = config
        # ignore sequence length of tokenizer
    elif tokenizer_base_model == 'tiktoken':
        tokenizer_base_model = 'tiktoken'
        config_tokenizer = None
    else:
        # get tokenizer specific objects
        config_tokenizer, _, max_seq_len_tokenizer = get_config(tokenizer_base_model, **config_kwargs,
                                                                raise_exception=False)
        if max_seq_len_tokenizer is not None:
            print("Using max_seq_len=%s defined by config for tokenizer %s" % (
                max_seq_len_tokenizer, tokenizer_base_model))
            max_seq_len = max_seq_len_tokenizer
        if config is None and max_seq_len is None:
            assert max_seq_len, "Must set max_seq_len if passing different tokenizer than model that cannot be found (config is None) e.g. because a private model"

        loader_kwargs_tokenizer = loader_kwargs.copy()
        loader_kwargs_tokenizer['model_name'] = tokenizer_base_model
        _, tokenizer_loader, _ = get_loaders(**loader_kwargs_tokenizer)

    tokenizer_kwargs = dict(local_files_only=local_files_only,
                            resume_download=resume_download,
                            token=use_auth_token,
                            trust_remote_code=trust_remote_code,
                            offload_folder=offload_folder,
                            revision=revision,
                            padding_side='left',
                            config=config_tokenizer,
                            )

    if load_exllama:
        tokenizer = tokenizer_loader
    elif tokenizer_base_model == 'tiktoken':
        assert max_seq_len is not None, "Please pass --max_seq_len=<max_seq_len> for unknown or tiktoken tokenizer for model %s" % base_model
        tokenizer = FakeTokenizer(model_max_length=max_seq_len - 50, is_openai=True)
        if max_output_seq_len is not None:
            tokenizer.max_output_len = max_output_seq_len
    elif config_tokenizer is not None and tokenizer_loader is not None and not isinstance(tokenizer_loader, str):
        if load_exllama:
            assert base_model == tokenizer_base_model
            tokenizer = tokenizer_loader
        else:
            tokenizer = tokenizer_loader.from_pretrained(tokenizer_base_model, **tokenizer_kwargs)
            if max_seq_len is None and hasattr(tokenizer, 'model_max_length'):
                print("Using max_seq_len=%s defined by tokenizer" % tokenizer.model_max_length)
                max_seq_len = tokenizer.model_max_length
            # sets raw (no cushion) limit
            # If using RoPE with scaling, then for non-exllama models (e.g. HF models),
            #  then config -> tokenizer will set model_max_length correctly
            set_model_max_len(max_seq_len, tokenizer, verbose=False)
            # if using fake tokenizer, not really accurate when lots of numbers, give a bit of buffer, else get:
            # Generation Failed: Input validation error: `inputs` must have less than 2048 tokens. Given: 2233
            tokenizer.model_max_length = int(tokenizer.model_max_length - 70)
    else:
        tokenizer = None

    # if base_model in ["HuggingFaceM4/idefics2-8b-chatty", "HuggingFaceM4/idefics2-8b"]:
    #    # work-around until https://huggingface.co/HuggingFaceM4/idefics2-8b-chatty/discussions/5 fixed
    #    tokenizer.chat_template = "{% for message in messages %}{{message['role'].capitalize()}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"

    if isinstance(inference_server, str) and inference_server.startswith("http"):
        inference_server, gr_client, hf_client = get_client_from_inference_server(inference_server,
                                                                                  base_model=base_model,
                                                                                  verbose=verbose)
        model = gr_client or hf_client
        if tokenizer is not None:
            return model, tokenizer, inference_server
        # tokenizer may still be None if not HF model

    if base_model in openai_gpts and not inference_server:
        raise ValueError("Must select inference server when choosing OpenAI models")
    if base_model in anthropic_gpts and not inference_server:
        raise ValueError("Must select inference server when choosing Anthropic models")
    if base_model in google_gpts and not inference_server:
        raise ValueError("Must select inference server when choosing Google models")
    if base_model in mistralai_gpts and not inference_server:
        raise ValueError("Must select inference server when choosing MistralAI models")
    if base_model in groq_gpts and not inference_server:
        raise ValueError("Must select inference server when choosing Groq models")

    # see if we can set max_seq_len and tokenizer for non-HF models or check at least if set when required
    inf_server_for_max_seq_len_handling = isinstance(inference_server, str) and (
            inference_server.startswith('openai') or
            inference_server.startswith('vllm') or
            inference_server.startswith('sglang') or
            inference_server.startswith('replicate') or
            inference_server.startswith('sagemaker') or
            inference_server.startswith('anthropic')
    )

    if inference_server.startswith('vllm') or \
            inference_server.startswith('sglang') or \
            inference_server.startswith('openai'):
        t0 = time.time()
        client, async_client, inf_type, deployment_type, base_url, api_version, api_key = \
            set_openai(inference_server, model_name=base_model)
        if not regenerate_clients:
            model = dict(client=client, async_client=async_client, inf_type=inf_type, deployment_type=deployment_type,
                         base_url=base_url, api_version=api_version, api_key=api_key)
        if verbose:
            print("Duration client %s: %s" % (base_model, time.time() - t0), flush=True)

    if inference_server.startswith('anthropic'):
        t0 = time.time()
        import anthropic
        base_url = os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com")
        api_key = os.getenv('ANTHROPIC_API_KEY')
        timeout = 600
        anthropic_kwargs = dict(base_url=base_url, api_key=api_key, timeout=timeout)
        client = anthropic.Anthropic(**anthropic_kwargs)
        async_client = anthropic.AsyncAnthropic(**anthropic_kwargs)
        if not regenerate_clients:
            model = dict(client=client, async_client=async_client, inf_type='anthropic', base_url=base_url,
                         api_key=api_key,
                         timeout=timeout)
        if verbose:
            print("Duration client %s: %s" % (base_model, time.time() - t0), flush=True)

    google_client = None
    if inference_server.startswith('google'):
        t0 = time.time()
        import google.generativeai as genai
        see_model = False
        models = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    name_split = m.name.split('models/')
                    if len(name_split) >= 2:
                        name = name_split[1]
                        models.append(name)
                        if name not in google_mapping:
                            if os.getenv('HARD_ASSERTS'):
                                raise ValueError("%s not in google_mapping" % name)
                            google_mapping[name] = 8192  # estimate
                            google_gpts.append(name)
                            prompt_type_to_model_name['google'].append(name)
                        see_model |= base_model == name
            assert see_model, "Did not find model=%s in API access: %s" % (base_model, models)
        except Exception as e:
            print("Can't automatically check Google models: %s" % str(e))
            assert base_model in google_mapping, "Unknown google model %s" % base_model

        api_key = os.getenv('GOOGLE_API_KEY')
        assert api_key, "Missing Google Gemini API key"
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel(base_model)
        async_client = genai.GenerativeModel(base_model)
        timeout = 600
        if not regenerate_clients:
            model = dict(client=client, async_client=async_client, inf_type='google', base_url=None, api_key=api_key,
                         timeout=timeout)
        if verbose:
            print("Duration client %s: %s" % (base_model, time.time() - t0), flush=True)
        google_client = client

    if inference_server.startswith('mistralai'):
        t0 = time.time()
        from mistralai.client import MistralClient
        from mistralai.async_client import MistralAsyncClient

        api_key = os.environ["MISTRAL_API_KEY"]
        assert api_key, "Missing MistralAI API key"
        client = MistralClient(api_key=api_key)

        try:
            list_models_response = client.list_models()
            see_model = False
            models = [x.id for x in dict(list_models_response)['data']]
            for name in models:
                see_model |= base_model == name
                if name not in mistralai_mapping:
                    if os.getenv('HARD_ASSERTS'):
                        raise ValueError("%s not in mistralai_mapping" % name)
                    mistralai_mapping[name] = 31768  # estimate
            assert see_model, "Did not find model=%s in API access: %s" % (base_model, models)
        except pydantic_core.ValidationError as e:
            print("mistrail ai issue: %s" % str(e))
            # https://github.com/mistralai/client-python/issues/83

        async_client = MistralAsyncClient(api_key=api_key)

        timeout = 600
        if not regenerate_clients:
            model = dict(client=client, async_client=async_client, inf_type='mistralai', base_url=None, api_key=api_key,
                         timeout=timeout)
        if verbose:
            print("Duration client %s: %s" % (base_model, time.time() - t0), flush=True)

    if inference_server.startswith('groq'):
        if len(inference_server.split(':')) == 2:
            groq_api_key = inference_server.split(':')[1]
            inference_server = inference_server.split(':')[0]
        else:
            groq_api_key = os.getenv('GROQ_API_KEY')

        t0 = time.time()
        from groq import Client, AsyncClient

        assert groq_api_key, "Missing Groq API key"
        client = Client(api_key=groq_api_key)

        async_client = AsyncClient(api_key=groq_api_key)

        timeout = 600
        if not regenerate_clients:
            model = dict(client=client, async_client=async_client, inf_type='groq', base_url=None, api_key=groq_api_key,
                         timeout=timeout)
        if verbose:
            print("Duration client %s: %s" % (base_model, time.time() - t0), flush=True)

    if inf_server_for_max_seq_len_handling or \
            inference_server.startswith('openai') or \
            base_model in openai_gpts or \
            inference_server.startswith('anthropic') or \
            base_model in anthropic_gpts or \
            inference_server.startswith('google') or \
            base_model in google_gpts or \
            inference_server.startswith('mistralai') or \
            base_model in mistralai_gpts or \
            inference_server.startswith('groq') or \
            base_model in groq_gpts:
        max_output_len = None
        if inference_server.startswith('openai') or base_model in openai_gpts:
            if inference_server.startswith('openai') and base_model in openai_gpts:
                client, async_client, inf_type, deployment_type, base_url, api_version, api_key = \
                    set_openai(inference_server, model_name=base_model)
                assert api_key, "No OpenAI key detected.  Set environment for OPENAI_API_KEY or add to inference server line: %s" % inference_server
            # Don't return None, None for model, tokenizer so triggers
            if base_model in model_token_mapping:
                if max_seq_len is None:
                    max_seq_len = model_token_mapping[base_model]
            else:
                print("Using unknown (or proxy) OpenAI model: %s for inference_server=%s" % (
                    base_model, inference_server))
            if base_model in model_token_mapping_outputs:
                if max_output_len is None:
                    max_output_len = model_token_mapping_outputs[base_model]
            else:
                if os.getenv('HARD_ASSERTS'):
                    assert max_output_seq_len is not None, "Must set max_output_seq_len"
                else:
                    max_output_seq_len = 8192  # estimate
                max_output_len = max_output_seq_len
        if inference_server.startswith('anthropic') or base_model in anthropic_gpts:
            if inference_server.startswith('anthropic'):
                assert os.getenv('ANTHROPIC_API_KEY'), "Set environment for ANTHROPIC_API_KEY"
            # Don't return None, None for model, tokenizer so triggers
            # include small token cushion
            if base_model in anthropic_mapping:
                if max_seq_len is None:
                    max_seq_len = anthropic_mapping[base_model]
            else:
                raise ValueError("Invalid base_model=%s for inference_server=%s" % (base_model, inference_server))
            if base_model in anthropic_mapping_outputs:
                if max_output_len is None:
                    max_output_len = anthropic_mapping_outputs[base_model]
            else:
                if os.getenv('HARD_ASSERTS'):
                    assert max_output_seq_len is not None, "Must set max_output_seq_len"
                else:
                    max_output_seq_len = 4096  # estimate
                max_output_len = max_output_seq_len
        if inference_server.startswith('google') or base_model in google_gpts:
            if inference_server.startswith('google'):
                assert os.getenv('GOOGLE_API_KEY'), "Set environment for GOOGLE_API_KEY"
            # Don't return None, None for model, tokenizer so triggers
            # include small token cushion
            if base_model in google_mapping:
                if max_seq_len is None:
                    max_seq_len = google_mapping[base_model]
            else:
                raise ValueError("Invalid base_model=%s for inference_server=%s" % (base_model, inference_server))
            if base_model in google_mapping_outputs:
                if max_output_len is None:
                    max_output_len = google_mapping_outputs[base_model]
            else:
                if os.getenv('HARD_ASSERTS'):
                    assert max_output_seq_len is not None, "Must set max_output_seq_len"
                else:
                    max_output_seq_len = 8192  # estimate
                max_output_len = max_output_seq_len

            if google_client:
                tokenizer = FakeTokenizer(model_max_length=max_seq_len,
                                          is_google=True,
                                          tokenizer=google_client.count_tokens)

        if inference_server.startswith('mistralai') or base_model in mistralai_gpts:
            if inference_server.startswith('mistralai'):
                assert os.getenv('MISTRAL_API_KEY'), "Set environment for MISTRAL_API_KEY"
            # Don't return None, None for model, tokenizer so triggers
            # include small token cushion
            if base_model in mistralai_mapping:
                if max_seq_len is None:
                    max_seq_len = mistralai_mapping[base_model]
            else:
                raise ValueError("Invalid base_model=%s for inference_server=%s" % (base_model, inference_server))
            if base_model in mistralai_mapping_outputs:
                if max_output_len is None:
                    max_output_len = mistralai_mapping_outputs[base_model]
            else:
                if os.getenv('HARD_ASSERTS'):
                    assert max_output_seq_len is not None, "Must set max_output_seq_len"
                else:
                    max_output_seq_len = 31768  # estimate
                max_output_len = max_output_seq_len

            try:
                from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
                tokenizer = MistralTokenizer.from_model(base_model)
                tokenizer.model_max_length = max_seq_len
                from mistral_common.protocol.instruct.request import ChatCompletionRequest
                encoded_tokenizer = tokenizer.encode_chat_completion(
                    ChatCompletionRequest(messages=[dict(role='user', content='Hello')]))
                assert len(encoded_tokenizer.tokens) > 0, "Invalid MistralAI tokenizer"
                tokenizer = FakeTokenizer(model_max_length=max_seq_len, is_mistral=True,
                                          tokenizer=tokenizer, encoding_name=base_model)

            except Exception as e:
                # FIXME: not all models, only some, so do what can
                print("Can't get native Mistral tokenizer for %s: %s" % (base_model, str(e)))
                tokenizer = None
            if tokenizer is None:
                tokenizer = FakeTokenizer(model_max_length=max_seq_len - 1000, is_hf=True,
                                          tokenizer=AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2',
                                                                                  token=use_auth_token,
                                                                                  trust_remote_code=trust_remote_code,
                                                                                  ))

        if inference_server.startswith('groq') or base_model in groq_gpts:
            if inference_server.startswith('groq'):
                assert os.getenv('GROQ_API_KEY'), "Set environment for GROQ_API_KEY"
            # Don't return None, None for model, tokenizer so triggers
            # include small token cushion
            if base_model in groq_mapping:
                if max_seq_len is None:
                    max_seq_len = groq_mapping[base_model]
            else:
                raise ValueError("Invalid base_model=%s for inference_server=%s" % (base_model, inference_server))
            if base_model in groq_mapping_outputs:
                if max_output_len is None:
                    max_output_len = groq_mapping_outputs[base_model]
            else:
                if os.getenv('HARD_ASSERTS'):
                    assert max_output_seq_len is not None, "Must set max_output_seq_len"
                else:
                    max_output_seq_len = 31768  # estimate
                max_output_len = max_output_seq_len

            if base_model == 'mixtral-8x7b-32768':
                tokenizer_base_model = 'mistralai/Mistral-7B-Instruct-v0.2'
            elif base_model == 'llama2-70b-4096':
                tokenizer_base_model = 'h2oai/h2ogpt-4096-llama2-7b'
            # elif base_model == 'gemma-7b-it':

            tokenizer = FakeTokenizer(model_max_length=max_seq_len, is_hf=True,
                                      tokenizer=AutoTokenizer.from_pretrained(tokenizer_base_model,
                                                                              token=use_auth_token,
                                                                              trust_remote_code=trust_remote_code,
                                                                              ))

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

        if inference_server.startswith('openai') or \
                base_model in openai_gpts or \
                inference_server.startswith('anthropic') or \
                base_model in anthropic_gpts or \
                inference_server.startswith('google') or \
                base_model in google_gpts or \
                inference_server.startswith('mistralai') or \
                base_model in mistralai_gpts or \
                inference_server.startswith('groq') or \
                base_model in groq_gpts:
            # must be set by now
            assert max_seq_len is not None, "max_seq_len should have been set for OpenAI or Anthropic or Google or MistralAI or Groq models by now."

        if tokenizer is None:
            # don't use fake (tiktoken) tokenizer for vLLM//replicate if know actual model with actual tokenizer
            # NOTE: Google reaches here because they only provide API to count tokens, no local code.
            assert max_seq_len is not None, "Please set max_seq_len in UI for context length, or pass to CLI --max_seq_len=<max_seq_len>"
            tokenizer = FakeTokenizer(model_max_length=max_seq_len - 50, is_openai=True)
        if max_output_len is not None:
            tokenizer.max_output_len = max_output_len

        if model is None:
            # if model None, means native inference server (and no concern about slowness of regenerating client)
            model = inference_server

        return model, tokenizer, inference_server

    if max_output_seq_len is not None:
        tokenizer.max_output_len = max_output_seq_len

    if inference_server and base_model in non_hf_types and tokenizer is None:
        assert max_seq_len is not None, "Please pass --max_seq_len=<max_seq_len> for non-HF model %s" % base_model
        tokenizer = FakeTokenizer(model_max_length=max_seq_len - 50, is_openai=True)
        return model, tokenizer, inference_server

    if inference_server and tokenizer is None:
        # for new openai, claude, etc. models
        assert max_seq_len is not None, "Please pass --max_seq_len=<max_seq_len> for non-HF model %s" % base_model
        tokenizer = FakeTokenizer(model_max_length=max_seq_len - 50, is_openai=True)
        return model, tokenizer, inference_server

    # shouldn't reach here if had inference server
    assert not inference_server, "Malformed inference_server=%s" % inference_server

    if base_model in non_hf_types:
        from gpt4all_llm import get_model_tokenizer_gpt4all
        model, tokenizer_llamacpp, device = get_model_tokenizer_gpt4all(base_model,
                                                                        n_jobs=n_jobs,
                                                                        gpu_id=gpu_id,
                                                                        n_gpus=n_gpus,
                                                                        max_seq_len=max_seq_len,
                                                                        llamacpp_dict=llamacpp_dict,
                                                                        llamacpp_path=llamacpp_path)
        # give chance to use tokenizer_base_model
        if tokenizer is None:
            tokenizer = tokenizer_llamacpp
        return model, tokenizer, device
    if load_exllama:
        return model_loader, tokenizer, 'cuda' if n_gpus != 0 else 'cpu'

    # get local torch-HF model
    return get_hf_model(load_8bit=load_8bit,
                        load_4bit=load_4bit,
                        low_bit_mode=low_bit_mode,
                        load_half=load_half,
                        use_flash_attention_2=use_flash_attention_2,
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
                        loader_kwargs=loader_kwargs,
                        gptq_dict=gptq_dict,
                        hf_model_dict=hf_model_dict,
                        force_seq2seq_type=force_seq2seq_type,
                        force_t5_type=force_t5_type,

                        verbose=verbose)


def get_hf_model(load_8bit: bool = False,
                 load_4bit: bool = False,
                 low_bit_mode: int = 1,
                 load_half: bool = True,
                 use_flash_attention_2: bool = True,
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
                 compile_model: bool = False,

                 llama_type: bool = False,
                 config_kwargs=None,
                 tokenizer_kwargs=None,
                 loader_kwargs=None,
                 gptq_dict=None,
                 hf_model_dict=None,
                 force_seq2seq_type=None,
                 force_t5_type=None,

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

    config, _, max_seq_len = get_config(base_model, return_model=False, raise_exception=True, **config_kwargs)

    model_loader, tokenizer_loader, conditional_type = get_loaders(**loader_kwargs)

    if not tokenizer_base_model:
        tokenizer_base_model = base_model
        # ignore sequence length of tokenizer
    else:
        loader_kwargs_tokenizer = loader_kwargs.copy()
        loader_kwargs_tokenizer['model_name'] = tokenizer_base_model
        _, tokenizer_loader, _ = get_loaders(**loader_kwargs_tokenizer)

    if tokenizer_loader is not None and not isinstance(tokenizer_loader, str):
        if load_exllama:
            tokenizer = tokenizer_loader
        else:
            # tokenizer_kwargs already contains config=config_tokenizer
            assert tokenizer_kwargs.get('config') is not None, "Tokenizer is invalid: %s" % tokenizer_base_model
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
                                     use_flash_attention_2=use_flash_attention_2,
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
        if n_gpus != 0 and not load_gptq:
            if load_8bit:
                from transformers import BitsAndBytesConfig
                model_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_8bit=load_8bit,
                )

            elif low_bit_mode == 1:
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
        if model_kwargs.get('quantization_config'):
            model_kwargs.pop('load_in_8bit', None)
            model_kwargs.pop('load_in_4bit', None)

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
                                                batch_size=1, use_safetensors=False,
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
                                use_safetensors=use_safetensors,
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
                    use_flash_attention_2: bool = True,
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
                    llamacpp_path: str = None,
                    llamacpp_dict: typing.Dict = None,
                    exllama_dict: typing.Dict = None,
                    gptq_dict: typing.Dict = None,
                    attention_sinks: bool = False,
                    sink_dict: typing.Dict = None,
                    truncation_generation: bool = False,
                    hf_model_dict: typing.Dict = None,
                    force_seq2seq_type: bool = False,
                    force_t5_type: bool = False,

                    verbose: bool = False,
                    ):
    if score_model is not None and score_model.strip():
        load_8bit = False
        load_4bit = False
        low_bit_mode = 1
        load_half = False
        use_flash_attention_2 = False
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
        regenerate_clients = True
        regenerate_gradio_clients = False
        llama_type = False
        max_seq_len = None
        max_output_seq_len = None
        rope_scaling = {}
        compile_model = False
        llamacpp_path = None
        llamacpp_dict = {}
        exllama_dict = {}
        gptq_dict = {}
        attention_sinks = False
        sink_dict = {}
        truncation_generation = False
        hf_model_dict = {}
        force_seq2seq_type = False
        force_t5_type = False

        smodel, stokenizer, sdevice = get_model(reward_type=True,
                                                **get_kwargs(get_model, exclude_names=['reward_type'],
                                                             **locals().copy()))
    else:
        smodel, stokenizer, sdevice = None, None, None
    return smodel, stokenizer, sdevice


def prep_model_state_none():
    model_state_none = model_state_none0.copy()
    model_state_none.update(other_model_state_defaults0)
    # for allowing rest of eval_func_param_names
    for k in eval_func_param_names:
        if k not in model_state_none:
            model_state_none[k] = None
    return model_state_none


def model_lock_to_state(model_dict1, cache_model_state=False, **kwargs):
    if model_dict1 is None:
        model_dict1 = {}
    if isinstance(model_dict1, str):
        model_dict1 = ast.literal_eval(model_dict1)
    if isinstance(model_dict1, list) and len(model_dict1) == 1:
        model_dict1 = model_dict1[0]
    if isinstance(model_dict1, list) and len(model_dict1) > 1:
        raise ValueError("Unexpected multiple model_dict entries: %s" % len(model_dict1))
    assert isinstance(model_dict1, dict)

    if cache_model_state:
        model_dict_json = json.dumps(model_dict1)

        # shouldn't need any objects
        kwargs_model_lock_to_state = kwargs.copy()
        for key in kwargs:
            try:
                json.dumps(kwargs[key])
            except TypeError:
                kwargs_model_lock_to_state.pop(key, None)
        kwargs_json = json.dumps(kwargs_model_lock_to_state)

        return _model_lock_to_state(model_dict_json, kwargs_json)
    else:
        return __model_lock_to_state(model_dict1, **kwargs)


@lru_cache()
def _model_lock_to_state(model_dict_json, kwargs_json):
    model_dict = json.loads(model_dict_json)
    kwargs = json.loads(kwargs_json)

    return __model_lock_to_state(model_dict, **kwargs)


def __model_lock_to_state(model_dict1, **kwargs):
    model_dict = model_dict1
    model_state_none = prep_model_state_none()
    model_list0 = [model_state_none]

    # handle defaults user didn't have to pass
    # special defaults, ignore defaults for these if not specifically set, replace with ''
    model_dict['base_model'] = model_dict.get('base_model', '')
    # display_name may be updated if need to dedup
    model_dict['display_name'] = model_dict.get('display_name', model_dict['base_model'])
    model_dict['tokenizer_base_model'] = model_dict.get('tokenizer_base_model', '')
    model_dict['lora_weights'] = model_dict.get('lora_weights', '')
    model_dict['inference_server'] = model_dict.get('inference_server', '')
    if kwargs['prepare_offline_level'] >= 2:
        if 'openai' not in model_dict['inference_server'] and 'replicate' not in model_dict['inference_server']:
            # assume want locally, but OpenAI and replicate are never local for model part
            model_dict['inference_server'] = ''
    prompt_type_infer = model_dict.get('prompt_type') in ['', None, unknown_prompt_type]
    model_dict['prompt_type'] = model_dict.get('prompt_type',
                                               model_list0[0]['prompt_type'])  # don't use mutated value
    # rest of generic defaults
    new_model_dict0 = copy.deepcopy(model_list0[0])
    for k in new_model_dict0:
        if k not in model_dict:
            model_dict[k] = new_model_dict0[k]
    # make so don't have to pass dict in dict so more like CLI for these options
    for key in llamacpp_inner_dict_keys:
        if key in model_dict:
            model_dict['llamacpp_dict'][key] = model_dict.pop(key)

    model_dict['llamacpp_dict'] = model_dict.get('llamacpp_dict', {})
    model_dict['base_model0'] = model_dict['base_model']
    model_dict['base_model'], model_dict['llamacpp_dict']['model_path_llama'], \
        model_dict['load_gptq'], \
        model_dict['load_awq'], \
        model_dict['llamacpp_dict']['n_gqa'] = \
        switch_a_roo_llama(model_dict['base_model'],
                           model_dict['llamacpp_dict'].get('model_path_llama'),
                           model_dict['load_gptq'],
                           model_dict['load_awq'],
                           model_dict['llamacpp_dict'].get('n_gqa', 0),
                           kwargs['llamacpp_path'])

    # try to infer, ignore empty initial state leading to get_generate_params -> 'plain'
    if prompt_type_infer:
        prompt_type1_trial = model_name_to_prompt_type(model_dict['base_model'],
                                                       model_dict['inference_server'],
                                                       model_name0=model_dict['base_model0'],
                                                       llamacpp_dict=model_dict['llamacpp_dict'])
        if prompt_type1_trial:
            model_dict['prompt_type'] = prompt_type1_trial
            get_prompt_kwargs = dict(context='', reduced=False,
                                     making_context=False,
                                     return_dict=True,
                                     system_prompt=kwargs['system_prompt'])
            model_dict['prompt_dict'], error0 = get_prompt(model_dict['prompt_type'], '',
                                                           **get_prompt_kwargs)
        else:
            model_dict['prompt_dict'] = kwargs['prompt_dict']
    else:
        model_dict['prompt_dict'] = kwargs['prompt_dict']
    model_dict['prompt_dict'] = model_dict.get('prompt_dict', model_dict['prompt_dict'])

    all_kwargs = kwargs.copy()
    all_kwargs.update(locals())
    all_kwargs.update(model_dict)
    if model_dict['base_model'] and not kwargs['login_mode_if_model0']:
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
        if kwargs['fail_if_cannot_connect']:
            raise RuntimeError("Could not connect, see logs")
        # skip
        return {}

    # have model
    model_state_trial = {}
    model_state_trial.update(model_dict)
    model_state_trial.update(dict(model=model0, tokenizer=tokenizer0, device=device))
    if model_state_trial['chat_template'] not in [None, ''] and hasattr(model_state_trial['tokenizer'],
                                                                        'apply_chat_template'):
        try:
            model_state_trial['tokenizer'].chat_template = base64_decode_jinja_template(
                model_state_trial['chat_template'])
            print("Overwrote chat template for %s with\n%s" % (
                model_state_trial['base_model'], model_state_trial['tokenizer'].chat_template))
            messages_test = [dict(role='user', content='Hi'),
                             dict(role='assistant', content='Hello! How can I help you today?')]
            prompt = model_state_trial['tokenizer'].apply_chat_template(messages_test, tokenize=False,
                                                                        add_generation_prompt=True)
            assert isinstance(prompt, str)
        except Exception as e:
            print("Could not overwrite %s template: %s" % (model_state_trial['base_model'], str(e)))
            model_state_trial['chat_template'] = get_chat_template(model_state_trial['tokenizer'])
            if kwargs['fail_if_cannot_connect']:
                raise
    elif has_chat_template(model_state_trial['tokenizer']):
        model_state_trial['chat_template'] = get_chat_template(model_state_trial['tokenizer'])

    model_state_trial['json_vllm'] = is_json_vllm(model_state_trial, model_state_trial['base_model'],
                                                  model_state_trial['inference_server'], verbose=kwargs['verbose'])
    model_state_trial['json'] = is_json_model(model_state_trial['base_model'],
                                              model_state_trial['inference_server'],
                                              json_vllm=model_state_trial['json_vllm'])
    model_state_trial['guided_vllm'] = model_state_trial['json_vllm']
    if model_state_trial['is_actually_vision_model'] is None:
        model_state_trial['is_actually_vision_model'] = is_vision_model(model_state_trial['base_model'])
    model_visible_vision_models = model_state_trial.get('visible_vision_models', kwargs['visible_vision_models'])
    if model_visible_vision_models is None:
        # '' would mean use no vision model, so don't use CLI in that case
        model_visible_vision_models = kwargs['visible_vision_models']
    if isinstance(model_visible_vision_models, str):
        model_visible_vision_models = [model_visible_vision_models]
    if kwargs['model_lock']:  # NOTE: Need real model lock here from kwargs
        all_visible_models = [x.get('visible_models') or x.get('base_model') for x in kwargs['model_lock']]
    else:
        all_visible_models = [kwargs['base_model']]
    if model_state_trial['is_vision_model'] is None:
        model_state_trial['is_vision_model'] = is_vision_model(model_state_trial['base_model'],
                                                               all_visible_models=all_visible_models,
                                                               visible_vision_models=model_visible_vision_models)
    if model_state_trial['images_num_max'] is None:
        if model_state_trial['is_actually_vision_model']:
            model_state_trial['images_num_max'] = images_num_max_dict.get(model_state_trial['base_model'],
                                                                          kwargs['images_num_max'] or 1) or 1
        elif model_state_trial['is_vision_model'] and model_visible_vision_models and len(
                model_visible_vision_models) > 0:
            model_state_trial['images_num_max'] = images_num_max_dict.get(model_visible_vision_models[0],
                                                                          kwargs['images_num_max'] or 1) or 1
        else:
            model_state_trial['images_num_max'] = 0

    if hasattr(tokenizer0, 'max_output_len') and tokenizer0.max_output_len is not None:
        model_state_trial['max_output_seq_len'] = tokenizer0.max_output_len

    auto_visible_vision_models = None
    if kwargs['visible_vision_models']:
        # if in UI, 'auto' is default, but CLI has another default, so use that if set
        auto_visible_vision_models = kwargs['visible_vision_models']
    if model_state_trial['is_actually_vision_model']:
        auto_visible_vision_models = model_state_trial['base_model']
    model_state_trial['auto_visible_vision_models'] = auto_visible_vision_models
    if isinstance(model_state_trial['auto_visible_vision_models'], list) and len(
            model_state_trial['auto_visible_vision_models']) >= 1:
        model_state_trial['auto_visible_vision_models'] = model_state_trial['auto_visible_vision_models'][0]

    diff_keys = set(list(model_state_none.keys())).symmetric_difference(model_state_trial.keys())
    assert len(model_state_none) == len(model_state_trial), diff_keys
    if kwargs['verbose']:
        print("Model %s" % model_dict, flush=True)
    return model_state_trial


def get_on_disk_models(llamacpp_path, use_auth_token, trust_remote_code):
    print("Begin auto-detect HF cache text generation models", flush=True)
    from huggingface_hub import scan_cache_dir
    hf_cache_info = scan_cache_dir()
    hf_models = [x.repo_id for x in hf_cache_info.repos if
                 x.repo_type == 'model' and x.size_on_disk > 100000 and x.nb_files > 0]

    # filter all models down to plausible text models
    # FIXME: Maybe better/faster way to doing this
    from transformers import AutoConfig
    text_hf_models = []
    for x in hf_models:
        try:
            config = AutoConfig.from_pretrained(x,
                                                token=use_auth_token,
                                                trust_remote_code=trust_remote_code)
            if hasattr(config, 'is_encoder_decoder') and config.is_encoder_decoder and x != 'lmsys/fastchat-t5-3b-v1.0':
                print("No loading model %s because is_encoder_decoder=True" % x)
                continue
            if hasattr(config, 'vocab_size'):
                text_hf_models.append(x)
        except Exception as e:
            print("No loading model %s because %s" % (x, str(e)))
            if 'Checkout your internet connection' in str(e):
                # do not continue if no internet
                break
    print("End auto-detect HF cache text generation models", flush=True)

    print("Begin auto-detect llama.cpp models", flush=True)
    llamacpp_path = os.getenv('LLAMACPP_PATH', llamacpp_path) or './'
    llamacpp_files = [os.path.join(llamacpp_path, f) for f in os.listdir(llamacpp_path) if
                      os.path.isfile(os.path.join(llamacpp_path, f))]
    print("End auto-detect llama.cpp models", flush=True)

    return text_hf_models + llamacpp_files
