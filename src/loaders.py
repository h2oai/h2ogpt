import functools
import json

from src.enums import t5_type
from src.utils import have_optimum


def get_loaders(model_name, reward_type, llama_type=None,
                load_gptq='',
                use_autogptq=False,
                load_awq='',
                load_exllama=False,
                config=None,
                rope_scaling=None, max_seq_len=None, model_name_exllama_if_no_config='',
                exllama_dict=None, gptq_dict=None,
                hf_model_dict={},
                ):
    # NOTE: Some models need specific new prompt_type
    # E.g. t5_xxl_true_nli_mixture has input format: "premise: PREMISE_TEXT hypothesis: HYPOTHESIS_TEXT".)
    if load_exllama:
        if exllama_dict is None:
            exllama_dict = {}
        from src.llm_exllama import H2OExLlamaTokenizer, H2OExLlamaGenerator
        from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
        import os, glob

        if config:
            # then use HF path
            from transformers import TRANSFORMERS_CACHE
            model_directory = os.path.join(TRANSFORMERS_CACHE, 'models--' + config.name_or_path.replace('/', '--'),
                                           'snapshots', config._commit_hash)
        else:
            # then use path in env file
            # Directory containing model, tokenizer, generator
            model_directory = model_name_exllama_if_no_config

        # download model
        revision = config._commit_hash
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_name, revision=revision)

        # Locate files we need within that directory
        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        assert os.path.isfile(tokenizer_path), "Missing %s" % tokenizer_path
        model_config_path = os.path.join(model_directory, "config.json")
        assert os.path.isfile(model_config_path), "Missing %s" % model_config_path
        st_pattern = os.path.join(model_directory, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]
        assert os.path.isfile(model_path), "Missing %s" % model_path

        # Create config, model, tokenizer and generator
        exconfig = ExLlamaConfig(model_config_path)  # create config from config.json
        rope_scaling = rope_scaling or {}
        exconfig.alpha_value = rope_scaling.get('alpha_value', 1)  # rope
        exconfig.compress_pos_emb = rope_scaling.get('compress_pos_emb', 1)  # related rope
        # update max_seq_len
        assert hasattr(config, 'max_position_embeddings') or hasattr(config,
                                                                     'max_sequence_length'), "Improve code if no such argument"
        if hasattr(config, 'max_position_embeddings'):
            exconfig.max_seq_len = int(config.max_position_embeddings * exconfig.alpha_value)
        else:
            exconfig.max_seq_len = int(config.max_sequence_length * exconfig.alpha_value)
        if 'Llama-2'.lower() in model_name.lower():
            # override bad defaults
            exconfig.max_seq_len = int(4096 * exconfig.alpha_value)
        if max_seq_len is not None:
            exconfig.max_seq_len = max_seq_len

        exconfig.model_path = model_path  # supply path to model weights file
        for k, v in exllama_dict.items():
            setattr(exconfig, k, v)
        if 'set_auto_map' in exllama_dict:
            exconfig.auto_map = [float(alloc) for alloc in exllama_dict['set_auto_map'].split(",")]

        model = ExLlama(exconfig)  # create ExLlama instance and load the weights
        tokenizer = H2OExLlamaTokenizer(tokenizer_path)  # create tokenizer from tokenizer model file
        tokenizer.model_max_length = exconfig.max_seq_len

        cache = ExLlamaCache(model)  # create cache for inference
        generator = H2OExLlamaGenerator(model, tokenizer, cache)  # create generator
        return generator, tokenizer, False
    if load_gptq and use_autogptq:
        if gptq_dict is None:
            gptq_dict = {}
        from transformers import AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM
        if 'use_triton' not in gptq_dict:
            gptq_dict['use_triton'] = False
        if 'llama-2-70B-chat-GPTQ' in model_name.lower() and 'inject_fused_attention' not in gptq_dict:
            gptq_dict.update(dict(inject_fused_attention=False))
        model_loader = functools.partial(AutoGPTQForCausalLM.from_quantized,
                                         quantize_config=None,
                                         **gptq_dict,
                                         )
        return model_loader, AutoTokenizer, False
    if load_gptq and not use_autogptq:
        assert have_optimum, "To use HF transformers GPTQ, please: pip install optimum"
    if load_awq:
        from transformers import AutoTokenizer
        from awq import AutoAWQForCausalLM
        model_loader = functools.partial(AutoAWQForCausalLM.from_quantized,
                                         fuse_layers=True,
                                         )
        return model_loader, AutoTokenizer, False
    if llama_type is None:
        llama_type = "llama" in model_name.lower()
    if llama_type and not load_gptq:
        from transformers import LlamaForCausalLM, LlamaTokenizer
        return functools.partial(LlamaForCausalLM.from_pretrained, **hf_model_dict), LlamaTokenizer, False
    elif 'distilgpt2' in model_name.lower():
        from transformers import AutoModelForCausalLM, AutoTokenizer
        return functools.partial(AutoModelForCausalLM.from_pretrained, **hf_model_dict), AutoTokenizer, False
    elif 'gpt2' in model_name.lower():
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        return functools.partial(GPT2LMHeadModel.from_pretrained, **hf_model_dict), GPT2Tokenizer, False
    elif 'mbart-' in model_name.lower():
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        return functools.partial(MBartForConditionalGeneration.from_pretrained, **hf_model_dict), MBart50TokenizerFast, True
    elif t5_type(model_name):
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        return functools.partial(T5ForConditionalGeneration.from_pretrained, **hf_model_dict), AutoTokenizer, True
    elif 'bigbird' in model_name:
        from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
        return functools.partial(BigBirdPegasusForConditionalGeneration.from_pretrained, **hf_model_dict), AutoTokenizer, True
    elif 'bart-large-cnn-samsum' in model_name or 'flan-t5-base-samsum' in model_name:
        from transformers import pipeline
        return pipeline, "summarization", False
    elif reward_type or 'OpenAssistant/reward-model'.lower() in model_name.lower():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        return functools.partial(AutoModelForSequenceClassification.from_pretrained, **hf_model_dict), AutoTokenizer, False
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_loader = functools.partial(AutoModelForCausalLM.from_pretrained, **hf_model_dict)
        tokenizer_loader = AutoTokenizer
        return model_loader, tokenizer_loader, False


def get_tokenizer(tokenizer_loader, tokenizer_base_model, local_files_only, resume_download, use_auth_token):
    tokenizer = tokenizer_loader.from_pretrained(tokenizer_base_model,
                                                 local_files_only=local_files_only,
                                                 resume_download=resume_download,
                                                 token=use_auth_token,
                                                 padding_side='left')

    tokenizer.pad_token_id = 0  # different from the eos token
    # when generating, we will use the logits of right-most token to predict the next token
    # so the padding should be on the left,
    # e.g. see: https://huggingface.co/transformers/v4.11.3/model_doc/t5.html#inference
    tokenizer.padding_side = "left"  # Allow batched inference

    return tokenizer
