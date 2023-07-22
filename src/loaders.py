import functools


def get_loaders(model_name, reward_type, llama_type=None, load_gptq='', load_exllama=False):
    # NOTE: Some models need specific new prompt_type
    # E.g. t5_xxl_true_nli_mixture has input format: "premise: PREMISE_TEXT hypothesis: HYPOTHESIS_TEXT".)
    if load_exllama:
        from src.llm_exllama import H2OExLlamaTokenizer, H2OExLlamaGenerator
        from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
        import os, glob

        # Directory containing model, tokenizer, generator
        model_directory = "Llama-2-7b-Chat-GPTQ/"  # FIXME:

        # Locate files we need within that directory
        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        model_config_path = os.path.join(model_directory, "config.json")
        st_pattern = os.path.join(model_directory, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]

        # Create config, model, tokenizer and generator

        config = ExLlamaConfig(model_config_path)               # create config from config.json
        # FIXME:
        config.alpha_value = 1.0  # rope
        config.compress_pos_emb = 1.0  # related rope
        config.model_path = model_path                          # supply path to model weights file

        model = ExLlama(config)                                 # create ExLlama instance and load the weights
        tokenizer = H2OExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file
        tokenizer.model_max_length = int(config.max_seq_len * config.alpha_value)

        cache = ExLlamaCache(model)                             # create cache for inference
        generator = H2OExLlamaGenerator(model, tokenizer, cache)   # create generator
        return generator, tokenizer
    if load_gptq:
        from transformers import AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM
        use_triton = False
        model_loader = functools.partial(AutoGPTQForCausalLM.from_quantized,
                                         quantize_config=None, use_triton=use_triton,
                                         )
        return model_loader, AutoTokenizer
    if llama_type is None:
        llama_type = "llama" in model_name.lower()
    if llama_type:
        from transformers import LlamaForCausalLM, LlamaTokenizer
        return LlamaForCausalLM.from_pretrained, LlamaTokenizer
    elif 'distilgpt2' in model_name.lower():
        from transformers import AutoModelForCausalLM, AutoTokenizer
        return AutoModelForCausalLM.from_pretrained, AutoTokenizer
    elif 'gpt2' in model_name.lower():
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        return GPT2LMHeadModel.from_pretrained, GPT2Tokenizer
    elif 'mbart-' in model_name.lower():
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        return MBartForConditionalGeneration.from_pretrained, MBart50TokenizerFast
    elif 't5' == model_name.lower() or \
            't5-' in model_name.lower() or \
            'flan-' in model_name.lower():
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        return T5ForConditionalGeneration.from_pretrained, AutoTokenizer
    elif 'bigbird' in model_name:
        from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
        return BigBirdPegasusForConditionalGeneration.from_pretrained, AutoTokenizer
    elif 'bart-large-cnn-samsum' in model_name or 'flan-t5-base-samsum' in model_name:
        from transformers import pipeline
        return pipeline, "summarization"
    elif reward_type or 'OpenAssistant/reward-model'.lower() in model_name.lower():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        return AutoModelForSequenceClassification.from_pretrained, AutoTokenizer
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_loader = AutoModelForCausalLM
        tokenizer_loader = AutoTokenizer
        return model_loader.from_pretrained, tokenizer_loader


def get_tokenizer(tokenizer_loader, tokenizer_base_model, local_files_only, resume_download, use_auth_token):
    tokenizer = tokenizer_loader.from_pretrained(tokenizer_base_model,
                                                 local_files_only=local_files_only,
                                                 resume_download=resume_download,
                                                 use_auth_token=use_auth_token,
                                                 padding_side='left')

    tokenizer.pad_token_id = 0  # different from the eos token
    # when generating, we will use the logits of right-most token to predict the next token
    # so the padding should be on the left,
    # e.g. see: https://huggingface.co/transformers/v4.11.3/model_doc/t5.html#inference
    tokenizer.padding_side = "left"  # Allow batched inference

    return tokenizer
