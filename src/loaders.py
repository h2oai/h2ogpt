import functools


def get_loaders(model_name, reward_type, llama_type=None, load_gptq=''):
    # NOTE: Some models need specific new prompt_type
    # E.g. t5_xxl_true_nli_mixture has input format: "premise: PREMISE_TEXT hypothesis: HYPOTHESIS_TEXT".)
    if load_gptq:
        from transformers import AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM
        use_triton = False
        functools.partial(AutoGPTQForCausalLM.from_quantized, quantize_config=None, use_triton=use_triton)
        return AutoGPTQForCausalLM.from_quantized, AutoTokenizer
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
