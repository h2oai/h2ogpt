def noop_load(*args, **kwargs):
    return None


def go_prepare_offline(*args, **kwargs):
    kwargs0 = kwargs['kwargs']
    # gen.py steps should have already obtained:
    #   model+tokenizers from base_model or model_lock if required
    #   tokenizers, including tokenizers for model_lock if using inference servers even if no LLM locally
    #   score_model or reward model
    #
    # Additional steps are related to document Q/A:
    # For simplicity use gradio functions,
    #  but not API calls that would require actual gradio app up and API usage that might have issues

    kwargs['max_quality'] = True
    embed = True
    h2ogpt_key = ''
    file_list = ['tests/driverslicense.jpeg', 'tests/CityofTshwaneWater.pdf', 'tests/example.xlsx']

    inputs2 = [kwargs['my_db_state0'],
               kwargs['selection_docs_state0'],
               kwargs['requests_state0'],
               kwargs0['langchain_mode'],
               kwargs0['chunk'],
               kwargs0['chunk_size'],
               embed,
               kwargs['image_audio_loaders_options'],
               kwargs['pdf_loaders_options'],
               kwargs['url_loaders_options'],
               kwargs['jq_schema0'],
               kwargs['extract_frames'],
               h2ogpt_key,
               ]

    for fileup_output in file_list:
        inputs1 = [fileup_output]
        add_file_kwargs = dict(fn=kwargs['update_db_func'],
                               inputs=inputs1 + inputs2)
        add_file_kwargs['fn'](*tuple(add_file_kwargs['inputs']))

        # ensure normal blip (not 2) obtained
        blip2 = 'CaptionBlip2'
        if blip2 in kwargs['image_audio_loaders_options']:
            image_audio_loaders_options = kwargs['image_audio_loaders_options'].copy()
            image_audio_loaders_options.remove(blip2)

        # ensure normal asr (not asrlarge) obtained
        asrlarge = 'ASRLarge'
        if asrlarge in kwargs['image_audio_loaders_options']:
            image_audio_loaders_options = kwargs['image_audio_loaders_options'].copy()
            image_audio_loaders_options.remove(asrlarge)

        inputs2[8] = kwargs['image_audio_loaders_options']
        add_file_kwargs = dict(fn=kwargs['update_db_func'],
                               inputs=inputs1 + inputs2)
        add_file_kwargs['fn'](*tuple(add_file_kwargs['inputs']))

    # FakeTokenizer etc. needs tiktoken for general tasks
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    assert encoding
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    assert encoding

    # sometimes summarization needs gpt2 still
    from transformers import AutoTokenizer
    model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert tokenizer

    # then run h2ogpt as:
    # HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python generate.py --gradio_offline_level=2 --share=False ...
