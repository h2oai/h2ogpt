import copy
import torch

from evaluate_params import eval_func_param_names, input_args_list
from gen import evaluate, check_locals
from prompter import non_hf_types
from utils import clear_torch_cache, NullContext, get_kwargs


def run_cli(  # for local function:
        base_model=None, lora_weights=None, inference_server=None, regenerate_clients=None,
        regenerate_gradio_clients=None,
        debug=None,
        examples=None, memory_restriction_level=None,
        # evaluate kwargs
        n_jobs=None, llamacpp_path=None, llamacpp_dict=None, exllama_dict=None, gptq_dict=None, attention_sinks=None,
        sink_dict=None, truncation_generation=None,
        hf_model_dict=None,
        force_seq2seq_type=None, force_t5_type=None,
        load_exllama=None,

        use_pymupdf=None,
        use_unstructured_pdf=None,
        use_pypdf=None,
        enable_pdf_ocr=None,
        enable_pdf_doctr=None,
        enable_image=None,
        visible_image_models=None,

        try_pdf_as_html=None,
        # for some evaluate args
        load_awq='',
        stream_output=None, async_output=None, num_async=None, stream_map=None,
        prompt_type=None, prompt_dict=None, chat_template=None, system_prompt=None,
        temperature=None, top_p=None, top_k=None, penalty_alpha=None, num_beams=None,
        max_new_tokens=None, min_new_tokens=None, early_stopping=None, max_time=None, repetition_penalty=None,
        num_return_sequences=None, do_sample=None, seed=None, chat=None,
        langchain_mode=None, langchain_action=None, langchain_agents=None,
        document_subset=None, document_choice=None,
        document_source_substrings=None,
        document_source_substrings_op=None,
        document_content_substrings=None,
        document_content_substrings_op=None,
        top_k_docs=None, chunk=None, chunk_size=None,
        pre_prompt_query=None, prompt_query=None,
        pre_prompt_summary=None, prompt_summary=None, hyde_llm_prompt=None,

        user_prompt_for_fake_system_prompt=None,
        json_object_prompt=None,
        json_object_prompt_simpler=None,
        json_code_prompt=None,
        json_code_prompt_if_no_schema=None,
        json_schema_instruction=None,

        image_audio_loaders=None,
        pdf_loaders=None,
        url_loaders=None,
        jq_schema=None,
        extract_frames=None,
        extract_frames0=None,
        guided_whitespace_pattern0=None,
        metadata_in_context0=None,
        llava_prompt=None,
        visible_models=None,
        h2ogpt_key=None,
        add_search_to_context=None,
        chat_conversation=None,
        text_context_list=None,
        docs_ordering_type=None,
        min_max_new_tokens=None,
        max_input_tokens=None,
        max_total_input_tokens=None,
        docs_token_handling=None,
        docs_joiner=None,
        hyde_level=None,
        hyde_template=None,
        hyde_show_only_final=None,
        hyde_show_intermediate_in_accordion=None,
        map_reduce_show_intermediate_in_accordion=None,
        doc_json_mode=None,
        metadata_in_context=None,
        chatbot_role=None,
        speaker=None,
        tts_language=None,
        tts_speed=None,
        image_file=None,
        image_control=None,
        images_num_max=None,
        image_resolution=None,
        image_format=None,
        rotate_align_resize_image=None,
        video_frame_period=None,
        image_batch_image_prompt=None,
        image_batch_final_prompt=None,
        image_batch_stream=None,
        visible_vision_models=None,
        video_file=None,

        response_format=None,
        guided_json=None,
        guided_regex=None,
        guided_choice=None,
        guided_grammar=None,
        guided_whitespace_pattern=None,

        # for evaluate kwargs
        captions_model=None,
        caption_loader=None,
        doctr_loader=None,
        pix2struct_loader=None,
        llava_model=None,
        image_model_dict=None,

        asr_model=None,
        asr_loader=None,
        image_audio_loaders_options0=None,
        pdf_loaders_options0=None,
        url_loaders_options0=None,
        jq_schema0=None,
        keep_sources_in_context=None,
        gradio_errors_to_chatbot=None,
        allow_chat_system_prompt=None,
        src_lang=None, tgt_lang=None, concurrency_count=None, save_dir=None, sanitize_bot_response=None,
        model_state0=None,
        use_auth_token=None,
        trust_remote_code=None,
        score_model_state0=None,
        max_max_new_tokens=None,
        is_public=None,
        max_max_time=None,
        raise_generate_gpu_exceptions=None, load_db_if_exists=None, use_llm_if_no_docs=None,
        my_db_state0=None, selection_docs_state0=None, dbs=None, langchain_modes=None, langchain_mode_paths=None,
        detect_user_path_changes_every_query=None,
        use_openai_embedding=None, use_openai_model=None,
        hf_embedding_model=None, migrate_embedding_model=None,
        cut_distance=None,
        answer_with_sources=None,
        append_sources_to_answer=None,
        append_sources_to_chat=None,
        sources_show_text_in_accordion=None,
        top_k_docs_max_show=None,
        show_link_in_sources=None,
        langchain_instruct_mode=None,
        add_chat_history_to_context=None,
        context=None, iinput=None,
        db_type=None, first_para=None, text_limit=None, verbose=None,
        gradio=None, cli=None,
        use_cache=None,
        auto_reduce_chunks=None, max_chunks=None, headsize=None,
        model_lock=None, force_langchain_evaluate=None,
        model_state_none=None,
        # unique to this function:
        cli_loop=None,
):
    # avoid noisy command line outputs
    import warnings
    warnings.filterwarnings("ignore")
    import logging
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    from_ui = False
    check_locals(**locals().copy())

    score_model = ""  # FIXME: For now, so user doesn't have to pass
    verifier_server = ""  # FIXME: For now, so user doesn't have to pass
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = 'cpu' if n_gpus == 0 else 'cuda'
    context_class = NullContext if n_gpus > 1 or n_gpus == 0 else torch.device

    with context_class(device):
        from functools import partial

        requests_state0 = {}
        roles_state0 = None
        args = (None, my_db_state0, selection_docs_state0, requests_state0, roles_state0)
        assert len(args) == len(input_args_list)
        example1 = examples[-1]  # pick reference example
        all_generations = []
        all_sources = []
        if not context:
            context = ''
        if chat_conversation is None:
            chat_conversation = []

        fun = partial(evaluate,
                      *args,
                      **get_kwargs(evaluate, exclude_names=input_args_list + eval_func_param_names,
                                   **locals().copy()))

        while True:
            clear_torch_cache(allow_skip=True)
            instruction = input("\nEnter an instruction: ")
            if instruction == "exit":
                break

            eval_vars = copy.deepcopy(example1)
            eval_vars[eval_func_param_names.index('instruction')] = \
                eval_vars[eval_func_param_names.index('instruction_nochat')] = instruction
            eval_vars[eval_func_param_names.index('iinput')] = \
                eval_vars[eval_func_param_names.index('iinput_nochat')] = iinput
            eval_vars[eval_func_param_names.index('context')] = context

            # grab other parameters, like langchain_mode
            for k in eval_func_param_names:
                if k in locals().copy():
                    eval_vars[eval_func_param_names.index(k)] = locals().copy()[k]

            gener = fun(*tuple(eval_vars))
            outr = ''
            res_old = ''
            for gen_output in gener:
                res = gen_output['response']
                sources = gen_output.get('sources', 'Failure of Generation')
                if base_model not in non_hf_types or base_model in ['llama']:
                    if not stream_output:
                        print(res)
                    else:
                        # then stream output for gradio that has full output each generation, so need here to show only new chars
                        diff = res[len(res_old):]
                        print(diff, end='', flush=True)
                        res_old = res
                    outr = res  # don't accumulate
                else:
                    outr += res  # just is one thing
                    if sources:
                        # show sources at end after model itself had streamed to std rest of response
                        print('\n\n' + str(sources), flush=True)
            all_generations.append(outr + '\n')
            all_sources.append(sources)
            if not cli_loop:
                break
            if add_chat_history_to_context:
                # for CLI keep track of conversation
                chat_conversation.extend([[instruction, outr]])
    return all_generations, all_sources
