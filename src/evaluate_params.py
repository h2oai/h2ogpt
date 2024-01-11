input_args_list = ['model_state', 'my_db_state', 'selection_docs_state', 'requests_state', 'roles_state']

no_default_param_names = [
    'instruction',
    'iinput',
    'context',
    'instruction_nochat',
    'iinput_nochat',
    'h2ogpt_key',
]

gen_hyper0 = ['num_beams',
              'max_new_tokens',
              'min_new_tokens',
              'early_stopping',
              'max_time',
              'repetition_penalty',
              'num_return_sequences',
              'do_sample',
              ]
gen_hyper = ['temperature',
             'top_p',
             'top_k',
             'penalty_alpha'] + gen_hyper0
reader_names = ['image_audio_loaders', 'pdf_loaders', 'url_loaders', 'jq_schema', 'extract_frames', 'llava_prompt']

eval_func_param_names = ['instruction',
                         'iinput',
                         'context',
                         'stream_output',
                         'prompt_type',
                         'prompt_dict'] + \
                        gen_hyper + \
                        ['chat',
                         'instruction_nochat',
                         'iinput_nochat',
                         'langchain_mode',
                         'add_chat_history_to_context',
                         'langchain_action',
                         'langchain_agents',
                         'top_k_docs',
                         'chunk',
                         'chunk_size',
                         'document_subset',
                         'document_choice',
                         'document_source_substrings',
                         'document_source_substrings_op',
                         'document_content_substrings',
                         'document_content_substrings_op',

                         'pre_prompt_query',
                         'prompt_query',
                         'pre_prompt_summary',
                         'prompt_summary',
                         'hyde_llm_prompt',
                         'system_prompt',
                         ] + \
                        reader_names + \
                        ['visible_models',
                         'h2ogpt_key',
                         'add_search_to_context',

                         'chat_conversation',
                         'text_context_list',
                         'docs_ordering_type',
                         'min_max_new_tokens',
                         'max_input_tokens',
                         'max_total_input_tokens',
                         'docs_token_handling',
                         'docs_joiner',
                         'hyde_level',
                         'hyde_template',
                         'hyde_show_only_final',
                         'doc_json_mode',

                         'chatbot_role',
                         'speaker',
                         'tts_language',
                         'tts_speed',
                         ]

# form evaluate defaults for submit_nochat_api
eval_func_param_names_defaults = eval_func_param_names.copy()
for k in no_default_param_names:
    if k in eval_func_param_names_defaults:
        eval_func_param_names_defaults.remove(k)

eval_extra_columns = ['prompt', 'response', 'score']

# override default_kwargs if user_kwargs None for args evaluate() uses that are not just in model_state
# ensure prompt_type consistent with prep_bot(), so nochat API works same way
# see how default_kwargs is set in gradio_runner.py
key_overrides = ['prompt_type', 'prompt_dict']
