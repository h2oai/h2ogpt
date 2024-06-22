import ast
import base64
import copy
import functools
import inspect
import itertools
import json
import os
import platform
import pprint
import random
import shutil
import sys
import time
import traceback
import uuid
import filelock
import numpy as np
import pandas as pd
import requests
import ujson

from iterators import TimeoutIterator

from gradio_utils.css import get_css
from gradio_utils.prompt_form import make_chatbots, get_chatbot_name

from src.gradio_funcs import visible_models_to_model_choice, clear_embeddings, fix_text_for_gradio, get_response, \
    my_db_state_done, update_langchain_mode_paths, process_audio, is_valid_key, is_from_ui, get_llm_history, prep_bot, \
    allow_empty_instruction, update_prompt, gen1_fake, get_one_key, get_fun_with_dict_str_plain, bot, choose_exc

from src.db_utils import set_userid, get_username_direct, get_userid_direct, fetch_user, upsert_user
from src.tts_utils import combine_audios
from src.vision.utils_vision import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

# This is a hack to prevent Gradio from phoning home when it gets imported
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'


def my_get(url, **kwargs):
    print('Gradio HTTP request redirected to localhost :)', flush=True)
    kwargs.setdefault('allow_redirects', True)
    return requests.api.request('get', 'http://127.0.0.1/', **kwargs)


original_get = requests.get
requests.get = my_get
import gradio as gr

requests.get = original_get


def fix_pydantic_duplicate_validators_error():
    try:
        from pydantic import class_validators

        class_validators.in_ipython = lambda: True  # type: ignore[attr-defined]
    except ImportError:
        pass


fix_pydantic_duplicate_validators_error()

from enums import DocumentSubset, no_model_str, no_lora_str, no_server_str, LangChainAction, LangChainMode, \
    DocumentChoice, langchain_modes_intrinsic, LangChainTypes, langchain_modes_non_db, gr_to_lg, invalid_key_msg, \
    LangChainAgent, docs_ordering_types, docs_token_handlings, docs_joiner_default, split_google, response_formats, \
    summary_prefix, extract_prefix, unknown_prompt_type, my_db_state0, requests_state0, noneset, \
    is_vision_model, is_video_model, is_json_model
from gradio_themes import H2oTheme, SoftTheme, get_h2o_title, get_simple_title, \
    get_dark_js, get_heap_js, wrap_js_to_lambda, \
    spacing_xsm, radius_xsm, text_xsm
from prompter import prompt_type_to_model_name, prompt_types_strings, non_hf_types, \
    get_prompt, model_names_curated, get_system_prompts, get_llava_prompts, get_llm_history
from utils import flatten_list, zip_data, s3up, clear_torch_cache, get_torch_allocated, system_info_print, \
    ping, makedirs, get_kwargs, system_info, ping_gpu, get_url, \
    save_generate_output, url_alive, remove, dict_to_html, text_to_html, lg_to_gr, str_to_dict, have_serpapi, \
    have_librosa, have_gradio_pdf, have_pyrubberband, is_gradio_version4, have_fiftyone, n_gpus_global, \
    get_accordion_named, get_is_gradio_h2oai, is_uuid4, get_show_username
from gen import get_model, languages_covered, evaluate, score_qa, inputs_kwargs_list, \
    get_max_max_new_tokens, get_minmax_top_k_docs, history_to_context, langchain_actions, langchain_agents_list, \
    switch_a_roo_llama, get_model_max_length_from_tokenizer, \
    get_model_retry, remove_refs, get_on_disk_models, model_name_to_prompt_type, get_inf_models
from evaluate_params import eval_func_param_names, no_default_param_names, eval_func_param_names_defaults, \
    input_args_list

from apscheduler.schedulers.background import BackgroundScheduler


def get_prompt_type1(is_public, **kwargs):
    prompt_types_strings_used = prompt_types_strings.copy()
    if kwargs['model_lock']:
        prompt_types_strings_used += [no_model_str]
        default_prompt_type = kwargs['prompt_type'] or no_model_str
    else:
        default_prompt_type = kwargs['prompt_type'] or unknown_prompt_type
    prompt_type = gr.Dropdown(prompt_types_strings_used,
                              value=default_prompt_type,
                              label="Choose/Select Prompt Type",
                              info="Auto-Detected if known (plain means failed to detect)",
                              visible=not kwargs['model_lock'],
                              interactive=not is_public,
                              )
    return prompt_type


def get_prompt_type2(is_public, **kwargs):
    prompt_types_strings_used = prompt_types_strings.copy()
    if kwargs['model_lock']:
        prompt_types_strings_used += [no_model_str]
        default_prompt_type = kwargs['prompt_type'] or no_model_str
    else:
        default_prompt_type = kwargs['prompt_type'] or unknown_prompt_type
    prompt_type2 = gr.Dropdown(prompt_types_strings_used,
                               value=default_prompt_type,
                               label="Choose/Select Prompt Type Model 2",
                               info="Auto-Detected if known (plain means failed to detect)",
                               visible=False and not kwargs['model_lock'],
                               interactive=not is_public)
    return prompt_type2


def ask_block(kwargs, instruction_label, visible_upload, file_types, mic_sources_kwargs, mic_kwargs, noqueue_kwargs2,
              submit_kwargs, stop_kwargs):
    with gr.Row():
        with gr.Column(scale=50):
            with gr.Row(elem_id="prompt-form-row"):
                label_instruction = 'Ask or Ingest'
                instruction = gr.Textbox(
                    lines=kwargs['input_lines'],
                    label=label_instruction,
                    info=instruction_label,
                    # info=None,
                    elem_id='prompt-form',
                    container=True,
                    visible=False,
                )
                instruction_mm = gr.MultimodalTextbox(
                    lines=kwargs['input_lines'],
                    label=label_instruction,
                    info=instruction_label,
                    # info=None,
                    elem_id='prompt-form',
                    container=True,
                    submit_btn=False,
                )
                mw0 = 20
                mic_button = gr.Button(
                    elem_id="microphone-button" if kwargs['enable_stt'] else None,
                    value="🔴",
                    size="sm",
                    min_width=mw0,
                    visible=kwargs['enable_stt'])
                attach_button = gr.UploadButton(
                    elem_id="attach-button" if visible_upload else None,
                    value=None,
                    label="Upload",
                    size="sm",
                    min_width=mw0,
                    file_types=['.' + x for x in file_types],
                    file_count="multiple",
                    visible=visible_upload)
                add_button = gr.Button(
                    elem_id="add-button" if visible_upload and not kwargs[
                        'actions_in_sidebar'] else None,
                    value="Ingest",
                    size="sm",
                    min_width=mw0,
                    visible=visible_upload and not kwargs['actions_in_sidebar'])

            # AUDIO
            if kwargs['enable_stt']:
                def action(btn, instruction1, audio_state1, stt_continue_mode=1):
                    # print("B0: %s %s" % (audio_state1[0], instruction1), flush=True)
                    """Changes button text on click"""
                    if btn == '🔴':
                        audio_state1[3] = 'on'
                        # print("A: %s %s" % (audio_state1[0], instruction1), flush=True)
                        if stt_continue_mode == 1:
                            audio_state1[0] = instruction1
                            audio_state1[1] = instruction1
                            audio_state1[2] = None
                        return '⭕', instruction1, audio_state1
                    else:
                        audio_state1[3] = 'off'
                        if stt_continue_mode == 1:
                            audio_state1[0] = None  # indicates done for race case
                            instruction1 = audio_state1[1]
                            audio_state1[2] = []
                        # print("B1: %s %s" % (audio_state1[0], instruction1), flush=True)
                        return '🔴', instruction1, audio_state1

                # while audio state used, entries are pre_text, instruction source, and audio chunks, condition
                audio_state0 = [None, None, None, 'off']
                audio_state = gr.State(value=audio_state0)
                audio_output = gr.HTML(visible=False)
                audio = gr.Audio(**mic_sources_kwargs, streaming=True, visible=False,
                                 # max_length=30 if is_public else None,
                                 elem_id='audio',
                                 # waveform_options=dict(show_controls=True),
                                 )
                mic_button_kwargs = dict(fn=functools.partial(action,
                                                              stt_continue_mode=kwargs[
                                                                  'stt_continue_mode']),
                                         inputs=[mic_button, instruction,
                                                 audio_state],
                                         outputs=[mic_button, instruction,
                                                  audio_state],
                                         api_name=False,
                                         show_progress='hidden')
                # JS first, then python, but all in one click instead of using .then() that will delay
                mic_button.click(fn=lambda: None, **mic_kwargs, **noqueue_kwargs2) \
                    .then(**mic_button_kwargs)
                audio.stream(fn=kwargs['transcriber_func'],
                             inputs=[audio_state, audio],
                             outputs=[audio_state, instruction],
                             show_progress='hidden')

        submit_buttons = gr.Row(equal_height=False, visible=kwargs['visible_submit_buttons'])
        with submit_buttons:
            mw1 = 50
            mw2 = 50
            with gr.Column(min_width=mw1):
                submit = gr.Button(value='Submit', variant='primary', size='sm',
                                   min_width=mw1, elem_id="submit")
                stop_btn = gr.Button(value="Stop", variant='secondary', size='sm',
                                     min_width=mw1, elem_id='stop')
                save_chat_btn = gr.Button("Save", size='sm', min_width=mw1)
            with gr.Column(min_width=mw2):
                retry_btn = gr.Button("Redo", size='sm', min_width=mw2)
                undo = gr.Button("Undo", size='sm', min_width=mw2)
                clear_chat_btn = gr.Button(value="Clear", size='sm', min_width=mw2)

            if kwargs['enable_stt'] and (
                    kwargs['tts_action_phrases'] or kwargs['tts_stop_phrases']):
                def detect_words(action_text1, stop_text1, text):
                    got_action_word = False
                    action_words = kwargs['tts_action_phrases']
                    if action_words:
                        for action_word in action_words:
                            if action_word.lower() in text.lower():
                                text = text[:text.lower().index(action_word.lower())]
                                print("Got action: %s %s" % (action_text1, text), flush=True)
                                got_action_word = True
                    if got_action_word:
                        action_text1 = action_text1 + '.'

                    got_stop_word = False
                    stop_words = kwargs['tts_stop_phrases']
                    if stop_words:
                        for stop_word in stop_words:
                            if stop_word.lower() in text.lower():
                                text = text[:text.lower().index(stop_word.lower())]
                                print("Got stop: %s %s" % (stop_text1, text), flush=True)
                                got_stop_word = True

                    if got_stop_word:
                        stop_text1 = stop_text1 + '.'

                    return action_text1, stop_text1, text

                action_text = gr.Textbox(value='', visible=False)
                stop_text = gr.Textbox(value='', visible=False)

                # avoid if no action word, may take extra time
                instruction.change(fn=detect_words,
                                   inputs=[action_text, stop_text, instruction],
                                   outputs=[action_text, stop_text, instruction])

                def clear_audio_state():
                    return audio_state0

                action_text.change(fn=clear_audio_state, outputs=audio_state) \
                    .then(fn=lambda: None, **submit_kwargs)
                stop_text.change(fn=clear_audio_state, outputs=audio_state) \
                    .then(fn=lambda: None, **stop_kwargs)
    return attach_button, add_button, submit_buttons, instruction, instruction_mm, submit, retry_btn, undo, clear_chat_btn, save_chat_btn, stop_btn


def go_gradio(**kwargs):
    page_title = kwargs['page_title']
    model_label_prefix = kwargs['model_label_prefix']
    allow_api = kwargs['allow_api']
    is_public = kwargs['is_public']
    is_hf = kwargs['is_hf']
    memory_restriction_level = kwargs['memory_restriction_level']
    n_gpus = kwargs['n_gpus']
    admin_pass = kwargs['admin_pass']
    model_states = kwargs['model_states']
    dbs = kwargs['dbs']
    db_type = kwargs['db_type']
    visible_langchain_actions = kwargs['visible_langchain_actions']
    visible_langchain_agents = kwargs['visible_langchain_agents']
    allow_upload_to_user_data = kwargs['allow_upload_to_user_data']
    allow_upload_to_my_data = kwargs['allow_upload_to_my_data']
    enable_sources_list = kwargs['enable_sources_list']
    enable_url_upload = kwargs['enable_url_upload']
    enable_text_upload = kwargs['enable_text_upload']
    use_openai_embedding = kwargs['use_openai_embedding']
    hf_embedding_model = kwargs['hf_embedding_model']
    load_db_if_exists = kwargs['load_db_if_exists']
    migrate_embedding_model = kwargs['migrate_embedding_model']
    captions_model = kwargs['captions_model']
    caption_loader = kwargs['caption_loader']
    doctr_loader = kwargs['doctr_loader']
    llava_model = kwargs['llava_model']
    asr_model = kwargs['asr_model']
    asr_loader = kwargs['asr_loader']

    n_jobs = kwargs['n_jobs']
    verbose = kwargs['verbose']

    # for dynamic state per user session in gradio
    model_state0 = kwargs['model_state0']
    score_model_state0 = kwargs['score_model_state0']
    selection_docs_state0 = kwargs['selection_docs_state0']
    visible_models_state0 = kwargs['visible_models_state0']
    visible_vision_models_state0 = kwargs['visible_vision_models_state0']
    visible_image_models_state0 = kwargs['visible_image_models_state0']
    roles_state0 = kwargs['roles_state0']
    # For Heap analytics
    is_heap_analytics_enabled = kwargs['enable_heap_analytics']
    heap_app_id = kwargs['heap_app_id']

    # easy update of kwargs needed for evaluate() etc.
    queue = True
    allow_upload = allow_upload_to_user_data or allow_upload_to_my_data
    allow_upload_api = allow_api and allow_upload

    h2ogpt_key1 = get_one_key(kwargs['h2ogpt_api_keys'], kwargs['enforce_h2ogpt_api_key'])

    kwargs.update(locals().copy())

    # import control
    if kwargs['langchain_mode'] != 'Disabled':
        from gpt_langchain import file_types, have_arxiv
    else:
        have_arxiv = False
        file_types = []

    if 'mbart-' in kwargs['model_lower']:
        instruction_label_nochat = "Text to translate"
    else:
        instruction_label_nochat = "Instruction (Shift-Enter or push Submit to send message," \
                                   " use Enter for multiple input lines)"

    if kwargs['visible_h2ogpt_links']:
        description = """<iframe src="https://ghbtns.com/github-btn.html?user=h2oai&repo=h2ogpt&type=star&count=true&size=small" frameborder="0" scrolling="0" width="280" height="20" title="GitHub"></iframe><small><a href="https://github.com/h2oai/h2ogpt">h2oGPT</a> <a href="https://evalgpt.ai/">LLM Leaderboard</a> <a href="https://github.com/h2oai/h2o-llmstudio">LLM Studio</a><br /><a href="https://codellama.h2o.ai">CodeLlama</a> <br /><a href="https://huggingface.co/h2oai">🤗 Models</a> <br /><a href="https://h2o.ai/platform/enterprise-h2ogpt/">h2oGPTe</a>"""
    else:
        description = None
    description_bottom = "If this host is busy, try<br>[Multi-Model](https://gpt.h2o.ai)<br>[CodeLlama](https://codellama.h2o.ai)<br>[Llama2 70B](https://llama.h2o.ai)<br>[Falcon 40B](https://falcon.h2o.ai)<br>[HF Spaces1](https://huggingface.co/spaces/h2oai/h2ogpt-chatbot)<br>[HF Spaces2](https://huggingface.co/spaces/h2oai/h2ogpt-chatbot2)<br>"
    if is_hf:
        description_bottom += '''<a href="https://huggingface.co/spaces/h2oai/h2ogpt-chatbot?duplicate=true"><img src="https://bit.ly/3gLdBN6" style="white-space: nowrap" alt="Duplicate Space"></a>'''
    task_info_md = ''
    css_code = get_css(kwargs, select_string='\"Select_%s\"' % kwargs['max_visible_models'] if kwargs[
        'max_visible_models'] else '\"Select_Any\"')

    if kwargs['gradio_offline_level'] >= 0:
        # avoid GoogleFont that pulls from internet
        if kwargs['gradio_offline_level'] == 1:
            # front end would still have to download fonts or have cached it at some point
            base_font = 'Source Sans Pro'
        else:
            base_font = 'Helvetica'
        theme_kwargs = dict(font=(base_font, 'ui-sans-serif', 'system-ui', 'sans-serif'),
                            font_mono=('IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'))
    else:
        theme_kwargs = dict()
    if kwargs['gradio_size'] == 'xsmall':
        theme_kwargs.update(dict(spacing_size=spacing_xsm, text_size=text_xsm, radius_size=radius_xsm))
    elif kwargs['gradio_size'] in [None, 'small']:
        theme_kwargs.update(dict(spacing_size=gr.themes.sizes.spacing_sm, text_size=gr.themes.sizes.text_sm,
                                 radius_size=gr.themes.sizes.spacing_sm))
    elif kwargs['gradio_size'] == 'large':
        theme_kwargs.update(dict(spacing_size=gr.themes.sizes.spacing_lg, text_size=gr.themes.sizes.text_lg),
                            radius_size=gr.themes.sizes.spacing_lg)
    elif kwargs['gradio_size'] == 'medium':
        theme_kwargs.update(dict(spacing_size=gr.themes.sizes.spacing_md, text_size=gr.themes.sizes.text_md,
                                 radius_size=gr.themes.sizes.spacing_md))

    theme = H2oTheme(**theme_kwargs) if kwargs['h2ocolors'] else SoftTheme(**theme_kwargs)
    demo = gr.Blocks(theme=theme, css=css_code, title=page_title, analytics_enabled=False)
    callback = gr.CSVLogger()

    # modify, if model lock then don't show models, then need prompts in expert
    kwargs['visible_models_tab'] = kwargs['visible_models_tab'] and not bool(kwargs['model_lock'])

    # Initial model options
    if kwargs['visible_all_prompter_models']:
        model_options0 = flatten_list(list(prompt_type_to_model_name.values())) + kwargs['extra_model_options']
    else:
        model_options0 = []
        if kwargs['visible_curated_models']:
            model_options0.extend(model_names_curated)
        model_options0.extend(kwargs['extra_model_options'])
    if kwargs['base_model'].strip() and kwargs['base_model'].strip() not in model_options0:
        model_options0 = [kwargs['base_model'].strip()] + model_options0
    if kwargs['add_disk_models_to_ui'] and kwargs['visible_models_tab'] and not kwargs['model_lock']:
        model_options0.extend(get_on_disk_models(llamacpp_path=kwargs['llamacpp_path'],
                                                 use_auth_token=kwargs['use_auth_token'],
                                                 trust_remote_code=kwargs['trust_remote_code']))
    model_options0 = sorted(set(model_options0))

    # Initial LORA options
    lora_options = kwargs['extra_lora_options']
    if kwargs['lora_weights'].strip() and kwargs['lora_weights'].strip() not in lora_options:
        lora_options = [kwargs['lora_weights'].strip()] + lora_options

    # Initial server options
    server_options = kwargs['extra_server_options']
    if kwargs['inference_server'].strip() and kwargs['inference_server'].strip() not in server_options:
        server_options = [kwargs['inference_server'].strip()] + server_options
    if os.getenv('OPENAI_API_KEY'):
        if 'openai_chat' not in server_options:
            server_options += ['openai_chat']
        if 'openai' not in server_options:
            server_options += ['openai']

    # always add in no lora case
    # add fake space so doesn't go away in gradio dropdown
    model_options0 = [no_model_str] + sorted(model_options0)
    lora_options = [no_lora_str] + sorted(lora_options)
    server_options = [no_server_str] + sorted(server_options)
    # always add in no model case so can free memory
    # add fake space so doesn't go away in gradio dropdown

    # transcribe, will be detranscribed before use by evaluate()
    if not kwargs['base_model'].strip():
        kwargs['base_model'] = no_model_str

    if not kwargs['lora_weights'].strip():
        kwargs['lora_weights'] = no_lora_str

    if not kwargs['inference_server'].strip():
        kwargs['inference_server'] = no_server_str

    # transcribe for gradio
    kwargs['gpu_id'] = str(kwargs['gpu_id'])

    no_model_msg = 'h2oGPT [   !!! Please Load Model in Models Tab !!!   ]'
    chat_name0 = get_chatbot_name(kwargs.get("base_model"),
                                  kwargs.get("display_name"),
                                  kwargs.get("llamacpp_dict", {}).get("model_path_llama"),
                                  kwargs.get("inference_server"),
                                  kwargs.get("prompt_type"),
                                  kwargs.get("model_label_prefix"),
                                  )
    output_label0 = chat_name0 if kwargs.get('base_model') else no_model_msg
    output_label0_model2 = no_model_msg

    default_kwargs = {k: kwargs[k] for k in eval_func_param_names_defaults}
    # ensure prompt_type consistent with prep_bot(), so nochat API works same way
    default_kwargs['prompt_type'], default_kwargs['prompt_dict'] = \
        update_prompt(default_kwargs['prompt_type'], default_kwargs['prompt_dict'],
                      model_state1=model_state0,
                      which_model=visible_models_to_model_choice(kwargs['visible_models'], model_states),
                      global_scope=True,  # don't assume state0 is the prompt for all models
                      **kwargs,
                      )
    for k in no_default_param_names:
        default_kwargs[k] = ''

    def dummy_fun(x):
        # need dummy function to block new input from being sent until output is done,
        # else gets input_list at time of submit that is old, and shows up as truncated in chatbot
        return x

    def update_auth_selection(auth_user, selection_docs_state1, save=False):
        # in-place update of both
        if 'selection_docs_state' not in auth_user:
            auth_user['selection_docs_state'] = selection_docs_state0
        for k, v in auth_user['selection_docs_state'].items():
            if isinstance(selection_docs_state1[k], dict):
                if save:
                    auth_user['selection_docs_state'][k].clear()
                    auth_user['selection_docs_state'][k].update(selection_docs_state1[k])
                else:
                    if not kwargs['update_selection_state_from_cli']:
                        selection_docs_state1[k].clear()
                    selection_docs_state1[k].update(auth_user['selection_docs_state'][k])
            elif isinstance(selection_docs_state1[k], list):
                if save:
                    auth_user['selection_docs_state'][k].clear()
                    auth_user['selection_docs_state'][k].extend(selection_docs_state1[k])
                else:
                    if not kwargs['update_selection_state_from_cli']:
                        selection_docs_state1[k].clear()
                    selection_docs_state1[k].extend(auth_user['selection_docs_state'][k])
                    newlist = sorted(set(selection_docs_state1[k]))
                    selection_docs_state1[k].clear()
                    selection_docs_state1[k].extend(newlist)
            else:
                raise RuntimeError("Bad type: %s" % selection_docs_state1[k])

    # BEGIN AUTH THINGS
    def get_auth_password(username1, auth_filename):
        with filelock.FileLock(auth_filename + '.lock'):
            auth_dict = {}
            if os.path.isfile(auth_filename):
                if auth_filename.endswith('.db'):
                    auth_dict = fetch_user(auth_filename, username1, verbose=verbose)
                else:
                    try:
                        with open(auth_filename, 'rt') as f:
                            auth_dict = json.load(f)
                    except json.decoder.JSONDecodeError as e:
                        print("Auth exception: %s" % str(e), flush=True)
                        shutil.move(auth_filename, auth_filename + '.bak' + str(uuid.uuid4()))
                        auth_dict = {}
        return auth_dict.get(username1, {}).get('password')

    def auth_func(username1, password1, auth_pairs=None, auth_filename=None,
                  auth_access=None,
                  auth_freeze=None,
                  guest_name=None,
                  selection_docs_state1=None,
                  selection_docs_state00=None,
                  id0=None,
                  **kwargs):
        assert auth_freeze is not None
        if selection_docs_state1 is None:
            selection_docs_state1 = selection_docs_state00
        assert selection_docs_state1 is not None
        assert auth_filename and isinstance(auth_filename, str), "Auth file must be a non-empty string, got: %s" % str(
            auth_filename)
        if auth_access == 'open' and guest_name and username1.startswith(guest_name):
            return True
        if username1 == '':
            # some issue with login
            return False
        if guest_name and username1.startswith(guest_name):
            # for random access with persistent password in auth case
            # username1 here only for auth check, rest of time full guest name used
            username1 = guest_name
        with filelock.FileLock(auth_filename + '.lock'):
            auth_dict = {}
            if os.path.isfile(auth_filename):
                print("Auth access: %s" % username1)
                if auth_filename.endswith('.db'):
                    auth_dict = fetch_user(auth_filename, username1, verbose=verbose)
                else:
                    try:
                        with open(auth_filename, 'rt') as f:
                            auth_dict = json.load(f)
                    except json.decoder.JSONDecodeError as e:
                        print("Auth exception: %s" % str(e), flush=True)
                        shutil.move(auth_filename, auth_filename + '.bak' + str(uuid.uuid4()))
                        auth_dict = {}
            if username1 in auth_dict and username1 in auth_pairs:
                if password1 == auth_dict[username1]['password'] and password1 == auth_pairs[username1]:
                    auth_user = auth_dict[username1]
                    update_auth_selection(auth_user, selection_docs_state1)
                    save_auth_dict(auth_dict, auth_filename, username1)
                    return True
                else:
                    return False
            elif username1 in auth_dict:
                if password1 == auth_dict[username1]['password']:
                    auth_user = auth_dict[username1]
                    update_auth_selection(auth_user, selection_docs_state1)
                    save_auth_dict(auth_dict, auth_filename, username1)
                    return True
                else:
                    return False
            elif username1 in auth_pairs:
                # copy over CLI auth to file so only one state to manage
                auth_dict[username1] = dict(password=auth_pairs[username1], userid=id0 or str(uuid.uuid4()))
                auth_user = auth_dict[username1]
                update_auth_selection(auth_user, selection_docs_state1)
                save_auth_dict(auth_dict, auth_filename, username1)
                return True
            else:
                if auth_access == 'closed':
                    return False
                # open access
                auth_dict[username1] = dict(password=password1, userid=id0 or str(uuid.uuid4()))
                auth_user = auth_dict[username1]
                update_auth_selection(auth_user, selection_docs_state1)
                save_auth_dict(auth_dict, auth_filename, username1)
                if auth_access == 'open':
                    return True
                else:
                    raise RuntimeError("Invalid auth_access: %s" % auth_access)

    def auth_func_open(*args, **kwargs):
        return True

    def get_username(requests_state1):
        username1 = None
        if 'username' in requests_state1:
            username1 = requests_state1['username']
        return username1

    def get_userid_auth_func(requests_state1, auth_filename=None, auth_access=None, guest_name=None, id0=None,
                             **kwargs):
        username1 = get_username(requests_state1)
        if auth_filename and isinstance(auth_filename, str):
            if username1:
                if username1.startswith(guest_name):
                    return str(uuid.uuid4())
                with filelock.FileLock(auth_filename + '.lock'):
                    if os.path.isfile(auth_filename):
                        if auth_filename.endswith('.db'):
                            auth_dict = fetch_user(auth_filename, username1, verbose=verbose)
                        else:
                            with open(auth_filename, 'rt') as f:
                                auth_dict = json.load(f)
                        if username1 in auth_dict:
                            return auth_dict[username1]['userid']
        # if here, then not persistently associated with username1,
        # but should only be one-time asked if going to persist within a single session!
        return id0 or username1 or str(uuid.uuid4())

    get_userid_auth = functools.partial(get_userid_auth_func,
                                        auth_filename=kwargs['auth_filename'],
                                        auth_access=kwargs['auth_access'],
                                        guest_name=kwargs['guest_name'],
                                        )
    if kwargs['auth_access'] == 'closed':
        auth_message1 = "Closed access"
    else:
        auth_message1 = "WELCOME to %s!  Open access" \
                        " (%s/%s or any unique user/pass)" % (page_title, kwargs['guest_name'], kwargs['guest_name'])

    if kwargs['auth_message'] is not None:
        auth_message = kwargs['auth_message']
    else:
        auth_message = auth_message1

    # always use same callable
    auth_pairs0 = {}
    if isinstance(kwargs['auth'], list):
        for k, v in kwargs['auth']:
            auth_pairs0[k] = v
    authf = functools.partial(auth_func,
                              auth_pairs=auth_pairs0,
                              auth_filename=kwargs['auth_filename'],
                              auth_access=kwargs['auth_access'],
                              auth_freeze=kwargs['auth_freeze'],
                              guest_name=kwargs['guest_name'],
                              selection_docs_state00=copy.deepcopy(selection_docs_state0))

    def get_request_state(requests_state1, request, db1s):
        # if need to get state, do it now
        if not requests_state1:
            requests_state1 = requests_state0.copy()
        if requests:
            if not requests_state1.get('headers', '') and hasattr(request, 'headers'):
                requests_state1.update(request.headers)
            if not requests_state1.get('host', '') and hasattr(request, 'host'):
                requests_state1.update(dict(host=request.host))
            if not requests_state1.get('host2', '') and hasattr(request, 'client') and hasattr(request.client, 'host'):
                requests_state1.update(dict(host2=request.client.host))
            if not requests_state1.get('username', '') and hasattr(request, 'username'):
                # use already-defined username instead of keep changing to new uuid
                # should be same as in requests_state1
                db_username = get_username_direct(db1s)

                if request.username and split_google in request.username:
                    assert len(request.username.split(split_google)) >= 2  # 3 if already got pic out
                    username = split_google.join(request.username.split(split_google)[0:2])  # no picture
                else:
                    username = request.username

                requests_state1.update(dict(username=username or db_username or str(uuid.uuid4())))
            if not requests_state1.get('picture', ''):
                if request.username and split_google in request.username and len(
                        request.username.split(split_google)) == 3:
                    picture = split_google.join(request.username.split(split_google)[2:3])  # picture
                else:
                    picture = None

                requests_state1.update(dict(picture=picture))
        requests_state1 = {str(k): str(v) for k, v in requests_state1.items()}
        return requests_state1

    def user_state_setup(db1s, requests_state1, guest_name1, request: gr.Request, *args):
        requests_state1 = get_request_state(requests_state1, request, db1s)
        set_userid(db1s, requests_state1, get_userid_auth, guest_name=guest_name1)
        args_list = [db1s, requests_state1] + list(args)
        return tuple(args_list)

    # END AUTH THINGS

    image_audio_loaders_options0, image_audio_loaders_options, \
        pdf_loaders_options0, pdf_loaders_options, \
        url_loaders_options0, url_loaders_options = lg_to_gr(**kwargs)
    jq_schema0 = '.[]'

    def click_js():
        return """function audioRecord() {
    var xPathRes = document.evaluate ('//*[contains(@class, "record")]', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null); 
    xPathRes.singleNodeValue.click();}"""

    def click_submit():
        return """function check() {
  document.getElementById("submit").click();
}"""

    def click_stop():
        return """function check() {
  document.getElementById("stop").click();
}"""

    if is_gradio_version4:
        noqueue_kwargs = dict(concurrency_limit=None)
        noqueue_kwargs2 = dict(concurrency_limit=None)
        noqueue_kwargs_curl = dict(queue=False)
        mic_kwargs = dict(js=click_js())
        submit_kwargs = dict(js=click_submit())
        stop_kwargs = dict(js=click_stop())
        dark_kwargs = dict(js=wrap_js_to_lambda(0, get_dark_js()))
        queue_kwargs = dict(default_concurrency_limit=kwargs['concurrency_count'])
        mic_sources_kwargs = dict(sources=['microphone'],
                                  waveform_options=dict(show_controls=False, show_recording_waveform=False))
    else:
        noqueue_kwargs = dict(queue=False)
        noqueue_kwargs2 = dict()
        noqueue_kwargs_curl = dict(queue=False)
        mic_kwargs = dict(_js=click_js())
        submit_kwargs = dict(_js=click_submit())
        stop_kwargs = dict(_js=click_stop())
        dark_kwargs = dict(_js=wrap_js_to_lambda(0, get_dark_js()))
        queue_kwargs = dict(concurrency_count=kwargs['concurrency_count'])
        mic_sources_kwargs = dict(source='microphone')

    if kwargs['model_lock']:
        have_vision_models = any([is_vision_model(x.get('base_model', '')) for x in kwargs['model_lock']])
    else:
        have_vision_models = is_vision_model(kwargs['base_model'])

    is_gradio_h2oai = get_is_gradio_h2oai()

    # image control prep
    image_gen_visible = kwargs['enable_imagegen']
    image_change_visible = kwargs['enable_imagechange']
    image_control_panels_visible = False  # WIP
    image_tab_visible = image_control_panels_visible and (image_gen_visible or image_change_visible)
    visible_image_models_visible = len(visible_image_models_state0) > 1
    visible_image_models_kwargs = dict(choices=visible_image_models_state0,
                                       label="Visible ImageGen Models",
                                       value=visible_image_models_state0[
                                           0] if visible_image_models_state0 else None,
                                       interactive=True,
                                       multiselect=False,
                                       visible=visible_image_models_visible,
                                       filterable=False,
                                       max_choices=None,
                                       )

    with demo:
        support_state_callbacks = hasattr(gr.State(), 'callback')

        # avoid actual model/tokenizer here or anything that would be bad to deepcopy
        # https://github.com/gradio-app/gradio/issues/3558
        def model_state_done(state):
            if isinstance(state, dict) and 'model' in state and hasattr(state['model'], 'cpu'):
                state['model'].cpu()
                state['model'] = None
                clear_torch_cache()

        model_state_cb = dict(callback=model_state_done) if support_state_callbacks else {}
        model_state_default = dict(model='model', tokenizer='tokenizer', device=kwargs['device'],
                                   base_model=kwargs['base_model'],
                                   display_name=kwargs['base_model'],
                                   tokenizer_base_model=kwargs['tokenizer_base_model'],
                                   lora_weights=kwargs['lora_weights'],
                                   inference_server=kwargs['inference_server'],
                                   prompt_type=kwargs['prompt_type'],
                                   prompt_dict=kwargs['prompt_dict'],
                                   visible_models=visible_models_to_model_choice(kwargs['visible_models'],
                                                                                 model_states),
                                   h2ogpt_key=None,
                                   # only apply at runtime when doing API call with gradio inference server
                                   )
        [model_state_default.update({k: v}) for k, v in kwargs['model_state_none'].items() if
         k not in model_state_default]
        model_state = gr.State(value=model_state_default, **model_state_cb)

        my_db_state_cb = dict(callback=my_db_state_done) if support_state_callbacks else {}

        model_state2 = gr.State(kwargs['model_state_none'].copy())
        model_options_state = gr.State([model_options0], **model_state_cb)
        lora_options_state = gr.State([lora_options])
        server_options_state = gr.State([server_options])
        my_db_state = gr.State(my_db_state0, **my_db_state_cb)
        chat_state = gr.State({})
        if kwargs['enable_tts'] and kwargs['tts_model'].startswith('tts_models/'):
            from src.tts_coqui import get_role_to_wave_map
            roles_state0 = roles_state0 if roles_state0 else get_role_to_wave_map()
        else:
            roles_state0 = {}
        roles_state = gr.State(roles_state0)
        docs_state00 = kwargs['document_choice'] + [DocumentChoice.ALL.value]
        docs_state0 = []
        [docs_state0.append(x) for x in docs_state00 if x not in docs_state0]
        docs_state = gr.State(docs_state0)
        viewable_docs_state0 = ['None']
        viewable_docs_state = gr.State(viewable_docs_state0)
        selection_docs_state0 = update_langchain_mode_paths(selection_docs_state0)
        selection_docs_state = gr.State(selection_docs_state0)
        requests_state = gr.State(requests_state0)

        if description is None:
            description = ''
        markdown_logo = f"""
                {get_h2o_title(page_title, description, visible_h2ogpt_qrcode=kwargs['visible_h2ogpt_qrcode'])
        if kwargs['h2ocolors'] else get_simple_title(page_title, description)}
                """
        if kwargs['visible_h2ogpt_logo']:
            gr.Markdown(markdown_logo)

        # go button visible if
        base_wanted = kwargs['base_model'] != no_model_str and kwargs['login_mode_if_model0']
        go_btn = gr.Button(value="ENTER", visible=base_wanted, variant="primary")

        nas = ' '.join(['NA'] * len(kwargs['model_states']))
        res_value = "Response Score: NA" if not kwargs[
            'model_lock'] else "Response Scores: %s" % nas

        user_can_do_sum = kwargs['langchain_mode'] != LangChainMode.DISABLED.value and \
                          (kwargs['visible_side_bar'] or kwargs['visible_system_tab'])
        if user_can_do_sum:
            extra_prompt_form = ".  Just Click Submit for simple Summarize/Extract"
        else:
            extra_prompt_form = ""
        if allow_upload:
            extra_prompt_form += ".  Clicking Ingest adds text as URL/ArXiv/YouTube/Text."
        if kwargs['input_lines'] > 1:
            instruction_label = "Shift-Enter to Submit, Enter adds lines%s" % extra_prompt_form
        else:
            instruction_label = "Enter to Submit, Shift-Enter adds lines%s" % extra_prompt_form

        def get_langchain_choices(selection_docs_state1):
            langchain_modes = selection_docs_state1['langchain_modes']

            if is_hf:
                # don't show 'wiki' since only usually useful for internal testing at moment
                no_show_modes = ['Disabled', 'wiki']
            else:
                no_show_modes = ['Disabled']
            allowed_modes = langchain_modes.copy()
            # allowed_modes = [x for x in allowed_modes if x in dbs]
            allowed_modes += ['LLM']
            if allow_upload_to_my_data and 'MyData' not in allowed_modes:
                allowed_modes += ['MyData']
            if allow_upload_to_user_data and 'UserData' not in allowed_modes:
                allowed_modes += ['UserData']
            choices = [x for x in langchain_modes if x in allowed_modes and x not in no_show_modes]
            return choices

        def get_df_langchain_mode_paths(selection_docs_state1, db1s, dbs1=None):
            langchain_choices1 = get_langchain_choices(selection_docs_state1)
            langchain_mode_paths = selection_docs_state1['langchain_mode_paths']
            langchain_mode_paths = {k: v for k, v in langchain_mode_paths.items() if k in langchain_choices1}
            if langchain_mode_paths:
                langchain_mode_paths = langchain_mode_paths.copy()
                for langchain_mode1 in langchain_modes_non_db:
                    langchain_mode_paths.pop(langchain_mode1, None)
                df1 = pd.DataFrame.from_dict(langchain_mode_paths.items(), orient='columns')
                df1.columns = ['Collection', 'Path']
                df1 = df1.set_index('Collection')
            else:
                df1 = pd.DataFrame(None)
            langchain_mode_types = selection_docs_state1['langchain_mode_types']
            langchain_mode_types = {k: v for k, v in langchain_mode_types.items() if k in langchain_choices1}
            if langchain_mode_types:
                langchain_mode_types = langchain_mode_types.copy()
                for langchain_mode1 in langchain_modes_non_db:
                    langchain_mode_types.pop(langchain_mode1, None)

                df2 = pd.DataFrame.from_dict(langchain_mode_types.items(), orient='columns')
                df2.columns = ['Collection', 'Type']
                df2 = df2.set_index('Collection')

                from src.gpt_langchain import get_persist_directory, load_embed
                persist_directory_dict = {}
                embed_dict = {}
                chroma_version_dict = {}
                for langchain_mode3 in langchain_mode_types:
                    langchain_type3 = langchain_mode_types.get(langchain_mode3, LangChainTypes.EITHER.value)
                    # this also makes a directory, but may not use it later
                    persist_directory3, langchain_type3 = get_persist_directory(langchain_mode3,
                                                                                langchain_type=langchain_type3,
                                                                                db1s=db1s, dbs=dbs1)
                    got_embedding3, use_openai_embedding3, hf_embedding_model3 = load_embed(
                        persist_directory=persist_directory3, use_openai_embedding=use_openai_embedding)
                    persist_directory_dict[langchain_mode3] = persist_directory3
                    embed_dict[langchain_mode3] = 'OpenAI' if not hf_embedding_model3 else hf_embedding_model3

                    if os.path.isfile(os.path.join(persist_directory3, 'chroma.sqlite3')):
                        chroma_version_dict[langchain_mode3] = 'ChromaDB>=0.4'
                    elif os.path.isdir(os.path.join(persist_directory3, 'index')):
                        chroma_version_dict[langchain_mode3] = 'ChromaDB<0.4'
                    elif not os.listdir(persist_directory3):
                        if db_type == 'chroma':
                            chroma_version_dict[langchain_mode3] = 'ChromaDB>=0.4'  # will be
                        elif db_type == 'chroma_old':
                            chroma_version_dict[langchain_mode3] = 'ChromaDB<0.4'  # will be
                        else:
                            chroma_version_dict[langchain_mode3] = 'Weaviate'  # will be
                        if isinstance(hf_embedding_model, dict):
                            hf_embedding_model3 = hf_embedding_model['name']
                        else:
                            hf_embedding_model3 = 'OpenAI' if not hf_embedding_model else hf_embedding_model
                        assert isinstance(hf_embedding_model3, str)
                        embed_dict[langchain_mode3] = hf_embedding_model3  # will be
                    else:
                        chroma_version_dict[langchain_mode3] = 'Weaviate'

                df3 = pd.DataFrame.from_dict(persist_directory_dict.items(), orient='columns')
                df3.columns = ['Collection', 'Directory']
                df3 = df3.set_index('Collection')

                df4 = pd.DataFrame.from_dict(embed_dict.items(), orient='columns')
                df4.columns = ['Collection', 'Embedding']
                df4 = df4.set_index('Collection')

                df5 = pd.DataFrame.from_dict(chroma_version_dict.items(), orient='columns')
                df5.columns = ['Collection', 'DB']
                df5 = df5.set_index('Collection')
            else:
                df2 = pd.DataFrame(None)
                df3 = pd.DataFrame(None)
                df4 = pd.DataFrame(None)
                df5 = pd.DataFrame(None)
            df_list = [df2, df1, df3, df4, df5]
            df_list = [x for x in df_list if x.shape[1] > 0]
            if len(df_list) > 1:
                df = df_list[0].join(df_list[1:]).replace(np.nan, '').reset_index()
            elif len(df_list) == 0:
                df = df_list[0].replace(np.nan, '').reset_index()
            else:
                df = pd.DataFrame(None)
            return df

        normal_block = gr.Row(visible=not base_wanted, equal_height=False, elem_id="col_container")
        with normal_block:
            side_bar = gr.Column(elem_id="sidebar", scale=1, min_width=100, visible=kwargs['visible_side_bar'])
            with side_bar:
                with gr.Accordion("Chats", open=False, visible=True):
                    radio_chats = gr.Radio(value=None, label="Saved Chats", show_label=False,
                                           visible=True, interactive=True,
                                           type='value')
                    visible_speak_me = kwargs['enable_tts'] and kwargs['predict_from_text_func'] is not None
                    speak_human_button = gr.Button("Speak Instruction", visible=visible_speak_me, size='sm')
                    speak_bot_button = gr.Button("Speak Response", visible=visible_speak_me, size='sm')
                    speak_text_api_button = gr.Button("Speak Text API", visible=False)
                    speak_text_plain_api_button = gr.Button("Speak Text Plain API", visible=False)
                    stop_speak_button = gr.Button("Stop/Clear Speak", visible=visible_speak_me, size='sm')
                    if kwargs['enable_tts'] and kwargs['tts_model'].startswith('tts_models/'):
                        from src.tts_coqui import get_roles
                        chatbot_role = get_roles(choices=list(roles_state.value.keys()), value=kwargs['chatbot_role'])
                    else:
                        chatbot_role = gr.Dropdown(choices=['None'], visible=False, value='None')
                    if kwargs['enable_tts'] and kwargs['tts_model'].startswith('microsoft'):
                        from src.tts import get_speakers_gr
                        speaker = get_speakers_gr(value=kwargs['speaker'])
                    else:
                        speaker = gr.Radio(visible=False)
                    min_tts_speed = 1.0 if not have_pyrubberband else 0.1
                    tts_speed = gr.Number(minimum=min_tts_speed, maximum=10.0, step=0.1,
                                          value=kwargs['tts_speed'],
                                          label='Speech Speed',
                                          visible=kwargs['enable_tts'] and not is_public,
                                          interactive=not is_public)

                upload_visible = kwargs['langchain_mode'] != 'Disabled' and allow_upload
                url_visible = kwargs['langchain_mode'] != 'Disabled' and allow_upload and enable_url_upload
                if have_arxiv and have_librosa:
                    url_label = 'URLs/ArXiv/Youtube'
                elif have_arxiv:
                    url_label = 'URLs/ArXiv'
                elif have_librosa:
                    url_label = 'URLs/Youtube'
                else:
                    url_label = 'URLs'
                text_visible = kwargs['langchain_mode'] != 'Disabled' and allow_upload and enable_text_upload
                fileup_output_text = gr.Textbox(visible=False)
                with gr.Accordion("Upload", open=False, visible=upload_visible and kwargs['actions_in_sidebar']):
                    fileup_output = gr.File(show_label=False,
                                            file_types=['.' + x for x in file_types],
                                            # file_types=['*', '*.*'],  # for iPhone etc. needs to be unconstrained else doesn't work with extension-based restrictions
                                            file_count="multiple",
                                            scale=1,
                                            min_width=0,
                                            elem_id="warning", elem_classes="feedback",
                                            )
                    if kwargs['actions_in_sidebar']:
                        max_quality = gr.Checkbox(label="Max Ingest Quality", value=kwargs['max_quality'],
                                                  visible=kwargs['visible_max_quality'] and not is_public)
                        gradio_upload_to_chatbot = gr.Checkbox(label="Add Doc to Chat",
                                                               value=kwargs['gradio_upload_to_chatbot'],
                                                               visible=kwargs[
                                                                           'visible_add_doc_to_chat'] and not is_public)
                    url_text = gr.Textbox(label=url_label,
                                          # placeholder="Enter Submits",
                                          max_lines=1,
                                          interactive=True,
                                          visible=kwargs['actions_in_sidebar'])
                    user_text_text = gr.Textbox(label='Paste Text',
                                                # placeholder="Enter Submits",
                                                interactive=True,
                                                visible=text_visible and kwargs['actions_in_sidebar'])

                database_visible = kwargs['langchain_mode'] != 'Disabled'
                langchain_choices0 = get_langchain_choices(selection_docs_state0)
                serp_visible = os.environ.get('SERPAPI_API_KEY') is not None and have_serpapi
                allowed_actions = [x for x in langchain_actions if x in visible_langchain_actions]
                default_action = allowed_actions[0] if len(allowed_actions) > 0 else None

                if not kwargs['actions_in_sidebar']:
                    max_quality = gr.Checkbox(label="Max Ingest Quality",
                                              value=kwargs['max_quality'],
                                              visible=kwargs['visible_max_quality'] and not is_public)
                    gradio_upload_to_chatbot = gr.Checkbox(label="Add Doc to Chat",
                                                           value=kwargs['gradio_upload_to_chatbot'],
                                                           visible=kwargs['visible_add_doc_to_chat'])

                if not kwargs['actions_in_sidebar']:
                    add_chat_history_to_context = gr.Checkbox(label="Include Chat History",
                                                              value=kwargs[
                                                                  'add_chat_history_to_context'],
                                                              visible=kwargs['visible_chat_history'])
                    add_search_to_context = gr.Checkbox(label="Include Web Search",
                                                        value=kwargs['add_search_to_context'],
                                                        visible=serp_visible)
                resources_acc_label = "Resources" if not is_public else "Collections"
                langchain_mode_radio_kwargs = dict(
                    choices=langchain_choices0,
                    value=kwargs['langchain_mode'],
                    label="Collections",
                    show_label=True,
                    visible=kwargs['langchain_mode'] != 'Disabled',
                    min_width=100)
                if is_public:
                    langchain_mode = gr.Radio(**langchain_mode_radio_kwargs)
                with gr.Accordion(resources_acc_label, open=False, visible=database_visible and not is_public):
                    if not is_public:
                        langchain_mode = gr.Radio(**langchain_mode_radio_kwargs)
                    if kwargs['actions_in_sidebar']:
                        add_chat_history_to_context = gr.Checkbox(label="Chat History",
                                                                  value=kwargs['add_chat_history_to_context'])
                        add_search_to_context = gr.Checkbox(label="Web Search",
                                                            value=kwargs['add_search_to_context'],
                                                            visible=serp_visible)
                    document_subset = gr.Radio([x.name for x in DocumentSubset],
                                               label="Subset",
                                               value=DocumentSubset.Relevant.name,
                                               interactive=True,
                                               visible=kwargs['visible_document_subset'] and not is_public,
                                               )
                    if kwargs['actions_in_sidebar']:
                        langchain_action = gr.Radio(
                            allowed_actions,
                            value=default_action,
                            label="Action",
                            visible=len(allowed_actions) > 1 and kwargs['visible_langchain_action_radio'])
                    allowed_agents = [x for x in langchain_agents_list if x in visible_langchain_agents]
                    if os.getenv('OPENAI_API_KEY') is None and LangChainAgent.JSON.value in allowed_agents:
                        allowed_agents.remove(LangChainAgent.JSON.value)
                    if os.getenv('OPENAI_API_KEY') is None and LangChainAgent.PYTHON.value in allowed_agents:
                        allowed_agents.remove(LangChainAgent.PYTHON.value)
                    if LangChainAgent.PANDAS.value in allowed_agents:
                        allowed_agents.remove(LangChainAgent.PANDAS.value)
                    langchain_agents = gr.Dropdown(
                        allowed_agents,
                        value=None,
                        label="Agents",
                        multiselect=True,
                        interactive=True,
                        visible=not is_public and len(allowed_agents) > 0,
                        elem_id="langchain_agents",
                        filterable=False)

                can_db_filter = kwargs['langchain_mode'] != 'Disabled' and kwargs['db_type'] in ['chroma',
                                                                                                 'chroma_old']
                document_choice_kwargs = dict(choices=docs_state0,
                                              label="Document",
                                              value=[DocumentChoice.ALL.value],
                                              interactive=True,
                                              multiselect=True,
                                              visible=can_db_filter,
                                              elem_id="multi-selection",
                                              allow_custom_value=False,
                                              )
                if kwargs['document_choice_in_sidebar']:
                    document_choice = gr.Dropdown(**document_choice_kwargs)

                visible_doc_track = upload_visible and kwargs['visible_doc_track'] and not kwargs[
                    'large_file_count_mode']
                row_doc_track = gr.Row(visible=visible_doc_track)
                with row_doc_track:
                    if kwargs['langchain_mode'] in langchain_modes_non_db:
                        doc_counts_str = "Pure LLM Mode"
                    else:
                        doc_counts_str = "Name: %s\nDocs: Unset\nChunks: Unset" % kwargs['langchain_mode']
                    text_doc_count = gr.Textbox(lines=3, label="Doc Counts", value=doc_counts_str,
                                                visible=visible_doc_track)
                    text_file_last = gr.Textbox(lines=1, label="Newest Doc", value=None, visible=visible_doc_track)
                    new_files_last = gr.Textbox(label="New Docs full paths as dict of full file names and content",
                                                value='{}',
                                                visible=False)
                    text_viewable_doc_count = gr.Textbox(lines=2, label=None, visible=False)

                with gr.Accordion("Image/Video Query", open=False, visible=have_vision_models):
                    image_file = gr.Gallery(value=kwargs['image_file'] if kwargs['image_file'] and any(
                        kwargs['image_file'].endswith(y) for y in IMAGE_EXTENSIONS) else None,
                                          label='Upload',
                                          show_label=False,
                                          type='filepath',
                                          elem_id="warning", elem_classes="feedback",
                                          visible=False,
                                          )
                    video_file = gr.Video(value=None,
                                          label='Upload',
                                          show_label=False,
                                          elem_id="warning", elem_classes="feedback",
                                          visible=False,
                                          )
                    audio_file = gr.Audio(value=None,
                                          label='Upload',
                                          show_label=False,
                                          elem_id="warning", elem_classes="feedback",
                                          visible=False,
                                          )

            col_tabs = gr.Column(elem_id="col-tabs", scale=10)
            with col_tabs, gr.Tabs():
                if kwargs['chat_tabless']:
                    chat_tab = gr.Row(visible=True)
                else:
                    chat_tab = gr.TabItem("Chat", visible=kwargs['visible_chat_tab'])
                with chat_tab:
                    if kwargs['langchain_mode'] == 'Disabled':
                        text_output_nochat = gr.Textbox(lines=5, label=output_label0, show_copy_button=True,
                                                        visible=not kwargs['chat'])
                    else:
                        # text looks a bit worse, but HTML links work
                        text_output_nochat = gr.HTML(label=output_label0, visible=not kwargs['chat'])
                    with gr.Row():
                        # NOCHAT
                        instruction_nochat = gr.Textbox(
                            lines=kwargs['input_lines'],
                            label=instruction_label_nochat,
                            placeholder=kwargs['placeholder_instruction'],
                            visible=not kwargs['chat'],
                        )
                        iinput_nochat = gr.Textbox(lines=4, label="Input context for Instruction",
                                                   placeholder=kwargs['placeholder_input'],
                                                   value=kwargs['iinput'],
                                                   visible=not kwargs['chat'])
                        submit_nochat = gr.Button("Submit", size='sm', visible=not kwargs['chat'])
                        flag_btn_nochat = gr.Button("Flag", size='sm', visible=not kwargs['chat'])
                        score_text_nochat = gr.Textbox("Response Score: NA", show_label=False,
                                                       visible=not kwargs['chat'])

                        submit_nochat_api = gr.Button("Submit nochat API", visible=False)

                        submit_nochat_api_plain = gr.Button("Submit nochat API Plain", visible=False)
                        inputs_dict_str = gr.Textbox(label='API input for nochat', show_label=False, visible=False)
                        text_output_nochat_api = gr.Textbox(lines=5, label='API nochat output', visible=False,
                                                            show_copy_button=True)

                        submit_verifier = gr.Button("Submit verifier", visible=False)
                        verifier_inputs_dict_str = gr.Textbox(label='Verifier input', show_label=False, visible=False)
                        text_output_verifier = gr.Textbox(lines=5, label='Verifier output', visible=False,
                                                          show_copy_button=True)

                        visible_upload = (allow_upload_to_user_data or
                                          allow_upload_to_my_data) and \
                                         kwargs['langchain_mode'] != 'Disabled'

                        # CHAT
                        col_chat = gr.Column(visible=kwargs['chat'])
                        with col_chat:
                            if kwargs['visible_ask_anything_high']:
                                attach_button, add_button, submit_buttons, instruction, instruction_mm, \
                                submit, retry_btn, undo, clear_chat_btn, save_chat_btn, stop_btn = \
                                    ask_block(kwargs, instruction_label, visible_upload, file_types, mic_sources_kwargs,
                                              mic_kwargs, noqueue_kwargs2, submit_kwargs, stop_kwargs)
                            visible_model_choice = bool(kwargs['model_lock']) and \
                                                   len(model_states) > 1 and \
                                                   kwargs['visible_visible_models']
                            with gr.Row(visible=not kwargs['actions_in_sidebar'] or visible_model_choice):
                                visible_models = gr.Dropdown(kwargs['all_possible_display_names'],
                                                             label="Visible Models",
                                                             value=visible_models_state0,
                                                             interactive=True,
                                                             multiselect=True,
                                                             visible=visible_model_choice,
                                                             elem_id="multi-selection-models" if kwargs[
                                                                                                     'max_visible_models'] is None or is_gradio_h2oai else None,
                                                             filterable=len(kwargs['all_possible_display_names']) > 5,
                                                             max_choices=kwargs['max_visible_models'],
                                                             )
                                if not image_tab_visible:
                                    visible_image_models = gr.Dropdown(**visible_image_models_kwargs)
                                mw0 = 100
                                with gr.Column(min_width=mw0):
                                    if not kwargs['actions_in_sidebar']:
                                        langchain_action = gr.Radio(
                                            allowed_actions,
                                            value=default_action,
                                            label='Action',
                                            show_label=visible_model_choice,
                                            visible=kwargs['visible_langchain_action_radio'],
                                            min_width=mw0)

                            text_output, text_output2, text_outputs = make_chatbots(output_label0, output_label0_model2,
                                                                                    **kwargs)

                            if not kwargs['visible_ask_anything_high']:
                                attach_button, add_button, submit_buttons, instruction, instruction_mm, \
                                submit, retry_btn, undo, clear_chat_btn, save_chat_btn, stop_btn = \
                                    ask_block(kwargs, instruction_label, visible_upload, file_types, mic_sources_kwargs,
                                              mic_kwargs, noqueue_kwargs2, submit_kwargs, stop_kwargs)
                            with gr.Row():
                                with gr.Column(visible=kwargs['score_model']):
                                    score_text = gr.Textbox(res_value,
                                                            show_label=False,
                                                            visible=True)
                                    score_text2 = gr.Textbox("Response Score2: NA", show_label=False,
                                                             visible=False and not kwargs['model_lock'])

                doc_selection_tab = gr.TabItem("Document Selection", visible=kwargs['visible_doc_selection_tab'])
                with doc_selection_tab:
                    if kwargs['langchain_mode'] in langchain_modes_non_db:
                        if langchain_mode == LangChainMode.DISABLED.value:
                            inactive_collection = "#### Document Q/A Disabled -- Chat only mode"
                        else:
                            dlabel1 = 'Choose Resources->Collections and Pick Collection'
                            inactive_collection = "#### Not Chatting with Any Collection\n%s" % dlabel1
                        active_collection = gr.Markdown(value=inactive_collection)
                    else:
                        dlabel1 = 'Select Subset of Document(s) for Chat with Collection: %s' % kwargs['langchain_mode']
                        active_collection = gr.Markdown(
                            value="#### Chatting with Collection: %s" % kwargs['langchain_mode'])
                    if not kwargs['document_choice_in_sidebar']:
                        document_choice_kwargs.update(dict(label=dlabel1))
                        document_choice = gr.Dropdown(**document_choice_kwargs)
                    with gr.Row():
                        with gr.Column():
                            document_source_substrings = gr.Dropdown([], label='Source substrings (post-search filter)',
                                                                     # info='Post-search filter',
                                                                     interactive=True,
                                                                     multiselect=True,
                                                                     visible=can_db_filter,
                                                                     allow_custom_value=True,
                                                                     scale=0,
                                                                     )
                        with gr.Column():
                            document_source_substrings_op = gr.Dropdown(['and', 'or'],
                                                                        label='Source substrings operation',
                                                                        interactive=True,
                                                                        multiselect=False,
                                                                        visible=can_db_filter,
                                                                        allow_custom_value=False,
                                                                        scale=0,
                                                                        )
                        with gr.Column():
                            document_content_substrings = gr.Dropdown([],
                                                                      label='Content substrings (search-time filter)',
                                                                      # info="Search-time filter of list of words to pass to where_document={'$contains': word list}",
                                                                      interactive=True,
                                                                      multiselect=True,
                                                                      visible=can_db_filter,
                                                                      allow_custom_value=True,
                                                                      scale=0,
                                                                      )
                        with gr.Column():
                            document_content_substrings_op = gr.Dropdown(['and', 'or'],
                                                                         label='Content substrings operation',
                                                                         interactive=True,
                                                                         multiselect=False,
                                                                         visible=can_db_filter,
                                                                         allow_custom_value=False,
                                                                         scale=0,
                                                                         )
                    sources_visible = kwargs['langchain_mode'] != 'Disabled' and enable_sources_list
                    with gr.Row():
                        with gr.Column(scale=1):
                            get_sources_btn = gr.Button(value="Update UI with Document(s) from DB", scale=0, size='sm',
                                                        visible=sources_visible and kwargs['large_file_count_mode'])
                            # handle API get sources
                            get_sources_api_btn = gr.Button(visible=False)
                            get_sources_api_text = gr.Textbox(visible=False)

                            get_document_api_btn = gr.Button(visible=False)
                            get_document_api_text = gr.Textbox(visible=False)

                            show_sources_btn = gr.Button(value="Show Sources from DB", scale=0, size='sm',
                                                         visible=sources_visible and kwargs['large_file_count_mode'])
                            delete_sources_btn = gr.Button(value="Delete Selected (not by substrings) Sources from DB",
                                                           scale=0, size='sm',
                                                           visible=sources_visible)
                            refresh_sources_btn = gr.Button(value="Update DB with new/changed files on disk", scale=0,
                                                            size='sm',
                                                            visible=sources_visible and allow_upload_to_user_data)
                        with gr.Column(scale=4):
                            pass
                    visible_add_remove_collection = visible_upload
                    with gr.Row():
                        with gr.Column(scale=1):
                            add_placeholder = "e.g. UserData2, shared, user_path2" \
                                if not is_public else "e.g. MyData2, personal (optional)"
                            remove_placeholder = "e.g. UserData2" if not is_public else "e.g. MyData2"
                            new_langchain_mode_text = gr.Textbox(value="", visible=visible_add_remove_collection,
                                                                 label='Add Collection',
                                                                 placeholder=add_placeholder,
                                                                 interactive=True)
                            remove_langchain_mode_text = gr.Textbox(value="", visible=visible_add_remove_collection,
                                                                    label='Remove Collection from UI',
                                                                    placeholder=remove_placeholder,
                                                                    interactive=True)
                            purge_langchain_mode_text = gr.Textbox(value="", visible=visible_add_remove_collection,
                                                                   label='Purge Collection (UI, DB, & source files)',
                                                                   placeholder=remove_placeholder,
                                                                   interactive=True)
                            sync_sources_btn = gr.Button(
                                value="Synchronize DB and UI [only required if did not login and have shared docs]",
                                scale=0, size='sm',
                                visible=sources_visible and allow_upload_to_user_data and not kwargs[
                                    'large_file_count_mode'])
                            load_langchain = gr.Button(
                                value="Load Collections State [only required if logged in another user ", scale=0,
                                size='sm',
                                visible=False and allow_upload_to_user_data and
                                        kwargs['langchain_mode'] != 'Disabled')
                        with gr.Column(scale=5):
                            if kwargs['langchain_mode'] != 'Disabled' and visible_add_remove_collection:
                                df0 = get_df_langchain_mode_paths(selection_docs_state0, None, dbs1=dbs)
                            else:
                                df0 = pd.DataFrame(None)
                            langchain_mode_path_text = gr.Dataframe(value=df0,
                                                                    visible=visible_add_remove_collection,
                                                                    label='LangChain Mode-Path',
                                                                    show_label=False,
                                                                    interactive=False)

                    sources_row = gr.Row(visible=kwargs['langchain_mode'] != 'Disabled' and enable_sources_list,
                                         equal_height=False)
                    with sources_row:
                        with gr.Column(scale=1):
                            file_source = gr.File(interactive=False,
                                                  label="Download File w/Sources")
                        with gr.Column(scale=2):
                            sources_text = gr.HTML(label='Sources Added')

                    doc_exception_text = gr.Textbox(value="", label='Document Exceptions',
                                                    interactive=False,
                                                    visible=kwargs['langchain_mode'] != 'Disabled')
                    if have_arxiv and have_librosa:
                        file_types_extra = ' URL YouTube ArXiv TEXT'
                    elif have_librosa:
                        file_types_extra = ' URL YouTube TEXT'
                    elif have_arxiv:
                        file_types_extra = ' URL ArXiv TEXT'
                    else:
                        file_types_extra = ' URL TEXT'
                    file_types_str = ' '.join(file_types) + file_types_extra
                    gr.Textbox(value=file_types_str, label='Document Types Supported',
                               lines=2,
                               interactive=False,
                               visible=kwargs['langchain_mode'] != 'Disabled')

                doc_view_tab = gr.TabItem("Document Viewer", visible=kwargs['visible_doc_view_tab'])
                with doc_view_tab:
                    with gr.Row(visible=kwargs['langchain_mode'] != 'Disabled'):
                        with gr.Column(scale=2):
                            get_viewable_sources_btn = gr.Button(value="Update UI with Document(s) from DB", scale=0,
                                                                 size='sm',
                                                                 visible=sources_visible and kwargs[
                                                                     'large_file_count_mode'])
                            view_document_choice = gr.Dropdown(viewable_docs_state0,
                                                               label="Select Single Document to View",
                                                               value=None,
                                                               interactive=True,
                                                               multiselect=False,
                                                               visible=True,
                                                               elem_id="single-selection",
                                                               )
                            info_view_raw = "Raw text shown if render of original doc fails"
                            if is_public:
                                info_view_raw += " (Up to %s chunks in public portal)" % kwargs['max_raw_chunks']
                            view_raw_text_checkbox = gr.Checkbox(label="View Database Text", value=False,
                                                                 info=info_view_raw,
                                                                 visible=kwargs['db_type'] in ['chroma', 'chroma_old'])
                        with gr.Column(scale=4):
                            pass
                    doc_view = gr.HTML(visible=False)
                    doc_view2 = gr.Dataframe(visible=False)
                    doc_view3 = gr.JSON(visible=False)
                    doc_view4 = gr.Markdown(visible=False)
                    doc_view5 = gr.HTML(visible=False)
                    if have_gradio_pdf:
                        from gradio_pdf import PDF
                        doc_view6 = PDF(visible=False)
                    else:
                        doc_view6 = gr.HTML(visible=False)
                    doc_view7 = gr.Audio(visible=False)
                    doc_view8 = gr.Video(visible=False)

                image_tab = gr.TabItem("Image Control", visible=image_tab_visible)
                with image_tab:
                    if image_tab_visible:
                        visible_image_models = gr.Dropdown(**visible_image_models_kwargs)
                    with gr.Row(visible=image_control_panels_visible):
                        image_control = gr.Image(label="Input Image", type='filepath', elem_id="warning",
                                                 elem_classes="feedback")
                        image_style = gr.Image(label="Style Image", type='filepath', elem_id="warning",
                                               elem_classes="feedback")
                        image_output = gr.Image(label="Output Image", type='filepath', elem_id="warning",
                                                elem_classes="feedback")
                    image_prompt = gr.Textbox(label="Prompt", visible=image_control_panels_visible and \
                                                                      (image_gen_visible or image_change_visible))
                    with gr.Row(visible=image_control_panels_visible):
                        generate_btn = gr.Button("Generate by Prompt", visible=image_gen_visible)
                        change_btn = gr.Button("Change Image by Prompt", visible=image_change_visible)
                        style_btn = gr.Button("Apply Style", visible=False)
                        # image_upload = # FIXME, go into db

                chat_history_tab = gr.TabItem("Chat History", visible=kwargs['visible_chat_history_tab'])
                with chat_history_tab:
                    with gr.Row():
                        with gr.Column(scale=1):
                            remove_chat_btn = gr.Button(value="Remove Selected Saved Chats", visible=True, size='sm')
                            flag_btn = gr.Button("Flag Current Chat", size='sm')
                            export_chats_btn = gr.Button(value="Export Chats to Download", size='sm')
                        with gr.Column(scale=4):
                            pass
                    with gr.Row():
                        chats_file = gr.File(interactive=False, label="Download Exported Chats")
                        chatsup_output = gr.File(label="Upload Chat File(s)",
                                                 file_types=['.json'],
                                                 file_count='multiple',
                                                 elem_id="warning", elem_classes="feedback")
                    with gr.Row():
                        if 'mbart-' in kwargs['model_lower']:
                            src_lang = gr.Dropdown(list(languages_covered().keys()),
                                                   value=kwargs['src_lang'],
                                                   label="Input Language")
                            tgt_lang = gr.Dropdown(list(languages_covered().keys()),
                                                   value=kwargs['tgt_lang'],
                                                   label="Output Language")

                    chat_exception_text = gr.Textbox(value="", visible=True, label='Chat Exceptions',
                                                     interactive=False)
                    with gr.Row():
                        count_chat_tokens_btn = gr.Button(value="Count Chat Tokens",
                                                          visible=not is_public and not kwargs['model_lock'],
                                                          interactive=not is_public, size='sm')
                        chat_token_count = gr.Textbox(label="Chat Token Count Result", value=None,
                                                      visible=not is_public and not kwargs['model_lock'],
                                                      interactive=False)
                expert_tab = gr.TabItem("Expert", visible=kwargs['visible_expert_tab'])
                with expert_tab:
                    gr.Markdown("Prompt Control")
                    with gr.Row():
                        with gr.Column():
                            if not kwargs['visible_models_tab']:
                                # only show here if no models tab
                                prompt_type = get_prompt_type1(**kwargs)
                                prompt_type2 = get_prompt_type2(**kwargs)

                            system_prompt_type = gr.Dropdown(label="System Prompt Type",
                                                             info="Choose System Prompt Type",
                                                             value=kwargs['system_prompt'],
                                                             choices=get_system_prompts(),
                                                             filterable=True,
                                                             )
                            system_prompt = gr.Textbox(label='System Prompt',
                                                       info="Filled by choice above, or can enter your own custom system prompt.  auto means automatic, which will auto-switch to DocQA prompt when using collections.",
                                                       value=kwargs['system_prompt'], lines=2)

                            def show_sys(x):
                                return x

                            system_prompt_type.change(fn=show_sys, inputs=system_prompt_type, outputs=system_prompt,
                                                      **noqueue_kwargs)

                            context = gr.Textbox(lines=2, label="System Pre-Context",
                                                 info="Directly pre-appended without prompt processing (before Pre-Conversation)",
                                                 value=kwargs['context'])
                            chat_conversation = gr.Textbox(lines=2, label="Pre-Conversation",
                                                           info="Pre-append conversation for instruct/chat models as List of tuple of (human, bot)",
                                                           value=kwargs['chat_conversation'])
                            text_context_list = gr.Textbox(lines=2, label="Text Doc Q/A",
                                                           info="List of strings, for document Q/A, for bypassing database (i.e. also works in LLM Mode)",
                                                           value=kwargs['chat_conversation'],
                                                           visible=not is_public,  # primarily meant for API
                                                           )
                            iinput = gr.Textbox(lines=2, label="Input for Instruct prompt types",
                                                info="If given for document query, added after query",
                                                value=kwargs['iinput'],
                                                placeholder=kwargs['placeholder_input'],
                                                interactive=not is_public)
                        with gr.Column():
                            pre_prompt_query = gr.Textbox(label="Query Pre-Prompt",
                                                          info="In prompt template, added before document text chunks",
                                                          value=kwargs['pre_prompt_query'] or '')
                            prompt_query = gr.Textbox(label="Query Prompt",
                                                      info="Added after documents",
                                                      value=kwargs['prompt_query'] or '')
                            pre_prompt_summary = gr.Textbox(label="Summary Pre-Prompt",
                                                            info="In prompt template, added before documents",
                                                            value=kwargs['pre_prompt_summary'] or '')
                            prompt_summary = gr.Textbox(label="Summary Prompt",
                                                        info="In prompt template, added after documents text chunks (if query given, 'Focusing on {query}, ' is pre-appended)",
                                                        value=kwargs['prompt_summary'] or '')
                            hyde_llm_prompt = gr.Textbox(label="HYDE LLM Prompt",
                                                         info="When doing HYDE, this is first prompt, and in template the user query comes right after this.",
                                                         value=kwargs['hyde_llm_prompt'] or '')
                            llava_prompt_type = gr.Dropdown(label="LLaVa LLM Prompt Type",
                                                            info="Pick pre-defined LLaVa prompt",
                                                            value=kwargs['llava_prompt'],
                                                            choices=get_llava_prompts(),
                                                            filterable=True,
                                                            )
                            llava_prompt = gr.Textbox(label="LLaVa LLM Prompt",
                                                      info="LLaVa prompt",
                                                      value=kwargs['llava_prompt'],
                                                      lines=2)
                            user_prompt_for_fake_system_prompt = gr.Textbox(label="User System Prompt",
                                                                            info="user part of pre-conversation if LLM doesn't handle system prompt.",
                                                                            value=kwargs[
                                                                                      'user_prompt_for_fake_system_prompt'] or '')
                            json_object_prompt = gr.Textbox(label="JSON Object Prompt",
                                                            info="prompt for getting LLM to do JSON object",
                                                            value=kwargs['json_object_prompt'] or '')
                            json_object_prompt_simpler = gr.Textbox(label="Simpler JSON Object Prompt",
                                                                    info="Simpler prompt for getting LLM to do JSON object (for MistralAI)",
                                                                    value=kwargs['json_object_prompt_simpler'] or '')
                            json_code_prompt = gr.Textbox(label="JSON Code Prompt",
                                                          info="prompt for getting LLm to do JSON in code block",
                                                          value=kwargs['json_code_prompt'] or '')
                            json_code_prompt_if_no_schema = gr.Textbox(label="Schema instructions Prompt",
                                                                       info="prompt for LLM to use when no schema but need schema to obey rules",
                                                                       value=kwargs[
                                                                                 'json_code_prompt_if_no_schema'] or '')
                            json_schema_instruction = gr.Textbox(label="JSON Schema Prompt",
                                                                 info="prompt for LLM to use schema",
                                                                 value=kwargs['json_schema_instruction'] or '')

                            def show_llava(x):
                                return x

                            llava_prompt_type.change(fn=show_llava, inputs=llava_prompt_type, outputs=llava_prompt,
                                                     **noqueue_kwargs)

                    gr.Markdown("Document Control")
                    with gr.Row(visible=not is_public):
                        image_audio_loaders = gr.CheckboxGroup(image_audio_loaders_options,
                                                               label="Force Image-Audio Reader",
                                                               value=image_audio_loaders_options0)
                        pdf_loaders = gr.CheckboxGroup(pdf_loaders_options,
                                                       label="Force PDF Reader",
                                                       value=pdf_loaders_options0)
                        url_loaders = gr.CheckboxGroup(url_loaders_options,
                                                       label="Force URL Reader",
                                                       info="Set env CRAWL_DEPTH to control depth for Scrape, default is 1 (given page + links on that page)",
                                                       value=url_loaders_options0)
                        jq_schema = gr.Textbox(label="JSON jq_schema", value=jq_schema0)
                        extract_frames = gr.Slider(value=kwargs['extract_frames'] if not is_public else 5,
                                                   step=1,
                                                   minimum=0,
                                                   maximum=5 if is_public else max(kwargs['extract_frames'], 1000),
                                                   label="Number of unique images to extract from videos",
                                                   info="If 0, just audio extracted if enabled",
                                                   visible=have_fiftyone)

                        min_top_k_docs, max_top_k_docs, label_top_k_docs = get_minmax_top_k_docs(is_public, True)
                        top_k_docs = gr.Slider(minimum=min_top_k_docs, maximum=max_top_k_docs, step=1,
                                               value=kwargs['top_k_docs'],
                                               label=label_top_k_docs,
                                               # info="For LangChain",
                                               visible=kwargs['langchain_mode'] != 'Disabled',
                                               interactive=not is_public)
                        chunk = gr.components.Checkbox(value=kwargs['chunk'],
                                                       label="Whether to chunk documents",
                                                       info="For LangChain",
                                                       visible=kwargs['langchain_mode'] != 'Disabled',
                                                       interactive=not is_public)
                        chunk_size = gr.Number(value=kwargs['chunk_size'],
                                               label="Chunk size for document chunking",
                                               info="For LangChain (ignored if chunk=False)",
                                               minimum=128,
                                               maximum=2048,
                                               visible=kwargs['langchain_mode'] != 'Disabled',
                                               interactive=not is_public,
                                               precision=0)
                        docs_ordering_type = gr.Radio(
                            docs_ordering_types,
                            value=kwargs['docs_ordering_type'],
                            label="Document Sorting in LLM Context",
                            visible=True)
                        docs_token_handling = gr.Radio(
                            docs_token_handlings,
                            value=kwargs['docs_token_handling'],
                            label="Document Handling Mode for filling LLM Context",
                            visible=True)
                        docs_joiner = gr.Textbox(label="String to join lists and documents",
                                                 value=kwargs['docs_joiner'] or docs_joiner_default)
                        max_hyde_level = 0 if is_public else 5
                        hyde_level = gr.Slider(minimum=0, maximum=max_hyde_level, step=1,
                                               value=kwargs['hyde_level'],
                                               label='HYDE level',
                                               info="Whether to use HYDE approach for LLM getting answer to embed (0=disabled, 1=non-doc LLM answer, 2=doc-based LLM answer)",
                                               visible=kwargs['langchain_mode'] != 'Disabled',
                                               interactive=not is_public)
                        hyde_template = gr.components.Textbox(value='auto',
                                                              label="HYDE Embedding Template",
                                                              info="HYDE approach for LLM getting answer to embed ('auto' means automatic, else enter template like '{query}'",
                                                              visible=True)
                        hyde_show_only_final = gr.components.Checkbox(value=kwargs['hyde_show_only_final'],
                                                                      label="Only final HYDE shown",
                                                                      info="Whether to only show final HYDE result",
                                                                      visible=True)
                        doc_json_mode = gr.components.Checkbox(value=kwargs['doc_json_mode'],
                                                               label="JSON docs mode",
                                                               info="Whether to pass JSON to and get JSON back from LLM",
                                                               visible=True)
                        metadata_in_context = gr.components.Textbox(value=str(kwargs['metadata_in_context']),
                                                                    label="Metadata keys to include in LLM context (all, auto, or [key1, key2, ...] where strings are quoted)",
                                                                    visible=True)

                        embed = gr.components.Checkbox(value=True,
                                                       label="Embed text",
                                                       info="For LangChain, whether to embed text",
                                                       visible=False)
                    gr.Markdown("LLM Control")
                    with gr.Row():
                        stream_output = gr.components.Checkbox(label="Stream output",
                                                               value=kwargs['stream_output'])
                        do_sample = gr.Checkbox(label="Sample",
                                                info="Enable sampler (required for use of temperature, top_p, top_k).  If temperature=0 is set, this is forced to False.",
                                                value=kwargs['do_sample'],
                                                visible=False)
                        seed = gr.Number(value=0,
                                         minimum=0,
                                         step=1,
                                         label="Seed for sampling.  0 makes random seed",
                                         )
                        max_time = gr.Slider(minimum=0, maximum=kwargs['max_max_time'], step=1,
                                             value=min(kwargs['max_max_time'],
                                                       kwargs['max_time']), label="Max. time",
                                             info="Max. time to search optimal output.")
                        temperature = gr.Slider(minimum=0, maximum=2,
                                                value=kwargs['temperature'],
                                                label="Temperature",
                                                info="Lower is deterministic, higher more creative (e.g. 0.3 to 0.75)")
                        top_p = gr.Slider(minimum=1e-3, maximum=1.0,
                                          value=kwargs['top_p'], label="Top p",
                                          info="Cumulative probability of tokens to sample from (e.g. 0.7)")
                        top_k = gr.Slider(
                            minimum=1, maximum=100, step=1,
                            value=kwargs['top_k'], label="Top k",
                            info='Num. tokens to sample from (e.g. 5 to 70)'
                        )
                        penalty_alpha = gr.Slider(
                            minimum=0.0, maximum=2.0, step=0.01,
                            value=kwargs['penalty_alpha'], label="penalty_alpha",
                            info='penalty_alpha>0 and top_k>1 enables contrastive search'
                        )
                        # FIXME: https://github.com/h2oai/h2ogpt/issues/106
                        if os.getenv('TESTINGFAIL'):
                            max_beams = 8 if not (memory_restriction_level or is_public) else 1
                        else:
                            max_beams = 1
                        num_beams = gr.Slider(minimum=1, maximum=max_beams, step=1,
                                              value=min(max_beams, kwargs['num_beams']), label="Beams",
                                              info="Number of searches for optimal overall probability.  "
                                                   "Uses more GPU memory/compute",
                                              interactive=False, visible=max_beams > 1)
                        max_max_new_tokens = get_max_max_new_tokens(model_state0, **kwargs)
                        max_new_tokens = gr.Slider(
                            minimum=1, maximum=max_max_new_tokens, step=1,
                            value=min(max_max_new_tokens, kwargs['max_new_tokens']), label="Max output length",
                        )
                        min_new_tokens = gr.Slider(
                            minimum=0, maximum=max_max_new_tokens, step=1,
                            value=min(max_max_new_tokens, kwargs['min_new_tokens']), label="Min output length",
                        )
                        max_new_tokens2 = gr.Slider(
                            minimum=1, maximum=max_max_new_tokens, step=1,
                            value=min(max_max_new_tokens, kwargs['max_new_tokens']), label="Max output length 2",
                            visible=False and not kwargs['model_lock'],
                        )
                        min_new_tokens2 = gr.Slider(
                            minimum=0, maximum=max_max_new_tokens, step=1,
                            value=min(max_max_new_tokens, kwargs['min_new_tokens']), label="Min output length 2",
                            visible=False and not kwargs['model_lock'],
                        )
                        min_max_new_tokens = gr.Slider(
                            minimum=1, maximum=max_max_new_tokens, step=1,
                            value=min(max_max_new_tokens, kwargs['min_max_new_tokens']),
                            label="Min. of Max output length",
                            visible=not is_public,
                        )
                        max_input_tokens = gr.Number(
                            minimum=-1 if not is_public else kwargs['max_input_tokens'],
                            maximum=128 * 1024 if not is_public else kwargs['max_input_tokens'],
                            step=1,
                            value=kwargs.get('max_input_tokens', -1),
                            label="Max input length (treat as if model has more limited context, e.g. for context-filling when top_k_docs=-1)",
                            visible=not is_public,
                        )
                        max_total_input_tokens = gr.Number(
                            minimum=-1 if not is_public else kwargs['max_total_input_tokens'],
                            maximum=128 * 1024 if not is_public else kwargs['max_total_input_tokens'],
                            step=1,
                            value=kwargs.get('max_total_input_tokens', -1),
                            label="Max input length across all LLM calls when doing summarization/extraction",
                            visible=not is_public,
                        )
                        early_stopping = gr.Checkbox(label="EarlyStopping", info="Stop early in beam search",
                                                     value=kwargs['early_stopping'], visible=max_beams > 1)
                        repetition_penalty = gr.Slider(minimum=0.01, maximum=3.0,
                                                       value=kwargs['repetition_penalty'],
                                                       label="Repetition Penalty")
                        num_return_sequences = gr.Slider(minimum=1, maximum=10, step=1,
                                                         value=kwargs['num_return_sequences'],
                                                         label="Number Returns", info="Must be <= num_beams",
                                                         interactive=not is_public, visible=max_beams > 1)
                        chat = gr.components.Checkbox(label="Chat mode", value=kwargs['chat'],
                                                      visible=False,  # no longer support nochat in UI
                                                      interactive=not is_public,
                                                      )

                        response_format = gr.Radio(response_formats,
                                                   label="response_format",
                                                   value=kwargs['response_format'],
                                                   interactive=True,
                                                   visible=True,
                                                   )
                        guided_json = gr.components.Textbox(value=kwargs['guided_json'],
                                                            label="guided_json as string, will be converted to dict via json.loads",
                                                            info="https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters-for-chat-api",
                                                            visible=True)
                        guided_regex = gr.components.Textbox(value=kwargs['guided_regex'],
                                                             label="guided_regex",
                                                             info="https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters-for-chat-api",
                                                             visible=True)
                        guided_choice = gr.components.Textbox(value=kwargs['guided_choice'],
                                                              label="guided_choice",
                                                              info="https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters-for-chat-api",
                                                              visible=True)
                        guided_grammar = gr.components.Textbox(value=kwargs['guided_grammar'],
                                                               label="guided_grammar",
                                                               info="https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters-for-chat-api",
                                                               visible=True)
                        guided_whitespace_pattern = gr.components.Textbox(
                            value=kwargs['guided_whitespace_pattern'] or '',
                            label="guided_whitespace_pattern, empty string means None",
                            info="https://github.com/vllm-project/vllm/pull/4305/files",
                            visible=not is_public)
                        images_num_max = gr.Number(
                            label='Number of Images per LLM call, -1 is auto mode, 0 is avoid using images',
                            value=kwargs['images_num_max'] or 0,
                            visible=not is_public)
                        image_resolution = gr.Textbox(label='Resolution in (nx, ny)', value=kwargs['image_resolution'],
                                                      visible=not is_public)
                        image_format = gr.Textbox(label='Image format', value=kwargs['image_format'],
                                                  visible=not is_public)
                        rotate_align_resize_image = gr.Checkbox(
                            label="Whether to apply rotation, align, resize before giving to LLM.",
                            value=kwargs['rotate_align_resize_image'],
                            visible=not is_public)
                        video_frame_period = gr.Number(label="Period of frames to use from video.  0 means auto",
                                                       value=kwargs['video_frame_period'] or 0,
                                                       visible=not is_public)

                        image_batch_image_prompt = gr.Textbox(label="Image batch prompt",
                                                              value=kwargs['image_batch_image_prompt'])
                        image_batch_final_prompt = gr.Textbox(label="Image batch prompt",
                                                              value=kwargs['image_batch_final_prompt'])

                        visible_vision_models = gr.Dropdown(['auto'] + kwargs['all_possible_vision_display_names'],
                                                            label="Visible Image Models",
                                                            # value=visible_vision_models_state0,  # not changing yet
                                                            value='auto',
                                                            interactive=True,
                                                            multiselect=False,
                                                            visible=visible_model_choice and not is_public,
                                                            filterable=len(
                                                                kwargs['all_possible_vision_display_names']) > 5,
                                                            )
                        image_batch_stream = gr.Checkbox(label="Whether to stream batching of images.",
                                                         value=kwargs['image_batch_stream'],
                                                         visible=not is_public)

                    clone_visible = visible = kwargs['enable_tts'] and kwargs['tts_model'].startswith('tts_models/')
                    if clone_visible:
                        markdown_label = "Speech Control and Voice Cloning"
                    else:
                        markdown_label = "Speech Control"
                    audio_visible = kwargs['enable_tts'] and kwargs['tts_model']
                    gr.Markdown(markdown_label, visible=audio_visible)
                    with gr.Row(visible=audio_visible):
                        if audio_visible:
                            speech_human = gr.Audio(value=None,
                                                    label="Generated Human Speech",
                                                    type="numpy",
                                                    streaming=True,
                                                    interactive=False,
                                                    show_label=True,
                                                    autoplay=True,
                                                    elem_id='human_audio',
                                                    visible=audio_visible)
                            speech_bot = gr.Audio(value=None,
                                                  label="Generated Bot Speech",
                                                  type="numpy",
                                                  streaming=True,
                                                  interactive=False,
                                                  show_label=True,
                                                  autoplay=True,
                                                  elem_id='bot_audio',
                                                  visible=audio_visible)
                            speech_bot2 = gr.Audio(value=None,
                                                   label="Generated Bot 2 Speech",
                                                   type="numpy",
                                                   streaming=True,
                                                   interactive=False,
                                                   show_label=True,
                                                   autoplay=False,
                                                   visible=False,
                                                   elem_id='bot2_audio')
                            text_speech = gr.Textbox(visible=False)
                            text_speech_out = gr.Textbox(visible=False)
                        else:
                            # Ensure not streaming media, just webconnect, if not doing TTS
                            speech_human = gr.Textbox(visible=False)
                            speech_bot = gr.Textbox(visible=False)
                            speech_bot2 = gr.Textbox(visible=False)
                            text_speech = gr.Textbox(visible=False)
                            text_speech_out = gr.Textbox(visible=False)
                        speak_inputs_dict_str = gr.Textbox(label='API input for speak_text_plain_api', show_label=False,
                                                           visible=False)

                        if kwargs['enable_tts'] and kwargs['tts_model'].startswith('tts_models/'):
                            from src.tts_coqui import get_languages_gr
                            tts_language = get_languages_gr(visible=True, value=kwargs['tts_language'])
                        else:
                            tts_language = gr.Dropdown(visible=False)

                        if audio_visible:
                            model_base = os.getenv('H2OGPT_MODEL_BASE', 'models/')
                            female_voice = os.path.join(model_base, "female.wav")
                            ref_voice_clone = gr.Audio(
                                label="File for Clone (x resets)",
                                type="filepath",
                                value=female_voice if os.path.isfile(female_voice) else None,
                                # max_length=30 if is_public else None,
                                visible=clone_visible,
                            )
                            ref_voice_clone.upload(process_audio, inputs=ref_voice_clone, outputs=ref_voice_clone)
                        else:
                            ref_voice_clone = gr.Textbox(visible=False)

                        if audio_visible:
                            mic_voice_clone = gr.Audio(
                                label="Mic for Clone (x resets)",
                                type="filepath",
                                **mic_sources_kwargs,
                                # max_length=30 if is_public else None,
                                visible=clone_visible,
                            )
                            mic_voice_clone.upload(process_audio, inputs=mic_voice_clone, outputs=mic_voice_clone)
                        else:
                            mic_voice_clone = gr.Textbox(visible=False)
                        choose_mic_voice_clone = gr.Checkbox(
                            label="Use Mic for Cloning",
                            value=False,
                            info="If unchecked, uses File",
                            visible=clone_visible,
                        )
                        role_name_to_add = gr.Textbox(value='', info="Name of Speaker to add", label="Speaker Style",
                                                      visible=clone_visible)
                        add_role = gr.Button(value="Clone Voice for new Speech Style", visible=clone_visible)

                        def add_role_func(name, file, mic, roles1, use_mic):
                            if use_mic and os.path.isfile(mic):
                                roles1[name] = mic
                            elif os.path.isfile(file):
                                roles1[name] = file
                            roles1[name] = process_audio(roles1[name])
                            return gr.Dropdown(choices=list(roles1.keys())), roles1

                        add_role_event = add_role.click(add_role_func,
                                                        inputs=[role_name_to_add, ref_voice_clone, mic_voice_clone,
                                                                roles_state,
                                                                choose_mic_voice_clone],
                                                        outputs=[chatbot_role, roles_state],
                                                        api_name='add_role' if allow_api else False,
                                                        **noqueue_kwargs2,
                                                        )
                models_tab = gr.TabItem("Models", visible=kwargs['visible_models_tab'])
                with models_tab:
                    load_msg = "Load (Download) Model" if not is_public \
                        else "LOAD-UNLOAD DISABLED FOR HOSTED DEMO"
                    if kwargs['base_model'] not in ['', None, no_model_str] and kwargs['inference_server'] in ['', None,
                                                                                                               no_server_str]:
                        load_msg += '   [WARNING: Avoid --base_model on CLI for memory efficient Load-Unload]'
                    load_msg2 = load_msg + "2"
                    variant_load_msg = 'primary' if not is_public else 'secondary'
                    with gr.Row():
                        n_gpus_list = [str(x) for x in list(range(-1, n_gpus))]
                        with gr.Column():
                            with gr.Row():
                                with gr.Column(scale=10, visible=not kwargs['model_lock']):
                                    load_models_button = gr.Button('Load Model Names from Server',
                                                                   variant=variant_load_msg, scale=0,
                                                                   size='sm', interactive=not is_public)
                                    load_model_button = gr.Button(load_msg, variant=variant_load_msg, scale=0,
                                                                  size='sm', interactive=not is_public)
                                    unload_model_button = gr.Button("UnLoad Model", variant=variant_load_msg, scale=0,
                                                                    size='sm', interactive=not is_public)
                                    with gr.Row():
                                        with gr.Column():
                                            model_choice = gr.Dropdown(model_options_state.value[0],
                                                                       label="Choose/Enter Base Model (HF name, TheBloke, file, URL)",
                                                                       value=kwargs['base_model'] or
                                                                             model_options_state.value[0],
                                                                       allow_custom_value=not is_public)
                                            lora_choice = gr.Dropdown(lora_options_state.value[0],
                                                                      label="Choose/Enter LORA",
                                                                      value=kwargs['lora_weights'] or
                                                                            lora_options_state.value[0],
                                                                      visible=kwargs['show_lora'],
                                                                      allow_custom_value=not is_public)
                                            server_choice = gr.Dropdown(server_options_state.value[0],
                                                                        label="Choose/Enter Server",
                                                                        value=kwargs['inference_server'] or
                                                                              server_options_state.value[0],
                                                                        visible=not is_public,
                                                                        allow_custom_value=not is_public)
                                            if kwargs['visible_models_tab']:
                                                prompt_type = get_prompt_type1(**kwargs)
                                        with gr.Column():
                                            model_used = gr.Textbox(label="Current Model", value=kwargs['base_model'],
                                                                    interactive=False)
                                            lora_used = gr.Textbox(label="Current LORA", value=kwargs['lora_weights'],
                                                                   visible=kwargs['show_lora'], interactive=False)
                                            server_used = gr.Textbox(label="Current Server",
                                                                     value=kwargs['inference_server'],
                                                                     visible=bool(
                                                                         kwargs['inference_server']) and not is_public,
                                                                     interactive=False)
                                with gr.Column(scale=1, visible=not kwargs['model_lock']):
                                    with gr.Accordion("Precision", open=False, visible=True):
                                        model_load8bit_checkbox = gr.components.Checkbox(
                                            label="Load 8-bit [requires support]",
                                            value=kwargs['load_8bit'], interactive=not is_public)
                                        model_load4bit_checkbox = gr.components.Checkbox(
                                            label="Load 4-bit [requires support]",
                                            value=kwargs['load_4bit'], interactive=not is_public)
                                        model_low_bit_mode = gr.Slider(value=kwargs['low_bit_mode'],
                                                                       minimum=0, maximum=4, step=1,
                                                                       label="low_bit_mode",
                                                                       info="0: no quantization config 1: change compute 2: nf4 3: double quant 4: 2 and 3")
                                    with gr.Accordion("GPU", open=False, visible=n_gpus != 0):
                                        model_use_cpu_checkbox = gr.components.Checkbox(
                                            label="Use CPU even if have GPUs",
                                            value=False,
                                            interactive=not is_public)
                                        model_use_gpu_id_checkbox = gr.components.Checkbox(
                                            label="Choose Devices [If not Checked, use all GPUs]",
                                            value=kwargs['use_gpu_id'],
                                            interactive=not is_public)
                                        llama_multi_gpu_info = "LLaMa.cpp does not support multi-GPU GPU selection, run h2oGPT with env CUDA_VISIBLE_DEVICES set to which GPU to use, else all are used."
                                        model_gpu = gr.Dropdown(n_gpus_list,
                                                                label="GPU ID [-1 = all GPUs, if Choose is enabled]",
                                                                info=llama_multi_gpu_info,
                                                                value=kwargs['gpu_id'],
                                                                interactive=not is_public)
                                    with gr.Accordion("Add-ons", open=False, visible=True):
                                        model_attention_sinks = gr.components.Checkbox(
                                            label="Enable Attention Sinks [requires support]",
                                            value=kwargs['attention_sinks'], interactive=not is_public)
                                        model_truncation_generation = gr.components.Checkbox(
                                            label="Truncate generation (disable for attention sinks, enforced if required)",
                                            value=kwargs['truncation_generation'], interactive=not is_public)
                                        model_sink_dict = gr.Textbox(value=str(kwargs['sink_dict'] or {}),
                                                                     label="sink_dict")
                                        model_load_gptq = gr.Textbox(label="gptq",
                                                                     info="For TheBloke, use: model",
                                                                     value=kwargs['load_gptq'],
                                                                     visible=kwargs['use_autogptq'],
                                                                     interactive=not is_public)
                                        model_gptq_dict = gr.Textbox(value=str(kwargs['gptq_dict'] or {}),
                                                                     info="E.g. {'inject_fused_attention':False, 'disable_exllama': True}",
                                                                     label="gptq_dict",
                                                                     visible=kwargs['use_autogptq'])
                                        model_load_awq = gr.Textbox(label="awq", value=kwargs['load_awq'],
                                                                    info="For TheBloke, use: model",
                                                                    interactive=not is_public)
                                        model_load_exllama_checkbox = gr.components.Checkbox(
                                            label="Load with exllama [requires support]",
                                            value=kwargs['load_exllama'], interactive=not is_public)
                                        model_exllama_dict = gr.Textbox(value=str(kwargs['exllama_dict'] or {}),
                                                                        label="exllama_dict",
                                                                        info="E.g. to split across 2 GPUs: {'set_auto_map':20,20}")
                                    hf_label = "HuggingFace" if kwargs['use_autogptq'] else "HuggingFace (inc. GPTQ)"
                                    with gr.Accordion(hf_label, open=False, visible=True):
                                        model_safetensors_checkbox = gr.components.Checkbox(
                                            label="Safetensors [required sometimes, e.g. GPTQ from TheBloke]",
                                            value=kwargs['use_safetensors'], interactive=not is_public)
                                        model_hf_model_dict = gr.Textbox(value=str(kwargs['hf_model_dict'] or {}),
                                                                         label="hf_model_dict")
                                        model_force_seq2seq_type = gr.components.Checkbox(
                                            label="Force sequence to sequence")
                                        model_force_force_t5_type = gr.components.Checkbox(
                                            label="Force T5 Conditional")
                                        model_revision = gr.Textbox(label="revision",
                                                                    value=kwargs['revision'],
                                                                    info="Hash on HF to use",
                                                                    interactive=not is_public)
                                    with gr.Accordion("Current or Custom Model Prompt", open=False, visible=True):
                                        prompt_dict = gr.Textbox(label="Current Prompt (or Custom)",
                                                                 value=pprint.pformat(kwargs['prompt_dict'] or {},
                                                                                      indent=4),
                                                                 interactive=not is_public, lines=6)
                                    with gr.Accordion("Current or Custom Context Length", open=False, visible=True):
                                        max_seq_len = gr.Number(value=kwargs['max_seq_len'] or -1,
                                                                minimum=-1,
                                                                maximum=2 ** 18,
                                                                precision=0,
                                                                info="If standard LLaMa-2, choose up to 4096 (-1 means choose max of model)",
                                                                label="max_seq_len")
                                        max_seq_len_used = gr.Number(value=kwargs['max_seq_len'] or -1,
                                                                     label="Current Max. Seq. Length",
                                                                     interactive=False)
                                        rope_scaling = gr.Textbox(value=str(kwargs['rope_scaling'] or {}),
                                                                  label="rope_scaling",
                                                                  info="Not required if in config.json.  E.g. {'type':'linear', 'factor':4} for HF and {'alpha_value':4} for exllama")
                                    acc_llama = gr.Accordion("LLaMa.cpp & GPT4All", open=False,
                                                             visible=kwargs['show_llama'])
                                    with acc_llama:
                                        # with row_llama:
                                        model_path_llama = gr.Textbox(value=kwargs['llamacpp_dict']['model_path_llama'],
                                                                      lines=4,
                                                                      label="Choose LLaMa.cpp Model Path/URL (for Base Model: llama)",
                                                                      visible=kwargs['show_llama'])
                                        n_gpu_layers = gr.Number(value=kwargs['llamacpp_dict']['n_gpu_layers'],
                                                                 minimum=0, maximum=100,
                                                                 label="LLaMa.cpp Num. GPU Layers Offloaded",
                                                                 visible=kwargs['show_llama'])
                                        n_batch = gr.Number(value=kwargs['llamacpp_dict']['n_batch'],
                                                            minimum=0, maximum=2048,
                                                            label="LLaMa.cpp Batch Size",
                                                            visible=kwargs['show_llama'])
                                        n_gqa = gr.Number(value=kwargs['llamacpp_dict']['n_gqa'],
                                                          minimum=0, maximum=32,
                                                          label="LLaMa.cpp Num. Group Query Attention (8 for 70B LLaMa2)",
                                                          visible=kwargs['show_llama'])
                                        llamacpp_dict_more = gr.Textbox(value="{}",
                                                                        lines=4,
                                                                        label="Dict for other LLaMa.cpp/GPT4All options",
                                                                        visible=kwargs['show_llama'])
                                        model_name_gptj = gr.Textbox(value=kwargs['llamacpp_dict']['model_name_gptj'],
                                                                     label="Choose GPT4All GPTJ Model Path/URL (for Base Model: gptj)",
                                                                     visible=kwargs['show_gpt4all'])
                                        model_name_gpt4all_llama = gr.Textbox(
                                            value=kwargs['llamacpp_dict']['model_name_gpt4all_llama'],
                                            label="Choose GPT4All LLaMa Model Path/URL (for Base Model: gpt4all_llama)",
                                            visible=kwargs['show_gpt4all'])
                        col_model2 = gr.Column(visible=False)
                        with col_model2:
                            with gr.Row():
                                with gr.Column(scale=10, visible=not kwargs['model_lock']):
                                    load_models_button2 = gr.Button('Load Model Names from Server2',
                                                                    variant=variant_load_msg, scale=0,
                                                                    size='sm', interactive=not is_public)
                                    load_model_button2 = gr.Button(load_msg2, variant=variant_load_msg, scale=0,
                                                                   size='sm', interactive=not is_public)
                                    unload_model_button2 = gr.Button("UnLoad Model2", variant=variant_load_msg, scale=0,
                                                                     size='sm', interactive=not is_public)
                                    with gr.Row():
                                        with gr.Column():
                                            model_choice2 = gr.Dropdown(model_options_state.value[0],
                                                                        label="Choose/Enter Model 2 (HF name, TheBloke, file, URL)",
                                                                        value=no_model_str,
                                                                        allow_custom_value=not is_public)
                                            lora_choice2 = gr.Dropdown(lora_options_state.value[0],
                                                                       label="Choose/Enter LORA 2",
                                                                       value=no_lora_str,
                                                                       visible=kwargs['show_lora'],
                                                                       allow_custom_value=not is_public)
                                            server_choice2 = gr.Dropdown(server_options_state.value[0],
                                                                         label="Choose/Enter Server 2",
                                                                         value=no_server_str,
                                                                         visible=not is_public,
                                                                         allow_custom_value=not is_public)
                                            if kwargs['visible_models_tab']:
                                                prompt_type2 = get_prompt_type2(**kwargs)
                                        with gr.Column():
                                            # no model/lora loaded ever in model2 by default
                                            model_used2 = gr.Textbox(label="Current Model 2", value=no_model_str,
                                                                     interactive=False)
                                            lora_used2 = gr.Textbox(label="Current LORA (Model 2)", value=no_lora_str,
                                                                    visible=kwargs['show_lora'], interactive=False)
                                            server_used2 = gr.Textbox(label="Current Server (Model 2)",
                                                                      value=no_server_str,
                                                                      interactive=False,
                                                                      visible=not is_public)
                                with gr.Column(scale=1, visible=not kwargs['model_lock']):
                                    with gr.Accordion("Precision", open=False, visible=True):
                                        model_load8bit_checkbox2 = gr.components.Checkbox(
                                            label="Load 8-bit (Model 2) [requires support]",
                                            value=kwargs['load_8bit'], interactive=not is_public)
                                        model_load4bit_checkbox2 = gr.components.Checkbox(
                                            label="Load 4-bit (Model 2) [requires support]",
                                            value=kwargs['load_4bit'], interactive=not is_public)
                                        model_low_bit_mode2 = gr.Slider(value=kwargs['low_bit_mode'],
                                                                        # ok that same as Model 1
                                                                        minimum=0, maximum=4, step=1,
                                                                        label="low_bit_mode (Model 2)")
                                    with gr.Accordion("GPU", open=False, visible=n_gpus != 0):
                                        model_use_cpu_checkbox2 = gr.components.Checkbox(
                                            label="Use CPU even if have GPUs (Model 2)",
                                            value=False,
                                            interactive=not is_public)
                                        model_use_gpu_id_checkbox2 = gr.components.Checkbox(
                                            label="Choose Devices (Model 2) [If not Checked, use all GPUs]",
                                            value=kwargs['use_gpu_id'],
                                            interactive=not is_public)
                                        model_gpu2 = gr.Dropdown(n_gpus_list,
                                                                 label="GPU ID (Model 2) [-1 = all GPUs, if choose is enabled]",
                                                                 info=llama_multi_gpu_info,
                                                                 value=kwargs['gpu_id'],
                                                                 interactive=not is_public)
                                    with gr.Accordion("Add-ons", open=False, visible=True):
                                        model_attention_sinks2 = gr.components.Checkbox(
                                            label="Enable Attention Sinks [requires support] (Model 2)",
                                            value=kwargs['attention_sinks'], interactive=not is_public)
                                        model_truncation_generation2 = gr.components.Checkbox(
                                            label="Truncate generation (disable for attention sinks) (Model 2)",
                                            value=kwargs['truncation_generation'], interactive=not is_public)
                                        model_sink_dict2 = gr.Textbox(value=str(kwargs['sink_dict'] or {}),
                                                                      label="sink_dict (Model 2)")
                                        model_load_gptq2 = gr.Textbox(label="gptq (Model 2)",
                                                                      info="For TheBloke models, use: model",
                                                                      value=kwargs['load_gptq'],
                                                                      visible=kwargs['use_autogptq'],
                                                                      interactive=not is_public)
                                        model_gptq_dict2 = gr.Textbox(value=str(kwargs['gptq_dict'] or {}),
                                                                      info="E.g. {'inject_fused_attention':False, 'disable_exllama': True}",
                                                                      visible=kwargs['use_autogptq'],
                                                                      label="gptq_dict (Model 2)")
                                        model_load_awq2 = gr.Textbox(label="awq (Model 2)", value='',
                                                                     interactive=not is_public)
                                        model_load_exllama_checkbox2 = gr.components.Checkbox(
                                            label="Load with exllama (Model 2) [requires support]",
                                            value=False, interactive=not is_public)
                                        model_exllama_dict2 = gr.Textbox(value=str(kwargs['exllama_dict'] or {}),
                                                                         label="exllama_dict (Model 2)")
                                    with gr.Accordion(hf_label, open=False, visible=True):
                                        model_safetensors_checkbox2 = gr.components.Checkbox(
                                            label="Safetensors (Model 2) [requires support]",
                                            value=False, interactive=not is_public)
                                        model_hf_model_dict2 = gr.Textbox(value=str(kwargs['hf_model_dict'] or {}),
                                                                          label="hf_model_dict (Model 2)")
                                        model_force_seq2seq_type2 = gr.components.Checkbox(
                                            label="Force sequence to sequence (Model 2)")
                                        model_force_force_t5_type2 = gr.components.Checkbox(
                                            label="Force T5 Conditional (Model 2)")
                                        model_revision2 = gr.Textbox(label="revision (Model 2)", value='',
                                                                     interactive=not is_public)
                                    with gr.Accordion("Current or Custom Model Prompt", open=False, visible=True):
                                        prompt_dict2 = gr.Textbox(label="Current Prompt (or Custom) (Model 2)",
                                                                  value=pprint.pformat(kwargs['prompt_dict'] or {},
                                                                                       indent=4),
                                                                  interactive=not is_public, lines=4)
                                    with gr.Accordion("Current or Custom Context Length", open=False, visible=True):
                                        max_seq_len2 = gr.Number(value=kwargs['max_seq_len'] or -1,
                                                                 minimum=-1,
                                                                 maximum=2 ** 18,
                                                                 info="If standard LLaMa-2, choose up to 4096 (-1 means choose max of model)",
                                                                 label="max_seq_len Model 2")
                                        max_seq_len_used2 = gr.Number(value=-1,
                                                                      label="mCurrent Max. Seq. Length (Model 2)",
                                                                      interactive=False)
                                        rope_scaling2 = gr.Textbox(value=str(kwargs['rope_scaling'] or {}),
                                                                   label="rope_scaling Model 2")
                                    acc_llama2 = gr.Accordion("LLaMa.cpp & GPT4All", open=False,
                                                              visible=kwargs['show_llama'])
                                    with acc_llama2:
                                        model_path_llama2 = gr.Textbox(
                                            value=kwargs['llamacpp_dict']['model_path_llama'],
                                            label="Choose LLaMa.cpp Model 2 Path/URL (for Base Model: llama)",
                                            lines=4,
                                            visible=kwargs['show_llama'])
                                        n_gpu_layers2 = gr.Number(value=kwargs['llamacpp_dict']['n_gpu_layers'],
                                                                  minimum=0, maximum=100,
                                                                  label="LLaMa.cpp Num. GPU 2 Layers Offloaded",
                                                                  visible=kwargs['show_llama'])
                                        n_batch2 = gr.Number(value=kwargs['llamacpp_dict']['n_batch'],
                                                             minimum=0, maximum=2048,
                                                             label="LLaMa.cpp Model 2 Batch Size",
                                                             visible=kwargs['show_llama'])
                                        n_gqa2 = gr.Number(value=kwargs['llamacpp_dict']['n_gqa'],
                                                           minimum=0, maximum=32,
                                                           label="LLaMa.cpp Model 2 Num. Group Query Attention (8 for 70B LLaMa2)",
                                                           visible=kwargs['show_llama'])
                                        llamacpp_dict_more2 = gr.Textbox(value="{}",
                                                                         lines=4,
                                                                         label="Model 2 Dict for other LLaMa.cpp/GPT4All options",
                                                                         visible=kwargs['show_llama'])
                                        model_name_gptj2 = gr.Textbox(value=kwargs['llamacpp_dict']['model_name_gptj'],
                                                                      label="Choose GPT4All GPTJ Model 2 Path/URL (for Base Model: gptj)",
                                                                      visible=kwargs['show_gpt4all'])
                                        model_name_gpt4all_llama2 = gr.Textbox(
                                            value=kwargs['llamacpp_dict']['model_name_gpt4all_llama'],
                                            label="Choose GPT4All LLaMa Model 2 Path/URL (for Base Model: gpt4all_llama)",
                                            visible=kwargs['show_gpt4all'])

                    compare_checkbox = gr.components.Checkbox(label="Compare Two Models",
                                                              value=kwargs['model_lock'],
                                                              visible=not is_public and not kwargs['model_lock'])
                    with gr.Row(visible=not kwargs['model_lock'] and kwargs['enable_add_models_to_list_ui']):
                        with gr.Column(scale=50):
                            new_model = gr.Textbox(label="New Model name/path/URL", interactive=not is_public)
                        with gr.Column(scale=50):
                            new_lora = gr.Textbox(label="New LORA name/path/URL", visible=kwargs['show_lora'],
                                                  interactive=not is_public)
                        with gr.Column(scale=50):
                            new_server = gr.Textbox(label="New Server url:port", interactive=not is_public)
                        with gr.Row():
                            add_model_lora_server_button = gr.Button("Add new Model, Lora, Server url:port", scale=0,
                                                                     variant=variant_load_msg,
                                                                     size='sm', interactive=not is_public)
                system_tab = gr.TabItem("System", visible=kwargs['visible_system_tab'])
                with system_tab:
                    with gr.Row():
                        with gr.Column(scale=1):
                            side_bar_text = gr.Textbox('on' if kwargs['visible_side_bar'] else 'off',
                                                       visible=False, interactive=False)
                            side_bar_btn = gr.Button("Toggle SideBar", variant="secondary", size="sm")
                            doc_count_text = gr.Textbox('on' if kwargs['visible_doc_track'] else 'off',
                                                        visible=False, interactive=False)
                            doc_count_btn = gr.Button("Toggle SideBar Document Count/Show Newest", variant="secondary",
                                                      size="sm",
                                                      visible=langchain_mode != LangChainMode.DISABLED.value)
                            submit_buttons_text = gr.Textbox('on' if kwargs['visible_submit_buttons'] else 'off',
                                                             visible=False, interactive=False)
                            submit_buttons_btn = gr.Button("Toggle Submit Buttons", variant="secondary", size="sm")
                            visible_models_text = gr.Textbox('on' if kwargs['visible_visible_models'] and \
                                                                     visible_model_choice else 'off',
                                                             visible=False, interactive=False)
                            visible_model_btn = gr.Button("Toggle Visible Models", variant="secondary", size="sm")

                            col_tabs_scale = gr.Slider(minimum=1, maximum=20, value=10, step=1, label='Window Size')
                            text_outputs_height = gr.Slider(minimum=100, maximum=2000, value=kwargs['height'] or 400,
                                                            step=50, label='Chat Height')
                            pdf_height = gr.Slider(minimum=100, maximum=3000, value=kwargs['pdf_height'] or 800,
                                                   step=50, label='PDF Viewer Height',
                                                   visible=have_gradio_pdf and langchain_mode != LangChainMode.DISABLED.value)
                            dark_mode_btn = gr.Button("Dark Mode", variant="secondary", size="sm")

                            # gr.TabItem(s):
                            with gr.Row():
                                # can make less visible but not make what was invisible into visible since button will not be visible
                                chat_tab_text = gr.Textbox('on' if kwargs['visible_chat_tab'] else 'off',
                                                           visible=False, interactive=False)
                                chat_tab_btn = gr.Button("Toggle Chat Tab", variant="secondary", size="sm",
                                                         visible=kwargs['visible_chat_tab'])
                                doc_selection_tab_text = gr.Textbox('on' if kwargs['visible_doc_view_tab'] else 'off',
                                                                    visible=False, interactive=False)
                                doc_selection_btn = gr.Button("Toggle Document Selection Tab", variant="secondary",
                                                              size="sm", visible=kwargs['visible_doc_view_tab'])
                                doc_view_tab_text = gr.Textbox('on' if kwargs['visible_doc_view_tab'] else 'off',
                                                               visible=False, interactive=False)
                                doc_view_tab_btn = gr.Button("Toggle Document View tab", variant="secondary", size="sm",
                                                             visible=kwargs['visible_doc_view_tab'])
                                chat_history_tab_text = gr.Textbox(
                                    'on' if kwargs['visible_chat_history_tab'] else 'off',
                                    visible=False, interactive=False)
                                chat_history_btn = gr.Button("Toggle Chat History Tab", variant="secondary", size="sm",
                                                             visible=kwargs['visible_chat_history_tab'])
                                expert_tab_text = gr.Textbox('on' if kwargs['visible_expert_tab'] else 'off',
                                                             visible=False, interactive=False)
                                expert_tab_btn = gr.Button("Toggle Expert Tab", variant="secondary", size="sm",
                                                           visible=kwargs['visible_expert_tab'])
                                models_tab_text = gr.Textbox('on' if kwargs['visible_models_tab'] else 'off',
                                                             visible=False, interactive=False)
                                models_tab_btn = gr.Button("Toggle Models Tab", variant="secondary", size="sm",
                                                           visible=kwargs['visible_models_tab'])
                                system_tab_text = gr.Textbox('on' if kwargs['visible_system_tab'] else 'off',
                                                             visible=False, interactive=False)
                                # too confusing to allow system to turn itself off, can't recover, so only allow CLI to control if visible, not in UI
                                system_tab_btn = gr.Button("Toggle Systems Tab", variant="secondary", size="sm",
                                                           visible=False and kwargs['visible_system_tab'])
                                tos_tab_text = gr.Textbox('on' if kwargs['visible_tos_tab'] else 'off',
                                                          visible=False, interactive=False)
                                tos_tab_btn = gr.Button("Toggle ToS Tab", variant="secondary", size="sm",
                                                        visible=kwargs['visible_tos_tab'])
                                login_tab_text = gr.Textbox('on' if kwargs['visible_login_tab'] else 'off',
                                                            visible=False, interactive=False)
                                login_tab_btn = gr.Button("Toggle Login Tab", variant="secondary", size="sm",
                                                          visible=kwargs['visible_login_tab'])
                                hosts_tab_text = gr.Textbox('on' if kwargs['visible_hosts_tab'] else 'off',
                                                            visible=False, interactive=False)
                                hosts_tab_btn = gr.Button("Toggle Hosts Tab", variant="secondary", size="sm",
                                                          visible=kwargs['visible_hosts_tab'])

                        with gr.Column(scale=4):
                            pass
                    system_visible0 = not is_public and not admin_pass
                    admin_row = gr.Row()
                    with admin_row:
                        with gr.Column(scale=1):
                            admin_pass_textbox = gr.Textbox(label="Admin Password",
                                                            type='password',
                                                            visible=not system_visible0)
                        guest_name = gr.Textbox(value=kwargs['guest_name'], visible=False)
                        with gr.Column(scale=4):
                            pass
                    system_row = gr.Row(visible=system_visible0)
                    with system_row:
                        with gr.Accordion("Admin", open=False, visible=True):
                            with gr.Column():
                                close_btn = gr.Button(value="Shutdown h2oGPT", size='sm',
                                                      visible=kwargs['close_button'] and kwargs[
                                                          'h2ogpt_pid'] is not None)
                                with gr.Row():
                                    system_btn = gr.Button(value='Get System Info', size='sm')
                                    system_text = gr.Textbox(label='System Info', interactive=False,
                                                             show_copy_button=True)
                                with gr.Row():
                                    system_input = gr.Textbox(label='System Info Dict Password', interactive=True,
                                                              visible=not is_public)
                                    system_btn2 = gr.Button(value='Get System Info Dict', visible=not is_public,
                                                            size='sm')
                                    system_text2 = gr.Textbox(label='System Info Dict', interactive=False,
                                                              visible=not is_public, show_copy_button=True)
                                with gr.Row():
                                    system_btn3 = gr.Button(value='Get Hash', visible=not is_public, size='sm')
                                    system_text3 = gr.Textbox(label='Hash', interactive=False,
                                                              visible=not is_public, show_copy_button=True)

                                    def get_hash():
                                        return kwargs['git_hash']

                                    system_event = system_btn3.click(get_hash,
                                                                     outputs=system_text3,
                                                                     api_name='system_hash' if allow_api else False,
                                                                     **noqueue_kwargs_curl,
                                                                     )

                                    system_btn4 = gr.Button(value='Get Model Names', visible=not is_public, size='sm')
                                    system_text4 = gr.Textbox(label='Model Names', interactive=False,
                                                              visible=not is_public, show_copy_button=True)

                                with gr.Row():
                                    zip_btn = gr.Button("Zip", size='sm')
                                    zip_text = gr.Textbox(label="Zip file name", interactive=False)
                                    file_output = gr.File(interactive=False, label="Zip file to Download")
                                with gr.Row():
                                    s3up_btn = gr.Button("S3UP", size='sm')
                                    s3up_text = gr.Textbox(label='S3UP result', interactive=False)

                tos_tab = gr.TabItem("Terms of Service", visible=kwargs['visible_tos_tab'])
                with tos_tab:
                    description = ""
                    description += """<p><b> DISCLAIMERS: </b><ul><i><li>The model was trained on The Pile and other data, which may contain objectionable content.  Use at own risk.</i></li>"""
                    if kwargs['load_8bit']:
                        description += """<i><li> Model is loaded in 8-bit and has other restrictions on this host. UX can be worse than non-hosted version.</i></li>"""
                    description += """<i><li>Conversations may be used to improve h2oGPT.  Do not share sensitive information.</i></li>"""
                    if 'h2ogpt-research' in kwargs['base_model']:
                        description += """<i><li>Research demonstration only, not used for commercial purposes.</i></li>"""
                    description += """<i><li>By using h2oGPT, you accept our <a href="https://github.com/h2oai/h2ogpt/blob/main/docs/tos.md">Terms of Service</a></i></li></ul></p>"""
                    gr.Markdown(value=description, show_label=False)

                login_tab = gr.TabItem("Log-in/out" if kwargs['auth'] else "Login", visible=kwargs['visible_login_tab'])
                with login_tab:
                    extra_login = "\nDaily maintenance at midnight PST will not allow reconnection to state otherwise." if is_public else ""
                    gr.Markdown(
                        value="#### Login page to persist your state (database, documents, chat, chat history, model list)%s" % extra_login)
                    username_text = gr.Textbox(label="Username")
                    password_text = gr.Textbox(label="Password", type='password', visible=True)
                    login_msg = "Login (pick unique user/pass to persist your state)" if kwargs[
                                                                                             'auth_access'] == 'open' else "Login (closed access)"
                    login_btn = gr.Button(value=login_msg)
                    num_lock_button = gr.Button(visible=False)
                    num_model_lock_value_output = gr.Number(value=len(text_outputs), visible=False, precision=0)
                    login_result_text = gr.Text(label="Login Result", interactive=False)
                    # WIP
                    if (kwargs['auth'] or kwargs['google_auth']) and is_gradio_h2oai:
                        gr.Button("Logout", link="/logout")
                    if kwargs['enforce_h2ogpt_api_key'] and kwargs['enforce_h2ogpt_ui_key']:
                        label_h2ogpt_key = "h2oGPT Token for API and UI access"
                    elif kwargs['enforce_h2ogpt_api_key']:
                        label_h2ogpt_key = "h2oGPT Token for API access"
                    elif kwargs['enforce_h2ogpt_ui_key']:
                        label_h2ogpt_key = "h2oGPT Token for UI access"
                    else:
                        label_h2ogpt_key = 'Unused'
                    h2ogpt_key = gr.Text(value='',
                                         # do not use kwargs['h2ogpt_key'] here, that's only for gradio inference server
                                         label=label_h2ogpt_key,
                                         type='password',
                                         visible=kwargs['enforce_h2ogpt_ui_key'],  # only show if need for UI
                                         )

                hosts_tab = gr.TabItem("Hosts", visible=kwargs['visible_hosts_tab'])
                with hosts_tab:
                    gr.Markdown(f"""
                        {description_bottom}
                        {task_info_md}
                        """)

        def zip_data_check_key(admin_pass_textbox1,
                               h2ogpt_key2,
                               root_dirs=None,
                               enforce_h2ogpt_api_key=None,
                               enforce_h2ogpt_ui_key=None,
                               h2ogpt_api_keys=None, requests_state1=None):
            valid_key = is_valid_key(enforce_h2ogpt_api_key,
                                     enforce_h2ogpt_ui_key,
                                     h2ogpt_api_keys,
                                     h2ogpt_key2,
                                     requests_state1=requests_state1,
                                     )
            from_ui = is_from_ui(requests_state1)
            if not valid_key:
                raise ValueError(invalid_key_msg)
            assert admin_pass_textbox1 == admin_pass or not admin_pass
            return zip_data(root_dirs=root_dirs)

        zip_data_func = functools.partial(zip_data_check_key,
                                          root_dirs=['flagged_data_points', kwargs['save_dir']],
                                          enforce_h2ogpt_api_key=kwargs['enforce_h2ogpt_api_key'],
                                          enforce_h2ogpt_ui_key=kwargs['enforce_h2ogpt_ui_key'],
                                          h2ogpt_api_keys=kwargs['h2ogpt_api_keys'],
                                          )
        # Get flagged data
        zip_data1 = functools.partial(zip_data_func)
        zip_event = zip_btn.click(zip_data1, inputs=[admin_pass_textbox, h2ogpt_key],
                                  outputs=[file_output, zip_text],
                                  **noqueue_kwargs,
                                  api_name=False,
                                  )

        def s3up_check_key(zip_text, admin_pass_textbox1, h2ogpt_key1,
                           enforce_h2ogpt_api_key=None,
                           enforce_h2ogpt_ui_key=None,
                           h2ogpt_api_keys=None, requests_state1=None):
            valid_key = is_valid_key(enforce_h2ogpt_api_key,
                                     enforce_h2ogpt_ui_key,
                                     h2ogpt_api_keys,
                                     h2ogpt_key1,
                                     requests_state1=requests_state1,
                                     )
            from_ui = is_from_ui(requests_state1)
            if not valid_key:
                raise ValueError(invalid_key_msg)
            assert admin_pass_textbox1 == admin_pass or not admin_pass
            return s3up(zip_text)

        s3up_check_key_func = functools.partial(s3up_check_key, enforce_h2ogpt_api_key=kwargs['enforce_h2ogpt_api_key'],
                                                enforce_h2ogpt_ui_key=kwargs['enforce_h2ogpt_ui_key'],
                                                h2ogpt_api_keys=kwargs['h2ogpt_api_keys'],
                                                )

        s3up_event = s3up_btn.click(s3up_check_key_func, inputs=[zip_text, admin_pass_textbox, h2ogpt_key],
                                    outputs=s3up_text,
                                    **noqueue_kwargs,
                                    api_name=False,
                                    )

        def clear_file_list():
            return None

        def set_loaders(max_quality1,
                        image_audio_loaders_options1=None,
                        pdf_loaders_options1=None,
                        url_loaders_options1=None,
                        image_audio_loaders_options01=None,
                        pdf_loaders_options01=None,
                        url_loaders_options01=None,
                        ):
            if not max_quality1:
                return image_audio_loaders_options01, pdf_loaders_options01, url_loaders_options01
            else:
                return image_audio_loaders_options1, pdf_loaders_options1, url_loaders_options1

        set_loaders_func = functools.partial(set_loaders,
                                             image_audio_loaders_options1=image_audio_loaders_options,
                                             pdf_loaders_options1=pdf_loaders_options,
                                             url_loaders_options1=url_loaders_options,
                                             image_audio_loaders_options01=image_audio_loaders_options0,
                                             pdf_loaders_options01=pdf_loaders_options0,
                                             url_loaders_options01=url_loaders_options0,
                                             )

        max_quality.change(fn=set_loaders_func,
                           inputs=max_quality,
                           outputs=[image_audio_loaders, pdf_loaders, url_loaders])

        def fun_mm(instruction_mm1, instruction1, image_file1, video_file1, audio_file1):
            if 'text' in instruction_mm1:
                instruction1 = instruction_mm1['text']
            if 'url' in instruction_mm1:
                instruction1 = instruction_mm1['url']
            if 'files' in instruction_mm1:
                image_file2 = [x for x in instruction_mm1['files'] if any(x.endswith(y) for y in IMAGE_EXTENSIONS)]
                if image_file2:
                    image_file1 = image_file2
                video_file2 = [x for x in instruction_mm1['files'] if any(x.endswith(y) for y in VIDEO_EXTENSIONS)]
                if video_file2:
                    video_file1 = video_file2
                if isinstance(video_file1, list):
                    if video_file1:
                        video_file1 = video_file1[0]
                    else:
                        video_file1 = None
            return instruction1, image_file1, video_file1, audio_file1

        instruction_mm.submit(fn=fun_mm,
                              inputs=[instruction_mm, instruction, image_file, video_file, audio_file],
                              outputs=[instruction, image_file, video_file, audio_file]). \
                              then(lambda: gr.MultimodalTextbox(interactive=True), None, [instruction_mm])

        # Add to UserData or custom user db
        update_db_func = functools.partial(update_user_db_gr,
                                           dbs=dbs,
                                           db_type=db_type,
                                           use_openai_embedding=use_openai_embedding,
                                           hf_embedding_model=hf_embedding_model,
                                           migrate_embedding_model=migrate_embedding_model,
                                           captions_model=captions_model,
                                           caption_loader=caption_loader,
                                           doctr_loader=doctr_loader,
                                           llava_model=llava_model,
                                           asr_model=asr_model,
                                           asr_loader=asr_loader,
                                           verbose=kwargs['verbose'],
                                           n_jobs=kwargs['n_jobs'],
                                           get_userid_auth=get_userid_auth,
                                           image_audio_loaders_options0=image_audio_loaders_options0,
                                           pdf_loaders_options0=pdf_loaders_options0,
                                           url_loaders_options0=url_loaders_options0,
                                           jq_schema0=jq_schema0,
                                           enforce_h2ogpt_api_key=kwargs['enforce_h2ogpt_api_key'],
                                           enforce_h2ogpt_ui_key=kwargs['enforce_h2ogpt_ui_key'],
                                           h2ogpt_api_keys=kwargs['h2ogpt_api_keys'],
                                           is_public=is_public,
                                           use_pymupdf=kwargs['use_pymupdf'],
                                           use_unstructured_pdf=kwargs['use_unstructured_pdf'],
                                           use_pypdf=kwargs['use_pypdf'],
                                           enable_pdf_ocr=kwargs['enable_pdf_ocr'],
                                           enable_pdf_doctr=kwargs['enable_pdf_doctr'],
                                           try_pdf_as_html=kwargs['try_pdf_as_html'],
                                           gradio_upload_to_chatbot_num_max=kwargs['gradio_upload_to_chatbot_num_max'],
                                           allow_upload_to_my_data=kwargs['allow_upload_to_my_data'],
                                           allow_upload_to_user_data=kwargs['allow_upload_to_user_data'],
                                           function_server=kwargs['function_server'],
                                           function_server_port=kwargs['function_server_port'],
                                           function_api_key=h2ogpt_key1 if not kwargs['function_api_key'] else kwargs[
                                               'function_api_key'],
                                           )
        add_file_outputs = [fileup_output, langchain_mode]
        add_file_kwargs = dict(fn=update_db_func,
                               inputs=[fileup_output, my_db_state, selection_docs_state, requests_state,
                                       langchain_mode, chunk, chunk_size, embed,
                                       image_audio_loaders,
                                       pdf_loaders,
                                       url_loaders,
                                       jq_schema,
                                       extract_frames,
                                       llava_prompt,
                                       h2ogpt_key,
                                       ],
                               outputs=add_file_outputs + [sources_text, doc_exception_text, text_file_last,
                                                           new_files_last],
                               queue=queue,
                               api_name='add_file' if allow_upload_api else False)

        # then no need for add buttons, only single changeable db
        user_state_kwargs = dict(fn=user_state_setup,
                                 inputs=[my_db_state, requests_state, guest_name, langchain_mode],
                                 outputs=[my_db_state, requests_state, langchain_mode],
                                 show_progress='minimal')
        eventdb1a = fileup_output.upload(**user_state_kwargs)
        eventdb1 = eventdb1a.then(**add_file_kwargs, show_progress='full')

        event_attach1 = attach_button.upload(**user_state_kwargs)
        attach_file_kwargs = add_file_kwargs.copy()
        attach_file_kwargs['inputs'][0] = attach_button
        attach_file_kwargs['outputs'][0] = attach_button
        attach_file_kwargs['api_name'] = 'attach_file'
        event_attach2 = event_attach1.then(**attach_file_kwargs, show_progress='full')

        sync1 = sync_sources_btn.click(**user_state_kwargs)

        # deal with challenge to have fileup_output itself as input
        add_file_kwargs2 = dict(fn=update_db_func,
                                inputs=[fileup_output_text, my_db_state, selection_docs_state, requests_state,
                                        langchain_mode, chunk, chunk_size, embed,
                                        image_audio_loaders,
                                        pdf_loaders,
                                        url_loaders,
                                        jq_schema,
                                        extract_frames,
                                        llava_prompt,
                                        h2ogpt_key,
                                        ],
                                outputs=add_file_outputs + [sources_text, doc_exception_text, text_file_last,
                                                            new_files_last],
                                queue=queue,
                                api_name='add_file_api' if allow_upload_api else None)
        eventdb1_api = fileup_output_text.submit(**add_file_kwargs2, show_progress='full')

        # note for update_user_db_func output is ignored for db

        def clear_textbox():
            return gr.Textbox(value='')

        update_user_db_url_func = functools.partial(update_db_func, is_url=True,
                                                    is_txt=not kwargs['actions_in_sidebar'])

        add_url_outputs = [url_text, langchain_mode]
        add_url_kwargs = dict(fn=update_user_db_url_func,
                              inputs=[url_text, my_db_state, selection_docs_state, requests_state,
                                      langchain_mode, chunk, chunk_size, embed,
                                      image_audio_loaders,
                                      pdf_loaders,
                                      url_loaders,
                                      jq_schema,
                                      extract_frames,
                                      llava_prompt,
                                      h2ogpt_key,
                                      ],
                              outputs=add_url_outputs + [sources_text, doc_exception_text, text_file_last,
                                                         new_files_last],
                              queue=queue,
                              api_name='add_url' if allow_upload_api else False)

        user_text_submit_kwargs = dict(fn=user_state_setup,
                                       inputs=[my_db_state, requests_state, guest_name, url_text, url_text],
                                       outputs=[my_db_state, requests_state, url_text],
                                       queue=queue,
                                       show_progress='minimal')
        eventdb2a = url_text.submit(**user_text_submit_kwargs)
        # work around https://github.com/gradio-app/gradio/issues/4733
        eventdb2 = eventdb2a.then(**add_url_kwargs, show_progress='full')

        # small button version
        add_url_kwargs_btn = add_url_kwargs.copy()
        add_url_kwargs_btn.update(api_name='add_url_btn' if allow_upload_api else False)

        def copy_text(instruction1):
            return gr.Textbox(value=''), instruction1

        eventdb2a_btn = add_button.click(copy_text, inputs=instruction, outputs=[instruction, url_text],
                                         **noqueue_kwargs2)
        eventdb2a_btn2 = eventdb2a_btn.then(**user_text_submit_kwargs)
        eventdb2_btn = eventdb2a_btn2.then(**add_url_kwargs_btn, show_progress='full')

        update_user_db_txt_func = functools.partial(update_db_func, is_txt=True, is_url=False)
        add_text_outputs = [user_text_text, langchain_mode]
        add_text_kwargs = dict(fn=update_user_db_txt_func,
                               inputs=[user_text_text, my_db_state, selection_docs_state, requests_state,
                                       langchain_mode, chunk, chunk_size, embed,
                                       image_audio_loaders,
                                       pdf_loaders,
                                       url_loaders,
                                       jq_schema,
                                       extract_frames,
                                       llava_prompt,
                                       h2ogpt_key,
                                       ],
                               outputs=add_text_outputs + [sources_text, doc_exception_text, text_file_last,
                                                           new_files_last],
                               queue=queue,
                               api_name='add_text' if allow_upload_api else False
                               )
        eventdb3a = user_text_text.submit(fn=user_state_setup,
                                          inputs=[my_db_state, requests_state, guest_name, user_text_text,
                                                  user_text_text],
                                          outputs=[my_db_state, requests_state, user_text_text],
                                          queue=queue,
                                          show_progress='minimal')
        eventdb3 = eventdb3a.then(**add_text_kwargs, show_progress='full')

        db_events = [eventdb1a, eventdb1, eventdb1_api,
                     eventdb2a, eventdb2,
                     eventdb2a_btn, eventdb2_btn,
                     eventdb3a, eventdb3]
        db_events.extend([event_attach1, event_attach2])

        get_sources_fun_kwargs = dict(dbs=dbs, docs_state0=docs_state0,
                                      load_db_if_exists=load_db_if_exists,
                                      db_type=db_type,
                                      use_openai_embedding=use_openai_embedding,
                                      hf_embedding_model=hf_embedding_model,
                                      migrate_embedding_model=migrate_embedding_model,
                                      verbose=verbose,
                                      get_userid_auth=get_userid_auth,
                                      n_jobs=n_jobs,
                                      enforce_h2ogpt_api_key=kwargs['enforce_h2ogpt_api_key'],
                                      enforce_h2ogpt_ui_key=kwargs['enforce_h2ogpt_ui_key'],
                                      h2ogpt_api_keys=kwargs['h2ogpt_api_keys'],
                                      )

        get_sources1 = functools.partial(get_sources_gr, **get_sources_fun_kwargs)

        # if change collection source, must clear doc selections from it to avoid inconsistency
        def clear_doc_choice(langchain_mode1):
            if langchain_mode1 in langchain_modes_non_db:
                label1 = 'Choose Resources->Collections and Pick Collection' if not kwargs[
                    'document_choice_in_sidebar'] else "Document"
                active_collection1 = "#### Not Chatting with Any Collection\n%s" % label1
            else:
                label1 = 'Select Subset of Document(s) for Chat with Collection: %s' % langchain_mode1 if not kwargs[
                    'document_choice_in_sidebar'] else "Document"
                active_collection1 = "#### Chatting with Collection: %s" % langchain_mode1
            return gr.Dropdown(choices=docs_state0, value=[DocumentChoice.ALL.value],
                               label=label1), gr.Markdown(value=active_collection1)

        lg_change_event = langchain_mode.change(clear_doc_choice, inputs=langchain_mode,
                                                outputs=[document_choice, active_collection],
                                                queue=not kwargs['large_file_count_mode'])

        def resize_col_tabs(x):
            return gr.Dropdown(scale=x)

        col_tabs_scale.change(fn=resize_col_tabs, inputs=col_tabs_scale, outputs=col_tabs, **noqueue_kwargs)

        def resize_chatbots(x, num_model_lock=0):
            if num_model_lock == 0:
                num_model_lock = 3  # 2 + 1 (which is dup of first)
            else:
                num_model_lock = 2 + num_model_lock
            return tuple([gr.update(height=x)] * num_model_lock)

        resize_chatbots_func = functools.partial(resize_chatbots, num_model_lock=len(text_outputs))
        text_outputs_height.change(fn=resize_chatbots_func, inputs=text_outputs_height,
                                   outputs=[text_output, text_output2] + text_outputs, **noqueue_kwargs)

        def resize_pdf_viewer_func(x):
            return gr.update(height=x)

        pdf_height.change(fn=resize_pdf_viewer_func, inputs=pdf_height, outputs=doc_view6, **noqueue_kwargs2)

        def update_dropdown(x):
            if DocumentChoice.ALL.value in x:
                x.remove(DocumentChoice.ALL.value)
            source_list = [DocumentChoice.ALL.value] + x
            return gr.Dropdown(choices=source_list, value=[DocumentChoice.ALL.value])

        get_sources_kwargs = dict(fn=get_sources1,
                                  inputs=[my_db_state, selection_docs_state, requests_state, langchain_mode,
                                          h2ogpt_key],
                                  outputs=[file_source, docs_state, text_doc_count],
                                  queue=queue)

        eventdb7a = get_sources_btn.click(user_state_setup,
                                          inputs=[my_db_state, requests_state, guest_name, get_sources_btn,
                                                  get_sources_btn],
                                          outputs=[my_db_state, requests_state, get_sources_btn],
                                          show_progress='minimal')
        eventdb7 = eventdb7a.then(**get_sources_kwargs,
                                  api_name='get_sources' if allow_api else False) \
            .then(fn=update_dropdown, inputs=docs_state, outputs=document_choice)

        get_sources_api_args = dict(fn=functools.partial(get_sources1, api=True),
                                    inputs=[my_db_state, selection_docs_state, requests_state, langchain_mode,
                                            h2ogpt_key],
                                    outputs=get_sources_api_text,
                                    queue=queue)
        get_sources_api_btn.click(**get_sources_api_args,
                                  api_name='get_sources_api' if allow_api else False)

        # show button, else only show when add.
        # Could add to above get_sources for download/dropdown, but bit much maybe
        show_sources1_fun_kwargs = dict(dbs=dbs,
                                        load_db_if_exists=load_db_if_exists,
                                        db_type=db_type,
                                        use_openai_embedding=use_openai_embedding,
                                        hf_embedding_model=hf_embedding_model,
                                        migrate_embedding_model=migrate_embedding_model,
                                        verbose=verbose,
                                        get_userid_auth=get_userid_auth,
                                        n_jobs=n_jobs,
                                        enforce_h2ogpt_api_key=kwargs['enforce_h2ogpt_api_key'],
                                        enforce_h2ogpt_ui_key=kwargs['enforce_h2ogpt_ui_key'],
                                        h2ogpt_api_keys=kwargs['h2ogpt_api_keys'],
                                        )
        show_sources1 = functools.partial(get_source_files_given_langchain_mode_gr,
                                          **show_sources1_fun_kwargs,
                                          )
        eventdb8a = show_sources_btn.click(user_state_setup,
                                           inputs=[my_db_state, requests_state, guest_name, show_sources_btn,
                                                   show_sources_btn],
                                           outputs=[my_db_state, requests_state, show_sources_btn],
                                           show_progress='minimal')
        show_sources_kwargs = dict(fn=show_sources1,
                                   inputs=[my_db_state, selection_docs_state, requests_state, langchain_mode,
                                           h2ogpt_key],
                                   outputs=sources_text)
        eventdb8 = eventdb8a.then(**show_sources_kwargs,
                                  api_name='show_sources' if allow_api else False)

        def update_viewable_dropdown(x):
            return gr.Dropdown(choices=x,
                               value=viewable_docs_state0[0] if len(viewable_docs_state0) > 0 else None)

        get_viewable_sources1_fun_kwargs = dict(dbs=dbs, docs_state0=viewable_docs_state0,
                                                load_db_if_exists=load_db_if_exists,
                                                db_type=db_type,
                                                use_openai_embedding=use_openai_embedding,
                                                hf_embedding_model=hf_embedding_model,
                                                migrate_embedding_model=migrate_embedding_model,
                                                verbose=kwargs['verbose'],
                                                get_userid_auth=get_userid_auth,
                                                n_jobs=n_jobs,
                                                enforce_h2ogpt_api_key=kwargs['enforce_h2ogpt_api_key'],
                                                enforce_h2ogpt_ui_key=kwargs['enforce_h2ogpt_ui_key'],
                                                h2ogpt_api_keys=kwargs['h2ogpt_api_keys'],
                                                )

        get_viewable_sources1 = functools.partial(get_sources_gr, **get_viewable_sources1_fun_kwargs)
        get_viewable_sources_args = dict(fn=get_viewable_sources1,
                                         inputs=[my_db_state, selection_docs_state, requests_state, langchain_mode,
                                                 h2ogpt_key],
                                         outputs=[file_source, viewable_docs_state, text_viewable_doc_count],
                                         queue=queue)
        eventdb12a = get_viewable_sources_btn.click(user_state_setup,
                                                    inputs=[my_db_state, requests_state, guest_name,
                                                            get_viewable_sources_btn, get_viewable_sources_btn],
                                                    outputs=[my_db_state, requests_state, get_viewable_sources_btn],
                                                    show_progress='minimal')
        viewable_kwargs = dict(fn=update_viewable_dropdown, inputs=viewable_docs_state, outputs=view_document_choice)
        eventdb12 = eventdb12a.then(**get_viewable_sources_args,
                                    api_name='get_viewable_sources' if allow_api else False) \
            .then(**viewable_kwargs)

        view_doc_select_kwargs = dict(fn=user_state_setup,
                                      inputs=[my_db_state, requests_state, guest_name,
                                              view_document_choice],
                                      outputs=[my_db_state, requests_state],
                                      show_progress='minimal')
        eventdb_viewa = view_document_choice.select(**view_doc_select_kwargs)
        show_doc_func = functools.partial(show_doc,
                                          dbs1=dbs,
                                          load_db_if_exists1=load_db_if_exists,
                                          db_type1=db_type,
                                          use_openai_embedding1=use_openai_embedding,
                                          hf_embedding_model1=hf_embedding_model,
                                          migrate_embedding_model_or_db1=migrate_embedding_model,
                                          verbose1=verbose,
                                          get_userid_auth1=get_userid_auth,
                                          max_raw_chunks=kwargs['max_raw_chunks'],
                                          api=False,
                                          n_jobs=n_jobs,
                                          enforce_h2ogpt_api_key=kwargs['enforce_h2ogpt_api_key'],
                                          enforce_h2ogpt_ui_key=kwargs['enforce_h2ogpt_ui_key'],
                                          h2ogpt_api_keys=kwargs['h2ogpt_api_keys'],
                                          )
        # Note: Not really useful for API, so no api_name
        show_doc_kwargs = dict(fn=show_doc_func,
                               inputs=[my_db_state, selection_docs_state, requests_state, langchain_mode,
                                       view_document_choice, view_raw_text_checkbox,
                                       text_context_list, pdf_height,
                                       h2ogpt_key],
                               outputs=[doc_view, doc_view2, doc_view3, doc_view4,
                                        doc_view5, doc_view6, doc_view7, doc_view8])
        eventdb_viewa.then(**show_doc_kwargs)

        view_raw_text_checkbox.change(**view_doc_select_kwargs) \
            .then(**show_doc_kwargs)

        show_doc_func_api = functools.partial(show_doc_func, api=True)
        get_document_api_btn.click(fn=show_doc_func_api,
                                   inputs=[my_db_state, selection_docs_state, requests_state, langchain_mode,
                                           view_document_choice, view_raw_text_checkbox,
                                           text_context_list, pdf_height,
                                           h2ogpt_key],
                                   outputs=get_document_api_text, api_name='get_document_api')

        # Get inputs to evaluate() and make_db()
        # don't deepcopy, can contain model itself
        all_kwargs = kwargs.copy()
        all_kwargs.update(locals().copy())

        refresh_sources1 = functools.partial(update_and_get_source_files_given_langchain_mode_gr,
                                             captions_model=captions_model,
                                             caption_loader=caption_loader,
                                             doctr_loader=doctr_loader,
                                             llava_model=llava_model,
                                             asr_model=asr_model,
                                             asr_loader=asr_loader,
                                             dbs=dbs,
                                             first_para=kwargs['first_para'],
                                             hf_embedding_model=hf_embedding_model,
                                             use_openai_embedding=use_openai_embedding,
                                             migrate_embedding_model=migrate_embedding_model,
                                             text_limit=kwargs['text_limit'],
                                             db_type=db_type,
                                             load_db_if_exists=load_db_if_exists,
                                             n_jobs=n_jobs, verbose=verbose,
                                             get_userid_auth=get_userid_auth,
                                             image_audio_loaders_options0=image_audio_loaders_options0,
                                             pdf_loaders_options0=pdf_loaders_options0,
                                             url_loaders_options0=url_loaders_options0,
                                             jq_schema0=jq_schema0,
                                             use_pymupdf=kwargs['use_pymupdf'],
                                             use_unstructured_pdf=kwargs['use_unstructured_pdf'],
                                             use_pypdf=kwargs['use_pypdf'],
                                             enable_pdf_ocr=kwargs['enable_pdf_ocr'],
                                             enable_pdf_doctr=kwargs['enable_pdf_doctr'],
                                             try_pdf_as_html=kwargs['try_pdf_as_html'],
                                             enforce_h2ogpt_api_key=kwargs['enforce_h2ogpt_api_key'],
                                             enforce_h2ogpt_ui_key=kwargs['enforce_h2ogpt_ui_key'],
                                             h2ogpt_api_keys=kwargs['h2ogpt_api_keys'],
                                             )
        eventdb9a = refresh_sources_btn.click(user_state_setup,
                                              inputs=[my_db_state, requests_state, guest_name,
                                                      refresh_sources_btn, refresh_sources_btn],
                                              outputs=[my_db_state, requests_state, refresh_sources_btn],
                                              show_progress='minimal')
        eventdb9 = eventdb9a.then(fn=refresh_sources1,
                                  inputs=[my_db_state, selection_docs_state, requests_state,
                                          langchain_mode, chunk, chunk_size,
                                          image_audio_loaders,
                                          pdf_loaders,
                                          url_loaders,
                                          jq_schema,
                                          extract_frames,
                                          llava_prompt,
                                          h2ogpt_key,
                                          ],
                                  outputs=sources_text,
                                  api_name='refresh_sources' if allow_api else False)

        delete_sources1 = functools.partial(del_source_files_given_langchain_mode_gr,
                                            dbs=dbs,
                                            load_db_if_exists=load_db_if_exists,
                                            db_type=db_type,
                                            use_openai_embedding=use_openai_embedding,
                                            hf_embedding_model=hf_embedding_model,
                                            migrate_embedding_model=migrate_embedding_model,
                                            verbose=verbose,
                                            get_userid_auth=get_userid_auth,
                                            n_jobs=n_jobs,
                                            enforce_h2ogpt_api_key=kwargs['enforce_h2ogpt_api_key'],
                                            enforce_h2ogpt_ui_key=kwargs['enforce_h2ogpt_ui_key'],
                                            h2ogpt_api_keys=kwargs['h2ogpt_api_keys'],
                                            )
        eventdb90a = delete_sources_btn.click(user_state_setup,
                                              inputs=[my_db_state, requests_state, guest_name,
                                                      delete_sources_btn, delete_sources_btn],
                                              outputs=[my_db_state, requests_state, delete_sources_btn],
                                              show_progress='minimal', **noqueue_kwargs2)
        eventdb90 = eventdb90a.then(fn=delete_sources1,
                                    inputs=[my_db_state, selection_docs_state, requests_state, document_choice,
                                            langchain_mode,
                                            h2ogpt_key],
                                    outputs=sources_text,
                                    api_name='delete_sources' if allow_api else False)
        db_events.extend([eventdb90a, eventdb90])

        def check_admin_pass(x):
            return gr.update(visible=x == admin_pass)

        def close_admin(x):
            return gr.update(visible=not (x == admin_pass))

        def get_num_model_lock_value():
            return len(text_outputs)

        num_lock_button.click(get_num_model_lock_value, inputs=None, outputs=num_model_lock_value_output,
                              api_name='num_model_lock', **noqueue_kwargs2)

        eventdb_logina = login_btn.click(user_state_setup,
                                         inputs=[my_db_state, requests_state, guest_name, login_btn, login_btn],
                                         outputs=[my_db_state, requests_state, login_btn],
                                         show_progress='minimal', **noqueue_kwargs2)

        def login(db1s, selection_docs_state1, requests_state1, roles_state1,
                  model_options_state1, lora_options_state1, server_options_state1,
                  chat_state1, langchain_mode1,
                  h2ogpt_key2, visible_models1,

                  side_bar_text1, doc_count_text1, submit_buttons_text1, visible_models_text1,
                  chat_tab_text1, doc_selection_tab_text1, doc_view_tab_text1, chat_history_tab_text1,
                  expert_tab_text1, models_tab_text1, system_tab_text1, tos_tab_text1,
                  login_tab_text1, hosts_tab_text1,

                  username1, password1,
                  text_output1, text_output21, *text_outputs1,
                  auth_filename=None, num_model_lock=0, pre_authorized=False):
            # use full auth login to allow new users if open access etc.
            if pre_authorized:
                username1 = requests_state1.get('username')
                password1 = get_auth_password(username1, auth_filename)
                if password1 in [None, '']:
                    password1 = username1
                authorized1 = True
            else:
                authorized1 = False

            # need to store even if pre authorized, so can keep track of state
            authorized2 = authf(username1, password1, selection_docs_state1=selection_docs_state1,
                                id0=get_userid_direct(db1s))
            authorized1 += authorized2

            if authorized1:
                if not isinstance(requests_state1, dict):
                    requests_state1 = {}
                requests_state1['username'] = username1
                set_userid_gr(db1s, requests_state1, get_userid_auth)
                username2 = get_username(requests_state1)
                text_outputs1 = list(text_outputs1)

                success1, text_result, text_output1, text_output21, text_outputs1, \
                    langchain_mode1, \
                    h2ogpt_key2, visible_models1, \
                    side_bar_text1, doc_count_text1, submit_buttons_text1, visible_models_text1, \
                    chat_tab_text1, doc_selection_tab_text1, doc_view_tab_text1, chat_history_tab_text1, \
                    expert_tab_text1, models_tab_text1, system_tab_text1, tos_tab_text1, \
                    login_tab_text1, hosts_tab_text1 = \
                    load_auth(db1s, requests_state1, auth_filename, selection_docs_state1=selection_docs_state1,
                              roles_state1=roles_state1,
                              model_options_state1=model_options_state1,
                              lora_options_state1=lora_options_state1,
                              server_options_state1=server_options_state1,
                              chat_state1=chat_state1, langchain_mode1=langchain_mode1,
                              h2ogpt_key2=h2ogpt_key2, visible_models1=visible_models1,

                              side_bar_text1=side_bar_text1, doc_count_text1=doc_count_text1,
                              submit_buttons_text1=submit_buttons_text1, visible_models_text1=visible_models_text1,
                              chat_tab_text1=chat_tab_text1, doc_selection_tab_text1=doc_selection_tab_text1,
                              doc_view_tab_text1=doc_view_tab_text1, chat_history_tab_text1=chat_history_tab_text1,
                              expert_tab_text1=expert_tab_text1, models_tab_text1=models_tab_text1,
                              system_tab_text1=system_tab_text1, tos_tab_text1=tos_tab_text1,
                              login_tab_text1=login_tab_text1, hosts_tab_text1=hosts_tab_text1,

                              text_output1=text_output1, text_output21=text_output21,
                              text_outputs1=text_outputs1,
                              username_override=username1, password_to_check=password1,
                              num_model_lock=num_model_lock)
            else:
                success1 = False
                text_result = "Wrong password for user %s" % username1
            df_langchain_mode_paths1 = get_df_langchain_mode_paths(selection_docs_state1, db1s, dbs1=dbs)
            if success1:
                requests_state1['username'] = username1
            if (requests_state1['username'] == get_userid_direct(db1s)) and is_uuid4(requests_state1['username']):
                # still pre-login if both are same hash
                label_instruction1 = 'Ask or Ingest'
            else:
                username = requests_state1['username']
                if username and split_google in username:
                    real_name = split_google.join(username.split(split_google)[0:1])
                else:
                    real_name = username
                label_instruction1 = 'Ask or Ingest, %s' % real_name
            if kwargs['chat_tabless']:
                chat_tab_text1 = 'on'
            return db1s, selection_docs_state1, requests_state1, roles_state1, \
                model_options_state1, lora_options_state1, server_options_state1, \
                chat_state1, \
                text_result, \
                gr.update(label=label_instruction1), \
                df_langchain_mode_paths1, \
                gr.update(choices=list(roles_state1.keys())), \
                gr.update(choices=list(chat_state1.keys()), value=None), \
                gr.update(choices=get_langchain_choices(selection_docs_state1),
                          value=langchain_mode1), \
                h2ogpt_key2, visible_models1, \
                gr.update(visible=True if side_bar_text1 == 'on' else False), \
                gr.update(visible=True if doc_count_text1 == 'on' else False), \
                gr.update(visible=True if submit_buttons_text1 == 'on' else False), \
                gr.update(visible=True if visible_models_text1 == 'on' else False), \
                gr.update(visible=True if chat_tab_text1 == 'on' else False), \
                gr.update(visible=True if doc_selection_tab_text1 == 'on' else False), \
                gr.update(visible=True if doc_view_tab_text1 == 'on' else False), \
                gr.update(visible=True if chat_history_tab_text1 == 'on' else False), \
                gr.update(visible=True if expert_tab_text1 == 'on' else False), \
                gr.update(visible=True if models_tab_text1 == 'on' else False), \
                gr.update(visible=True if system_tab_text1 == 'on' else False), \
                gr.update(visible=True if tos_tab_text1 == 'on' else False), \
                gr.update(visible=True if login_tab_text1 == 'on' else False), \
                gr.update(visible=True if hosts_tab_text1 == 'on' else False), \
                text_output1, text_output21, *tuple(text_outputs1)

        login_func = functools.partial(login,
                                       auth_filename=kwargs['auth_filename'],
                                       num_model_lock=len(text_outputs),
                                       pre_authorized=False,
                                       )
        load_login_func = functools.partial(login,
                                            auth_filename=kwargs['auth_filename'],
                                            num_model_lock=len(text_outputs),
                                            pre_authorized=True,
                                            )
        # FIXME: get_client() in openai server backend.py needs updating if login_inputs changes
        login_inputs = [my_db_state, selection_docs_state, requests_state, roles_state,
                        model_options_state, lora_options_state, server_options_state,
                        chat_state, langchain_mode,
                        h2ogpt_key, visible_models,

                        side_bar_text, doc_count_text, submit_buttons_text, visible_models_text,
                        chat_tab_text, doc_selection_tab_text, doc_view_tab_text, chat_history_tab_text,
                        expert_tab_text, models_tab_text, system_tab_text, tos_tab_text,
                        login_tab_text, hosts_tab_text,

                        username_text, password_text,
                        text_output, text_output2] + text_outputs
        login_outputs = [my_db_state, selection_docs_state, requests_state, roles_state,
                         model_options_state, lora_options_state, server_options_state,
                         chat_state,
                         login_result_text,
                         instruction,
                         langchain_mode_path_text,
                         chatbot_role,
                         radio_chats, langchain_mode,
                         h2ogpt_key, visible_models,

                         side_bar, row_doc_track, submit_buttons, visible_models,
                         chat_tab, doc_selection_tab, doc_view_tab, chat_history_tab,
                         expert_tab, models_tab, system_tab, tos_tab,
                         login_tab, hosts_tab,

                         text_output, text_output2] + text_outputs
        eventdb_loginb = eventdb_logina.then(login_func,
                                             inputs=login_inputs,
                                             outputs=login_outputs,
                                             queue=not kwargs['large_file_count_mode'],
                                             api_name='login')

        admin_pass_textbox.submit(check_admin_pass, inputs=admin_pass_textbox, outputs=system_row,
                                  **noqueue_kwargs) \
            .then(close_admin, inputs=admin_pass_textbox, outputs=admin_row, **noqueue_kwargs)

        def load_auth(db1s, requests_state1, auth_filename=None, selection_docs_state1=None,
                      roles_state1=None,
                      model_options_state1=None,
                      lora_options_state1=None,
                      server_options_state1=None,
                      chat_state1=None, langchain_mode1=None,
                      h2ogpt_key2=None, visible_models1=None,

                      side_bar_text1=None, doc_count_text1=None, submit_buttons_text1=None, visible_models_text1=None,
                      chat_tab_text1=None, doc_selection_tab_text1=None, doc_view_tab_text1=None,
                      chat_history_tab_text1=None,
                      expert_tab_text1=None, models_tab_text1=None, system_tab_text1=None, tos_tab_text1=None,
                      login_tab_text1=None, hosts_tab_text1=None,

                      text_output1=None, text_output21=None,
                      text_outputs1=None,
                      username_override=None, password_to_check=None,
                      num_model_lock=None):
            # in-place assignment
            if not auth_filename:
                return False, "No auth file", text_output1, text_output21, text_outputs1, \
                    langchain_mode1, h2ogpt_key2, visible_models1, \
                    side_bar_text1, doc_count_text1, submit_buttons_text1, visible_models_text1, \
                    chat_tab_text1, doc_selection_tab_text1, doc_view_tab_text1, chat_history_tab_text1, \
                    expert_tab_text1, models_tab_text1, system_tab_text1, tos_tab_text1, \
                    login_tab_text1, hosts_tab_text1
            # if first time here, need to set userID
            set_userid_gr(db1s, requests_state1, get_userid_auth)
            if username_override:
                username1 = username_override
            else:
                username1 = get_username(requests_state1)
            success1 = False
            with filelock.FileLock(auth_filename + '.lock'):
                if os.path.isfile(auth_filename):
                    if auth_filename.endswith('.db'):
                        auth_dict = fetch_user(auth_filename, username1, verbose=verbose)
                    else:
                        with open(auth_filename, 'rt') as f:
                            auth_dict = ujson.load(f)
                    if username1 in auth_dict:
                        auth_user = auth_dict[username1]
                        if password_to_check:
                            if auth_user['password'] != password_to_check:
                                return False, "Invalid password for user %s" % username1, \
                                    text_output1, text_output21, text_outputs1, \
                                    langchain_mode1, h2ogpt_key2, visible_models1, \
                                    side_bar_text1, doc_count_text1, submit_buttons_text1, visible_models_text1, \
                                    chat_tab_text1, doc_selection_tab_text1, doc_view_tab_text1, chat_history_tab_text1, \
                                    expert_tab_text1, models_tab_text1, system_tab_text1, tos_tab_text1, \
                                    login_tab_text1, hosts_tab_text1
                        if username_override:
                            # then use original user id
                            set_userid_direct_gr(db1s, auth_dict[username1]['userid'], username1)
                        if 'selection_docs_state' in auth_user:
                            update_auth_selection(auth_user, selection_docs_state1)
                        if 'roles_state' in auth_user:
                            roles_state1.update(auth_user['roles_state'])
                        if 'model_options_state' in auth_user and \
                                model_options_state1 and \
                                auth_user['model_options_state']:
                            model_options_state1[0].extend(auth_user['model_options_state'][0])
                            model_options_state1[0] = [x for x in model_options_state1[0] if
                                                       x != no_model_str and x]
                            model_options_state1[0] = [no_model_str] + sorted(set(model_options_state1[0]))
                        if 'lora_options_state' in auth_user and \
                                lora_options_state1 and \
                                auth_user['lora_options_state']:
                            lora_options_state1[0].extend(auth_user['lora_options_state'][0])
                            lora_options_state1[0] = [x for x in lora_options_state1[0] if x != no_lora_str and x]
                            lora_options_state1[0] = [no_lora_str] + sorted(set(lora_options_state1[0]))
                        if 'server_options_state' in auth_user and \
                                server_options_state1 and \
                                auth_user['server_options_state']:
                            server_options_state1[0].extend(auth_user['server_options_state'][0])
                            server_options_state1[0] = [x for x in server_options_state1[0] if
                                                        x != no_server_str and x]
                            server_options_state1[0] = [no_server_str] + sorted(set(server_options_state1[0]))
                        if 'chat_state' in auth_user:
                            chat_state1.update(auth_user['chat_state'])
                        if 'text_output' in auth_user:
                            text_output1 = auth_user['text_output']
                        if 'text_output2' in auth_user:
                            text_output21 = auth_user['text_output2']
                        if 'text_outputs' in auth_user:
                            text_outputs1 = auth_user['text_outputs']
                        if 'langchain_mode' in auth_user:
                            langchain_mode1 = auth_user['langchain_mode']
                        if 'h2ogpt_key' in auth_user:
                            h2ogpt_key2 = auth_user['h2ogpt_key']
                        if 'visible_models' in auth_user:
                            visible_models1 = auth_user['visible_models']

                        # other toggles
                        if 'side_bar_text' in auth_user:
                            side_bar_text1 = auth_user['side_bar_text']
                        if 'doc_count_text' in auth_user:
                            doc_count_text1 = auth_user['doc_count_text']
                        if 'submit_buttons_text' in auth_user:
                            submit_buttons_text1 = auth_user['submit_buttons_text']
                        if 'visible_models_text' in auth_user:
                            visible_models_text1 = auth_user['visible_models_text']

                        # gr.TabItem(s)
                        if 'chat_tab_text' in auth_user:
                            chat_tab_text1 = auth_user['chat_tab_text']
                        if 'doc_selection_tab_text' in auth_user:
                            doc_selection_tab_text1 = auth_user['doc_selection_tab_text']
                        if 'doc_view_tab_text' in auth_user:
                            doc_view_tab_text1 = auth_user['doc_view_tab_text']
                        if 'chat_history_tab_text' in auth_user:
                            chat_history_tab_text1 = auth_user['chat_history_tab_text']
                        if 'expert_tab_text' in auth_user:
                            expert_tab_text1 = auth_user['expert_tab_text']
                        if 'models_tab_text' in auth_user:
                            models_tab_text1 = auth_user['models_tab_text']
                        if 'system_tab_text' in auth_user:
                            system_tab_text1 = auth_user['system_tab_text']
                        if 'tos_tab_text' in auth_user:
                            tos_tab_text1 = auth_user['tos_tab_text']
                        if 'login_tab_text' in auth_user:
                            login_tab_text1 = auth_user['login_tab_text']
                        if 'hosts_tab_text' in auth_user:
                            hosts_tab_text1 = auth_user['hosts_tab_text']

                        text_result = "Successful login for %s" % get_show_username(username1)
                        success1 = True
                    else:
                        text_result = "No user %s" % get_show_username(username1)
                else:
                    text_result = "No auth file"
            if num_model_lock is not None:
                if len(text_outputs1) > num_model_lock:
                    text_outputs1 = text_outputs1[:num_model_lock]
                elif len(text_outputs1) < num_model_lock:
                    text_outputs1 = text_outputs1 + [[]] * (num_model_lock - len(text_outputs1))
            else:
                text_outputs1 = []
            # ensure when load, even if unused, that has good state.  Can't be [[]]
            if text_output1 is None:
                text_output1 = []
            if text_output1 and len(text_output1) > 0 and not text_output1[0]:
                text_output1 = []
            if text_output21 is None or not text_output21 and len(text_output21) > 0 and not text_output21[0]:
                text_output21 = []
            if text_output21 is None:
                text_output21 = []
            for i in range(len(text_outputs1)):
                if text_outputs1[i] is None:
                    text_outputs1[i] = []
                if not text_outputs1[i] and len(text_outputs1[i]) > 0 and not text_outputs1[i][0]:
                    text_outputs1[i] = []
            return success1, text_result, text_output1, text_output21, text_outputs1, \
                langchain_mode1, h2ogpt_key2, visible_models1, \
                side_bar_text1, doc_count_text1, submit_buttons_text1, visible_models_text1, \
                chat_tab_text1, doc_selection_tab_text1, doc_view_tab_text1, chat_history_tab_text1, \
                expert_tab_text1, models_tab_text1, system_tab_text1, tos_tab_text1, \
                login_tab_text1, hosts_tab_text1

        def save_auth_dict(auth_dict, auth_filename, username1):
            if auth_filename.endswith('.db'):
                upsert_user(auth_filename, username1, auth_dict[username1], verbose=verbose)
            else:
                backup_file = auth_filename + '.bak' + str(uuid.uuid4())
                if os.path.isfile(auth_filename):
                    shutil.copy(auth_filename, backup_file)
                try:
                    with open(auth_filename, 'wt') as f:
                        f.write(ujson.dumps(auth_dict, indent=2))
                    remove(backup_file)
                except BaseException as e:
                    print("Failure to save auth %s, restored backup: %s: %s" % (auth_filename, backup_file, str(e)),
                          flush=True)
                    shutil.copy(backup_file, auth_dict)
                    if os.getenv('HARD_ASSERTS'):
                        # unexpected in testing or normally
                        raise

        def save_auth(selection_docs_state1, requests_state1, roles_state1,
                      model_options_state1, lora_options_state1, server_options_state1,
                      chat_state1, langchain_mode1,
                      h2ogpt_key1, visible_models1,

                      side_bar_text1, doc_count_text1, submit_buttons_text1, visible_models_text1,
                      chat_tab_text1, doc_selection_tab_text1, doc_view_tab_text1, chat_history_tab_text1,
                      expert_tab_text1, models_tab_text1, system_tab_text1, tos_tab_text1,
                      login_tab_text1, hosts_tab_text1,

                      text_output1, text_output21,
                      text_outputs1,
                      auth_filename=None, auth_access=None, auth_freeze=None, guest_name=None,
                      ):
            if auth_freeze:
                return
            if not auth_filename:
                return
            # save to auth file
            username1 = get_username(requests_state1)
            with filelock.FileLock(auth_filename + '.lock'):
                if os.path.isfile(auth_filename):
                    if auth_filename.endswith('.db'):
                        auth_dict = fetch_user(auth_filename, username1, verbose=verbose)
                    else:
                        with open(auth_filename, 'rt') as f:
                            auth_dict = ujson.load(f)
                    if username1 in auth_dict:
                        auth_user = auth_dict[username1]
                        if selection_docs_state1:
                            update_auth_selection(auth_user, selection_docs_state1, save=True)
                        if roles_state1:
                            # overwrite
                            auth_user['roles_state'] = roles_state1
                        if model_options_state1:
                            # overwrite
                            auth_user['model_options_state'] = model_options_state1
                        if lora_options_state1:
                            # overwrite
                            auth_user['lora_options_state'] = lora_options_state1
                        if server_options_state1:
                            # overwrite
                            auth_user['server_options_state'] = server_options_state1
                        if chat_state1:
                            # overwrite
                            auth_user['chat_state'] = chat_state1
                        if text_output1:
                            auth_user['text_output'] = text_output1
                        if text_output21:
                            auth_user['text_output2'] = text_output21
                        if text_outputs1:
                            auth_user['text_outputs'] = text_outputs1
                        if langchain_mode1:
                            auth_user['langchain_mode'] = langchain_mode1
                        if h2ogpt_key1:
                            auth_user['h2ogpt_key'] = h2ogpt_key1
                        if visible_models1:
                            auth_user['visible_models'] = visible_models1

                        # other toggles
                        if side_bar_text1:
                            auth_user['side_bar_text'] = side_bar_text1
                        if doc_count_text1:
                            auth_user['doc_count_text'] = doc_count_text1
                        if submit_buttons_text1:
                            auth_user['submit_buttons_text'] = submit_buttons_text1
                        if visible_models_text1:
                            auth_user['visible_models_text'] = visible_models_text1

                        # gr.TabItem(s)
                        if chat_tab_text1:
                            auth_user['chat_tab_text'] = chat_tab_text1
                        if doc_selection_tab_text1:
                            auth_user['doc_selection_tab_text'] = doc_selection_tab_text1
                        if doc_view_tab_text1:
                            auth_user['doc_view_tab_text'] = doc_view_tab_text1
                        if chat_history_tab_text1:
                            auth_user['chat_history_tab_text'] = chat_history_tab_text1
                        if expert_tab_text1:
                            auth_user['expert_tab_text'] = expert_tab_text1
                        if models_tab_text1:
                            auth_user['models_tab_text'] = models_tab_text1
                        if system_tab_text1:
                            auth_user['system_tab_text'] = system_tab_text1
                        if tos_tab_text1:
                            auth_user['tos_tab_text'] = tos_tab_text1
                        if login_tab_text1:
                            auth_user['login_tab_text'] = login_tab_text1
                        if hosts_tab_text1:
                            auth_user['hosts_tab_text'] = hosts_tab_text1

                        save_auth_dict(auth_dict, auth_filename, username1)

        def save_auth_wrap(*args, **kwargs):
            save_auth(args[0], args[1], args[2],
                      args[3], args[4], args[5],
                      args[6], args[7],
                      args[8], args[9],

                      # other toggles
                      args[10], args[11], args[12], args[13],

                      # gr.TabItem(s)
                      args[14], args[15], args[16], args[17],
                      args[18], args[19], args[20], args[21],
                      args[22], args[23],
                      # text_output, text_output2
                      args[24], args[25],
                      # text_outputs
                      args[26:],
                      **kwargs
                      )

        save_auth_func = functools.partial(save_auth_wrap,
                                           auth_filename=kwargs['auth_filename'],
                                           auth_access=kwargs['auth_access'],
                                           auth_freeze=kwargs['auth_freeze'],
                                           guest_name=kwargs['guest_name'],
                                           )

        save_auth_kwargs = dict(fn=save_auth_func,
                                inputs=[selection_docs_state, requests_state, roles_state,
                                        model_options_state, lora_options_state, server_options_state,
                                        chat_state, langchain_mode,
                                        h2ogpt_key, visible_models,
                                        side_bar_text, doc_count_text, submit_buttons_text, visible_models_text,
                                        chat_tab_text, doc_selection_tab_text, doc_view_tab_text, chat_history_tab_text,
                                        expert_tab_text, models_tab_text, system_tab_text, tos_tab_text,
                                        login_tab_text, hosts_tab_text,
                                        text_output, text_output2] + text_outputs
                                )
        lg_change_event_auth = lg_change_event.then(**save_auth_kwargs)
        add_role_event_save_event = add_role_event.then(**save_auth_kwargs)

        h2ogpt_key.blur(**save_auth_kwargs)
        h2ogpt_key.submit(**save_auth_kwargs)

        def get_model_lock_visible_list(visible_models1, all_possible_display_names):
            visible_list = []
            for modeli, model in enumerate(all_possible_display_names):
                if visible_models1 is None or \
                        isinstance(model, str) and model in visible_models1 or \
                        isinstance(modeli, int) and modeli in visible_models1:
                    visible_list.append(True)
                else:
                    visible_list.append(False)
            return visible_list

        def set_visible_models(visible_models1, compare_checkbox1, visible_models_text1, num_model_lock=0,
                               all_possible_display_names=None):
            if num_model_lock == 0:
                num_model_lock = 3  # 2 + 1 (which is dup of first)
                ret_list = [gr.update(visible=True)] * num_model_lock
                if not compare_checkbox1:
                    ret_list[1] = gr.update(visible=False)
                # in case switched from lock to not
                visible_models_text1 = 'off'
            else:
                assert isinstance(all_possible_display_names, list)
                assert num_model_lock == len(all_possible_display_names)
                visible_list = [False, False] + get_model_lock_visible_list(visible_models1,
                                                                            all_possible_display_names)
                ret_list = [gr.update(visible=x) for x in visible_list]
            ret_list.insert(0, visible_models_text1)
            ret_list.insert(0, gr.update(visible=visible_models_text1 == 'on'))
            return tuple(ret_list)

        visible_models_func = functools.partial(set_visible_models,
                                                num_model_lock=len(text_outputs),
                                                all_possible_display_names=kwargs['all_possible_display_names'])
        visible_models.change(fn=visible_models_func,
                              inputs=[visible_models, compare_checkbox, visible_models_text],
                              outputs=[visible_models, visible_models_text, text_output, text_output2] + text_outputs,
                              ).then(**save_auth_kwargs)

        def add_langchain_mode(db1s, selection_docs_state1, requests_state1, langchain_mode1, y,
                               h2ogpt_key1,
                               auth_filename=None, auth_freeze=None, guest_name=None,
                               enforce_h2ogpt_api_key=True,
                               enforce_h2ogpt_ui_key=True,
                               h2ogpt_api_keys=[],
                               ):
            valid_key = is_valid_key(enforce_h2ogpt_api_key,
                                     enforce_h2ogpt_ui_key,
                                     h2ogpt_api_keys,
                                     h2ogpt_key1,
                                     requests_state1=requests_state1,
                                     )
            from_ui = is_from_ui(requests_state1)
            if not valid_key:
                raise ValueError(invalid_key_msg)
            assert auth_filename is not None
            assert auth_freeze is not None

            set_userid_gr(db1s, requests_state1, get_userid_auth)
            username1 = get_username(requests_state1)
            for k in db1s:
                set_dbid_gr(db1s[k])
            langchain_modes = selection_docs_state1['langchain_modes']
            langchain_mode_paths = selection_docs_state1['langchain_mode_paths']
            langchain_mode_types = selection_docs_state1['langchain_mode_types']

            user_path = None
            valid = True
            y2 = y.strip().replace(' ', '').split(',')
            if len(y2) >= 1:
                langchain_mode2 = y2[0]
                if len(langchain_mode2) >= 3 and langchain_mode2.isalnum():
                    # real restriction is:
                    # ValueError: Expected collection name that (1) contains 3-63 characters, (2) starts and ends with an alphanumeric character, (3) otherwise contains only alphanumeric characters, underscores or hyphens (-), (4) contains no two consecutive periods (..) and (5) is not a valid IPv4 address, got me
                    # but just make simpler
                    # assume personal if don't have user_path
                    langchain_mode_type = y2[1] if len(y2) > 1 else LangChainTypes.PERSONAL.value
                    user_path = y2[2] if len(y2) > 2 else None  # assume None if don't have user_path
                    if user_path in ['', "''"]:
                        # transcribe UI input
                        user_path = None
                    if langchain_mode_type not in [x.value for x in list(LangChainTypes)]:
                        textbox = "Invalid type %s" % langchain_mode_type
                        valid = False
                        langchain_mode2 = langchain_mode1
                    elif langchain_mode_type == LangChainTypes.SHARED.value and username1.startswith(guest_name):
                        textbox = "Guests cannot add shared collections"
                        valid = False
                        langchain_mode2 = langchain_mode1
                    elif user_path is not None and langchain_mode_type == LangChainTypes.PERSONAL.value:
                        textbox = "Do not pass user_path for personal/scratch types"
                        valid = False
                        langchain_mode2 = langchain_mode1
                    elif user_path is not None and username1.startswith(guest_name):
                        textbox = "Guests cannot add collections with path"
                        valid = False
                        langchain_mode2 = langchain_mode1
                    elif langchain_mode2 in langchain_modes_intrinsic:
                        user_path = None
                        textbox = "Invalid access to use internal name: %s" % langchain_mode2
                        valid = False
                        langchain_mode2 = langchain_mode1
                    elif user_path and allow_upload_to_user_data or not user_path and allow_upload_to_my_data:
                        if user_path:
                            user_path = makedirs(user_path, exist_ok=True, use_base=True)
                        langchain_mode_paths.update({langchain_mode2: user_path})
                        langchain_mode_types.update({langchain_mode2: langchain_mode_type})
                        if langchain_mode2 not in langchain_modes:
                            langchain_modes.append(langchain_mode2)
                        textbox = ''
                    else:
                        valid = False
                        langchain_mode2 = langchain_mode1
                        textbox = "Invalid access.  user allowed: %s " \
                                  "personal/scratch allowed: %s" % (allow_upload_to_user_data, allow_upload_to_my_data)
                else:
                    valid = False
                    langchain_mode2 = langchain_mode1
                    textbox = "Invalid, collection must be >=3 characters and alphanumeric"
            else:
                valid = False
                langchain_mode2 = langchain_mode1
                textbox = "Invalid, must be like UserData2, user_path2"
            selection_docs_state1 = update_langchain_mode_paths(selection_docs_state1)
            df_langchain_mode_paths1 = get_df_langchain_mode_paths(selection_docs_state1, db1s, dbs1=dbs)
            choices = get_langchain_choices(selection_docs_state1)

            if valid and not user_path:
                # needs to have key for it to make it known different from userdata case in _update_user_db()
                from src.gpt_langchain import length_db1
                db1s[langchain_mode2] = [None] * length_db1()
            if valid:
                chat_state1 = None
                roles_state1 = None
                model_options_state1 = None
                lora_options_state1 = None
                server_options_state1 = None
                text_output1, text_output21, text_outputs1 = None, None, None
                h2ogpt_key2, visible_models2 = None, None
                save_auth_func(selection_docs_state1, requests_state1, roles_state1,
                               model_options_state1, lora_options_state1, server_options_state1,
                               chat_state1, langchain_mode2,
                               h2ogpt_key2, visible_models2,
                               None, None, None, None,
                               None, None, None, None,
                               None, None, None, None,
                               None, None,
                               text_output1, text_output21, text_outputs1,
                               )

            return db1s, selection_docs_state1, gr.update(choices=choices,
                                                          value=langchain_mode2), textbox, df_langchain_mode_paths1

        def remove_langchain_mode(db1s, selection_docs_state1, requests_state1,
                                  langchain_mode1, langchain_mode2,
                                  h2ogpt_key2,
                                  dbsu=None, auth_filename=None, auth_freeze=None,
                                  guest_name=None,
                                  purge=False,
                                  enforce_h2ogpt_api_key=True,
                                  enforce_h2ogpt_ui_key=True,
                                  h2ogpt_api_keys=[],
                                  ):
            valid_key = is_valid_key(enforce_h2ogpt_api_key,
                                     enforce_h2ogpt_ui_key,
                                     h2ogpt_api_keys,
                                     h2ogpt_key2,
                                     requests_state1=requests_state1,
                                     )
            from_ui = is_from_ui(requests_state1)
            if not valid_key:
                raise ValueError(invalid_key_msg)

            assert auth_filename is not None
            assert auth_freeze is not None

            set_userid_gr(db1s, requests_state1, get_userid_auth)
            for k in db1s:
                set_dbid_gr(db1s[k])
            assert dbsu is not None
            langchain_modes = selection_docs_state1['langchain_modes']
            langchain_mode_paths = selection_docs_state1['langchain_mode_paths']
            langchain_mode_types = selection_docs_state1['langchain_mode_types']
            langchain_type2 = langchain_mode_types.get(langchain_mode2, LangChainTypes.EITHER.value)

            changed_state = False
            textbox = "Invalid access, cannot remove %s" % langchain_mode2
            in_scratch_db = langchain_mode2 in db1s
            in_user_db = dbsu is not None and langchain_mode2 in dbsu
            if in_scratch_db and not allow_upload_to_my_data or \
                    in_user_db and not allow_upload_to_user_data or \
                    langchain_mode2 in langchain_modes_intrinsic:
                can_remove = False
                can_purge = False
                if langchain_mode2 in langchain_modes_intrinsic:
                    can_purge = True
            else:
                can_remove = True
                can_purge = True

            # change global variables
            if langchain_mode2 in langchain_modes or langchain_mode2 in langchain_mode_paths or langchain_mode2 in db1s:
                if can_purge and purge:
                    # remove source files
                    from src.gpt_langchain import get_sources, del_from_db
                    sources_file, source_list, num_chunks, num_sources_str, db = \
                        get_sources(db1s, selection_docs_state1,
                                    requests_state1, langchain_mode2, dbs=dbsu,
                                    docs_state0=docs_state0,
                                    load_db_if_exists=load_db_if_exists,
                                    db_type=db_type,
                                    use_openai_embedding=use_openai_embedding,
                                    hf_embedding_model=hf_embedding_model,
                                    migrate_embedding_model=migrate_embedding_model,
                                    verbose=verbose,
                                    get_userid_auth=get_userid_auth,
                                    n_jobs=n_jobs)
                    del_from_db(db, source_list, db_type=db_type)
                    for fil in source_list:
                        if os.path.isfile(fil):
                            print("Purged %s" % fil, flush=True)
                            remove(fil)
                    # remove db directory
                    from src.gpt_langchain import get_persist_directory
                    persist_directory, langchain_type2 = \
                        get_persist_directory(langchain_mode2, langchain_type=langchain_type2,
                                              db1s=db1s, dbs=dbsu)
                    print("removed persist_directory %s" % persist_directory, flush=True)
                    remove(persist_directory)
                    textbox = "Purged, but did not remove %s" % langchain_mode2
                if can_remove:
                    if langchain_mode2 in langchain_modes:
                        langchain_modes.remove(langchain_mode2)
                    if langchain_mode2 in langchain_mode_paths:
                        langchain_mode_paths.pop(langchain_mode2)
                    if langchain_mode2 in langchain_mode_types:
                        langchain_mode_types.pop(langchain_mode2)
                    if langchain_mode2 in db1s and langchain_mode2 != LangChainMode.MY_DATA.value:
                        # don't remove last MyData, used as user hash
                        db1s.pop(langchain_mode2)
                    textbox = ""
                    changed_state = True
            else:
                textbox = "%s is not visible" % langchain_mode2

            # update
            selection_docs_state1 = update_langchain_mode_paths(selection_docs_state1)
            df_langchain_mode_paths1 = get_df_langchain_mode_paths(selection_docs_state1, db1s, dbs1=dbs)

            if changed_state:
                chat_state1 = None
                roles_state1 = None
                model_options_state1 = None
                lora_options_state1 = None
                server_options_state1 = None
                text_output1, text_output21, text_outputs1 = None, None, None
                h2ogpt_key2, visible_models2 = None, None
                save_auth_func(selection_docs_state1, requests_state1, roles_state1,
                               model_options_state1, lora_options_state1, server_options_state1,
                               chat_state1, langchain_mode2,
                               h2ogpt_key2, visible_models2,
                               None, None, None, None,
                               None, None, None, None,
                               None, None, None, None,
                               None, None,
                               text_output1, text_output21, text_outputs1,
                               )

            return db1s, selection_docs_state1, \
                gr.update(choices=get_langchain_choices(selection_docs_state1),
                          value=langchain_mode2), textbox, df_langchain_mode_paths1

        eventdb20a = new_langchain_mode_text.submit(user_state_setup,
                                                    inputs=[my_db_state, requests_state, guest_name,
                                                            new_langchain_mode_text, new_langchain_mode_text],
                                                    outputs=[my_db_state, requests_state, new_langchain_mode_text],
                                                    show_progress='minimal')
        add_langchain_mode_func = functools.partial(add_langchain_mode,
                                                    auth_filename=kwargs['auth_filename'],
                                                    auth_freeze=kwargs['auth_freeze'],
                                                    guest_name=kwargs['guest_name'],
                                                    enforce_h2ogpt_api_key=kwargs['enforce_h2ogpt_api_key'],
                                                    enforce_h2ogpt_ui_key=kwargs['enforce_h2ogpt_ui_key'],
                                                    h2ogpt_api_keys=kwargs['h2ogpt_api_keys'],
                                                    )
        eventdb20b = eventdb20a.then(fn=add_langchain_mode_func,
                                     inputs=[my_db_state, selection_docs_state, requests_state,
                                             langchain_mode,
                                             new_langchain_mode_text,
                                             h2ogpt_key],
                                     outputs=[my_db_state, selection_docs_state, langchain_mode,
                                              new_langchain_mode_text,
                                              langchain_mode_path_text],
                                     api_name='new_langchain_mode_text'
                                     if allow_api and (allow_upload_to_user_data or allow_upload_to_my_data) else False)
        db_events.extend([eventdb20a, eventdb20b])

        remove_langchain_mode_func = functools.partial(remove_langchain_mode,
                                                       dbsu=dbs,
                                                       auth_filename=kwargs['auth_filename'],
                                                       auth_freeze=kwargs['auth_freeze'],
                                                       guest_name=kwargs['guest_name'],
                                                       enforce_h2ogpt_api_key=kwargs['enforce_h2ogpt_api_key'],
                                                       enforce_h2ogpt_ui_key=kwargs['enforce_h2ogpt_ui_key'],
                                                       h2ogpt_api_keys=kwargs['h2ogpt_api_keys'],
                                                       )
        eventdb21a = remove_langchain_mode_text.submit(user_state_setup,
                                                       inputs=[my_db_state,
                                                               requests_state, guest_name,
                                                               remove_langchain_mode_text, remove_langchain_mode_text],
                                                       outputs=[my_db_state,
                                                                requests_state, remove_langchain_mode_text],
                                                       show_progress='minimal')
        remove_langchain_mode_kwargs = dict(fn=remove_langchain_mode_func,
                                            inputs=[my_db_state, selection_docs_state, requests_state,
                                                    langchain_mode,
                                                    remove_langchain_mode_text,
                                                    h2ogpt_key],
                                            outputs=[my_db_state, selection_docs_state, langchain_mode,
                                                     remove_langchain_mode_text,
                                                     langchain_mode_path_text])
        eventdb21b = eventdb21a.then(**remove_langchain_mode_kwargs,
                                     api_name='remove_langchain_mode_text'
                                     if allow_api and (allow_upload_to_user_data or allow_upload_to_my_data) else False)
        db_events.extend([eventdb21a, eventdb21b])

        eventdb22a = purge_langchain_mode_text.submit(user_state_setup,
                                                      inputs=[my_db_state,
                                                              requests_state, guest_name,
                                                              purge_langchain_mode_text, purge_langchain_mode_text],
                                                      outputs=[my_db_state,
                                                               requests_state, purge_langchain_mode_text],
                                                      show_progress='minimal')
        purge_langchain_mode_func = functools.partial(remove_langchain_mode_func, purge=True)
        purge_langchain_mode_kwargs = dict(fn=purge_langchain_mode_func,
                                           inputs=[my_db_state, selection_docs_state, requests_state,
                                                   langchain_mode,
                                                   purge_langchain_mode_text,
                                                   h2ogpt_key],
                                           outputs=[my_db_state, selection_docs_state, langchain_mode,
                                                    purge_langchain_mode_text,
                                                    langchain_mode_path_text])
        # purge_langchain_mode_kwargs = remove_langchain_mode_kwargs.copy()
        # purge_langchain_mode_kwargs['fn'] = functools.partial(remove_langchain_mode_kwargs['fn'], purge=True)
        eventdb22b = eventdb22a.then(**purge_langchain_mode_kwargs,
                                     api_name='purge_langchain_mode_text'
                                     if allow_api and (allow_upload_to_user_data or allow_upload_to_my_data) else False)
        eventdb22b_auth = eventdb22b.then(**save_auth_kwargs)
        db_events.extend([eventdb22a, eventdb22b, eventdb22b_auth])

        def load_langchain_gr(db1s, selection_docs_state1, requests_state1, langchain_mode1,
                              h2ogpt_key3,
                              auth_filename=None,
                              enforce_h2ogpt_api_key=kwargs['enforce_h2ogpt_api_key'],
                              enforce_h2ogpt_ui_key=kwargs['enforce_h2ogpt_ui_key'],
                              h2ogpt_api_keys=kwargs['h2ogpt_api_keys'],
                              ):
            valid_key = is_valid_key(enforce_h2ogpt_api_key,
                                     enforce_h2ogpt_ui_key,
                                     h2ogpt_api_keys,
                                     h2ogpt_key3,
                                     requests_state1=requests_state1,
                                     )
            from_ui = is_from_ui(requests_state1)
            if not valid_key:
                raise ValueError(invalid_key_msg)

            load_auth(db1s, requests_state1, auth_filename, selection_docs_state1=selection_docs_state1)

            selection_docs_state1 = update_langchain_mode_paths(selection_docs_state1)
            df_langchain_mode_paths1 = get_df_langchain_mode_paths(selection_docs_state1, db1s, dbs1=dbs)
            return selection_docs_state1, \
                gr.update(choices=get_langchain_choices(selection_docs_state1),
                          value=langchain_mode1), df_langchain_mode_paths1

        eventdbloadla = load_langchain.click(user_state_setup,
                                             inputs=[my_db_state, requests_state, guest_name, langchain_mode],
                                             outputs=[my_db_state, requests_state, langchain_mode],
                                             show_progress='minimal')
        load_langchain_gr_func = functools.partial(load_langchain_gr,
                                                   auth_filename=kwargs['auth_filename'])
        eventdbloadlb = eventdbloadla.then(fn=load_langchain_gr_func,
                                           inputs=[my_db_state, selection_docs_state, requests_state, langchain_mode,
                                                   h2ogpt_key],
                                           outputs=[selection_docs_state, langchain_mode, langchain_mode_path_text],
                                           api_name='load_langchain' if allow_api and allow_upload_to_user_data else False)

        if not kwargs['large_file_count_mode']:
            # FIXME: Could add all these functions, inputs, outputs into single function for snappier GUI
            # all update events when not doing large file count mode
            # Note: Login touches langchain_mode, which triggers all these
            lg_change_event2 = lg_change_event_auth.then(**get_sources_kwargs)
            lg_change_event3 = lg_change_event2.then(fn=update_dropdown, inputs=docs_state, outputs=document_choice)
            lg_change_event4 = lg_change_event3.then(**show_sources_kwargs)
            lg_change_event5 = lg_change_event4.then(**get_viewable_sources_args)
            lg_change_event6 = lg_change_event5.then(**viewable_kwargs)

            # add url text
            eventdb2c = eventdb2.then(**get_sources_kwargs)
            eventdb2d = eventdb2c.then(fn=update_dropdown, inputs=docs_state, outputs=document_choice)
            eventdb2e = eventdb2d.then(**show_sources_kwargs)
            eventdb2f = eventdb2e.then(**get_viewable_sources_args)
            eventdb2g = eventdb2f.then(**viewable_kwargs)

            def docs_to_message(new_files_last1):
                from src.gpt_langchain import image_types, audio_types
                # already filtered by what can show in gradio
                # https://github.com/gradio-app/gradio/issues/3728
                added_history = []
                for k, v in new_files_last1.items():
                    if any(k.endswith(x) for x in image_types):
                        user_message1 = (k,)
                        if v.startswith("The image"):
                            bot_message1 = "Thank you for uploading the Image.  %s" % v
                        else:
                            bot_message1 = "Thank you for uploading the Image.  Looks like: %s" % v
                    elif any(k.endswith(x) for x in audio_types):
                        user_message1 = (k,)
                        bot_message1 = "Thank you for uploading the Audio.  Sounds like it says: %s" % v
                    else:
                        user_message1 = "Upload %s" % k
                        bot_message1 = "Thank you for uploading the File.  Description:\n\n%s" % v
                    added_history.extend([[user_message1, bot_message1]])
                return added_history

            def update_chatbots(*args,
                                num_model_lock=0,
                                all_possible_display_names=None,
                                for_errors=False,
                                gradio_errors_to_chatbot=False):
                args_list = list(args)

                gradio_upload_to_chatbot1 = args_list[0]
                gradio_errors_to_chatbot1 = gradio_errors_to_chatbot and for_errors
                do_show = gradio_upload_to_chatbot1 or gradio_errors_to_chatbot1
                added_history = []

                if not for_errors and str(args_list[1]).strip():
                    new_files_last1 = ast.literal_eval(args_list[1]) if isinstance(args_list[1], str) else {}
                    assert isinstance(new_files_last1, dict)
                    added_history = docs_to_message(new_files_last1)
                elif str(args_list[1]).strip():
                    added_history = [(None, get_accordion_named(args_list[1],
                                                                "Document Ingestion (maybe partial) Failure.  Click Undo to remove this message.",
                                                                font_size=2))]

                compare_checkbox1 = args_list[2]

                if num_model_lock > 0:
                    visible_models1 = args_list[3]
                    assert isinstance(visible_models1, list)
                    assert isinstance(all_possible_display_names, list)
                    visible_list = get_model_lock_visible_list(visible_models1, all_possible_display_names)
                    visible_list = [False, False] + visible_list

                    history_list = args_list[-num_model_lock - 2:]
                    assert len(all_possible_display_names) + 2 == len(history_list)
                else:
                    visible_list = [True, compare_checkbox1]
                    history_list = args_list[-num_model_lock - 2:]

                assert len(history_list) > 0, "Bad history list: %s" % history_list
                if do_show and added_history:
                    for hi, history in enumerate(history_list):
                        if not visible_list[hi]:
                            continue
                        # gradio_upload_to_chatbot_num_max
                        history_list[hi].extend(added_history)
                if len(history_list) > 1:
                    return tuple(history_list)
                else:
                    return history_list[0]

            update_chatbots_func = functools.partial(update_chatbots,
                                                     num_model_lock=len(text_outputs),
                                                     all_possible_display_names=kwargs['all_possible_display_names']
                                                     )
            update_chatbots_kwargs = dict(fn=update_chatbots_func,
                                          inputs=[gradio_upload_to_chatbot,
                                                  new_files_last,
                                                  compare_checkbox,
                                                  visible_models,
                                                  text_output, text_output2] + text_outputs,
                                          outputs=[text_output, text_output2] + text_outputs
                                          )

            update_chatbots_errors_func = functools.partial(update_chatbots,
                                                            num_model_lock=len(text_outputs),
                                                            all_possible_display_names=kwargs[
                                                                'all_possible_display_names'],
                                                            for_errors=True,
                                                            gradio_errors_to_chatbot=kwargs['gradio_errors_to_chatbot'],
                                                            )
            update_chatbots_errors_kwargs = dict(fn=update_chatbots_errors_func,
                                                 inputs=[gradio_upload_to_chatbot,
                                                         doc_exception_text,
                                                         compare_checkbox,
                                                         visible_models,
                                                         text_output, text_output2] + text_outputs,
                                                 outputs=[text_output, text_output2] + text_outputs
                                                 )

            # Ingest, add button
            eventdb2c_btn = eventdb2_btn.then(**get_sources_kwargs)
            eventdb2d_btn = eventdb2c_btn.then(fn=update_dropdown, inputs=docs_state, outputs=document_choice)
            eventdb2e_btn = eventdb2d_btn.then(**show_sources_kwargs)
            eventdb2f_btn = eventdb2e_btn.then(**get_viewable_sources_args)
            eventdb2g_btn = eventdb2f_btn.then(**viewable_kwargs)
            eventdb2h_btn = eventdb2g_btn.then(**update_chatbots_kwargs)
            if kwargs['gradio_errors_to_chatbot']:
                eventdb2i_btn = eventdb2h_btn.then(**update_chatbots_errors_kwargs)

            # file upload
            eventdb1c = eventdb1.then(**get_sources_kwargs)
            eventdb1d = eventdb1c.then(fn=update_dropdown, inputs=docs_state, outputs=document_choice)
            eventdb1e = eventdb1d.then(**show_sources_kwargs)
            eventdb1f = eventdb1e.then(**get_viewable_sources_args)
            eventdb1g = eventdb1f.then(**viewable_kwargs)
            eventdb1h = eventdb1g.then(**update_chatbots_kwargs)
            if kwargs['gradio_errors_to_chatbot']:
                eventdb1i = eventdb1h.then(**update_chatbots_errors_kwargs)

            # add text by hitting enter
            eventdb3c = eventdb3.then(**get_sources_kwargs)
            eventdb3d = eventdb3c.then(fn=update_dropdown, inputs=docs_state, outputs=document_choice)
            eventdb3e = eventdb3d.then(**show_sources_kwargs)
            eventdb3f = eventdb3e.then(**get_viewable_sources_args)
            eventdb3g = eventdb3f.then(**viewable_kwargs)

            # delete
            eventdb90ua = eventdb90.then(**get_sources_kwargs)
            eventdb90ub = eventdb90ua.then(fn=update_dropdown, inputs=docs_state, outputs=document_choice)
            eventdb90uc = eventdb90ub.then(**show_sources_kwargs)
            eventdb90ud = eventdb90uc.then(**get_viewable_sources_args)
            eventdb90ue = eventdb90ud.then(**viewable_kwargs)

            # add langchain mode
            eventdb20c = eventdb20b.then(**get_sources_kwargs)
            eventdb20d = eventdb20c.then(fn=update_dropdown, inputs=docs_state, outputs=document_choice)
            eventdb20e = eventdb20d.then(**show_sources_kwargs)
            eventdb20f = eventdb20e.then(**get_viewable_sources_args)
            eventdb20g = eventdb20f.then(**viewable_kwargs)

            # remove langchain mode
            eventdb21c = eventdb21b.then(**get_sources_kwargs)
            eventdb21d = eventdb21c.then(fn=update_dropdown, inputs=docs_state, outputs=document_choice)
            eventdb21e = eventdb21d.then(**show_sources_kwargs)
            eventdb21f = eventdb21e.then(**get_viewable_sources_args)
            eventdb21g = eventdb21f.then(**viewable_kwargs)

            # purge collection
            eventdb22c = eventdb22b_auth.then(**get_sources_kwargs)
            eventdb22d = eventdb22c.then(fn=update_dropdown, inputs=docs_state, outputs=document_choice)
            eventdb22e = eventdb22d.then(**show_sources_kwargs)
            eventdb22f = eventdb22e.then(**get_viewable_sources_args)
            eventdb22g = eventdb22f.then(**viewable_kwargs)

            # attach
            event_attach3 = event_attach2.then(**get_sources_kwargs)
            event_attach4 = event_attach3.then(fn=update_dropdown, inputs=docs_state, outputs=document_choice)
            event_attach5 = event_attach4.then(**show_sources_kwargs)
            event_attach6 = event_attach5.then(**get_viewable_sources_args)
            event_attach7 = event_attach6.then(**viewable_kwargs)
            event_attach8 = event_attach7.then(**update_chatbots_kwargs)

            sync2 = sync1.then(**get_sources_kwargs)
            sync3 = sync2.then(fn=update_dropdown, inputs=docs_state, outputs=document_choice)
            sync4 = sync3.then(**show_sources_kwargs)
            sync5 = sync4.then(**get_viewable_sources_args)
            sync6 = sync5.then(**viewable_kwargs)

            def update_model_dropdown(model_options_state1, lora_options_state1, server_options_state1,
                                      model_choice1, lora_choice1, server_choice1,
                                      model_choice12, lora_choice12, server_choice12):
                return gr.Dropdown(choices=model_options_state1[0], value=model_choice1), \
                    gr.Dropdown(choices=lora_options_state1[0], value=lora_choice1), \
                    gr.Dropdown(choices=server_options_state1[0], value=server_choice1), \
                    gr.Dropdown(choices=model_options_state1[0], value=model_choice12), \
                    gr.Dropdown(choices=lora_options_state1[0], value=lora_choice12), \
                    gr.Dropdown(choices=server_options_state1[0], value=server_choice12)

            eventdb_loginbb = eventdb_loginb.then(**get_sources_kwargs)
            eventdb_loginc = eventdb_loginbb.then(fn=update_dropdown, inputs=docs_state, outputs=document_choice)
            # FIXME: Fix redundancy
            eventdb_logind = eventdb_loginc.then(**show_sources_kwargs)
            eventdb_logine = eventdb_logind.then(**get_viewable_sources_args)
            eventdb_loginf = eventdb_logine.then(**viewable_kwargs)
            eventdb_loginh = eventdb_loginf.then(fn=update_model_dropdown,
                                                 inputs=[model_options_state, lora_options_state, server_options_state,
                                                         model_choice, lora_choice, server_choice,
                                                         model_choice2, lora_choice2, server_choice2,
                                                         ],
                                                 outputs=[model_choice, lora_choice, server_choice,
                                                          model_choice2, lora_choice2, server_choice2,
                                                          ]
                                                 )

            db_events.extend([lg_change_event_auth,
                              lg_change_event, lg_change_event2, lg_change_event3, lg_change_event4, lg_change_event5,
                              lg_change_event6] +
                             [eventdb2c, eventdb2d, eventdb2e, eventdb2f, eventdb2g] +
                             [eventdb1c, eventdb1d, eventdb1e, eventdb1f, eventdb1g] +
                             [eventdb3c, eventdb3d, eventdb3e, eventdb3f, eventdb3g] +
                             [eventdb90ua, eventdb90ub, eventdb90uc, eventdb90ud, eventdb90ue] +
                             [eventdb20c, eventdb20d, eventdb20e, eventdb20f, eventdb20g] +
                             [eventdb21c, eventdb21d, eventdb21e, eventdb21f, eventdb21g] +
                             [eventdb22b_auth, eventdb22c, eventdb22d, eventdb22e, eventdb22f, eventdb22g] +
                             [event_attach3, event_attach4, event_attach5, event_attach6, event_attach7] +
                             [sync1, sync2, sync3, sync4, sync5, sync6] +
                             [eventdb_logina, eventdb_loginb, eventdb_loginbb,
                              eventdb_loginc, eventdb_logind, eventdb_logine,
                              eventdb_loginf]
                             ,
                             )

        inputs_list, inputs_dict = get_inputs_list(all_kwargs, kwargs['model_lower'], model_id=1)
        inputs_list2, inputs_dict2 = get_inputs_list(all_kwargs, kwargs['model_lower'], model_id=2)
        from functools import partial
        kwargs_evaluate = {k: v for k, v in all_kwargs.items() if k in inputs_kwargs_list}
        kwargs_evaluate.update(dict(from_ui=True))  # default except for evaluate_nochat
        # ensure present
        for k in inputs_kwargs_list:
            assert k in kwargs_evaluate, "Missing %s" % k

        kwargs_evaluate_nochat = kwargs_evaluate.copy()
        # nominally never want sources appended for API calls, which is what nochat used for primarily
        kwargs_evaluate_nochat.update(dict(append_sources_to_answer=False,
                                           from_ui=False, append_sources_to_chat=False,
                                           selection_docs_state0=selection_docs_state0,
                                           requests_state0=requests_state0,
                                           roles_state0=roles_state0,
                                           model_states=model_states,
                                           is_public=is_public,
                                           verbose=verbose,
                                           ))
        from src.gradio_funcs import evaluate_nochat
        fun = partial(evaluate_nochat,
                      default_kwargs1=default_kwargs,
                      str_api=False,
                      kwargs=kwargs,
                      **kwargs_evaluate_nochat)
        fun_with_dict_str = partial(evaluate_nochat,
                                    default_kwargs1=default_kwargs,
                                    str_api=True,
                                    kwargs=kwargs,
                                    **kwargs_evaluate_nochat
                                    )

        fun_with_dict_str_plain = get_fun_with_dict_str_plain(default_kwargs, kwargs, **kwargs_evaluate_nochat)
        fun_with_dict_verifier = partial(fun_with_dict_str_plain,
                                         verifier=True,
                                         )

        dark_mode_btn.click(
            None,
            None,
            None,
            api_name="dark" if allow_api else False,
            **dark_kwargs,
            **noqueue_kwargs,
        )

        # Handle uploads from API
        upload_api_btn = gr.UploadButton("Upload File Results", visible=False)
        file_upload_api = gr.File(visible=False)
        file_upload_text = gr.Textbox(visible=False)

        def upload_file(files):
            if isinstance(files, list):
                file_paths = [file.name for file in files]
            else:
                file_paths = files.name
            return file_paths, file_paths

        upload_api_btn.upload(fn=upload_file,
                              inputs=upload_api_btn,
                              outputs=[file_upload_api, file_upload_text],
                              api_name='upload_api' if allow_upload_api else False)

        def visible_toggle(x):
            x = 'off' if x == 'on' else 'on'
            return x, gr.update(visible=True if x == 'on' else False)

        side_bar_btn.click(fn=visible_toggle,
                           inputs=side_bar_text,
                           outputs=[side_bar_text, side_bar],
                           **noqueue_kwargs).then(**save_auth_kwargs)

        doc_count_btn.click(fn=visible_toggle,
                            inputs=doc_count_text,
                            outputs=[doc_count_text, row_doc_track],
                            **noqueue_kwargs).then(**save_auth_kwargs)

        submit_buttons_btn.click(fn=visible_toggle,
                                 inputs=submit_buttons_text,
                                 outputs=[submit_buttons_text, submit_buttons],
                                 **noqueue_kwargs).then(**save_auth_kwargs)

        visible_model_btn.click(fn=visible_toggle,
                                inputs=visible_models_text,
                                outputs=[visible_models_text, visible_models],
                                **noqueue_kwargs).then(**save_auth_kwargs)

        chat_tab_btn.click(fn=visible_toggle,
                           inputs=chat_tab_text,
                           outputs=[chat_tab_text, chat_tab],
                           **noqueue_kwargs).then(**save_auth_kwargs)

        doc_selection_btn.click(fn=visible_toggle,
                                inputs=doc_selection_tab_text,
                                outputs=[doc_selection_tab_text, doc_selection_tab],
                                **noqueue_kwargs).then(**save_auth_kwargs)

        doc_view_tab_btn.click(fn=visible_toggle,
                               inputs=doc_view_tab_text,
                               outputs=[doc_view_tab_text, doc_view_tab],
                               **noqueue_kwargs).then(**save_auth_kwargs)

        chat_history_btn.click(fn=visible_toggle,
                               inputs=chat_history_tab_text,
                               outputs=[chat_history_tab_text, chat_history_tab],
                               **noqueue_kwargs).then(**save_auth_kwargs)

        expert_tab_btn.click(fn=visible_toggle,
                             inputs=expert_tab_text,
                             outputs=[expert_tab_text, expert_tab],
                             **noqueue_kwargs).then(**save_auth_kwargs)

        models_tab_btn.click(fn=visible_toggle,
                             inputs=models_tab_text,
                             outputs=[models_tab_text, models_tab],
                             **noqueue_kwargs).then(**save_auth_kwargs)

        system_tab_btn.click(fn=visible_toggle,
                             inputs=system_tab_text,
                             outputs=[system_tab_text, system_tab],
                             **noqueue_kwargs).then(**save_auth_kwargs)

        tos_tab_btn.click(fn=visible_toggle,
                          inputs=tos_tab_text,
                          outputs=[tos_tab_text, tos_tab],
                          **noqueue_kwargs).then(**save_auth_kwargs)

        login_tab_btn.click(fn=visible_toggle,
                            inputs=login_tab_text,
                            outputs=[login_tab_text, login_tab],
                            **noqueue_kwargs).then(**save_auth_kwargs)

        hosts_tab_btn.click(fn=visible_toggle,
                            inputs=hosts_tab_text,
                            outputs=[hosts_tab_text, hosts_tab],
                            **noqueue_kwargs).then(**save_auth_kwargs)

        # examples after submit or any other buttons for chat or no chat
        if kwargs['examples'] is not None and kwargs['show_examples']:
            gr.Examples(examples=kwargs['examples'], inputs=inputs_list)

        # Score
        def score_last_response(*args, nochat=False, num_model_lock=0):
            try:
                if num_model_lock > 0:
                    # then lock way
                    args_list = list(args).copy()
                    outputs = args_list[-num_model_lock:]
                    score_texts1 = []
                    for output in outputs:
                        # same input, put into form good for _score_last_response()
                        args_list[-1] = output
                        score_texts1.append(
                            _score_last_response(*tuple(args_list), nochat=nochat,
                                                 num_model_lock=num_model_lock, prefix=''))
                    if len(score_texts1) > 1:
                        return "Response Scores: %s" % ' '.join(score_texts1)
                    else:
                        return "Response Scores: %s" % score_texts1[0]
                else:
                    return _score_last_response(*args, nochat=nochat, num_model_lock=num_model_lock)
            finally:
                clear_torch_cache(allow_skip=True)

        def _score_last_response(*args, nochat=False, num_model_lock=0, prefix='Response Score: '):
            """ Similar to user() """
            args_list = list(args)
            smodel = score_model_state0['model']
            stokenizer = score_model_state0['tokenizer']
            sdevice = score_model_state0['device']
            reward_model = score_model_state0['reward_model']

            if not nochat:
                history = args_list[-1]
                history = get_llm_history(history)
                if smodel is not None and \
                        stokenizer is not None and \
                        sdevice is not None and \
                        history is not None and len(history) > 0 and \
                        history[-1] is not None and \
                        len(history[-1]) >= 2:
                    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

                    question = history[-1][0]
                    answer = history[-1][1]
                else:
                    return '%sNA' % prefix
            else:
                answer = args_list[-1]
                instruction_nochat_arg_id = eval_func_param_names.index('instruction_nochat')
                question = args_list[instruction_nochat_arg_id]

            if question is None:
                return '%sBad Question' % prefix
            if answer is None:
                return '%sBad Answer' % prefix
            score = score_qa(smodel, stokenizer, question, answer, memory_restriction_level=memory_restriction_level)
            if reward_model:
                if isinstance(score, str):
                    return '%sNA' % prefix
                return '{}{:.1%}'.format(prefix, score)
            else:
                # any text
                return score

        def noop_score_last_response(*args, **kwargs):
            return "Response Score: Disabled"

        if kwargs['score_model']:
            score_fun = score_last_response
        else:
            score_fun = noop_score_last_response

        score_args = dict(fn=score_fun,
                          inputs=inputs_list + [text_output],
                          outputs=[score_text],
                          )
        score_args2 = dict(fn=partial(score_fun),
                           inputs=inputs_list2 + [text_output2],
                           outputs=[score_text2],
                           )
        score_fun_func = functools.partial(score_fun, num_model_lock=len(text_outputs))
        all_score_args = dict(fn=score_fun_func,
                              inputs=inputs_list + text_outputs,
                              outputs=score_text,
                              )

        score_args_nochat = dict(fn=partial(score_fun, nochat=True),
                                 inputs=inputs_list + [text_output_nochat],
                                 outputs=[score_text_nochat],
                                 )

        def update_history(*args, undo=False, retry=False, sanitize_user_prompt=False):
            """
            User that fills history for bot
            :param args:
            :param undo:
            :param retry:
            :param sanitize_user_prompt:
            :return:
            """
            args_list = list(args)
            user_message = args_list[eval_func_param_names.index('instruction')]  # chat only
            input1 = args_list[eval_func_param_names.index('iinput')]  # chat only
            prompt_type1 = args_list[eval_func_param_names.index('prompt_type')]
            langchain_mode1 = args_list[eval_func_param_names.index('langchain_mode')]
            langchain_action1 = args_list[eval_func_param_names.index('langchain_action')]
            langchain_agents1 = args_list[eval_func_param_names.index('langchain_agents')]
            document_subset1 = args_list[eval_func_param_names.index('document_subset')]
            document_choice1 = args_list[eval_func_param_names.index('document_choice')]
            if not prompt_type1:
                # shouldn't have to specify if CLI launched model
                prompt_type1 = kwargs['prompt_type']
                # apply back
                args_list[eval_func_param_names.index('prompt_type')] = prompt_type1
            if input1 and not user_message.endswith(':'):
                user_message1 = user_message + ":" + input1
            elif input1:
                user_message1 = user_message + input1
            else:
                user_message1 = user_message
            if sanitize_user_prompt:
                pass
                # requirements.txt has comment that need to re-enable the below 2 lines
                # from better_profanity import profanity
                # user_message1 = profanity.censor(user_message1)

            history = args_list[-1]
            if history is None:
                # bad history
                history = []
            history = history.copy()

            if undo:
                history = get_llm_history(history)
                if len(history) > 0:
                    history.pop()
                return history
            if retry:
                history = get_llm_history(history)
                if history:
                    history[-1][1] = None
                    if isinstance(history[-1][0], (tuple, list)):
                        if history[-1][0] is None:
                            history[-1][0] = ''
                        elif isinstance(history[-1][0], (tuple, list)):
                            history[-1][0] = history[-1][0][0]
                return history
            if user_message1 in ['', None, '\n']:
                if not allow_empty_instruction(langchain_mode1, document_subset1, langchain_action1):
                    # reject non-retry submit/enter
                    return history
            user_message1 = fix_text_for_gradio(user_message1)
            if not user_message1 and langchain_action1 == LangChainAction.SUMMARIZE_MAP.value:
                user_message1 = '%s%s, Subset: %s, Documents: %s' % (
                    summary_prefix, langchain_mode1, document_subset1, document_choice1)
            if not user_message1 and langchain_action1 == LangChainAction.EXTRACT.value:
                user_message1 = '%s%s, Subset: %s, Documents: %s' % (
                    extract_prefix, langchain_mode1, document_subset1, document_choice1)
            return history + [[user_message1, None]]

        def user(*args, undo=False, retry=False, sanitize_user_prompt=False):
            return update_history(*args, undo=undo, retry=retry, sanitize_user_prompt=sanitize_user_prompt)

        def all_user(*args, undo=False, retry=False, sanitize_user_prompt=False, num_model_lock=0,
                     all_possible_display_names=None):
            args_list = list(args)

            visible_models1 = args_list[eval_func_param_names.index('visible_models')]
            assert isinstance(all_possible_display_names, list)
            visible_list = get_model_lock_visible_list(visible_models1, all_possible_display_names)

            history_list = args_list[-num_model_lock:]
            assert len(all_possible_display_names) == len(history_list)
            assert len(history_list) > 0, "Bad history list: %s" % history_list
            for hi, history in enumerate(history_list):
                if not visible_list[hi]:
                    continue
                if num_model_lock > 0:
                    hargs = args_list[:-num_model_lock].copy()
                else:
                    hargs = args_list.copy()
                hargs += [history]
                history_list[hi] = update_history(*hargs, undo=undo, retry=retry,
                                                  sanitize_user_prompt=sanitize_user_prompt)
            if len(history_list) > 1:
                return tuple(history_list)
            else:
                return history_list[0]

        def all_bot(*args, retry=False, model_states1=None, all_possible_display_names=None):
            args_list = list(args).copy()
            chatbots = args_list[-len(model_states1):]
            args_list0 = args_list[:-len(model_states1)]  # same for all models
            exceptions = []
            stream_output1 = args_list[eval_func_param_names.index('stream_output')]
            max_time1 = args_list[eval_func_param_names.index('max_time')]
            langchain_mode1 = args_list[eval_func_param_names.index('langchain_mode')]

            visible_models1 = args_list[eval_func_param_names.index('visible_models')]
            assert isinstance(all_possible_display_names, list)
            assert len(all_possible_display_names) == len(model_states1)
            visible_list = get_model_lock_visible_list(visible_models1, all_possible_display_names)

            langchain_action1 = args_list[eval_func_param_names.index('langchain_action')]

            image_files_to_delete = []

            isize = len(input_args_list) + 1  # states + chat history
            db1s = None
            requests_state1 = None
            valid_key = False
            h2ogpt_key1 = ''
            sources_all = []
            exceptions = []
            save_dicts = []
            audios = []  # in case not streaming, since audio is always streaming, need to accumulate for when yield
            chatbot_role1 = None
            try:
                gen_list = []
                num_visible_bots = sum(visible_list)
                first_visible = True
                for chatboti, (chatbot1, model_state1) in enumerate(zip(chatbots, model_states1)):
                    args_list1 = args_list0.copy()
                    # insert at -2 so is at -3, and after chatbot1 added, at -4
                    args_list1.insert(-isize + 2, model_state1)
                    # if at start, have None in response still, replace with '' so client etc. acts like normal
                    # assumes other parts of code treat '' and None as if no response yet from bot
                    # can't do this later in bot code as racy with threaded generators
                    if chatbot1 is None:
                        chatbot1 = []
                    if len(chatbot1) > 0 and len(chatbot1[-1]) == 2 and chatbot1[-1][1] is None:
                        chatbot1[-1][1] = ''
                    args_list1.append(chatbot1)
                    # so consistent with prep_bot()
                    # with model_state1 at -3, my_db_state1 at -2, and history(chatbot) at -1
                    # langchain_mode1 and my_db_state1 and requests_state1 should be same for every bot
                    history, fun1, langchain_mode1, db1s, requests_state1, \
                        valid_key, h2ogpt_key1, \
                        max_time1, stream_output1, \
                        chatbot_role1, speaker1, tts_language1, roles_state1, tts_speed1, \
                        langchain_action1, \
                        image_files_to_delete = \
                        prep_bot(*tuple(args_list1), retry=retry, which_model=chatboti, kwargs_eval=kwargs_evaluate,
                                 kwargs=kwargs, verbose=verbose)
                    if num_visible_bots == 1:
                        # no need to lag, will be faster this way
                        lag = 0
                    else:
                        lag = 1e-3
                    if visible_list[chatboti]:
                        gen1 = get_response(fun1, history,
                                            chatbot_role1 if first_visible else 'None',
                                            speaker1 if first_visible else 'None',
                                            tts_language1 if first_visible else 'autodetect',
                                            roles_state1 if first_visible else {},
                                            tts_speed1 if first_visible else 1.0,
                                            langchain_action1,
                                            kwargs=kwargs,
                                            api=False,
                                            verbose=verbose,
                                            )
                        # FIXME: only first visible chatbot is allowed to speak for now
                        first_visible = False
                        # always use stream or not, so do not block any iterator/generator
                        gen1 = TimeoutIterator(gen1, timeout=lag, sentinel=None, raise_on_exception=False,
                                               whichi=chatboti)
                        # else timeout will truncate output for non-streaming case
                    else:
                        gen1 = gen1_fake(fun1, history)
                    gen_list.append(gen1)
            finally:
                pass

            bots = bots_old = chatbots.copy()
            bot_strs = bot_strs_old = str(chatbots)
            exceptions = exceptions_old = [''] * len(bots_old)
            exceptions_str = '\n'.join(
                ['Model %s: %s' % (iix, choose_exc(x)) for iix, x in enumerate(exceptions) if
                 x not in [None, '', 'None']])
            exceptions_each_str = [''] * len(bots_old)
            exceptions_old_str = exceptions_str
            sources = sources_all_old = [[]] * len(bots_old)
            sources_str = sources_str_all_old = [''] * len(bots_old)
            sources_str_all = [None] * len(bots_old)
            prompt_raw = prompt_raw_all_old = [''] * len(bots_old)
            llm_answers = llm_answers_all_old = [{}] * len(bots_old)
            save_dicts = save_dicts_old = [{}] * len(bots_old)
            if kwargs['tts_model'].startswith('microsoft'):
                from src.tts_utils import prepare_speech, get_no_audio
                no_audio = get_no_audio(sr=16000)
            elif kwargs['tts_model'].startswith('tts_models/'):
                from src.tts_utils import prepare_speech, get_no_audio
                no_audio = get_no_audio(sr=24000)
            else:
                no_audio = None

            tgen0 = time.time()
            last_yield = None
            try:
                for res1 in itertools.zip_longest(*gen_list):
                    do_yield = False
                    bots = [x[0] if x is not None and not isinstance(x, BaseException) else y
                            for x, y in zip(res1, bots_old)]
                    bot_strs = [str(x) for x in bots]
                    could_yield = any(x != y for x, y in zip(bot_strs, bot_strs_old))
                    if kwargs['gradio_ui_stream_chunk_size'] <= 0:
                        do_yield |= could_yield
                    else:
                        enough_data = any(abs(len(x) - len(y)) > kwargs['gradio_ui_stream_chunk_size']
                                          for x, y in zip(bot_strs, bot_strs_old))
                        beyond_min_time = last_yield is None or \
                                          last_yield is not None and \
                                          (time.time() - last_yield) > kwargs['gradio_ui_stream_chunk_min_seconds']
                        do_yield |= enough_data and beyond_min_time
                        enough_time = last_yield is None or \
                                      last_yield is not None and \
                                      (time.time() - last_yield) > kwargs['gradio_ui_stream_chunk_seconds']
                        do_yield |= enough_time and could_yield
                        # DEBUG: print("do_yield: %s : %s %s %s" % (do_yield, enough_data, beyond_min_time, enough_time), flush=True)
                    if do_yield:
                        bot_strs_old = bot_strs.copy()

                    def larger_str(x, y):
                        return x if len(x) > len(y) else y

                    exceptions = [x[1] if x is not None and not isinstance(x, BaseException) else larger_str(str(x), y)
                                  for x, y in zip(res1, exceptions_old)]
                    exceptions_each_str = [
                        get_accordion_named(choose_exc(x), "Generate Error", font_size=2) if x not in ['', None,
                                                                                                       'None'] else ''
                        for x in exceptions]
                    do_yield |= any(
                        x != y for x, y in zip(exceptions, exceptions_old) if (x not in noneset or y not in noneset))
                    exceptions_old = exceptions.copy()

                    sources_all = [x[2] if x is not None and not isinstance(x, BaseException) else y
                                   for x, y in zip(res1, sources_all_old)]
                    sources_all_old = sources_all.copy()

                    sources_str_all = [x[3] if x is not None and not isinstance(x, BaseException) else y
                                       for x, y in zip(res1, sources_str_all_old)]
                    sources_str_all_old = sources_str_all.copy()

                    prompt_raw_all = [x[4] if x is not None and not isinstance(x, BaseException) else y
                                      for x, y in zip(res1, prompt_raw_all_old)]
                    prompt_raw_all_old = prompt_raw_all.copy()

                    llm_answers_all = [x[5] if x is not None and not isinstance(x, BaseException) else y
                                       for x, y in zip(res1, llm_answers_all_old)]
                    llm_answers_all_old = llm_answers_all.copy()

                    save_dicts = [x[6] if x is not None and not isinstance(x, BaseException) else y
                                  for x, y in zip(res1, save_dicts_old)]
                    save_dicts_old = save_dicts.copy()

                    exceptions_str = '\n'.join(
                        ['Model %s: %s' % (iix, choose_exc(x)) for iix, x in enumerate(exceptions) if
                         x not in noneset])

                    audios_gen = [x[7] if x is not None and not isinstance(x, BaseException) else None for x in
                                  res1]
                    audios_gen = [x for x in audios_gen if x is not None]
                    if os.getenv('HARD_ASSERTS'):
                        # FIXME: should only be 0 or 1 speaker in all_bot mode for now
                        assert len(audios_gen) in [0, 1], "Wrong len audios_gen: %s" % len(audios_gen)
                    audio1 = audios_gen[0] if len(audios_gen) == 1 else no_audio
                    do_yield |= audio1 != no_audio

                    # yield back to gradio only is bots + exceptions, rest are consumed locally
                    if stream_output1 and do_yield:
                        audio1 = combine_audios(audios, audio=audio1, sr=24000 if chatbot_role1 else 16000,
                                                expect_bytes=kwargs['return_as_byte'], verbose=verbose)
                        audios = []  # reset accumulation
                        # update bots_old
                        bots_old = bots.copy()
                        if len(bots) > 1:
                            yield tuple(bots + [exceptions_str, audio1])
                        else:
                            yield bots[0], exceptions_str, audio1
                        last_yield = time.time()
                    else:
                        audios.append(audio1)
                    if time.time() - tgen0 > max_time1 + 10:  # don't use actual, so inner has chance to complete
                        if verbose:
                            print("Took too long all_bot: %s" % (time.time() - tgen0), flush=True)
                        break
                if exceptions:
                    exceptions_reduced = [x for x in exceptions if x not in ['', None, 'None']]
                    if exceptions_reduced:
                        print("Generate exceptions: %s" % exceptions_reduced, flush=True)

                # yield if anything left over as can happen (FIXME: Understand better)
                final_audio = combine_audios(audios, audio=no_audio,
                                             expect_bytes=kwargs['return_as_byte'], verbose=verbose)
                # add error accordion
                for boti, bot1 in enumerate(bots):
                    if bots[boti] and bots[boti][-1] and len(bots[boti][-1]) == 2 and exceptions_each_str[boti]:
                        if not bots[boti][-1][1]:
                            bots[boti][-1][1] = exceptions_each_str[boti]
                        else:
                            bots[boti].append((None, exceptions_each_str[boti]))
                    if kwargs['append_sources_to_chat'] and sources_str_all[boti]:
                        bots[boti].append((None, sources_str_all[boti]))

                if len(bots) > 1:
                    yield tuple(bots + [exceptions_str, final_audio])
                else:
                    yield bots[0], exceptions_str, final_audio
            finally:
                clear_torch_cache(allow_skip=True)
                clear_embeddings(langchain_mode1, db_type, db1s, dbs)
            for image_file1 in image_files_to_delete:
                if os.path.isfile(image_file1):
                    remove(image_file1)

            # save
            for sources, error, save_dict, model_name in zip(sources_all, exceptions, save_dicts,
                                                             all_possible_display_names):
                if 'extra_dict' not in save_dict:
                    save_dict['extra_dict'] = {}
                if requests_state1:
                    save_dict['extra_dict'].update(requests_state1)
                else:
                    save_dict['extra_dict'].update(dict(username='NO_REQUEST'))
                save_dict['error'] = error
                save_dict['sources'] = sources
                save_dict['which_api'] = 'all_bot_%s' % model_name
                save_dict['valid_key'] = valid_key
                save_dict['h2ogpt_key'] = h2ogpt_key1
                save_dict['save_dir'] = kwargs['save_dir']
                save_generate_output(**save_dict)

        # NORMAL MODEL
        user_args = dict(fn=functools.partial(user, sanitize_user_prompt=kwargs['sanitize_user_prompt']),
                         inputs=inputs_list + [text_output],
                         outputs=text_output,
                         )
        bot_args = dict(
            fn=functools.partial(bot, kwargs_evaluate=kwargs_evaluate, kwargs=kwargs, db_type=db_type, dbs=dbs,
                                 verbose=verbose),
            inputs=inputs_list + [model_state, my_db_state, selection_docs_state, requests_state,
                                  roles_state] + [
                       text_output],
            outputs=[text_output, chat_exception_text, speech_bot],
        )
        retry_bot_args = dict(
            fn=functools.partial(bot, retry=True, kwargs_evaluate=kwargs_evaluate, kwargs=kwargs, db_type=db_type,
                                 dbs=dbs, verbose=verbose),
            inputs=inputs_list + [model_state, my_db_state, selection_docs_state, requests_state,
                                  roles_state] + [
                       text_output],
            outputs=[text_output, chat_exception_text, speech_bot],
        )
        retry_user_args = dict(
            fn=functools.partial(user, retry=True, sanitize_user_prompt=kwargs['sanitize_user_prompt']),
            inputs=inputs_list + [text_output],
            outputs=text_output,
        )
        undo_user_args = dict(
            fn=functools.partial(user, undo=True, sanitize_user_prompt=kwargs['sanitize_user_prompt']),
            inputs=inputs_list + [text_output],
            outputs=text_output,
        )

        # MODEL2
        user_args2 = dict(fn=functools.partial(user, sanitize_user_prompt=kwargs['sanitize_user_prompt']),
                          inputs=inputs_list2 + [text_output2],
                          outputs=text_output2,
                          )
        bot_args2 = dict(
            fn=functools.partial(bot, kwargs_evaluate=kwargs_evaluate, kwargs=kwargs, db_type=db_type, dbs=dbs,
                                 verbose=verbose),
            inputs=inputs_list2 + [model_state2, my_db_state, selection_docs_state, requests_state,
                                   roles_state] + [
                       text_output2],
            outputs=[text_output2, chat_exception_text, speech_bot2],
        )
        retry_bot_args2 = dict(
            fn=functools.partial(bot, retry=True, kwargs_evaluate=kwargs_evaluate, kwargs=kwargs, db_type=db_type,
                                 dbs=dbs, verbose=verbose),
            inputs=inputs_list2 + [model_state2, my_db_state, selection_docs_state,
                                   requests_state, roles_state] + [
                       text_output2],
            outputs=[text_output2, chat_exception_text, speech_bot2],
        )
        retry_user_args2 = dict(fn=functools.partial(user, retry=True),
                                inputs=inputs_list2 + [text_output2],
                                outputs=text_output2,
                                )
        undo_user_args2 = dict(fn=functools.partial(user, undo=True),
                               inputs=inputs_list2 + [text_output2],
                               outputs=text_output2,
                               )

        # MODEL N
        all_user_args = dict(fn=functools.partial(all_user,
                                                  sanitize_user_prompt=kwargs['sanitize_user_prompt'],
                                                  num_model_lock=len(text_outputs),
                                                  all_possible_display_names=kwargs['all_possible_display_names']
                                                  ),
                             inputs=inputs_list + text_outputs,
                             outputs=text_outputs,
                             )
        all_bot_args = dict(fn=functools.partial(all_bot, model_states1=model_states,
                                                 all_possible_display_names=kwargs['all_possible_display_names']),
                            inputs=inputs_list + [my_db_state, selection_docs_state, requests_state, roles_state] +
                                   text_outputs,
                            outputs=text_outputs + [chat_exception_text, speech_bot],
                            )
        all_retry_bot_args = dict(fn=functools.partial(all_bot, model_states1=model_states,
                                                       all_possible_display_names=kwargs[
                                                           'all_possible_display_names'],
                                                       retry=True),
                                  inputs=inputs_list + [my_db_state, selection_docs_state, requests_state,
                                                        roles_state] +
                                         text_outputs,
                                  outputs=text_outputs + [chat_exception_text, speech_bot],
                                  )
        all_retry_user_args = dict(fn=functools.partial(all_user, retry=True,
                                                        sanitize_user_prompt=kwargs['sanitize_user_prompt'],
                                                        num_model_lock=len(text_outputs),
                                                        all_possible_display_names=kwargs[
                                                            'all_possible_display_names']
                                                        ),
                                   inputs=inputs_list + text_outputs,
                                   outputs=text_outputs,
                                   )
        all_undo_user_args = dict(fn=functools.partial(all_user, undo=True,
                                                       sanitize_user_prompt=kwargs['sanitize_user_prompt'],
                                                       num_model_lock=len(text_outputs),
                                                       all_possible_display_names=kwargs['all_possible_display_names']
                                                       ),
                                  inputs=inputs_list + text_outputs,
                                  outputs=text_outputs,
                                  )

        def clear_instruct():
            return gr.Textbox(value='')

        def deselect_radio_chats():
            return gr.update(value=None)

        def clear_all():
            return gr.Textbox(value=''), gr.Textbox(value=''), gr.update(value=None), \
                gr.Textbox(value=''), gr.Textbox(value='')

        if kwargs['model_states']:
            submits1 = submits2 = submits3 = []
            submits4 = []

            triggers = [instruction, submit, retry_btn]
            fun_source = [instruction.submit, submit.click, retry_btn.click]
            fun_name = ['instruction', 'submit', 'retry']
            user_args = [all_user_args, all_user_args, all_retry_user_args]
            bot_args = [all_bot_args, all_bot_args, all_retry_bot_args]
            for userargs1, botarg1, funn1, funs1, trigger1, in zip(user_args, bot_args, fun_name, fun_source, triggers):
                submit_event11 = funs1(fn=user_state_setup,
                                       inputs=[my_db_state, requests_state, guest_name, trigger1, trigger1],
                                       outputs=[my_db_state, requests_state, trigger1],
                                       queue=queue)
                submit_event1a = submit_event11.then(**userargs1, queue=queue,
                                                     api_name='%s' % funn1 if allow_api else False)
                # if hit enter on new instruction for submitting new query, no longer the saved chat
                submit_event1b = submit_event1a.then(clear_all, inputs=None,
                                                     outputs=[instruction, iinput, radio_chats, score_text,
                                                              score_text2],
                                                     queue=queue)
                submit_event1c = submit_event1b.then(**botarg1,
                                                     api_name='%s_bot' % funn1 if allow_api else False,
                                                     queue=queue)
                submit_event1d = submit_event1c.then(**all_score_args,
                                                     api_name='%s_bot_score' % funn1 if allow_api else False,
                                                     queue=queue)
                submit_event1d.then(**save_auth_kwargs)

                submits1.extend([submit_event1a, submit_event1b, submit_event1c, submit_event1d])

            # if undo, no longer the saved chat
            submit_event4 = undo.click(fn=user_state_setup,
                                       inputs=[my_db_state, requests_state, guest_name, undo, undo],
                                       outputs=[my_db_state, requests_state, undo],
                                       queue=queue) \
                .then(**all_undo_user_args, api_name='undo' if allow_api else False) \
                .then(clear_all, inputs=None, outputs=[instruction, iinput, radio_chats, score_text,
                                                       score_text2], queue=queue) \
                .then(**all_score_args, api_name='undo_score' if allow_api else False) \
                .then(**save_auth_kwargs)
            submits4 = [submit_event4]

        else:
            # in case 2nd model, consume instruction first, so can clear quickly
            # bot doesn't consume instruction itself, just history from user, so why works
            submit_event11 = instruction.submit(fn=user_state_setup,
                                                inputs=[my_db_state, requests_state, guest_name, instruction,
                                                        instruction],
                                                outputs=[my_db_state, requests_state, instruction],
                                                queue=queue)
            submit_event1a = submit_event11.then(**user_args, queue=queue,
                                                 api_name='instruction' if allow_api else False)
            # if hit enter on new instruction for submitting new query, no longer the saved chat
            submit_event1a2 = submit_event1a.then(deselect_radio_chats, inputs=None, outputs=radio_chats, queue=queue)
            submit_event1b = submit_event1a2.then(**user_args2, api_name='instruction2' if allow_api else False)
            submit_event1c = submit_event1b.then(clear_instruct, None, instruction) \
                .then(clear_instruct, None, iinput)
            submit_event1d = submit_event1c.then(**bot_args, api_name='instruction_bot' if allow_api else False,
                                                 queue=queue)
            submit_event1e = submit_event1d.then(**score_args,
                                                 api_name='instruction_bot_score' if allow_api else False,
                                                 queue=queue)
            submit_event1f = submit_event1e.then(**bot_args2, api_name='instruction_bot2' if allow_api else False,
                                                 queue=queue)
            submit_event1g = submit_event1f.then(**score_args2,
                                                 api_name='instruction_bot_score2' if allow_api else False, queue=queue)
            submit_event1g.then(**save_auth_kwargs)

            submits1 = [submit_event1a, submit_event1a2, submit_event1b, submit_event1c, submit_event1d,
                        submit_event1e,
                        submit_event1f, submit_event1g]

            submit_event21 = submit.click(fn=user_state_setup,
                                          inputs=[my_db_state, requests_state, guest_name, submit, submit],
                                          outputs=[my_db_state, requests_state, submit],
                                          queue=queue)
            submit_event2a = submit_event21.then(**user_args, api_name='submit' if allow_api else False)
            # if submit new query, no longer the saved chat
            submit_event2a2 = submit_event2a.then(deselect_radio_chats, inputs=None, outputs=radio_chats, queue=queue)
            submit_event2b = submit_event2a2.then(**user_args2, api_name='submit2' if allow_api else False)
            submit_event2c = submit_event2b.then(clear_all, inputs=None,
                                                 outputs=[instruction, iinput, radio_chats, score_text, score_text2],
                                                 queue=queue)
            submit_event2d = submit_event2c.then(**bot_args, api_name='submit_bot' if allow_api else False, queue=queue)
            submit_event2e = submit_event2d.then(**score_args,
                                                 api_name='submit_bot_score' if allow_api else False,
                                                 queue=queue)
            submit_event2f = submit_event2e.then(**bot_args2, api_name='submit_bot2' if allow_api else False,
                                                 queue=queue)
            submit_event2g = submit_event2f.then(**score_args2,
                                                 api_name='submit_bot_score2' if allow_api else False,
                                                 queue=queue)
            submit_event2g.then(**save_auth_kwargs)

            submits2 = [submit_event2a, submit_event2a2, submit_event2b, submit_event2c, submit_event2d,
                        submit_event2e,
                        submit_event2f, submit_event2g]

            submit_event31 = retry_btn.click(fn=user_state_setup,
                                             inputs=[my_db_state, requests_state, guest_name, retry_btn, retry_btn],
                                             outputs=[my_db_state, requests_state, retry_btn],
                                             queue=queue)
            submit_event3a = submit_event31.then(**retry_user_args,
                                                 api_name='retry' if allow_api else False)
            # if retry, no longer the saved chat
            submit_event3a2 = submit_event3a.then(deselect_radio_chats, inputs=None, outputs=radio_chats, queue=queue)
            submit_event3b = submit_event3a2.then(**retry_user_args2, api_name='retry2' if allow_api else False)
            submit_event3c = submit_event3b.then(clear_instruct, None, instruction) \
                .then(clear_instruct, None, iinput)
            submit_event3d = submit_event3c.then(**retry_bot_args, api_name='retry_bot' if allow_api else False,
                                                 queue=queue)
            submit_event3e = submit_event3d.then(**score_args,
                                                 api_name='retry_bot_score' if allow_api else False,
                                                 queue=queue)
            submit_event3f = submit_event3e.then(**retry_bot_args2, api_name='retry_bot2' if allow_api else False,
                                                 queue=queue)
            submit_event3g = submit_event3f.then(**score_args2,
                                                 api_name='retry_bot_score2' if allow_api else False,
                                                 queue=queue)
            submit_event3g.then(**save_auth_kwargs)

            submits3 = [submit_event3a, submit_event3a2, submit_event3b, submit_event3c, submit_event3d,
                        submit_event3e,
                        submit_event3f, submit_event3g]

            # if undo, no longer the saved chat
            submit_event4 = undo.click(fn=user_state_setup,
                                       inputs=[my_db_state, requests_state, guest_name, undo, undo],
                                       outputs=[my_db_state, requests_state, undo],
                                       queue=queue) \
                .then(**undo_user_args, api_name='undo' if allow_api else False) \
                .then(**undo_user_args2, api_name='undo2' if allow_api else False) \
                .then(clear_all, inputs=None, outputs=[instruction, iinput, radio_chats, score_text,
                                                       score_text2], queue=queue) \
                .then(**score_args, api_name='undo_score' if allow_api else False) \
                .then(**score_args2, api_name='undo_score2' if allow_api else False) \
                .then(**save_auth_kwargs)
            submits4 = [submit_event4]

        # MANAGE CHATS
        def dedup(short_chat, short_chats):
            if short_chat not in short_chats:
                return short_chat
            for i in range(1, 1000):
                short_chat_try = short_chat + "_" + str(i)
                if short_chat_try not in short_chats:
                    return short_chat_try
            # fallback and hope for best
            short_chat = short_chat + "_" + str(random.random())
            return short_chat

        def get_short_chat(x, short_chats, short_len=20, words=4):
            if x and len(x[0]) == 2 and x[0][0] is not None:
                short_chat = ' '.join(x[0][0][:short_len].split(' ')[:words]).strip()
                if not short_chat:
                    # e.g.summarization, try using answer
                    short_chat = ' '.join(x[0][1][:short_len].split(' ')[:words]).strip()
                    if not short_chat:
                        short_chat = 'Unk'
                short_chat = dedup(short_chat, short_chats)
            else:
                short_chat = None
            return short_chat

        def is_chat_same(x, y):
            # <p> etc. added in chat, try to remove some of that to help avoid dup entries when hit new conversation
            is_same = True
            # length of conversation has to be same
            if len(x) != len(y):
                return False
            if len(x) != len(y):
                return False
            for stepx, stepy in zip(x, y):
                if len(stepx) != len(stepy):
                    # something off with a conversation
                    return False
                for stepxx, stepyy in zip(stepx, stepy):
                    if len(stepxx) != len(stepyy):
                        # something off with a conversation
                        return False
                    if len(stepxx) != 2:
                        # something off
                        return False
                    if len(stepyy) != 2:
                        # something off
                        return False
                    questionx = str(stepxx[0]).replace('<p>', '').replace('</p>', '') if stepxx[0] is not None else None
                    answerx = str(stepxx[1]).replace('<p>', '').replace('</p>', '') if stepxx[1] is not None else None

                    questiony = str(stepyy[0]).replace('<p>', '').replace('</p>', '') if stepyy[0] is not None else None
                    answery = str(stepyy[1]).replace('<p>', '').replace('</p>', '') if stepyy[1] is not None else None

                    if questionx != questiony or answerx != answery:
                        return False
            return is_same

        def save_chat(*args, chat_is_list=False, auth_filename=None, auth_freeze=None, raise_if_none=True):
            args_list = list(args)
            db1s = args_list[0]
            requests_state1 = args_list[1]
            args_list = args_list[2:]
            if not chat_is_list:
                # list of chatbot histories,
                # can't pass in list with list of chatbot histories and state due to gradio limits
                chat_list = args_list[:-1]
            else:
                assert len(args_list) == 2
                chat_list = args_list[0]
            # if old chat file with single chatbot, get into shape
            if isinstance(chat_list, list) and len(chat_list) > 0 and isinstance(chat_list[0], list) and len(
                    chat_list[0]) == 2 and isinstance(chat_list[0][0], str) and isinstance(chat_list[0][1], str):
                chat_list = [chat_list]
            # remove None histories
            chat_list_not_none = [x for x in chat_list if x and len(x) > 0 and len(x[0]) == 2 and x[0][1] is not None]
            chat_list_none = [x for x in chat_list if x not in chat_list_not_none]
            if len(chat_list_none) > 0 and len(chat_list_not_none) == 0:
                if raise_if_none:
                    raise ValueError("Invalid chat file")
                else:
                    chat_state1 = args_list[-1]
                    choices = list(chat_state1.keys()).copy()
                    return chat_state1, gr.update(choices=choices, value=None)
            # dict with keys of short chat names, values of list of list of chatbot histories
            chat_state1 = args_list[-1]
            short_chats = list(chat_state1.keys())
            if len(chat_list_not_none) > 0:
                # make short_chat key from only first history, based upon question that is same anyways
                chat_first = chat_list_not_none[0]
                short_chat = get_short_chat(chat_first, short_chats)
                if short_chat:
                    old_chat_lists = list(chat_state1.values())
                    already_exists = any([is_chat_same(chat_list, x) for x in old_chat_lists])
                    if not already_exists:
                        chat_state1[short_chat] = chat_list.copy()

            # reverse so newest at top
            choices = list(chat_state1.keys()).copy()
            choices.reverse()

            # save saved chats and chatbots to auth file
            selection_docs_state1 = None
            langchain_mode2 = None
            roles_state1 = None
            model_options_state1 = None
            lora_options_state1 = None
            server_options_state1 = None
            text_output1 = chat_list[0]
            text_output21 = chat_list[1]
            text_outputs1 = chat_list[2:]
            h2ogpt_key2, visible_models2 = None, None
            save_auth_func(selection_docs_state1, requests_state1, roles_state1,
                           model_options_state1, lora_options_state1, server_options_state1,
                           chat_state1, langchain_mode2,
                           h2ogpt_key2, visible_models2,
                           None, None, None, None,
                           None, None, None, None,
                           None, None, None, None,
                           None, None,
                           text_output1, text_output21, text_outputs1,
                           )

            return chat_state1, gr.update(choices=choices, value=None)

        def switch_chat(chat_key, chat_state1, num_model_lock=0):
            chosen_chat = chat_state1[chat_key]
            # deal with possible different size of chat list vs. current list
            ret_chat = [None] * (2 + num_model_lock)
            for chati in range(0, 2 + num_model_lock):
                ret_chat[chati % len(ret_chat)] = chosen_chat[chati % len(chosen_chat)]
            return tuple(ret_chat)

        def clear_texts(*args):
            return tuple([[]] * len(args))

        def clear_scores():
            return gr.Textbox(value=res_value), \
                gr.Textbox(value='Response Score: NA'), \
                gr.Textbox(value='Response Score: NA')

        switch_chat_fun = functools.partial(switch_chat, num_model_lock=len(text_outputs))
        radio_chats.input(switch_chat_fun,
                          inputs=[radio_chats, chat_state],
                          outputs=[text_output, text_output2] + text_outputs) \
            .then(clear_scores, outputs=[score_text, score_text2, score_text_nochat])

        def remove_chat(chat_key, chat_state1):
            if isinstance(chat_key, str):
                chat_state1.pop(chat_key, None)
            return gr.update(choices=list(chat_state1.keys()), value=None), chat_state1

        remove_chat_event = remove_chat_btn.click(remove_chat,
                                                  inputs=[radio_chats, chat_state],
                                                  outputs=[radio_chats, chat_state],
                                                  **noqueue_kwargs, api_name='remove_chat')

        def get_chats1(chat_state1):
            base = 'chats'
            base = makedirs(base, exist_ok=True, tmp_ok=True, use_base=True)
            filename = os.path.join(base, 'chats_%s.json' % str(uuid.uuid4()))
            with open(filename, "wt") as f:
                f.write(json.dumps(chat_state1, indent=2))
            return filename

        export_chat_event = export_chats_btn.click(get_chats1, inputs=chat_state, outputs=chats_file,
                                                   **noqueue_kwargs2,
                                                   api_name='export_chats' if allow_api else False)

        def add_chats_from_file(db1s, requests_state1, file, chat_state1, radio_chats1, chat_exception_text1,
                                auth_filename=None, auth_freeze=None):
            if not file:
                return None, chat_state1, gr.update(choices=list(chat_state1.keys()), value=None), chat_exception_text1
            if isinstance(file, str):
                files = [file]
            else:
                files = file
            if not files:
                return None, chat_state1, gr.update(choices=list(chat_state1.keys()), value=None), chat_exception_text1
            chat_exception_list = []
            for file1 in files:
                try:
                    if hasattr(file1, 'name'):
                        file1 = file1.name
                    with open(file1, "rt") as f:
                        new_chats = json.loads(f.read())
                        for chat1_k, chat1_v in new_chats.items():
                            # ignore chat1_k, regenerate and de-dup to avoid loss
                            chat_state1, _ = save_chat(db1s, requests_state1, chat1_v, chat_state1, chat_is_list=True,
                                                       raise_if_none=True)
                except BaseException as e:
                    t, v, tb = sys.exc_info()
                    ex = ''.join(traceback.format_exception(t, v, tb))
                    ex_str = "File %s exception: %s" % (file1, str(e))
                    print(ex_str, flush=True)
                    chat_exception_list.append(ex_str)
                    chat_exception_text1 = '\n'.join(chat_exception_list)
            # save chat to auth file
            selection_docs_state1 = None
            langchain_mode2 = None
            roles_state1 = None
            model_options_state1 = None
            lora_options_state1 = None
            server_options_state1 = None
            text_output1, text_output21, text_outputs1 = None, None, None
            h2ogpt_key2, visible_models2 = None, None
            save_auth_func(selection_docs_state1, requests_state1, roles_state1,
                           model_options_state1, lora_options_state1, server_options_state1,
                           chat_state1, langchain_mode2,
                           h2ogpt_key2, visible_models2,
                           None, None, None, None,
                           None, None, None, None,
                           None, None, None, None,
                           None, None,
                           text_output1, text_output21, text_outputs1,
                           )
            return None, chat_state1, gr.update(choices=list(chat_state1.keys()), value=None), chat_exception_text1

        # note for update_user_db_func output is ignored for db
        chatup_change_eventa = chatsup_output.change(user_state_setup,
                                                     inputs=[my_db_state, requests_state, guest_name, langchain_mode],
                                                     outputs=[my_db_state, requests_state, langchain_mode],
                                                     show_progress='minimal')
        add_chats_from_file_func = functools.partial(add_chats_from_file,
                                                     auth_filename=kwargs['auth_filename'],
                                                     auth_freeze=kwargs['auth_freeze'],
                                                     )
        chatup_change_event = chatup_change_eventa.then(add_chats_from_file_func,
                                                        inputs=[my_db_state, requests_state] +
                                                               [chatsup_output, chat_state, radio_chats,
                                                                chat_exception_text],
                                                        outputs=[chatsup_output, chat_state, radio_chats,
                                                                 chat_exception_text],
                                                        **noqueue_kwargs,
                                                        api_name='add_to_chats' if allow_api else False)

        clear_chat_event = clear_chat_btn.click(fn=clear_texts,
                                                inputs=[text_output, text_output2] + text_outputs,
                                                outputs=[text_output, text_output2] + text_outputs,
                                                **noqueue_kwargs, api_name='clear' if allow_api else False) \
            .then(deselect_radio_chats, inputs=None, outputs=radio_chats, **noqueue_kwargs) \
            .then(clear_scores, outputs=[score_text, score_text2, score_text_nochat])

        clear_eventa = save_chat_btn.click(user_state_setup,
                                           inputs=[my_db_state, requests_state, guest_name, langchain_mode],
                                           outputs=[my_db_state, requests_state, langchain_mode],
                                           show_progress='minimal', **noqueue_kwargs2)
        save_chat_func = functools.partial(save_chat,
                                           auth_filename=kwargs['auth_filename'],
                                           auth_freeze=kwargs['auth_freeze'],
                                           raise_if_none=False,
                                           )
        clear_event = clear_eventa.then(save_chat_func,
                                        inputs=[my_db_state, requests_state] +
                                               [text_output, text_output2] + text_outputs +
                                               [chat_state],
                                        outputs=[chat_state, radio_chats],
                                        api_name='save_chat' if allow_api else False)
        if kwargs['score_model']:
            clear_event2 = clear_event.then(clear_scores, outputs=[score_text, score_text2, score_text_nochat])

        # NOTE: clear of instruction/iinput for nochat has to come after score,
        # because score for nochat consumes actual textbox, while chat consumes chat history filled by user()
        no_chat_args = dict(fn=fun,
                            inputs=[model_state, my_db_state, selection_docs_state, requests_state,
                                    roles_state] + inputs_list,
                            outputs=text_output_nochat,
                            queue=queue,
                            )
        submit_event_nochat = submit_nochat.click(**no_chat_args, api_name='submit_nochat' if allow_api else False) \
            .then(**score_args_nochat, api_name='instruction_bot_score_nochat' if allow_api else False, queue=queue) \
            .then(clear_instruct, None, instruction_nochat) \
            .then(clear_instruct, None, iinput_nochat)
        # copy of above with text box submission
        submit_event_nochat2 = instruction_nochat.submit(**no_chat_args) \
            .then(**score_args_nochat, queue=queue) \
            .then(clear_instruct, None, instruction_nochat) \
            .then(clear_instruct, None, iinput_nochat)

        submit_event_nochat_api = submit_nochat_api.click(fun_with_dict_str,
                                                          inputs=[model_state, my_db_state, selection_docs_state,
                                                                  requests_state, roles_state,
                                                                  inputs_dict_str],
                                                          outputs=text_output_nochat_api,
                                                          queue=True,  # required for generator
                                                          api_name='submit_nochat_api' if allow_api else False)

        submit_event_nochat_api_plain = submit_nochat_api_plain.click(fun_with_dict_str_plain,
                                                                      inputs=inputs_dict_str,
                                                                      outputs=text_output_nochat_api,
                                                                      **noqueue_kwargs_curl,
                                                                      api_name='submit_nochat_plain_api' if allow_api else False)

        submit_event_verifier = submit_verifier.click(fun_with_dict_verifier,
                                                      inputs=verifier_inputs_dict_str,
                                                      outputs=text_output_verifier,
                                                      **noqueue_kwargs,
                                                      api_name='submit_verifier' if allow_api else False)

        def load_model(model_name, lora_weights, server_name,
                       model_state_old,
                       prompt_type_old,
                       load_8bit, load_4bit, low_bit_mode,
                       load_gptq, load_awq, load_exllama, use_safetensors, revision,
                       use_cpu,
                       use_gpu_id, gpu_id,
                       max_seq_len1, rope_scaling1,
                       model_path_llama1, model_name_gptj1, model_name_gpt4all_llama1,
                       n_gpu_layers1, n_batch1, n_gqa1, llamacpp_dict_more1,
                       system_prompt1,
                       exllama_dict, gptq_dict, attention_sinks, sink_dict, truncation_generation, hf_model_dict,
                       force_seq2seq_type, force_t5_type,
                       model_options_state1, lora_options_state1, server_options_state1,
                       unload=False):
            if unload:
                model_name = no_model_str
                lora_weights = no_lora_str
                server_name = no_server_str
            exllama_dict = str_to_dict(exllama_dict)
            gptq_dict = str_to_dict(gptq_dict)
            sink_dict = str_to_dict(sink_dict)
            hf_model_dict = str_to_dict(hf_model_dict)

            # switch-a-roo on base_model so can pass GGUF/GGML as base model
            model_name0 = model_name
            model_name, model_path_llama1, load_gptq, load_awq, n_gqa1 = \
                switch_a_roo_llama(model_name, model_path_llama1, load_gptq, load_awq, n_gqa1,
                                   kwargs['llamacpp_path'])

            # after getting results, we always keep all items related to llama.cpp, gptj, gpt4all inside llamacpp_dict
            llamacpp_dict = str_to_dict(llamacpp_dict_more1)
            llamacpp_dict.update(dict(model_path_llama=model_path_llama1,
                                      model_name_gptj=model_name_gptj1,
                                      model_name_gpt4all_llama=model_name_gpt4all_llama1,
                                      n_gpu_layers=n_gpu_layers1,
                                      n_batch=n_batch1,
                                      n_gqa=n_gqa1,
                                      ))
            if model_name == 'llama' and not model_path_llama1:
                raise ValueError("Must set model_path_llama if model_name==llama")
            if model_name == 'gptj' and not model_name_gptj:
                raise ValueError("Must set model_name_gptj if model_name==llama")
            if model_name == 'gpt4all_llama' and not model_name_gpt4all_llama:
                raise ValueError("Must set model_name_gpt4all_llama if model_name==llama")

            # ensure no API calls reach here
            if is_public:
                raise RuntimeError("Illegal access for %s" % model_name)
            # ensure old model removed from GPU memory
            if kwargs['debug']:
                print("Pre-switch pre-del GPU memory: %s" % get_torch_allocated(), flush=True)

            model0 = model_state0['model']
            if isinstance(model_state_old['model'], str) and \
                    model0 is not None and \
                    hasattr(model0, 'cpu'):
                # best can do, move model loaded at first to CPU
                model0.cpu()

            if model_state_old['model'] is not None and \
                    not isinstance(model_state_old['model'], str):
                if hasattr(model_state_old['model'], 'cpu'):
                    try:
                        model_state_old['model'].cpu()
                    except Exception as e:
                        # sometimes hit NotImplementedError: Cannot copy out of meta tensor; no data!
                        print("Unable to put model on CPU: %s" % str(e), flush=True)
                del model_state_old['model']
                model_state_old['model'] = None

            if model_state_old['tokenizer'] is not None and not isinstance(model_state_old['tokenizer'], str):
                del model_state_old['tokenizer']
                model_state_old['tokenizer'] = None

            clear_torch_cache(allow_skip=False)
            if kwargs['debug']:
                print("Pre-switch post-del GPU memory: %s" % get_torch_allocated(), flush=True)
            if not model_name:
                model_name = no_model_str
            if model_name == no_model_str:
                # no-op if no model, just free memory
                # no detranscribe needed for model, never go into evaluate
                lora_weights = no_lora_str
                server_name = no_server_str
                prompt_type_old = ''
                model_path_llama1 = ''
                model_name_gptj1 = ''
                model_name_gpt4all_llama1 = ''
                load_gptq = ''
                load_awq = ''
                return kwargs['model_state_none'].copy(), \
                    model_name, lora_weights, server_name, \
                    prompt_type_old, max_seq_len1, \
                    gr.Slider(maximum=256), \
                    gr.Slider(maximum=256), \
                    model_path_llama1, model_name_gptj1, model_name_gpt4all_llama1, \
                    load_gptq, load_awq, n_gqa1, \
                    n_batch1, n_gpu_layers1, llamacpp_dict_more1, \
                    model_options_state1, lora_options_state1, server_options_state1

            # don't deepcopy, can contain model itself
            all_kwargs1 = all_kwargs.copy()
            all_kwargs1['base_model'] = model_name.strip()
            all_kwargs1['load_8bit'] = load_8bit
            all_kwargs1['load_4bit'] = load_4bit
            all_kwargs1['low_bit_mode'] = low_bit_mode
            all_kwargs1['load_gptq'] = load_gptq
            all_kwargs1['load_awq'] = load_awq
            all_kwargs1['load_exllama'] = load_exllama
            all_kwargs1['use_safetensors'] = use_safetensors
            all_kwargs1['revision'] = None if not revision else revision  # transcribe, don't pass ''
            all_kwargs1['use_gpu_id'] = use_gpu_id
            all_kwargs1['gpu_id'] = int(gpu_id) if gpu_id not in [None, 'None'] else None  # detranscribe
            all_kwargs1['llamacpp_dict'] = llamacpp_dict
            all_kwargs1['exllama_dict'] = exllama_dict
            all_kwargs1['gptq_dict'] = gptq_dict
            all_kwargs1['attention_sinks'] = attention_sinks
            all_kwargs1['sink_dict'] = sink_dict
            all_kwargs1['truncation_generation'] = truncation_generation
            all_kwargs1['hf_model_dict'] = hf_model_dict
            all_kwargs1['force_seq2seq_type'] = force_seq2seq_type
            all_kwargs1['force_t5_type'] = force_t5_type
            # reasonable default for easy UI/UX even if not optimal
            if 'llama2' in model_name and max_seq_len1 in [-1, None]:
                max_seq_len1 = 4096
            elif 'llama3' in model_name and max_seq_len1 in [-1, None]:
                max_seq_len1 = 8192
            elif 'mistral' in model_name and max_seq_len1 in [-1, None]:
                max_seq_len1 = 4096
            else:
                max_seq_len1 = 4096
            all_kwargs1['max_seq_len'] = int(max_seq_len1) if max_seq_len1 is not None and max_seq_len1 > 0 else None
            try:
                all_kwargs1['rope_scaling'] = str_to_dict(rope_scaling1)  # transcribe
            except:
                print("Failed to use user input for rope_scaling dict", flush=True)
                all_kwargs1['rope_scaling'] = {}
            if use_cpu:
                all_kwargs1['n_gpus'] = 0
            elif use_gpu_id and all_kwargs1['gpu_id']:
                all_kwargs1['n_gpus'] = 1
            else:
                all_kwargs1['n_gpus'] = n_gpus_global
            prompt_type1 = model_name_to_prompt_type(model_name,
                                                     server_name,
                                                     model_name0=model_name0,
                                                     llamacpp_dict=llamacpp_dict,
                                                     prompt_type_old=prompt_type_old)

            # detranscribe
            if lora_weights == no_lora_str:
                lora_weights = ''
            all_kwargs1['lora_weights'] = lora_weights.strip()
            if server_name == no_server_str:
                server_name = ''
            all_kwargs1['inference_server'] = server_name.strip()

            gradio_model_kwargs = dict(reward_type=False,
                                       **get_kwargs(get_model, exclude_names=['reward_type'],
                                                    **all_kwargs1))
            model1, tokenizer1, device1 = get_model_retry(**gradio_model_kwargs)
            clear_torch_cache()

            tokenizer_base_model = model_name
            prompt_dict1, error0 = get_prompt(prompt_type1, '',
                                              context='', reduced=False, making_context=False,
                                              return_dict=True, system_prompt=system_prompt1)
            model_state_new = dict(model=model1, tokenizer=tokenizer1, device=device1,
                                   base_model=model_name,
                                   display_name=model_name,
                                   tokenizer_base_model=tokenizer_base_model,
                                   lora_weights=lora_weights, inference_server=server_name,
                                   prompt_type=prompt_type1, prompt_dict=prompt_dict1,
                                   # FIXME: not typically required, unless want to expose adding h2ogpt endpoint in UI
                                   visible_models=None, h2ogpt_key=None,
                                   )
            [model_state_new.update({k: v}) for k, v in kwargs['model_state_none'].items() if k not in model_state_new]
            max_seq_len1new = get_model_max_length_from_tokenizer(tokenizer1)

            max_max_new_tokens1 = get_max_max_new_tokens(model_state_new, **kwargs)

            # FIXME: Ensure stored in login state
            if model_options_state1 and model_name0 not in model_options_state1[0]:
                model_options_state1[0].extend([model_name0])
            if lora_options_state1 and lora_weights not in lora_options_state1[0]:
                lora_options_state1[0].extend([lora_weights])
            if server_options_state1 and server_name not in server_options_state1[0]:
                server_options_state1[0].extend([server_name])

            if kwargs['debug']:
                print("Post-switch GPU memory: %s" % get_torch_allocated(), flush=True)
            return model_state_new, model_name, lora_weights, server_name, \
                prompt_type1, max_seq_len1new, \
                gr.Slider(maximum=max_max_new_tokens1), \
                gr.Slider(maximum=max_max_new_tokens1), \
                model_path_llama1, model_name_gptj1, model_name_gpt4all_llama1, \
                load_gptq, load_awq, n_gqa1, \
                n_batch1, n_gpu_layers1, llamacpp_dict_more1, \
                model_options_state1, lora_options_state1, server_options_state1

        def get_prompt_str(prompt_type1, prompt_dict1, system_prompt1, which=0):
            if prompt_type1 in ['', None]:
                print("Got prompt_type %s: %s" % (which, prompt_type1), flush=True)
                return str({})
            prompt_dict1, prompt_dict_error = get_prompt(prompt_type1, prompt_dict1, context='',
                                                         reduced=False, making_context=False, return_dict=True,
                                                         system_prompt=system_prompt1)
            if prompt_dict_error:
                return str(prompt_dict_error)
            else:
                # return so user can manipulate if want and use as custom
                return str(prompt_dict1)

        get_prompt_str_func1 = functools.partial(get_prompt_str, which=1)
        get_prompt_str_func2 = functools.partial(get_prompt_str, which=2)
        prompt_type.change(fn=get_prompt_str_func1, inputs=[prompt_type, prompt_dict, system_prompt],
                           outputs=prompt_dict, **noqueue_kwargs)
        prompt_type2.change(fn=get_prompt_str_func2, inputs=[prompt_type2, prompt_dict2, system_prompt],
                            outputs=prompt_dict2,
                            **noqueue_kwargs)

        def dropdown_prompt_type_list(x):
            return gr.Dropdown(value=x)

        def chatbot_list(x, model_used_in, model_path_llama_in, inference_server_in, prompt_type_in,
                         model_label_prefix_in=''):
            chat_name = get_chatbot_name(model_used_in, model_used_in, model_path_llama_in, inference_server_in,
                                         prompt_type_in,
                                         model_label_prefix=model_label_prefix_in)
            return gr.Textbox(label=chat_name)

        load_model_inputs = [model_choice, lora_choice, server_choice, model_state, prompt_type,
                             model_load8bit_checkbox, model_load4bit_checkbox, model_low_bit_mode,
                             model_load_gptq, model_load_awq, model_load_exllama_checkbox,
                             model_safetensors_checkbox, model_revision,
                             model_use_cpu_checkbox,
                             model_use_gpu_id_checkbox, model_gpu,
                             max_seq_len, rope_scaling,
                             model_path_llama, model_name_gptj, model_name_gpt4all_llama,
                             n_gpu_layers, n_batch, n_gqa, llamacpp_dict_more,
                             system_prompt,
                             model_exllama_dict, model_gptq_dict,
                             model_attention_sinks, model_sink_dict,
                             model_truncation_generation,
                             model_hf_model_dict,
                             model_force_seq2seq_type,
                             model_force_force_t5_type,
                             model_options_state, lora_options_state, server_options_state,
                             ]
        load_model_outputs = [model_state, model_used, lora_used, server_used,
                              # if prompt_type changes, prompt_dict will change via change rule
                              prompt_type, max_seq_len_used,
                              max_new_tokens, min_new_tokens,
                              model_path_llama, model_name_gptj, model_name_gpt4all_llama,
                              model_load_gptq, model_load_awq, n_gqa,
                              n_batch, n_gpu_layers, llamacpp_dict_more,
                              model_options_state, lora_options_state, server_options_state,
                              ]
        load_model_args = dict(fn=load_model,
                               inputs=load_model_inputs, outputs=load_model_outputs)
        unload_model_args = dict(fn=functools.partial(load_model, unload=True),
                                 inputs=load_model_inputs, outputs=load_model_outputs)
        prompt_update_args = dict(fn=dropdown_prompt_type_list, inputs=prompt_type, outputs=prompt_type)
        chatbot_update_args = dict(
            fn=functools.partial(chatbot_list, model_label_prefix_in=kwargs['model_label_prefix']),
            inputs=[text_output, model_used, model_path_llama, server_used, prompt_type],
            outputs=text_output)
        nochat_update_args = dict(
            fn=functools.partial(chatbot_list, model_label_prefix_in=kwargs['model_label_prefix']),
            inputs=[text_output_nochat, model_used, model_path_llama, server_used, prompt_type],
            outputs=text_output_nochat)
        load_model_event = load_model_button.click(**load_model_args,
                                                   api_name='load_model' if allow_api and not is_public else False) \
            .then(**prompt_update_args) \
            .then(**chatbot_update_args) \
            .then(**nochat_update_args) \
            .then(clear_torch_cache) \
            .then(**save_auth_kwargs)

        unload_model_event = unload_model_button.click(**unload_model_args,
                                                       api_name='unload_model' if allow_api and not is_public else False) \
            .then(**prompt_update_args) \
            .then(**chatbot_update_args) \
            .then(**nochat_update_args) \
            .then(clear_torch_cache)

        load_model_inputs2 = [model_choice2, lora_choice2, server_choice2, model_state2, prompt_type2,
                              model_load8bit_checkbox2, model_load4bit_checkbox2, model_low_bit_mode2,
                              model_load_gptq2, model_load_awq2, model_load_exllama_checkbox2,
                              model_safetensors_checkbox2, model_revision2,
                              model_use_cpu_checkbox2,
                              model_use_gpu_id_checkbox2, model_gpu2,
                              max_seq_len2, rope_scaling2,
                              model_path_llama2, model_name_gptj2, model_name_gpt4all_llama2,
                              n_gpu_layers2, n_batch2, n_gqa2, llamacpp_dict_more2,
                              system_prompt,
                              model_exllama_dict2, model_gptq_dict2,
                              model_attention_sinks2, model_sink_dict2,
                              model_truncation_generation2,
                              model_hf_model_dict2,
                              model_force_seq2seq_type2,
                              model_force_force_t5_type2,
                              model_options_state, lora_options_state, server_options_state,
                              ]
        load_model_outputs2 = [model_state2, model_used2, lora_used2, server_used2,
                               # if prompt_type2 changes, prompt_dict2 will change via change rule
                               prompt_type2, max_seq_len_used2,
                               max_new_tokens2, min_new_tokens2,
                               model_path_llama2, model_name_gptj2, model_name_gpt4all_llama2,
                               model_load_gptq2, model_load_awq2, n_gqa2,
                               n_batch2, n_gpu_layers2, llamacpp_dict_more2,
                               model_options_state, lora_options_state, server_options_state,
                               ]
        load_model_args2 = dict(fn=load_model,
                                inputs=load_model_inputs2, outputs=load_model_outputs2)
        unload_model_args2 = dict(fn=functools.partial(load_model, unload=True),
                                  inputs=load_model_inputs2, outputs=load_model_outputs2)
        prompt_update_args2 = dict(fn=dropdown_prompt_type_list, inputs=prompt_type2, outputs=prompt_type2)
        chatbot_update_args2 = dict(
            fn=functools.partial(chatbot_list, model_label_prefix_in=kwargs['model_label_prefix']),
            inputs=[text_output2, model_used2, model_path_llama2, server_used2, prompt_type2],
            outputs=text_output2)
        load_model_event2 = load_model_button2.click(**load_model_args2,
                                                     api_name='load_model2' if allow_api and not is_public else False) \
            .then(**prompt_update_args2) \
            .then(**chatbot_update_args2) \
            .then(clear_torch_cache) \
            .then(**save_auth_kwargs)

        unload_model_event2 = unload_model_button2.click(**unload_model_args2,
                                                         api_name='unload_model2' if allow_api and not is_public else False) \
            .then(**prompt_update_args) \
            .then(**chatbot_update_args) \
            .then(**nochat_update_args) \
            .then(clear_torch_cache)

        def dropdown_model_lora_server_list(model_list0, model_x,
                                            lora_list0, lora_x,
                                            server_list0, server_x,
                                            model_used1, lora_used1, server_used1,
                                            model_used2, lora_used2, server_used2,
                                            ):
            model_new_state = [model_list0[0] + [model_x]]
            model_new_options = [*model_new_state[0]]
            if no_model_str in model_new_options:
                model_new_options.remove(no_model_str)
            model_new_options = [no_model_str] + sorted(model_new_options)
            x1 = model_x if model_used1 == no_model_str else model_used1
            x2 = model_x if model_used2 == no_model_str else model_used2
            ret1 = [gr.Dropdown(value=x1, choices=model_new_options),
                    gr.Dropdown(value=x2, choices=model_new_options),
                    '', model_new_state]

            lora_new_state = [lora_list0[0] + [lora_x]]
            lora_new_options = [*lora_new_state[0]]
            if no_lora_str in lora_new_options:
                lora_new_options.remove(no_lora_str)
            lora_new_options = [no_lora_str] + sorted(lora_new_options)
            # don't switch drop-down to added lora if already have model loaded
            x1 = lora_x if model_used1 == no_model_str else lora_used1
            x2 = lora_x if model_used2 == no_model_str else lora_used2
            ret2 = [gr.Dropdown(value=x1, choices=lora_new_options),
                    gr.Dropdown(value=x2, choices=lora_new_options),
                    '', lora_new_state]

            server_new_state = [server_list0[0] + [server_x]]
            server_new_options = [*server_new_state[0]]
            if no_server_str in server_new_options:
                server_new_options.remove(no_server_str)
            server_new_options = [no_server_str] + sorted(server_new_options)
            # don't switch drop-down to added server if already have model loaded
            x1 = server_x if model_used1 == no_model_str else server_used1
            x2 = server_x if model_used2 == no_model_str else server_used2
            ret3 = [gr.Dropdown(value=x1, choices=server_new_options),
                    gr.Dropdown(value=x2, choices=server_new_options),
                    '', server_new_state]

            return tuple(ret1 + ret2 + ret3)

        add_model_lora_server_event = \
            add_model_lora_server_button.click(fn=dropdown_model_lora_server_list,
                                               inputs=[model_options_state, new_model] +
                                                      [lora_options_state, new_lora] +
                                                      [server_options_state, new_server] +
                                                      [model_used, lora_used, server_used] +
                                                      [model_used2, lora_used2, server_used2],
                                               outputs=[model_choice, model_choice2, new_model, model_options_state] +
                                                       [lora_choice, lora_choice2, new_lora, lora_options_state] +
                                                       [server_choice, server_choice2, new_server,
                                                        server_options_state],
                                               **noqueue_kwargs)

        def get_inf_models_gr(model_options_state1, model_choice1, server1):
            models_new = get_inf_models(server1, verbose=verbose)
            model_options_state1[0].extend(models_new)
            if no_model_str in model_options_state1[0]:
                model_options_state1[0].remove(no_model_str)
            model_options_state1[0] = [no_model_str] + sorted(set(model_options_state1[0]))
            if models_new:
                model_choice1 = models_new[0]  # pick new one
            return model_options_state1, gr.Dropdown(choices=model_options_state1[0], value=model_choice1)

        load_models_button.click(get_inf_models_gr, inputs=[model_options_state, model_choice, server_choice],
                                 outputs=[model_options_state, model_choice])
        load_models_button2.click(get_inf_models_gr, inputs=[model_options_state, model_choice2, server_choice2],
                                  outputs=[model_options_state, model_choice2])

        go_event = go_btn.click(lambda: gr.update(visible=False), None, go_btn, api_name="go" if allow_api else False,
                                **noqueue_kwargs) \
            .then(lambda: gr.update(visible=True), None, normal_block, **noqueue_kwargs) \
            .then(**load_model_args, **noqueue_kwargs).then(**prompt_update_args, **noqueue_kwargs)

        def compare_textbox_fun(x):
            return gr.Textbox(visible=x)

        def compare_column_fun(x):
            return gr.Column(visible=x)

        def compare_prompt_fun(x):
            return gr.Dropdown(visible=x)

        def slider_fun(x):
            return gr.Slider(visible=x)

        compare_checkbox.select(compare_textbox_fun, compare_checkbox, text_output2,
                                api_name="compare_checkbox" if allow_api else False) \
            .then(compare_column_fun, compare_checkbox, col_model2) \
            .then(compare_prompt_fun, compare_checkbox, prompt_type2) \
            .then(compare_textbox_fun, compare_checkbox, score_text2) \
            .then(slider_fun, compare_checkbox, max_new_tokens2) \
            .then(slider_fun, compare_checkbox, min_new_tokens2)
        # FIXME: add score_res2 in condition, but do better

        # callback for logging flagged input/output
        callback.setup(inputs_list + [text_output, text_output2] + text_outputs, "flagged_data_points")
        flag_btn.click(lambda *args: callback.flag(args), inputs_list + [text_output, text_output2] + text_outputs,
                       None,
                       preprocess=False,
                       api_name='flag' if allow_api else False, **noqueue_kwargs)
        flag_btn_nochat.click(lambda *args: callback.flag(args), inputs_list + [text_output_nochat], None,
                              preprocess=False,
                              api_name='flag_nochat' if allow_api else False, **noqueue_kwargs)

        def get_system_info():
            if is_public:
                time.sleep(10)  # delay to avoid spam since **noqueue_kwargs
            return gr.Textbox(value=system_info_print())

        system_event = system_btn.click(get_system_info, outputs=system_text,
                                        api_name='system_info' if kwargs['system_api_open'] else False,
                                        **noqueue_kwargs)

        def shutdown_func(admin_pass_textbox1, h2ogpt_pid):
            assert admin_pass_textbox1 == admin_pass or not admin_pass
            if kwargs['close_button']:
                import psutil
                parent = psutil.Process(h2ogpt_pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()

        api_name_shutdown = 'shutdown' if kwargs['shutdown_via_api'] and \
                                          allow_api and \
                                          not is_public and \
                                          kwargs['h2ogpt_pid'] is not None else False
        shutdown_event = close_btn.click(functools.partial(shutdown_func, h2ogpt_pid=kwargs['h2ogpt_pid']),
                                         inputs=[admin_pass_textbox], outputs=None,
                                         api_name=api_name_shutdown,
                                         **noqueue_kwargs)

        def get_system_info_dict(system_input1, **kwargs1):
            if system_input1 != os.getenv("ADMIN_PASS", ""):
                return json.dumps({})
            exclude_list = ['admin_pass', 'examples']
            sys_dict = {k: v for k, v in kwargs1.items() if
                        isinstance(v, (str, int, bool, float)) and k not in exclude_list}
            try:
                sys_dict.update(system_info())
            except Exception as e:
                # protection
                print("Exception: %s" % str(e), flush=True)
            return json.dumps(sys_dict)

        system_kwargs = all_kwargs.copy()
        system_kwargs.update(dict(command=str(' '.join(sys.argv))))
        get_system_info_dict_func = functools.partial(get_system_info_dict, **all_kwargs)

        system_dict_event = system_btn2.click(get_system_info_dict_func,
                                              inputs=system_input,
                                              outputs=system_text2,
                                              api_name='system_info_dict' if kwargs['system_api_open'] else False,
                                              **noqueue_kwargs,  # queue to avoid spam
                                              )

        def get_model_names():
            key_list = ['display_name', 'base_model', 'prompt_type', 'prompt_dict'] + list(
                kwargs['other_model_state_defaults'].keys())
            # don't want to expose backend inference server IP etc.
            # key_list += ['inference_server']
            if len(model_states) >= 1:
                local_model_states = model_states
            elif model_state0 is not None:
                local_model_states = [model_state0]
            else:
                local_model_states = []
            for model_state3 in local_model_states:
                base_model = model_state3.get('base_model', '')
                inference_server = model_state3.get('inference_server', '')
                model_state3['llm'] = True
                model_state3['rag'] = True
                model_state3['image'] = model_state3.get('is_vision_model', False)
                model_state3['actually_image'] = model_state3.get('is_actually_vision_model', False)
                model_state3['video'] = is_video_model(base_model) or model_state3['image']
                model_state3['actually_video'] = is_video_model(base_model)
                model_state3['json'] = model_state3.get('json', False)
                model_state3['auto_visible_vision_models'] = model_state3.get('auto_visible_vision_models', False)

            key_list.extend(['llm', 'rag', 'image', 'actually_image', 'video', 'actually_video', 'json', 'auto_visible_vision_models'])
            return [{k: x[k] for k in key_list if k in x} for x in local_model_states]

        models_list_event = system_btn4.click(get_model_names,
                                              outputs=system_text4,
                                              api_name='model_names' if allow_api else False,
                                              **noqueue_kwargs,
                                              )

        def count_chat_tokens(model_state1, chat1, prompt_type1, prompt_dict1,
                              system_prompt1, chat_conversation1,
                              memory_restriction_level1=0,
                              keep_sources_in_context1=False,
                              ):
            if model_state1 and not isinstance(model_state1['tokenizer'], str):
                tokenizer = model_state1['tokenizer']
            elif model_state0 and not isinstance(model_state0['tokenizer'], str):
                tokenizer = model_state0['tokenizer']
            else:
                tokenizer = None
            if tokenizer is not None:
                langchain_mode1 = 'LLM'
                add_chat_history_to_context1 = True
                # fake user message to mimic bot()
                chat1 = copy.deepcopy(chat1)
                chat1 = chat1 + [['user_message1', None]]
                model_max_length1 = tokenizer.model_max_length
                context1, chat1 = history_to_context(chat1,
                                                     langchain_mode=langchain_mode1,
                                                     add_chat_history_to_context=add_chat_history_to_context1,
                                                     prompt_type=prompt_type1,
                                                     prompt_dict=prompt_dict1,
                                                     model_max_length=model_max_length1,
                                                     memory_restriction_level=memory_restriction_level1,
                                                     keep_sources_in_context=keep_sources_in_context1,
                                                     system_prompt=system_prompt1,
                                                     chat_conversation=chat_conversation1,
                                                     hyde_level=None,
                                                     gradio_errors_to_chatbot=kwargs['gradio_errors_to_chatbot'])
                tokens = tokenizer(context1, return_tensors="pt")['input_ids']
                if len(tokens.shape) == 1:
                    return str(tokens.shape[0])
                elif len(tokens.shape) == 2:
                    return str(tokens.shape[1])
                else:
                    return "N/A"
            else:
                return "N/A"

        count_chat_tokens_func = functools.partial(count_chat_tokens,
                                                   memory_restriction_level1=memory_restriction_level,
                                                   keep_sources_in_context1=kwargs['keep_sources_in_context'])
        count_tokens_event = count_chat_tokens_btn.click(fn=count_chat_tokens_func,
                                                         inputs=[model_state, text_output, prompt_type, prompt_dict,
                                                                 system_prompt, chat_conversation],
                                                         outputs=chat_token_count,
                                                         api_name='count_tokens' if allow_api else False)

        speak_events = []
        if kwargs['enable_tts'] and kwargs['predict_from_text_func'] is not None:
            if kwargs['tts_model'].startswith('tts_models/'):
                speak_human_event = speak_human_button.click(kwargs['predict_from_text_func'],
                                                             inputs=[instruction, chatbot_role, tts_language,
                                                                     roles_state, tts_speed],
                                                             outputs=speech_human,
                                                             api_name=False,  # not for API
                                                             )
                speak_events.extend([speak_human_event])
            elif kwargs['tts_model'].startswith('microsoft'):
                speak_human_event = speak_human_button.click(kwargs['predict_from_text_func'],
                                                             inputs=[instruction, speaker, tts_speed],
                                                             outputs=speech_human,
                                                             api_name=False,  # not for API
                                                             )
                speak_events.extend([speak_human_event])

        def wrap_pred_func(chatbot_role1, speaker1, tts_language1, roles_state1, tts_speed1,
                           visible_models1, text_output1, text_output21, *args,
                           all_models=[]):
            # FIXME: Choose first visible
            text_outputs1 = list(args)
            text_outputss = [text_output1, text_output21] + text_outputs1
            text_outputss = [x[-1][1] for x in text_outputss if len(x) >= 1 and len(x[-1]) == 2 and x[-1][1]]
            response = text_outputss[0] if text_outputss else ''

            keep_sources_in_context1 = False
            langchain_mode1 = None  # so always tries
            hyde_level1 = None  # so always tries
            response = remove_refs(response, keep_sources_in_context1, langchain_mode1, hyde_level1,
                                   kwargs['gradio_errors_to_chatbot'])

            if kwargs['enable_tts'] and kwargs['predict_from_text_func'] is not None and response:
                if kwargs['tts_model'].startswith('tts_models/') and chatbot_role1 not in [None, 'None']:
                    yield from kwargs['predict_from_text_func'](response, chatbot_role1, tts_language1, roles_state1,
                                                                tts_speed1)
                elif kwargs['tts_model'].startswith('microsoft') and speaker1 not in [None, 'None']:
                    yield from kwargs['predict_from_text_func'](response, speaker1, tts_speed1)

        def _wrap_pred_func_api(chatbot_role1, speaker1, tts_language1, tts_speed1,
                                response, roles_state1):
            if kwargs['tts_model'].startswith('microsoft') and speaker1 not in [None, "None"]:
                sr1 = 16000
            elif kwargs['tts_model'].startswith('tts_models/') and chatbot_role1 not in [None, "None"]:
                sr1 = 24000
            else:
                return
            if kwargs['enable_tts'] and kwargs['predict_from_text_func'] is not None and response:
                if kwargs['tts_model'].startswith('tts_models/') and chatbot_role1 not in [None, 'None']:
                    yield from kwargs['predict_from_text_func'](response, chatbot_role1, tts_language1, roles_state1,
                                                                tts_speed1,
                                                                return_prefix_every_yield=False,
                                                                include_audio0=False,
                                                                return_dict=True,
                                                                sr=sr1)
                elif kwargs['tts_model'].startswith('microsoft') and speaker1 not in [None, 'None']:
                    yield from kwargs['predict_from_text_func'](response, speaker1, tts_speed1,
                                                                return_prefix_every_yield=False,
                                                                include_audio0=False,
                                                                return_dict=True,
                                                                sr=sr1)

        def wrap_pred_func_api(chatbot_role1, speaker1, tts_language1, tts_speed1,
                               response, stream_output1, h2ogpt_key1, roles_state1, requests_state1):
            # check key
            valid_key = is_valid_key(kwargs['enforce_h2ogpt_api_key'],
                                     kwargs['enforce_h2ogpt_ui_key'],
                                     kwargs['h2ogpt_api_keys'],
                                     h2ogpt_key1,
                                     requests_state1=requests_state1)
            kwargs['from_ui'] = is_from_ui(requests_state1)
            if not valid_key:
                raise ValueError(invalid_key_msg)

            if stream_output1:
                yield from _wrap_pred_func_api(chatbot_role1, speaker1, tts_language1, tts_speed1,
                                               response, roles_state1)
            else:
                audios = []
                for audio1 in _wrap_pred_func_api(chatbot_role1, speaker1, tts_language1, tts_speed1,
                                                  response, roles_state1):
                    audios.append(audio1)
                srs = [x['sr'] for x in audios]
                if len(srs) > 0:
                    sr = srs[0]
                    audios = [x['audio'] for x in audios]
                    audios = combine_audios(audios, audio=None, sr=sr, expect_bytes=kwargs['return_as_byte'],
                                            verbose=verbose)
                    yield dict(audio=audios, sr=sr)

        def wrap_pred_func_plain_api(*args1):
            args_dict = ast.literal_eval(args1[0])
            args_dict['requests_state'] = requests_state0.copy()
            args_dict['roles_state'] = roles_state.value.copy()

            input_args_list_speak = ['chatbot_role', 'speaker', 'tts_language', 'tts_speed',
                                     'prompt', 'stream_output', 'h2ogpt_key',
                                     'roles_state', 'requests_state']
            assert len(args_dict) == len(input_args_list_speak)

            # fix order and make into list
            args_dict = {k: args_dict[k] for k in input_args_list_speak}
            args_list = list(args_dict.values())

            ret = yield from wrap_pred_func_api(*tuple(args_list))
            return ret

        speak_bot_event = speak_bot_button.click(wrap_pred_func,
                                                 inputs=[chatbot_role, speaker, tts_language, roles_state, tts_speed,
                                                         visible_models, text_output,
                                                         text_output2] + text_outputs,
                                                 outputs=speech_bot,
                                                 api_name=False,  # not for API
                                                 )
        speak_events.extend([speak_bot_event])

        speak_text_api_event1 = speak_text_api_button.click(**user_state_kwargs)
        speak_text_api_event = speak_text_api_event1.then(wrap_pred_func_api,
                                                          inputs=[chatbot_role, speaker, tts_language, tts_speed,
                                                                  text_speech, stream_output, h2ogpt_key,
                                                                  roles_state, requests_state],
                                                          outputs=text_speech_out,
                                                          api_name='speak_text_api' if allow_api else False,
                                                          )

        speak_text_plain_api_event = speak_text_plain_api_button.click(wrap_pred_func_plain_api,
                                                                       inputs=speak_inputs_dict_str,
                                                                       outputs=text_speech_out,
                                                                       api_name='speak_text_plain_api' if allow_api else False,
                                                                       **noqueue_kwargs_curl,
                                                                       )

        def stop_audio_func():
            return None, None

        if kwargs['enable_tts']:
            stop_speak_button.click(stop_audio_func,
                                    outputs=[speech_human, speech_bot],
                                    cancels=speak_events, **noqueue_kwargs2)

        # don't pass text_output, don't want to clear output, just stop it
        # cancel only stops outer generation, not inner generation or non-generation
        clear_torch_cache_func_soft = functools.partial(clear_torch_cache, allow_skip=True)
        stop_event = stop_btn.click(lambda: None, None, None,
                                    cancels=submits1 + submits2 + submits3 + submits4 +
                                            [submit_event_nochat, submit_event_nochat2] +
                                            [eventdb1, eventdb2, eventdb3] +
                                            [eventdb7a, eventdb7, eventdb8a, eventdb8, eventdb9a, eventdb9, eventdb12a,
                                             eventdb12] +
                                            db_events +
                                            [eventdbloadla, eventdbloadlb] +
                                            [clear_event] +
                                            [submit_event_nochat_api, submit_event_nochat] +
                                            [load_model_event, load_model_event2] +
                                            [count_tokens_event] +
                                            speak_events
                                    ,
                                    **noqueue_kwargs, api_name='stop' if allow_api else False) \
            .then(clear_torch_cache_func_soft, **noqueue_kwargs) \
            .then(stop_audio_func, outputs=[speech_human, speech_bot])

        if kwargs['auth'] is not None:
            auth = authf
            load_func = user_state_setup
            load_inputs = [my_db_state, requests_state, guest_name, login_btn, login_btn]
            load_outputs = [my_db_state, requests_state, login_btn]
        else:
            auth = None
            load_func = user_state_setup
            load_inputs = [my_db_state, requests_state, guest_name, login_btn, login_btn]
            load_outputs = [my_db_state, requests_state, login_btn]
            # auth = None
            # load_func, load_inputs, load_outputs = None, None, None

        app_js = wrap_js_to_lambda(
            len(load_inputs) if load_inputs else 0,
            get_dark_js() if kwargs['dark'] else None,
            get_heap_js(heap_app_id) if is_heap_analytics_enabled else None)

        load_kwargs = dict(js=app_js) if is_gradio_version4 else dict(_js=app_js)
        load_event = demo.load(fn=load_func, inputs=load_inputs, outputs=load_outputs, **load_kwargs)

        if load_func:
            load_event2 = load_event.then(load_login_func,
                                          inputs=login_inputs,
                                          outputs=login_outputs)
        if load_func and auth:
            if not kwargs['large_file_count_mode']:
                get_sources_fun_kwargs_login = get_sources_fun_kwargs.copy()
                get_sources_fun_kwargs_login['for_login'] = True
                get_sources1_login = functools.partial(get_sources_gr, **get_sources_fun_kwargs_login)
                get_sources_kwargs_login = dict(fn=get_sources1_login,
                                                inputs=[my_db_state, selection_docs_state, requests_state,
                                                        langchain_mode,
                                                        h2ogpt_key],
                                                outputs=[file_source, docs_state, text_doc_count],
                                                queue=queue)
                load_event3 = load_event2.then(**get_sources_kwargs_login)
                load_event4 = load_event3.then(fn=update_dropdown, inputs=docs_state, outputs=document_choice)
                show_sources1_fun_kwargs_login = show_sources1_fun_kwargs.copy()
                show_sources1_fun_kwargs_login['for_login'] = True
                show_sources1_login = functools.partial(get_source_files_given_langchain_mode_gr,
                                                        **show_sources1_fun_kwargs_login,
                                                        )
                show_sources_kwargs_login = dict(fn=show_sources1_login,
                                                 inputs=[my_db_state, selection_docs_state, requests_state,
                                                         langchain_mode,
                                                         h2ogpt_key],
                                                 outputs=sources_text)
                load_event5 = load_event4.then(**show_sources_kwargs_login)

                get_viewable_sources1_fun_kwargs_login = get_viewable_sources1_fun_kwargs.copy()
                get_viewable_sources1_fun_kwargs_login['for_login'] = True
                get_viewable_sources1_login = functools.partial(get_sources_gr,
                                                                **get_viewable_sources1_fun_kwargs_login)
                get_viewable_sources_args_login = dict(fn=get_viewable_sources1_login,
                                                       inputs=[my_db_state, selection_docs_state, requests_state,
                                                               langchain_mode,
                                                               h2ogpt_key],
                                                       outputs=[file_source, viewable_docs_state,
                                                                text_viewable_doc_count],
                                                       queue=queue)

                load_event6 = load_event5.then(**get_viewable_sources_args_login)
                load_event7 = load_event6.then(**viewable_kwargs)

        def wrap_transcribe_func_api(audio_obj1, stream_output1, h2ogpt_key1, requests_state1):
            # check key
            valid_key = is_valid_key(kwargs['enforce_h2ogpt_api_key'],
                                     kwargs['enforce_h2ogpt_ui_key'],
                                     kwargs['h2ogpt_api_keys'],
                                     h2ogpt_key1,
                                     requests_state1=requests_state1)
            kwargs['from_ui'] = is_from_ui(requests_state1)
            if not valid_key:
                raise ValueError(invalid_key_msg)

            audio_api_state0 = ['', '', None, 'on']
            state_text = kwargs['transcriber_func'](audio_api_state0, audio_obj1)
            text = state_text[1]
            yield text

        audio_api_output = gr.Textbox(value='', visible=False)
        audio_api_input = gr.Textbox(value='', visible=False)
        audio_api_btn = gr.Button(visible=False)
        audio_api_btn.click(fn=wrap_transcribe_func_api,
                            inputs=[audio_api_input, stream_output, h2ogpt_key, requests_state],
                            outputs=[audio_api_output],
                            api_name='transcribe_audio_api',
                            show_progress='hidden')

        def wrap_embedding_func_api(text, h2ogpt_key1, is_list1, requests_state1):
            # check key
            valid_key = is_valid_key(kwargs['enforce_h2ogpt_api_key'],
                                     kwargs['enforce_h2ogpt_ui_key'],
                                     kwargs['h2ogpt_api_keys'],
                                     h2ogpt_key1,
                                     requests_state1=requests_state1)
            kwargs['from_ui'] = is_from_ui(requests_state1)
            if not valid_key:
                raise ValueError(invalid_key_msg)

            assert not kwargs['use_openai_embedding'], "Should not be using OpenAI embeddings."
            is_list1 = ast.literal_eval(is_list1)
            if is_list1:
                text = ast.literal_eval(text)
            else:
                text = [text]
            embedding = kwargs['hf_embedding_model']['model'].embed_documents(text)
            return embedding

        embed_api_output = gr.Textbox(value='', visible=False)
        embed_api_input = gr.Textbox(value='', visible=False)
        embed_api_btn = gr.Button(visible=False)
        is_list = gr.Textbox(value='False', visible=False)
        embed_api_btn.click(fn=wrap_embedding_func_api,
                            inputs=[embed_api_input, h2ogpt_key, is_list, requests_state],
                            outputs=[embed_api_output],
                            api_name='embed_api',
                            show_progress='hidden')

    demo.queue(**queue_kwargs, api_open=kwargs['api_open'])
    favicon_file = "h2o-logo.svg"
    favicon_path = kwargs['favicon_path'] or favicon_file
    if not os.path.isfile(favicon_file):
        print("favicon_path1=%s not found" % favicon_file, flush=True)
        alt_path = os.path.dirname(os.path.abspath(__file__))
        favicon_path = os.path.join(alt_path, favicon_file)
        if not os.path.isfile(favicon_path):
            print("favicon_path2: %s not found in %s" % (favicon_file, alt_path), flush=True)
            alt_path = os.path.dirname(alt_path)
            favicon_path = os.path.join(alt_path, favicon_file)
            if not os.path.isfile(favicon_path):
                print("favicon_path3: %s not found in %s" % (favicon_file, alt_path), flush=True)
                favicon_path = None

    if kwargs['prepare_offline_level'] > 0:
        from src.prepare_offline import go_prepare_offline
        go_prepare_offline(**locals().copy())
        return

    scheduler = BackgroundScheduler()
    if kwargs['clear_torch_cache_level'] in [0, 1]:
        interval_time = 120
        clear_torch_cache_func_periodic = clear_torch_cache_func_soft
    else:
        interval_time = 20
        clear_torch_cache_func_periodic = clear_torch_cache
    # don't require ever clear torch cache
    scheduler.add_job(func=clear_torch_cache_func_periodic, trigger="interval", seconds=interval_time)
    if is_public and \
            kwargs['base_model'] not in non_hf_types:
        # FIXME: disable for gptj, langchain or gpt4all modify print itself
        # FIXME: and any multi-threaded/async print will enter model output!
        scheduler.add_job(func=ping, trigger="interval", seconds=60)
    if os.getenv('PING_GPU'):
        scheduler.add_job(func=ping_gpu, trigger="interval", seconds=60 * 10)
    scheduler.start()

    # import control
    if kwargs['langchain_mode'] == 'Disabled' and \
            os.environ.get("TEST_LANGCHAIN_IMPORT") and \
            kwargs['base_model'] not in non_hf_types:
        assert 'gpt_langchain' not in sys.modules, "Dev bug, import of langchain when should not have"
        assert 'langchain' not in sys.modules, "Dev bug, import of langchain when should not have"

    # set port in case GRADIO_SERVER_PORT was already set in prior main() call,
    # gradio does not listen if change after import
    # Keep None if not set so can find an open port above used ports
    server_port = os.getenv('GRADIO_SERVER_PORT')
    if server_port is not None:
        server_port = int(server_port)

    # NOTE: Dynamically added paths won't work unless relative to root and not public
    allowed_paths = []
    allowed_paths += [os.path.abspath(v) for k, v in kwargs['langchain_mode_paths'].items() if v]
    allowed_paths += [os.path.abspath(x) for x in kwargs['extra_allowed_paths']]
    blocked_paths = [os.path.abspath(x) for x in kwargs['blocked_paths']]

    max_threads = max(128, 4 * kwargs['concurrency_count']) if isinstance(kwargs['concurrency_count'],
                                                                          int) else 128

    if kwargs['google_auth']:
        import uvicorn
        from gradio_utils.google_auth import get_app
        app_kwargs = dict(
            favicon_path=favicon_path,
            # prevent_thread_lock=True,
            allowed_paths=allowed_paths if allowed_paths else None,
            blocked_paths=blocked_paths if blocked_paths else None,
        )
        app = get_app(demo,
                      markdown_logo=markdown_logo,
                      visible_h2ogpt_logo=kwargs['visible_h2ogpt_logo'],
                      page_title=page_title,
                      )
        uvicorn.run(app,
                    # share not allowed
                    host=kwargs['server_name'],
                    port=server_port or 7860,
                    # show_error not allowed
                    ws_max_queue=max_threads,
                    # workers=max_threads, # https://github.com/tiangolo/fastapi/issues/1495#issuecomment-635681976
                    root_path=kwargs['root_path'],
                    ssl_keyfile=kwargs['ssl_keyfile'],
                    # ssl_verify=kwargs['ssl_verify'], # https://github.com/gradio-app/gradio/issues/2790#issuecomment-2004984763
                    ssl_certfile=kwargs['ssl_certfile'],
                    ssl_keyfile_password=kwargs['ssl_keyfile_password'],
                    limit_concurrency=None,
                    )
    else:
        demo.launch(share=kwargs['share'],
                    server_name=kwargs['server_name'],
                    server_port=server_port,
                    show_error=True,
                    favicon_path=favicon_path,
                    prevent_thread_lock=True,
                    auth=auth,
                    auth_message=auth_message,
                    root_path=kwargs['root_path'],
                    ssl_keyfile=kwargs['ssl_keyfile'],
                    ssl_verify=kwargs['ssl_verify'],
                    ssl_certfile=kwargs['ssl_certfile'],
                    ssl_keyfile_password=kwargs['ssl_keyfile_password'],
                    max_threads=max_threads,
                    allowed_paths=allowed_paths if allowed_paths else None,
                    blocked_paths=blocked_paths if blocked_paths else None,
                    )

    showed_server_name = 'localhost' if kwargs['server_name'] == "0.0.0.0" else kwargs['server_name']
    if kwargs['verbose'] or not (kwargs['base_model'] in ['gptj', 'gpt4all_llama']):
        print("Started Gradio Server and/or GUI: server_name: %s port: %s" % (showed_server_name,
                                                                              server_port),
              flush=True)
    if server_port is None:
        server_port = '7860'

    if kwargs['open_browser']:
        # Open URL in a new tab, if a browser window is already open.
        import webbrowser
        webbrowser.open_new_tab(demo.local_url)
    else:
        print("Use local URL: %s" % demo.local_url, flush=True)

    if kwargs['openai_server'] or kwargs['function_server']:
        url_split = demo.local_url.split(':')
        if len(url_split) == 3:
            gradio_prefix = ':'.join(url_split[0:1]).replace('//', '')
            gradio_host = ':'.join(url_split[1:2]).replace('//', '')
            gradio_port = ':'.join(url_split[2:]).split('/')[0]
        else:
            gradio_prefix = 'http'
            gradio_host = ':'.join(url_split[0:1])
            gradio_port = ':'.join(url_split[1:]).split('/')[0]
        # ensure can reach out
        if platform.system() in ['Darwin', 'Windows']:
            openai_host = gradio_host if gradio_host not in ['localhost', '127.0.0.1'] else '0.0.0.0'
        else:
            if gradio_host in ['localhost', '127.0.0.1']:
                openai_host = gradio_host = '0.0.0.0'
            else:
                openai_host = gradio_host
        from openai_server.server_start import run

        run_kwargs = dict(wait=False,
                          multiple_workers_gunicorn=kwargs['multiple_workers_gunicorn'],
                          host=openai_host,
                          gradio_prefix=gradio_prefix,
                          gradio_host=gradio_host,
                          gradio_port=gradio_port,
                          h2ogpt_key=h2ogpt_key1,
                          auth=kwargs['auth'],
                          auth_access=kwargs['auth_access'],
                          guest_name=kwargs['guest_name'],
                          main_kwargs=json.dumps(kwargs['main_kwargs']),
                          verbose=verbose,
                          )

        if kwargs['openai_server']:
            if verbose:
                print("Starting up OpenAI proxy server")
            if kwargs['openai_workers'] == 1:
                from openai_server.server import app as openai_app
            else:
                openai_app = 'server:app'
            run(**run_kwargs, port=kwargs['openai_port'], app=openai_app, is_openai_server=True,
                workers=kwargs['openai_workers'],
                )

        if kwargs['function_server']:
            if verbose:
                print("Starting up Function server")
            if kwargs['function_server_workers'] == 1:
                from openai_server.function_server import app as function_app
            else:
                function_app = 'function_server:app'
            run(**run_kwargs, port=kwargs['function_server_port'], app=function_app, is_openai_server=False,
                workers=kwargs['function_server_workers'],
                )

    if kwargs['block_gradio_exit']:
        demo.block_thread()


def show_doc(db1s, selection_docs_state1, requests_state1,
             langchain_mode1,
             single_document_choice1,
             view_raw_text_checkbox1,
             text_context_list1,
             pdf_height,
             h2ogpt_key1,
             dbs1=None,
             load_db_if_exists1=None,
             db_type1=None,
             use_openai_embedding1=None,
             hf_embedding_model1=None,
             migrate_embedding_model_or_db1=None,
             verbose1=False,
             get_userid_auth1=None,
             max_raw_chunks=1000000,
             api=False,
             n_jobs=-1,
             enforce_h2ogpt_api_key=True,
             enforce_h2ogpt_ui_key=True,
             h2ogpt_api_keys=[],
             ):
    valid_key = is_valid_key(enforce_h2ogpt_api_key,
                             enforce_h2ogpt_ui_key,
                             h2ogpt_api_keys,
                             h2ogpt_key1,
                             requests_state1=requests_state1)
    from_ui = is_from_ui(requests_state1)
    if not valid_key:
        raise ValueError(invalid_key_msg)

    file = single_document_choice1
    document_choice1 = [single_document_choice1]
    content = None
    db_documents = []
    db_metadatas = []
    if db_type1 in ['chroma', 'chroma_old']:
        assert langchain_mode1 is not None
        langchain_mode_paths = selection_docs_state1['langchain_mode_paths']
        langchain_mode_types = selection_docs_state1['langchain_mode_types']
        from src.gpt_langchain import set_userid, get_any_db, get_docs_and_meta
        set_userid(db1s, requests_state1, get_userid_auth1)
        top_k_docs = -1
        db = get_any_db(db1s, langchain_mode1, langchain_mode_paths, langchain_mode_types,
                        dbs=dbs1,
                        load_db_if_exists=load_db_if_exists1,
                        db_type=db_type1,
                        use_openai_embedding=use_openai_embedding1,
                        hf_embedding_model=hf_embedding_model1,
                        migrate_embedding_model=migrate_embedding_model_or_db1,
                        for_sources_list=True,
                        verbose=verbose1,
                        n_jobs=n_jobs,
                        )
        query_action = False  # long chunks like would be used for summarize
        # the below is as or filter, so will show doc or by chunk, unrestricted
        from langchain_community.vectorstores import Chroma
        if isinstance(db, Chroma):
            # chroma >= 0.4
            if view_raw_text_checkbox1:
                one_filter = \
                    [{"source": {"$eq": x}, "chunk_id": {"$gte": 0}} if query_action else {"source": {"$eq": x},
                                                                                           "chunk_id": {
                                                                                               "$gte": -1}}
                     for x in document_choice1][0]
            else:
                one_filter = \
                    [{"source": {"$eq": x}, "chunk_id": {"$gte": 0}} if query_action else {"source": {"$eq": x},
                                                                                           "chunk_id": {
                                                                                               "$eq": -1}}
                     for x in document_choice1][0]
            filter_kwargs = dict(filter={"$and": [dict(source=one_filter['source']),
                                                  dict(chunk_id=one_filter['chunk_id'])]})
        else:
            # migration for chroma < 0.4
            one_filter = \
                [{"source": {"$eq": x}, "chunk_id": {"$gte": 0}} if query_action else {"source": {"$eq": x},
                                                                                       "chunk_id": {
                                                                                           "$eq": -1}}
                 for x in document_choice1][0]
            if view_raw_text_checkbox1:
                # like or, full raw all chunk types
                filter_kwargs = dict(filter=one_filter)
            else:
                filter_kwargs = dict(filter={"$and": [dict(source=one_filter['source']),
                                                      dict(chunk_id=one_filter['chunk_id'])]})
        db_documents, db_metadatas = get_docs_and_meta(db, top_k_docs, filter_kwargs=filter_kwargs,
                                                       text_context_list=text_context_list1)
        # order documents
        from langchain.docstore.document import Document
        docs_with_score = [(Document(page_content=result[0], metadata=result[1] or {}), 0)
                           for result in zip(db_documents, db_metadatas)]
        doc_chunk_ids = [x.get('chunk_id', -1) for x in db_metadatas]
        doc_page_ids = [x.get('page', 0) for x in db_metadatas]
        doc_hashes = [x.get('doc_hash', 'None') for x in db_metadatas]
        docs_with_score = [x for hx, px, cx, x in
                           sorted(zip(doc_hashes, doc_page_ids, doc_chunk_ids, docs_with_score),
                                  key=lambda x: (x[0], x[1], x[2]))
                           # if cx == -1
                           ]
        db_metadatas = [x[0].metadata for x in docs_with_score][:max_raw_chunks]
        db_documents = [x[0].page_content for x in docs_with_score][:max_raw_chunks]
        # done reordering
        if view_raw_text_checkbox1:
            content = [dict_to_html(x) + '\n' + text_to_html(y) for x, y in zip(db_metadatas, db_documents)]
        else:
            content = [text_to_html(y) for x, y in zip(db_metadatas, db_documents)]
        content = '\n'.join(content)
        content = f"""<!DOCTYPE html>
<html>
<head>
<title>{file}</title>
</head>
<body>
{content}
</body>
</html>"""
    if api:
        if view_raw_text_checkbox1:
            return dict(contents=db_documents, metadatas=db_metadatas)
        else:
            contents = [text_to_html(y, api=api) for y in db_documents]
            metadatas = [dict_to_html(x, api=api) for x in db_metadatas]
            return dict(contents=contents, metadatas=metadatas)
    else:
        assert not api, "API mode for get_document only supported for chroma"

    dummy1 = gr.update(visible=False, value=None)
    # backup is text dump of db version
    if content:
        dummy_ret = dummy1, dummy1, dummy1, dummy1, gr.update(visible=True, value=content), dummy1, dummy1, dummy1
        if view_raw_text_checkbox1:
            return dummy_ret
    else:
        dummy_ret = dummy1, dummy1, dummy1, dummy1, dummy1, dummy1, dummy1, dummy1

    if not isinstance(file, str):
        return dummy_ret

    if file.lower().endswith('.html') or file.lower().endswith('.mhtml') or file.lower().endswith('.htm') or \
            file.lower().endswith('.xml'):
        try:
            with open(file, 'rt') as f:
                content = f.read()
            return gr.update(visible=True, value=content), dummy1, dummy1, dummy1, dummy1, dummy1, dummy1, dummy1
        except:
            return dummy_ret

    if file.lower().endswith('.md'):
        try:
            with open(file, 'rt') as f:
                content = f.read()
            return dummy1, dummy1, dummy1, gr.update(visible=True, value=content), dummy1, dummy1, dummy1, dummy1
        except:
            return dummy_ret

    if file.lower().endswith('.py'):
        try:
            with open(file, 'rt') as f:
                content = f.read()
            content = f"```python\n{content}\n```"
            return dummy1, dummy1, dummy1, gr.update(visible=True, value=content), dummy1, dummy1, dummy1, dummy1
        except:
            return dummy_ret

    if file.lower().endswith('.txt') or file.lower().endswith('.rst') or file.lower().endswith(
            '.rtf') or file.lower().endswith('.toml'):
        try:
            with open(file, 'rt') as f:
                content = f.read()
            # content = f"```text\n{content}\n```"
            content = text_to_html(content, api=api)
            return dummy1, dummy1, dummy1, gr.update(visible=True, value=content), dummy1, dummy1, dummy1, dummy1
        except:
            return dummy_ret

    func = None
    if file.lower().endswith(".csv"):
        func = pd.read_csv
    elif file.lower().endswith(".pickle"):
        func = pd.read_pickle
    elif file.lower().endswith(".xls") or file.lower().endswith("xlsx"):
        func = pd.read_excel
    elif file.lower().endswith('.json'):
        func = pd.read_json
    # pandas doesn't show full thing, even if html view shows broken things still better
    # elif file.lower().endswith('.xml'):
    #    func = pd.read_xml
    if func is not None:
        try:
            df = func(file).head(100)
        except:
            # actual JSON required
            with open(file, 'rt') as f:
                json_blob = f.read()
            return dummy1, dummy1, gr.update(visible=True, value=json_blob), dummy1, dummy1, dummy1, dummy1, dummy1
        return dummy1, gr.update(visible=True, value=df), dummy1, dummy1, dummy1, dummy1, dummy1, dummy1
    port = int(os.getenv('GRADIO_SERVER_PORT', '7860'))
    import pathlib
    if not file.startswith('http'):
        absolute_path_string = os.path.abspath(file)
        url_path = pathlib.Path(absolute_path_string).as_uri()
        url = get_url(absolute_path_string, from_str=True)
        img_url = url.replace("""<a href=""", """<img src=""")
    else:
        img_url = """<img src="%s" alt="%s">""" % (file, file)
    from src.gpt_langchain import image_types, audio_types, video_types
    if any([file.lower().endswith('.' + x) for x in image_types]):
        return gr.update(visible=True, value=img_url), dummy1, dummy1, dummy1, dummy1, dummy1, dummy1, dummy1
    elif any([file.lower().endswith('.' + x) for x in video_types]):
        return dummy1, dummy1, dummy1, dummy1, dummy1, dummy1, dummy1, gr.update(visible=True, value=file)
    elif any([file.lower().endswith('.' + x) for x in audio_types]):
        return dummy1, dummy1, dummy1, dummy1, dummy1, dummy1, gr.update(visible=True, value=file), dummy1
    elif file.lower().endswith('.pdf') or 'arxiv.org/pdf' in file:

        # account for when use `wget -b -m -k -o wget.log -e robots=off`
        if url_alive('http://' + file):
            file = 'http://' + file
        if url_alive('https://' + file):
            file = 'https://' + file

        if file.lower().startswith('http') or file.lower().startswith('https'):
            # if file is online, then might as well use google(?)
            document1 = file
            return gr.update(visible=True,
                             value=f"""<iframe width="1000" height="{pdf_height}" src="https://docs.google.com/viewerng/viewer?url={document1}&embedded=true" frameborder="0" height="100%" width="100%">
</iframe>
"""), dummy1, dummy1, dummy1, dummy1, dummy1, dummy1, dummy1
        elif have_gradio_pdf and os.path.isfile(file):
            from gradio_pdf import PDF
            return dummy1, dummy1, dummy1, dummy1, dummy1, PDF(file, visible=True, label=file, show_label=True,
                                                               height=pdf_height), dummy1, dummy1
        else:
            return dummy_ret
    else:
        return dummy_ret


def get_inputs_list(inputs_dict, model_lower, model_id=1):
    """
    map gradio objects in locals() to inputs for evaluate().
    :param inputs_dict:
    :param model_lower:
    :param model_id: Which model (1 or 2) of 2
    :return:
    """
    inputs_list_names = list(inspect.signature(evaluate).parameters)
    inputs_list = []
    inputs_dict_out = {}
    for k in inputs_list_names:
        if k == 'kwargs':
            continue
        if k in input_args_list + inputs_kwargs_list:
            # these are added at use time for args or partial for kwargs, not taken as input
            continue
        if 'mbart-' not in model_lower and k in ['src_lang', 'tgt_lang']:
            continue
        if model_id == 2:
            if k == 'prompt_type':
                k = 'prompt_type2'
            if k == 'prompt_used':
                k = 'prompt_used2'
            if k == 'max_new_tokens':
                k = 'max_new_tokens2'
            if k == 'min_new_tokens':
                k = 'min_new_tokens2'
        inputs_list.append(inputs_dict[k])
        inputs_dict_out[k] = inputs_dict[k]
    return inputs_list, inputs_dict_out


def update_user_db_gr(file, db1s, selection_docs_state1, requests_state1,
                      langchain_mode, chunk, chunk_size, embed,

                      image_audio_loaders,
                      pdf_loaders,
                      url_loaders,
                      jq_schema,
                      extract_frames,
                      llava_prompt,
                      h2ogpt_key,

                      captions_model=None,
                      caption_loader=None,
                      doctr_loader=None,
                      llava_model=None,
                      asr_model=None,
                      asr_loader=None,

                      dbs=None,
                      get_userid_auth=None,
                      **kwargs):
    valid_key = is_valid_key(kwargs.pop('enforce_h2ogpt_api_key', None),
                             kwargs.pop('enforce_h2ogpt_ui_key', None),
                             kwargs.pop('h2ogpt_api_keys', []),
                             h2ogpt_key,
                             requests_state1=requests_state1)
    kwargs['from_ui'] = is_from_ui(requests_state1)
    if not valid_key:
        raise ValueError(invalid_key_msg)
    loaders_dict, captions_model, asr_model = gr_to_lg(image_audio_loaders,
                                                       pdf_loaders,
                                                       url_loaders,
                                                       captions_model=captions_model,
                                                       asr_model=asr_model,
                                                       **kwargs,
                                                       )
    if jq_schema is None:
        jq_schema = kwargs['jq_schema0']
    loaders_dict.update(dict(captions_model=captions_model,
                             caption_loader=caption_loader,
                             doctr_loader=doctr_loader,
                             llava_model=llava_model,
                             llava_prompt=llava_prompt,
                             asr_model=asr_model,
                             asr_loader=asr_loader,
                             jq_schema=jq_schema,
                             extract_frames=extract_frames,
                             ))
    kwargs.pop('image_audio_loaders_options0', None)
    kwargs.pop('pdf_loaders_options0', None)
    kwargs.pop('url_loaders_options0', None)
    kwargs.pop('jq_schema0', None)
    if not embed:
        kwargs['use_openai_embedding'] = False
        kwargs['hf_embedding_model'] = 'fake'
        kwargs['migrate_embedding_model'] = False

    # avoid dups after loaders_dict updated with new results
    for k, v in loaders_dict.items():
        if k in kwargs:
            kwargs.pop(k, None)

    from src.gpt_langchain import update_user_db
    return update_user_db(file, db1s, selection_docs_state1, requests_state1,
                          langchain_mode=langchain_mode, chunk=chunk, chunk_size=chunk_size,
                          **loaders_dict,
                          dbs=dbs,
                          get_userid_auth=get_userid_auth,
                          **kwargs)


def get_sources_gr(db1s, selection_docs_state1, requests_state1, langchain_mode, h2ogpt_key1,
                   dbs=None, docs_state0=None,
                   load_db_if_exists=None,
                   db_type=None,
                   use_openai_embedding=None,
                   hf_embedding_model=None,
                   migrate_embedding_model=None,
                   verbose=False,
                   get_userid_auth=None,
                   api=False,
                   n_jobs=-1,
                   enforce_h2ogpt_api_key=True,
                   enforce_h2ogpt_ui_key=True,
                   h2ogpt_api_keys=[],
                   for_login=False,
                   ):
    valid_key = is_valid_key(enforce_h2ogpt_api_key,
                             enforce_h2ogpt_ui_key,
                             h2ogpt_api_keys,
                             h2ogpt_key1,
                             requests_state1=requests_state1,
                             )
    from_ui = is_from_ui(requests_state1)
    if not valid_key:
        if for_login:
            from src.utils_langchain import make_sources_file
            sources_file = make_sources_file(langchain_mode, '')
            return sources_file, [], ''
        else:
            raise ValueError(invalid_key_msg)

    from src.gpt_langchain import get_sources
    sources_file, source_list, num_chunks, num_sources_str, db = \
        get_sources(db1s, selection_docs_state1, requests_state1, langchain_mode,
                    dbs=dbs, docs_state0=docs_state0,
                    load_db_if_exists=load_db_if_exists,
                    db_type=db_type,
                    use_openai_embedding=use_openai_embedding,
                    hf_embedding_model=hf_embedding_model,
                    migrate_embedding_model=migrate_embedding_model,
                    verbose=verbose,
                    get_userid_auth=get_userid_auth,
                    n_jobs=n_jobs,
                    )
    if api:
        return source_list
    if langchain_mode in langchain_modes_non_db:
        doc_counts_str = "LLM Mode\nNo Collection"
    else:
        doc_counts_str = "Collection: %s\nDocs: %s\nChunks: %d" % (langchain_mode, num_sources_str, num_chunks)
    return sources_file, source_list, doc_counts_str


def get_source_files_given_langchain_mode_gr(db1s, selection_docs_state1, requests_state1,
                                             langchain_mode,
                                             h2ogpt_key,
                                             dbs=None,
                                             load_db_if_exists=None,
                                             db_type=None,
                                             use_openai_embedding=None,
                                             hf_embedding_model=None,
                                             migrate_embedding_model=None,
                                             verbose=False,
                                             get_userid_auth=None,
                                             n_jobs=-1,
                                             enforce_h2ogpt_api_key=True,
                                             enforce_h2ogpt_ui_key=True,
                                             h2ogpt_api_keys=[],
                                             for_login=False,
                                             ):
    valid_key = is_valid_key(enforce_h2ogpt_api_key,
                             enforce_h2ogpt_ui_key,
                             h2ogpt_api_keys,
                             h2ogpt_key,
                             requests_state1=requests_state1,
                             )
    from_ui = is_from_ui(requests_state1)
    if not valid_key:
        if for_login:
            return "Sources: N/A"
        else:
            raise ValueError(invalid_key_msg)

    from src.gpt_langchain import get_source_files_given_langchain_mode
    return get_source_files_given_langchain_mode(db1s, selection_docs_state1, requests_state1, None,
                                                 langchain_mode,
                                                 dbs=dbs,
                                                 load_db_if_exists=load_db_if_exists,
                                                 db_type=db_type,
                                                 use_openai_embedding=use_openai_embedding,
                                                 hf_embedding_model=hf_embedding_model,
                                                 migrate_embedding_model=migrate_embedding_model,
                                                 verbose=verbose,
                                                 get_userid_auth=get_userid_auth,
                                                 delete_sources=False,
                                                 n_jobs=n_jobs)


def del_source_files_given_langchain_mode_gr(db1s, selection_docs_state1, requests_state1, document_choice1,
                                             langchain_mode,
                                             h2ogpt_key1,
                                             dbs=None,
                                             load_db_if_exists=None,
                                             db_type=None,
                                             use_openai_embedding=None,
                                             hf_embedding_model=None,
                                             migrate_embedding_model=None,
                                             verbose=False,
                                             get_userid_auth=None,
                                             n_jobs=-1,
                                             enforce_h2ogpt_api_key=True,
                                             enforce_h2ogpt_ui_key=True,
                                             h2ogpt_api_keys=[],
                                             ):
    valid_key = is_valid_key(enforce_h2ogpt_api_key,
                             enforce_h2ogpt_ui_key,
                             h2ogpt_api_keys,
                             h2ogpt_key1,
                             requests_state1=requests_state1,
                             )
    from_ui = is_from_ui(requests_state1)
    if not valid_key:
        raise ValueError(invalid_key_msg)

    from src.gpt_langchain import get_source_files_given_langchain_mode
    return get_source_files_given_langchain_mode(db1s, selection_docs_state1, requests_state1, document_choice1,
                                                 langchain_mode,
                                                 dbs=dbs,
                                                 load_db_if_exists=load_db_if_exists,
                                                 db_type=db_type,
                                                 use_openai_embedding=use_openai_embedding,
                                                 hf_embedding_model=hf_embedding_model,
                                                 migrate_embedding_model=migrate_embedding_model,
                                                 verbose=verbose,
                                                 get_userid_auth=get_userid_auth,
                                                 delete_sources=True,
                                                 n_jobs=n_jobs)


def update_and_get_source_files_given_langchain_mode_gr(db1s,
                                                        selection_docs_state,
                                                        requests_state,
                                                        langchain_mode, chunk, chunk_size,

                                                        image_audio_loaders,
                                                        pdf_loaders,
                                                        url_loaders,
                                                        jq_schema,
                                                        extract_frames,
                                                        llava_prompt,

                                                        h2ogpt_key1,

                                                        captions_model=None,
                                                        caption_loader=None,
                                                        doctr_loader=None,
                                                        llava_model=None,
                                                        asr_model=None,
                                                        asr_loader=None,

                                                        dbs=None, first_para=None,
                                                        hf_embedding_model=None,
                                                        use_openai_embedding=None,
                                                        migrate_embedding_model=None,
                                                        text_limit=None,
                                                        db_type=None, load_db_if_exists=None,
                                                        n_jobs=None, verbose=None, get_userid_auth=None,
                                                        image_audio_loaders_options0=None,
                                                        pdf_loaders_options0=None,
                                                        url_loaders_options0=None,
                                                        jq_schema0=None,
                                                        use_pymupdf=None,
                                                        use_unstructured_pdf=None,
                                                        use_pypdf=None,
                                                        enable_pdf_ocr=None,
                                                        enable_pdf_doctr=None,
                                                        try_pdf_as_html=None,
                                                        enforce_h2ogpt_api_key=True,
                                                        enforce_h2ogpt_ui_key=True,
                                                        h2ogpt_api_keys=[],
                                                        ):
    valid_key = is_valid_key(enforce_h2ogpt_api_key,
                             enforce_h2ogpt_ui_key,
                             h2ogpt_api_keys,
                             h2ogpt_key1,
                             requests_state1=requests_state,
                             )
    from_ui = is_from_ui(requests_state)
    if not valid_key:
        raise ValueError(invalid_key_msg)

    from src.gpt_langchain import update_and_get_source_files_given_langchain_mode

    loaders_dict, captions_model, asr_model = gr_to_lg(image_audio_loaders,
                                                       pdf_loaders,
                                                       url_loaders,
                                                       use_pymupdf=use_pymupdf,
                                                       use_unstructured_pdf=use_unstructured_pdf,
                                                       use_pypdf=use_pypdf,
                                                       enable_pdf_ocr=enable_pdf_ocr,
                                                       enable_pdf_doctr=enable_pdf_doctr,
                                                       try_pdf_as_html=try_pdf_as_html,
                                                       image_audio_loaders_options0=image_audio_loaders_options0,
                                                       pdf_loaders_options0=pdf_loaders_options0,
                                                       url_loaders_options0=url_loaders_options0,
                                                       captions_model=captions_model,
                                                       asr_model=asr_model,
                                                       )
    if jq_schema is None:
        jq_schema = jq_schema0
    loaders_dict.update(dict(captions_model=captions_model,
                             caption_loader=caption_loader,
                             doctr_loader=doctr_loader,
                             llava_model=llava_model,
                             llava_prompt=llava_prompt,
                             asr_loader=asr_loader,
                             jq_schema=jq_schema,
                             extract_frames=extract_frames,
                             ))

    return update_and_get_source_files_given_langchain_mode(db1s,
                                                            selection_docs_state,
                                                            requests_state,
                                                            langchain_mode, chunk, chunk_size,
                                                            **loaders_dict,
                                                            dbs=dbs, first_para=first_para,
                                                            hf_embedding_model=hf_embedding_model,
                                                            use_openai_embedding=use_openai_embedding,
                                                            migrate_embedding_model=migrate_embedding_model,
                                                            text_limit=text_limit,
                                                            db_type=db_type, load_db_if_exists=load_db_if_exists,
                                                            n_jobs=n_jobs, verbose=verbose,
                                                            get_userid_auth=get_userid_auth)


def set_userid_gr(db1s, requests_state1, get_userid_auth):
    from src.gpt_langchain import set_userid
    return set_userid(db1s, requests_state1, get_userid_auth)


def set_dbid_gr(db1):
    from src.gpt_langchain import set_dbid
    return set_dbid(db1)


def set_userid_direct_gr(db1s, userid, username):
    from src.gpt_langchain import set_userid_direct
    return set_userid_direct(db1s, userid, username)
