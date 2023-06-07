import copy
import functools
import inspect
import json
import os
import pprint
import random
import sys
import traceback
import uuid
import filelock
import pandas as pd
import requests
import tabulate

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

from gradio_themes import H2oTheme, SoftTheme, get_h2o_title, get_simple_title, get_dark_js
from prompter import Prompter, \
    prompt_type_to_model_name, prompt_types_strings, inv_prompt_type_to_model_lower, generate_prompt, non_hf_types, \
    get_prompt
from utils import get_githash, flatten_list, zip_data, s3up, clear_torch_cache, get_torch_allocated, system_info_print, \
    ping, get_short_name, get_url, makedirs, get_kwargs, DocumentChoices
from generate import get_model, languages_covered, evaluate, eval_func_param_names, score_qa, langchain_modes, \
    inputs_kwargs_list, get_cutoffs, scratch_base_dir, evaluate_from_str, no_default_param_names, \
    eval_func_param_names_defaults, get_max_max_new_tokens

from apscheduler.schedulers.background import BackgroundScheduler


def go_gradio(**kwargs):
    allow_api = kwargs['allow_api']
    is_public = kwargs['is_public']
    is_hf = kwargs['is_hf']
    memory_restriction_level = kwargs['memory_restriction_level']
    n_gpus = kwargs['n_gpus']
    admin_pass = kwargs['admin_pass']
    model_state0 = kwargs['model_state0']
    score_model_state0 = kwargs['score_model_state0']
    dbs = kwargs['dbs']
    db_type = kwargs['db_type']
    visible_langchain_modes = kwargs['visible_langchain_modes']
    allow_upload_to_user_data = kwargs['allow_upload_to_user_data']
    allow_upload_to_my_data = kwargs['allow_upload_to_my_data']
    enable_sources_list = kwargs['enable_sources_list']
    enable_url_upload = kwargs['enable_url_upload']
    enable_text_upload = kwargs['enable_text_upload']
    use_openai_embedding = kwargs['use_openai_embedding']
    hf_embedding_model = kwargs['hf_embedding_model']
    enable_captions = kwargs['enable_captions']
    captions_model = kwargs['captions_model']
    enable_ocr = kwargs['enable_ocr']
    caption_loader = kwargs['caption_loader']

    # easy update of kwargs needed for evaluate() etc.
    queue = True
    allow_upload = allow_upload_to_user_data or allow_upload_to_my_data
    kwargs.update(locals())

    if 'mbart-' in kwargs['model_lower']:
        instruction_label_nochat = "Text to translate"
    else:
        instruction_label_nochat = "Instruction (Shift-Enter or push Submit to send message," \
                                   " use Enter for multiple input lines)"
    if kwargs['input_lines'] > 1:
        instruction_label = "You (Shift-Enter or push Submit to send message, use Enter for multiple input lines)"
    else:
        instruction_label = "You (Enter or push Submit to send message, shift-enter for more lines)"

    title = 'h2oGPT'
    if 'h2ogpt-research' in kwargs['base_model']:
        title += " [Research demonstration]"
    more_info = """For more information, visit our GitHub pages: [h2oGPT](https://github.com/h2oai/h2ogpt) and [H2O-LLMStudio](https://github.com/h2oai/h2o-llmstudio)<br>"""
    if is_public:
        more_info += """<iframe src="https://ghbtns.com/github-btn.html?user=h2oai&repo=h2ogpt&type=star&count=true&size=small" frameborder="0" scrolling="0" width="150" height="20" title="GitHub"></iframe>"""
    if kwargs['verbose']:
        description = f"""Model {kwargs['base_model']} Instruct dataset.
                      For more information, visit our GitHub pages: [h2oGPT](https://github.com/h2oai/h2ogpt) and [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio).
                      Command: {str(' '.join(sys.argv))}
                      Hash: {get_githash()}
                      """
    else:
        description = more_info
    description += "If this host is busy, try [12B](https://gpt.h2o.ai), [Falcon 40B](http://falcon.h2o.ai), [HF Spaces1 12B](https://huggingface.co/spaces/h2oai/h2ogpt-chatbot) or [HF Spaces2 12B](https://huggingface.co/spaces/h2oai/h2ogpt-chatbot2)<br>"
    description += """<p>By using h2oGPT, you accept our [Terms of Service](https://github.com/h2oai/h2ogpt/blob/main/docs/tos.md)</p>"""
    if is_hf:
        description += '''<a href="https://huggingface.co/spaces/h2oai/h2ogpt-chatbot?duplicate=true"><img src="https://bit.ly/3gLdBN6" style="white-space: nowrap" alt="Duplicate Space"></a>'''

    if kwargs['verbose']:
        task_info_md = f"""
        ### Task: {kwargs['task_info']}"""
    else:
        task_info_md = ''

    if kwargs['h2ocolors']:
        css_code = """footer {visibility: hidden;}
    body{background:linear-gradient(#f5f5f5,#e5e5e5);}
    body.dark{background:linear-gradient(#000000,#0d0d0d);}
    """
    else:
        css_code = """footer {visibility: hidden}"""
    css_code += """
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600&display=swap');
body.dark{#warning {background-color: #555555};}
#small_btn {
    margin: 0.6em 0em 0.55em 0;
    max-width: 20em;
    min-width: 5em !important;
    height: 5em;
    font-size: 14px !important
}"""

    if kwargs['gradio_avoid_processing_markdown']:
        from gradio_client import utils as client_utils
        from gradio.components import Chatbot

        # gradio has issue with taking too long to process input/output for markdown etc.
        # Avoid for now, allow raw html to render, good enough for chatbot.
        def _postprocess_chat_messages(self, chat_message: str):
            if chat_message is None:
                return None
            elif isinstance(chat_message, (tuple, list)):
                filepath = chat_message[0]
                mime_type = client_utils.get_mimetype(filepath)
                filepath = self.make_temp_copy_if_needed(filepath)
                return {
                    "name": filepath,
                    "mime_type": mime_type,
                    "alt_text": chat_message[1] if len(chat_message) > 1 else None,
                    "data": None,  # These last two fields are filled in by the frontend
                    "is_file": True,
                }
            elif isinstance(chat_message, str):
                return chat_message
            else:
                raise ValueError(f"Invalid message for Chatbot component: {chat_message}")

        Chatbot._postprocess_chat_messages = _postprocess_chat_messages

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

    theme = H2oTheme(**theme_kwargs) if kwargs['h2ocolors'] else SoftTheme(**theme_kwargs)
    demo = gr.Blocks(theme=theme, css=css_code, title="h2oGPT", analytics_enabled=False)
    callback = gr.CSVLogger()

    model_options = flatten_list(list(prompt_type_to_model_name.values())) + kwargs['extra_model_options']
    if kwargs['base_model'].strip() not in model_options:
        lora_options = [kwargs['base_model'].strip()] + model_options
    lora_options = kwargs['extra_lora_options']
    if kwargs['lora_weights'].strip() not in lora_options:
        lora_options = [kwargs['lora_weights'].strip()] + lora_options
    # always add in no lora case
    # add fake space so doesn't go away in gradio dropdown
    no_lora_str = no_model_str = '[None/Remove]'
    lora_options = [no_lora_str] + kwargs['extra_lora_options']  # FIXME: why double?
    # always add in no model case so can free memory
    # add fake space so doesn't go away in gradio dropdown
    model_options = [no_model_str] + model_options

    # transcribe, will be detranscribed before use by evaluate()
    if not kwargs['lora_weights'].strip():
        kwargs['lora_weights'] = no_lora_str

    if not kwargs['base_model'].strip():
        kwargs['base_model'] = no_model_str

    # transcribe for gradio
    kwargs['gpu_id'] = str(kwargs['gpu_id'])

    no_model_msg = 'h2oGPT [   !!! Please Load Model in Models Tab !!!   ]'
    output_label0 = f'h2oGPT [Model: {kwargs.get("base_model")}]' if kwargs.get(
        'base_model') else no_model_msg
    output_label0_model2 = no_model_msg

    default_kwargs = {k: kwargs[k] for k in eval_func_param_names_defaults}
    for k in no_default_param_names:
        default_kwargs[k] = ''

    with demo:
        # avoid actual model/tokenizer here or anything that would be bad to deepcopy
        # https://github.com/gradio-app/gradio/issues/3558
        model_state = gr.State(['model', 'tokenizer', kwargs['device'], kwargs['base_model']])
        model_state2 = gr.State([None, None, None, None])
        model_options_state = gr.State([model_options])
        lora_options_state = gr.State([lora_options])
        my_db_state = gr.State([None, None])
        chat_state = gr.State({})
        # make user default first and default choice, dedup
        docs_state00 = kwargs['document_choice'] + [x.name for x in list(DocumentChoices)]
        docs_state0 = []
        [docs_state0.append(x) for x in docs_state00 if x not in docs_state0]
        docs_state = gr.State(docs_state0)  # first is chosen as default
        gr.Markdown(f"""
            {get_h2o_title(title) if kwargs['h2ocolors'] else get_simple_title(title)}

            {description}
            {task_info_md}
            """)
        if is_hf:
            gr.HTML(
            )

        # go button visible if
        base_wanted = kwargs['base_model'] != no_model_str and kwargs['login_mode_if_model0']
        go_btn = gr.Button(value="ENTER", visible=base_wanted, variant="primary")
        normal_block = gr.Row(visible=not base_wanted)
        with normal_block:
            with gr.Tabs():
                with gr.Row():
                    col_nochat = gr.Column(visible=not kwargs['chat'])
                    with col_nochat:  # FIXME: for model comparison, and check rest
                        if kwargs['langchain_mode'] == 'Disabled':
                            text_output_nochat = gr.Textbox(lines=5, label=output_label0).style(show_copy_button=True)
                        else:
                            # text looks a bit worse, but HTML links work
                            text_output_nochat = gr.HTML(label=output_label0)
                        instruction_nochat = gr.Textbox(
                            lines=kwargs['input_lines'],
                            label=instruction_label_nochat,
                            placeholder=kwargs['placeholder_instruction'],
                        )
                        iinput_nochat = gr.Textbox(lines=4, label="Input context for Instruction",
                                                   placeholder=kwargs['placeholder_input'])
                        submit_nochat = gr.Button("Submit")
                        flag_btn_nochat = gr.Button("Flag")
                        if not kwargs['auto_score']:
                            with gr.Column(visible=kwargs['score_model']):
                                score_btn_nochat = gr.Button("Score last prompt & response")
                                score_text_nochat = gr.Textbox("Response Score: NA", show_label=False)
                        else:
                            with gr.Column(visible=kwargs['score_model']):
                                score_text_nochat = gr.Textbox("Response Score: NA", show_label=False)
                    col_chat = gr.Column(visible=kwargs['chat'])
                    with col_chat:
                        with gr.Row():
                            text_output = gr.Chatbot(label=output_label0).style(height=kwargs['height'] or 400)
                            text_output2 = gr.Chatbot(label=output_label0_model2, visible=False).style(
                                height=kwargs['height'] or 400)
                        with gr.Row():
                            with gr.Column(scale=50):
                                instruction = gr.Textbox(
                                    lines=kwargs['input_lines'],
                                    label=instruction_label,
                                    placeholder=kwargs['placeholder_instruction'],
                                )
                            with gr.Row():
                                submit = gr.Button(value='Submit').style(full_width=False, size='sm')
                                stop_btn = gr.Button(value="Stop").style(full_width=False, size='sm')
                        with gr.Row():
                            clear = gr.Button("Save Chat / New Chat")
                            flag_btn = gr.Button("Flag")
                            if not kwargs['auto_score']:  # FIXME: For checkbox model2
                                with gr.Column(visible=kwargs['score_model']):
                                    with gr.Row():
                                        score_btn = gr.Button("Score last prompt & response").style(
                                            full_width=False, size='sm')
                                        score_text = gr.Textbox("Response Score: NA", show_label=False)
                                    score_res2 = gr.Row(visible=False)
                                    with score_res2:
                                        score_btn2 = gr.Button("Score last prompt & response 2").style(
                                            full_width=False, size='sm')
                                        score_text2 = gr.Textbox("Response Score2: NA", show_label=False)
                            else:
                                with gr.Column(visible=kwargs['score_model']):
                                    score_text = gr.Textbox("Response Score: NA", show_label=False)
                                    score_text2 = gr.Textbox("Response Score2: NA", show_label=False, visible=False)
                            retry = gr.Button("Regenerate")
                            undo = gr.Button("Undo")
                    submit_nochat_api = gr.Button("Submit nochat API", visible=False)
                    inputs_dict_str = gr.Textbox(label='API input for nochat', show_label=False, visible=False)
                    text_output_nochat_api = gr.Textbox(lines=5, label='API nochat output', visible=False).style(
                        show_copy_button=True)
                with gr.TabItem("Chat"):
                    with gr.Row():
                        if 'mbart-' in kwargs['model_lower']:
                            src_lang = gr.Dropdown(list(languages_covered().keys()),
                                                   value=kwargs['src_lang'],
                                                   label="Input Language")
                            tgt_lang = gr.Dropdown(list(languages_covered().keys()),
                                                   value=kwargs['tgt_lang'],
                                                   label="Output Language")
                    radio_chats = gr.Radio(value=None, label="Saved Chats", visible=True, interactive=True,
                                           type='value')
                    with gr.Row():
                        clear_chat_btn = gr.Button(value="Clear Chat", visible=True).style(size='sm')
                        export_chats_btn = gr.Button(value="Export Chats to Download").style(size='sm')
                        remove_chat_btn = gr.Button(value="Remove Selected Chat", visible=True).style(size='sm')
                        add_to_chats_btn = gr.Button("Import Chats from Upload").style(size='sm')
                    with gr.Row():
                        chats_file = gr.File(interactive=False, label="Download Exported Chats")
                        chatsup_output = gr.File(label="Upload Chat File(s)",
                                                 file_types=['.json'],
                                                 file_count='multiple',
                                                 elem_id="warning", elem_classes="feedback")
                with gr.TabItem("Data Source"):
                    langchain_readme = get_url('https://github.com/h2oai/h2ogpt/blob/main/docs/README_LangChain.md',
                                               from_str=True)
                    gr.HTML(value=f"""LangChain Support Disabled<p>
                            Run:<p>
                            <code>
                            python generate.py --langchain_mode=MyData
                            </code>
                            <p>
                            For more options see: {langchain_readme}""",
                            visible=kwargs['langchain_mode'] == 'Disabled', interactive=False)
                    data_row1 = gr.Row(visible=kwargs['langchain_mode'] != 'Disabled')
                    with data_row1:
                        if is_hf:
                            # don't show 'wiki' since only usually useful for internal testing at moment
                            no_show_modes = ['Disabled', 'wiki']
                        else:
                            no_show_modes = ['Disabled']
                        allowed_modes = visible_langchain_modes.copy()
                        allowed_modes = [x for x in allowed_modes if x in dbs]
                        allowed_modes += ['ChatLLM', 'LLM']
                        if allow_upload_to_my_data and 'MyData' not in allowed_modes:
                            allowed_modes += ['MyData']
                        if allow_upload_to_user_data and 'UserData' not in allowed_modes:
                            allowed_modes += ['UserData']
                        langchain_mode = gr.Radio(
                            [x for x in langchain_modes if x in allowed_modes and x not in no_show_modes],
                            value=kwargs['langchain_mode'],
                            label="Data Collection of Sources",
                            visible=kwargs['langchain_mode'] != 'Disabled')
                    data_row2 = gr.Row(visible=kwargs['langchain_mode'] != 'Disabled')
                    with data_row2:
                        with gr.Column(scale=50):
                            document_choice = gr.Dropdown(docs_state.value,
                                                          label="Choose Subset of Doc(s) in Collection [click get sources to update]",
                                                          value=docs_state.value[0],
                                                          interactive=True,
                                                          multiselect=True,
                                                          )
                        with gr.Row(visible=kwargs['langchain_mode'] != 'Disabled' and enable_sources_list):
                            get_sources_btn = gr.Button(value="Get Sources",
                                                        ).style(full_width=False, size='sm')
                            show_sources_btn = gr.Button(value="Show Sources",
                                                         ).style(full_width=False, size='sm')
                            refresh_sources_btn = gr.Button(value="Refresh Sources",
                                                            ).style(full_width=False, size='sm')

                    # import control
                    if kwargs['langchain_mode'] != 'Disabled':
                        from gpt_langchain import file_types, have_arxiv
                    else:
                        have_arxiv = False
                        file_types = []

                    upload_row = gr.Row(visible=kwargs['langchain_mode'] != 'Disabled' and allow_upload).style(
                        equal_height=False)
                    with upload_row:
                        with gr.Column():
                            file_types_str = '[' + ' '.join(file_types) + ']'
                            fileup_output = gr.File(label=f'Upload {file_types_str}',
                                                    file_types=file_types,
                                                    file_count="multiple",
                                                    elem_id="warning", elem_classes="feedback")
                            with gr.Row():
                                add_to_shared_db_btn = gr.Button("Add File(s) to UserData",
                                                                 visible=allow_upload_to_user_data, elem_id='small_btn')
                                add_to_my_db_btn = gr.Button("Add File(s) to Scratch MyData",
                                                             visible=allow_upload_to_my_data,
                                                             elem_id='small_btn' if allow_upload_to_user_data else None,
                                                             ).style(
                                    size='sm' if not allow_upload_to_user_data else None)
                        with gr.Column(
                                visible=kwargs['langchain_mode'] != 'Disabled' and allow_upload and enable_url_upload):
                            url_label = 'URL (http/https) or ArXiv:' if have_arxiv else 'URL (http/https)'
                            url_text = gr.Textbox(label=url_label, interactive=True)
                            with gr.Row():
                                url_user_btn = gr.Button(value='Add URL content to Shared UserData',
                                                         visible=allow_upload_to_user_data, elem_id='small_btn')
                                url_my_btn = gr.Button(value='Add URL content to Scratch MyData',
                                                       visible=allow_upload_to_my_data,
                                                       elem_id='small_btn' if allow_upload_to_user_data else None,
                                                       ).style(size='sm' if not allow_upload_to_user_data else None)
                        with gr.Column(
                                visible=kwargs['langchain_mode'] != 'Disabled' and allow_upload and enable_text_upload):
                            user_text_text = gr.Textbox(label='Paste Text [Shift-Enter more lines]', interactive=True)
                            with gr.Row():
                                user_text_user_btn = gr.Button(value='Add Text to Shared UserData',
                                                               visible=allow_upload_to_user_data,
                                                               elem_id='small_btn')
                                user_text_my_btn = gr.Button(value='Add Text to Scratch MyData',
                                                             visible=allow_upload_to_my_data,
                                                             elem_id='small_btn' if allow_upload_to_user_data else None,
                                                             ).style(
                                    size='sm' if not allow_upload_to_user_data else None)
                        with gr.Column(visible=False):
                            # WIP:
                            with gr.Row(visible=False).style(equal_height=False):
                                github_textbox = gr.Textbox(label="Github URL")
                                with gr.Row(visible=True):
                                    github_shared_btn = gr.Button(value="Add Github to Shared UserData",
                                                                  visible=allow_upload_to_user_data,
                                                                  elem_id='small_btn')
                                    github_my_btn = gr.Button(value="Add Github to Scratch MyData",
                                                              visible=allow_upload_to_my_data, elem_id='small_btn')
                    sources_row3 = gr.Row(visible=kwargs['langchain_mode'] != 'Disabled' and enable_sources_list).style(
                        equal_height=False)
                    with sources_row3:
                        with gr.Column(scale=1):
                            file_source = gr.File(interactive=False,
                                                  label="Download File w/Sources [click get sources to make file]")
                        with gr.Column(scale=2):
                            pass
                    sources_row = gr.Row(visible=kwargs['langchain_mode'] != 'Disabled' and enable_sources_list).style(
                        equal_height=False)
                    with sources_row:
                        sources_text = gr.HTML(label='Sources Added', interactive=False)

                with gr.TabItem("Expert"):
                    with gr.Row():
                        with gr.Column():
                            stream_output = gr.components.Checkbox(label="Stream output",
                                                                   value=kwargs['stream_output'])
                            prompt_type = gr.Dropdown(prompt_types_strings,
                                                      value=kwargs['prompt_type'], label="Prompt Type",
                                                      visible=not is_public)
                            prompt_type2 = gr.Dropdown(prompt_types_strings,
                                                       value=kwargs['prompt_type'], label="Prompt Type Model 2",
                                                       visible=not is_public and False)
                            do_sample = gr.Checkbox(label="Sample",
                                                    info="Enable sampler, required for use of temperature, top_p, top_k",
                                                    value=kwargs['do_sample'])
                            temperature = gr.Slider(minimum=0.01, maximum=3,
                                                    value=kwargs['temperature'],
                                                    label="Temperature",
                                                    info="Lower is deterministic (but may lead to repeats), Higher more creative (but may lead to hallucinations)")
                            top_p = gr.Slider(minimum=0, maximum=1,
                                              value=kwargs['top_p'], label="Top p",
                                              info="Cumulative probability of tokens to sample from")
                            top_k = gr.Slider(
                                minimum=0, maximum=100, step=1,
                                value=kwargs['top_k'], label="Top k",
                                info='Num. tokens to sample from'
                            )
                            # FIXME: https://github.com/h2oai/h2ogpt/issues/106
                            if os.getenv('TESTINGFAIL'):
                                max_beams = 8 if not (memory_restriction_level or is_public) else 1
                            else:
                                max_beams = 1
                            num_beams = gr.Slider(minimum=1, maximum=max_beams, step=1,
                                                  value=min(max_beams, kwargs['num_beams']), label="Beams",
                                                  info="Number of searches for optimal overall probability.  "
                                                       "Uses more GPU memory/compute")
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
                                visible=False,
                            )
                            min_new_tokens2 = gr.Slider(
                                minimum=0, maximum=max_max_new_tokens, step=1,
                                value=min(max_max_new_tokens, kwargs['min_new_tokens']), label="Min output length 2",
                                visible=False,
                            )
                            early_stopping = gr.Checkbox(label="EarlyStopping", info="Stop early in beam search",
                                                         value=kwargs['early_stopping'])
                            max_max_time = 60 * 20 if not is_public else 60 * 2
                            if is_hf:
                                max_max_time = min(max_max_time, 60 * 1)
                            max_time = gr.Slider(minimum=0, maximum=max_max_time, step=1,
                                                 value=min(max_max_time, kwargs['max_time']), label="Max. time",
                                                 info="Max. time to search optimal output.")
                            repetition_penalty = gr.Slider(minimum=0.01, maximum=3.0,
                                                           value=kwargs['repetition_penalty'],
                                                           label="Repetition Penalty")
                            num_return_sequences = gr.Slider(minimum=1, maximum=10, step=1,
                                                             value=kwargs['num_return_sequences'],
                                                             label="Number Returns", info="Must be <= num_beams",
                                                             visible=not is_public)
                            iinput = gr.Textbox(lines=4, label="Input",
                                                placeholder=kwargs['placeholder_input'],
                                                visible=not is_public)
                            context = gr.Textbox(lines=3, label="System Pre-Context",
                                                 info="Directly pre-appended without prompt processing",
                                                 visible=not is_public)
                            chat = gr.components.Checkbox(label="Chat mode", value=kwargs['chat'],
                                                          visible=not is_public)
                            count_chat_tokens_btn = gr.Button(value="Count Chat Tokens", visible=not is_public)
                            chat_token_count = gr.Textbox(label="Chat Token Count", value=None,
                                                          visible=not is_public, interactive=False)
                            chunk = gr.components.Checkbox(value=kwargs['chunk'],
                                                           label="Whether to chunk documents",
                                                           info="For LangChain",
                                                           visible=not is_public)
                            top_k_docs = gr.Slider(minimum=0, maximum=100, step=1,
                                                   value=kwargs['top_k_docs'],
                                                   label="Number of document chunks",
                                                   info="For LangChain",
                                                   visible=not is_public)
                            chunk_size = gr.Number(value=kwargs['chunk_size'],
                                                   label="Chunk size for document chunking",
                                                   info="For LangChain (ignored if chunk=False)",
                                                   visible=not is_public,
                                                   precision=0)

                with gr.TabItem("Models"):
                    load_msg = "Load-Unload Model/LORA [unload works if did not use --base_model]" if not is_public \
                        else "LOAD-UNLOAD DISABLED FOR HOSTED DEMO"
                    load_msg2 = "Load-Unload Model/LORA 2 [unload works if did not use --base_model]" if not is_public \
                        else "LOAD-UNLOAD DISABLED FOR HOSTED DEMO 2"
                    compare_checkbox = gr.components.Checkbox(label="Compare Mode",
                                                              value=False, visible=not is_public)
                    with gr.Row():
                        n_gpus_list = [str(x) for x in list(range(-1, n_gpus))]
                        with gr.Column():
                            with gr.Row():
                                with gr.Column(scale=20):
                                    model_choice = gr.Dropdown(model_options_state.value[0], label="Choose Model",
                                                               value=kwargs['base_model'])
                                    lora_choice = gr.Dropdown(lora_options_state.value[0], label="Choose LORA",
                                                              value=kwargs['lora_weights'], visible=kwargs['show_lora'])
                                with gr.Column(scale=1):
                                    load_model_button = gr.Button(load_msg).style(full_width=False, size='sm')
                                    model_load8bit_checkbox = gr.components.Checkbox(
                                        label="Load 8-bit [requires support]",
                                        value=kwargs['load_8bit'])
                                    model_infer_devices_checkbox = gr.components.Checkbox(
                                        label="Choose Devices [If not Checked, use all GPUs]",
                                        value=kwargs['infer_devices'])
                                    model_gpu = gr.Dropdown(n_gpus_list,
                                                            label="GPU ID [-1 = all GPUs, if Choose is enabled]",
                                                            value=kwargs['gpu_id'])
                                    model_used = gr.Textbox(label="Current Model", value=kwargs['base_model'],
                                                            interactive=False)
                                    lora_used = gr.Textbox(label="Current LORA", value=kwargs['lora_weights'],
                                                           visible=kwargs['show_lora'], interactive=False)
                                    prompt_dict = gr.Textbox(label="Prompt (or Custom)",
                                                             value=pprint.pformat(kwargs['prompt_dict'], indent=4),
                                                             interactive=True, lines=4)
                        col_model2 = gr.Column(visible=False)
                        with col_model2:
                            with gr.Row():
                                with gr.Column(scale=20):
                                    model_choice2 = gr.Dropdown(model_options_state.value[0], label="Choose Model 2",
                                                                value=no_model_str)
                                    lora_choice2 = gr.Dropdown(lora_options_state.value[0], label="Choose LORA 2",
                                                               value=no_lora_str,
                                                               visible=kwargs['show_lora'])
                                with gr.Column(scale=1):
                                    load_model_button2 = gr.Button(load_msg2).style(full_width=False, size='sm')
                                    model_load8bit_checkbox2 = gr.components.Checkbox(
                                        label="Load 8-bit 2 [requires support]",
                                        value=kwargs['load_8bit'])
                                    model_infer_devices_checkbox2 = gr.components.Checkbox(
                                        label="Choose Devices 2 [If not Checked, use all GPUs]",
                                        value=kwargs[
                                            'infer_devices'])
                                    model_gpu2 = gr.Dropdown(n_gpus_list,
                                                             label="GPU ID 2 [-1 = all GPUs, if choose is enabled]",
                                                             value=kwargs['gpu_id'])
                                    # no model/lora loaded ever in model2 by default
                                    model_used2 = gr.Textbox(label="Current Model 2", value=no_model_str)
                                    lora_used2 = gr.Textbox(label="Current LORA 2", value=no_lora_str,
                                                            visible=kwargs['show_lora'])
                                    prompt_dict2 = gr.Textbox(label="Prompt (or Custom) 2",
                                                              value=pprint.pformat(kwargs['prompt_dict'], indent=4),
                                                              interactive=True, lines=4)
                    with gr.Row():
                        with gr.Column(scale=50):
                            new_model = gr.Textbox(label="New Model HF name/path")
                        with gr.Row():
                            add_model_button = gr.Button("Add new model name").style(full_width=False, size='sm')
                        with gr.Column(scale=50):
                            new_lora = gr.Textbox(label="New LORA HF name/path", visible=kwargs['show_lora'])
                        with gr.Row():
                            add_lora_button = gr.Button("Add new LORA name", visible=kwargs['show_lora']).style(
                                full_width=False, size='sm')
                with gr.TabItem("System"):
                    admin_row = gr.Row()
                    with admin_row:
                        admin_pass_textbox = gr.Textbox(label="Admin Password", type='password', visible=is_public)
                        admin_btn = gr.Button(value="Admin Access", visible=is_public)
                    system_row = gr.Row(visible=not is_public)
                    with system_row:
                        with gr.Column():
                            with gr.Row():
                                system_btn = gr.Button(value='Get System Info')
                                system_text = gr.Textbox(label='System Info', interactive=False).style(
                                    show_copy_button=True)

                            with gr.Row():
                                zip_btn = gr.Button("Zip")
                                zip_text = gr.Textbox(label="Zip file name", interactive=False)
                                file_output = gr.File(interactive=False, label="Zip file to Download")
                            with gr.Row():
                                s3up_btn = gr.Button("S3UP")
                                s3up_text = gr.Textbox(label='S3UP result', interactive=False)
                with gr.TabItem("Disclaimers"):
                    description = ""
                    description += """<p><b> DISCLAIMERS: </b><ul><i><li>The model was trained on The Pile and other data, which may contain objectionable content.  Use at own risk.</i></li>"""
                    if kwargs['load_8bit']:
                        description += """<i><li> Model is loaded in 8-bit and has other restrictions on this host. UX can be worse than non-hosted version.</i></li>"""
                    description += """<i><li>Conversations may be used to improve h2oGPT.  Do not share sensitive information.</i></li>"""
                    if 'h2ogpt-research' in kwargs['base_model']:
                        description += """<i><li>Research demonstration only, not used for commercial purposes.</i></li>"""
                    description += """<i><li>By using h2oGPT, you accept our <a href="https://github.com/h2oai/h2ogpt/blob/main/docs/tos.md">Terms of Service</a></i></li></ul></p>"""
                    gr.Markdown(value=description, show_label=False, interactive=False)

        # Get flagged data
        zip_data1 = functools.partial(zip_data, root_dirs=['flagged_data_points', kwargs['save_dir']])
        zip_btn.click(zip_data1, inputs=None, outputs=[file_output, zip_text], queue=False,
                      api_name='zip_data' if allow_api else None)
        s3up_btn.click(s3up, inputs=zip_text, outputs=s3up_text, queue=False,
                       api_name='s3up_data' if allow_api else None)

        def make_add_visible(x):
            return gr.update(visible=x is not None)

        def clear_file_list():
            return None

        def make_invisible():
            return gr.update(visible=False)

        def make_visible():
            return gr.update(visible=True)

        def update_radio_to_user():
            return gr.update(value='UserData')

        # Add to UserData
        update_user_db_func = functools.partial(update_user_db,
                                                dbs=dbs, db_type=db_type, langchain_mode='UserData',
                                                use_openai_embedding=use_openai_embedding,
                                                hf_embedding_model=hf_embedding_model,
                                                enable_captions=enable_captions,
                                                captions_model=captions_model,
                                                enable_ocr=enable_ocr,
                                                caption_loader=caption_loader,
                                                verbose=kwargs['verbose'],
                                                )

        # note for update_user_db_func output is ignored for db
        add_to_shared_db_btn.click(update_user_db_func,
                                   inputs=[fileup_output, my_db_state, add_to_shared_db_btn, add_to_my_db_btn,
                                           chunk, chunk_size],
                                   outputs=[add_to_shared_db_btn, add_to_my_db_btn, sources_text], queue=queue,
                                   api_name='add_to_shared' if allow_api else None) \
            .then(clear_file_list, outputs=fileup_output, queue=queue) \
            .then(update_radio_to_user, inputs=None, outputs=langchain_mode, queue=False)

        # .then(make_invisible, outputs=add_to_shared_db_btn, queue=queue)
        # .then(make_visible, outputs=upload_button, queue=queue)

        def clear_textbox():
            return gr.Textbox.update(value='')

        update_user_db_url_func = functools.partial(update_user_db_func, is_url=True)
        url_user_btn.click(update_user_db_url_func,
                           inputs=[url_text, my_db_state, add_to_shared_db_btn, add_to_my_db_btn,
                                   chunk, chunk_size],
                           outputs=[add_to_shared_db_btn, add_to_my_db_btn, sources_text], queue=queue,
                           api_name='add_url_to_shared' if allow_api else None) \
            .then(clear_textbox, outputs=url_text, queue=queue) \
            .then(update_radio_to_user, inputs=None, outputs=langchain_mode, queue=False)

        update_user_db_txt_func = functools.partial(update_user_db_func, is_txt=True)
        user_text_user_btn.click(update_user_db_txt_func,
                                 inputs=[user_text_text, my_db_state, add_to_shared_db_btn, add_to_my_db_btn,
                                         chunk, chunk_size],
                                 outputs=[add_to_shared_db_btn, add_to_my_db_btn, sources_text], queue=queue,
                                 api_name='add_text_to_shared' if allow_api else None) \
            .then(clear_textbox, outputs=user_text_text, queue=queue) \
            .then(update_radio_to_user, inputs=None, outputs=langchain_mode, queue=False)

        # Add to MyData
        def update_radio_to_my():
            return gr.update(value='MyData')

        update_my_db_func = functools.partial(update_user_db, dbs=dbs, db_type=db_type, langchain_mode='MyData',
                                              use_openai_embedding=use_openai_embedding,
                                              hf_embedding_model=hf_embedding_model,
                                              enable_captions=enable_captions,
                                              captions_model=captions_model,
                                              enable_ocr=enable_ocr,
                                              caption_loader=caption_loader,
                                              verbose=kwargs['verbose'],
                                              )

        add_to_my_db_btn.click(update_my_db_func,
                               inputs=[fileup_output, my_db_state, add_to_shared_db_btn, add_to_my_db_btn,
                                       chunk, chunk_size],
                               outputs=[my_db_state, add_to_shared_db_btn, add_to_my_db_btn, sources_text], queue=queue,
                               api_name='add_to_my' if allow_api else None) \
            .then(clear_file_list, outputs=fileup_output, queue=queue) \
            .then(update_radio_to_my, inputs=None, outputs=langchain_mode, queue=False)
        # .then(make_invisible, outputs=add_to_shared_db_btn, queue=queue)
        # .then(make_visible, outputs=upload_button, queue=queue)

        update_my_db_url_func = functools.partial(update_my_db_func, is_url=True)
        url_my_btn.click(update_my_db_url_func,
                         inputs=[url_text, my_db_state, add_to_shared_db_btn, add_to_my_db_btn,
                                 chunk, chunk_size],
                         outputs=[my_db_state, add_to_shared_db_btn, add_to_my_db_btn, sources_text], queue=queue,
                         api_name='add_url_to_my' if allow_api else None) \
            .then(clear_textbox, outputs=url_text, queue=queue) \
            .then(update_radio_to_my, inputs=None, outputs=langchain_mode, queue=False)

        update_my_db_txt_func = functools.partial(update_my_db_func, is_txt=True)
        user_text_my_btn.click(update_my_db_txt_func,
                               inputs=[user_text_text, my_db_state, add_to_shared_db_btn, add_to_my_db_btn,
                                       chunk, chunk_size],
                               outputs=[my_db_state, add_to_shared_db_btn, add_to_my_db_btn, sources_text], queue=queue,
                               api_name='add_txt_to_my' if allow_api else None) \
            .then(clear_textbox, outputs=user_text_text, queue=queue) \
            .then(update_radio_to_my, inputs=None, outputs=langchain_mode, queue=False)

        get_sources1 = functools.partial(get_sources, dbs=dbs, docs_state0=docs_state0)

        # if change collection source, must clear doc selections from it to avoid inconsistency
        def clear_doc_choice():
            return gr.Dropdown.update(choices=docs_state0, value=[docs_state0[0]])

        langchain_mode.change(clear_doc_choice, inputs=None, outputs=document_choice)

        def update_dropdown(x):
            return gr.Dropdown.update(choices=x, value=[docs_state0[0]])

        get_sources_btn.click(get_sources1, inputs=[my_db_state, langchain_mode], outputs=[file_source, docs_state],
                              queue=queue,
                              api_name='get_sources' if allow_api else None) \
            .then(fn=update_dropdown, inputs=docs_state, outputs=document_choice)
        # show button, else only show when add.  Could add to above get_sources for download/dropdown, but bit much maybe
        show_sources1 = functools.partial(get_source_files_given_langchain_mode, dbs=dbs)
        show_sources_btn.click(fn=show_sources1, inputs=[my_db_state, langchain_mode], outputs=sources_text,
                               api_name='show_sources' if allow_api else None)

        # Get inputs to evaluate() and make_db()
        # don't deepcopy, can contain model itself
        all_kwargs = kwargs.copy()
        all_kwargs.update(locals())

        refresh_sources1 = functools.partial(update_and_get_source_files_given_langchain_mode,
                                             **get_kwargs(update_and_get_source_files_given_langchain_mode,
                                                          exclude_names=['db1', 'langchain_mode'],
                                                          **all_kwargs))
        refresh_sources_btn.click(fn=refresh_sources1, inputs=[my_db_state, langchain_mode], outputs=sources_text,
                                  api_name='refresh_sources' if allow_api else None)

        def check_admin_pass(x):
            return gr.update(visible=x == admin_pass)

        def close_admin(x):
            return gr.update(visible=not (x == admin_pass))

        admin_btn.click(check_admin_pass, inputs=admin_pass_textbox, outputs=system_row, queue=False) \
            .then(close_admin, inputs=admin_pass_textbox, outputs=admin_row, queue=False)

        inputs_list, inputs_dict = get_inputs_list(all_kwargs, kwargs['model_lower'], model_id=1)
        inputs_list2, inputs_dict2 = get_inputs_list(all_kwargs, kwargs['model_lower'], model_id=2)
        from functools import partial
        kwargs_evaluate = {k: v for k, v in all_kwargs.items() if k in inputs_kwargs_list}
        # ensure present
        for k in inputs_kwargs_list:
            assert k in kwargs_evaluate, "Missing %s" % k

        def evaluate_gradio(*args1, **kwargs1):
            for res_dict in evaluate(*args1, **kwargs1):
                yield '<br>' + res_dict['response'].replace("\n", "<br>")

        fun = partial(evaluate_gradio,
                      **kwargs_evaluate)
        fun2 = partial(evaluate_gradio,
                       **kwargs_evaluate)
        fun_with_dict_str = partial(evaluate_from_str,
                                    default_kwargs=default_kwargs,
                                    **kwargs_evaluate
                                    )

        dark_mode_btn = gr.Button("Dark Mode", variant="primary").style(
            size="sm",
        )
        # FIXME: Could add exceptions for non-chat but still streaming
        exception_text = gr.Textbox(value="", visible=kwargs['chat'], label='Chat Exceptions', interactive=False)
        dark_mode_btn.click(
            None,
            None,
            None,
            _js=get_dark_js(),
            api_name="dark" if allow_api else None,
            queue=False,
        )

        # Control chat and non-chat blocks, which can be independently used by chat checkbox swap
        def col_nochat_fun(x):
            return gr.Column.update(visible=not x)

        def col_chat_fun(x):
            return gr.Column.update(visible=x)

        def context_fun(x):
            return gr.Textbox.update(visible=not x)

        chat.select(col_nochat_fun, chat, col_nochat, api_name="chat_checkbox" if allow_api else None) \
            .then(col_chat_fun, chat, col_chat) \
            .then(context_fun, chat, context) \
            .then(col_chat_fun, chat, exception_text)

        # examples after submit or any other buttons for chat or no chat
        if kwargs['examples'] is not None and kwargs['show_examples']:
            gr.Examples(examples=kwargs['examples'], inputs=inputs_list)

        # Score
        def score_last_response(*args, nochat=False, model2=False):
            """ Similar to user() """
            args_list = list(args)

            if memory_restriction_level > 0:
                max_length_tokenize = 768 - 256 if memory_restriction_level <= 2 else 512 - 256
            else:
                max_length_tokenize = 2048 - 256
            cutoff_len = max_length_tokenize * 4  # restrict deberta related to max for LLM
            smodel = score_model_state0[0]
            stokenizer = score_model_state0[1]
            sdevice = score_model_state0[2]
            if not nochat:
                history = args_list[-1]
                if history is None:
                    if not model2:
                        # maybe only doing first model, no need to complain
                        print("Bad history in scoring last response, fix for now", flush=True)
                    history = []
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
                    return 'Response Score: NA'
            else:
                answer = args_list[-1]
                instruction_nochat_arg_id = eval_func_param_names.index('instruction_nochat')
                question = args_list[instruction_nochat_arg_id]

            if question is None:
                return 'Response Score: Bad Question'
            if answer is None:
                return 'Response Score: Bad Answer'
            score = score_qa(smodel, stokenizer, max_length_tokenize, question, answer, cutoff_len)
            if isinstance(score, str):
                return 'Response Score: NA'
            return 'Response Score: {:.1%}'.format(score)

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
        score_args2 = dict(fn=partial(score_fun, model2=True),
                           inputs=inputs_list2 + [text_output2],
                           outputs=[score_text2],
                           )

        score_args_nochat = dict(fn=partial(score_fun, nochat=True),
                                 inputs=inputs_list + [text_output_nochat],
                                 outputs=[score_text_nochat],
                                 )
        if not kwargs['auto_score']:
            score_event = score_btn.click(**score_args, queue=queue, api_name='score' if allow_api else None) \
                .then(**score_args2, queue=queue, api_name='score2' if allow_api else None)
            score_event_nochat = score_btn_nochat.click(**score_args_nochat, queue=queue,
                                                        api_name='score_nochat' if allow_api else None)

        def user(*args, undo=False, sanitize_user_prompt=True, model2=False):
            """
            User that fills history for bot
            :param args:
            :param undo:
            :param sanitize_user_prompt:
            :param model2:
            :return:
            """
            args_list = list(args)
            user_message = args_list[eval_func_param_names.index('instruction')]  # chat only
            input1 = args_list[eval_func_param_names.index('iinput')]  # chat only
            context1 = args_list[eval_func_param_names.index('context')]
            prompt_type1 = args_list[eval_func_param_names.index('prompt_type')]
            prompt_dict1 = args_list[eval_func_param_names.index('prompt_dict')]
            chat1 = args_list[eval_func_param_names.index('chat')]
            stream_output1 = args_list[eval_func_param_names.index('stream_output')]
            if input1 and not user_message.endswith(':'):
                user_message1 = user_message + ":" + input1
            elif input1:
                user_message1 = user_message + input1
            else:
                user_message1 = user_message
            if sanitize_user_prompt:
                from better_profanity import profanity
                user_message1 = profanity.censor(user_message1)
            # FIXME: WIP to use desired seperator when user enters nothing
            prompter = Prompter(prompt_type1, prompt_dict1, debug=kwargs['debug'], chat=chat1,
                                stream_output=stream_output1)
            if user_message1 in ['']:
                # e.g. when user just hits enter in textbox,
                # else will have <human>: <bot>: on single line, which seems to be "ok" for LLM but not usual
                user_message1 = '\n'
            # ensure good visually, else markdown ignores multiple \n
            user_message1 = user_message1.replace('\n', '<br>')

            history = args_list[-1]
            if undo and history:
                history.pop()
            args_list = args_list[:-1]  # FYI, even if unused currently
            if history is None:
                if not model2:
                    # no need to complain so often unless model1
                    print("Bad history, fix for now", flush=True)
                history = []
            # ensure elements not mixed across models as output,
            # even if input is currently same source
            history = history.copy()
            if undo:
                return history
            else:
                # FIXME: compare, same history for now
                return history + [[user_message1, None]]

        def history_to_context(history, langchain_mode1, prompt_type1, prompt_dict1, chat1, model_max_length1):
            # ensure output will be unique to models
            _, _, _, max_prompt_length = get_cutoffs(memory_restriction_level,
                                                     for_context=True, model_max_length=model_max_length1)
            history = copy.deepcopy(history)

            context1 = ''
            if max_prompt_length is not None and langchain_mode1 not in ['LLM']:
                context1 = ''
                # - 1 below because current instruction already in history from user()
                for histi in range(0, len(history) - 1):
                    data_point = dict(instruction=history[histi][0], input='', output=history[histi][1])
                    prompt, pre_response, terminate_response, chat_sep = generate_prompt(data_point,
                                                                                         prompt_type1,
                                                                                         prompt_dict1,
                                                                                         chat1, reduced=True)
                    # md -> back to text, maybe not super important if model trained enough
                    if not kwargs['keep_sources_in_context']:
                        from gpt_langchain import source_prefix, source_postfix
                        import re
                        prompt = re.sub(f'{re.escape(source_prefix)}.*?{re.escape(source_postfix)}', '', prompt,
                                        flags=re.DOTALL)
                        if prompt.endswith('\n<p>'):
                            prompt = prompt[:-4]
                    prompt = prompt.replace('<br>', chat_sep)
                    if not prompt.endswith(chat_sep):
                        prompt += chat_sep
                    # most recent first, add older if can
                    # only include desired chat history
                    if len(prompt + context1) > max_prompt_length:
                        break
                    context1 = prompt + context1

                _, pre_response, terminate_response, chat_sep = generate_prompt({}, prompt_type1, prompt_dict1,
                                                                                chat1, reduced=True)
                if context1 and not context1.endswith(chat_sep):
                    context1 += chat_sep  # ensure if terminates abruptly, then human continues on next line
            return context1

        def get_model_max_length(model_state1):
            if model_state1 and not isinstance(model_state1[1], str):
                tokenizer = model_state1[1]
            elif model_state0 and not isinstance(model_state0[1], str):
                tokenizer = model_state0[1]
            else:
                tokenizer = None
            if tokenizer is not None:
                return tokenizer.model_max_length
            else:
                return 2000

        def bot(*args, retry=False):
            """
            bot that consumes history for user input
            instruction (from input_list) itself is not consumed by bot
            :param args:
            :param retry:
            :return:
            """
            # don't deepcopy, can contain model itself
            args_list = list(args).copy()
            model_state1 = args_list[-3]
            my_db_state1 = args_list[-2]
            history = args_list[-1]

            if model_state1[0] is None or model_state1[0] == no_model_str:
                history = []
                yield history, ''
                return

            args_list = args_list[:-3]  # only keep rest needed for evaluate()
            langchain_mode1 = args_list[eval_func_param_names.index('langchain_mode')]
            if retry and history:
                history.pop()
                if not args_list[eval_func_param_names.index('do_sample')]:
                    # if was not sampling, no point in retry unless change to sample
                    args_list[eval_func_param_names.index('do_sample')] = True
            if not history:
                print("No history", flush=True)
                history = []
                yield history, ''
                return
            instruction1 = history[-1][0]
            if not instruction1:
                # reject empty query, can sometimes go nuts
                history = []
                yield history, ''
                return
            prompt_type1 = args_list[eval_func_param_names.index('prompt_type')]
            prompt_dict1 = args_list[eval_func_param_names.index('prompt_dict')]
            chat1 = args_list[eval_func_param_names.index('chat')]
            model_max_length1 = get_model_max_length(model_state1)
            context1 = history_to_context(history, langchain_mode1, prompt_type1, prompt_dict1, chat1,
                                          model_max_length1)
            args_list[0] = instruction1  # override original instruction with history from user
            args_list[2] = context1
            fun1 = partial(evaluate,
                           model_state1,
                           my_db_state1,
                           **kwargs_evaluate)
            try:
                for output_fun in fun1(*tuple(args_list)):
                    output = output_fun['response']
                    extra = output_fun['sources']  # FIXME: can show sources in separate text box etc.
                    # ensure good visually, else markdown ignores multiple \n
                    bot_message = output.replace('\n', '<br>')
                    history[-1][1] = bot_message
                    yield history, ''
            except StopIteration:
                yield history, ''
            except RuntimeError as e:
                if "generator raised StopIteration" in str(e):
                    # assume last entry was bad, undo
                    history.pop()
                    yield history, ''
                else:
                    if history and len(history) > 0 and len(history[0]) > 1 and history[-1][1] is None:
                        history[-1][1] = ''
                    yield history, str(e)
                    raise
            except Exception as e:
                # put error into user input
                ex = "Exception: %s" % str(e)
                if history and len(history) > 0 and len(history[0]) > 1 and history[-1][1] is None:
                    history[-1][1] = ''
                yield history, ex
                raise
            return

        # NORMAL MODEL
        user_args = dict(fn=functools.partial(user, sanitize_user_prompt=kwargs['sanitize_user_prompt']),
                         inputs=inputs_list + [text_output],
                         outputs=text_output,
                         )
        bot_args = dict(fn=bot,
                        inputs=inputs_list + [model_state, my_db_state] + [text_output],
                        outputs=[text_output, exception_text],
                        )
        retry_bot_args = dict(fn=functools.partial(bot, retry=True),
                              inputs=inputs_list + [model_state, my_db_state] + [text_output],
                              outputs=[text_output, exception_text],
                              )
        undo_user_args = dict(fn=functools.partial(user, undo=True),
                              inputs=inputs_list + [text_output],
                              outputs=text_output,
                              )

        # MODEL2
        user_args2 = dict(fn=functools.partial(user, sanitize_user_prompt=kwargs['sanitize_user_prompt'], model2=True),
                          inputs=inputs_list2 + [text_output2],
                          outputs=text_output2,
                          )
        bot_args2 = dict(fn=bot,
                         inputs=inputs_list2 + [model_state2, my_db_state] + [text_output2],
                         outputs=[text_output2, exception_text],
                         )
        retry_bot_args2 = dict(fn=functools.partial(bot, retry=True),
                               inputs=inputs_list2 + [model_state2, my_db_state] + [text_output2],
                               outputs=[text_output2, exception_text],
                               )
        undo_user_args2 = dict(fn=functools.partial(user, undo=True),
                               inputs=inputs_list2 + [text_output2],
                               outputs=text_output2,
                               )

        def clear_instruct():
            return gr.Textbox.update(value='')

        if kwargs['auto_score']:
            score_args_submit = score_args
            score_args2_submit = score_args2
        else:
            score_args_submit = dict(fn=lambda: None, inputs=None, outputs=None)
            score_args2_submit = dict(fn=lambda: None, inputs=None, outputs=None)

        def deselect_radio_chats():
            return gr.update(value=None)

        # in case 2nd model, consume instruction first, so can clear quickly
        # bot doesn't consume instruction itself, just history from user, so why works
        submit_event1a = instruction.submit(**user_args, queue=queue,
                                            api_name='instruction' if allow_api else None)
        submit_event1b = submit_event1a.then(**user_args2, api_name='instruction2' if allow_api else None)
        submit_event1c = submit_event1b.then(clear_instruct, None, instruction) \
            .then(clear_instruct, None, iinput)
        submit_event1d = submit_event1c.then(**bot_args, api_name='instruction_bot' if allow_api else None,
                                             queue=queue)
        submit_event1e = submit_event1d.then(**score_args_submit,
                                             api_name='instruction_bot_score' if allow_api else None,
                                             queue=queue)
        submit_event1f = submit_event1e.then(**bot_args2, api_name='instruction_bot2' if allow_api else None,
                                             queue=queue)
        submit_event1g = submit_event1f.then(**score_args2_submit,
                                             api_name='instruction_bot_score2' if allow_api else None, queue=queue)
        submit_event1h = submit_event1g.then(clear_torch_cache)
        # if hit enter on new instruction for submitting new query, no longer the saved chat
        submit_event1i = submit_event1h.then(deselect_radio_chats, inputs=None, outputs=radio_chats, queue=False)

        submit_event2a = submit.click(**user_args, api_name='submit' if allow_api else None)
        submit_event2b = submit_event2a.then(**user_args2, api_name='submit2' if allow_api else None)
        submit_event2c = submit_event2b.then(clear_instruct, None, instruction) \
            .then(clear_instruct, None, iinput)
        submit_event2d = submit_event2c.then(**bot_args, api_name='submit_bot' if allow_api else None, queue=queue)
        submit_event2e = submit_event2d.then(**score_args_submit, api_name='submit_bot_score' if allow_api else None,
                                             queue=queue)
        submit_event2f = submit_event2e.then(**bot_args2, api_name='submit_bot2' if allow_api else None, queue=queue)
        submit_event2g = submit_event2f.then(**score_args2_submit, api_name='submit_bot_score2' if allow_api else None,
                                             queue=queue)
        submit_event2h = submit_event2g.then(clear_torch_cache)
        # if submit new query, no longer the saved chat
        submit_event2i = submit_event2h.then(deselect_radio_chats, inputs=None, outputs=radio_chats, queue=False)

        submit_event3a = retry.click(**user_args, api_name='retry' if allow_api else None)
        submit_event3b = submit_event3a.then(**user_args2, api_name='retry2' if allow_api else None)
        submit_event3c = submit_event3b.then(clear_instruct, None, instruction) \
            .then(clear_instruct, None, iinput)
        submit_event3d = submit_event3c.then(**retry_bot_args, api_name='retry_bot' if allow_api else None,
                                             queue=queue)
        submit_event3e = submit_event3d.then(**score_args_submit, api_name='retry_bot_score' if allow_api else None,
                                             queue=queue)
        submit_event3f = submit_event3e.then(**retry_bot_args2, api_name='retry_bot2' if allow_api else None,
                                             queue=queue)
        submit_event3g = submit_event3f.then(**score_args2_submit, api_name='retry_bot_score2' if allow_api else None,
                                             queue=queue)
        submit_event3h = submit_event3g.then(clear_torch_cache)
        # if retry, no longer the saved chat
        submit_event3i = submit_event3h.then(deselect_radio_chats, inputs=None, outputs=radio_chats, queue=False)

        submit_event4 = undo.click(**undo_user_args, api_name='undo' if allow_api else None) \
            .then(**undo_user_args2, api_name='undo2' if allow_api else None) \
            .then(clear_instruct, None, instruction) \
            .then(clear_instruct, None, iinput) \
            .then(**score_args_submit, api_name='undo_score' if allow_api else None) \
            .then(**score_args2_submit, api_name='undo_score2' if allow_api else None) \
            .then(deselect_radio_chats, inputs=None, outputs=radio_chats, queue=False)  # if undo, no longer the saved chat

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
            for stepx, stepy in zip(x, y):
                if len(stepx) != len(stepy):
                    # something off with a conversation
                    return False
                if len(stepx) != 2:
                    # something off
                    return False
                if len(stepy) != 2:
                    # something off
                    return False
                questionx = stepx[0].replace('<p>', '').replace('</p>', '') if stepx[0] is not None else None
                answerx = stepx[1].replace('<p>', '').replace('</p>', '') if stepx[1] is not None else None

                questiony = stepy[0].replace('<p>', '').replace('</p>', '') if stepy[0] is not None else None
                answery = stepy[1].replace('<p>', '').replace('</p>', '') if stepy[1] is not None else None

                if questionx != questiony or answerx != answery:
                    return False
            return is_same

        def save_chat(chat1, chat2, chat_state1):
            short_chats = list(chat_state1.keys())
            for chati in [chat1, chat2]:
                if chati and len(chati) > 0 and len(chati[0]) == 2 and chati[0][1] is not None:
                    short_chat = get_short_chat(chati, short_chats)
                    if short_chat:
                        already_exists = any([is_chat_same(chati, x) for x in chat_state1.values()])
                        if not already_exists:
                            chat_state1[short_chat] = chati
            return chat_state1

        def update_radio_chats(chat_state1):
            return gr.update(choices=list(chat_state1.keys()), value=None)

        def switch_chat(chat_key, chat_state1):
            chosen_chat = chat_state1[chat_key]
            return chosen_chat, chosen_chat

        radio_chats.input(switch_chat, inputs=[radio_chats, chat_state], outputs=[text_output, text_output2])

        def remove_chat(chat_key, chat_state1):
            chat_state1.pop(chat_key, None)
            return chat_state1

        remove_chat_btn.click(remove_chat, inputs=[radio_chats, chat_state], outputs=chat_state) \
            .then(update_radio_chats, inputs=chat_state, outputs=radio_chats)

        def get_chats1(chat_state1):
            base = 'chats'
            makedirs(base, exist_ok=True)
            filename = os.path.join(base, 'chats_%s.json' % str(uuid.uuid4()))
            with open(filename, "wt") as f:
                f.write(json.dumps(chat_state1, indent=2))
            return filename

        export_chats_btn.click(get_chats1, inputs=chat_state, outputs=chats_file, queue=False,
                               api_name='export_chats' if allow_api else None)

        def add_chats_from_file(file, chat_state1, add_btn):
            if not file:
                return chat_state1, add_btn
            if isinstance(file, str):
                files = [file]
            else:
                files = file
            if not files:
                return chat_state1, add_btn
            for file1 in files:
                try:
                    if hasattr(file1, 'name'):
                        file1 = file1.name
                    with open(file1, "rt") as f:
                        new_chats = json.loads(f.read())
                        for chat1_k, chat1_v in new_chats.items():
                            # ignore chat1_k, regenerate and de-dup to avoid loss
                            chat_state1 = save_chat(chat1_v, None, chat_state1)
                except BaseException as e:
                    print("Add chats exception: %s" % str(e), flush=True)
            return chat_state1, add_btn

        # note for update_user_db_func output is ignored for db
        add_to_chats_btn.click(add_chats_from_file,
                               inputs=[chatsup_output, chat_state, add_to_chats_btn],
                               outputs=[chat_state, add_to_my_db_btn], queue=False,
                               api_name='add_to_chats' if allow_api else None) \
            .then(clear_file_list, outputs=chatsup_output, queue=False) \
            .then(update_radio_chats, inputs=chat_state, outputs=radio_chats, queue=False)

        clear_chat_btn.click(lambda: None, None, text_output, queue=False, api_name='clear' if allow_api else None) \
            .then(lambda: None, None, text_output2, queue=False, api_name='clear2' if allow_api else None) \
            .then(deselect_radio_chats, inputs=None, outputs=radio_chats, queue=False)

        # does both models
        clear.click(save_chat, inputs=[text_output, text_output2, chat_state], outputs=chat_state,
                    api_name='save_chat' if allow_api else None) \
            .then(update_radio_chats, inputs=chat_state, outputs=radio_chats,
                  api_name='update_chats' if allow_api else None) \
            .then(lambda: None, None, text_output, queue=False, api_name='clearB' if allow_api else None) \
            .then(lambda: None, None, text_output2, queue=False, api_name='clearB2' if allow_api else None)
        # NOTE: clear of instruction/iinput for nochat has to come after score,
        # because score for nochat consumes actual textbox, while chat consumes chat history filled by user()
        no_chat_args = dict(fn=fun,
                            inputs=[model_state, my_db_state] + inputs_list,
                            outputs=text_output_nochat,
                            queue=queue,
                            )
        submit_event_nochat = submit_nochat.click(**no_chat_args, api_name='submit_nochat' if allow_api else None) \
            .then(**score_args_nochat, api_name='instruction_bot_score_nochat' if allow_api else None, queue=queue) \
            .then(clear_instruct, None, instruction_nochat) \
            .then(clear_instruct, None, iinput_nochat) \
            .then(clear_torch_cache)
        # copy of above with text box submission
        submit_event_nochat2 = instruction_nochat.submit(**no_chat_args) \
            .then(**score_args_nochat, queue=queue) \
            .then(clear_instruct, None, instruction_nochat) \
            .then(clear_instruct, None, iinput_nochat) \
            .then(clear_torch_cache)

        submit_event_nochat_api = submit_nochat_api.click(fun_with_dict_str,
                                                          inputs=[model_state, my_db_state, inputs_dict_str],
                                                          outputs=text_output_nochat_api,
                                                          queue=True,  # required for generator
                                                          api_name='submit_nochat_api' if allow_api else None) \
            .then(clear_torch_cache)

        def load_model(model_name, lora_weights, model_state_old, prompt_type_old, load_8bit, infer_devices, gpu_id):
            # ensure old model removed from GPU memory
            if kwargs['debug']:
                print("Pre-switch pre-del GPU memory: %s" % get_torch_allocated(), flush=True)

            model0 = model_state0[0]
            if isinstance(model_state_old[0], str) and model0 is not None:
                # best can do, move model loaded at first to CPU
                model0.cpu()

            if model_state_old[0] is not None and not isinstance(model_state_old[0], str):
                try:
                    model_state_old[0].cpu()
                except Exception as e:
                    # sometimes hit NotImplementedError: Cannot copy out of meta tensor; no data!
                    print("Unable to put model on CPU: %s" % str(e), flush=True)
                del model_state_old[0]
                model_state_old[0] = None

            if model_state_old[1] is not None and not isinstance(model_state_old[1], str):
                del model_state_old[1]
                model_state_old[1] = None

            clear_torch_cache()
            if kwargs['debug']:
                print("Pre-switch post-del GPU memory: %s" % get_torch_allocated(), flush=True)

            if model_name is None or model_name == no_model_str:
                # no-op if no model, just free memory
                # no detranscribe needed for model, never go into evaluate
                lora_weights = no_lora_str
                return [None, None, None, model_name], model_name, lora_weights, prompt_type_old

            # don't deepcopy, can contain model itself
            all_kwargs1 = all_kwargs.copy()
            all_kwargs1['base_model'] = model_name.strip()
            all_kwargs1['load_8bit'] = load_8bit
            all_kwargs1['infer_devices'] = infer_devices
            all_kwargs1['gpu_id'] = int(gpu_id)  # detranscribe
            model_lower = model_name.strip().lower()
            if model_lower in inv_prompt_type_to_model_lower:
                prompt_type1 = inv_prompt_type_to_model_lower[model_lower]
            else:
                prompt_type1 = prompt_type_old

            # detranscribe
            if lora_weights == no_lora_str:
                lora_weights = ''

            all_kwargs1['lora_weights'] = lora_weights.strip()
            model1, tokenizer1, device1 = get_model(reward_type=False,
                                                    **get_kwargs(get_model, exclude_names=['reward_type'],
                                                                 **all_kwargs1))
            clear_torch_cache()

            model_state_new = [model1, tokenizer1, device1, model_name]

            max_max_new_tokens1 = get_max_max_new_tokens(model_state_new, **kwargs)

            if kwargs['debug']:
                print("Post-switch GPU memory: %s" % get_torch_allocated(), flush=True)
            return model_state_new, model_name, lora_weights, prompt_type1, \
                gr.Slider.update(maximum=max_max_new_tokens1), \
                gr.Slider.update(maximum=max_max_new_tokens1)

        def get_prompt_str(prompt_type1, prompt_dict1):
            prompt_dict1, prompt_dict_error = get_prompt(prompt_type1, prompt_dict1, chat=False, context='',
                                                         reduced=False, return_dict=True)
            if prompt_dict_error:
                return str(prompt_dict_error)
            else:
                # return so user can manipulate if want and use as custom
                return str(prompt_dict1)

        prompt_type.change(fn=get_prompt_str, inputs=[prompt_type, prompt_dict], outputs=prompt_dict)
        prompt_type2.change(fn=get_prompt_str, inputs=[prompt_type2, prompt_dict2], outputs=prompt_dict2)

        def dropdown_prompt_type_list(x):
            return gr.Dropdown.update(value=x)

        def chatbot_list(x, model_used_in):
            return gr.Textbox.update(label=f'h2oGPT [Model: {model_used_in}]')

        load_model_args = dict(fn=load_model,
                               inputs=[model_choice, lora_choice, model_state, prompt_type,
                                       model_load8bit_checkbox, model_infer_devices_checkbox, model_gpu],
                               outputs=[model_state, model_used, lora_used,
                                        # if prompt_type changes, prompt_dict will change via change rule
                                        prompt_type, max_new_tokens, min_new_tokens,
                                        ])
        prompt_update_args = dict(fn=dropdown_prompt_type_list, inputs=prompt_type, outputs=prompt_type)
        chatbot_update_args = dict(fn=chatbot_list, inputs=[text_output, model_used], outputs=text_output)
        nochat_update_args = dict(fn=chatbot_list, inputs=[text_output_nochat, model_used], outputs=text_output_nochat)
        if not is_public:
            load_model_event = load_model_button.click(**load_model_args, api_name='load_model' if allow_api else None) \
                .then(**prompt_update_args) \
                .then(**chatbot_update_args) \
                .then(**nochat_update_args) \
                .then(clear_torch_cache)

        load_model_args2 = dict(fn=load_model,
                                inputs=[model_choice2, lora_choice2, model_state2, prompt_type2,
                                        model_load8bit_checkbox2, model_infer_devices_checkbox2, model_gpu2],
                                outputs=[model_state2, model_used2, lora_used2,
                                         # if prompt_type2 changes, prompt_dict2 will change via change rule
                                         prompt_type2, max_new_tokens2, min_new_tokens2
                                         ])
        prompt_update_args2 = dict(fn=dropdown_prompt_type_list, inputs=prompt_type2, outputs=prompt_type2)
        chatbot_update_args2 = dict(fn=chatbot_list, inputs=[text_output2, model_used2], outputs=text_output2)
        if not is_public:
            load_model_event2 = load_model_button2.click(**load_model_args2,
                                                         api_name='load_model2' if allow_api else None) \
                .then(**prompt_update_args2) \
                .then(**chatbot_update_args2) \
                .then(clear_torch_cache)

        def dropdown_model_list(list0, x):
            new_state = [list0[0] + [x]]
            new_options = [*new_state[0]]
            return gr.Dropdown.update(value=x, choices=new_options), \
                gr.Dropdown.update(value=x, choices=new_options), \
                '', new_state

        add_model_event = add_model_button.click(fn=dropdown_model_list,
                                                 inputs=[model_options_state, new_model],
                                                 outputs=[model_choice, model_choice2, new_model, model_options_state],
                                                 queue=False)

        def dropdown_lora_list(list0, x, model_used1, lora_used1, model_used2, lora_used2):
            new_state = [list0[0] + [x]]
            new_options = [*new_state[0]]
            # don't switch drop-down to added lora if already have model loaded
            x1 = x if model_used1 == no_model_str else lora_used1
            x2 = x if model_used2 == no_model_str else lora_used2
            return gr.Dropdown.update(value=x1, choices=new_options), \
                gr.Dropdown.update(value=x2, choices=new_options), \
                '', new_state

        add_lora_event = add_lora_button.click(fn=dropdown_lora_list,
                                               inputs=[lora_options_state, new_lora, model_used, lora_used, model_used2,
                                                       lora_used2],
                                               outputs=[lora_choice, lora_choice2, new_lora, lora_options_state],
                                               queue=False)

        go_btn.click(lambda: gr.update(visible=False), None, go_btn, api_name="go" if allow_api else None, queue=False) \
            .then(lambda: gr.update(visible=True), None, normal_block, queue=False) \
            .then(**load_model_args, queue=False).then(**prompt_update_args, queue=False)

        def compare_textbox_fun(x):
            return gr.Textbox.update(visible=x)

        def compare_column_fun(x):
            return gr.Column.update(visible=x)

        def compare_prompt_fun(x):
            return gr.Dropdown.update(visible=x)

        def slider_fun(x):
            return gr.Slider.update(visible=x)

        compare_checkbox.select(compare_textbox_fun, compare_checkbox, text_output2,
                                api_name="compare_checkbox" if allow_api else None) \
            .then(compare_column_fun, compare_checkbox, col_model2) \
            .then(compare_prompt_fun, compare_checkbox, prompt_type2) \
            .then(compare_textbox_fun, compare_checkbox, score_text2) \
            .then(slider_fun, compare_checkbox, max_new_tokens2) \
            .then(slider_fun, compare_checkbox, min_new_tokens2)
        # FIXME: add score_res2 in condition, but do better

        # callback for logging flagged input/output
        callback.setup(inputs_list + [text_output, text_output2], "flagged_data_points")
        flag_btn.click(lambda *args: callback.flag(args), inputs_list + [text_output, text_output2], None,
                       preprocess=False,
                       api_name='flag' if allow_api else None, queue=False)
        flag_btn_nochat.click(lambda *args: callback.flag(args), inputs_list + [text_output_nochat], None,
                              preprocess=False,
                              api_name='flag_nochat' if allow_api else None, queue=False)

        def get_system_info():
            return gr.Textbox.update(value=system_info_print())

        system_event = system_btn.click(get_system_info, outputs=system_text,
                                        api_name='system_info' if allow_api else None, queue=False)

        # don't pass text_output, don't want to clear output, just stop it
        # cancel only stops outer generation, not inner generation or non-generation
        stop_btn.click(lambda: None, None, None,
                       cancels=[submit_event1d, submit_event1f,
                                submit_event2d, submit_event2f,
                                submit_event3d, submit_event3f,
                                submit_event_nochat,
                                submit_event_nochat2,
                                ],
                       queue=False, api_name='stop' if allow_api else None).then(clear_torch_cache, queue=False)

        def count_chat_tokens(model_state1, chat1, prompt_type1, prompt_dict1):
            if model_state1 and not isinstance(model_state1[1], str):
                tokenizer = model_state1[1]
            elif model_state0 and not isinstance(model_state0[1], str):
                tokenizer = model_state0[1]
            else:
                tokenizer = None
            if tokenizer is not None:
                langchain_mode1 = 'ChatLLM'
                # fake user message to mimic bot()
                chat1 = copy.deepcopy(chat1)
                chat1 = chat1 + [['user_message1', None]]
                model_max_length1 = tokenizer.model_max_length
                context1 = history_to_context(chat1, langchain_mode1, prompt_type1, prompt_dict1, chat1,
                                              model_max_length1)
                return str(tokenizer(context1, return_tensors="pt")['input_ids'].shape[1])
            else:
                return "N/A"

        count_chat_tokens_btn.click(fn=count_chat_tokens, inputs=[model_state, text_output, prompt_type, prompt_dict],
                                    outputs=chat_token_count, api_name='count_tokens' if allow_api else None)

        demo.load(None, None, None, _js=get_dark_js() if kwargs['h2ocolors'] else None)

    demo.queue(concurrency_count=kwargs['concurrency_count'], api_open=kwargs['api_open'])
    favicon_path = "h2o-logo.svg"

    scheduler = BackgroundScheduler()
    scheduler.add_job(func=clear_torch_cache, trigger="interval", seconds=20)
    if is_public and \
            kwargs['base_model'] not in non_hf_types:
        # FIXME: disable for gptj, langchain or gpt4all modify print itself
        # FIXME: and any multi-threaded/async print will enter model output!
        scheduler.add_job(func=ping, trigger="interval", seconds=60)
    scheduler.start()

    # import control
    if kwargs['langchain_mode'] == 'Disabled' and \
            os.environ.get("TEST_LANGCHAIN_IMPORT") and \
            kwargs['base_model'] not in non_hf_types:
        assert 'gpt_langchain' not in sys.modules, "Dev bug, import of langchain when should not have"
        assert 'langchain' not in sys.modules, "Dev bug, import of langchain when should not have"

    demo.launch(share=kwargs['share'], server_name="0.0.0.0", show_error=True,
                favicon_path=favicon_path, prevent_thread_lock=True,
                auth=kwargs['auth'])
    if kwargs['verbose']:
        print("Started GUI", flush=True)
    if kwargs['block_gradio_exit']:
        demo.block_thread()


input_args_list = ['model_state', 'my_db_state']


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


def get_sources(db1, langchain_mode, dbs=None, docs_state0=None):
    if langchain_mode in ['ChatLLM', 'LLM']:
        source_files_added = "NA"
        source_list = []
    elif langchain_mode in ['wiki_full']:
        source_files_added = "Not showing wiki_full, takes about 20 seconds and makes 4MB file." \
                             "  Ask jon.mckinney@h2o.ai for file if required."
        source_list = []
    elif langchain_mode == 'MyData' and len(db1) > 0 and db1[0] is not None:
        db_get = db1[0].get()
        source_list = sorted(set([x['source'] for x in db_get['metadatas']]))
        source_files_added = '\n'.join(source_list)
    elif langchain_mode in dbs and dbs[langchain_mode] is not None:
        db1 = dbs[langchain_mode]
        db_get = db1.get()
        source_list = sorted(set([x['source'] for x in db_get['metadatas']]))
        source_files_added = '\n'.join(source_list)
    else:
        source_list = []
        source_files_added = "None"
    sources_file = 'sources_%s_%s' % (langchain_mode, str(uuid.uuid4()))
    with open(sources_file, "wt") as f:
        f.write(source_files_added)
    source_list = docs_state0 + source_list
    return sources_file, source_list


def update_user_db(file, db1, x, y, *args, dbs=None, langchain_mode='UserData', **kwargs):
    try:
        return _update_user_db(file, db1, x, y, *args, dbs=dbs, langchain_mode=langchain_mode, **kwargs)
    except BaseException as e:
        print(traceback.format_exc(), flush=True)
        # gradio has issues if except, so fail semi-gracefully, else would hang forever in processing textbox
        ex_str = "Exception: %s" % str(e)
        source_files_added = """\
        <html>
          <body>
            <p>
               Sources: <br>
            </p>
               <div style="overflow-y: auto;height:400px">
               {0}
               </div>
          </body>
        </html>
        """.format(ex_str)
        if langchain_mode == 'MyData':
            return db1, x, y, source_files_added
        else:
            return x, y, source_files_added


def _update_user_db(file, db1, x, y, chunk, chunk_size, dbs=None, db_type=None, langchain_mode='UserData',
                    use_openai_embedding=None,
                    hf_embedding_model=None,
                    caption_loader=None,
                    enable_captions=None,
                    captions_model=None,
                    enable_ocr=None,
                    verbose=None,
                    is_url=None, is_txt=None):
    assert use_openai_embedding is not None
    assert hf_embedding_model is not None
    assert caption_loader is not None
    assert enable_captions is not None
    assert captions_model is not None
    assert enable_ocr is not None
    assert verbose is not None

    assert isinstance(dbs, dict), "Wrong type for dbs: %s" % str(type(dbs))
    assert db_type in ['faiss', 'chroma'], "db_type %s not supported" % db_type
    from gpt_langchain import add_to_db, get_db, path_to_docs
    # handle case of list of temp buffer
    if isinstance(file, list) and len(file) > 0 and hasattr(file[0], 'name'):
        file = [x.name for x in file]
    # handle single file of temp buffer
    if hasattr(file, 'name'):
        file = file.name
    if verbose:
        print("Adding %s" % file, flush=True)
    sources = path_to_docs(file if not is_url and not is_txt else None,
                           verbose=verbose,
                           chunk=chunk, chunk_size=chunk_size,
                           url=file if is_url else None,
                           text=file if is_txt else None,
                           enable_captions=enable_captions,
                           captions_model=captions_model,
                           enable_ocr=enable_ocr,
                           caption_loader=caption_loader,
                           )
    exceptions = [x for x in sources if x.metadata.get('exception')]
    sources = [x for x in sources if 'exception' not in x.metadata]

    with filelock.FileLock("db_%s.lock" % langchain_mode.replace(' ', '_')):
        if langchain_mode == 'MyData':
            if db1[0] is not None:
                # then add
                db, num_new_sources, new_sources_metadata = add_to_db(db1[0], sources, db_type=db_type)
            else:
                assert len(db1) == 2 and db1[1] is None, "Bad MyData db: %s" % db1
                # then create
                # assign fresh hash for this user session, so not shared
                # if added has to original state and didn't change, then would be shared db for all users
                db1[1] = str(uuid.uuid4())
                persist_directory = os.path.join(scratch_base_dir, 'db_dir_%s_%s' % (langchain_mode, db1[1]))
                db1[0] = get_db(sources, use_openai_embedding=use_openai_embedding,
                                db_type=db_type,
                                persist_directory=persist_directory,
                                langchain_mode=langchain_mode,
                                hf_embedding_model=hf_embedding_model)
                if db1[0] is None:
                    db1[1] = None
            source_files_added = get_source_files(db=db1[0], exceptions=exceptions)
            return db1, x, y, source_files_added
        else:
            persist_directory = 'db_dir_%s' % langchain_mode
            if langchain_mode in dbs and dbs[langchain_mode] is not None:
                # then add
                db, num_new_sources, new_sources_metadata = add_to_db(dbs[langchain_mode], sources, db_type=db_type)
            else:
                # then create
                db = get_db(sources, use_openai_embedding=use_openai_embedding,
                            db_type=db_type,
                            persist_directory=persist_directory,
                            langchain_mode=langchain_mode,
                            hf_embedding_model=hf_embedding_model)
                dbs[langchain_mode] = db
            # NOTE we do not return db, because function call always same code path
            # return dbs[langchain_mode], x, y
            # db in this code path is updated in place
            source_files_added = get_source_files(db=dbs[langchain_mode], exceptions=exceptions)
            return x, y, source_files_added


def get_db(db1, langchain_mode, dbs=None):
    with filelock.FileLock("db_%s.lock" % langchain_mode.replace(' ', '_')):
        if langchain_mode in ['wiki_full']:
            # NOTE: avoid showing full wiki.  Takes about 30 seconds over about 90k entries, but not useful for now
            db = None
        elif langchain_mode == 'MyData' and len(db1) > 0 and db1[0] is not None:
            db = db1[0]
        elif langchain_mode in dbs and dbs[langchain_mode] is not None:
            db = dbs[langchain_mode]
        else:
            db = None
    return db


def get_source_files_given_langchain_mode(db1, langchain_mode='UserData', dbs=None):
    db = get_db(db1, langchain_mode, dbs=dbs)
    return get_source_files(db=db, exceptions=None)


def get_source_files(db=None, exceptions=None, metadatas=None):
    if exceptions is None:
        exceptions = []

    # only should be one source, not confused
    assert db is not None or metadatas is not None

    if metadatas is None:
        source_label = "Sources:"
        if db is not None:
            metadatas = db.get()['metadatas']
        else:
            metadatas = []
        adding_new = False
    else:
        source_label = "New Sources:"
        adding_new = True

    # below automatically de-dups
    from gpt_langchain import get_url
    small_dict = {get_url(x['source'], from_str=True, short_name=True): get_short_name(x.get('head')) for x in
                  metadatas}
    # if small_dict is empty dict, that's ok
    df = pd.DataFrame(small_dict.items(), columns=['source', 'head'])
    df.index = df.index + 1
    df.index.name = 'index'
    source_files_added = tabulate.tabulate(df, headers='keys', tablefmt='unsafehtml')

    if exceptions:
        exception_metadatas = [x.metadata for x in exceptions]
        small_dict = {get_url(x['source'], from_str=True, short_name=True): get_short_name(x.get('exception')) for x in
                      exception_metadatas}
        # if small_dict is empty dict, that's ok
        df = pd.DataFrame(small_dict.items(), columns=['source', 'exception'])
        df.index = df.index + 1
        df.index.name = 'index'
        exceptions_html = tabulate.tabulate(df, headers='keys', tablefmt='unsafehtml')
    else:
        exceptions_html = ''

    if metadatas and exceptions:
        source_files_added = """\
        <html>
          <body>
            <p>
               {0} <br>
            </p>
               <div style="overflow-y: auto;height:400px">
               {1}
               {2}
               </div>
          </body>
        </html>
        """.format(source_label, source_files_added, exceptions_html)
    elif metadatas:
        source_files_added = """\
        <html>
          <body>
            <p>
               {0} <br>
            </p>
               <div style="overflow-y: auto;height:400px">
               {1}
               </div>
          </body>
        </html>
        """.format(source_label, source_files_added)
    elif exceptions_html:
        source_files_added = """\
        <html>
          <body>
            <p>
               Exceptions: <br>
            </p>
               <div style="overflow-y: auto;height:400px">
               {0}
               </div>
          </body>
        </html>
        """.format(exceptions_html)
    else:
        if adding_new:
            source_files_added = "No New Sources"
        else:
            source_files_added = "No Sources"

    return source_files_added


def update_and_get_source_files_given_langchain_mode(db1, langchain_mode, dbs=None, first_para=None,
                                                     text_limit=None, chunk=None, chunk_size=None,
                                                     user_path=None, db_type=None, load_db_if_exists=None,
                                                     n_jobs=None, verbose=None):
    db = get_db(db1, langchain_mode, dbs=dbs)

    from gpt_langchain import make_db
    db, num_new_sources, new_sources_metadata = make_db(use_openai_embedding=False,
                                                        hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                                                        first_para=first_para, text_limit=text_limit,
                                                        chunk=chunk,
                                                        chunk_size=chunk_size,
                                                        langchain_mode=langchain_mode,
                                                        user_path=user_path,
                                                        db_type=db_type,
                                                        load_db_if_exists=load_db_if_exists,
                                                        db=db,
                                                        n_jobs=n_jobs,
                                                        verbose=verbose)
    # return only new sources with text saying such
    return get_source_files(db=None, exceptions=None, metadatas=new_sources_metadata)
