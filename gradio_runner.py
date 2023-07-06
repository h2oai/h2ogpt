import copy
import functools
import inspect
import itertools
import json
import os
import pprint
import random
import shutil
import sys
import time
import traceback
import typing
import uuid
import filelock
import pandas as pd
import requests
import tabulate
from iterators import TimeoutIterator

from gradio_utils.css import get_css
from gradio_utils.prompt_form import make_prompt_form, make_chatbots

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

from enums import DocumentChoices, no_model_str, no_lora_str, no_server_str, LangChainMode
from gradio_themes import H2oTheme, SoftTheme, get_h2o_title, get_simple_title, get_dark_js, spacing_xsm, radius_xsm, \
    text_xsm
from prompter import prompt_type_to_model_name, prompt_types_strings, inv_prompt_type_to_model_lower, non_hf_types, \
    get_prompt
from utils import get_githash, flatten_list, zip_data, s3up, clear_torch_cache, get_torch_allocated, system_info_print, \
    ping, get_short_name, get_url, makedirs, get_kwargs, remove, system_info, ping_gpu
from generate import get_model, languages_covered, evaluate, eval_func_param_names, score_qa, langchain_modes, \
    inputs_kwargs_list, scratch_base_dir, evaluate_from_str, no_default_param_names, \
    eval_func_param_names_defaults, get_max_max_new_tokens, get_minmax_top_k_docs, history_to_context

from apscheduler.schedulers.background import BackgroundScheduler


def fix_text_for_gradio(text, fix_new_lines=False, fix_latex_dollars=True):
    if fix_latex_dollars:
        ts = text.split('```')
        for parti, part in enumerate(ts):
            inside = parti % 2 == 1
            if not inside:
                ts[parti] = ts[parti].replace('$', '﹩')
        text = '```'.join(ts)

    if fix_new_lines:
        # let Gradio handle code, since got improved recently
        ## FIXME: below conflicts with Gradio, but need to see if can handle multiple \n\n\n etc. properly as is.
        # ensure good visually, else markdown ignores multiple \n
        # handle code blocks
        ts = text.split('```')
        for parti, part in enumerate(ts):
            inside = parti % 2 == 1
            if not inside:
                ts[parti] = ts[parti].replace('\n', '<br>')
        text = '```'.join(ts)
    return text


def go_gradio(**kwargs):
    allow_api = kwargs['allow_api']
    is_public = kwargs['is_public']
    is_hf = kwargs['is_hf']
    memory_restriction_level = kwargs['memory_restriction_level']
    n_gpus = kwargs['n_gpus']
    admin_pass = kwargs['admin_pass']
    model_state0 = kwargs['model_state0']
    model_states = kwargs['model_states']
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

    title = 'h2oGPT'
    more_info = """<iframe src="https://ghbtns.com/github-btn.html?user=h2oai&repo=h2ogpt&type=star&count=true&size=small" frameborder="0" scrolling="0" width="250" height="20" title="GitHub"></iframe><small><a href="https://github.com/h2oai/h2ogpt">h2oGPT</a>  <a href="https://github.com/h2oai/h2o-llmstudio">H2O LLM Studio</a><br><a href="https://huggingface.co/h2oai">🤗 Models</a>"""
    if kwargs['verbose']:
        description = f"""Model {kwargs['base_model']} Instruct dataset.
                      For more information, visit our GitHub pages: [h2oGPT](https://github.com/h2oai/h2ogpt) and [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio).
                      Command: {str(' '.join(sys.argv))}
                      Hash: {get_githash()}
                      """
    else:
        description = more_info
    description_bottom = "If this host is busy, try [LLaMa 65B](https://llama.h2o.ai), [Falcon 40B](https://gpt.h2o.ai), [Falcon 40B](http://falcon.h2o.ai), [HF Spaces1 12B](https://huggingface.co/spaces/h2oai/h2ogpt-chatbot) or [HF Spaces2 12B](https://huggingface.co/spaces/h2oai/h2ogpt-chatbot2)<br>"
    description_bottom += """<p>By using h2oGPT, you accept our [Terms of Service](https://github.com/h2oai/h2ogpt/blob/main/docs/tos.md)</p>"""
    if is_hf:
        description_bottom += '''<a href="https://huggingface.co/spaces/h2oai/h2ogpt-chatbot?duplicate=true"><img src="https://bit.ly/3gLdBN6" style="white-space: nowrap" alt="Duplicate Space"></a>'''

    if kwargs['verbose']:
        task_info_md = f"""
        ### Task: {kwargs['task_info']}"""
    else:
        task_info_md = ''

    css_code = get_css(kwargs)

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
    elif kwargs['gradio_size'] == 'small':
        theme_kwargs.update(dict(spacing_size=gr.themes.sizes.spacing_sm, text_size=gr.themes.sizes.text_sm,
                                 radius_size=gr.themes.sizes.spacing_sm))
    elif kwargs['gradio_size'] == 'large':
        theme_kwargs.update(dict(spacing_size=gr.themes.sizes.spacing_lg, text_size=gr.themes.sizes.text_lg),
                            radius_size=gr.themes.sizes.spacing_lg)
    elif kwargs['gradio_size'] == 'medium':
        theme_kwargs.update(dict(spacing_size=gr.themes.sizes.spacing_md, text_size=gr.themes.sizes.text_md,
                                 radius_size=gr.themes.sizes.spacing_md))

    theme = H2oTheme(**theme_kwargs) if kwargs['h2ocolors'] else SoftTheme(**theme_kwargs)
    demo = gr.Blocks(theme=theme, css=css_code, title="h2oGPT", analytics_enabled=False)
    callback = gr.CSVLogger()

    model_options = flatten_list(list(prompt_type_to_model_name.values())) + kwargs['extra_model_options']
    if kwargs['base_model'].strip() not in model_options:
        model_options = [kwargs['base_model'].strip()] + model_options
    lora_options = kwargs['extra_lora_options']
    if kwargs['lora_weights'].strip() not in lora_options:
        lora_options = [kwargs['lora_weights'].strip()] + lora_options
    server_options = kwargs['extra_server_options']
    if kwargs['inference_server'].strip() not in server_options:
        server_options = [kwargs['inference_server'].strip()] + server_options
    if os.getenv('OPENAI_API_KEY'):
        if 'openai_chat' not in server_options:
            server_options += ['openai_chat']
        if 'openai' not in server_options:
            server_options += ['openai']

    # always add in no lora case
    # add fake space so doesn't go away in gradio dropdown
    model_options = [no_model_str] + model_options
    lora_options = [no_lora_str] + lora_options
    server_options = [no_server_str] + server_options
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
    output_label0 = f'h2oGPT [Model: {kwargs.get("base_model")}]' if kwargs.get(
        'base_model') else no_model_msg
    output_label0_model2 = no_model_msg

    default_kwargs = {k: kwargs[k] for k in eval_func_param_names_defaults}
    for k in no_default_param_names:
        default_kwargs[k] = ''

    def dummy_fun(x):
        # need dummy function to block new input from being sent until output is done,
        # else gets input_list at time of submit that is old, and shows up as truncated in chatbot
        return x

    with demo:
        # avoid actual model/tokenizer here or anything that would be bad to deepcopy
        # https://github.com/gradio-app/gradio/issues/3558
        model_state = gr.State(
            dict(model='model', tokenizer='tokenizer', device=kwargs['device'],
                 base_model=kwargs['base_model'],
                 tokenizer_base_model=kwargs['tokenizer_base_model'],
                 lora_weights=kwargs['lora_weights'],
                 inference_server=kwargs['inference_server'],
                 prompt_type=kwargs['prompt_type'],
                 prompt_dict=kwargs['prompt_dict'],
                 )
        )
        model_state2 = gr.State(kwargs['model_state_none'].copy())
        model_options_state = gr.State([model_options])
        lora_options_state = gr.State([lora_options])
        server_options_state = gr.State([server_options])
        my_db_state = gr.State([None, None])
        chat_state = gr.State({})
        # make user default first and default choice, dedup
        docs_state00 = kwargs['document_choice'] + [x.name for x in list(DocumentChoices)]
        docs_state0 = []
        [docs_state0.append(x) for x in docs_state00 if x not in docs_state0]
        docs_state = gr.State(docs_state0)  # first is chosen as default
        gr.Markdown(f"""
            {get_h2o_title(title, description) if kwargs['h2ocolors'] else get_simple_title(title, description)}
            """)

        # go button visible if
        base_wanted = kwargs['base_model'] != no_model_str and kwargs['login_mode_if_model0']
        go_btn = gr.Button(value="ENTER", visible=base_wanted, variant="primary")

        nas = ' '.join(['NA'] * len(kwargs['model_states']))
        res_value = "Response Score: NA" if not kwargs[
            'model_lock'] else "Response Scores: %s" % nas

        normal_block = gr.Row(visible=not base_wanted)
        with normal_block:
            with gr.Tabs():
                with gr.Row():
                    col_nochat = gr.Column(visible=not kwargs['chat'])
                    with col_nochat:  # FIXME: for model comparison, and check rest
                        if kwargs['langchain_mode'] == 'Disabled':
                            text_output_nochat = gr.Textbox(lines=5, label=output_label0, show_copy_button=True)
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
                        with gr.Column(visible=kwargs['score_model']):
                            score_text_nochat = gr.Textbox("Response Score: NA", show_label=False)

                    col_chat = gr.Column(visible=kwargs['chat'])
                    with col_chat:
                        instruction, submit, stop_btn = make_prompt_form(kwargs)
                        text_output, text_output2, text_outputs = make_chatbots(output_label0, output_label0_model2,
                                                                                **kwargs)

                        with gr.Row():
                            clear = gr.Button("Save Chat / New Chat")
                            flag_btn = gr.Button("Flag")
                            with gr.Column(visible=kwargs['score_model']):
                                score_text = gr.Textbox(res_value,
                                                        show_label=False,
                                                        visible=True)
                                score_text2 = gr.Textbox("Response Score2: NA", show_label=False,
                                                         visible=False and not kwargs['model_lock'])
                            retry_btn = gr.Button("Regenerate")
                            undo = gr.Button("Undo")
                    submit_nochat_api = gr.Button("Submit nochat API", visible=False)
                    inputs_dict_str = gr.Textbox(label='API input for nochat', show_label=False, visible=False)
                    text_output_nochat_api = gr.Textbox(lines=5, label='API nochat output', visible=False,
                                                        show_copy_button=True)
                with gr.TabItem("Documents"):
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
                            get_sources_btn = gr.Button(value="Get Sources", scale=0, size='sm')
                            show_sources_btn = gr.Button(value="Show Sources", scale=0, size='sm')
                            refresh_sources_btn = gr.Button(value="Refresh Sources", scale=0, size='sm')

                    # import control
                    if kwargs['langchain_mode'] != 'Disabled':
                        from gpt_langchain import file_types, have_arxiv
                    else:
                        have_arxiv = False
                        file_types = []

                    upload_row = gr.Row(visible=kwargs['langchain_mode'] != 'Disabled' and allow_upload,
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
                                                                 visible=allow_upload_to_user_data,
                                                                 elem_id='small_btn')
                                add_to_my_db_btn = gr.Button("Add File(s) to Scratch MyData",
                                                             visible=allow_upload_to_my_data and
                                                                     allow_upload_to_user_data,
                                                             elem_id='small_btn' if allow_upload_to_user_data else None,
                                                             size='sm' if not allow_upload_to_user_data else None)
                        with gr.Column(
                                visible=kwargs['langchain_mode'] != 'Disabled' and allow_upload and enable_url_upload):
                            url_label = 'URL (http/https) or ArXiv:' if have_arxiv else 'URL (http/https)'
                            url_text = gr.Textbox(label=url_label,
                                                  placeholder="Click Add to Submit" if
                                                  allow_upload_to_my_data and
                                                  allow_upload_to_user_data else
                                                  "Enter to Submit",
                                                  max_lines=1,
                                                  interactive=True)
                            with gr.Row():
                                url_user_btn = gr.Button(value='Add URL content to Shared UserData',
                                                         visible=allow_upload_to_user_data and allow_upload_to_my_data,
                                                         elem_id='small_btn')
                                url_my_btn = gr.Button(value='Add URL content to Scratch MyData',
                                                       visible=allow_upload_to_my_data and allow_upload_to_user_data,
                                                       elem_id='small_btn' if allow_upload_to_user_data else None,
                                                       size='sm' if not allow_upload_to_user_data else None)
                        with gr.Column(
                                visible=kwargs['langchain_mode'] != 'Disabled' and allow_upload and enable_text_upload):
                            user_text_text = gr.Textbox(label='Paste Text [Shift-Enter more lines]',
                                                        placeholder="Click Add to Submit" if
                                                        allow_upload_to_my_data and
                                                        allow_upload_to_user_data else
                                                        "Enter to Submit, Shift-Enter for more lines",
                                                        interactive=True)
                            with gr.Row():
                                user_text_user_btn = gr.Button(value='Add Text to Shared UserData',
                                                               visible=allow_upload_to_user_data and allow_upload_to_my_data,
                                                               elem_id='small_btn')
                                user_text_my_btn = gr.Button(value='Add Text to Scratch MyData',
                                                             visible=allow_upload_to_my_data and allow_upload_to_user_data,
                                                             elem_id='small_btn' if allow_upload_to_user_data else None,
                                                             size='sm' if not allow_upload_to_user_data else None)
                        with gr.Column(visible=False):
                            # WIP:
                            with gr.Row(visible=False, equal_height=False):
                                github_textbox = gr.Textbox(label="Github URL")
                                with gr.Row(visible=True):
                                    github_shared_btn = gr.Button(value="Add Github to Shared UserData",
                                                                  visible=allow_upload_to_user_data,
                                                                  elem_id='small_btn')
                                    github_my_btn = gr.Button(value="Add Github to Scratch MyData",
                                                              visible=allow_upload_to_my_data, elem_id='small_btn')
                    sources_row = gr.Row(visible=kwargs['langchain_mode'] != 'Disabled' and enable_sources_list,
                                         equal_height=False)
                    with sources_row:
                        with gr.Column(scale=1):
                            file_source = gr.File(interactive=False,
                                                  label="Download File w/Sources [click get sources to make file]")
                        with gr.Column(scale=2):
                            sources_text = gr.HTML(label='Sources Added', interactive=False)

                with gr.TabItem("Chat History"):
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
                        clear_chat_btn = gr.Button(value="Clear Chat", visible=True, size='sm')
                        export_chats_btn = gr.Button(value="Export Chats to Download", size='sm')
                        remove_chat_btn = gr.Button(value="Remove Selected Chat", visible=True, size='sm')
                        add_to_chats_btn = gr.Button("Import Chats from Upload", size='sm')
                    with gr.Row():
                        chats_file = gr.File(interactive=False, label="Download Exported Chats")
                        chatsup_output = gr.File(label="Upload Chat File(s)",
                                                 file_types=['.json'],
                                                 file_count='multiple',
                                                 elem_id="warning", elem_classes="feedback")

                with gr.TabItem("Expert"):
                    with gr.Row():
                        with gr.Column():
                            stream_output = gr.components.Checkbox(label="Stream output",
                                                                   value=kwargs['stream_output'])
                            prompt_type = gr.Dropdown(prompt_types_strings,
                                                      value=kwargs['prompt_type'], label="Prompt Type",
                                                      visible=not kwargs['model_lock'],
                                                      interactive=not is_public,
                                                      )
                            prompt_type2 = gr.Dropdown(prompt_types_strings,
                                                       value=kwargs['prompt_type'], label="Prompt Type Model 2",
                                                       visible=False and not kwargs['model_lock'],
                                                       interactive=not is_public)
                            do_sample = gr.Checkbox(label="Sample",
                                                    info="Enable sampler, required for use of temperature, top_p, top_k",
                                                    value=kwargs['do_sample'])
                            temperature = gr.Slider(minimum=0.01, maximum=2,
                                                    value=kwargs['temperature'],
                                                    label="Temperature",
                                                    info="Lower is deterministic (but may lead to repeats), Higher more creative (but may lead to hallucinations)")
                            top_p = gr.Slider(minimum=1e-3, maximum=1.0 - 1e-3,
                                              value=kwargs['top_p'], label="Top p",
                                              info="Cumulative probability of tokens to sample from")
                            top_k = gr.Slider(
                                minimum=1, maximum=100, step=1,
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
                                                       "Uses more GPU memory/compute",
                                                  interactive=False)
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
                            early_stopping = gr.Checkbox(label="EarlyStopping", info="Stop early in beam search",
                                                         value=kwargs['early_stopping'])
                            max_time = gr.Slider(minimum=0, maximum=kwargs['max_max_time'], step=1,
                                                 value=min(kwargs['max_max_time'],
                                                           kwargs['max_time']), label="Max. time",
                                                 info="Max. time to search optimal output.")
                            repetition_penalty = gr.Slider(minimum=0.01, maximum=3.0,
                                                           value=kwargs['repetition_penalty'],
                                                           label="Repetition Penalty")
                            num_return_sequences = gr.Slider(minimum=1, maximum=10, step=1,
                                                             value=kwargs['num_return_sequences'],
                                                             label="Number Returns", info="Must be <= num_beams",
                                                             interactive=not is_public)
                            iinput = gr.Textbox(lines=4, label="Input",
                                                placeholder=kwargs['placeholder_input'],
                                                interactive=not is_public)
                            context = gr.Textbox(lines=3, label="System Pre-Context",
                                                 info="Directly pre-appended without prompt processing",
                                                 interactive=not is_public)
                            chat = gr.components.Checkbox(label="Chat mode", value=kwargs['chat'],
                                                          visible=not kwargs['model_lock'],
                                                          interactive=not is_public,
                                                          )
                            count_chat_tokens_btn = gr.Button(value="Count Chat Tokens",
                                                              visible=not is_public and not kwargs['model_lock'],
                                                              interactive=not is_public)
                            chat_token_count = gr.Textbox(label="Chat Token Count", value=None,
                                                          visible=not is_public and not kwargs['model_lock'],
                                                          interactive=False)
                            chunk = gr.components.Checkbox(value=kwargs['chunk'],
                                                           label="Whether to chunk documents",
                                                           info="For LangChain",
                                                           visible=kwargs['langchain_mode'] != 'Disabled',
                                                           interactive=not is_public)
                            min_top_k_docs, max_top_k_docs, label_top_k_docs = get_minmax_top_k_docs(is_public)
                            top_k_docs = gr.Slider(minimum=min_top_k_docs, maximum=max_top_k_docs, step=1,
                                                   value=kwargs['top_k_docs'],
                                                   label=label_top_k_docs,
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

                with gr.TabItem("Models"):
                    model_lock_msg = gr.Textbox(lines=1, label="Model Lock Notice",
                                                placeholder="Started in model_lock mode, no model changes allowed.",
                                                visible=bool(kwargs['model_lock']), interactive=False)
                    load_msg = "Load-Unload Model/LORA [unload works if did not use --base_model]" if not is_public \
                        else "LOAD-UNLOAD DISABLED FOR HOSTED DEMO"
                    load_msg2 = "Load-Unload Model/LORA 2 [unload works if did not use --base_model]" if not is_public \
                        else "LOAD-UNLOAD DISABLED FOR HOSTED DEMO 2"
                    variant_load_msg = 'primary' if not is_public else 'secondary'
                    compare_checkbox = gr.components.Checkbox(label="Compare Mode",
                                                              value=kwargs['model_lock'],
                                                              visible=not is_public and not kwargs['model_lock'])
                    with gr.Row():
                        n_gpus_list = [str(x) for x in list(range(-1, n_gpus))]
                        with gr.Column():
                            with gr.Row():
                                with gr.Column(scale=20, visible=not kwargs['model_lock']):
                                    model_choice = gr.Dropdown(model_options_state.value[0], label="Choose Model",
                                                               value=kwargs['base_model'])
                                    lora_choice = gr.Dropdown(lora_options_state.value[0], label="Choose LORA",
                                                              value=kwargs['lora_weights'], visible=kwargs['show_lora'])
                                    server_choice = gr.Dropdown(server_options_state.value[0], label="Choose Server",
                                                                value=kwargs['inference_server'], visible=not is_public)
                                with gr.Column(scale=1, visible=not kwargs['model_lock']):
                                    load_model_button = gr.Button(load_msg, variant=variant_load_msg, scale=0,
                                                                  size='sm', interactive=not is_public)
                                    model_load8bit_checkbox = gr.components.Checkbox(
                                        label="Load 8-bit [requires support]",
                                        value=kwargs['load_8bit'], interactive=not is_public)
                                    model_infer_devices_checkbox = gr.components.Checkbox(
                                        label="Choose Devices [If not Checked, use all GPUs]",
                                        value=kwargs['infer_devices'], interactive=not is_public)
                                    model_gpu = gr.Dropdown(n_gpus_list,
                                                            label="GPU ID [-1 = all GPUs, if Choose is enabled]",
                                                            value=kwargs['gpu_id'], interactive=not is_public)
                                    model_used = gr.Textbox(label="Current Model", value=kwargs['base_model'],
                                                            interactive=False)
                                    lora_used = gr.Textbox(label="Current LORA", value=kwargs['lora_weights'],
                                                           visible=kwargs['show_lora'], interactive=False)
                                    server_used = gr.Textbox(label="Current Server",
                                                             value=kwargs['inference_server'],
                                                             visible=bool(kwargs['inference_server']) and not is_public,
                                                             interactive=False)
                                    prompt_dict = gr.Textbox(label="Prompt (or Custom)",
                                                             value=pprint.pformat(kwargs['prompt_dict'], indent=4),
                                                             interactive=not is_public, lines=4)
                        col_model2 = gr.Column(visible=False)
                        with col_model2:
                            with gr.Row():
                                with gr.Column(scale=20, visible=not kwargs['model_lock']):
                                    model_choice2 = gr.Dropdown(model_options_state.value[0], label="Choose Model 2",
                                                                value=no_model_str)
                                    lora_choice2 = gr.Dropdown(lora_options_state.value[0], label="Choose LORA 2",
                                                               value=no_lora_str,
                                                               visible=kwargs['show_lora'])
                                    server_choice2 = gr.Dropdown(server_options_state.value[0], label="Choose Server 2",
                                                                 value=no_server_str,
                                                                 visible=not is_public)
                                with gr.Column(scale=1, visible=not kwargs['model_lock']):
                                    load_model_button2 = gr.Button(load_msg2, variant=variant_load_msg, scale=0,
                                                                   size='sm', interactive=not is_public)
                                    model_load8bit_checkbox2 = gr.components.Checkbox(
                                        label="Load 8-bit 2 [requires support]",
                                        value=kwargs['load_8bit'], interactive=not is_public)
                                    model_infer_devices_checkbox2 = gr.components.Checkbox(
                                        label="Choose Devices 2 [If not Checked, use all GPUs]",
                                        value=kwargs[
                                            'infer_devices'], interactive=not is_public)
                                    model_gpu2 = gr.Dropdown(n_gpus_list,
                                                             label="GPU ID 2 [-1 = all GPUs, if choose is enabled]",
                                                             value=kwargs['gpu_id'], interactive=not is_public)
                                    # no model/lora loaded ever in model2 by default
                                    model_used2 = gr.Textbox(label="Current Model 2", value=no_model_str,
                                                             interactive=False)
                                    lora_used2 = gr.Textbox(label="Current LORA 2", value=no_lora_str,
                                                            visible=kwargs['show_lora'], interactive=False)
                                    server_used2 = gr.Textbox(label="Current Server 2", value=no_server_str,
                                                              interactive=False,
                                                              visible=not is_public)
                                    prompt_dict2 = gr.Textbox(label="Prompt (or Custom) 2",
                                                              value=pprint.pformat(kwargs['prompt_dict'], indent=4),
                                                              interactive=not is_public, lines=4)
                    with gr.Row(visible=not kwargs['model_lock']):
                        with gr.Column(scale=50):
                            new_model = gr.Textbox(label="New Model name/path", interactive=not is_public)
                        with gr.Column(scale=50):
                            new_lora = gr.Textbox(label="New LORA name/path", visible=kwargs['show_lora'],
                                                  interactive=not is_public)
                        with gr.Column(scale=50):
                            new_server = gr.Textbox(label="New Server url:port", interactive=not is_public)
                        with gr.Row():
                            add_model_lora_server_button = gr.Button("Add new Model, Lora, Server url:port", scale=0,
                                                                     size='sm', interactive=not is_public)
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
                                system_text = gr.Textbox(label='System Info', interactive=False, show_copy_button=True)
                            with gr.Row():
                                system_input = gr.Textbox(label='System Info Dict Password', interactive=True,
                                                          visible=not is_public)
                                system_btn2 = gr.Button(value='Get System Info Dict', visible=not is_public)
                                system_text2 = gr.Textbox(label='System Info Dict', interactive=False,
                                                          visible=not is_public, show_copy_button=True)
                            with gr.Row():
                                system_btn3 = gr.Button(value='Get Hash', visible=not is_public)
                                system_text3 = gr.Textbox(label='Hash', interactive=False,
                                                          visible=not is_public, show_copy_button=True)

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

        gr.Markdown(f"""
            {description_bottom}
            {task_info_md}
            """)

        # Get flagged data
        zip_data1 = functools.partial(zip_data, root_dirs=['flagged_data_points', kwargs['save_dir']])
        zip_btn.click(zip_data1, inputs=None, outputs=[file_output, zip_text], queue=False,
                      api_name='zip_data' if allow_api else None)
        s3up_btn.click(s3up, inputs=zip_text, outputs=s3up_text, queue=False,
                       api_name='s3up_data' if allow_api else None)

        def clear_file_list():
            return None

        def make_non_interactive(*args):
            if len(args) == 1:
                return gr.update(interactive=False)
            else:
                return tuple([gr.update(interactive=False)] * len(args))

        def make_interactive(*args):
            if len(args) == 1:
                return gr.update(interactive=True)
            else:
                return tuple([gr.update(interactive=True)] * len(args))

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
                                                user_path=kwargs['user_path'],
                                                )
        add_file_outputs = [fileup_output, langchain_mode, add_to_shared_db_btn, add_to_my_db_btn]
        add_file_kwargs = dict(fn=update_user_db_func,
                               inputs=[fileup_output, my_db_state, add_to_shared_db_btn,
                                       add_to_my_db_btn,
                                       chunk, chunk_size],
                               outputs=add_file_outputs + [sources_text],
                               queue=queue,
                               api_name='add_to_shared' if allow_api and allow_upload_to_user_data else None)

        if allow_upload_to_user_data and not allow_upload_to_my_data:
            func1 = fileup_output.change
        else:
            func1 = add_to_shared_db_btn.click
        # then no need for add buttons, only single changeable db
        eventdb1a = func1(make_non_interactive, inputs=add_file_outputs, outputs=add_file_outputs,
                          show_progress='minimal')
        eventdb1 = eventdb1a.then(**add_file_kwargs, show_progress='minimal')
        eventdb1.then(make_interactive, inputs=add_file_outputs, outputs=add_file_outputs, show_progress='minimal')

        # note for update_user_db_func output is ignored for db

        def clear_textbox():
            return gr.Textbox.update(value='')

        update_user_db_url_func = functools.partial(update_user_db_func, is_url=True)

        add_url_outputs = [url_text, langchain_mode, url_user_btn, url_my_btn]
        add_url_kwargs = dict(fn=update_user_db_url_func,
                              inputs=[url_text, my_db_state, url_user_btn, url_my_btn,
                                      chunk, chunk_size],
                              outputs=add_url_outputs + [sources_text],
                              queue=queue,
                              api_name='add_url_to_shared' if allow_api and allow_upload_to_user_data else None)

        if allow_upload_to_user_data and not allow_upload_to_my_data:
            func2 = url_text.submit
        else:
            func2 = url_user_btn.click
        eventdb2a = func2(fn=dummy_fun, inputs=url_text, outputs=url_text, queue=queue,
                          show_progress='minimal')
        # work around https://github.com/gradio-app/gradio/issues/4733
        eventdb2b = eventdb2a.then(make_non_interactive, inputs=add_url_outputs, outputs=add_url_outputs,
                                   show_progress='minimal')
        eventdb2 = eventdb2b.then(**add_url_kwargs, show_progress='minimal')
        eventdb2.then(make_interactive, inputs=add_url_outputs, outputs=add_url_outputs, show_progress='minimal')

        update_user_db_txt_func = functools.partial(update_user_db_func, is_txt=True)
        add_text_outputs = [user_text_text, langchain_mode, user_text_user_btn, user_text_my_btn]
        add_text_kwargs = dict(fn=update_user_db_txt_func,
                               inputs=[user_text_text, my_db_state, user_text_user_btn, user_text_my_btn,
                                       chunk, chunk_size],
                               outputs=add_text_outputs + [sources_text],
                               queue=queue,
                               api_name='add_text_to_shared' if allow_api and allow_upload_to_user_data else None
                               )
        if allow_upload_to_user_data and not allow_upload_to_my_data:
            func3 = user_text_text.submit
        else:
            func3 = user_text_user_btn.click

        eventdb3a = func3(fn=dummy_fun, inputs=user_text_text, outputs=user_text_text, queue=queue,
                          show_progress='minimal')
        eventdb3b = eventdb3a.then(make_non_interactive, inputs=add_text_outputs, outputs=add_text_outputs,
                                   show_progress='minimal')
        eventdb3 = eventdb3b.then(**add_text_kwargs, show_progress='minimal')
        eventdb3.then(make_interactive, inputs=add_text_outputs, outputs=add_text_outputs,
                      show_progress='minimal')

        update_my_db_func = functools.partial(update_user_db, dbs=dbs, db_type=db_type, langchain_mode='MyData',
                                              use_openai_embedding=use_openai_embedding,
                                              hf_embedding_model=hf_embedding_model,
                                              enable_captions=enable_captions,
                                              captions_model=captions_model,
                                              enable_ocr=enable_ocr,
                                              caption_loader=caption_loader,
                                              verbose=kwargs['verbose'],
                                              user_path=kwargs['user_path'],
                                              )

        add_my_file_outputs = [fileup_output, langchain_mode, my_db_state, add_to_shared_db_btn, add_to_my_db_btn]
        add_my_file_kwargs = dict(fn=update_my_db_func,
                                  inputs=[fileup_output, my_db_state, add_to_shared_db_btn, add_to_my_db_btn,
                                          chunk, chunk_size],
                                  outputs=add_my_file_outputs + [sources_text],
                                  queue=queue,
                                  api_name='add_to_my' if allow_api and allow_upload_to_my_data else None)

        if not allow_upload_to_user_data and allow_upload_to_my_data:
            func4 = fileup_output.change
        else:
            func4 = add_to_my_db_btn.click

        eventdb4a = func4(make_non_interactive, inputs=add_my_file_outputs,
                          outputs=add_my_file_outputs,
                          show_progress='minimal')
        eventdb4 = eventdb4a.then(**add_my_file_kwargs, show_progress='minimal')
        eventdb4.then(make_interactive, inputs=add_my_file_outputs, outputs=add_my_file_outputs,
                      show_progress='minimal')

        update_my_db_url_func = functools.partial(update_my_db_func, is_url=True)
        add_my_url_outputs = [url_text, langchain_mode, my_db_state, url_user_btn, url_my_btn]
        add_my_url_kwargs = dict(fn=update_my_db_url_func,
                                 inputs=[url_text, my_db_state, url_user_btn, url_my_btn,
                                         chunk, chunk_size],
                                 outputs=add_my_url_outputs + [sources_text],
                                 queue=queue,
                                 api_name='add_url_to_my' if allow_api and allow_upload_to_my_data else None)
        if not allow_upload_to_user_data and allow_upload_to_my_data:
            func5 = url_text.submit
        else:
            func5 = url_my_btn.click
        eventdb5a = func5(fn=dummy_fun, inputs=url_text, outputs=url_text, queue=queue,
                          show_progress='minimal')
        eventdb5b = eventdb5a.then(make_non_interactive, inputs=add_my_url_outputs, outputs=add_my_url_outputs,
                                   show_progress='minimal')
        eventdb5 = eventdb5b.then(**add_my_url_kwargs, show_progress='minimal')
        eventdb5.then(make_interactive, inputs=add_my_url_outputs, outputs=add_my_url_outputs,
                      show_progress='minimal')

        update_my_db_txt_func = functools.partial(update_my_db_func, is_txt=True)

        add_my_text_outputs = [user_text_text, langchain_mode, my_db_state, user_text_user_btn,
                               user_text_my_btn]
        add_my_text_kwargs = dict(fn=update_my_db_txt_func,
                                  inputs=[user_text_text, my_db_state, user_text_user_btn, user_text_my_btn,
                                          chunk, chunk_size],
                                  outputs=add_my_text_outputs + [sources_text],
                                  queue=queue,
                                  api_name='add_txt_to_my' if allow_api and allow_upload_to_my_data else None)
        if not allow_upload_to_user_data and allow_upload_to_my_data:
            func6 = user_text_text.submit
        else:
            func6 = user_text_my_btn.click

        eventdb6a = func6(fn=dummy_fun, inputs=user_text_text, outputs=user_text_text, queue=queue,
                          show_progress='minimal')
        eventdb6b = eventdb6a.then(make_non_interactive, inputs=add_my_text_outputs, outputs=add_my_text_outputs,
                                   show_progress='minimal')
        eventdb6 = eventdb6b.then(**add_my_text_kwargs, show_progress='minimal')
        eventdb6.then(make_interactive, inputs=add_my_text_outputs, outputs=add_my_text_outputs,
                      show_progress='minimal')

        get_sources1 = functools.partial(get_sources, dbs=dbs, docs_state0=docs_state0)

        # if change collection source, must clear doc selections from it to avoid inconsistency
        def clear_doc_choice():
            return gr.Dropdown.update(choices=docs_state0, value=[docs_state0[0]])

        langchain_mode.change(clear_doc_choice, inputs=None, outputs=document_choice)

        def update_dropdown(x):
            return gr.Dropdown.update(choices=x, value=[docs_state0[0]])

        eventdb7 = get_sources_btn.click(get_sources1, inputs=[my_db_state, langchain_mode],
                                         outputs=[file_source, docs_state],
                                         queue=queue,
                                         api_name='get_sources' if allow_api else None) \
            .then(fn=update_dropdown, inputs=docs_state, outputs=document_choice)
        # show button, else only show when add.  Could add to above get_sources for download/dropdown, but bit much maybe
        show_sources1 = functools.partial(get_source_files_given_langchain_mode, dbs=dbs)
        eventdb8 = show_sources_btn.click(fn=show_sources1, inputs=[my_db_state, langchain_mode], outputs=sources_text,
                                          api_name='show_sources' if allow_api else None)

        # Get inputs to evaluate() and make_db()
        # don't deepcopy, can contain model itself
        all_kwargs = kwargs.copy()
        all_kwargs.update(locals())

        refresh_sources1 = functools.partial(update_and_get_source_files_given_langchain_mode,
                                             **get_kwargs(update_and_get_source_files_given_langchain_mode,
                                                          exclude_names=['db1', 'langchain_mode'],
                                                          **all_kwargs))
        eventdb9 = refresh_sources_btn.click(fn=refresh_sources1, inputs=[my_db_state, langchain_mode],
                                             outputs=sources_text,
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
                if kwargs['langchain_mode'] == 'Disabled':
                    yield fix_text_for_gradio(res_dict['response'])
                else:
                    yield '<br>' + fix_text_for_gradio(res_dict['response'])

        fun = partial(evaluate_gradio,
                      **kwargs_evaluate)
        fun2 = partial(evaluate_gradio,
                       **kwargs_evaluate)
        fun_with_dict_str = partial(evaluate_from_str,
                                    default_kwargs=default_kwargs,
                                    **kwargs_evaluate
                                    )

        dark_mode_btn = gr.Button("Dark Mode", variant="primary", size="sm")
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
            return gr.Column.update(visible=bool(x))

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
                clear_torch_cache()

        def _score_last_response(*args, nochat=False, num_model_lock=0, prefix='Response Score: '):
            """ Similar to user() """
            args_list = list(args)
            smodel = score_model_state0['model']
            stokenizer = score_model_state0['tokenizer']
            sdevice = score_model_state0['device']

            if memory_restriction_level > 0:
                max_length_tokenize = 768 - 256 if memory_restriction_level <= 2 else 512 - 256
            elif hasattr(stokenizer, 'model_max_length'):
                max_length_tokenize = stokenizer.model_max_length
            else:
                # limit to 1024, not worth OOMing on reward score
                max_length_tokenize = 2048 - 1024
            cutoff_len = max_length_tokenize * 4  # restrict deberta related to max for LLM

            if not nochat:
                history = args_list[-1]
                if history is None:
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
                    return '%sNA' % prefix
            else:
                answer = args_list[-1]
                instruction_nochat_arg_id = eval_func_param_names.index('instruction_nochat')
                question = args_list[instruction_nochat_arg_id]

            if question is None:
                return '%sBad Question' % prefix
            if answer is None:
                return '%sBad Answer' % prefix
            try:
                score = score_qa(smodel, stokenizer, max_length_tokenize, question, answer, cutoff_len)
            finally:
                clear_torch_cache()
            if isinstance(score, str):
                return '%sNA' % prefix
            return '{}{:.1%}'.format(prefix, score)

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
            :param sanitize_user_prompt:
            :param model2:
            :return:
            """
            args_list = list(args)
            user_message = args_list[eval_func_param_names.index('instruction')]  # chat only
            input1 = args_list[eval_func_param_names.index('iinput')]  # chat only
            prompt_type1 = args_list[eval_func_param_names.index('prompt_type')]
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
                from better_profanity import profanity
                user_message1 = profanity.censor(user_message1)

            history = args_list[-1]
            if history is None:
                # bad history
                history = []
            history = history.copy()

            if undo:
                if len(history) > 0:
                    history.pop()
                return history
            if retry:
                if history:
                    history[-1][1] = None
                return history
            if user_message1 in ['', None, '\n']:
                # reject non-retry submit/enter
                return history
            user_message1 = fix_text_for_gradio(user_message1)
            return history + [[user_message1, None]]

        def user(*args, undo=False, retry=False, sanitize_user_prompt=False):
            return update_history(*args, undo=undo, retry=retry, sanitize_user_prompt=sanitize_user_prompt)

        def all_user(*args, undo=False, retry=False, sanitize_user_prompt=False, num_model_lock=0):
            args_list = list(args)
            history_list = args_list[-num_model_lock:]
            assert len(history_list) > 0, "Bad history list: %s" % history_list
            for hi, history in enumerate(history_list):
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

        def get_model_max_length(model_state1):
            if model_state1 and not isinstance(model_state1["tokenizer"], str):
                tokenizer = model_state1["tokenizer"]
            elif model_state0 and not isinstance(model_state0["tokenizer"], str):
                tokenizer = model_state0["tokenizer"]
            else:
                tokenizer = None
            if tokenizer is not None:
                return tokenizer.model_max_length
            else:
                return 2000

        def prep_bot(*args, retry=False):
            """

            :param args:
            :param retry:
            :return: last element is True if should run bot, False if should just yield history
            """
            # don't deepcopy, can contain model itself
            args_list = list(args).copy()
            model_state1 = args_list[-3]
            my_db_state1 = args_list[-2]
            history = args_list[-1]
            langchain_mode1 = args_list[eval_func_param_names.index('langchain_mode')]

            if model_state1['model'] is None or model_state1['model'] == no_model_str:
                return history, None, None, None

            args_list = args_list[:-3]  # only keep rest needed for evaluate()
            if not history:
                print("No history", flush=True)
                history = []
                return history, None, None, None
            instruction1 = history[-1][0]
            if retry and history:
                # if retry, pop history and move onto bot stuff
                instruction1 = history[-1][0]
                history[-1][1] = None
            elif not instruction1:
                # if not retrying, then reject empty query
                return history, None, None, None
            elif len(history) > 0 and history[-1][1] not in [None, '']:
                # reject submit button if already filled and not retrying
                # None when not filling with '' to keep client happy
                return history, None, None, None

            # shouldn't have to specify in API prompt_type if CLI launched model, so prefer global CLI one if have it
            prompt_type1 = kwargs.get('prompt_type', args_list[eval_func_param_names.index('prompt_type')])
            # prefer model specific prompt type instead of global one, and apply back to args_list for evaluate()
            args_list[eval_func_param_names.index('prompt_type')] = prompt_type1 = \
                model_state1.get('prompt_type', prompt_type1)

            prompt_dict1 = kwargs.get('prompt_dict', args_list[eval_func_param_names.index('prompt_dict')])
            args_list[eval_func_param_names.index('prompt_dict')] = prompt_dict1 = \
                model_state1.get('prompt_dict', prompt_dict1)

            chat1 = args_list[eval_func_param_names.index('chat')]
            model_max_length1 = get_model_max_length(model_state1)
            context1 = history_to_context(history, langchain_mode1, prompt_type1, prompt_dict1, chat1,
                                          model_max_length1, memory_restriction_level,
                                          kwargs['keep_sources_in_context'])
            args_list[0] = instruction1  # override original instruction with history from user
            args_list[2] = context1

            fun1 = partial(evaluate,
                           model_state1,
                           my_db_state1,
                           *tuple(args_list),
                           **kwargs_evaluate)

            return history, fun1, langchain_mode1, my_db_state1

        def get_response(fun1, history):
            """
            bot that consumes history for user input
            instruction (from input_list) itself is not consumed by bot
            :return:
            """
            if not fun1:
                yield history, ''
                return
            try:
                for output_fun in fun1():
                    output = output_fun['response']
                    extra = output_fun['sources']  # FIXME: can show sources in separate text box etc.
                    # ensure good visually, else markdown ignores multiple \n
                    bot_message = fix_text_for_gradio(output)
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
            finally:
                clear_torch_cache()
            return

        def clear_embeddings(langchain_mode1, my_db):
            # clear any use of embedding that sits on GPU, else keeps accumulating GPU usage even if clear torch cache
            if db_type == 'chroma' and langchain_mode1 not in ['ChatLLM', 'LLM', 'Disabled', None, '']:
                from gpt_langchain import clear_embedding
                db = dbs.get('langchain_mode1')
                if db is not None and not isinstance(db, str):
                    clear_embedding(db)
                if langchain_mode1 == LangChainMode.MY_DATA.value and my_db is not None:
                    clear_embedding(my_db[0])

        def bot(*args, retry=False):
            history, fun1, langchain_mode1, my_db_state1 = prep_bot(*args, retry=retry)
            try:
                for res in get_response(fun1, history):
                    yield res
            finally:
                clear_embeddings(langchain_mode1, my_db_state1)

        def all_bot(*args, retry=False, model_states1=None):
            args_list = list(args).copy()
            chatbots = args_list[-len(model_states1):]
            args_list0 = args_list[:-len(model_states1)]  # same for all models
            exceptions = []
            stream_output1 = args_list[eval_func_param_names.index('stream_output')]
            max_time1 = args_list[eval_func_param_names.index('max_time')]
            langchain_mode1 = args_list[eval_func_param_names.index('langchain_mode')]
            my_db_state1 = None  # will be filled below by some bot
            try:
                gen_list = []
                for chatbot1, model_state1 in zip(chatbots, model_states1):
                    args_list1 = args_list0.copy()
                    args_list1.insert(-1, model_state1)  # insert at -1 so is at -2
                    # if at start, have None in response still, replace with '' so client etc. acts like normal
                    # assumes other parts of code treat '' and None as if no response yet from bot
                    # can't do this later in bot code as racy with threaded generators
                    if len(chatbot1) > 0 and len(chatbot1[-1]) == 2 and chatbot1[-1][1] is None:
                        chatbot1[-1][1] = ''
                    args_list1.append(chatbot1)
                    # so consistent with prep_bot()
                    # with model_state1 at -3, my_db_state1 at -2, and history(chatbot) at -1
                    # langchain_mode1 and my_db_state1 should be same for every bot
                    history, fun1, langchain_mode1, my_db_state1 = prep_bot(*tuple(args_list1), retry=retry)
                    gen1 = get_response(fun1, history)
                    if stream_output1:
                        gen1 = TimeoutIterator(gen1, timeout=0.01, sentinel=None, raise_on_exception=False)
                    # else timeout will truncate output for non-streaming case
                    gen_list.append(gen1)

                bots_old = chatbots.copy()
                exceptions_old = [''] * len(bots_old)
                tgen0 = time.time()
                for res1 in itertools.zip_longest(*gen_list):
                    if time.time() - tgen0 > max_time1:
                        break

                    bots = [x[0] if x is not None and not isinstance(x, BaseException) else y for x, y in
                            zip(res1, bots_old)]
                    bots_old = bots.copy()

                    def larger_str(x, y):
                        return x if len(x) > len(y) else y

                    exceptions = [x[1] if x is not None and not isinstance(x, BaseException) else larger_str(str(x), y)
                                  for x, y in zip(res1, exceptions_old)]
                    exceptions_old = exceptions.copy()

                    def choose_exc(x):
                        # don't expose ports etc. to exceptions window
                        if is_public:
                            return "Endpoint unavailable or failed"
                        else:
                            return x

                    exceptions_str = '\n'.join(
                        ['Model %s: %s' % (iix, choose_exc(x)) for iix, x in enumerate(exceptions) if
                         x not in [None, '', 'None']])
                    if len(bots) > 1:
                        yield tuple(bots + [exceptions_str])
                    else:
                        yield bots[0], exceptions_str
                if exceptions:
                    exceptions = [x for x in exceptions if x not in ['', None, 'None']]
                    if exceptions:
                        print("Generate exceptions: %s" % exceptions, flush=True)
            finally:
                clear_torch_cache()
                clear_embeddings(langchain_mode1, my_db_state1)

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
        retry_user_args = dict(fn=functools.partial(user, retry=True),
                               inputs=inputs_list + [text_output],
                               outputs=text_output,
                               )
        undo_user_args = dict(fn=functools.partial(user, undo=True),
                              inputs=inputs_list + [text_output],
                              outputs=text_output,
                              )

        # MODEL2
        user_args2 = dict(fn=functools.partial(user, sanitize_user_prompt=kwargs['sanitize_user_prompt']),
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
                                                  ),
                             inputs=inputs_list + text_outputs,
                             outputs=text_outputs,
                             )
        all_bot_args = dict(fn=functools.partial(all_bot, model_states1=model_states),
                            inputs=inputs_list + [my_db_state] + text_outputs,
                            outputs=text_outputs + [exception_text],
                            )
        all_retry_bot_args = dict(fn=functools.partial(all_bot, model_states1=model_states, retry=True),
                                  inputs=inputs_list + [my_db_state] + text_outputs,
                                  outputs=text_outputs + [exception_text],
                                  )
        all_retry_user_args = dict(fn=functools.partial(all_user, retry=True,
                                                        sanitize_user_prompt=kwargs['sanitize_user_prompt'],
                                                        num_model_lock=len(text_outputs),
                                                        ),
                                   inputs=inputs_list + text_outputs,
                                   outputs=text_outputs,
                                   )
        all_undo_user_args = dict(fn=functools.partial(all_user, undo=True,
                                                       sanitize_user_prompt=kwargs['sanitize_user_prompt'],
                                                       num_model_lock=len(text_outputs),
                                                       ),
                                  inputs=inputs_list + text_outputs,
                                  outputs=text_outputs,
                                  )

        def clear_instruct():
            return gr.Textbox.update(value='')

        def deselect_radio_chats():
            return gr.update(value=None)

        def clear_all():
            return gr.Textbox.update(value=''), gr.Textbox.update(value=''), gr.update(value=None), \
                gr.Textbox.update(value=''), gr.Textbox.update(value='')

        if kwargs['model_states']:
            submits1 = submits2 = submits3 = []
            submits4 = []

            fun_source = [instruction.submit, submit.click, retry_btn.click]
            fun_name = ['instruction', 'submit', 'retry']
            user_args = [all_user_args, all_user_args, all_retry_user_args]
            bot_args = [all_bot_args, all_bot_args, all_retry_bot_args]
            for userargs1, botarg1, funn1, funs1 in zip(user_args, bot_args, fun_name, fun_source):
                submit_event11 = funs1(fn=dummy_fun,
                                       inputs=instruction, outputs=instruction, queue=queue)
                submit_event1a = submit_event11.then(**userargs1, queue=queue,
                                                     api_name='%s' % funn1 if allow_api else None)
                # if hit enter on new instruction for submitting new query, no longer the saved chat
                submit_event1b = submit_event1a.then(clear_all, inputs=None,
                                                     outputs=[instruction, iinput, radio_chats, score_text,
                                                              score_text2],
                                                     queue=queue)
                submit_event1c = submit_event1b.then(**botarg1,
                                                     api_name='%s_bot' % funn1 if allow_api else None,
                                                     queue=queue)
                submit_event1d = submit_event1c.then(**all_score_args,
                                                     api_name='%s_bot_score' % funn1 if allow_api else None,
                                                     queue=queue)

                submits1.extend([submit_event1a, submit_event1b, submit_event1c, submit_event1d])

            # if undo, no longer the saved chat
            submit_event4 = undo.click(fn=dummy_fun,
                                       inputs=instruction, outputs=instruction, queue=queue) \
                .then(**all_undo_user_args, api_name='undo' if allow_api else None) \
                .then(clear_all, inputs=None, outputs=[instruction, iinput, radio_chats, score_text,
                                                       score_text2], queue=queue) \
                .then(**all_score_args, api_name='undo_score' if allow_api else None)
            submits4 = [submit_event4]

        else:
            # in case 2nd model, consume instruction first, so can clear quickly
            # bot doesn't consume instruction itself, just history from user, so why works
            submit_event11 = instruction.submit(fn=dummy_fun,
                                                inputs=instruction, outputs=instruction, queue=queue)
            submit_event1a = submit_event11.then(**user_args, queue=queue,
                                                 api_name='instruction' if allow_api else None)
            # if hit enter on new instruction for submitting new query, no longer the saved chat
            submit_event1a2 = submit_event1a.then(deselect_radio_chats, inputs=None, outputs=radio_chats, queue=queue)
            submit_event1b = submit_event1a2.then(**user_args2, api_name='instruction2' if allow_api else None)
            submit_event1c = submit_event1b.then(clear_instruct, None, instruction) \
                .then(clear_instruct, None, iinput)
            submit_event1d = submit_event1c.then(**bot_args, api_name='instruction_bot' if allow_api else None,
                                                 queue=queue)
            submit_event1e = submit_event1d.then(**score_args,
                                                 api_name='instruction_bot_score' if allow_api else None,
                                                 queue=queue)
            submit_event1f = submit_event1e.then(**bot_args2, api_name='instruction_bot2' if allow_api else None,
                                                 queue=queue)
            submit_event1g = submit_event1f.then(**score_args2,
                                                 api_name='instruction_bot_score2' if allow_api else None, queue=queue)

            submits1 = [submit_event1a, submit_event1a2, submit_event1b, submit_event1c, submit_event1d,
                        submit_event1e,
                        submit_event1f, submit_event1g]

            submit_event21 = submit.click(fn=dummy_fun,
                                          inputs=instruction, outputs=instruction, queue=queue)
            submit_event2a = submit_event21.then(**user_args, api_name='submit' if allow_api else None)
            # if submit new query, no longer the saved chat
            submit_event2a2 = submit_event2a.then(deselect_radio_chats, inputs=None, outputs=radio_chats, queue=queue)
            submit_event2b = submit_event2a2.then(**user_args2, api_name='submit2' if allow_api else None)
            submit_event2c = submit_event2b.then(clear_all, inputs=None,
                                                 outputs=[instruction, iinput, radio_chats, score_text, score_text2],
                                                 queue=queue)
            submit_event2d = submit_event2c.then(**bot_args, api_name='submit_bot' if allow_api else None, queue=queue)
            submit_event2e = submit_event2d.then(**score_args,
                                                 api_name='submit_bot_score' if allow_api else None,
                                                 queue=queue)
            submit_event2f = submit_event2e.then(**bot_args2, api_name='submit_bot2' if allow_api else None,
                                                 queue=queue)
            submit_event2g = submit_event2f.then(**score_args2,
                                                 api_name='submit_bot_score2' if allow_api else None,
                                                 queue=queue)

            submits2 = [submit_event2a, submit_event2a2, submit_event2b, submit_event2c, submit_event2d,
                        submit_event2e,
                        submit_event2f, submit_event2g]

            submit_event31 = retry_btn.click(fn=dummy_fun,
                                             inputs=instruction, outputs=instruction, queue=queue)
            submit_event3a = submit_event31.then(**user_args, api_name='retry' if allow_api else None)
            # if retry, no longer the saved chat
            submit_event3a2 = submit_event3a.then(deselect_radio_chats, inputs=None, outputs=radio_chats, queue=queue)
            submit_event3b = submit_event3a2.then(**user_args2, api_name='retry2' if allow_api else None)
            submit_event3c = submit_event3b.then(clear_instruct, None, instruction) \
                .then(clear_instruct, None, iinput)
            submit_event3d = submit_event3c.then(**retry_bot_args, api_name='retry_bot' if allow_api else None,
                                                 queue=queue)
            submit_event3e = submit_event3d.then(**score_args,
                                                 api_name='retry_bot_score' if allow_api else None,
                                                 queue=queue)
            submit_event3f = submit_event3e.then(**retry_bot_args2, api_name='retry_bot2' if allow_api else None,
                                                 queue=queue)
            submit_event3g = submit_event3f.then(**score_args2,
                                                 api_name='retry_bot_score2' if allow_api else None,
                                                 queue=queue)

            submits3 = [submit_event3a, submit_event3a2, submit_event3b, submit_event3c, submit_event3d,
                        submit_event3e,
                        submit_event3f, submit_event3g]

            # if undo, no longer the saved chat
            submit_event4 = undo.click(fn=dummy_fun,
                                       inputs=instruction, outputs=instruction, queue=queue) \
                .then(**undo_user_args, api_name='undo' if allow_api else None) \
                .then(**undo_user_args2, api_name='undo2' if allow_api else None) \
                .then(clear_all, inputs=None, outputs=[instruction, iinput, radio_chats, score_text,
                                                       score_text2], queue=queue) \
                .then(**score_args, api_name='undo_score' if allow_api else None) \
                .then(**score_args2, api_name='undo_score2' if allow_api else None)
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
                    questionx = stepxx[0].replace('<p>', '').replace('</p>', '') if stepxx[0] is not None else None
                    answerx = stepxx[1].replace('<p>', '').replace('</p>', '') if stepxx[1] is not None else None

                    questiony = stepyy[0].replace('<p>', '').replace('</p>', '') if stepyy[0] is not None else None
                    answery = stepyy[1].replace('<p>', '').replace('</p>', '') if stepyy[1] is not None else None

                    if questionx != questiony or answerx != answery:
                        return False
            return is_same

        def save_chat(*args):
            args_list = list(args)
            chat_list = args_list[:-1]  # list of chatbot histories
            # remove None histories
            chat_list_not_none = [x for x in chat_list if x and len(x) > 0 and len(x[0]) == 2 and x[0][1] is not None]
            chat_state1 = args_list[
                -1]  # dict with keys of short chat names, values of list of list of chatbot histories
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
                # clear chat_list so saved and then new conversation starts
                chat_list = [[]] * len(chat_list)
            ret_list = chat_list + [chat_state1]
            return tuple(ret_list)

        def update_radio_chats(chat_state1):
            return gr.update(choices=list(chat_state1.keys()), value=None)

        def switch_chat(chat_key, chat_state1, num_model_lock=0):
            chosen_chat = chat_state1[chat_key]
            # deal with possible different size of chat list vs. current list
            ret_chat = [None] * (2 + num_model_lock)
            for chati in range(0, 2 + num_model_lock):
                ret_chat[chati % len(ret_chat)] = chosen_chat[chati % len(chosen_chat)]
            return tuple(ret_chat)

        def clear_texts(*args):
            return tuple([gr.Textbox.update(value='')] * len(args))

        def clear_scores():
            return gr.Textbox.update(value=res_value), \
                gr.Textbox.update(value='Response Score: NA'), \
                gr.Textbox.update(value='Response Score: NA')

        switch_chat_fun = functools.partial(switch_chat, num_model_lock=len(text_outputs))
        radio_chats.input(switch_chat_fun,
                          inputs=[radio_chats, chat_state],
                          outputs=[text_output, text_output2] + text_outputs) \
            .then(clear_scores, outputs=[score_text, score_text2, score_text_nochat])

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
                            _, chat_state1 = save_chat(chat1_v, chat_state1)
                except BaseException as e:
                    t, v, tb = sys.exc_info()
                    ex = ''.join(traceback.format_exception(t, v, tb))
                    print("Add chats exception: %s" % str(ex), flush=True)
            return chat_state1, add_btn

        # note for update_user_db_func output is ignored for db
        add_to_chats_btn.click(add_chats_from_file,
                               inputs=[chatsup_output, chat_state, add_to_chats_btn],
                               outputs=[chat_state, add_to_my_db_btn], queue=False,
                               api_name='add_to_chats' if allow_api else None) \
            .then(clear_file_list, outputs=chatsup_output, queue=False) \
            .then(update_radio_chats, inputs=chat_state, outputs=radio_chats, queue=False)

        clear_chat_btn.click(fn=clear_texts,
                             inputs=[text_output, text_output2] + text_outputs,
                             outputs=[text_output, text_output2] + text_outputs,
                             queue=False, api_name='clear' if allow_api else None) \
            .then(deselect_radio_chats, inputs=None, outputs=radio_chats, queue=False) \
            .then(clear_scores, outputs=[score_text, score_text2, score_text_nochat])

        # does both models
        clear.click(save_chat,
                    inputs=[text_output, text_output2] + text_outputs + [chat_state],
                    outputs=[text_output, text_output2] + text_outputs + [chat_state],
                    api_name='save_chat' if allow_api else None) \
            .then(update_radio_chats, inputs=chat_state, outputs=radio_chats,
                  api_name='update_chats' if allow_api else None) \
            .then(clear_scores, outputs=[score_text, score_text2, score_text_nochat])

        # NOTE: clear of instruction/iinput for nochat has to come after score,
        # because score for nochat consumes actual textbox, while chat consumes chat history filled by user()
        no_chat_args = dict(fn=fun,
                            inputs=[model_state, my_db_state] + inputs_list,
                            outputs=text_output_nochat,
                            queue=queue,
                            )
        submit_event_nochat = submit_nochat.click(**no_chat_args, api_name='submit_nochat' if allow_api else None) \
            .then(clear_torch_cache) \
            .then(**score_args_nochat, api_name='instruction_bot_score_nochat' if allow_api else None, queue=queue) \
            .then(clear_instruct, None, instruction_nochat) \
            .then(clear_instruct, None, iinput_nochat) \
            .then(clear_torch_cache)
        # copy of above with text box submission
        submit_event_nochat2 = instruction_nochat.submit(**no_chat_args) \
            .then(clear_torch_cache) \
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

        def load_model(model_name, lora_weights, server_name, model_state_old, prompt_type_old, load_8bit,
                       infer_devices, gpu_id):
            # ensure old model removed from GPU memory
            if kwargs['debug']:
                print("Pre-switch pre-del GPU memory: %s" % get_torch_allocated(), flush=True)

            model0 = model_state0['model']
            if isinstance(model_state_old['model'], str) and model0 is not None:
                # best can do, move model loaded at first to CPU
                model0.cpu()

            if model_state_old['model'] is not None and not isinstance(model_state_old['model'], str):
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

            clear_torch_cache()
            if kwargs['debug']:
                print("Pre-switch post-del GPU memory: %s" % get_torch_allocated(), flush=True)

            if model_name is None or model_name == no_model_str:
                # no-op if no model, just free memory
                # no detranscribe needed for model, never go into evaluate
                lora_weights = no_lora_str
                server_name = no_server_str
                return [None, None, None, model_name, server_name], \
                    model_name, lora_weights, server_name, prompt_type_old, \
                    gr.Slider.update(maximum=256), \
                    gr.Slider.update(maximum=256)

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
            if server_name == no_server_str:
                server_name = ''
            all_kwargs1['inference_server'] = server_name.strip()

            model1, tokenizer1, device1 = get_model(reward_type=False,
                                                    **get_kwargs(get_model, exclude_names=['reward_type'],
                                                                 **all_kwargs1))
            clear_torch_cache()

            tokenizer_base_model = model_name
            prompt_dict1, error0 = get_prompt(prompt_type1, '',
                                              chat=False, context='', reduced=False, making_context=False,
                                              return_dict=True)
            model_state_new = dict(model=model1, tokenizer=tokenizer1, device=device1,
                                   base_model=model_name, tokenizer_base_model=tokenizer_base_model,
                                   lora_weights=lora_weights, inference_server=server_name,
                                   prompt_type=prompt_type1, prompt_dict=prompt_dict1,
                                   )

            max_max_new_tokens1 = get_max_max_new_tokens(model_state_new, **kwargs)

            if kwargs['debug']:
                print("Post-switch GPU memory: %s" % get_torch_allocated(), flush=True)
            return model_state_new, model_name, lora_weights, server_name, prompt_type1, \
                gr.Slider.update(maximum=max_max_new_tokens1), \
                gr.Slider.update(maximum=max_max_new_tokens1)

        def get_prompt_str(prompt_type1, prompt_dict1, which=0):
            if prompt_type1 in ['', None]:
                print("Got prompt_type %s: %s" % (which, prompt_type1), flush=True)
                return str({})
            prompt_dict1, prompt_dict_error = get_prompt(prompt_type1, prompt_dict1, chat=False, context='',
                                                         reduced=False, making_context=False, return_dict=True)
            if prompt_dict_error:
                return str(prompt_dict_error)
            else:
                # return so user can manipulate if want and use as custom
                return str(prompt_dict1)

        get_prompt_str_func1 = functools.partial(get_prompt_str, which=1)
        get_prompt_str_func2 = functools.partial(get_prompt_str, which=2)
        prompt_type.change(fn=get_prompt_str_func1, inputs=[prompt_type, prompt_dict], outputs=prompt_dict)
        prompt_type2.change(fn=get_prompt_str_func2, inputs=[prompt_type2, prompt_dict2], outputs=prompt_dict2)

        def dropdown_prompt_type_list(x):
            return gr.Dropdown.update(value=x)

        def chatbot_list(x, model_used_in):
            return gr.Textbox.update(label=f'h2oGPT [Model: {model_used_in}]')

        load_model_args = dict(fn=load_model,
                               inputs=[model_choice, lora_choice, server_choice, model_state, prompt_type,
                                       model_load8bit_checkbox, model_infer_devices_checkbox, model_gpu],
                               outputs=[model_state, model_used, lora_used, server_used,
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
                                inputs=[model_choice2, lora_choice2, server_choice2, model_state2, prompt_type2,
                                        model_load8bit_checkbox2, model_infer_devices_checkbox2, model_gpu2],
                                outputs=[model_state2, model_used2, lora_used2, server_used2,
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

        def dropdown_model_lora_server_list(model_list0, model_x,
                                            lora_list0, lora_x,
                                            server_list0, server_x,
                                            model_used1, lora_used1, server_used1,
                                            model_used2, lora_used2, server_used2,
                                            ):
            model_new_state = [model_list0[0] + [model_x]]
            model_new_options = [*model_new_state[0]]
            x1 = model_x if model_used1 == no_model_str else model_used1
            x2 = model_x if model_used2 == no_model_str else model_used2
            ret1 = [gr.Dropdown.update(value=x1, choices=model_new_options),
                    gr.Dropdown.update(value=x2, choices=model_new_options),
                    '', model_new_state]

            lora_new_state = [lora_list0[0] + [lora_x]]
            lora_new_options = [*lora_new_state[0]]
            # don't switch drop-down to added lora if already have model loaded
            x1 = lora_x if model_used1 == no_model_str else lora_used1
            x2 = lora_x if model_used2 == no_model_str else lora_used2
            ret2 = [gr.Dropdown.update(value=x1, choices=lora_new_options),
                    gr.Dropdown.update(value=x2, choices=lora_new_options),
                    '', lora_new_state]

            server_new_state = [server_list0[0] + [server_x]]
            server_new_options = [*server_new_state[0]]
            # don't switch drop-down to added server if already have model loaded
            x1 = server_x if model_used1 == no_model_str else server_used1
            x2 = server_x if model_used2 == no_model_str else server_used2
            ret3 = [gr.Dropdown.update(value=x1, choices=server_new_options),
                    gr.Dropdown.update(value=x2, choices=server_new_options),
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
        callback.setup(inputs_list + [text_output, text_output2] + text_outputs, "flagged_data_points")
        flag_btn.click(lambda *args: callback.flag(args), inputs_list + [text_output, text_output2] + text_outputs,
                       None,
                       preprocess=False,
                       api_name='flag' if allow_api else None, queue=False)
        flag_btn_nochat.click(lambda *args: callback.flag(args), inputs_list + [text_output_nochat], None,
                              preprocess=False,
                              api_name='flag_nochat' if allow_api else None, queue=False)

        def get_system_info():
            if is_public:
                time.sleep(10)  # delay to avoid spam since queue=False
            return gr.Textbox.update(value=system_info_print())

        system_event = system_btn.click(get_system_info, outputs=system_text,
                                        api_name='system_info' if allow_api else None, queue=False)

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

        get_system_info_dict_func = functools.partial(get_system_info_dict, **all_kwargs)

        system_dict_event = system_btn2.click(get_system_info_dict_func,
                                              inputs=system_input,
                                              outputs=system_text2,
                                              api_name='system_info_dict' if allow_api else None,
                                              queue=False,  # queue to avoid spam
                                              )

        def get_hash():
            return kwargs['git_hash']

        system_btn3.click(get_hash,
                          outputs=system_text3,
                          api_name='system_hash' if allow_api else None,
                          queue=False,
                          )

        # don't pass text_output, don't want to clear output, just stop it
        # cancel only stops outer generation, not inner generation or non-generation
        stop_btn.click(lambda: None, None, None,
                       cancels=submits1 + submits2 + submits3 +
                               submits4 +
                               [submit_event_nochat, submit_event_nochat2] +
                               [eventdb1, eventdb2, eventdb3,
                                eventdb4, eventdb5, eventdb6] +
                               [eventdb7, eventdb8, eventdb9]
                       ,
                       queue=False, api_name='stop' if allow_api else None).then(clear_torch_cache, queue=False)

        def count_chat_tokens(model_state1, chat1, prompt_type1, prompt_dict1,
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
                langchain_mode1 = 'ChatLLM'
                # fake user message to mimic bot()
                chat1 = copy.deepcopy(chat1)
                chat1 = chat1 + [['user_message1', None]]
                model_max_length1 = tokenizer.model_max_length
                context1 = history_to_context(chat1, langchain_mode1, prompt_type1, prompt_dict1, chat1,
                                              model_max_length1,
                                              memory_restriction_level1, keep_sources_in_context1)
                return str(tokenizer(context1, return_tensors="pt")['input_ids'].shape[1])
            else:
                return "N/A"

        count_chat_tokens_func = functools.partial(count_chat_tokens,
                                                   memory_restriction_level1=memory_restriction_level,
                                                   keep_sources_in_context1=kwargs['keep_sources_in_context'])
        count_chat_tokens_btn.click(fn=count_chat_tokens,
                                    inputs=[model_state, text_output, prompt_type, prompt_dict],
                                    outputs=chat_token_count, api_name='count_tokens' if allow_api else None)

        demo.load(None, None, None, _js=get_dark_js() if kwargs['h2ocolors'] and False else None)  # light best

    demo.queue(concurrency_count=kwargs['concurrency_count'], api_open=kwargs['api_open'])
    favicon_path = "h2o-logo.svg"
    if not os.path.isfile(favicon_path):
        print("favicon_path=%s not found" % favicon_path, flush=True)
        favicon_path = None

    scheduler = BackgroundScheduler()
    scheduler.add_job(func=clear_torch_cache, trigger="interval", seconds=20)
    if is_public and \
            kwargs['base_model'] not in non_hf_types:
        # FIXME: disable for gptj, langchain or gpt4all modify print itself
        # FIXME: and any multi-threaded/async print will enter model output!
        scheduler.add_job(func=ping, trigger="interval", seconds=60)
    scheduler.add_job(func=ping_gpu, trigger="interval", seconds=60 * 10)
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
        from gpt_langchain import get_metadatas
        metadatas = get_metadatas(db1[0])
        source_list = sorted(set([x['source'] for x in metadatas]))
        source_files_added = '\n'.join(source_list)
    elif langchain_mode in dbs and dbs[langchain_mode] is not None:
        from gpt_langchain import get_metadatas
        db1 = dbs[langchain_mode]
        metadatas = get_metadatas(db1)
        source_list = sorted(set([x['source'] for x in metadatas]))
        source_files_added = '\n'.join(source_list)
    else:
        source_list = []
        source_files_added = "None"
    sources_dir = "sources_dir"
    makedirs(sources_dir)
    sources_file = os.path.join(sources_dir, 'sources_%s_%s' % (langchain_mode, str(uuid.uuid4())))
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
            return None, langchain_mode, db1, x, y, source_files_added
        else:
            return None, langchain_mode, x, y, source_files_added
    finally:
        clear_torch_cache()


def _update_user_db(file, db1, x, y, chunk, chunk_size, dbs=None, db_type=None, langchain_mode='UserData',
                    user_path=None,
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

    if dbs is None:
        dbs = {}
    assert isinstance(dbs, dict), "Wrong type for dbs: %s" % str(type(dbs))
    # assert db_type in ['faiss', 'chroma'], "db_type %s not supported" % db_type
    from gpt_langchain import add_to_db, get_db, path_to_docs
    # handle case of list of temp buffer
    if isinstance(file, list) and len(file) > 0 and hasattr(file[0], 'name'):
        file = [x.name for x in file]
    # handle single file of temp buffer
    if hasattr(file, 'name'):
        file = file.name
    if not isinstance(file, (list, tuple, typing.Generator)) and isinstance(file, str):
        file = [file]

    if langchain_mode == 'UserData' and user_path is not None:
        # move temp files from gradio upload to stable location
        for fili, fil in enumerate(file):
            if isinstance(fil, str):
                if fil.startswith('/tmp/gradio/'):
                    new_fil = os.path.join(user_path, os.path.basename(fil))
                    if os.path.isfile(new_fil):
                        remove(new_fil)
                    try:
                        shutil.move(fil, new_fil)
                    except FileExistsError:
                        pass
                    file[fili] = new_fil

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
                db, num_new_sources, new_sources_metadata = add_to_db(db1[0], sources, db_type=db_type,
                                                                      use_openai_embedding=use_openai_embedding,
                                                                      hf_embedding_model=hf_embedding_model)
            else:
                # in testing expect:
                # assert len(db1) == 2 and db1[1] is None, "Bad MyData db: %s" % db1
                # for production hit, when user gets clicky:
                assert len(db1) == 2, "Bad MyData db: %s" % db1
                # then create
                # assign fresh hash for this user session, so not shared
                # if added has to original state and didn't change, then would be shared db for all users
                db1[1] = str(uuid.uuid4())
                persist_directory = os.path.join(scratch_base_dir, 'db_dir_%s_%s' % (langchain_mode, db1[1]))
                db = get_db(sources, use_openai_embedding=use_openai_embedding,
                            db_type=db_type,
                            persist_directory=persist_directory,
                            langchain_mode=langchain_mode,
                            hf_embedding_model=hf_embedding_model)
            if db is None:
                db1[1] = None
            else:
                db1[0] = db
            source_files_added = get_source_files(db=db1[0], exceptions=exceptions)
            return None, langchain_mode, db1, x, y, source_files_added
        else:
            from gpt_langchain import get_persist_directory
            persist_directory = get_persist_directory(langchain_mode)
            if langchain_mode in dbs and dbs[langchain_mode] is not None:
                # then add
                db, num_new_sources, new_sources_metadata = add_to_db(dbs[langchain_mode], sources, db_type=db_type,
                                                                      use_openai_embedding=use_openai_embedding,
                                                                      hf_embedding_model=hf_embedding_model)
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
            return None, langchain_mode, x, y, source_files_added


def get_db(db1, langchain_mode, dbs=None):
    with filelock.FileLock("db_%s.lock" % langchain_mode.replace(' ', '_')):
        if langchain_mode in ['wiki_full']:
            # NOTE: avoid showing full wiki.  Takes about 30 seconds over about 90k entries, but not useful for now
            db = None
        elif langchain_mode == 'MyData' and len(db1) > 0 and db1[0] is not None:
            db = db1[0]
        elif dbs is not None and langchain_mode in dbs and dbs[langchain_mode] is not None:
            db = dbs[langchain_mode]
        else:
            db = None
    return db


def get_source_files_given_langchain_mode(db1, langchain_mode='UserData', dbs=None):
    db = get_db(db1, langchain_mode, dbs=dbs)
    if langchain_mode in ['ChatLLM', 'LLM'] or db is None:
        return "Sources: N/A"
    return get_source_files(db=db, exceptions=None)


def get_source_files(db=None, exceptions=None, metadatas=None):
    if exceptions is None:
        exceptions = []

    # only should be one source, not confused
    # assert db is not None or metadatas is not None
    # clicky user
    if db is None and metadatas is None:
        return "No Sources at all"

    if metadatas is None:
        source_label = "Sources:"
        if db is not None:
            from gpt_langchain import get_metadatas
            metadatas = get_metadatas(db)
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
