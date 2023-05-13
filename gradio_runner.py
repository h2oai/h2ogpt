import copy
import functools
import inspect
import os
import sys

from gradio_themes import H2oTheme, SoftTheme, get_h2o_title, get_simple_title, get_dark_js
from prompter import Prompter
from utils import get_githash, flatten_list, zip_data, s3up, clear_torch_cache, get_torch_allocated, system_info_print, \
    ping
from finetune import prompt_type_to_model_name, prompt_types_strings, generate_prompt, inv_prompt_type_to_model_lower
from generate import get_model, languages_covered, evaluate, eval_func_param_names, score_qa

import gradio as gr
from apscheduler.schedulers.background import BackgroundScheduler


def go_gradio(**kwargs):
    allow_api = kwargs['allow_api']
    is_public = kwargs['is_public']
    is_hf = kwargs['is_hf']
    is_low_mem = kwargs['is_low_mem']
    n_gpus = kwargs['n_gpus']
    admin_pass = kwargs['admin_pass']
    model_state0 = kwargs['model_state0']
    score_model_state0 = kwargs['score_model_state0']
    queue = True

    # easy update of kwargs needed for evaluate() etc.
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
    if kwargs['verbose']:
        description = f"""Model {kwargs['base_model']} Instruct dataset.
                      For more information, visit our GitHub pages: [h2oGPT](https://github.com/h2oai/h2ogpt) and [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio).
                      Command: {str(' '.join(sys.argv))}
                      Hash: {get_githash()}
                      """
    else:
        description = "For more information, visit our GitHub pages: [h2oGPT](https://github.com/h2oai/h2ogpt) and [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio)<br>"
    description += "If this host is busy, try [gpt.h2o.ai 12B](https://gpt.h2o.ai) and [30B](http://gpt2.h2o.ai) and [HF Spaces1 12B](https://huggingface.co/spaces/h2oai/h2ogpt-chatbot) and [HF Spaces2 12B](https://huggingface.co/spaces/h2oai/h2ogpt-chatbot2)<br>"
    description += """<p>By using h2oGPT, you accept our [Terms of Service](https://github.com/h2oai/h2ogpt/blob/main/tos.md)</p>"""

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

    theme = H2oTheme() if kwargs['h2ocolors'] else SoftTheme()
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

    with demo:
        # avoid actual model/tokenizer here or anything that would be bad to deepcopy
        # https://github.com/gradio-app/gradio/issues/3558
        model_state = gr.State(['model', 'tokenizer', kwargs['device'], kwargs['base_model']])
        model_state2 = gr.State([None, None, None, None])
        model_options_state = gr.State([model_options])
        lora_options_state = gr.State([lora_options])
        gr.Markdown(f"""
            {get_h2o_title(title) if kwargs['h2ocolors'] else get_simple_title(title)}

            {description}
            {task_info_md}
            """)
        if is_hf:
            gr.HTML(
                '''<center><a href="https://huggingface.co/spaces/h2oai/h2ogpt-chatbot?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>Duplicate this Space to skip the queue and run in a private space</center>''')

        # go button visible if
        base_wanted = kwargs['base_model'] != no_model_str and kwargs['login_mode_if_model0']
        go_btn = gr.Button(value="ENTER", visible=base_wanted, variant="primary")
        normal_block = gr.Row(visible=not base_wanted)
        with normal_block:
            with gr.Tabs():
                with gr.Row():
                    col_nochat = gr.Column(visible=not kwargs['chat'])
                    with col_nochat:  # FIXME: for model comparison, and check rest
                        text_output_nochat = gr.Textbox(lines=5, label=output_label0)
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
                            clear = gr.Button("New Conversation")
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
                with gr.TabItem("Input/Output"):
                    with gr.Row():
                        if 'mbart-' in kwargs['model_lower']:
                            src_lang = gr.Dropdown(list(languages_covered().keys()),
                                                   value=kwargs['src_lang'],
                                                   label="Input Language")
                            tgt_lang = gr.Dropdown(list(languages_covered().keys()),
                                                   value=kwargs['tgt_lang'],
                                                   label="Output Language")
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
                                 max_beams = 8 if not (is_low_mem or is_public) else 1
                            else:
                                max_beams = 1
                            num_beams = gr.Slider(minimum=1, maximum=max_beams, step=1,
                                                  value=min(max_beams, kwargs['num_beams']), label="Beams",
                                                  info="Number of searches for optimal overall probability.  "
                                                       "Uses more GPU memory/compute")
                            max_max_new_tokens = 2048 if not is_low_mem else kwargs['max_new_tokens']
                            max_new_tokens = gr.Slider(
                                minimum=1, maximum=max_max_new_tokens, step=1,
                                value=min(max_max_new_tokens, kwargs['max_new_tokens']), label="Max output length",
                            )
                            min_new_tokens = gr.Slider(
                                minimum=0, maximum=max_max_new_tokens, step=1,
                                value=min(max_max_new_tokens, kwargs['min_new_tokens']), label="Min output length",
                            )
                            early_stopping = gr.Checkbox(label="EarlyStopping", info="Stop early in beam search",
                                                         value=kwargs['early_stopping'])
                            max_max_time = 60 * 5 if not is_public else 60 * 2
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

                with gr.TabItem("Models"):
                    load_msg = "Load-Unload Model/LORA" if not is_public \
                        else "LOAD-UNLOAD DISABLED FOR HOSTED DEMO"
                    load_msg2 = "Load-Unload Model/LORA 2" if not is_public \
                        else "LOAD-UNLOAD DISABLED FOR HOSTED DEMO 2"
                    compare_checkbox = gr.components.Checkbox(label="Compare Mode",
                                                              value=False, visible=not is_public)
                    with gr.Row():
                        n_gpus_list = [str(x) for x in list(range(-1, n_gpus))]
                        with gr.Column():
                            with gr.Row():
                                with gr.Column(scale=50):
                                    model_choice = gr.Dropdown(model_options_state.value[0], label="Choose Model",
                                                               value=kwargs['base_model'])
                                    lora_choice = gr.Dropdown(lora_options_state.value[0], label="Choose LORA",
                                                              value=kwargs['lora_weights'], visible=kwargs['show_lora'])
                                with gr.Column(scale=1):
                                    load_model_button = gr.Button(load_msg)
                                    model_load8bit_checkbox = gr.components.Checkbox(
                                        label="Load 8-bit [requires support]",
                                        value=kwargs['load_8bit'])
                                    model_infer_devices_checkbox = gr.components.Checkbox(
                                        label="Choose Devices [If not Checked, use all GPUs]",
                                        value=kwargs['infer_devices'])
                                    model_gpu = gr.Dropdown(n_gpus_list,
                                                            label="GPU ID 2 [-1 = all GPUs, if Choose is enabled]",
                                                            value=kwargs['gpu_id'])
                                    model_used = gr.Textbox(label="Current Model", value=kwargs['base_model'],
                                                            interactive=False)
                                    lora_used = gr.Textbox(label="Current LORA", value=kwargs['lora_weights'],
                                                           visible=kwargs['show_lora'], interactive=False)
                            with gr.Row():
                                with gr.Column(scale=50):
                                    new_model = gr.Textbox(label="New Model HF name/path")
                                    new_lora = gr.Textbox(label="New LORA HF name/path", visible=kwargs['show_lora'])
                                with gr.Column(scale=1):
                                    add_model_button = gr.Button("Add new model name")
                                    add_lora_button = gr.Button("Add new LORA name", visible=kwargs['show_lora'])
                        col_model2 = gr.Column(visible=False)
                        with col_model2:
                            with gr.Row():
                                with gr.Column(scale=50):
                                    model_choice2 = gr.Dropdown(model_options_state.value[0], label="Choose Model 2",
                                                                value=no_model_str)
                                    lora_choice2 = gr.Dropdown(lora_options_state.value[0], label="Choose LORA 2",
                                                               value=no_lora_str,
                                                               visible=kwargs['show_lora'])
                                with gr.Column(scale=1):
                                    load_model_button2 = gr.Button(load_msg2)
                                    model_load8bit_checkbox2 = gr.components.Checkbox(
                                        label="Load 8-bit 2 [requires support]",
                                        value=kwargs['load_8bit'])
                                    model_infer_devices_checkbox2 = gr.components.Checkbox(
                                        label="Choose Devices 2 [If not Checked, use all GPUs]",
                                        value=kwargs[
                                            'infer_devices'])
                                    model_gpu2 = gr.Dropdown(n_gpus_list,
                                                             label="GPU ID [-1 = all GPUs, if choose is enabled]",
                                                             value=kwargs['gpu_id'])
                                    # no model/lora loaded ever in model2 by default
                                    model_used2 = gr.Textbox(label="Current Model 2", value=no_model_str)
                                    lora_used2 = gr.Textbox(label="Current LORA 2", value=no_lora_str,
                                                            visible=kwargs['show_lora'])
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
                                system_text = gr.Textbox(label='System Info', interactive=False)

                            with gr.Row():
                                zip_btn = gr.Button("Zip")
                                zip_text = gr.Textbox(label="Zip file name", interactive=False)
                                file_output = gr.File()
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
                    description += """<i><li>By using h2oGPT, you accept our <a href="https://github.com/h2oai/h2ogpt/blob/main/tos.md">Terms of Service</a></i></li></ul></p>"""
                    gr.Markdown(value=description, show_label=False, interactive=False)

        # Get flagged data
        zip_data1 = functools.partial(zip_data, root_dirs=['flagged_data_points', kwargs['save_dir']])
        zip_btn.click(zip_data1, inputs=None, outputs=[file_output, zip_text], queue=False)
        s3up_btn.click(s3up, inputs=zip_text, outputs=s3up_text, queue=False)

        def check_admin_pass(x):
            return gr.update(visible=x == admin_pass)

        def close_admin(x):
            return gr.update(visible=not (x == admin_pass))

        admin_btn.click(check_admin_pass, inputs=admin_pass_textbox, outputs=system_row, queue=False) \
            .then(close_admin, inputs=admin_pass_textbox, outputs=admin_row, queue=False)

        # Get inputs to evaluate()
        # don't deepcopy, can contain model itself
        all_kwargs = kwargs.copy()
        all_kwargs.update(locals())
        inputs_list = get_inputs_list(all_kwargs, kwargs['model_lower'])
        from functools import partial
        kwargs_evaluate = {k: v for k, v in all_kwargs.items() if k in inputs_kwargs_list}
        # ensure present
        for k in inputs_kwargs_list:
            assert k in kwargs_evaluate, "Missing %s" % k
        fun = partial(evaluate,
                      **kwargs_evaluate)
        fun2 = partial(evaluate,
                       **kwargs_evaluate)

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

            max_length_tokenize = 512 if is_low_mem else 2048
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
                           inputs=inputs_list + [text_output2],
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
            prompter = Prompter(prompt_type1, debug=kwargs['debug'], chat=chat1, stream_output=stream_output1)
            if user_message1 in ['']:
                # e.g. when user just hits enter in textbox,
                # else will have <human>: <bot>: on single line, which seems to be "ok" for LLM but not usual
                user_message1 = '\n'

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
            history = args_list[-1]  # model_state is -2
            if retry and history:
                history.pop()
            if not history:
                print("No history", flush=True)
                history = [['', None]]
                yield history, ''
                return
            # ensure output will be unique to models
            history = copy.deepcopy(history)
            instruction1 = history[-1][0]
            context1 = ''
            if kwargs['chat_history'] > 0:
                prompt_type_arg_id = eval_func_param_names.index('prompt_type')
                prompt_type1 = args_list[prompt_type_arg_id]
                chat_arg_id = eval_func_param_names.index('chat')
                chat1 = args_list[chat_arg_id]
                context1 = ''
                for histi in range(len(history) - 1):
                    data_point = dict(instruction=history[histi][0], input='', output=history[histi][1])
                    prompt, pre_response, terminate_response, chat_sep = generate_prompt(data_point, prompt_type1,
                                                                                         chat1, reduced=True)
                    # md -> back to text, maybe not super improtant if model trained enough
                    prompt = prompt.replace('<br>', chat_sep)
                    context1 += prompt
                    if not context1.endswith(chat_sep):
                        context1 += chat_sep

                _, pre_response, terminate_response, chat_sep = generate_prompt({}, prompt_type1, chat1,
                                                                                reduced=True)
                if context1 and not context1.endswith(chat_sep):
                    context1 += chat_sep  # ensure if terminates abruptly, then human continues on next line
            args_list[0] = instruction1  # override original instruction with history from user
            # only include desired chat history
            args_list[2] = context1[-kwargs['chat_history']:]
            model_state1 = args_list[-2]
            if model_state1[0] is None or model_state1[0] == no_model_str:
                history = [['', None]]
                yield history, ''
                return
            args_list = args_list[:-2]
            fun1 = partial(evaluate,
                           model_state1,
                           **kwargs_evaluate)
            try:
                for output in fun1(*tuple(args_list)):
                    bot_message = output
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
                        inputs=inputs_list + [model_state] + [text_output],
                        outputs=[text_output, exception_text],
                        )
        retry_bot_args = dict(fn=functools.partial(bot, retry=True),
                              inputs=inputs_list + [model_state] + [text_output],
                              outputs=[text_output, exception_text],
                              )
        undo_user_args = dict(fn=functools.partial(user, undo=True),
                              inputs=inputs_list + [text_output],
                              outputs=text_output,
                              )

        # MODEL2
        user_args2 = dict(fn=functools.partial(user, sanitize_user_prompt=kwargs['sanitize_user_prompt'], model2=True),
                          inputs=inputs_list + [text_output2],
                          outputs=text_output2,
                          )
        bot_args2 = dict(fn=bot,
                         inputs=inputs_list + [model_state2] + [text_output2],
                         outputs=[text_output2, exception_text],
                         )
        retry_bot_args2 = dict(fn=functools.partial(bot, retry=True),
                               inputs=inputs_list + [model_state2] + [text_output2],
                               outputs=[text_output2, exception_text],
                               )
        undo_user_args2 = dict(fn=functools.partial(user, undo=True),
                               inputs=inputs_list + [text_output2],
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

        # in case 2nd model, consume instruction first, so can clear quickly
        # bot doesn't consume instruction itself, just history from user, so why works
        submit_event1a = instruction.submit(**user_args, queue=queue,
                                            api_name='instruction' if allow_api else None)
        submit_event1b = submit_event1a.then(**user_args2, api_name='instruction2' if allow_api else None)
        submit_event1c = submit_event1b.then(clear_instruct, None, instruction) \
            .then(clear_instruct, None, iinput)
        submit_event1d = submit_event1c.then(**bot_args, api_name='instruction_bot' if allow_api else None,
                                             queue=queue)
        submit_event1e = submit_event1d.then(**score_args_submit, api_name='instruction_bot_score' if allow_api else None,
                                             queue=queue)
        submit_event1f = submit_event1e.then(**bot_args2, api_name='instruction_bot2' if allow_api else None,
                                             queue=queue)
        submit_event1g = submit_event1f.then(**score_args2_submit,
                                             api_name='instruction_bot_score2' if allow_api else None, queue=queue)
        submit_event1h = submit_event1g.then(clear_torch_cache)

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

        submit_event4 = undo.click(**undo_user_args, api_name='undo' if allow_api else None) \
            .then(**undo_user_args2, api_name='undo2' if allow_api else None) \
            .then(clear_instruct, None, instruction) \
            .then(clear_instruct, None, iinput) \
            .then(**score_args_submit, api_name='undo_score' if allow_api else None) \
            .then(**score_args2_submit, api_name='undo_score2' if allow_api else None)

        # does both models
        clear.click(lambda: None, None, text_output, queue=False, api_name='clear' if allow_api else None) \
            .then(lambda: None, None, text_output2, queue=False, api_name='clear2' if allow_api else None)
        # NOTE: clear of instruction/iinput for nochat has to come after score,
        # because score for nochat consumes actual textbox, while chat consumes chat history filled by user()
        submit_event_nochat = submit_nochat.click(fun, inputs=[model_state] + inputs_list,
                                                  outputs=text_output_nochat,
                                                  queue=queue,
                                                  api_name='submit_nochat' if allow_api else None) \
            .then(**score_args_nochat, api_name='instruction_bot_score_nochat' if allow_api else None, queue=queue) \
            .then(clear_instruct, None, instruction_nochat) \
            .then(clear_instruct, None, iinput_nochat) \
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
            model1, tokenizer1, device1 = get_model(**all_kwargs1)
            clear_torch_cache()

            if kwargs['debug']:
                print("Post-switch GPU memory: %s" % get_torch_allocated(), flush=True)
            return [model1, tokenizer1, device1, model_name], model_name, lora_weights, prompt_type1

        def dropdown_prompt_type_list(x):
            return gr.Dropdown.update(value=x)

        def chatbot_list(x, model_used_in):
            return gr.Textbox.update(label=f'h2oGPT [Model: {model_used_in}]')

        load_model_args = dict(fn=load_model,
                               inputs=[model_choice, lora_choice, model_state, prompt_type,
                                       model_load8bit_checkbox, model_infer_devices_checkbox, model_gpu],
                               outputs=[model_state, model_used, lora_used, prompt_type])
        prompt_update_args = dict(fn=dropdown_prompt_type_list, inputs=prompt_type, outputs=prompt_type)
        chatbot_update_args = dict(fn=chatbot_list, inputs=[text_output, model_used], outputs=text_output)
        nochat_update_args = dict(fn=chatbot_list, inputs=[text_output_nochat, model_used], outputs=text_output_nochat)
        if not is_public:
            load_model_event = load_model_button.click(**load_model_args) \
                .then(**prompt_update_args) \
                .then(**chatbot_update_args) \
                .then(**nochat_update_args) \
                .then(clear_torch_cache)

        load_model_args2 = dict(fn=load_model,
                                inputs=[model_choice2, lora_choice2, model_state2, prompt_type2,
                                        model_load8bit_checkbox2, model_infer_devices_checkbox2, model_gpu2],
                                outputs=[model_state2, model_used2, lora_used2, prompt_type2])
        prompt_update_args2 = dict(fn=dropdown_prompt_type_list, inputs=prompt_type2, outputs=prompt_type2)
        chatbot_update_args2 = dict(fn=chatbot_list, inputs=[text_output2, model_used2], outputs=text_output2)
        if not is_public:
            load_model_event2 = load_model_button2.click(**load_model_args2) \
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

        compare_checkbox.select(compare_textbox_fun, compare_checkbox, text_output2,
                                api_name="compare_checkbox" if allow_api else None) \
            .then(compare_column_fun, compare_checkbox, col_model2) \
            .then(compare_prompt_fun, compare_checkbox, prompt_type2) \
            .then(compare_textbox_fun, compare_checkbox, score_text2)
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
                                submit_event_nochat],
                       queue=False, api_name='stop' if allow_api else None).then(clear_torch_cache, queue=False)
        demo.load(None, None, None, _js=get_dark_js() if kwargs['h2ocolors'] else None)

    demo.queue(concurrency_count=kwargs['concurrency_count'], api_open=kwargs['api_open'])
    favicon_path = "h2o-logo.svg"

    scheduler = BackgroundScheduler()
    scheduler.add_job(func=clear_torch_cache, trigger="interval", seconds=20)
    if is_public:
        scheduler.add_job(func=ping, trigger="interval", seconds=60)
    scheduler.start()

    demo.launch(share=kwargs['share'], server_name="0.0.0.0", show_error=True,
                favicon_path=favicon_path, prevent_thread_lock=True,
                auth=kwargs['auth'])
    print("Started GUI", flush=True)
    if kwargs['block_gradio_exit']:
        demo.block_thread()


input_args_list = ['model_state']
inputs_kwargs_list = ['debug', 'save_dir', 'sanitize_bot_response', 'model_state0', 'is_low_mem',
                      'raise_generate_gpu_exceptions', 'chat_context', 'concurrency_count', 'lora_weights']


def get_inputs_list(inputs_dict, model_lower):
    """
    map gradio objects in locals() to inputs for evaluate().
    :param inputs_dict:
    :param model_lower:
    :return:
    """
    inputs_list_names = list(inspect.signature(evaluate).parameters)
    inputs_list = []
    for k in inputs_list_names:
        if k == 'kwargs':
            continue
        if k in input_args_list + inputs_kwargs_list:
            # these are added via partial, not taken as input
            continue
        if 'mbart-' not in model_lower and k in ['src_lang', 'tgt_lang']:
            continue
        inputs_list.append(inputs_dict[k])
    return inputs_list
