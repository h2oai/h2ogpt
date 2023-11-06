import os
import math

import gradio as gr


def get_chatbot_name(base_model, model_path_llama, inference_server='', debug=False):
    if not debug:
        inference_server = ''
    else:
        inference_server = ' : ' + inference_server
    if base_model == 'llama':
        model_path_llama = os.path.basename(model_path_llama)
        return f'h2oGPT [Model: {model_path_llama}{inference_server}]'
    else:
        return f'h2oGPT [Model: {base_model}{inference_server}]'


def make_chatbots(output_label0, output_label0_model2, **kwargs):
    visible_models = kwargs['visible_models']
    all_models = kwargs['all_possible_visible_models']

    text_outputs = []
    chat_kwargs = []
    min_width = 250 if kwargs['gradio_size'] in ['small', 'large', 'medium'] else 160
    for model_state_locki, model_state_lock in enumerate(kwargs['model_states']):
        output_label = get_chatbot_name(model_state_lock["base_model"],
                                        model_state_lock["model_path_llama"],
                                        model_state_lock["inference_server"],
                                        debug=bool(os.environ.get('DEBUG_MODEL_LOCK', 0)))
        chat_kwargs.append(dict(render_markdown=kwargs.get('render_markdown', True),
                                label=output_label,
                                elem_classes='chatsmall',
                                height=kwargs['height'] or 400,
                                min_width=min_width,
                                show_copy_button=kwargs['show_copy_button'],
                                visible=kwargs['model_lock'] and (visible_models is None or
                                                                  model_state_locki in visible_models or
                                                                  all_models[model_state_locki] in visible_models
                                                                  )))

    # base view on initial visible choice
    if visible_models:
        len_visible = len(visible_models)
    else:
        len_visible = len(kwargs['model_states'])
    if kwargs['model_lock_columns'] == -1:
        kwargs['model_lock_columns'] = len_visible
    if kwargs['model_lock_columns'] is None:
        kwargs['model_lock_columns'] = 3

    ncols = kwargs['model_lock_columns']
    if kwargs['model_states'] == 0:
        nrows = 0
    else:
        nrows = math.ceil(len_visible / kwargs['model_lock_columns'])

    if kwargs['model_lock_columns'] == 0:
        # not using model_lock
        pass
    elif nrows <= 1:
        with gr.Row():
            for chat_kwargs1, model_state_lock in zip(chat_kwargs, kwargs['model_states']):
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
    elif nrows == kwargs['model_states']:
        with gr.Row():
            for chat_kwargs1, model_state_lock in zip(chat_kwargs, kwargs['model_states']):
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
    elif nrows > 0:
        len_chatbots = len(kwargs['model_states'])
        nrows = math.ceil(len_chatbots / kwargs['model_lock_columns'])
        for nrowi in range(nrows):
            with gr.Row():
                for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                    if mii < nrowi * len_chatbots / nrows or mii >= (1 + nrowi) * len_chatbots / nrows:
                        continue
                    text_outputs.append(gr.Chatbot(**chat_kwargs1))
    if len(kwargs['model_states']) > 0:
        assert len(text_outputs) == len(kwargs['model_states'])

    no_model_lock_chat_kwargs = dict(render_markdown=kwargs.get('render_markdown', True),
                                     elem_classes='chatsmall',
                                     height=kwargs['height'] or 400,
                                     min_width=min_width,
                                     show_copy_button=kwargs['show_copy_button'],
                                     )
    with gr.Row():
        text_output = gr.Chatbot(label=output_label0,
                                 visible=not kwargs['model_lock'],
                                 **no_model_lock_chat_kwargs,
                                 )
        text_output2 = gr.Chatbot(label=output_label0_model2,
                                  visible=False and not kwargs['model_lock'],
                                  **no_model_lock_chat_kwargs)
    return text_output, text_output2, text_outputs
