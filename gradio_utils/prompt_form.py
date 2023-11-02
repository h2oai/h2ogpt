import os
import math

import gradio as gr


def make_chatbots(output_label0, output_label0_model2, **kwargs):
    visible_models = kwargs['visible_models']
    all_models = kwargs['all_models']

    text_outputs = []
    chat_kwargs = []
    min_width = 250 if kwargs['gradio_size'] in ['small', 'large', 'medium'] else 160
    for model_state_locki, model_state_lock in enumerate(kwargs['model_states']):
        if os.environ.get('DEBUG_MODEL_LOCK'):
            model_name = model_state_lock["base_model"] + " : " + model_state_lock["inference_server"]
        else:
            model_name = model_state_lock["base_model"]
        output_label = f'h2oGPT [{model_name}]'
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
    elif nrows == 2:
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii >= len_visible / 2:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii < len_visible / 2:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
    elif nrows == 3:
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii >= 1 * len_visible / 3:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii < 1 * len_visible / 3 or mii >= 2 * len_visible / 3:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii < 2 * len_visible / 3:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
    elif nrows >= 4:
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii >= 1 * len_visible / 4:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii < 1 * len_visible / 4 or mii >= 2 * len_visible / 4:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii < 2 * len_visible / 4 or mii >= 3 * len_visible / 4:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii < 3 * len_visible / 4:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1))

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
