import os
import math

import gradio as gr


def make_chatbots(output_label0, output_label0_model2, **kwargs):
    text_outputs = []
    chat_kwargs = []
    for model_state_lock in kwargs['model_states']:
        if os.environ.get('DEBUG_MODEL_LOCK'):
            model_name = model_state_lock["base_model"] + " : " + model_state_lock["inference_server"]
        else:
            model_name = model_state_lock["base_model"]
        output_label = f'h2oGPT [{model_name}]'
        min_width = 250 if kwargs['gradio_size'] in ['small', 'large', 'medium'] else 160
        chat_kwargs.append(dict(label=output_label, visible=kwargs['model_lock'], elem_classes='chatsmall',
                                height=kwargs['height'] or 400, min_width=min_width))

    if kwargs['model_lock_columns'] == -1:
        kwargs['model_lock_columns'] = len(kwargs['model_states'])
    if kwargs['model_lock_columns'] is None:
        kwargs['model_lock_columns'] = 3

    ncols = kwargs['model_lock_columns']
    if kwargs['model_states'] == 0:
        nrows = 0
    else:
        nrows = math.ceil(len(kwargs['model_states']) / kwargs['model_lock_columns'])

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
                if mii >= len(kwargs['model_states']) / 2:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii < len(kwargs['model_states']) / 2:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
    elif nrows == 3:
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii >= 1 * len(kwargs['model_states']) / 3:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii < 1 * len(kwargs['model_states']) / 3 or mii >= 2 * len(kwargs['model_states']) / 3:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii < 2 * len(kwargs['model_states']) / 3:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
    elif nrows >= 4:
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii >= 1 * len(kwargs['model_states']) / 4:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii < 1 * len(kwargs['model_states']) / 4 or mii >= 2 * len(kwargs['model_states']) / 4:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii < 2 * len(kwargs['model_states']) / 4 or mii >= 3 * len(kwargs['model_states']) / 4:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii < 3 * len(kwargs['model_states']) / 4:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1))

    with gr.Row():
        text_output = gr.Chatbot(label=output_label0, visible=not kwargs['model_lock'], height=kwargs['height'] or 400)
        text_output2 = gr.Chatbot(label=output_label0_model2,
                                  visible=False and not kwargs['model_lock'], height=kwargs['height'] or 400)
    return text_output, text_output2, text_outputs
