import math

import gradio as gr


def make_chatbots(output_label0, output_label0_model2, **kwargs):
    text_outputs = []
    chat_kwargs = []
    style_kwargs = dict(height=kwargs['height'] or 400)
    for model_state_lock in kwargs['model_states']:
        output_label = f'h2oGPT [{model_state_lock["base_model"]}]'
        chat_kwargs.append(dict(label=output_label, visible=kwargs['model_lock'], elem_classes='chatsmall'))

    if kwargs['model_lock_columns'] == -1:
        kwargs['model_lock_columns'] = len(kwargs['model_states'])

    ncols = kwargs['model_lock_columns']
    nrows = math.ceil(len(kwargs['model_states']) / kwargs['model_lock_columns'])

    if nrows <= 1:
        with gr.Row():
            with gr.Column():
                for chat_kwargs1, model_state_lock in zip(chat_kwargs, kwargs['model_states']):
                    text_outputs.append(gr.Chatbot(**chat_kwargs1).style(**style_kwargs))
    elif nrows == kwargs['model_states']:
        with gr.Row():
            for chat_kwargs1, model_state_lock in zip(chat_kwargs, kwargs['model_states']):
                text_outputs.append(gr.Chatbot(**chat_kwargs1).style(**style_kwargs))
    elif nrows == 2:
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii >= len(kwargs['model_states']) / 2:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1).style(**style_kwargs))
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii < len(kwargs['model_states']) / 2:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1).style(**style_kwargs))
    elif nrows == 3:
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii >= 1 * len(kwargs['model_states']) / 3:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1).style(**style_kwargs))
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii < 1 * len(kwargs['model_states']) / 3 or mii >= 2 * len(kwargs['model_states']) / 3:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1).style(**style_kwargs))
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii < 2 * len(kwargs['model_states']) / 3:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1).style(**style_kwargs))
    elif nrows == 4:
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii >= 1 * len(kwargs['model_states']) / 4:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1).style(**style_kwargs))
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii < 1 * len(kwargs['model_states']) / 4 or mii >= 2 * len(kwargs['model_states']) / 4:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1).style(**style_kwargs))
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii < 2 * len(kwargs['model_states']) / 4 or mii >= 3 * len(kwargs['model_states']) / 4:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1).style(**style_kwargs))
        with gr.Row():
            for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                if mii < 3 * len(kwargs['model_states']) / 4:
                    continue
                text_outputs.append(gr.Chatbot(**chat_kwargs1).style(**style_kwargs))

    with gr.Row():
        text_output = gr.Chatbot(label=output_label0, visible=not kwargs['model_lock']).style(
            height=kwargs['height'] or 400)
        text_output2 = gr.Chatbot(label=output_label0_model2,
                                  visible=False and not kwargs['model_lock']).style(
            height=kwargs['height'] or 400)
    return text_output, text_output2, text_outputs


def make_prompt_form(kwargs):
    if kwargs['input_lines'] > 1:
        instruction_label = "press Shift-Enter or click Submit to send message, press Enter for multiple input lines"
    else:
        instruction_label = "press Enter or click Submit to send message, press Shift-Enter for more lines"

    with gr.Row(elem_id='prompt-form-area'):
        with gr.Column(scale=50):
            instruction = gr.Textbox(
                lines=kwargs['input_lines'],
                label='Ask anything',
                placeholder=kwargs['placeholder_instruction'],
                info=instruction_label,
                elem_id='prompt-form'
            )
            instruction.style(container=True)
        with gr.Row():
            submit = gr.Button(value='Submit', variant='primary').style(full_width=False, size='sm')
            stop_btn = gr.Button(value="Stop", variant='secondary').style(full_width=False, size='sm')

    return instruction, submit, stop_btn
