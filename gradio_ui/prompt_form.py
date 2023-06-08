import gradio as gr


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
