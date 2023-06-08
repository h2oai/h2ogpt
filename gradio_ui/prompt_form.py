import gradio as gr


def make_prompt_form(kwargs):
    if kwargs['input_lines'] > 1:
        instruction_label = "You (Shift-Enter or push Submit to send message, use Enter for multiple input lines)"
    else:
        instruction_label = "You (Enter or push Submit to send message, shift-enter for more lines)"

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

    return instruction, submit, stop_btn
