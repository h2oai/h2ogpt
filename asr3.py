import gradio as gr


def click_js():
    return """function audioRecord() {
    var xPathRes = document.evaluate ('//*[@id="audio"]//button', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null); 
    xPathRes.singleNodeValue.click();}"""


def action(btn):
    """Changes button text on click"""
    if btn == 'Speak': return 'Stop'
    else: return 'Speak'


def check_btn(btn):
    """Checks for correct button text before invoking transcribe()"""
    if btn != 'Speak': raise Exception('Recording...')


def transcribe():
    return 'Success'


with gr.Blocks() as demo:
    msg = gr.Textbox()
    audio_box = gr.Audio(label="Audio", source="microphone", type="filepath", elem_id='audio')

    with gr.Row():
        audio_btn = gr.Button('Speak')
        clear = gr.Button("Clear")

    audio_btn.click(fn=action, inputs=audio_btn, outputs=audio_btn).\
              then(fn=lambda: None, _js=click_js()).\
              then(fn=check_btn, inputs=audio_btn).\
              success(fn=transcribe, outputs=msg)

    clear.click(lambda: None, None, msg, queue=False)

demo.queue().launch(debug=True)