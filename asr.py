import gradio as gr
from transformers import pipeline
import time

pipe = pipeline("automatic-speech-recognition")

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-medium",
  chunk_length_s=30,
  device='cuda:0',
)


def transcribe(audio, state=""):
    print(audio)
    time.sleep(2)
    text = pipe(audio)["text"]
    state += text + " "
    return state, state


with gr.Blocks() as demo:
  state = gr.State(value="")
  with gr.Row():
      with gr.Column():
        audio = gr.Audio(source="microphone", type="filepath")
      with gr.Column():
        textbox = gr.Textbox()
  audio.stream(fn=transcribe, inputs=[audio, state], outputs=[textbox, state], show_progress=False)

demo.launch(debug=True)