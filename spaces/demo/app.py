import gradio as gr
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Open Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
)

auth_token = os.environ.get("SECRET_TOKEN") or True

from h2oai_pipeline import H2OTextGenerationPipeline

model_name = "h2oai/h2ogpt-oig-oasst1-512-6_9b"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True, use_auth_token=auth_token)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, use_auth_token=auth_token)

generate_text = H2OTextGenerationPipeline(model=model, tokenizer=tokenizer)


def generate(query):
    return generate_text(query, max_new_tokens=150)[0]['generated_text']


examples = [
    "Why is drinking water so healthy?",
    "Is there such a thing as Shallow Learning?",
    "Tell me a funny joke in German",
    "What does the 402 error mean?",
    "Can penguins fly?",
    "What's the secret to a happy life?",
    "Is it easy to train large language models?"
]


def process_example(args):
    for x in generate(args):
        pass
    return x

css = ".generating {visibility: hidden}"

with gr.Blocks(theme=theme) as demo:
    gr.Markdown(
        """<h1><center>h2oGPT</center></h1>
"""
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                instruction = gr.Textbox(placeholder="Enter your question here", label="Question", elem_id="q-input")
            with gr.Row():
                with gr.Row():
                    submit = gr.Button("Generate Answer")
    with gr.Row():
        with gr.Column():
            with gr.Box():
                gr.Markdown("**h2oGPT**")
                output = gr.Markdown()
    with gr.Row():
        gr.Examples(
                    examples=examples,
                    inputs=[instruction],
                    cache_examples=False,
                    fn=process_example,
                    outputs=[output],
                )
    submit.click(generate, inputs=[instruction], outputs=[output], api_name='submit')
    instruction.submit(generate, inputs=[instruction], outputs=[output])

demo.queue(concurrency_count=16).launch(debug=True)
