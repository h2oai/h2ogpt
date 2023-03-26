import sys

import fire
import torch
from peft import PeftModel
from transformers import GenerationConfig
import gradio as gr


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


from finetune import get_loaders, get_prompt, example_data_point


def main(
        load_8bit: bool = False,
        base_model: str = "EleutherAI/gpt-j-6B",
        lora_weights: str = "",
        prompt_type: int= 1 ,
        llama_type: bool = False,
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model="
    )
    model_loader, tokenizer_loader = get_loaders(llama_type=llama_type)

    tokenizer = tokenizer_loader.from_pretrained(base_model)
    if device == "cuda":
        model = model_loader.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
    elif device == "mps":
        model = model_loader.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        model = model_loader.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

    # unwind broken decapoda-research config
    if llama_type:
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            instruction,
            input=None,
            prompt_type_choice=0,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
            **kwargs,
    ):
        prompt, preresponse = generate_test_prompt(instruction, prompt_type_choice, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        if prompt_type == -1:
            return output
        else:
            return output.split(preresponse)[1].strip()

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2, label="Instruction", placeholder="Who is smarter, Einstein or Newton?"
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(minimum=-1, maximum=1, value=prompt_type, step=1, label="Prompt Type"),
            gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
            gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(minimum=1, maximum=4, step=1, value=4, label="Beams"),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="H2O-LLM",
        description="Model %s Instruct dataset.  "
                    "For more information, visit [the project's website](https://github.com/h2oai/h2o-llm)." % base_model,
    ).launch()


def generate_test_prompt(instruction, prompt_type, input=None):
    promptA, promptB, PreInstruct, PreInput, PreResponse = get_prompt(prompt_type)

    if input:
        return f"""{promptA}
{PreInstruct}
{instruction}
{PreInput}
{input}
{PreResponse}
""", PreResponse.strip()
    else:
        return f"""{promptB}
{PreInstruct}
{instruction}
{PreResponse}
""", PreResponse.strip()


def test_test_prompt(prompt_type=0):
    print(generate_test_prompt(example_data_point['instruction'], prompt_type, example_data_point['input']))


if __name__ == "__main__":
    print("""
    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port=1234 generate.py --base_model='EleutherAI/gpt-j-6B' --lora_weights=lora-alpaca_6B
    python generate.py --base_model='EleutherAI/gpt-j-6B' --lora_weights='lora-alpaca_6B'
    python generate.py --base_model='EleutherAI/gpt-neox-20b' --lora_weights='lora-alpaca_20B'
    
    # generate without lora weights, no prompt
    python generate.py --base_model='EleutherAI/gpt-neox-20b' --prompt_type=-1
    python generate.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --prompt_type=-1
    
    """, flush=True)
    fire.Fire(main)
