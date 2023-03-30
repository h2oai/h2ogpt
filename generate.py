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

from finetune import get_loaders, example_data_points, generate_prompt, get_githash


def main(
        load_8bit: bool = False,
        load_half: bool = True,
        base_model: str = "EleutherAI/gpt-j-6B",
        tokenizer_base_model: str = None,
        lora_weights: str = "",
        prompt_type: int = 1,
        temperature: float = 0.1,
        top_p: float = 0.75,
        top_k: int = 40,
        num_beams: int = 4,
        llama_type: bool = False,
        debug: bool = False,
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model="
    )
    model_loader, tokenizer_loader = get_loaders(llama_type=llama_type)
    if tokenizer_base_model is None:
        tokenizer_base_model = base_model

    tokenizer = tokenizer_loader.from_pretrained(tokenizer_base_model)
    if device == "cuda":
        # directly to GPU
        if load_8bit:
            model = model_loader.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            model = model_loader.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            ).to(device)
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        if not load_8bit and load_half:
            model.half()
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

    if device != "cuda":
        # NOTE: if cuda, already done at once into GPU
        if not load_8bit and load_half:
            model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            instruction,
            input=None,
            prompt_type_choice=0,
            temperature_choice=0.1,
            top_p_choice=0.75,
            top_k_choice=40,
            num_beams_choice=4,
            max_new_tokens=128,
            do_sample=False,
            **kwargs,
    ):
        data_point = dict(instruction=instruction, input=input)
        prompt, pre_response, terminate_response = generate_prompt(data_point, prompt_type_choice)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature_choice,
            top_p=top_p_choice,
            top_k=top_k_choice,
            num_beams=num_beams_choice,
            do_sample=do_sample,
            **kwargs,
        )
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        s = outputs.sequences[0]
        output = tokenizer.decode(s)

        if debug:
            print("prompt: ", prompt, flush=True)
            print("output: ", output, flush=True)

        def clean_response(response):
            meaningless_words = ['<pad>', '</s>', '<|endoftext|>']
            for word in meaningless_words:
                response = response.replace(word, "")
            response = response.strip("\n")
            return response
        output = clean_response(output)
        if prompt_type_choice == -1:
            return output
        else:
            # find first instance of prereponse
            # prompt sometimes has odd characters, that mutate length,
            # so can't go by length alone
            output = output.split(pre_response)[1]
            if terminate_response:
                finds = []
                for term in terminate_response:
                    finds.append(output.find(term))
                finds = [x for x in finds if x >= 0]
                if len(finds) > 0:
                    termi = finds[0]
                    return output[:termi].strip()
                else:
                    return output.strip()
            else:
                return output.strip()

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2, label="Instruction", placeholder="Who is smarter, Einstein or Newton?"
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(minimum=-1, maximum=3, value=prompt_type, step=1, label="Prompt Type"),
            gr.components.Slider(minimum=0, maximum=1, value=temperature, label="Temperature"),
            gr.components.Slider(minimum=0, maximum=1, value=top_p, label="Top p"),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=top_k, label="Top k"
            ),
            gr.components.Slider(minimum=1, maximum=8, step=1, value=num_beams, label="Beams",
                                 info="Uses more GPU memory/compute"),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Checkbox(label="Sample", info="Do sample"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="H2O-LLM",
        description="Model %s Instruct dataset.  "
                    "For more information, visit [the project's website](https://github.com/h2oai/h2o-llm)."
                    "\nCommand: %s\nHash: %s" % (base_model, str(' '.join(sys.argv)), get_githash()),
    ).launch(share=True)


def test_test_prompt(prompt_type=0, data_point=0):
    example_data_point = example_data_points[data_point]
    example_data_point.pop('output', None)
    return generate_prompt(example_data_point, prompt_type)


if __name__ == "__main__":
    print("""
    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port=1234 generate.py --base_model='EleutherAI/gpt-j-6B' --lora_weights=lora-alpaca_6B
    python generate.py --base_model='EleutherAI/gpt-j-6B' --lora_weights='lora-alpaca_6B'
    python generate.py --base_model='EleutherAI/gpt-neox-20b' --lora_weights='lora-alpaca_20B'
    
    # generate without lora weights, no prompt
    python generate.py --base_model='EleutherAI/gpt-neox-20b' --prompt_type=-1
    python generate.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --prompt_type=2

    python generate.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --prompt_type=3 --lora_weights='lora_20B_daifaq'
    # OpenChatKit settings:
    python generate.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --prompt_type=2 --debug=True --num_beams=1 --temperature=0.6 --top_k=40 --top_p=1.0
    
    """, flush=True)
    fire.Fire(main)
