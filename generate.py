import sys
from typing import Union

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

from finetune import get_loaders, example_data_points, generate_prompt, get_githash, prompt_types, prompt_types_strings


def main(
        load_8bit: bool = False,
        load_half: bool = True,
        base_model: str = "EleutherAI/gpt-j-6B",
        tokenizer_base_model: str = None,
        lora_weights: str = "",
        prompt_type: Union[int, str] = None,

        # input to generation
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        num_beams: int = None,
        repetition_penalty: float = None,
        num_return_sequences: int = None,
        do_sample: bool = None,
        max_length: int = None,

        llama_type: bool = None,
        debug: bool = False,
        share: bool = True,
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model="
    )
    llama_type = llama_type or "llama" in base_model
    model_loader, tokenizer_loader = get_loaders(llama_type=llama_type, model_name=base_model)
    if tokenizer_base_model is None:
        tokenizer_base_model = base_model

    if tokenizer_loader is not None and not isinstance(tokenizer_loader, str):
        tokenizer = tokenizer_loader.from_pretrained(tokenizer_base_model)
    else:
        tokenizer = tokenizer_loader

    if isinstance(tokenizer, str):
        # already a pipeline, tokenizer_loader is string for task
        model = model_loader(tokenizer,
                             model=base_model,
                             device=0 if device == "cuda" else -1,
                             torch_dtype=torch.float16)
    elif device == "cuda":
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
    if 'gpt2' in base_model.lower():
        # add special tokens that otherwise all share the same id
        tokenizer.add_special_tokens({'bos_token': '<bos>',
                                      'eos_token': '<eos>',
                                      'pad_token': '<pad>'})

    if device != "cuda":
        # NOTE: if cuda, already done at once into GPU
        if not load_8bit and load_half:
            model.half()  # seems to fix bugs for some users.

    if not isinstance(tokenizer, str):
        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    def evaluate(*args, **kwargs):
        try:
            return _evaluate(*args, **kwargs)
        except Exception as e:
            t, v, tb = sys.exc_info()
            import traceback
            ex = ''.join(traceback.format_exception(t, v, tb))
            return str(ex)

    def _evaluate(
            instruction,
            input=None,
            prompt_type_choice=prompt_type,
            temperature_choice=temperature,
            top_p_choice=top_p,
            top_k_choice=top_k,
            num_beams_choice=num_beams,
            max_new_tokens=max_length,
            repetition_penalty_choice=repetition_penalty,
            num_return_sequences_choice=num_return_sequences,
            do_sample_choice=do_sample,
            **kwargs,
    ):
        data_point = dict(instruction=instruction, input=input)
        prompt, pre_response, terminate_response = generate_prompt(data_point, prompt_type_choice)
        if isinstance(tokenizer, str):
            # pipeline
            if tokenizer == "summarization":
                key = 'summary_text'
            else:
                raise RuntimeError("No such task type %s" % tokenizer)
            return model(prompt, max_length=max_length)[0][key]

        inputs = tokenizer(prompt, return_tensors="pt")
        if debug:
            print('input_ids length', len(inputs["input_ids"]), flush=True)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature_choice,
            top_p=top_p_choice,
            top_k=top_k_choice,
            num_beams=num_beams_choice,
            do_sample=do_sample_choice,
            repetition_penalty=repetition_penalty_choice,
            num_return_sequences=num_return_sequences_choice,
            **kwargs,
        )
        with torch.no_grad():
            gen_kwargs = dict(input_ids=input_ids,
                              generation_config=generation_config,
                              return_dict_in_generate=True,
                              output_scores=True,
                              max_new_tokens=max_new_tokens,
                              pad_token_id=tokenizer.eos_token_id,
                              )
            if 'gpt2' in base_model.lower():
                gen_kwargs.update(dict(bos_token_id=tokenizer.bos_token_id))
            outputs = model.generate(**gen_kwargs)
        outputs = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in outputs.sequences]
        output = '\n\n'.join(outputs)

        if debug:
            print("prompt: ", prompt, flush=True)
            print("output: ", output, flush=True)

        def clean_response(response):
            meaningless_words = ['<pad>', '</s>', '<|endoftext|>', '”\n']
            for word in meaningless_words:
                response = response.replace(word, "")
            response = response.strip("\n")
            return response
        output = clean_response(output)
        if prompt_type_choice in [0, '0', 'plain']:
            return output
        else:
            # find first instance of prereponse
            # prompt sometimes has odd characters, that mutate length,
            # so can't go by length alone
            if pre_response:
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

    # get defaults
    model_lower = base_model.lower()
    placeholder_instruction, placeholder_input, \
    prompt_type, temperature, top_p, top_k, num_beams, \
    max_length, repetition_penalty, num_return_sequences, \
    do_sample, \
    examples = \
        get_generate_params(model_lower,
                            prompt_type, temperature, top_p, top_k, num_beams,
                            max_length, repetition_penalty, num_return_sequences,
                            do_sample,
                            )

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2, label="Instruction", placeholder=placeholder_instruction,
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder=placeholder_input),
            gr.components.Dropdown(prompt_types_strings, value=prompt_type, step=1, label="Prompt Type"),
            gr.components.Slider(minimum=0, maximum=3, value=temperature, label="Temperature"),
            gr.components.Slider(minimum=0, maximum=1, value=top_p, label="Top p"),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=top_k, label="Top k"
            ),
            gr.components.Slider(minimum=1, maximum=8, step=1, value=num_beams, label="Beams",
                                 info="Uses more GPU memory/compute"),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
            gr.components.Slider(minimum=0.01, maximum=3.0, value=repetition_penalty, label="Repetition Penalty"),
            gr.components.Slider(minimum=1, maximum=10, step=1, value=num_return_sequences, label="Num. Returns"),
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
        server_name="0.0.0.0",
        examples=examples,
    ).launch(share=share, show_error=True)


def get_generate_params(model_lower,
                        prompt_type, temperature, top_p, top_k, num_beams,
                        max_length, repetition_penalty, num_return_sequences,
                        do_sample):
    use_defaults = False
    use_default_examples = False
    examples = None

    if 't5-' in model_lower or 't5' == model_lower or 'flan-' in model_lower:
        placeholder_instruction = "The square root of x is the cube root of y. What is y to the power of 2, if x = 4?"
        placeholder_input = ""
        use_defaults = True
        use_default_examples = True
    elif 'bart-large-cnn-samsum' in model_lower:
        placeholder_instruction = """Jeff: Can I train a ? Transformers model on Amazon SageMaker? 
Philipp: Sure you can use the new Hugging Face Deep Learning Container. 
Jeff: ok.
Jeff: and how can I get started? 
Jeff: where can I find documentation? 
Philipp: ok, ok you can find everything here. https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face"""
        placeholder_input = ""
        use_defaults = True
        use_default_examples = False
        examples = [[placeholder_instruction, "", 'plain', 1.0, 1.0, 50, 1, 128, 1.0, 1, False]]
    else:
        placeholder_instruction = "Who is smarter, Einstein or Newton?"
        placeholder_input = ""
        prompt_type = prompt_type or 'instruct'
        temperature = 0.1 if temperature is None else temperature
        top_p = 0.75 if top_p is None else top_p
        top_k = 40 if top_k is None else top_k
        num_beams = num_beams or 4
        max_length = max_length or 128
        repetition_penalty = repetition_penalty or 1.0
        num_return_sequences = min(num_beams, num_return_sequences or 1)
        do_sample = False if do_sample is None else do_sample
    if use_defaults:
        prompt_type = prompt_type or 'plain'
        temperature = 1.0 if temperature is None else temperature
        top_p = 1.0 if top_p is None else top_p
        top_k = 50 if top_k is None else top_k
        num_beams = num_beams or 1
        max_length = max_length or 128
        repetition_penalty = repetition_penalty or 1.0
        num_return_sequences = min(num_beams, num_return_sequences or 1)
        do_sample = False if do_sample is None else do_sample

    if use_default_examples:
        examples = [
            ["Translate english to french", "Good morning", 'simple_instruct', 1.0, 1.0, 50, 1, 128, 1.0, 1, False],
            ['Translate to German:  My name is Arthur', '', 'plain', 1.0, 1.0, 50, 1, 128, 1.0, 1, False],
            ["Please answer to the following question. Who is going to be the next Ballon d'or?", '', 'plain', 1.0, 1.0, 50, 1, 128, 1.0, 1, False],
            ['Q: Can Geoffrey Hinton have a conversation with George Washington? Give the rationale before answering.', '', 'plain', 1.0, 1.0, 50, 1, 128, 1.0, 1, False],
            ['Please answer the following question. What is the boiling point of Nitrogen?', '', 'plain', 1.0, 1.0, 50, 1, 128, 1.0, 1, False],
            ['Answer the following yes/no question. Can you write a whole Haiku in a single tweet?', '', 'plain', 1.0, 1.0, 50, 1, 128, 1.0, 1, False],
            ['Answer the following yes/no question by reasoning step-by-step. Can you write a whole Haiku in a single tweet?', '', 'plain', 1.0, 1.0, 50, 1, 128, 1.0, 1, False],
            ["Q: ( False or not False or False ) is? A: Let's think step by step", '', 'plain', 1.0, 1.0, 50, 1, 128, 1.0, 1, False],
            ["Premise: At my age you will probably have learnt one lesson. Hypothesis:  It's not certain how many lessons you'll learn by your thirties. Does the premise entail the hypothesis?", '', 'plain', 1.0, 1.0, 50, 1, 128, 1.0, 1, False],
            ['The square root of x is the cube root of y. What is y to the power of 2, if x = 4?', '', 'plain', 1.0, 1.0, 50, 1, 128, 1.0, 1, False],
            ['Answer the following question by reasoning step by step.  The cafeteria had 23 apples. If they used 20 for lunch, and bought 6 more, how many apple do they have?', '', 'plain', 1.0, 1.0, 50, 1, 128, 1.0, 1, False],
        ]

    return placeholder_instruction, placeholder_input, \
           prompt_type, temperature, top_p, top_k, num_beams, \
           max_length, repetition_penalty, num_return_sequences, \
           do_sample, \
           examples


def test_test_prompt(prompt_type='instruct', data_point=0):
    example_data_point = example_data_points[data_point]
    example_data_point.pop('output', None)
    return generate_prompt(example_data_point, prompt_type)


if __name__ == "__main__":
    print("""
    WORLD_SIZE=4 CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 --master_port=1234 generate.py --base_model='EleutherAI/gpt-j-6B' --lora_weights=lora-alpaca_6B
    python generate.py --base_model='EleutherAI/gpt-j-6B' --lora_weights='lora-alpaca_6B'
    python generate.py --base_model='EleutherAI/gpt-neox-20b' --lora_weights='lora-alpaca_20B'
    
    # generate without lora weights, no prompt
    python generate.py --base_model='EleutherAI/gpt-neox-20b' --prompt_type='plain'
    python generate.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --prompt_type='dai_faq'

    python generate.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --prompt_type='dai_faq' --lora_weights='lora_20B_daifaq'
    # OpenChatKit settings:
    python generate.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --prompt_type='human_bot --debug=True --num_beams=1 --temperature=0.6 --top_k=40 --top_p=1.0

    python generate.py --base_model='distilgpt2' --prompt_type='plain' --debug=True --num_beams=1 --temperature=0.6 --top_k=40 --top_p=1.0 --share=False
    python generate.py --base_model='t5-large' --prompt_type='simple_instruct'
    python generate.py --base_model='philschmid/bart-large-cnn-samsum'

    """, flush=True)
    fire.Fire(main)
