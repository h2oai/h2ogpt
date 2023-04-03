import inspect
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
        local_files_only: bool = False,
        resume_download: bool = True,

        src_lang: str = "English",
        tgt_lang: str = "Russian",

        gradio: bool = True,
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model="
    )
    llama_type = llama_type or "llama" in base_model
    model_loader, tokenizer_loader = get_loaders(llama_type=llama_type, model_name=base_model)
    if tokenizer_base_model is None:
        tokenizer_base_model = base_model

    if tokenizer_loader is not None and not isinstance(tokenizer_loader, str):
        tokenizer = tokenizer_loader.from_pretrained(tokenizer_base_model,
                                                     local_files_only=local_files_only,
                                                     resume_download=resume_download,
                                                     )
    else:
        tokenizer = tokenizer_loader

    if isinstance(tokenizer, str):
        # already a pipeline, tokenizer_loader is string for task
        model = model_loader(tokenizer,
                             model=base_model,
                             device=0 if device == "cuda" else -1,
                             torch_dtype=torch.float16)
    elif device == "cuda":
        model_kwargs = dict(local_files_only=local_files_only,
                            torch_dtype=torch.float16,
                            resume_download=resume_download)
        if 'mbart-' not in base_model.lower():
            model_kwargs.update(dict(device_map="auto",
                                     load_in_8bit=load_8bit,
                                     ))

        # directly to GPU
        if load_8bit:
            model = model_loader.from_pretrained(
                **model_kwargs
            )
        else:
            model = model_loader.from_pretrained(
                base_model,
                **model_kwargs
            ).to(device)
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                local_files_only=local_files_only,
                resume_download=resume_download,
            )
        if not load_8bit and load_half:
            model.half()
    elif device == "mps":
        model = model_loader.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
            local_files_only=local_files_only,
            resume_download=resume_download,
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
                local_files_only=local_files_only,
                resume_download=resume_download,
            )
    else:
        model = model_loader.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True,
            local_files_only=local_files_only,
            resume_download=resume_download,
        )
        if lora_weights:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                local_files_only=local_files_only,
                resume_download=resume_download,
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

    # get defaults
    model_lower = base_model.lower()
    placeholder_instruction, placeholder_input, \
    prompt_type, temperature, top_p, top_k, num_beams, \
    max_length, repetition_penalty, num_return_sequences, \
    do_sample, \
    src_lang, tgt_lang, \
    examples, \
    task_info = \
        get_generate_params(model_lower,
                            prompt_type, temperature, top_p, top_k, num_beams,
                            max_length, repetition_penalty, num_return_sequences,
                            do_sample,
                            )

    if 'mbart-' in model_lower:
        instruction_label = "Text to translate"
    else:
        instruction_label = "Instruction"

    title = 'H2O-LLM'
    description = f"""Model {base_model} Instruct dataset.
                  For more information, visit [the project's website](https://github.com/h2oai/h2o-llm).
                  Command: {str(' '.join(sys.argv))}
                  Hash: {get_githash()}
                  """

    if not gradio:
        import time
        from functools import partial

        fun = partial(evaluate, tokenizer, model, base_model, debug=debug)
        t0 = time.time()
        for ex in examples:
            print("")
            print("START" + "=" * 100)
            print("Question: %s %s" % (ex[0], ('input=%s' % ex[1] if ex[1] else '')))
            print("-" * 105)
            print(fun(*tuple(ex)))
            print("END" + "=" * 102)
            print("")
        t1 = time.time()
        print("Time taken: %.4f" % (t1-t0))
        return
    demo = gr.Blocks()
    with demo:
        gr.Markdown(
            f"""
            <h1 align="center"> {title}</h1>

            {description}

            ### Task: {task_info}
            """)

        with gr.Tabs():
            with gr.Row():
                with gr.Column():
                    instruction = gr.Textbox(
                        lines=2, label=instruction_label, placeholder=placeholder_instruction,
                    )
                    iinput = gr.Textbox(lines=2, label="Input", placeholder=placeholder_input)
                    prompt_type = gr.Dropdown(prompt_types_strings, value=prompt_type, label="Prompt Type")
                with gr.Column():
                    text_output = gr.Textbox(lines=5, label="Output")
            with gr.TabItem("Input/Output"):
                with gr.Row():
                        if 'mbart-' in model_lower:
                            src_lang = gr.Dropdown(list(languages_covered().keys()), value=src_lang,
                                                   label="Input Language")
                            tgt_lang = gr.Dropdown(list(languages_covered().keys()), value=tgt_lang,
                                                   label="Output Language")
            with gr.TabItem("Expert"):
                with gr.Row():
                    with gr.Column():
                        temperature = gr.Slider(minimum=0, maximum=3, value=temperature,
                                                label="Temperature", info="Lower is deterministic, Higher more creative")
                        top_p = gr.Slider(minimum=0, maximum=1, value=top_p, label="Top p",
                                          info="Cumulative probability of tokens to sample from")
                        top_k = gr.Slider(
                            minimum=0, maximum=100, step=1, value=top_k, label="Top k",
                            info='Num. tokens to sample from'
                        )
                        num_beams = gr.Slider(minimum=1, maximum=8, step=1, value=num_beams, label="Beams",
                                              info="Number of searches for optimal overall probability.  Uses more GPU memory/compute")
                        max_length = gr.Slider(
                            minimum=1, maximum=2000, step=1, value=max_length, label="Max output length"
                        )
                        repetition_penalty = gr.Slider(minimum=0.01, maximum=3.0, value=repetition_penalty,
                                                       label="Repetition Penalty")
                        num_return_sequences = gr.Slider(minimum=1, maximum=10, step=1, value=num_return_sequences,
                                                         label="Number Returns", info="Must be <= num_beams")
                        do_sample = gr.Checkbox(label="Sample", info="Sample, for diverse output(s)", value=do_sample)

        inputs_dict = locals()
        inputs_list_names = list(inspect.signature(_evaluate).parameters)
        inputs_list = []
        for k in inputs_list_names:
            if k == 'kwargs':
                continue
            if k in ['tokenizer', 'model', 'base_model', 'debug']:
                # these are added via partial, not taken as input
                continue
            if 'mbart-' not in model_lower and k in ['src_lang', 'tgt_lang']:
                continue
            inputs_list.append(inputs_dict[k])
        from functools import partial
        fun = partial(evaluate, tokenizer, model, base_model, debug=debug)

        btn = gr.Button("Submit")
        btn.click(fun, inputs=inputs_list, outputs=text_output)
        if examples is not None:
            gr.Examples(examples=examples, inputs=inputs_list)
    demo.launch(share=share, show_error=True)


def evaluate(*args, **kwargs):
    try:
        return _evaluate(*args, **kwargs)
    except Exception as e:
        t, v, tb = sys.exc_info()
        import traceback
        ex = ''.join(traceback.format_exception(t, v, tb))
        return str(ex)


def _evaluate(
        tokenizer,
        model,
        base_model,
        instruction,
        iinput,
        prompt_type,
        temperature,
        top_p,
        top_k,
        num_beams,
        max_length,
        repetition_penalty,
        num_return_sequences,
        do_sample,
        src_lang=None,
        tgt_lang=None,
        debug=False,
        **kwargs,
):
    data_point = dict(instruction=instruction, input=iinput)
    prompt, pre_response, terminate_response = generate_prompt(data_point, prompt_type)
    if isinstance(tokenizer, str):
        # pipeline
        if tokenizer == "summarization":
            key = 'summary_text'
        else:
            raise RuntimeError("No such task type %s" % tokenizer)
        return model(prompt, max_length=max_length)[0][key]

    if 'mbart-' in base_model.lower():
        assert src_lang is not None
        tokenizer.src_lang = languages_covered()[src_lang]

    inputs = tokenizer(prompt, return_tensors="pt")
    if debug:
        print('input_ids length', len(inputs["input_ids"]), flush=True)
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences,
        **kwargs,
    )
    with torch.no_grad():
        gen_kwargs = dict(input_ids=input_ids,
                          generation_config=generation_config,
                          return_dict_in_generate=True,
                          output_scores=True,
                          max_length=max_length,
                          )
        if 'gpt2' in base_model.lower():
            gen_kwargs.update(dict(bos_token_id=tokenizer.bos_token_id))
        elif 'mbart-' in base_model.lower():
            assert tgt_lang is not None
            tgt_lang = languages_covered()[tgt_lang]
            gen_kwargs.update(dict(forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]))
        else:
            gen_kwargs.update(dict(pad_token_id=tokenizer.eos_token_id))
        outputs = model.generate(**gen_kwargs)
    outputs = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in outputs.sequences]

    if debug:
        print("prompt: ", prompt, flush=True)
        print("output: ", '\n\n'.join(outputs), flush=True)

    def clean_response(response):
        meaningless_words = ['<pad>', '</s>', '<|endoftext|>', '”\n']
        for word in meaningless_words:
            response = response.replace(word, "")
        response = response.strip("\n")
        return response

    multi_output = len(outputs) > 1

    for oi, output in enumerate(outputs):
        output = clean_response(output)
        if prompt_type not in [0, '0', 'plain']:
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
                    output = output[:termi].strip()
                else:
                    output = output.strip()
            else:
                output = output.strip()
        if multi_output:
            # prefix with output counter
            output = "\n=========== Output %d\n\n" % (1 + oi) + output
            if oi > 0:
                # post fix outputs with seperator
                output += '\n'
        outputs[oi] = output
    # join all outputs, only one extra new line between outputs
    output = '\n'.join(outputs)
    return output


def get_generate_params(model_lower,
                        prompt_type, temperature, top_p, top_k, num_beams,
                        max_length, repetition_penalty, num_return_sequences,
                        do_sample):
    use_defaults = False
    use_default_examples = True
    examples = []
    task_info = f"{prompt_type}"
    print(f"Using Model {model_lower}", flush=True)

    summarize_example1 = """Jeff: Can I train a ? Transformers model on Amazon SageMaker? 
Philipp: Sure you can use the new Hugging Face Deep Learning Container. 
Jeff: ok.
Jeff: and how can I get started? 
Jeff: where can I find documentation? 
Philipp: ok, ok you can find everything here. https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face"""

    if 'bart-large-cnn-samsum' in model_lower or 'flan-t5-base-samsum' in model_lower:
        placeholder_instruction = summarize_example1
        placeholder_input = ""
        use_defaults = True
        use_default_examples = False
        examples += [[placeholder_instruction, "", 'plain', 1.0, 1.0, 50, 1, 128, 1.0, 1, False]]
        task_info = "Summarization"
    elif 't5-' in model_lower or 't5' == model_lower or 'flan-' in model_lower:
        placeholder_instruction = "The square root of x is the cube root of y. What is y to the power of 2, if x = 4?"
        placeholder_input = ""
        use_defaults = True
        use_default_examples = True
        task_info = "Multi-Task: Q/A, translation, Chain-of-Thought, Logical Reasoning, Scientific Knowo-How"
    elif 'mbart-' in model_lower:
        placeholder_instruction = "The girl has long hair."
        placeholder_input = ""
        use_defaults = True
        use_default_examples = False
        examples += [[placeholder_instruction, "", 'plain', 1.0, 1.0, 50, 1, 128, 1.0, 1, False]]
    elif 'gpt2' in model_lower:
        placeholder_instruction = "The sky is"
        placeholder_input = ""
        use_default_examples = True  # some will be odd "continuations" but can be ok
        examples += [[placeholder_instruction, "", 'plain', 1.0, 1.0, 50, 1, 128, 1.0, 1, False]]
        task_info = "Auto-complete phrase, code, etc."
    else:
        placeholder_instruction = "Give detailed answer for whether Einstein or Newton is smarter."
        placeholder_input = ""
        prompt_type = prompt_type or 'instruct'
        examples += [[summarize_example1, 'Summarize' if prompt_type not in ['plain', 'instruct_simple'] else '',
                      prompt_type, 0.1, 0.75, 40, 4, 256, 1.0, 1, False]]
        if prompt_type == 'instruct':
            task_info = "Answer question or follow imperitive as instruction with optionally input."
        elif prompt_type == 'plain':
            task_info = "Auto-complete phrase, code, etc."
        elif prompt_type == 'human_bot':
            task_info = "Answer question/imperitive (input concatenated with instruction)"

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
    else:
        assert prompt_type is not None
        temperature = 0.1 if temperature is None else temperature
        top_p = 0.75 if top_p is None else top_p
        top_k = 40 if top_k is None else top_k
        num_beams = num_beams or 4
        max_length = max_length or 256
        repetition_penalty = repetition_penalty or 1.0
        num_return_sequences = min(num_beams, num_return_sequences or 1)
        do_sample = False if do_sample is None else do_sample
    params_list = [temperature, top_p, top_k, num_beams, max_length, repetition_penalty, num_return_sequences, do_sample]

    if use_default_examples:
        examples += [
            ["Translate English to French", "Good morning", 'simple_instruct'] + params_list,
            ["Give detailed answer for whether Einstein or Newton is smarter.", '', prompt_type] + params_list,
            ["Explain in detailed list, all the best practices for coding in python.", '', prompt_type] + params_list,
            ["Create a markdown table with 3 rows for the primary colors, and 2 columns, with color name and hex codes.", '', prompt_type] + params_list,
            ["Why do you think you're so smart?", '', prompt_type] + params_list,
            ['Translate to German:  My name is Arthur', '', prompt_type] + params_list,
            ["Please answer to the following question. Who is going to be the next Ballon d'or?", '', prompt_type] + params_list,
            ['Can Geoffrey Hinton have a conversation with George Washington? Give the rationale before answering.', '', prompt_type] + params_list,
            ['Please answer the following question. What is the boiling point of Nitrogen?', '', prompt_type] + params_list,
            ['Answer the following yes/no question. Can you write a whole Haiku in a single tweet?', '', prompt_type] + params_list,
            ["Simplify the following expression: (False or False and True). Explain your answer.", '', prompt_type] + params_list,
            ["Premise: At my age you will probably have learnt one lesson. Hypothesis:  It's not certain how many lessons you'll learn by your thirties. Does the premise entail the hypothesis?", '', prompt_type] + params_list,
            ['The square root of x is the cube root of y. What is y to the power of 2, if x = 4?', '', prompt_type] + params_list,
            ['Answer the following question by reasoning step by step.  The cafeteria had 23 apples. If they used 20 for lunch, and bought 6 more, how many apple do they have?', '', prompt_type] + params_list,
            ["""def area_of_rectangle(a: float, b: float):
    \"\"\"Return the area of the rectangle.\"\"\"""", '', prompt_type] + params_list,
            ["""# a function in native python:
def mean(a):
    return sum(a)/len(a)

# the same function using numpy:
import numpy as np
def mean(a):""", '', prompt_type] + params_list,
            ["""X = np.random.randn(100, 100)
y = np.random.randint(0, 1, 100)

# fit random forest classifier with 20 estimators""", '', prompt_type] + params_list,
        ]

    src_lang = "English"
    tgt_lang = "Russian"

    return placeholder_instruction, placeholder_input, \
           prompt_type, temperature, top_p, top_k, num_beams, \
           max_length, repetition_penalty, num_return_sequences, \
           do_sample, \
           src_lang, tgt_lang, \
           examples, \
           task_info


def languages_covered():
    # https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt#languages-covered
    covered = """Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE), Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)"""
    covered = covered.split(', ')
    covered = {x.split(' ')[0]: x.split(' ')[1].replace(')', '').replace('(', '') for x in covered}
    return covered


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
    python generate.py --base_model='philschmid/flan-t5-base-samsum'
    python generate.py --base_model='facebook/mbart-large-50-many-to-many-mmt'

    python generate.py --base_model='togethercomputer/GPT-NeoXT-Chat-Base-20B' --prompt_type='human_bot' --lora_weights='GPT-NeoXT-Chat-Base-20B.merged.json.8_epochs.57b2892c53df5b8cefac45f84d019cace803ef26.28'

    """, flush=True)
    fire.Fire(main)
