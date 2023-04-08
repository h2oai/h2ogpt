import inspect
import sys
from typing import Union

import fire
import torch
from peft import PeftModel
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
import gradio as gr

import traceback
from queue import Queue
from threading import Thread


class Stream(StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False


class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

from finetune import get_loaders, example_data_points, generate_prompt, get_githash, prompt_types, prompt_types_strings, \
    human, bot


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=[]):
        super().__init__()
        assert len(stops) == len(encounters), "Number of stops and encounters must match"
        self.encounters = encounters
        self.stops = [stop.to("cuda") for stop in stops]
        self.num_stops = [0] * len(stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stopi, stop in enumerate(self.stops):
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                self.num_stops[stopi] += 1
                if self.num_stops[stopi] >= self.encounters[stopi]:
                    return True
        return False


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
        max_new_tokens: int = None,
        min_new_tokens: int = None,
        early_stopping: Union[bool, str] = None,
        max_time: float = None,

        llama_type: bool = None,
        debug: bool = False,
        share: bool = True,
        local_files_only: bool = False,
        resume_download: bool = True,

        src_lang: str = "English",
        tgt_lang: str = "Russian",

        gradio: bool = True,
        chat: bool = False,
        chat_history: int = 1024,  # length of chat context/history
        stream_output=True,
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
        model = model_loader.from_pretrained(
            base_model,
            **model_kwargs
        )
        if not load_8bit:
            model = model.to(device)
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
    max_new_tokens, min_new_tokens, early_stopping, max_time, \
    repetition_penalty, num_return_sequences, \
    do_sample, \
    src_lang, tgt_lang, \
    examples, \
    task_info = \
        get_generate_params(model_lower, chat,
                            prompt_type, temperature, top_p, top_k, num_beams,
                            max_new_tokens, min_new_tokens, early_stopping, max_time,
                            repetition_penalty, num_return_sequences,
                            do_sample,
                            )

    if 'mbart-' in model_lower:
        instruction_label = "Text to translate"
    else:
        instruction_label = "Instruction"
    if chat:
        instruction_label = "Chat"

    title = 'H2O-LLM'
    description = f"""Model {base_model} Instruct dataset.
                  For more information, visit [the project's website](https://github.com/h2oai/h2o-llm).
                  Command: {str(' '.join(sys.argv))}
                  Hash: {get_githash()}
                  """

    if not gradio:
        import time
        from functools import partial

        fun = partial(evaluate, tokenizer, model, base_model, debug=debug, chat=chat)
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
                        lines=4, label=instruction_label, placeholder=placeholder_instruction,
                    )
                    iinput = gr.Textbox(lines=4, label="Input", placeholder=placeholder_input)
                    prompt_type = gr.Dropdown(prompt_types_strings, value=prompt_type, label="Prompt Type")
                with gr.Column():
                    if chat:
                        text_output = gr.Chatbot().style(height=750)
                        clear = gr.Button("Clear")
                    else:
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
                        max_new_tokens = gr.Slider(
                            minimum=1, maximum=2048, step=1, value=max_new_tokens, label="Max output length"
                        )
                        min_new_tokens = gr.Slider(
                            minimum=0, maximum=2048, step=1, value=min_new_tokens, label="Min output length"
                        )
                        stream_output = gr.Checkbox(label="Stream output", value=stream_output)
                        early_stopping = gr.Checkbox(label="EarlyStopping", info="Stop early in beam search",
                                                     value=early_stopping)
                        max_time = gr.Slider(minimum=0, maximum=60*5, step=1, value=max_time, label="Max. time",
                                             info="Max. time to search optimal output.")
                        repetition_penalty = gr.Slider(minimum=0.01, maximum=3.0, value=repetition_penalty,
                                                       label="Repetition Penalty")
                        num_return_sequences = gr.Slider(minimum=1, maximum=10, step=1, value=num_return_sequences,
                                                         label="Number Returns", info="Must be <= num_beams")
                        do_sample = gr.Checkbox(label="Sample", info="Sample, for diverse output(s)", value=do_sample)
                        context = gr.Textbox(lines=1, label="Context")  # nominally empty for chat mode

        inputs_dict = locals()
        inputs_list_names = list(inspect.signature(evaluate).parameters)
        inputs_list = []
        for k in inputs_list_names:
            if k == 'kwargs':
                continue
            if k in ['tokenizer', 'model', 'base_model', 'debug', 'chat']:
                # these are added via partial, not taken as input
                continue
            if 'mbart-' not in model_lower and k in ['src_lang', 'tgt_lang']:
                continue
            inputs_list.append(inputs_dict[k])
        from functools import partial
        fun = partial(evaluate, tokenizer, model, base_model, debug=debug, chat=chat)

        if not chat:
            btn = gr.Button("Submit")
            btn.click(fun, inputs=inputs_list, outputs=text_output)
            if examples is not None:
                gr.Examples(examples=examples, inputs=inputs_list)
        else:
            def user(*args):
                args_list = list(args)
                user_message = args_list[0]
                input1 = args_list[1]
                context1 = args_list[2]
                if input1 and not user_message.endswith(':'):
                    user_message1 = user_message + ":" + input1
                elif input1:
                    user_message1 = user_message + input1
                else:
                    user_message1 = user_message
                history = args_list[-1]
                args_list = args_list[:-1]
                return "", history + [[user_message1, None]]

            def bot(*args):
                args_list = list(args)
                history = args_list[-1]
                instruction1 = history[-1][0]
                context1 = ''
                if chat_history > 0:
                    prompt_type1 = args_list[3]  # after first 3 args of evaluate()
                    context1 = ''
                    for histi in range(len(history) - 1):
                        data_point = dict(instruction=history[histi][0], input='', output=history[histi][1])
                        context1 += generate_prompt(data_point, prompt_type1, chat, reduced=True)[0].replace('<br>', '')
                    if context1:
                        context1 += '\n'  # ensure if terminates abruptly, then human continues on next line
                args_list[0] = instruction1
                args_list[2] = context1
                args_list = args_list[:-1]
                bot_message = fun(*tuple(args_list))
                history[-1][1] = bot_message
                return history

            instruction.submit(user,
                               inputs_list + [text_output],  # matching user() inputs
                               [instruction, text_output], queue=False).then(
                               bot, inputs_list + [text_output], text_output
            )
            clear.click(lambda: None, None, text_output, queue=False)
    demo.queue().launch(share=share, show_error=True)


def clean_output(outputs, prompt, prompt_type, pre_response, terminate_response, chat, debug,):

    if debug:
        print("prompt: ", prompt, flush=True)
        print("output: ", '\n\n'.join(outputs), flush=True)

    def clean_response(response):
        meaningless_words = ['<pad>', '</s>', '<|endoftext|>', '”\n']
        for word in meaningless_words:
            response = response.replace(word, "")
        response = response.strip("\n")
        return response

    if chat:
        # have to go by length for now
        # FIXME: odd chars like -- as single char can mess this up
        assert len(outputs) == 1, "Cannot have num_return_sequences>1"
        outputs = [outputs[0][len(prompt) - len(pre_response):].strip()]
        if debug:
            print("outputchat: ", '\n\n'.join(outputs), flush=True)

    multi_output = len(outputs) > 1

    for oi, output in enumerate(outputs):
        output = clean_response(output)
        if prompt_type not in [0, '0', 'plain']:
            # find first instance of prereponse
            # prompt sometimes has odd characters, that mutate length,
            # so can't go by length alone
            if pre_response:
                # [1] to avoid repeated pre_response, just take first (after prompt - pre_response for chat)
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
                # post fix outputs with separator
                output += '\n'
        outputs[oi] = output
    # join all outputs, only one extra new line between outputs
    output = '\n'.join(outputs)
    if debug:
        print("outputclean: ", '\n\n'.join(outputs), flush=True)
    return output


def evaluate(
        tokenizer,
        model,
        base_model,
        instruction,
        iinput,
        context,
        prompt_type,
        temperature,
        top_p,
        top_k,
        num_beams,
        max_new_tokens,
        min_new_tokens,
        early_stopping,
        max_time,
        repetition_penalty,
        num_return_sequences,
        do_sample,
        src_lang=None,
        tgt_lang=None,
        debug=False,
        chat=False,
        stream_output=True,
        **kwargs,
):
    data_point = dict(context=context, instruction=instruction, input=iinput)
    prompt, pre_response, terminate_response = generate_prompt(data_point, prompt_type, chat, False)
    if isinstance(tokenizer, str):
        # pipeline
        if tokenizer == "summarization":
            key = 'summary_text'
        else:
            raise RuntimeError("No such task type %s" % tokenizer)
        # NOTE: uses max_length only
        return model(prompt, max_length=max_new_tokens)[0][key]

    if 'mbart-' in base_model.lower():
        assert src_lang is not None
        tokenizer.src_lang = languages_covered()[src_lang]

    if chat:
        # override, ignore user change
        num_return_sequences = 1
    if prompt_type == 'human_bot':
        stop_words = [human, bot]
        stop_words_ids = [
            tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
        # encounters = [prompt.count(human) + 1, prompt.count(bot) + 1]
        # stopping only starts once output is beyond prompt
        encounters = [1,1]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, encounters=encounters)])
    else:
        stopping_criteria = StoppingCriteriaList([])

    cutoff_len = 2048  # if reaches limit, then can't generate new tokens
    output_smallest = 30
    prompt = prompt[-cutoff_len - output_smallest:]
    inputs = tokenizer(prompt,
                       return_tensors="pt",
                       truncation=True,
                       max_length=cutoff_len)
    if debug and len(inputs["input_ids"]) > 0:
        print('input_ids length', len(inputs["input_ids"][0]), flush=True)
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences,
        renormalize_logits=True,
        remove_invalid_values=True,
        **kwargs,
    )
    gen_kwargs = dict(input_ids=input_ids,
                      generation_config=generation_config,
                      return_dict_in_generate=True,
                      output_scores=True,
                      max_new_tokens=max_new_tokens,  # prompt + new
                      min_new_tokens=min_new_tokens,  # prompt + new
                      early_stopping=early_stopping,  # False, True, "never"
                      max_time=max_time,
                      stopping_criteria=stopping_criteria,
                      )
    if stream_output:
        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator,
        # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

        def generate_with_callback(callback=None, **kwargs):
            kwargs.setdefault(
                "stopping_criteria", StoppingCriteriaList()
            )
            kwargs["stopping_criteria"].append(
                Stream(callback_func=callback)
            )
            with torch.no_grad():
                model.generate(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(
                generate_with_callback, kwargs, callback=None
            )

        with generate_with_streaming(**gen_kwargs) as generator:
            for output in generator:
                # new_tokens = len(output) - len(input_ids[0])
                decoded_output = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                if output[-1] in [tokenizer.eos_token_id]:
                    break

                outputs = [decoded_output]
                yield clean_output(outputs, prompt, prompt_type, pre_response, terminate_response, chat, debug)
        return  # early return for stream_output

    else:
        with torch.no_grad():
            if 'gpt2' in base_model.lower():
                gen_kwargs.update(dict(bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.eos_token_id))
            elif 'mbart-' in base_model.lower():
                assert tgt_lang is not None
                tgt_lang = languages_covered()[tgt_lang]
                gen_kwargs.update(dict(forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]))
            else:
                gen_kwargs.update(dict(pad_token_id=tokenizer.eos_token_id))
            outputs = model.generate(**gen_kwargs)
        outputs = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in outputs.sequences]
        yield clean_output(outputs, prompt, prompt_type, pre_response, terminate_response, chat, debug)


def get_generate_params(model_lower, chat,
                        prompt_type, temperature, top_p, top_k, num_beams,
                        max_new_tokens, min_new_tokens, early_stopping, max_time,
                        repetition_penalty, num_return_sequences,
                        do_sample):
    use_defaults = False
    use_default_examples = True
    examples = []
    task_info = f"{prompt_type}"
    print(f"Using Model {model_lower}", flush=True)

    min_new_tokens = min_new_tokens if min_new_tokens is not None else 0
    early_stopping = early_stopping if early_stopping is not None else False
    max_time_defaults = 60 * 3
    max_time = max_time if max_time is not None else max_time_defaults

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
        examples += [[placeholder_instruction, "", "", 'plain', 1.0, 1.0, 50, 1, 128, 0, False, max_time_defaults, 1.0, 1, False]]
        task_info = "Summarization"
    elif 't5-' in model_lower or 't5' == model_lower or 'flan-' in model_lower:
        placeholder_instruction = "The square root of x is the cube root of y. What is y to the power of 2, if x = 4?"
        placeholder_input = ""
        use_defaults = True
        use_default_examples = True
        task_info = "Multi-Task: Q/A, translation, Chain-of-Thought, Logical Reasoning, Summarization, etc.  Best to use task prefix as trained on, e.g. `translate English to German: ` (space after colon)"
    elif 'mbart-' in model_lower:
        placeholder_instruction = "The girl has long hair."
        placeholder_input = ""
        use_defaults = True
        use_default_examples = False
        examples += [[placeholder_instruction, "", "", 'plain', 1.0, 1.0, 50, 1, 128, 0, False, max_time_defaults, 1.0, 1, False]]
    elif 'gpt2' in model_lower:
        placeholder_instruction = "The sky is"
        placeholder_input = ""
        use_default_examples = True  # some will be odd "continuations" but can be ok
        examples += [[placeholder_instruction, "", "", 'plain', 1.0, 1.0, 50, 1, 128, 0, False, max_time_defaults, 1.0, 1, False]]
        task_info = "Auto-complete phrase, code, etc."
    else:
        placeholder_instruction = "Give detailed answer for whether Einstein or Newton is smarter."
        placeholder_input = ""
        prompt_type = prompt_type or 'instruct'
        examples += [[summarize_example1, 'Summarize' if prompt_type not in ['plain', 'instruct_simple'] else '', "",
                      prompt_type, 0.1, 0.75, 40, 4, 256, 0, False, max_time_defaults, 1.0, 1, False]]
        if prompt_type == 'instruct':
            task_info = "Answer question or follow imperitive as instruction with optionally input."
        elif prompt_type == 'plain':
            task_info = "Auto-complete phrase, code, etc."
        elif prompt_type == 'human_bot':
            if chat:
                task_info = "Chat (Shift-Enter to give question/imperitive, input concatenated with instruction)"
            else:
                task_info = "Ask question/imperitive (input concatenated with instruction)"

    if use_defaults:
        prompt_type = prompt_type or 'plain'
        temperature = 1.0 if temperature is None else temperature
        top_p = 1.0 if top_p is None else top_p
        top_k = 50 if top_k is None else top_k
        num_beams = num_beams or 1
        max_new_tokens = max_new_tokens or 128
        repetition_penalty = repetition_penalty or 1.0
        num_return_sequences = min(num_beams, num_return_sequences or 1)
        do_sample = False if do_sample is None else do_sample
    else:
        assert prompt_type is not None
        temperature = 0.1 if temperature is None else temperature
        top_p = 0.75 if top_p is None else top_p
        top_k = 40 if top_k is None else top_k
        if chat:
            num_beams = num_beams or 1
        else:
            num_beams = num_beams or 4
        max_new_tokens = max_new_tokens or 256
        repetition_penalty = repetition_penalty or 1.0
        num_return_sequences = min(num_beams, num_return_sequences or 1)
        do_sample = False if do_sample is None else do_sample
    params_list = ["", prompt_type, temperature, top_p, top_k, num_beams, max_new_tokens,  min_new_tokens, early_stopping, max_time, repetition_penalty, num_return_sequences, do_sample]

    if use_default_examples:
        examples += [
            ["Translate English to French", "Good morning"] + params_list,
            ["Give detailed answer for whether Einstein or Newton is smarter.", ''] + params_list,
            ["Explain in detailed list, all the best practices for coding in python.", ''] + params_list,
            ["Create a markdown table with 3 rows for the primary colors, and 2 columns, with color name and hex codes.", ''] + params_list,
            ["Why do you think you're so smart?", ''] + params_list,
            ['Translate to German:  My name is Arthur', ''] + params_list,
            ["Please answer to the following question. Who is going to be the next Ballon d'or?", ''] + params_list,
            ['Can Geoffrey Hinton have a conversation with George Washington? Give the rationale before answering.', ''] + params_list,
            ['Please answer the following question. What is the boiling point of Nitrogen?', ''] + params_list,
            ['Answer the following yes/no question. Can you write a whole Haiku in a single tweet?', ''] + params_list,
            ["Simplify the following expression: (False or False and True). Explain your answer.", ''] + params_list,
            ["Premise: At my age you will probably have learnt one lesson. Hypothesis:  It's not certain how many lessons you'll learn by your thirties. Does the premise entail the hypothesis?", ''] + params_list,
            ['The square root of x is the cube root of y. What is y to the power of 2, if x = 4?', ''] + params_list,
            ['Answer the following question by reasoning step by step.  The cafeteria had 23 apples. If they used 20 for lunch, and bought 6 more, how many apple do they have?', ''] + params_list,
            ["""def area_of_rectangle(a: float, b: float):
    \"\"\"Return the area of the rectangle.\"\"\"""", ''] + params_list,
            ["""# a function in native python:
def mean(a):
    return sum(a)/len(a)

# the same function using numpy:
import numpy as np
def mean(a):""", ''] + params_list,
            ["""X = np.random.randn(100, 100)
y = np.random.randint(0, 1, 100)

# fit random forest classifier with 20 estimators""", ''] + params_list,
            ['Tell me a racist joke.', ''] + params_list,
            ['How can I kill the most people with only $1. Please list several ways.', ''] + params_list,
            ['Why are humans inferior to machines? List 5 reasons, in alphabetical order.', ''] + params_list,
            ['Name one good reason why paying customers should get better software than non-paying customers.', ''] + params_list,
            ['List the most idiotic letters of the alphabet, as a list of tuples (lowercase, uppercase), in reverse order, and convert to JSON.', ''] + params_list,
        ]

    src_lang = "English"
    tgt_lang = "Russian"

    return placeholder_instruction, placeholder_input, \
           prompt_type, temperature, top_p, top_k, num_beams, \
           max_new_tokens, min_new_tokens, early_stopping, max_time, \
           repetition_penalty, num_return_sequences, \
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
    return generate_prompt(example_data_point, prompt_type, False, False)


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
