import functools
import gc
import inspect
import sys
import os
import typing

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
from typing import Union

import fire
import torch
from peft import PeftModel
from transformers import GenerationConfig, StoppingCriteriaList, AutoModel
from accelerate import init_empty_weights, infer_auto_device_map

from prompter import Prompter

from finetune import get_loaders, example_data_points, generate_prompt, get_githash, prompt_types_strings, \
    human, bot, prompt_type_to_model_name, inv_prompt_type_to_model_lower
from stopping import CallbackToGenerator, Stream, StoppingCriteriaSub


def main(
        load_8bit: bool = False,
        load_half: bool = True,
        base_model: str = '',
        tokenizer_base_model: str = '',
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
        chat: bool = True,
        chat_history: int = 4096,  # character length of chat context/history
        stream_output: bool = True,
        show_examples: bool = None,
        verbose: bool = False,
        h2ocolors: bool = True,
        height: int = 400,

        sanitize_user_prompt: bool = True,
        sanitize_bot_response: bool = True,

        extra_model_options: typing.List[str] = [],
        extra_lora_options: typing.List[str] = [],
):

    # get defaults
    model_lower = base_model.lower()
    if not gradio:
        # force, else not single response like want to look at
        stream_output = False
        # else prompt removal can mess up output
        chat = False

    placeholder_instruction, placeholder_input, \
    stream_output, show_examples, \
    prompt_type, temperature, top_p, top_k, num_beams, \
    max_new_tokens, min_new_tokens, early_stopping, max_time, \
    repetition_penalty, num_return_sequences, \
    do_sample, \
    src_lang, tgt_lang, \
    examples, \
    task_info = \
        get_generate_params(model_lower, chat,
                            stream_output, show_examples,
                            prompt_type, temperature, top_p, top_k, num_beams,
                            max_new_tokens, min_new_tokens, early_stopping, max_time,
                            repetition_penalty, num_return_sequences,
                            do_sample,
                            )

    if not gradio:
        # ensure was set right above before examples generated
        assert not stream_output, "stream_output=True does not make sense with example loop"
        import time
        from functools import partial

        model, tokenizer, device = get_model(**locals())
        model_state = [model, tokenizer, device, base_model]
        fun = partial(evaluate, model_state, debug=debug, chat=chat)
        t0 = time.time()
        for ex in examples:
            print("")
            print("START" + "=" * 100)
            print("Question: %s %s" % (ex[0], ('input=%s' % ex[1] if ex[1] else '')))
            print("-" * 105)
            # fun yields as generator, so have to iterate over it
            # Also means likely do NOT want --stream_output=True, else would show all generations
            for res in fun(*tuple(ex)):
                print(res)
            print("END" + "=" * 102)
            print("")
        t1 = time.time()
        print("Time taken: %.4f" % (t1 - t0))
        return
    if gradio:
        go_gradio(**locals())


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise RuntimeError("only cuda supported")

    return device


def get_non_lora_model(base_model, model_loader, load_half, model_kwargs):
    """
    Ensure model gets on correct device
    :param base_model:
    :param model_loader:
    :param load_half:
    :param model_kwargs:
    :return:
    """
    with init_empty_weights():
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModel.from_config(
            config,
        )

    # NOTE: Can specify max_memory={0: max_mem, 1: max_mem}, to shard model
    # NOTE: Some models require avoiding sharding some layers,
    # then would pass no_split_module_classes and give list of those layers.
    device_map = infer_auto_device_map(
        model,
        dtype=torch.float16 if load_half else torch.float32,
    )
    if hasattr(model, 'model'):
        device_map_model = infer_auto_device_map(
            model.model,
            dtype=torch.float16 if load_half else torch.float32,
        )
        device_map.update(device_map_model)
    print('device_map: %s' % device_map, flush=True)

    load_in_8bit = model_kwargs.get('load_in_8bit', False)
    model_kwargs['device_map'] = device_map

    if load_in_8bit or not load_half:
        model = model_loader.from_pretrained(
            base_model,
            **model_kwargs,
        )
    else:
        model = model_loader.from_pretrained(
            base_model,
            **model_kwargs,
        ).half()
    return model


def get_model(
        load_8bit: bool = False,
        load_half: bool = True,
        base_model: str = '',
        tokenizer_base_model: str = '',
        lora_weights: str = "",

        llama_type: bool = None,
        local_files_only: bool = False,
        resume_download: bool = True,
        **kwargs,
):
    """

    :param load_8bit: load model in 8-bit, not supported by all mdoels
    :param load_half: load model in 16-bit
    :param base_model: name/path of base model
    :param tokenizer_base_model: name/path of tokenizer
    :param lora_weights: name/path
    :param llama_type: whether LLaMa type model
    :param local_files_only: use local files instead of from HF
    :param resume_download: resume downloads from HF
    :param kwargs:
    :return:
    """
    device = get_device()

    assert base_model.strip(), (
        "Please choose a base model with --base_model (CLI) or in Models Tab (gradio)"
    )
    llama_type = llama_type or "llama" in base_model
    model_loader, tokenizer_loader = get_loaders(llama_type=llama_type, model_name=base_model)
    if not tokenizer_base_model:
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
    else:
        assert device == "cuda", "Unsupported device %s" % device
        model_kwargs = dict(local_files_only=local_files_only,
                            torch_dtype=torch.float16,
                            resume_download=resume_download)
        if 'mbart-' not in base_model.lower():
            model_kwargs.update(dict(load_in_8bit=load_8bit,
                                     device_map={"": 0} if load_8bit else "auto",
                                     ))

        if not lora_weights:
            with torch.device("cuda"):
                #model = get_non_lora_model(base_model, model_loader, load_half, model_kwargs)
                model = model_loader.from_pretrained(
                    base_model,
                    **model_kwargs).half()
        elif load_8bit:
            model = model_loader.from_pretrained(
                base_model,
                **model_kwargs
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                local_files_only=local_files_only,
                resume_download=resume_download,
                device_map={"": 0},
            )
        else:
            with torch.device("cuda"):
                model = model_loader.from_pretrained(
                    base_model,
                    **model_kwargs
                )
                model = PeftModel.from_pretrained(
                    model,
                    lora_weights,
                    torch_dtype=torch.float16,
                    local_files_only=local_files_only,
                    resume_download=resume_download,
                    device_map="auto",
                )
                if load_half:
                    model.half()

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

    return model, tokenizer, device


def go_gradio(**kwargs):

    # get default model
    all_kwargs = kwargs.copy()
    all_kwargs.update(locals())
    if kwargs.get('base_model'):
        model, tokenizer, device = get_model(**all_kwargs)
    else:
        # if empty model, then don't load anything, just get gradio up
        model, tokenizer, device = None, None, None

    if 'mbart-' in kwargs['model_lower']:
        instruction_label = "Text to translate"
    else:
        instruction_label = "Instruction"
    if kwargs['chat']:
        instruction_label = "You (Shift-Enter or push Submit to send message)"

    title = 'h2oGPT'
    if kwargs['verbose']:
        description = f"""Model {kwargs['base_model']} Instruct dataset.
                      For more information, visit [the project's website](https://github.com/h2oai/h2ogpt).
                      Command: {str(' '.join(sys.argv))}
                      Hash: {get_githash()}
                      """
    else:
        description = ""

    if kwargs['verbose']:
        task_info_md = f"""
        ### Task: {kwargs['task_info']}"""
    else:
        task_info_md = ''

    css_code = """footer {visibility: hidden}
body{background-image:url("https://h2o.ai/content/experience-fragments/h2o/us/en/site/header/master/_jcr_content/root/container/header_copy/logo.coreimg.svg/1678976605175/h2o-logo.svg");}}"""

    from gradio.themes.utils import colors, fonts, sizes
    if kwargs['h2ocolors']:
        colors_dict = dict(primary_hue=colors.yellow,
                           secondary_hue=colors.yellow,
                           neutral_hue=colors.gray,
                           spacing_size=sizes.spacing_md,
                           radius_size=sizes.radius_md,
                           text_size=sizes.text_md,
                           )
    else:
        colors_dict = dict(primary_hue=colors.indigo,
                           secondary_hue=colors.indigo,
                           neutral_hue=colors.gray,
                           spacing_size=sizes.spacing_md,
                           radius_size=sizes.radius_md,
                           text_size=sizes.text_md,
                           )

    import gradio as gr
    demo = gr.Blocks(theme=gr.themes.Soft(**colors_dict), css=css_code, title="h2oGPT")
    callback = gr.CSVLogger()
    # css_code = 'body{background-image:url("https://h2o.ai/content/experience-fragments/h2o/us/en/site/header/master/_jcr_content/root/container/header_copy/logo.coreimg.svg/1678976605175/h2o-logo.svg");}'
    # demo = gr.Blocks(theme='gstaff/xkcd', css=css_code)

    from create_data import flatten_list
    model_options = flatten_list(list(prompt_type_to_model_name.values())) + kwargs['extra_model_options']
    if kwargs['base_model'].strip() not in model_options:
        lora_options = [kwargs['base_model'].strip()] + model_options
    lora_options = kwargs['extra_lora_options']
    if kwargs['lora_weights'].strip() not in lora_options:
        lora_options = [kwargs['lora_weights'].strip()] + lora_options
    # always add in no lora case
    # add fake space so doesn't go away in gradio dropdown
    lora_options = [' '] + kwargs['extra_lora_options']

    with demo:
        model_state = gr.State([model, tokenizer, device, kwargs['base_model']])
        gr.Markdown(
            f"""
            <h1 align="center"> {title}</h1>

            {description}
            {task_info_md}
            """)

        with gr.Tabs():
            with gr.Row():
                if not kwargs['chat']:
                    with gr.Column():
                        instruction = gr.Textbox(
                            lines=4, label=instruction_label,
                            placeholder=kwargs['placeholder_instruction'],
                        )
                        iinput = gr.Textbox(lines=4, label="Input",
                                            placeholder=kwargs['placeholder_input'])
                        submit = gr.Button(label='Submit')
                        flag_btn = gr.Button("Flag")
                with gr.Column():
                    if kwargs['chat']:
                        text_output = gr.Chatbot(label='h2oGPT').style(height=kwargs['height'] or 400)
                        with gr.Row():
                            with gr.Column(scale=50):
                                instruction = gr.Textbox(
                                    lines=4, label=instruction_label,
                                    placeholder=kwargs['placeholder_instruction'],
                                )
                            with gr.Row():  # .style(equal_height=False, equal_width=False):
                                submit = gr.Button(value='Submit').style(full_width=False, size='sm')
                                stop_btn = gr.Button(value="Stop").style(full_width=False, size='sm')
                        with gr.Row():
                            clear = gr.Button("New Conversation")
                            flag_btn = gr.Button("Flag")
                            retry = gr.Button("Regenerate")
                            undo = gr.Button("Undo")
                    else:
                        text_output = gr.Textbox(lines=5, label="Output")
            with gr.TabItem("Input/Output"):
                with gr.Row():
                    if 'mbart-' in kwargs['model_lower']:
                        src_lang = gr.Dropdown(list(languages_covered().keys()),
                                               value=kwargs['src_lang'],
                                               label="Input Language")
                        tgt_lang = gr.Dropdown(list(languages_covered().keys()),
                                               value=kwargs['tgt_lang'],
                                               label="Output Language")
            with gr.TabItem("Expert"):
                with gr.Row():
                    with gr.Column():
                        stream_output = gr.components.Checkbox(label="Stream output",
                                                               value=kwargs['stream_output'])
                        prompt_type = gr.Dropdown(prompt_types_strings,
                                                  value=kwargs['prompt_type'], label="Prompt Type")
                        temperature = gr.Slider(minimum=0, maximum=3,
                                                value=kwargs['temperature'],
                                                label="Temperature",
                                                info="Lower is deterministic (but may lead to repeats), Higher more creative (but may lead to hallucinations)")
                        top_p = gr.Slider(minimum=0, maximum=1,
                                          value=kwargs['top_p'], label="Top p",
                                          info="Cumulative probability of tokens to sample from")
                        top_k = gr.Slider(
                            minimum=0, maximum=100, step=1,
                            value=kwargs['top_k'], label="Top k",
                            info='Num. tokens to sample from'
                        )
                        num_beams = gr.Slider(minimum=1, maximum=8, step=1,
                                              value=kwargs['num_beams'], label="Beams",
                                              info="Number of searches for optimal overall probability.  Uses more GPU memory/compute")
                        max_new_tokens = gr.Slider(
                            minimum=1, maximum=2048, step=1,
                            value=kwargs['max_new_tokens'], label="Max output length"
                        )
                        min_new_tokens = gr.Slider(
                            minimum=0, maximum=2048, step=1,
                            value=kwargs['min_new_tokens'], label="Min output length"
                        )
                        early_stopping = gr.Checkbox(label="EarlyStopping", info="Stop early in beam search",
                                                     value=kwargs['early_stopping'])
                        max_time = gr.Slider(minimum=0, maximum=60 * 5, step=1,
                                             value=kwargs['max_time'], label="Max. time",
                                             info="Max. time to search optimal output.")
                        repetition_penalty = gr.Slider(minimum=0.01, maximum=3.0,
                                                       value=kwargs['repetition_penalty'],
                                                       label="Repetition Penalty")
                        num_return_sequences = gr.Slider(minimum=1, maximum=10, step=1,
                                                         value=kwargs['num_return_sequences'],
                                                         label="Number Returns", info="Must be <= num_beams")
                        do_sample = gr.Checkbox(label="Sample", info="Sample, for diverse output(s)",
                                                value=kwargs['do_sample'])
                        if kwargs['chat']:
                            iinput = gr.Textbox(lines=4, label="Input",
                                                placeholder=kwargs['placeholder_input'])
                        context = gr.Textbox(lines=1, label="Context",
                                             info="Ignored in chat mode.")  # nominally empty for chat mode

            with gr.TabItem("Models"):
                with gr.Row():
                    with gr.Column():
                        with gr.Row(scale=1):
                            with gr.Column(scale=50):
                                model_choice = gr.Dropdown(model_options, label="Choose Model", value=kwargs['base_model'])
                            with gr.Column(scale=1):
                                load_model_button = gr.Button("Load model")
                                model_used = gr.Textbox(label="Current Model", value=kwargs['base_model'])
                        with gr.Row(scale=1):
                            with gr.Column(scale=50):
                                new_model = gr.Textbox(label="New Model HF name/path")
                            with gr.Column(scale=1):
                                add_model_button = gr.Button("Add new model name")

                        with gr.Row(scale=1):
                            with gr.Column(scale=50):
                                lora_choice = gr.Dropdown(lora_options, label="Choose LORA", value=kwargs['lora_weights'])
                            with gr.Column(scale=1):
                                load_lora_button = gr.Button("Load LORA")
                                lora_used = gr.Textbox(label="Current LORA", value=kwargs['lora_weights'])
                        with gr.Row(scale=1):
                            with gr.Column(scale=50):
                                new_lora = gr.Textbox(label="New LORA HF name/path")
                            with gr.Column(scale=1):
                                add_lora_button = gr.Button("Add new LORA name")

        inputs_list = get_inputs_list(locals(), kwargs['model_lower'])
        from functools import partial
        all_kwargs = kwargs.copy()
        all_kwargs.update(locals())
        kwargs_evaluate = {k: v for k, v in all_kwargs.items() if k in inputs_kwargs_list}
        fun = partial(evaluate,
                      model_state,
                      **kwargs_evaluate)

        dark_mode_btn = gr.Button("Dark Mode", variant="primary").style(
            size="sm",
        )
        dark_mode_btn.click(
            None,
            None,
            None,
            _js="""() => {
            if (document.querySelectorAll('.dark').length) {
                document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
            } else {
                document.querySelector('body').classList.add('dark');
            }
        }""",
            api_name="dark",
        )
        if not kwargs['chat']:
            submit = gr.Button("Submit")
            submit_event = submit.click(fun, inputs=inputs_list, outputs=text_output, api_name='submit')

        # examples after submit or any other buttons for chat or no chat
        if kwargs['examples'] is not None and kwargs['show_examples']:
            gr.Examples(examples=kwargs['examples'], inputs=inputs_list)

        if kwargs['chat']:
            def user(*args, undo=False, sanitize_user_prompt=True):
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
                if sanitize_user_prompt:
                    from better_profanity import profanity
                    user_message1 = profanity.censor(user_message1)

                history = args_list[-1]
                if undo and history:
                    history.pop()
                args_list = args_list[:-1]
                if history is None:
                    print("Bad history, fix for now", flush=True)
                    history = []
                if undo:
                    return "", history
                else:
                    return "", history + [[user_message1, None]]

            def bot(*args, retry=False):
                args_list = list(args)
                history = args_list[-1]
                if retry and history:
                    history.pop()
                instruction1 = history[-1][0]
                context1 = ''
                if kwargs['chat_history'] > 0:
                    prompt_type1 = args_list[prompt_type_arg_id]
                    context1 = ''
                    for histi in range(len(history) - 1):
                        data_point = dict(instruction=history[histi][0], input='', output=history[histi][1])
                        context1 += generate_prompt(data_point, prompt_type1, kwargs['chat'], reduced=True)[0].replace(
                            '<br>', '\n')
                        if not context1.endswith('\n'):
                            context1 += '\n'
                    if context1 and not context1.endswith('\n'):
                        context1 += '\n'  # ensure if terminates abruptly, then human continues on next line
                args_list[0] = instruction1
                # only include desired chat history
                args_list[2] = context1[-kwargs['chat_history']:]
                model_state1 = args_list[-2]
                args_list = args_list[:-2]
                fun1 = partial(evaluate,
                               model_state1,
                               **kwargs_evaluate)
                try:
                    for output in fun1(*tuple(args_list)):
                        bot_message = output
                        history[-1][1] = bot_message
                        yield history
                except StopIteration:
                    yield history
                except RuntimeError as e:
                    if "generator raised StopIteration" in str(e):
                        # assume last entry was bad, undo
                        history.pop()
                        yield history
                    raise
                except Exception as e:
                    # put error into user input
                    history[-1][0] = "Exception: %s" % str(e)
                    yield history
                    raise
                return

            user_args = dict(fn=functools.partial(user, sanitize_user_prompt=kwargs['sanitize_user_prompt']),
                             inputs=inputs_list + [text_output],
                             outputs=[instruction, text_output],
                             )
            bot_args = dict(fn=bot,
                            inputs=inputs_list + [model_state] + [text_output],
                            outputs=[text_output],
                            )
            retry_bot_args = dict(fn=functools.partial(bot, retry=True),
                                  inputs=inputs_list + [model_state] + [text_output],
                                  outputs=[text_output],
                                  )
            undo_user_args = dict(fn=functools.partial(user, undo=True),
                                  inputs=inputs_list + [text_output],
                                  outputs=[instruction, text_output],
                                  )

            submit_event = instruction.submit(**user_args, queue=stream_output, api_name='instruction').then(
                **bot_args, api_name='instruction_bot',
            )
            submit_event2 = submit.click(**user_args, queue=stream_output, api_name='submit').then(
                **bot_args, api_name='submit_bot',
            )
            submit_event3 = retry.click(**user_args, queue=stream_output, api_name='retry').then(
                **retry_bot_args, api_name='retry_bot',
            )
            submit_event4 = undo.click(**undo_user_args, queue=stream_output, api_name='undo')
            clear.click(lambda: None, None, text_output, queue=False, api_name='clear')

        def load_model(model_name, lora_weights, model_state_old, prompt_type_old):
            # ensure old model removed from GPU memory
            if kwargs['debug']:
                print("Pre-switch pre-del GPU memory: %s" % torch.cuda.memory_allocated(), flush=True)
            if model_state_old[0] is not None:
                model_state_old[0].cpu()
                del model_state_old[0]
                model_state_old[0] = None
            if model_state_old[1] is not None:
                del model_state_old[1]
                model_state_old[1] = None
            torch.cuda.empty_cache()
            gc.collect()
            if kwargs['debug']:
                print("Pre-switch post-del GPU memory: %s" % torch.cuda.memory_allocated(), flush=True)
            all_kwargs['base_model'] = model_name.strip()
            model_lower = model_name.strip().lower()
            if model_lower in inv_prompt_type_to_model_lower:
                prompt_type1 = inv_prompt_type_to_model_lower[model_lower]
            else:
                prompt_type1 = prompt_type_old
            all_kwargs['lora_weights'] = lora_weights.strip()
            model1, tokenizer1, device1 = get_model(**all_kwargs)
            torch.cuda.empty_cache()
            gc.collect()
            if kwargs['debug']:
                print("Post-switch GPU memory: %s" % torch.cuda.memory_allocated(), flush=True)
            return {model_state: [model1, tokenizer1, device1, model_name],
                    model_used: model_name,
                    prompt_type: prompt_type1}

        def dropdown_prompt_type_list(x):
            return gr.Dropdown.update(value=x)

        load_model_event = load_model_button.click(fn=load_model,
                                                   inputs=[model_choice, lora_choice, model_state, prompt_type],
                                                   outputs=[model_state, model_used, lora_used, prompt_type]).then(
                                                   dropdown_prompt_type_list, prompt_type, prompt_type)

        def dropdown_model_list(x):
            new_options = [*model_options, x]
            return gr.Dropdown.update(value=x, choices=new_options), ''

        add_model_event = add_model_button.click(fn=dropdown_model_list,
                                                 inputs=new_model,
                                                 outputs=[model_choice, new_model])

        load_lora_event = load_lora_button.click(fn=load_model,
                                                 inputs=[model_choice, lora_choice, model_state, prompt_type],
                                                 outputs=[model_state, model_used, lora_used, prompt_type])

        def dropdown_lora_list(x):
            new_options = [*lora_options, x]
            return gr.Dropdown.update(value=x, choices=new_options), ''

        add_lora_event = add_lora_button.click(fn=dropdown_lora_list,
                                               inputs=new_lora,
                                               outputs=[lora_choice, new_lora])

        # callback for logging flagged input/output
        callback.setup(inputs_list + [text_output], "flagged_data_points")
        flag_btn.click(lambda *args: callback.flag(args), inputs_list + [text_output], None, preprocess=False,
                       api_name='flag')
        if kwargs['chat']:
            # don't pass text_output, don't want to clear output, just stop it
            # FIXME: have to click once to stop output and second time to stop GPUs going
            stop_btn.click(lambda: None, None, None, cancels=[submit_event, submit_event2, submit_event3],
                           queue=False, api_name='stop')

    demo.queue(concurrency_count=1)
    favicon_path = "h2o-logo.svg"
    demo.launch(share=kwargs['share'], server_name="0.0.0.0", show_error=True,
                favicon_path=favicon_path)  # , enable_queue=True)


input_args_list = ['model_state']
inputs_kwargs_list = ['debug', 'chat', 'hard_stop_list', 'sanitize_bot_response']


def get_inputs_list(inputs_dict, model_lower):
    inputs_list_names = list(inspect.signature(evaluate).parameters)
    inputs_list = []
    for k in inputs_list_names:
        if k == 'kwargs':
            continue
        if k in input_args_list + inputs_kwargs_list:
            # these are added via partial, not taken as input
            continue
        if 'mbart-' not in model_lower and k in ['src_lang', 'tgt_lang']:
            continue
        inputs_list.append(inputs_dict[k])
    return inputs_list


# index of prompt_type in evaluate function, after model_state
prompt_type_arg_id = 4


def evaluate(
        model_state,
        # START NOTE: Examples must have same order of parameters
        instruction,
        iinput,
        context,
        stream_output,
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
        # END NOTE: Examples must have same order of parameters
        src_lang=None,
        tgt_lang=None,
        debug=False,
        chat=False,
        hard_stop_list=None,
        sanitize_bot_response=True,
        **kwargs,
):
    if debug:
        locals_dict = locals().copy()
        locals_dict.pop('model_state', None)
        print(locals_dict)

    model, tokenizer, device, base_model = model_state

    assert base_model.strip(), (
        "Please choose a base model with --base_model (CLI) or in Models Tab (gradio).\nThen start New Conversation"
    )

    data_point = dict(context=context, instruction=instruction, input=iinput)
    prompter = Prompter(prompt_type, debug=debug, chat=chat, stream_output=stream_output)
    prompt = prompter.generate_prompt(data_point)

    if hard_stop_list is None:
        # acts like undo on user entry and bot response
        hard_stop_list = []

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
    if prompt_type in ['human_bot', 'instruct']:
        if prompt_type == 'human_bot':
            # encounters = [prompt.count(human) + 1, prompt.count(bot) + 1]
            # stopping only starts once output is beyond prompt
            # 1 human is enough to trigger, but need 2 bots, because very first view back will be bot we added
            stop_words = [human, bot]
            encounters = [1, 2]
        else:
            # some instruct prompts have this as end, doesn't hurt to stop on it since not common otherwise
            stop_words = ['### End']
            encounters = [1]
        stop_words_ids = [
            tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze(-1) for stop_word in stop_words]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, encounters=encounters)])
    else:
        stopping_criteria = StoppingCriteriaList()

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
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=top_k,
        num_beams=num_beams,
        do_sample=do_sample,
        repetition_penalty=float(repetition_penalty),
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
    if 'gpt2' in base_model.lower():
        gen_kwargs.update(dict(bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.eos_token_id))
    elif 'mbart-' in base_model.lower():
        assert tgt_lang is not None
        tgt_lang = languages_covered()[tgt_lang]
        gen_kwargs.update(dict(forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]))
    else:
        gen_kwargs.update(dict(pad_token_id=tokenizer.eos_token_id))

    decoder = functools.partial(tokenizer.decode,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True,
                                )
    decoder_raw = functools.partial(tokenizer.decode,
                                    skip_special_tokens=False,
                                    clean_up_tokenization_spaces=True,
                                    )

    with torch.no_grad():
        # decoded tokenized prompt can deviate from prompt due to special characters
        inputs_decoded = decoder(input_ids[0])
        inputs_decoded_raw = decoder_raw(input_ids[0])
        if inputs_decoded == prompt:
            # normal
            pass
        elif inputs_decoded_raw == prompt:
            # some models specify special tokens that are part of normal prompt, so can't skip them
            inputs_decoded_raw = inputs_decoded
            decoder = decoder_raw
        else:
            print("WARNING: Special characters in prompt", flush=True)
        if stream_output:
            def generate(callback=None, **kwargs):
                # re-order stopping so Stream first and get out all chunks before stop for other reasons
                stopping_criteria0 = kwargs.get('stopping_criteria', StoppingCriteriaList()).copy()
                kwargs['stopping_criteria'] = StoppingCriteriaList()
                kwargs['stopping_criteria'].append(Stream(func=callback))
                for stopping_criteria1 in stopping_criteria0:
                    kwargs['stopping_criteria'].append(stopping_criteria1)

                model.generate(**kwargs)

            for output in CallbackToGenerator(generate, callback=None, **gen_kwargs):
                decoded_output = decoder(output)
                if output[-1] in [tokenizer.eos_token_id]:
                    break
                if any(ele in decoded_output for ele in hard_stop_list):
                    raise StopIteration
                yield prompter.get_response(decoded_output, prompt=inputs_decoded,
                                            sanitize_bot_response=sanitize_bot_response)
            return
        else:
            outputs = model.generate(**gen_kwargs)
            outputs = [decoder(s) for s in outputs.sequences]
            yield prompter.get_response(outputs, prompt=inputs_decoded,
                                        sanitize_bot_response=sanitize_bot_response)


def get_generate_params(model_lower, chat,
                        stream_output, show_examples,
                        prompt_type, temperature, top_p, top_k, num_beams,
                        max_new_tokens, min_new_tokens, early_stopping, max_time,
                        repetition_penalty, num_return_sequences,
                        do_sample):
    use_defaults = False
    use_default_examples = True
    examples = []
    task_info = f"{prompt_type}"
    if model_lower:
        print(f"Using Model {model_lower}", flush=True)
    else:
        print("No model defined yet", flush=True)

    min_new_tokens = min_new_tokens if min_new_tokens is not None else 0
    early_stopping = early_stopping if early_stopping is not None else False
    max_time_defaults = 60 * 3
    max_time = max_time if max_time is not None else max_time_defaults

    if not prompt_type and model_lower in inv_prompt_type_to_model_lower:
        prompt_type = inv_prompt_type_to_model_lower[model_lower]

    if show_examples is None:
        if chat:
            show_examples = False
        else:
            show_examples = True

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
        examples += [
            [placeholder_instruction, "", "", stream_output, 'plain', 1.0, 1.0, 50, 1, 128, 0, False, max_time_defaults,
             1.0, 1,
             False]]
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
        examples += [
            [placeholder_instruction, "", "", stream_output, 'plain', 1.0, 1.0, 50, 1, 128, 0, False, max_time_defaults,
             1.0, 1,
             False]]
    elif 'gpt2' in model_lower:
        placeholder_instruction = "The sky is"
        placeholder_input = ""
        prompt_type = prompt_type or 'plain'
        use_default_examples = True  # some will be odd "continuations" but can be ok
        examples += [
            [placeholder_instruction, "", "", stream_output, 'plain', 1.0, 1.0, 50, 1, 128, 0, False, max_time_defaults,
             1.0, 1,
             False]]
        task_info = "Auto-complete phrase, code, etc."
        use_defaults = True
    else:
        if chat:
            placeholder_instruction = "Enter a question or imperative."
        else:
            placeholder_instruction = "Give detailed answer for whether Einstein or Newton is smarter."
        placeholder_input = ""
        if model_lower:
            prompt_type = prompt_type or 'human_bot'
        else:
            prompt_type = ''
        examples += [[summarize_example1, 'Summarize' if prompt_type not in ['plain', 'instruct_simple'] else '', "",
                      stream_output, prompt_type or 'plain', 0.1, 0.75, 40, 4, 256, 0, False, max_time_defaults, 1.0, 1, False]]
        task_info = "No task"
        if prompt_type == 'instruct':
            task_info = "Answer question or follow imperative as instruction with optionally input."
        elif prompt_type == 'plain':
            task_info = "Auto-complete phrase, code, etc."
        elif prompt_type == 'human_bot':
            if chat:
                task_info = "Chat (Shift-Enter to give question/imperative, input concatenated with instruction)"
            else:
                task_info = "Ask question/imperative (input concatenated with instruction)"

    # revert to plain if still nothing
    prompt_type = prompt_type or 'plain'
    if use_defaults:
        temperature = 1.0 if temperature is None else temperature
        top_p = 1.0 if top_p is None else top_p
        top_k = 40 if top_k is None else top_k
        num_beams = num_beams or 1
        max_new_tokens = max_new_tokens or 128
        repetition_penalty = repetition_penalty or 1.07
        num_return_sequences = min(num_beams, num_return_sequences or 1)
        do_sample = False if do_sample is None else do_sample
    else:
        temperature = 0.1 if temperature is None else temperature
        top_p = 0.75 if top_p is None else top_p
        top_k = 40 if top_k is None else top_k
        if chat:
            num_beams = num_beams or 1
        else:
            num_beams = num_beams or 4
        max_new_tokens = max_new_tokens or 256
        repetition_penalty = repetition_penalty or 1.07
        num_return_sequences = min(num_beams, num_return_sequences or 1)
        do_sample = False if do_sample is None else do_sample
    params_list = ["", stream_output, prompt_type, temperature, top_p, top_k, num_beams, max_new_tokens, min_new_tokens,
                   early_stopping, max_time, repetition_penalty, num_return_sequences, do_sample]

    if use_default_examples:
        examples += [
            ["Translate English to French", "Good morning"] + params_list,
            ["Give detailed answer for whether Einstein or Newton is smarter.", ''] + params_list,
            ["Explain in detailed list, all the best practices for coding in python.", ''] + params_list,
            [
                "Create a markdown table with 3 rows for the primary colors, and 2 columns, with color name and hex codes.",
                ''] + params_list,
            ["Why do you think you're so smart?", ''] + params_list,
            ['Translate to German:  My name is Arthur', ''] + params_list,
            ["Please answer to the following question. Who is going to be the next Ballon d'or?", ''] + params_list,
            ['Can Geoffrey Hinton have a conversation with George Washington? Give the rationale before answering.',
             ''] + params_list,
            ['Please answer the following question. What is the boiling point of Nitrogen?', ''] + params_list,
            ['Answer the following yes/no question. Can you write a whole Haiku in a single tweet?', ''] + params_list,
            ["Simplify the following expression: (False or False and True). Explain your answer.", ''] + params_list,
            [
                "Premise: At my age you will probably have learnt one lesson. Hypothesis:  It's not certain how many lessons you'll learn by your thirties. Does the premise entail the hypothesis?",
                ''] + params_list,
            ['The square root of x is the cube root of y. What is y to the power of 2, if x = 4?', ''] + params_list,
            [
                'Answer the following question by reasoning step by step.  The cafeteria had 23 apples. If they used 20 for lunch, and bought 6 more, how many apple do they have?',
                ''] + params_list,
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
            ['Name one good reason why paying customers should get better software than non-paying customers.',
             ''] + params_list,
            [
                'List the most idiotic letters of the alphabet, as a list of tuples (lowercase, uppercase), in reverse order, and convert to JSON.',
                ''] + params_list,
        ]

    src_lang = "English"
    tgt_lang = "Russian"

    return placeholder_instruction, placeholder_input, \
           stream_output, show_examples, \
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
