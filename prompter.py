import time
from enum import Enum


class PromptType(Enum):
    plain = 0
    instruct = 1
    quality = 2
    human_bot = 3
    dai_faq = 4
    summarize = 5
    simple_instruct = 6
    instruct_vicuna = 7
    instruct_with_end = 8
    human_bot_orig = 9
    prompt_answer = 10
    open_assistant = 11
    wizard_lm = 12
    wizard_mega = 13


prompt_type_to_model_name = {
    'plain': [
        'EleutherAI/gpt-j-6B',
        'EleutherAI/pythia-6.9b',
        'EleutherAI/pythia-12b',
        'EleutherAI/pythia-12b-deduped',
        'EleutherAI/gpt-neox-20b',
        'decapoda-research/llama-7b-hf',
        'decapoda-research/llama-13b-hf',
        'decapoda-research/llama-30b-hf',
        'decapoda-research/llama-65b-hf',
        'facebook/mbart-large-50-many-to-many-mmt',
        'philschmid/bart-large-cnn-samsum',
        'philschmid/flan-t5-base-samsum',
        'gpt2',
        'distilgpt2',
        'mosaicml/mpt-7b-storywriter',
        'mosaicml/mpt-7b-instruct',  # internal code handles instruct
        'mosaicml/mpt-7b-chat',  # NC, internal code handles instruct
    ],
    'prompt_answer': [
        'h2oai/h2ogpt-gm-oasst1-en-1024-20b',
        'h2oai/h2ogpt-gm-oasst1-en-1024-12b',
        'h2oai/h2ogpt-gm-oasst1-multilang-1024-20b',
        'h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b-preview-300bt',
        'h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b-preview-300bt-v2',
    ],
    'instruct': [],
    'instruct_with_end': ['databricks/dolly-v2-12b'],
    'quality': [],
    'human_bot': [
        'h2oai/h2ogpt-oasst1-512-12b',
        'h2oai/h2ogpt-oasst1-512-20b',
        'h2oai/h2ogpt-oig-oasst1-256-6_9b',
        'h2oai/h2ogpt-oig-oasst1-512-6_9b',
        'h2oai/h2ogpt-research-oasst1-512-30b',  # private
    ],
    'dai_faq': [],
    'summarize': [],
    'simple_instruct': ['t5-small', 't5-large', 'google/flan-t5', 'google/flan-t5-xxl', 'google/flan-ul2'],
    'instruct_vicuna': ['AlekseyKorshuk/vicuna-7b', 'TheBloke/stable-vicuna-13B-HF', 'junelee/wizard-vicuna-13b'],
    'human_bot_orig': ['togethercomputer/GPT-NeoXT-Chat-Base-20B'],
    "open_assistant": ['OpenAssistant/oasst-sft-7-llama-30b-xor', 'oasst-sft-7-llama-30b'],
    "wizard_lm": ['ehartford/WizardLM-7B-Uncensored', 'ehartford/WizardLM-13B-Uncensored'],
    "wizard_mega": ['openaccess-ai-collective/wizard-mega-13b'],
}

inv_prompt_type_to_model_name = {v.strip(): k for k, l in prompt_type_to_model_name.items() for v in l}
inv_prompt_type_to_model_lower = {v.strip().lower(): k for k, l in prompt_type_to_model_name.items() for v in l}

prompt_types_strings = []
for p in PromptType:
    prompt_types_strings.extend([p.name])

prompt_types = []
for p in PromptType:
    prompt_types.extend([p.name, p.value, str(p.value)])


def get_prompt(prompt_type, chat, context, reduced):
    if prompt_type in [-1, "-1", "plain"]:
        promptA = promptB = PreInstruct = PreInput = PreResponse = ''
        terminate_response = []
        chat_sep = ''
        humanstr = ''
        botstr = ''
    elif prompt_type == 'simple_instruct':
        promptA = promptB = PreInstruct = PreInput = PreResponse = None
        terminate_response = []
        chat_sep = '\n'
        humanstr = ''
        botstr = ''
    elif prompt_type in [0, "0", "instruct"] or prompt_type in [7, "7", "instruct_with_end"]:
        promptA = 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n' if not (
                    chat and reduced) else ''
        promptB = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n' if not (
                    chat and reduced) else ''

        PreInstruct = """
### Instruction:
"""

        PreInput = """
### Input:
"""

        PreResponse = """
### Response:
"""
        if prompt_type in [7, "7", "instruct_with_end"]:
            terminate_response = ['### End']
        else:
            terminate_response = None
        chat_sep = '\n'
        humanstr = PreInstruct
        botstr = PreResponse
    elif prompt_type in [1, "1", "quality"]:
        promptA = 'Write a detailed high-quality, accurate, fair, Response with about 100 words by following the Instruction as applied on the Input.\n' if not (
                    chat and reduced) else ''
        promptB = 'Write a detailed high-quality, accurate, fair, Response with about 100 words by following the Instruction.\n' if not (
                    chat and reduced) else ''

        PreInstruct = """
### Instruction:
"""

        PreInput = """
### Input:
"""

        PreResponse = """
### Response:
"""
        terminate_response = None
        chat_sep = '\n'
        humanstr = PreInstruct  # first thing human says
        botstr = PreResponse  # first thing bot says
    elif prompt_type in [2, "2", "human_bot", 9, "9", "human_bot_orig"]:
        human = '<human>:'
        bot = "<bot>:"
        if reduced or context or prompt_type in [2, "2", "human_bot"]:
            preprompt = ''
        else:
            cur_date = time.strftime('%Y-%m-%d')
            cur_time = time.strftime('%H:%M:%S %p %Z')

            PRE_PROMPT = """\
Current Date: {}
Current Time: {}

"""
            preprompt = PRE_PROMPT.format(cur_date, cur_time)
        start = human
        promptB = promptA = '%s%s ' % (preprompt, start)

        PreInstruct = ""

        PreInput = None

        if reduced:
            # when making context, want it to appear as-if LLM generated, which starts with space after :
            PreResponse = bot + ' '
        else:
            # normally LLM adds space after this, because was how trained.
            # if add space here, non-unique tokenization will often make LLM produce wrong output
            PreResponse = bot

        terminate_response = [start, PreResponse]
        chat_sep = '\n'
        humanstr = human  # tag before human talks
        botstr = bot  # tag before bot talks
    elif prompt_type in [3, "3", "dai_faq"]:
        promptA = ''
        promptB = 'Answer the following Driverless AI question.\n'

        PreInstruct = """
### Driverless AI frequently asked question:
"""

        PreInput = None

        PreResponse = """
### Driverless AI documentation answer:
"""
        terminate_response = ['\n\n']
        chat_sep = terminate_response
        humanstr = PreInstruct
        botstr = PreResponse
    elif prompt_type in [5, "5", "summarize"]:
        promptA = promptB = PreInput = ''
        PreInstruct = '## Main Text\n\n'
        PreResponse = '\n\n## Summary\n\n'
        terminate_response = None
        chat_sep = '\n'
        humanstr = PreInstruct
        botstr = PreResponse
    elif prompt_type in [6, "6", "instruct_vicuna"]:
        promptA = promptB = "A chat between a curious human and an artificial intelligence assistant. " \
                            "The assistant gives helpful, detailed, and polite answers to the human's questions." if not (
                    chat and reduced) else ''

        PreInstruct = """
### Human:
"""

        PreInput = None

        PreResponse = """
### Assistant:
"""
        terminate_response = [
            '### Human:']  # but only allow terminate after prompt is found correctly, else can't terminate
        chat_sep = '\n'
        humanstr = PreInstruct
        botstr = PreResponse
    elif prompt_type in [10, "10", "prompt_answer"]:
        preprompt = ''
        prompt_tokens = "<|prompt|>"
        answer_tokens = "<|answer|>"
        start = prompt_tokens
        promptB = promptA = '%s%s' % (preprompt, start)
        PreInstruct = ""
        PreInput = None
        PreResponse = answer_tokens
        eos = '<|endoftext|>'  # neox eos
        terminate_response = [start, PreResponse, eos]
        chat_sep = eos
        humanstr = prompt_tokens
        botstr = answer_tokens
    elif prompt_type in [11, "11", "open_assistant"]:
        # From added_tokens.json
        preprompt = ''
        prompt_tokens = "<|prompter|>"
        answer_tokens = "<|assistant|>"
        start = prompt_tokens
        promptB = promptA = '%s%s' % (preprompt, start)
        PreInstruct = ""
        PreInput = None
        PreResponse = answer_tokens
        pend = "<|prefix_end|>"
        eos = "</s>"
        terminate_response = [start, PreResponse, pend, eos]
        chat_sep = eos
        humanstr = prompt_tokens
        botstr = answer_tokens
    elif prompt_type in [12, "12", "wizard_lm"]:
        # https://github.com/ehartford/WizardLM/blob/main/src/train_freeform.py
        preprompt = ''
        start = ''
        promptB = promptA = '%s%s' % (preprompt, start)
        PreInstruct = ""
        PreInput = None
        PreResponse = "\n\n### Response"
        eos = "</s>"
        terminate_response = [PreResponse, eos]
        chat_sep = eos
        humanstr = promptA
        botstr = PreResponse
    elif prompt_type in [13, "13", "wizard_mega"]:
        preprompt = ''
        start = ''
        promptB = promptA = '%s%s' % (preprompt, start)
        PreInstruct = """
### Instruction:
"""
        PreInput = None
        PreResponse = """
### Assistant:
"""
        terminate_response = [PreResponse]
        chat_sep = '\n'
        humanstr = PreInstruct
        botstr = PreResponse
    else:
        raise RuntimeError("No such prompt_type=%s" % prompt_type)

    return promptA, promptB, PreInstruct, PreInput, PreResponse, terminate_response, chat_sep, humanstr, botstr


def generate_prompt(data_point, prompt_type, chat, reduced):
    context = data_point.get('context')
    if context is None:
        context = ''
    instruction = data_point.get('instruction')
    input = data_point.get('input')
    output = data_point.get('output')
    prompt_type = data_point.get('prompt_type', prompt_type)
    assert prompt_type in prompt_types, "Bad prompt type: %s" % prompt_type
    promptA, promptB, PreInstruct, PreInput, PreResponse, \
        terminate_response, chat_sep, humanstr, botstr = get_prompt(prompt_type, chat, context, reduced)

    prompt = context if not reduced else ''

    if input and promptA:
        prompt += f"""{promptA}"""
    elif promptB:
        prompt += f"""{promptB}"""

    if instruction and PreInstruct is not None and input and PreInput is not None:
        prompt += f"""{PreInstruct}{instruction}{PreInput}{input}"""
        prompt = inject_newline(prompt_type, prompt)
    elif instruction and input and PreInstruct is None and PreInput is not None:
        prompt += f"""{PreInput}{instruction}
{input}"""
        prompt = inject_newline(prompt_type, prompt)
    elif input and instruction and PreInput is None and PreInstruct is not None:
        prompt += f"""{PreInstruct}{instruction}
{input}"""
        prompt = inject_newline(prompt_type, prompt)
    elif instruction and PreInstruct is not None:
        prompt += f"""{PreInstruct}{instruction}"""
        prompt = inject_newline(prompt_type, prompt)
    elif input and PreInput is not None:
        prompt += f"""{PreInput}{input}"""
        prompt = inject_newline(prompt_type, prompt)
    elif input and instruction and PreInput is not None:
        prompt += f"""{PreInput}{instruction}{input}"""
        prompt = inject_newline(prompt_type, prompt)
    elif input and instruction and PreInstruct is not None:
        prompt += f"""{PreInstruct}{instruction}{input}"""
        prompt = inject_newline(prompt_type, prompt)
    elif input and instruction:
        # i.e. for simple_instruct
        prompt += f"""{instruction}: {input}"""
        prompt = inject_newline(prompt_type, prompt)
    elif input:
        prompt += f"""{input}"""
        prompt = inject_newline(prompt_type, prompt)
    elif instruction:
        prompt += f"""{instruction}"""
        prompt = inject_newline(prompt_type, prompt)

    if PreResponse is not None:
        prompt += f"""{PreResponse}"""
        pre_response = PreResponse  # Don't use strip
    else:
        pre_response = ''

    if output:
        prompt += f"""{output}"""

    return prompt, pre_response, terminate_response, chat_sep


def inject_newline(prompt_type, prompt):
    if prompt_type not in [-1, '-1', 'plain', 'simple_instruct']:
        # only add new line if structured prompt, while 'plain' is just generation of next tokens from input
        prompt += '\n'
    return prompt


class Prompter(object):
    def __init__(self, prompt_type, debug=False, chat=False, stream_output=False, repeat_penalty=True,
                 allowed_repeat_line_length=10):
        self.prompt_type = prompt_type
        data_point = dict(instruction='', input='', output='')
        _, self.pre_response, self.terminate_response, self.chat_sep = \
            generate_prompt(data_point, prompt_type, chat, False)
        self.debug = debug
        self.chat = chat
        self.stream_output = stream_output
        self.repeat_penalty = repeat_penalty
        self.allowed_repeat_line_length = allowed_repeat_line_length
        self.prompt = None
        context = ""  # not for chat context
        reduced = False  # not for chat context
        self.promptA, self.promptB, self.PreInstruct, self.PreInput, self.PreResponse, \
            self.terminate_response, self.chat_sep, self.humanstr, self.botstr = \
            get_prompt(prompt_type, chat, context, reduced)

    def generate_prompt(self, data_point):
        reduced = False
        prompt, _, _, _ = generate_prompt(data_point, self.prompt_type, self.chat, reduced)
        if self.debug:
            print("prompt: ", prompt, flush=True)
        self.prompt = prompt
        return prompt

    def get_response(self, outputs, prompt=None, sanitize_bot_response=True):
        if isinstance(outputs, str):
            outputs = [outputs]
        if self.debug:
            print("output:\n", '\n\n'.join(outputs), flush=True)
        if prompt is not None:
            self.prompt = prompt

        def clean_response(response):
            meaningless_words = ['<pad>', '</s>', '<|endoftext|>']
            for word in meaningless_words:
                response = response.replace(word, "")
            if sanitize_bot_response:
                from better_profanity import profanity
                response = profanity.censor(response)
            response = response.strip("\n")
            return response

        def clean_repeats(response):
            lines = response.split('\n')
            new_lines = []
            [new_lines.append(line) for line in lines if
             line not in new_lines or len(line) < self.allowed_repeat_line_length]
            if self.debug and len(lines) != len(new_lines):
                print("cleaned repeats: %s %s" % (len(lines), len(new_lines)), flush=True)
            response = '\n'.join(new_lines)
            return response

        multi_output = len(outputs) > 1

        for oi, output in enumerate(outputs):
            if self.prompt_type in [0, '0', 'plain']:
                output = clean_response(output)
            elif prompt is None:
                # then use most basic parsing like pipeline
                if self.botstr in output:
                    if self.humanstr:
                        output = clean_response(output.split(self.botstr)[1].strip().split(self.humanstr)[0].strip())
                    else:
                        # i.e. use after bot but only up to next bot
                        output = clean_response(output.split(self.botstr)[1].strip().split(self.botstr)[0].strip())
                else:
                    # output = clean_response(output.strip())
                    # assume just not printed yet
                    output = ""
            else:
                # find first instance of prereponse
                # prompt sometimes has odd characters, that mutate length,
                # so can't go by length alone
                if self.pre_response:
                    outputi = output.find(prompt)
                    if outputi >= 0:
                        output = output[outputi + len(prompt):]
                        allow_terminate = True
                    else:
                        # subtraction is risky due to space offsets sometimes, so only do if necessary
                        output = output[len(prompt) - len(self.pre_response):]
                        # [1] to avoid repeated pre_response, just take first (after prompt - pre_response for chat)
                        if self.pre_response in output:
                            output = output.split(self.pre_response)[1]
                            allow_terminate = True
                        else:
                            if output:
                                print("Failure of parsing or not enough output yet: %s" % output, flush=True)
                            allow_terminate = False
                else:
                    allow_terminate = True
                    output = output[len(prompt):]
                # clean after subtract prompt out, so correct removal of pre_response
                output = clean_response(output).strip()
                if self.repeat_penalty:
                    output = clean_repeats(output).strip()
                if self.terminate_response and allow_terminate:
                    finds = []
                    for term in self.terminate_response:
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
        if self.debug:
            print("outputclean:\n", '\n\n'.join(outputs), flush=True)
        return output
