import os
import ast
import time
from enums import PromptType  # also supports imports from this file from other files

non_hf_types = ['gpt4all_llama', 'llama', 'gptj']

prompt_type_to_model_name = {
    'plain': [
        'EleutherAI/gpt-j-6B',
        'EleutherAI/pythia-6.9b',
        'EleutherAI/pythia-12b',
        'EleutherAI/pythia-12b-deduped',
        'EleutherAI/gpt-neox-20b',
        'openlm-research/open_llama_7b_700bt_preview',
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
    ],
    'gptj': ['gptj', 'gpt4all_llama'],
    'prompt_answer': [
        'h2oai/h2ogpt-gm-oasst1-en-1024-20b',
        'h2oai/h2ogpt-gm-oasst1-en-1024-12b',
        'h2oai/h2ogpt-gm-oasst1-multilang-1024-20b',
        'h2oai/h2ogpt-gm-oasst1-multilang-2048-falcon-7b',
        'h2oai/h2ogpt-gm-oasst1-multilang-2048-falcon-7b-v2',
        'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3',
        'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b',
        'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2',
        'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v1',
        'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2',
        'h2oai/h2ogpt-gm-oasst1-en-xgen-7b-8k',
        'h2oai/h2ogpt-gm-oasst1-multilang-xgen-7b-8k',
        'TheBloke/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2-GPTQ',
    ],
    'prompt_answer_openllama': [
        'h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b-preview-300bt',
        'h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b-preview-300bt-v2',
        'h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b-preview-700bt',
        'h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b',
        'h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-13b',
    ],
    'instruct': ['TheBloke/llama-30b-supercot-SuperHOT-8K-fp16'],  # https://huggingface.co/TheBloke/llama-30b-supercot-SuperHOT-8K-fp16#prompting
    'instruct_with_end': ['databricks/dolly-v2-12b'],
    'quality': [],
    'human_bot': [
        'h2oai/h2ogpt-oasst1-512-12b',
        'h2oai/h2ogpt-oasst1-512-20b',
        'h2oai/h2ogpt-oig-oasst1-256-6_9b',
        'h2oai/h2ogpt-oig-oasst1-512-6_9b',
        'h2oai/h2ogpt-oig-oasst1-256-6.9b',  # legacy
        'h2oai/h2ogpt-oig-oasst1-512-6.9b',  # legacy
        'h2oai/h2ogpt-research-oasst1-512-30b',
        'h2oai/h2ogpt-research-oasst1-llama-65b',
        'h2oai/h2ogpt-oasst1-falcon-40b',
        'h2oai/h2ogpt-oig-oasst1-falcon-40b',
    ],
    'dai_faq': [],
    'summarize': [],
    'simple_instruct': ['t5-small', 't5-large', 'google/flan-t5', 'google/flan-t5-xxl', 'google/flan-ul2'],
    'instruct_vicuna': ['AlekseyKorshuk/vicuna-7b', 'TheBloke/stable-vicuna-13B-HF', 'junelee/wizard-vicuna-13b'],
    'human_bot_orig': ['togethercomputer/GPT-NeoXT-Chat-Base-20B'],
    "open_assistant": ['OpenAssistant/oasst-sft-7-llama-30b-xor', 'oasst-sft-7-llama-30b'],
    "wizard_lm": ['ehartford/WizardLM-7B-Uncensored', 'ehartford/WizardLM-13B-Uncensored'],
    "wizard_mega": ['openaccess-ai-collective/wizard-mega-13b'],
    "instruct_simple": ['JosephusCheung/Guanaco'],
    "wizard_vicuna": ['ehartford/Wizard-Vicuna-13B-Uncensored'],
    "wizard2": ['llama'],
    "mptinstruct": ['mosaicml/mpt-30b-instruct', 'mosaicml/mpt-7b-instruct', 'mosaicml/mpt-30b-instruct'],
    "mptchat": ['mosaicml/mpt-7b-chat', 'mosaicml/mpt-30b-chat', 'TheBloke/mpt-30B-chat-GGML'],
    "vicuna11": ['lmsys/vicuna-33b-v1.3'],
    "falcon": ['tiiuae/falcon-40b-instruct', 'tiiuae/falcon-40b', 'tiiuae/falcon-7b-instruct', 'tiiuae/falcon-7b'],
    # could be plain, but default is correct prompt_type for default TheBloke model ggml-wizardLM-7B.q4_2.bin
}
if os.getenv('OPENAI_API_KEY'):
    prompt_type_to_model_name.update({
        "openai": ["text-davinci-003", "text-curie-001", "text-babbage-001", "text-ada-001"],
        "openai_chat": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
    })

inv_prompt_type_to_model_name = {v.strip(): k for k, l in prompt_type_to_model_name.items() for v in l}
inv_prompt_type_to_model_lower = {v.strip().lower(): k for k, l in prompt_type_to_model_name.items() for v in l}

prompt_types_strings = []
for p in PromptType:
    prompt_types_strings.extend([p.name])

prompt_types = []
for p in PromptType:
    prompt_types.extend([p.name, p.value, str(p.value)])


def get_prompt(prompt_type, prompt_dict, chat, context, reduced, making_context, return_dict=False):
    prompt_dict_error = ''
    generates_leading_space = False

    if prompt_type == PromptType.custom.name and not isinstance(prompt_dict, dict):
        try:
            prompt_dict = ast.literal_eval(prompt_dict)
        except BaseException as e:
            prompt_dict_error = str(e)
    if prompt_dict_error:
        promptA = None
        promptB = None
        PreInstruct = None
        PreInput = ''
        PreResponse = ''
        terminate_response = None
        chat_sep = ''
        chat_turn_sep = ''
        humanstr = ''
        botstr = ''
        generates_leading_space = False
    elif prompt_type in [PromptType.custom.value, str(PromptType.custom.value),
                         PromptType.custom.name]:
        promptA = prompt_dict.get('promptA', '')
        promptB = prompt_dict.get('promptB', '')
        PreInstruct = prompt_dict.get('PreInstruct', '')
        PreInput = prompt_dict.get('PreInput', '')
        PreResponse = prompt_dict.get('PreResponse', '')
        terminate_response = prompt_dict.get('terminate_response', None)
        chat_sep = prompt_dict.get('chat_sep', '\n')
        chat_turn_sep = prompt_dict.get('chat_turn_sep', '\n')
        humanstr = prompt_dict.get('humanstr', '')
        botstr = prompt_dict.get('botstr', '')
    elif prompt_type in [PromptType.plain.value, str(PromptType.plain.value),
                         PromptType.plain.name]:
        promptA = promptB = PreInstruct = PreInput = PreResponse = None
        terminate_response = []
        chat_turn_sep = chat_sep = ''
        # plain should have None for human/bot, so nothing truncated out, not '' that would truncate after first token
        humanstr = None
        botstr = None
    elif prompt_type == 'simple_instruct':
        promptA = promptB = PreInstruct = PreInput = PreResponse = None
        terminate_response = []
        chat_turn_sep = chat_sep = '\n'
        humanstr = None
        botstr = None
    elif prompt_type in [PromptType.instruct.value, str(PromptType.instruct.value),
                         PromptType.instruct.name] + [PromptType.instruct_with_end.value,
                                                      str(PromptType.instruct_with_end.value),
                                                      PromptType.instruct_with_end.name]:
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
        if prompt_type in [PromptType.instruct_with_end.value, str(PromptType.instruct_with_end.value),
                           PromptType.instruct_with_end.name]:
            terminate_response = ['### End']
        else:
            terminate_response = None
        chat_turn_sep = chat_sep = '\n'
        humanstr = PreInstruct
        botstr = PreResponse
    elif prompt_type in [PromptType.quality.value, str(PromptType.quality.value),
                         PromptType.quality.name]:
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
        chat_turn_sep = chat_sep = '\n'
        humanstr = PreInstruct  # first thing human says
        botstr = PreResponse  # first thing bot says
    elif prompt_type in [PromptType.human_bot.value, str(PromptType.human_bot.value),
                         PromptType.human_bot.name] + [PromptType.human_bot_orig.value,
                                                       str(PromptType.human_bot_orig.value),
                                                       PromptType.human_bot_orig.name]:
        human = '<human>:'
        bot = "<bot>:"
        if reduced or context or prompt_type in [PromptType.human_bot.value, str(PromptType.human_bot.value),
                                                 PromptType.human_bot.name]:
            preprompt = ''
        else:
            cur_date = time.strftime('%Y-%m-%d')
            cur_time = time.strftime('%H:%M:%S %p %Z')

            PRE_PROMPT = """\
Current Date: {}
Current Time: {}

"""
            preprompt = PRE_PROMPT.format(cur_date, cur_time)
        start = ''
        promptB = promptA = '%s%s' % (preprompt, start)

        PreInstruct = human + ' '

        PreInput = None

        if making_context:
            # when making context, want it to appear as-if LLM generated, which starts with space after :
            PreResponse = bot + ' '
        else:
            # normally LLM adds space after this, because was how trained.
            # if add space here, non-unique tokenization will often make LLM produce wrong output
            PreResponse = bot

        terminate_response = ['\n' + human, '\n' + bot, human, bot, PreResponse]
        chat_turn_sep = chat_sep = '\n'
        humanstr = human  # tag before human talks
        botstr = bot  # tag before bot talks
        generates_leading_space = True
    elif prompt_type in [PromptType.dai_faq.value, str(PromptType.dai_faq.value),
                         PromptType.dai_faq.name]:
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
        chat_turn_sep = chat_sep = terminate_response
        humanstr = PreInstruct
        botstr = PreResponse
    elif prompt_type in [PromptType.summarize.value, str(PromptType.summarize.value),
                         PromptType.summarize.name]:
        promptA = promptB = PreInput = ''
        PreInstruct = '## Main Text\n\n'
        PreResponse = '\n\n## Summary\n\n'
        terminate_response = None
        chat_turn_sep = chat_sep = '\n'
        humanstr = PreInstruct
        botstr = PreResponse
    elif prompt_type in [PromptType.instruct_vicuna.value, str(PromptType.instruct_vicuna.value),
                         PromptType.instruct_vicuna.name]:
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
        chat_turn_sep = chat_sep = '\n'
        humanstr = PreInstruct
        botstr = PreResponse
    elif prompt_type in [PromptType.prompt_answer.value, str(PromptType.prompt_answer.value),
                         PromptType.prompt_answer.name]:
        preprompt = ''
        prompt_tokens = "<|prompt|>"
        answer_tokens = "<|answer|>"
        start = ''
        promptB = promptA = '%s%s' % (preprompt, start)
        PreInstruct = prompt_tokens
        PreInput = None
        PreResponse = answer_tokens
        eos = '<|endoftext|>'  # neox eos
        humanstr = prompt_tokens
        botstr = answer_tokens
        terminate_response = [humanstr, PreResponse, eos]
        chat_sep = eos
        chat_turn_sep = eos
    elif prompt_type in [PromptType.prompt_answer_openllama.value, str(PromptType.prompt_answer_openllama.value),
                         PromptType.prompt_answer_openllama.name]:
        preprompt = ''
        prompt_tokens = "<|prompt|>"
        answer_tokens = "<|answer|>"
        start = ''
        promptB = promptA = '%s%s' % (preprompt, start)
        PreInstruct = prompt_tokens
        PreInput = None
        PreResponse = answer_tokens
        eos = '</s>'  # llama eos
        humanstr = prompt_tokens
        botstr = answer_tokens
        terminate_response = [humanstr, PreResponse, eos]
        chat_sep = eos
        chat_turn_sep = eos
    elif prompt_type in [PromptType.open_assistant.value, str(PromptType.open_assistant.value),
                         PromptType.open_assistant.name]:
        # From added_tokens.json
        preprompt = ''
        prompt_tokens = "<|prompter|>"
        answer_tokens = "<|assistant|>"
        start = ''
        promptB = promptA = '%s%s' % (preprompt, start)
        PreInstruct = prompt_tokens
        PreInput = None
        PreResponse = answer_tokens
        pend = "<|prefix_end|>"
        eos = "</s>"
        humanstr = prompt_tokens
        botstr = answer_tokens
        terminate_response = [humanstr, PreResponse, pend, eos]
        chat_turn_sep = chat_sep = eos
    elif prompt_type in [PromptType.wizard_lm.value, str(PromptType.wizard_lm.value),
                         PromptType.wizard_lm.name]:
        # https://github.com/ehartford/WizardLM/blob/main/src/train_freeform.py
        preprompt = ''
        start = ''
        promptB = promptA = '%s%s' % (preprompt, start)
        PreInstruct = ""
        PreInput = None
        PreResponse = "\n\n### Response\n"
        eos = "</s>"
        terminate_response = [PreResponse, eos]
        chat_turn_sep = chat_sep = eos
        humanstr = promptA
        botstr = PreResponse
    elif prompt_type in [PromptType.wizard_mega.value, str(PromptType.wizard_mega.value),
                         PromptType.wizard_mega.name]:
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
        chat_turn_sep = chat_sep = '\n'
        humanstr = PreInstruct
        botstr = PreResponse
    elif prompt_type in [PromptType.instruct_vicuna2.value, str(PromptType.instruct_vicuna2.value),
                         PromptType.instruct_vicuna2.name]:
        promptA = promptB = "" if not (chat and reduced) else ''

        PreInstruct = """
HUMAN:
"""

        PreInput = None

        PreResponse = """
ASSISTANT:
"""
        terminate_response = [
            'HUMAN:']  # but only allow terminate after prompt is found correctly, else can't terminate
        chat_turn_sep = chat_sep = '\n'
        humanstr = PreInstruct
        botstr = PreResponse
    elif prompt_type in [PromptType.instruct_vicuna3.value, str(PromptType.instruct_vicuna3.value),
                         PromptType.instruct_vicuna3.name]:
        promptA = promptB = "" if not (chat and reduced) else ''

        PreInstruct = """
### User:
"""

        PreInput = None

        PreResponse = """
### Assistant:
"""
        terminate_response = [
            '### User:']  # but only allow terminate after prompt is found correctly, else can't terminate
        chat_turn_sep = chat_sep = '\n'
        humanstr = PreInstruct
        botstr = PreResponse
    elif prompt_type in [PromptType.wizard2.value, str(PromptType.wizard2.value),
                         PromptType.wizard2.name]:
        # https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GGML
        preprompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.""" if not (
                chat and reduced) else ''
        start = ''
        promptB = promptA = '%s%s' % (preprompt, start)
        PreInstruct = """
### Instruction:
"""
        PreInput = None
        PreResponse = """
### Response:
"""
        terminate_response = [PreResponse]
        chat_turn_sep = chat_sep = '\n'
        humanstr = PreInstruct
        botstr = PreResponse
    elif prompt_type in [PromptType.wizard3.value, str(PromptType.wizard3.value),
                         PromptType.wizard3.name]:
        # https://huggingface.co/TheBloke/wizardLM-13B-1.0-GGML
        preprompt = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.""" if not (
                chat and reduced) else ''
        start = ''
        promptB = promptA = '%s%s' % (preprompt, start)
        PreInstruct = """USER: """
        PreInput = None
        PreResponse = """ASSISTANT: """
        terminate_response = [PreResponse]
        chat_turn_sep = chat_sep = '\n'
        humanstr = PreInstruct
        botstr = PreResponse
    elif prompt_type in [PromptType.wizard_vicuna.value, str(PromptType.wizard_vicuna.value),
                         PromptType.wizard_vicuna.name]:
        preprompt = ''
        start = ''
        promptB = promptA = '%s%s' % (preprompt, start)
        PreInstruct = """USER: """
        PreInput = None
        PreResponse = """ASSISTANT: """
        terminate_response = [PreResponse]
        chat_turn_sep = chat_sep = '\n'
        humanstr = PreInstruct
        botstr = PreResponse

    elif prompt_type in [PromptType.instruct_simple.value, str(PromptType.instruct_simple.value),
                         PromptType.instruct_simple.name]:
        promptB = promptA = '' if not (chat and reduced) else ''

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
        chat_turn_sep = chat_sep = '\n'
        humanstr = PreInstruct
        botstr = PreResponse
    elif prompt_type in [PromptType.openai.value, str(PromptType.openai.value),
                         PromptType.openai.name]:
        preprompt = """The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.""" if not (
                chat and reduced) else ''
        start = ''
        promptB = promptA = '%s%s' % (preprompt, start)
        PreInstruct = "\nHuman: "
        PreInput = None
        PreResponse = "\nAI:"
        terminate_response = [PreResponse] + [" Human:", " AI:"]
        chat_turn_sep = chat_sep = '\n'
        humanstr = PreInstruct
        botstr = PreResponse
    elif prompt_type in [PromptType.gptj.value, str(PromptType.gptj.value),
                         PromptType.gptj.name]:
        preprompt = "### Instruction:\n The prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response." if not (
                chat and reduced) else ''
        start = ''
        promptB = promptA = '%s%s' % (preprompt, start)
        PreInstruct = "\n### Prompt: "
        PreInput = None
        PreResponse = "\n### Response: "
        terminate_response = [PreResponse] + ["Prompt:", "Response:"]
        chat_turn_sep = chat_sep = '\n'
        humanstr = PreInstruct
        botstr = PreResponse
    elif prompt_type in [PromptType.openai_chat.value, str(PromptType.openai_chat.value),
                         PromptType.openai_chat.name]:
        # prompting and termination all handled by endpoint
        preprompt = """"""
        start = ''
        promptB = promptA = '%s%s' % (preprompt, start)
        PreInstruct = ""
        PreInput = None
        PreResponse = ""
        terminate_response = []
        chat_turn_sep = chat_sep = '\n'
        humanstr = None
        botstr = None
    elif prompt_type in [PromptType.vicuna11.value, str(PromptType.vicuna11.value),
                         PromptType.vicuna11.name]:
        preprompt = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. """ if not (
                chat and reduced) else ''
        start = ''
        promptB = promptA = '%s%s' % (preprompt, start)
        eos = '</s>'
        PreInstruct = """USER: """
        PreInput = None
        PreResponse = """ASSISTANT:"""
        terminate_response = [PreResponse]
        chat_sep = ' '
        chat_turn_sep = eos
        humanstr = PreInstruct
        botstr = PreResponse

        if making_context:
            # when making context, want it to appear as-if LLM generated, which starts with space after :
            PreResponse = PreResponse + ' '
        else:
            # normally LLM adds space after this, because was how trained.
            # if add space here, non-unique tokenization will often make LLM produce wrong output
            PreResponse = PreResponse
    elif prompt_type in [PromptType.mptinstruct.value, str(PromptType.mptinstruct.value),
                         PromptType.mptinstruct.name]:
        # https://huggingface.co/mosaicml/mpt-30b-instruct#formatting
        promptA = promptB = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n' if not (
                chat and reduced) else ''

        PreInstruct = """
### Instruction
"""

        PreInput = """
### Input
"""

        PreResponse = """
### Response
"""
        terminate_response = None
        chat_turn_sep = chat_sep = '\n'
        humanstr = PreInstruct
        botstr = PreResponse
    elif prompt_type in [PromptType.mptchat.value, str(PromptType.mptchat.value),
                         PromptType.mptchat.name]:
        # https://huggingface.co/TheBloke/mpt-30B-chat-GGML#prompt-template
        promptA = promptB = """<|im_start|>system\nA conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.\n<|im_end|>""" if not (
                chat and reduced) else ''

        PreInstruct = """<|im_start|>user
"""

        PreInput = None

        PreResponse = """<|im_end|><|im_start|>assistant
"""
        terminate_response = ['<|im_end|>']
        chat_sep = ''
        chat_turn_sep = '<|im_end|>'
        humanstr = PreInstruct
        botstr = PreResponse
    elif prompt_type in [PromptType.falcon.value, str(PromptType.falcon.value),
                         PromptType.falcon.name]:
        promptA = promptB = "" if not (chat and reduced) else ''

        PreInstruct = """User: """

        PreInput = None

        PreResponse = """Assistant:"""
        terminate_response = ['\nUser', "<|endoftext|>"]
        chat_sep = '\n\n'
        chat_turn_sep = '\n\n'
        humanstr = PreInstruct
        botstr = PreResponse
        if making_context:
            # when making context, want it to appear as-if LLM generated, which starts with space after :
            PreResponse = 'Assistant: '
        else:
            # normally LLM adds space after this, because was how trained.
            # if add space here, non-unique tokenization will often make LLM produce wrong output
            PreResponse = PreResponse
        # generates_leading_space = True
    elif prompt_type in [PromptType.guanaco.value, str(PromptType.guanaco.value),
                         PromptType.guanaco.name]:
        # https://huggingface.co/TheBloke/guanaco-65B-GPTQ
        promptA = promptB = "" if not (chat and reduced) else ''

        PreInstruct = """### Human: """

        PreInput = None

        PreResponse = """### Assistant:"""
        terminate_response = ['### Human:']  # but only allow terminate after prompt is found correctly, else can't terminate
        chat_turn_sep = chat_sep = '\n'
        humanstr = PreInstruct
        botstr = PreResponse
    else:
        raise RuntimeError("No such prompt_type=%s" % prompt_type)

    if isinstance(terminate_response, (tuple, list)):
        assert '' not in terminate_response, "Bad terminate_response"

    ret_dict = dict(promptA=promptA, promptB=promptB, PreInstruct=PreInstruct, PreInput=PreInput,
                    PreResponse=PreResponse, terminate_response=terminate_response, chat_sep=chat_sep,
                    chat_turn_sep=chat_turn_sep,
                    humanstr=humanstr, botstr=botstr,
                    generates_leading_space=generates_leading_space)

    if return_dict:
        return ret_dict, prompt_dict_error
    else:
        return tuple(list(ret_dict.values()))


def generate_prompt(data_point, prompt_type, prompt_dict, chat, reduced, making_context):
    context = data_point.get('context')
    if context is None:
        context = ''
    instruction = data_point.get('instruction')
    input = data_point.get('input')
    output = data_point.get('output')
    prompt_type = data_point.get('prompt_type', prompt_type)
    prompt_dict = data_point.get('prompt_dict', prompt_dict)
    assert prompt_type in prompt_types, "Bad prompt type: %s" % prompt_type
    promptA, promptB, PreInstruct, PreInput, PreResponse, \
        terminate_response, chat_sep, chat_turn_sep, humanstr, botstr, \
        generates_leading_space = get_prompt(prompt_type, prompt_dict, chat,
                                             context, reduced, making_context)

    # could avoid if reduce=True, but too complex for parent functions to handle
    prompt = context

    if input and promptA:
        prompt += f"""{promptA}"""
    elif promptB:
        prompt += f"""{promptB}"""

    if instruction and PreInstruct is not None and input and PreInput is not None:
        prompt += f"""{PreInstruct}{instruction}{PreInput}{input}"""
        prompt = inject_chatsep(prompt_type, prompt, chat_sep=chat_sep)
    elif instruction and input and PreInstruct is None and PreInput is not None:
        prompt += f"""{PreInput}{instruction}
{input}"""
        prompt = inject_chatsep(prompt_type, prompt, chat_sep=chat_sep)
    elif input and instruction and PreInput is None and PreInstruct is not None:
        prompt += f"""{PreInstruct}{instruction}
{input}"""
        prompt = inject_chatsep(prompt_type, prompt, chat_sep=chat_sep)
    elif instruction and PreInstruct is not None:
        prompt += f"""{PreInstruct}{instruction}"""
        prompt = inject_chatsep(prompt_type, prompt, chat_sep=chat_sep)
    elif input and PreInput is not None:
        prompt += f"""{PreInput}{input}"""
        prompt = inject_chatsep(prompt_type, prompt, chat_sep=chat_sep)
    elif input and instruction and PreInput is not None:
        prompt += f"""{PreInput}{instruction}{input}"""
        prompt = inject_chatsep(prompt_type, prompt, chat_sep=chat_sep)
    elif input and instruction and PreInstruct is not None:
        prompt += f"""{PreInstruct}{instruction}{input}"""
        prompt = inject_chatsep(prompt_type, prompt, chat_sep=chat_sep)
    elif input and instruction:
        # i.e. for simple_instruct
        prompt += f"""{instruction}: {input}"""
        prompt = inject_chatsep(prompt_type, prompt, chat_sep=chat_sep)
    elif input:
        prompt += f"""{input}"""
        prompt = inject_chatsep(prompt_type, prompt, chat_sep=chat_sep)
    elif instruction:
        prompt += f"""{instruction}"""
        prompt = inject_chatsep(prompt_type, prompt, chat_sep=chat_sep)

    if PreResponse is not None:
        prompt += f"""{PreResponse}"""
        pre_response = PreResponse  # Don't use strip
    else:
        pre_response = ''

    if output:
        prompt += f"""{output}"""

    return prompt, pre_response, terminate_response, chat_sep, chat_turn_sep


def inject_chatsep(prompt_type, prompt, chat_sep=None):
    if chat_sep:
        # only add new line if structured prompt, while 'plain' is just generation of next tokens from input
        prompt += chat_sep
    return prompt


class Prompter(object):
    def __init__(self, prompt_type, prompt_dict, debug=False, chat=False, stream_output=False, repeat_penalty=True,
                 allowed_repeat_line_length=10):
        self.prompt_type = prompt_type
        self.prompt_dict = prompt_dict
        self.debug = debug
        self.chat = chat
        self.stream_output = stream_output
        self.repeat_penalty = repeat_penalty
        self.allowed_repeat_line_length = allowed_repeat_line_length
        self.prompt = None
        context = ""  # not for chat context
        reduced = False  # not for chat context
        making_context = False  # not for chat context
        self.promptA, self.promptB, self.PreInstruct, self.PreInput, self.PreResponse, \
            self.terminate_response, self.chat_sep, self.chat_turn_sep, self.humanstr, self.botstr, \
            self.generates_leading_space = \
            get_prompt(self.prompt_type, self.prompt_dict, chat, context, reduced, making_context)
        self.pre_response = self.PreResponse

    def generate_prompt(self, data_point, reduced=None):
        """
        data_point['context'] is assumed to be like a system prompt or pre-conversation, not inserted after user prompt
        :param data_point:
        :param reduced:
        :return:
        """
        reduced = data_point.get('context') not in ['', None] if reduced is None else reduced
        making_context = False  # whether really making final prompt or just generating context
        prompt, _, _, _, _ = generate_prompt(data_point, self.prompt_type, self.prompt_dict, self.chat, reduced,
                                             making_context)
        if self.debug:
            print("prompt: %s" % prompt, flush=True)
        # if have context, should have always reduced and only preappend promptA/B here
        if data_point.get('context'):
            if data_point.get('input') and self.promptA:
                prompt = self.promptA + prompt
            elif self.promptB:
                prompt = self.promptB + prompt

        self.prompt = prompt
        return prompt

    def get_response(self, outputs, prompt=None, sanitize_bot_response=False):
        if isinstance(outputs, str):
            outputs = [outputs]
        if self.debug:
            print("output:\n%s" % '\n\n'.join(outputs), flush=True)
        if prompt is not None:
            self.prompt = prompt

        def clean_response(response):
            meaningless_words = ['<pad>', '</s>', '<|endoftext|>']
            for word in meaningless_words:
                response = response.replace(word, "")
            if sanitize_bot_response:
                from better_profanity import profanity
                response = profanity.censor(response)
            if self.generates_leading_space and isinstance(response, str) and len(response) > 0 and response[0] == ' ':
                response = response[1:]
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
            if self.prompt_type in [PromptType.plain.value, str(PromptType.plain.value), PromptType.plain.name]:
                output = clean_response(output)
            elif prompt is None:
                # then use most basic parsing like pipeline
                if not self.botstr:
                    pass
                elif self.botstr in output:
                    if self.humanstr:
                        output = clean_response(output.split(self.botstr)[1].split(self.humanstr)[0])
                    else:
                        # i.e. use after bot but only up to next bot
                        output = clean_response(output.split(self.botstr)[1].split(self.botstr)[0])
                else:
                    # output = clean_response(output)
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
                output = clean_response(output)
                if self.repeat_penalty:
                    output = clean_repeats(output)
                if self.terminate_response and allow_terminate:
                    finds = []
                    for term in self.terminate_response:
                        finds.append(output.find(term))
                    finds = [x for x in finds if x >= 0]
                    if len(finds) > 0:
                        termi = finds[0]
                        output = output[:termi]
                    else:
                        output = output
            if multi_output:
                # prefix with output counter
                output = "\n=========== Output %d\n\n" % (1 + oi) + output
                if oi > 0:
                    # post fix outputs with seperator
                    output += '\n'
            output = self.fix_text(self.prompt_type, output)
            outputs[oi] = output
        # join all outputs, only one extra new line between outputs
        output = '\n'.join(outputs)
        if self.debug:
            print("outputclean:\n%s" % '\n\n'.join(outputs), flush=True)
        return output

    @staticmethod
    def fix_text(prompt_type1, text1):
        if prompt_type1 == 'human_bot':
            # hack bug in vLLM with stopping, stops right, but doesn't return last token
            hfix = '<human'
            if text1.endswith(hfix):
                text1= text1[:-len(hfix)]
        return text1

