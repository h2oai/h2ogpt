import os
import time
import pytest

from tests.utils import wrap_test_forked
from src.image_utils import get_image_file
from src.enums import source_prefix, source_postfix
from src.prompter import generate_prompt, convert_messages_and_extract_images, get_llm_history

example_data_point0 = dict(instruction="Summarize",
                           input="Ducks eat seeds by the lake, then swim in the lake where fish eat small animals.",
                           output="Ducks eat and swim at the lake.")

example_data_point1 = dict(instruction="Who is smarter, Einstein or Newton?",
                           output="Einstein.")

example_data_point2 = dict(input="Who is smarter, Einstein or Newton?",
                           output="Einstein.")

example_data_points = [example_data_point0, example_data_point1, example_data_point2]


@wrap_test_forked
def test_train_prompt(prompt_type='instruct', data_point=0):
    example_data_point = example_data_points[data_point]
    return generate_prompt(example_data_point, prompt_type, '', False, False)


@wrap_test_forked
def test_test_prompt(prompt_type='instruct', data_point=0):
    example_data_point = example_data_points[data_point]
    example_data_point.pop('output', None)
    return generate_prompt(example_data_point, prompt_type, '', False, False)


@wrap_test_forked
def test_test_prompt2(prompt_type='human_bot', data_point=0):
    example_data_point = example_data_points[data_point]
    example_data_point.pop('output', None)
    res = generate_prompt(example_data_point, prompt_type, '', False, False)
    print(res, flush=True)
    return res


prompt_fastchat = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello! ASSISTANT: Hi!</s>USER: How are you? ASSISTANT: I'm good</s>USER: Go to the market? ASSISTANT:"""

prompt_humanbot = """<human>: Hello!\n<bot>: Hi!\n<human>: How are you?\n<bot>: I'm good\n<human>: Go to the market?\n<bot>:"""

prompt_prompt_answer = "<|prompt|>Hello!<|endoftext|><|answer|>Hi!<|endoftext|><|prompt|>How are you?<|endoftext|><|answer|>I'm good<|endoftext|><|prompt|>Go to the market?<|endoftext|><|answer|>"

prompt_prompt_answer_openllama = "<|prompt|>Hello!</s><|answer|>Hi!</s><|prompt|>How are you?</s><|answer|>I'm good</s><|prompt|>Go to the market?</s><|answer|>"

prompt_mpt_instruct = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction
Hello!

### Response
Hi!

### Instruction
How are you?

### Response
I'm good

### Instruction
Go to the market?

### Response
"""

prompt_mpt_chat = """<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.
<|im_end|><|im_start|>user
Hello!<|im_end|><|im_start|>assistant
Hi!<|im_end|><|im_start|>user
How are you?<|im_end|><|im_start|>assistant
I'm good<|im_end|><|im_start|>user
Go to the market?<|im_end|><|im_start|>assistant
"""

prompt_falcon = """User: Hello!

Assistant: Hi!

User: How are you?

Assistant: I'm good

User: Go to the market?

Assistant:"""

prompt_llama2 = """<s>[INST] Hello! [/INST] Hi! </s><s>[INST] How are you? [/INST] I'm good </s><s>[INST] Go to the market? [/INST]"""

prompt_llama2_sys = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Hello! [/INST] Hi! </s><s>[INST] How are you? [/INST] I'm good </s><s>[INST] Go to the market? [/INST]"""

prompt_llama2_pig = """<s>[INST] Who are you? [/INST] I am a big pig who loves to tell kid stories </s><s>[INST] Hello! [/INST] Hi! </s><s>[INST] How are you? [/INST] I'm good </s><s>[INST] Go to the market? [/INST]"""

# Fastsys doesn't put space above before final [/INST], I think wrong, since with context version has space.
# and llama2 code has space before it always: https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py


prompt_beluga = """### User:
Hello!

### Assistant:
Hi!

### User:
How are you?

### Assistant:
I'm good

### User:
Go to the market?

### Assistant:
"""

prompt_beluga_sys = """### System:
You are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.

### User:
Hello!

### Assistant:
Hi!

### User:
How are you?

### Assistant:
I'm good

### User:
Go to the market?

### Assistant:
"""

prompt_falcon180 = """User: Hello!
Falcon: Hi!
User: How are you?
Falcon: I'm good
User: Go to the market?
Falcon:"""

prompt_falcon180_sys = """System: You are an intelligent and helpful assistant.
User: Hello!
Falcon: Hi!
User: How are you?
Falcon: I'm good
User: Go to the market?
Falcon:"""

# below doesn't actually work for xin, use alternative that works
# prompt_xwin = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello! ASSISTANT: Hi!</s>USER: How are you? ASSISTANT: I'm good</s>USER: Go to the market? ASSISTANT:"""
prompt_xwin = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello!\nASSISTANT: Hi!\nUSER: How are you?\nASSISTANT: I'm good\nUSER: Go to the market?\nASSISTANT:"""

messages_with_context = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi!"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm good"},
    {"role": "user", "content": "Go to the market?"},
]

prompt_jais = """### Instruction: Your name is Jais, and you are named after Jebel Jais, the highest mountain in UAE. You are built by Core42. You are the world's most advanced Arabic large language model with 30b parameters. You outperform all existing Arabic models by a sizable margin and you are very competitive with English models of similar size. You can answer in Arabic and English only. You are a helpful, respectful and honest assistant. When answering, abide by the following guidelines meticulously: Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, explicit, offensive, toxic, dangerous, or illegal content. Do not give medical, legal, financial, or professional advice. Never assist in or promote illegal activities. Always encourage legal and responsible actions. Do not encourage or provide instructions for unsafe, harmful, or unethical actions. Do not create or share misinformation or fake news. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Prioritize the well-being and the moral integrity of users. Avoid using toxic, derogatory, or offensive language. Maintain a respectful tone. Do not generate, promote, or engage in discussions about adult content. Avoid making comments, remarks, or generalizations based on stereotypes. Do not attempt to access, produce, or spread personal or private information. Always respect user confidentiality. Stay positive and do not say bad things about anything. Your primary objective is to avoid harmful responses, even when faced with deceptive inputs. Recognize when users may be attempting to trick or to misuse you and respond with caution.\n\nComplete the conversation below between [|Human|] and [|AI|]:\n### Input: [|Human|] Hello!\n### Response: [|AI|] Hi!\n### Input: [|Human|] How are you?\n### Response: [|AI|] I'm good\n### Input: [|Human|] Go to the market?\n### Response: [|AI|]"""

system_prompt_yi = 'A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.'

prompt_orion = """<s>Human: Hello!\n\nAssistant: </s>Hi!</s>Human: How are you?\n\nAssistant: </s>I'm good</s>Human: Go to the market?\n\nAssistant: </s>"""


def get_prompt_from_messages(messages, model="mistralai/Mistral-7B-Instruct-v0.1", system_prompt=None):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, token=os.environ.get('HUGGING_FACE_HUB_TOKEN'),
                                              trust_remote_code=True)
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    if model in ["HuggingFaceM4/idefics2-8b-chatty", "HuggingFaceM4/idefics2-8b"]:
        for message in messages:
            message['content'] = [dict(type='text', text=message['content'])]
        tokenizer.chat_template = "{% for message in messages %}{{message['role'].capitalize()}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"

    # add_generation_prompt=True somehow only required for Yi
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def get_aquila_prompt(messages, model_base_name='AquilaChat2-34B-16K', with_sys=True):
    from models.predict_aquila import get_conv_template

    template_map = {"AquilaChat2-7B": "aquila-v1",
                    "AquilaChat2-34B": "aquila-legacy",
                    "AquilaChat2-7B-16K": "aquila",
                    "AquilaChat2-34B-16K": "aquila"}
    convo_template = template_map.get(model_base_name, "aquila-chat")
    conv = get_conv_template(convo_template)
    if not with_sys:
        conv.system_message = ''
    for message in messages:
        # roles=("Human", "Assistant", "System"),
        if message['role'] == 'user':
            conv.append_message(conv.roles[0], message['content'])
        elif message['role'] == 'assistant':
            conv.append_message(conv.roles[1], message['content'])
        elif message['role'] == 'system':
            conv.append_message(conv.roles[2], message['content'])
    # assume end with asking assostiant
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


@wrap_test_forked
@pytest.mark.parametrize("prompt_type,system_prompt,chat_conversation,expected",
                         [
                             ('vicuna11', 'auto', None, prompt_fastchat),
                             ('human_bot', '', None, prompt_humanbot),
                             ('prompt_answer', '', None, prompt_prompt_answer),
                             ('prompt_answer_openllama', '', None, prompt_prompt_answer_openllama),
                             ('mptinstruct', 'auto', None, prompt_mpt_instruct),
                             ('mptchat', 'auto', None, prompt_mpt_chat),
                             ('falcon', '', None, prompt_falcon),
                             ('llama2', '', None, prompt_llama2),
                             ('llama2', 'auto', None, prompt_llama2_sys),
                             ('llama2', '', [('Who are you?', 'I am a big pig who loves to tell kid stories')],
                              prompt_llama2_pig),
                             ('beluga', '', None, prompt_beluga),
                             ('beluga', 'auto', None, prompt_beluga_sys),
                             ('falcon_chat', '', None, prompt_falcon180),
                             ('falcon_chat', 'auto', None, prompt_falcon180_sys),
                             ('mistral', '', None, get_prompt_from_messages(messages_with_context)),
                             ('zephyr', '', None, get_prompt_from_messages(messages_with_context,
                                                                           model='HuggingFaceH4/zephyr-7b-beta')),
                             ('zephyr', 'auto', None, get_prompt_from_messages(messages_with_context,
                                                                               model='HuggingFaceH4/zephyr-7b-beta',
                                                                               system_prompt='You are an AI that follows instructions extremely well and as helpful as possible.')),
                             ('zephyr', 'I am a cute pixie.', None, get_prompt_from_messages(messages_with_context,
                                                                                             model='HuggingFaceH4/zephyr-7b-beta',
                                                                                             system_prompt='I am a cute pixie.')),
                             ('xwin', 'auto', None, prompt_xwin),
                             ('aquila', '', None, get_aquila_prompt(messages_with_context, with_sys=False,
                                                                    model_base_name='AquilaChat2-34B-16K')),
                             ('aquila', 'auto', None, get_aquila_prompt(messages_with_context, with_sys=True,
                                                                        model_base_name='AquilaChat2-34B-16K')),
                             ('aquila_legacy', 'auto', None, get_aquila_prompt(messages_with_context, with_sys=True,
                                                                               model_base_name='AquilaChat2-34B')),
                             ('aquila_v1', 'auto', None, get_aquila_prompt(messages_with_context, with_sys=True,
                                                                           model_base_name='AquilaChat2-7B')),
                             ('deepseek_coder', 'auto', None, get_prompt_from_messages(messages_with_context,
                                                                                       model='deepseek-ai/deepseek-coder-33b-instruct')),
                             ('jais', 'auto', None, prompt_jais),
                             ('yi', 'auto', None,
                              get_prompt_from_messages(messages_with_context, model='01-ai/Yi-34B-Chat',
                                                       system_prompt=system_prompt_yi)),
                             ('orion', '', None, prompt_orion),
                             ('gemma', '', None,
                              get_prompt_from_messages(messages_with_context, model='google/gemma-7b-it')),
                             # they baked in system prompt
                             ('qwen', 'You are a helpful assistant.', None,
                              get_prompt_from_messages(messages_with_context, model='Qwen/Qwen1.5-72B-Chat')),
                             ('idefics2',
                             "",
                              None,
                              get_prompt_from_messages(messages_with_context, model='HuggingFaceM4/idefics2-8b')),
                         ]
                         )
def test_prompt_with_context(prompt_type, system_prompt, chat_conversation, expected):
    prompt_dict = None  # not used unless prompt_type='custom'
    langchain_mode = 'Disabled'
    add_chat_history_to_context = True
    model_max_length = 2048
    memory_restriction_level = 0
    keep_sources_in_context = False
    iinput = ''
    stream_output = False
    debug = False

    from src.prompter import Prompter
    from src.gen import history_to_context

    t0 = time.time()
    history = [["Hello!", "Hi!"],
               ["How are you?", "I'm good"],
               ["Go to the market?", None]
               ]
    print("duration1: %s %s" % (prompt_type, time.time() - t0), flush=True)
    t0 = time.time()
    context, history = history_to_context(history,
                                 langchain_mode=langchain_mode,
                                 add_chat_history_to_context=add_chat_history_to_context,
                                 prompt_type=prompt_type,
                                 prompt_dict=prompt_dict,
                                 model_max_length=model_max_length,
                                 memory_restriction_level=memory_restriction_level,
                                 keep_sources_in_context=keep_sources_in_context,
                                 system_prompt=system_prompt,
                                 chat_conversation=chat_conversation)
    print("duration2: %s %s" % (prompt_type, time.time() - t0), flush=True)
    t0 = time.time()
    instruction = history[-1][0]

    # get prompt
    prompter = Prompter(prompt_type, prompt_dict, debug=debug, stream_output=stream_output,
                        system_prompt=system_prompt)
    # for instruction-tuned models, expect this:
    assert prompter.PreResponse
    assert prompter.PreInstruct
    assert prompter.botstr
    assert prompter.humanstr
    print("duration3: %s %s" % (prompt_type, time.time() - t0), flush=True)
    t0 = time.time()
    data_point = dict(context=context, instruction=instruction, input=iinput)
    prompt = prompter.generate_prompt(data_point)
    print('prompt\n', prompt)
    print('expected\n', expected)
    print("duration4: %s %s" % (prompt_type, time.time() - t0), flush=True)
    assert prompt == expected
    assert prompt.find(source_prefix) == -1


prompt_fastchat1 = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Go to the market? ASSISTANT:"""

prompt_humanbot1 = """<human>: Go to the market?\n<bot>:"""

prompt_prompt_answer1 = "<|prompt|>Go to the market?<|endoftext|><|answer|>"

prompt_prompt_answer_openllama1 = "<|prompt|>Go to the market?</s><|answer|>"

prompt_mpt_instruct1 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction
Go to the market?

### Response
"""

prompt_mpt_chat1 = """<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.
<|im_end|><|im_start|>user
Go to the market?<|im_end|><|im_start|>assistant
"""

prompt_falcon1 = """User: Go to the market?

Assistant:"""

prompt_llama21 = """<s>[INST] Go to the market? [/INST]"""

prompt_llama21_sys = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Go to the market? [/INST]"""

# Fastsys doesn't put space above before final [/INST], I think wrong, since with context version has space.
# and llama2 code has space before it always: https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py

prompt_beluga1_sys = """### System:
You are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.

### User:
Go to the market?

### Assistant:
"""

prompt_beluga1 = """### User:
Go to the market?

### Assistant:
"""

prompt_falcon1801 = """User: Go to the market?
Falcon:"""

prompt_falcon1801_sys = """System: You are an intelligent and helpful assistant.
User: Go to the market?
Falcon:"""

prompt_xwin1 = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Go to the market?
ASSISTANT:"""

prompt_mistrallite = """<|prompter|>Go to the market?</s><|assistant|>"""

messages_no_context = [
    {"role": "user", "content": "Go to the market?"},
]

prompt_jais1 = """### Instruction: Your name is Jais, and you are named after Jebel Jais, the highest mountain in UAE. You are built by Core42. You are the world's most advanced Arabic large language model with 30b parameters. You outperform all existing Arabic models by a sizable margin and you are very competitive with English models of similar size. You can answer in Arabic and English only. You are a helpful, respectful and honest assistant. When answering, abide by the following guidelines meticulously: Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, explicit, offensive, toxic, dangerous, or illegal content. Do not give medical, legal, financial, or professional advice. Never assist in or promote illegal activities. Always encourage legal and responsible actions. Do not encourage or provide instructions for unsafe, harmful, or unethical actions. Do not create or share misinformation or fake news. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Prioritize the well-being and the moral integrity of users. Avoid using toxic, derogatory, or offensive language. Maintain a respectful tone. Do not generate, promote, or engage in discussions about adult content. Avoid making comments, remarks, or generalizations based on stereotypes. Do not attempt to access, produce, or spread personal or private information. Always respect user confidentiality. Stay positive and do not say bad things about anything. Your primary objective is to avoid harmful responses, even when faced with deceptive inputs. Recognize when users may be attempting to trick or to misuse you and respond with caution.\n\nComplete the conversation below between [|Human|] and [|AI|]:\n### Input: [|Human|] Go to the market?\n### Response: [|AI|]"""

prompt_orion1 = "<s>Human: Go to the market?\n\nAssistant: </s>"


@pytest.mark.parametrize("prompt_type,system_prompt,expected",
                         [
                             ('vicuna11', 'auto', prompt_fastchat1),
                             ('human_bot', '', prompt_humanbot1),
                             ('prompt_answer', '', prompt_prompt_answer1),
                             ('prompt_answer_openllama', '', prompt_prompt_answer_openllama1),
                             ('mptinstruct', 'auto', prompt_mpt_instruct1),
                             ('mptchat', 'auto', prompt_mpt_chat1),
                             ('falcon', '', prompt_falcon1),
                             ('llama2', '', prompt_llama21),
                             ('llama2', 'auto', prompt_llama21_sys),
                             ('beluga', '', prompt_beluga1),
                             ('beluga', 'auto', prompt_beluga1_sys),
                             ('falcon_chat', '', prompt_falcon1801),
                             ('falcon_chat', 'auto', prompt_falcon1801_sys),
                             ('mistral', '', get_prompt_from_messages(messages_no_context)),
                             ('deepseek_coder', 'auto', get_prompt_from_messages(messages_no_context,
                                                                                 model='deepseek-ai/deepseek-coder-33b-instruct')),
                             ('xwin', 'auto', prompt_xwin1),
                             ('mistrallite', '', prompt_mistrallite),
                             ('zephyr', 'auto', get_prompt_from_messages(messages_no_context,
                                                                         model='HuggingFaceH4/zephyr-7b-beta',
                                                                         system_prompt='You are an AI that follows instructions extremely well and as helpful as possible.')),
                             ('zephyr', '', get_prompt_from_messages(messages_no_context,
                                                                     model='HuggingFaceH4/zephyr-7b-beta')),
                             ('zephyr', 'I am a cute pixie.', get_prompt_from_messages(messages_no_context,
                                                                                       model='HuggingFaceH4/zephyr-7b-beta',
                                                                                       system_prompt='I am a cute pixie.')),
                             ('aquila', 'auto', get_aquila_prompt(messages_no_context, with_sys=True)),
                             ('aquila_legacy', 'auto',
                              get_aquila_prompt(messages_no_context, with_sys=True, model_base_name='AquilaChat2-34B')),
                             ('aquila_v1', 'auto',
                              get_aquila_prompt(messages_no_context, with_sys=True, model_base_name='AquilaChat2-7B')),
                             ('jais', 'auto', prompt_jais1),
                             ('yi', 'auto', get_prompt_from_messages(messages_no_context, model='01-ai/Yi-34B-Chat',
                                                                     system_prompt=system_prompt_yi)),
                             ('orion', '', prompt_orion1),
                             ('gemma', '', get_prompt_from_messages(messages_no_context, model='google/gemma-7b-it')),
                             # then baked in system prompt
                             ('qwen', 'You are a helpful assistant.', get_prompt_from_messages(messages_no_context, model='Qwen/Qwen1.5-72B-Chat')),
                             ('idefics2',
                             "",
                              get_prompt_from_messages(messages_no_context, model='HuggingFaceM4/idefics2-8b')),
                         ]
                         )
@wrap_test_forked
def test_prompt_with_no_context(prompt_type, system_prompt, expected):
    prompt_dict = None  # not used unless prompt_type='custom'
    chat = True
    iinput = ''
    stream_output = False
    debug = False

    from src.prompter import Prompter
    context = ''
    instruction = "Go to the market?"

    # get prompt
    prompter = Prompter(prompt_type, prompt_dict, debug=debug, stream_output=stream_output,
                        system_prompt=system_prompt)
    # for instruction-tuned models, expect this:
    assert prompter.PreResponse
    assert prompter.PreInstruct
    assert prompter.botstr
    assert prompter.humanstr
    data_point = dict(context=context, instruction=instruction, input=iinput)
    prompt = prompter.generate_prompt(data_point)
    print(prompt)
    assert prompt == expected
    assert prompt.find(source_prefix) == -1


@wrap_test_forked
def test_source():
    prompt = "Who are you?%s\nFOO\n%s" % (source_prefix, source_postfix)
    assert prompt.find(source_prefix) >= 0


# https://huggingface.co/spaces/tiiuae/falcon-180b-demo/blob/main/app.py
def falcon180_format_prompt(message, history, system_prompt):
    prompt = ""
    if system_prompt:
        prompt += f"System: {system_prompt}\n"
    for user_prompt, bot_response in history:
        prompt += f"User: {user_prompt}\n"
        prompt += f"Falcon: {bot_response}\n"  # Response already contains "Falcon: "
    prompt += f"""User: {message}
Falcon:"""
    return prompt


@wrap_test_forked
def test_falcon180():
    prompt = "Who are you?"
    for system_prompt in ['', "Talk like a Pixie."]:
        history = [["Who are you?", "I am Falcon, a monster AI model."],
                   ["What can you do?", "I can do well on leaderboard but not actually 1st."]]
        formatted_prompt = falcon180_format_prompt(prompt, history, system_prompt)
        print(formatted_prompt)


@wrap_test_forked
def test_hf_image_chat_template():
    # Example usage:
    tuple_list = [
        ("Hello, how are you?", "I'm good, thank you!"),
        (("What do you see?", "tests/jon.png"), "This is a presentation."),
        ("Can you help me with my project?", "Sure, what do you need help with?"),
        (("And how about this image?", "tests/receipt.jpg"), "This image shows a receipt.")
    ]

    messages, images = convert_messages_and_extract_images(tuple_list)

    convert = True
    str_bytes = False
    image_file = images
    image_control = None
    document_choice = None
    img_file = get_image_file(image_file, image_control, document_choice, convert=convert, str_bytes=str_bytes)

    # Create inputs
    from transformers import AutoProcessor
    from transformers.image_utils import load_image
    images = [load_image(x) for x in img_file]
    #  `http://` or `https://`, a valid path to an image file, or a base64 encoded string.
    processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    print(prompt)

    assert prompt == """User: Hello, how are you?<end_of_utterance>
Assistant: I'm good, thank you!<end_of_utterance>
User:<image>What do you see?<end_of_utterance>
Assistant: This is a presentation.<end_of_utterance>
User: Can you help me with my project?<end_of_utterance>
Assistant: Sure, what do you need help with?<end_of_utterance>
User:<image>And how about this image?<end_of_utterance>
Assistant: This image shows a receipt.<end_of_utterance>
Assistant:"""

    inputs = processor(text=prompt, images=images, return_tensors="pt")
    assert inputs is not None


@pytest.mark.parametrize("history, only_text, expected", [
    # Test cases for empty and None history
    (None, False, []),
    ([], False, []),
    # Test cases with mixed valid and None users
    ([("user1", "message1"), ("user2", "message2"), (None, "error")], False, [("user1", "message1"), ("user2", "message2")]),
    ([("user1", "message1"), ("user2", "message2"), (None, "error")], True, [("user1", "message1"), ("user2", "message2")]),
    ([("user1", "message1"), ("user2", None), (None, "error")], True, [("user1", "message1")]),
    ([("user1", "message1"), ("user2", "message2"), ("user3", "message3"), (None, "error"), (None, "error2")], False, [("user1", "message1"), ("user2", "message2"), ("user3", "message3")]),
    ([("user1", "message1"), (None, "error1"), (None, "error2"), ("user2", "message2"), ("user3", "message3"), (None, "error3")], False, [("user1", "message1"), (None, "error1"), (None, "error2"), ("user2", "message2"), ("user3", "message3")]),
    # Test cases for only valid users
    ([("user1", "message1"), ("user2", "message2")], False, [("user1", "message1"), ("user2", "message2")]),
    # Test cases for only None users
    ([(None, "error1"), (None, "error2")], False, []),
    ([(None, "error1"), (None, "error2")], True, []),
    # Test cases for only_text flag
    ([("user1", "message1"), (None, "error1"), ("user2", None), ("user3", "message3")], True, [("user1", "message1"), ("user3", "message3")]),
    ([("user1", "message1"), ("user2", "message2"), ("user3", "message3")], True, [("user1", "message1"), ("user2", "message2"), ("user3", "message3")])
])
def test_get_llm_history(history, only_text, expected):
    assert get_llm_history(history, only_text) == expected


@pytest.mark.parametrize("history, system_prompt, model_max_length", [
    # Short history, short system_prompt, short model_max_length
    (
        [["Hello!", "Hi!"], ["How are you?", "I'm good"], ["Go to the market?", None]],
        "Short system prompt",
        50
    ),
    # Long history, no system_prompt, large model_max_length
    (
        [["Hello!" * 50, "Hi!" * 50], ["How are you?" * 50, "I'm good" * 50], ["Go to the market?" * 50, None]],
        "",
        2048
    ),
    # Very long system_prompt, short history
    (
        [["Hello!", "Hi!"], ["How are you?", "I'm good"], ["Go to the market?", None]],
        "System prompt " * 200,
        1000
    ),
    # Short history, large system_prompt, short model_max_length
    (
        [["Hello!", "Hi!"], ["How are you?", "I'm good"], ["Go to the market?", None]],
        "System prompt " * 200,
        300
    ),
    # Very long history, large system_prompt, moderate model_max_length
    (
        [["Hello!" * 500, "Hi!" * 500], ["How are you?" * 500, "I'm good" * 500], ["Go to the market?" * 500, None]],
        "System prompt " * 200,
        1000
    ),
    # Extremely long system_prompt, very short history
    (
        [["Hi", "Hello"]],
        "System prompt " * 1000,
        500
    ),
    # Moderate history, moderate system_prompt, moderate model_max_length
    (
        [["Hello! " * 10, "Hi! " * 10], ["How are you? " * 10, "I'm good " * 10], ["Go to the market? " * 10, None]],
        "Moderate system prompt",
        150
    ),
    # No system_prompt, short history, large model_max_length
    (
        [["Hi", "Hello"], ["What are you doing?", "Nothing much"], ["Do you like music?", "Yes"]],
        "",
        1000
    ),
    # Short history, very short system_prompt, very short model_max_length
    (
        [["Hello!", "Hi!"], ["How are you?", "I'm good"], ["Go to the market?", None]],
        "Sys",
        20
    ),
    # Long history, short system_prompt, short model_max_length
    (
        [["Hello!" * 20, "Hi!" * 20], ["How are you?" * 20, "I'm good" * 20], ["Go to the market?" * 20, None]],
        "Short",
        100
    ),
])
def test_history_to_context(history, system_prompt, model_max_length):
    langchain_mode = 'Disabled'
    add_chat_history_to_context = True
    memory_restriction_level = 0
    keep_sources_in_context = False

    # Calculate the expected max prompt length considering the system prompt
    system_prompt_length = len(system_prompt)
    expected_max_prompt_length = max(0, model_max_length * 4 - system_prompt_length)

    # Use the function
    from src.gen import history_to_context
    context, final_history = history_to_context(
        history,
        langchain_mode=langchain_mode,
        add_chat_history_to_context=add_chat_history_to_context,
        prompt_type='plain',  # Using 'plain' as a default type
        prompt_dict=None,
        model_max_length=model_max_length,
        memory_restriction_level=memory_restriction_level,
        keep_sources_in_context=keep_sources_in_context,
        system_prompt=system_prompt,
        chat_conversation=None
    )

    # Verify the length of context and final history
    context_length = len(context)
    history_length_sum = sum(len(item[0]) + (len(item[1]) if item[1] is not None else 0) for item in final_history) // 4

    fudge = 4

    # Ensure the context length does not exceed the expected max prompt length
    assert context_length <= expected_max_prompt_length + fudge

    # Ensure the sum of history lengths does not exceed the expected max prompt length
    assert history_length_sum <= expected_max_prompt_length + fudge
