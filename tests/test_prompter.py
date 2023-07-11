import time
import pytest

from tests.utils import wrap_test_forked
from src.enums import source_prefix, source_postfix
from src.prompter import generate_prompt

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
    return generate_prompt(example_data_point, prompt_type, '', False, False, False)


@wrap_test_forked
def test_test_prompt(prompt_type='instruct', data_point=0):
    example_data_point = example_data_points[data_point]
    example_data_point.pop('output', None)
    return generate_prompt(example_data_point, prompt_type, '', False, False, False)


@wrap_test_forked
def test_test_prompt2(prompt_type='human_bot', data_point=0):
    example_data_point = example_data_points[data_point]
    example_data_point.pop('output', None)
    res = generate_prompt(example_data_point, prompt_type, '', False, False, False)
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


@wrap_test_forked
@pytest.mark.parametrize("prompt_type,expected",
                         [
                             ('vicuna11', prompt_fastchat),
                             ('human_bot', prompt_humanbot),
                             ('prompt_answer', prompt_prompt_answer),
                             ('prompt_answer_openllama', prompt_prompt_answer_openllama),
                             ('mptinstruct', prompt_mpt_instruct),
                             ('mptchat', prompt_mpt_chat),
                             ('falcon', prompt_falcon),
                         ]
                         )
def test_prompt_with_context(prompt_type, expected):
    prompt_dict = None  # not used unless prompt_type='custom'
    langchain_mode = 'Disabled'
    chat = True
    model_max_length = 2048
    memory_restriction_level = 0
    keep_sources_in_context1 = False
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
    context = history_to_context(history, langchain_mode, prompt_type, prompt_dict, chat,
                                 model_max_length, memory_restriction_level,
                                 keep_sources_in_context1)
    print("duration2: %s %s" % (prompt_type, time.time() - t0), flush=True)
    t0 = time.time()
    instruction = history[-1][0]

    # get prompt
    prompter = Prompter(prompt_type, prompt_dict, debug=debug, chat=chat, stream_output=stream_output)
    print("duration3: %s %s" % (prompt_type, time.time() - t0), flush=True)
    t0 = time.time()
    data_point = dict(context=context, instruction=instruction, input=iinput)
    prompt = prompter.generate_prompt(data_point)
    print(prompt)
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


@pytest.mark.parametrize("prompt_type,expected",
                         [
                             ('vicuna11', prompt_fastchat1),
                             ('human_bot', prompt_humanbot1),
                             ('prompt_answer', prompt_prompt_answer1),
                             ('prompt_answer_openllama', prompt_prompt_answer_openllama1),
                             ('mptinstruct', prompt_mpt_instruct1),
                             ('mptchat', prompt_mpt_chat1),
                             ('falcon', prompt_falcon1),
                         ]
                         )
@wrap_test_forked
def test_prompt_with_no_context(prompt_type, expected):
    prompt_dict = None  # not used unless prompt_type='custom'
    chat = True
    iinput = ''
    stream_output = False
    debug = False

    from src.prompter import Prompter
    context = ''
    instruction = "Go to the market?"

    # get prompt
    prompter = Prompter(prompt_type, prompt_dict, debug=debug, chat=chat, stream_output=stream_output)
    data_point = dict(context=context, instruction=instruction, input=iinput)
    prompt = prompter.generate_prompt(data_point)
    print(prompt)
    assert prompt == expected
    assert prompt.find(source_prefix) == -1


@wrap_test_forked
def test_source():
    prompt = "Who are you?%s\nFOO\n%s" % (source_prefix, source_postfix)
    assert prompt.find(source_prefix) >= 0
