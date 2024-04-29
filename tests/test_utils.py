import functools
import json
import sys
import time

import pytest

from src.enums import invalid_json_str, user_prompt_for_fake_system_prompt0
from src.prompter import apply_chat_template
from src.utils import get_list_or_str, read_popen_pipes, get_token_count, reverse_ucurve_list, undo_reverse_ucurve_list, \
    is_uuid4, has_starting_code_block, extract_code_block_content, looks_like_json, get_json, is_full_git_hash
from tests.utils import wrap_test_forked
import subprocess as sp


@wrap_test_forked
def test_get_list_or_str():
    assert get_list_or_str(['foo', 'bar']) == ['foo', 'bar']
    assert get_list_or_str('foo') == 'foo'
    assert get_list_or_str("['foo', 'bar']") == ['foo', 'bar']


@wrap_test_forked
def test_stream_popen1():
    cmd_python = sys.executable + " -i -q -u"
    cmd = cmd_python + " -c print('hi')"
    # cmd = cmd.split(' ')

    with sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, text=True, shell=True) as p:
        for out_line, err_line in read_popen_pipes(p):
            print(out_line, end='')
            print(err_line, end='')

        p.poll()


@wrap_test_forked
def test_stream_popen2():
    script = """for i in 0 1 2 3 4 5
do
    echo "This messages goes to stdout $i"
    sleep 1
    echo This message goes to stderr >&2
    sleep 1
done
"""
    with open('pieces.sh', 'wt') as f:
        f.write(script)
    with sp.Popen(["./pieces.sh"], stdout=sp.PIPE, stderr=sp.PIPE, text=True, shell=True) as p:
        for out_line, err_line in read_popen_pipes(p):
            print(out_line, end='')
            print(err_line, end='')
        p.poll()


@pytest.mark.parametrize("text_context_list",
                         ['text_context_list1', 'text_context_list2', 'text_context_list3', 'text_context_list4',
                          'text_context_list5', 'text_context_list6'])
@pytest.mark.parametrize("system_prompt", ['auto', ''])
@pytest.mark.parametrize("context", ['context1', 'context2'])
@pytest.mark.parametrize("iinput", ['iinput1', 'iinput2'])
@pytest.mark.parametrize("chat_conversation", ['chat_conversation1', 'chat_conversation2'])
@pytest.mark.parametrize("instruction", ['instruction1', 'instruction2'])
@wrap_test_forked
def test_limited_prompt(instruction, chat_conversation, iinput, context, system_prompt, text_context_list):
    instruction1 = 'Who are you?'
    instruction2 = ' '.join(['foo_%s ' % x for x in range(0, 500)])
    instruction = instruction1 if instruction == 'instruction1' else instruction2

    iinput1 = 'Extra instruction info'
    iinput2 = ' '.join(['iinput_%s ' % x for x in range(0, 500)])
    iinput = iinput1 if iinput == 'iinput1' else iinput2

    context1 = 'context'
    context2 = ' '.join(['context_%s ' % x for x in range(0, 500)])
    context = context1 if context == 'context1' else context2

    chat_conversation1 = []
    chat_conversation2 = [['user_conv_%s ' % x, 'bot_conv_%s ' % x] for x in range(0, 500)]
    chat_conversation = chat_conversation1 if chat_conversation == 'chat_conversation1' else chat_conversation2

    text_context_list1 = []
    text_context_list2 = ['doc_%s ' % x for x in range(0, 500)]
    text_context_list3 = ['doc_%s ' % x for x in range(0, 10)]
    text_context_list4 = ['documentmany_%s ' % x for x in range(0, 10000)]
    import random, string
    text_context_list5 = [
        'documentlong_%s_%s' % (x, ''.join(random.choices(string.ascii_letters + string.digits, k=300))) for x in
        range(0, 20)]
    text_context_list6 = [
        'documentlong_%s_%s' % (x, ''.join(random.choices(string.ascii_letters + string.digits, k=4000))) for x in
        range(0, 1)]
    if text_context_list == 'text_context_list1':
        text_context_list = text_context_list1
    elif text_context_list == 'text_context_list2':
        text_context_list = text_context_list2
    elif text_context_list == 'text_context_list3':
        text_context_list = text_context_list3
    elif text_context_list == 'text_context_list4':
        text_context_list = text_context_list4
    elif text_context_list == 'text_context_list5':
        text_context_list = text_context_list5
    elif text_context_list == 'text_context_list6':
        text_context_list = text_context_list6
    else:
        raise ValueError("No such %s" % text_context_list)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('h2oai/h2ogpt-4096-llama2-7b-chat')

    prompt_type = 'llama2'
    prompt_dict = None
    debug = False
    chat = True
    stream_output = True
    from src.prompter import Prompter
    prompter = Prompter(prompt_type, prompt_dict, debug=debug,
                        stream_output=stream_output,
                        system_prompt=system_prompt,
                        tokenizer=tokenizer)

    min_max_new_tokens = 512  # like in get_limited_prompt()
    max_input_tokens = -1
    max_new_tokens = 1024
    model_max_length = 4096

    from src.gen import get_limited_prompt
    estimated_full_prompt, \
        instruction, iinput, context, \
        num_prompt_tokens, max_new_tokens, \
        num_prompt_tokens0, num_prompt_tokens_actual, \
        history_to_use_final, external_handle_chat_conversation, \
        top_k_docs_trial, one_doc_size, truncation_generation, system_prompt = \
        get_limited_prompt(instruction, iinput, tokenizer,
                           prompter=prompter,
                           max_new_tokens=max_new_tokens,
                           context=context,
                           chat_conversation=chat_conversation,
                           text_context_list=text_context_list,
                           model_max_length=model_max_length,
                           min_max_new_tokens=min_max_new_tokens,
                           max_input_tokens=max_input_tokens,
                           verbose=True)
    print('%s -> %s or %s: len(history_to_use_final): %s top_k_docs_trial=%s one_doc_size: %s' % (num_prompt_tokens0,
                                                                                                  num_prompt_tokens,
                                                                                                  num_prompt_tokens_actual,
                                                                                                  len(history_to_use_final),
                                                                                                  top_k_docs_trial,
                                                                                                  one_doc_size),
          flush=True, file=sys.stderr)
    assert num_prompt_tokens <= model_max_length + min_max_new_tokens
    # actual might be less due to token merging for characters across parts, but not more
    assert num_prompt_tokens >= num_prompt_tokens_actual
    assert num_prompt_tokens_actual <= model_max_length

    if top_k_docs_trial > 0:
        text_context_list = text_context_list[:top_k_docs_trial]
    elif one_doc_size is not None:
        text_context_list = [text_context_list[0][:one_doc_size]]
    else:
        text_context_list = []
    assert sum([get_token_count(x, tokenizer) for x in text_context_list]) <= model_max_length


@wrap_test_forked
def test_reverse_ucurve():
    ab = []
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = [2, 4, 6, 8, 7, 5, 3, 1]
    ab.append([a, b])
    a = [1]
    b = [1]
    ab.append([a, b])
    a = [1, 2]
    b = [2, 1]
    ab.append([a, b])
    a = [1, 2, 3]
    b = [2, 3, 1]
    ab.append([a, b])
    a = [1, 2, 3, 4]
    b = [2, 4, 3, 1]
    ab.append([a, b])

    for a, b in ab:
        assert reverse_ucurve_list(a) == b
        assert undo_reverse_ucurve_list(b) == a


@wrap_test_forked
def check_gradio():
    import gradio as gr
    assert gr.__h2oai__


def test_is_uuid4():
    # Example usage:
    test_strings = [
        "f47ac10b-58cc-4372-a567-0e02b2c3d479",  # Valid UUID v4
        "not-a-uuid",  # Invalid
        "12345678-1234-1234-1234-123456789abc",  # Valid UUID v4
        "xyz"  # Invalid
    ]
    # "f47ac10b-58cc-4372-a567-0e02b2c3d479": True (Valid UUID v4)
    # "not-a-uuid": False (Invalid)
    # "12345678-1234-1234-1234-123456789abc": False (Invalid, even though it resembles a UUID, it doesn't follow the version 4 UUID pattern)
    # "xyz": False (Invalid)

    # Check each string and print whether it's a valid UUID v4
    assert [is_uuid4(s) for s in test_strings] == [True, False, False, False]


def test_is_git_hash():
    # Example usage:
    hashes = ["1a3b5c7d9e1a3b5c7d9e1a3b5c7d9e1a3b5c7d9e", "1G3b5c7d9e1a3b5c7d9e1a3b5c7d9e1a3b5c7d9e", "1a3b5c7d"]

    assert [is_full_git_hash(h) for h in hashes] == [True, False, False]


def test_chat_template():
    instruction = "Who are you?"
    system_prompt = "Be kind"
    history_to_use = [('Are you awesome?', "Yes I'm awesome.")]
    other_base_models = ['h2oai/mixtral-gm-rag-experimental-v2']
    supports_system_prompt = ['meta-llama/Llama-2-7b-chat-hf', 'openchat/openchat-3.5-1210', 'SeaLLMs/SeaLLM-7B-v2',
                              'h2oai/h2ogpt-gm-experimental']
    base_models = supports_system_prompt + other_base_models

    for base_model in base_models:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        prompt = apply_chat_template(instruction, system_prompt, history_to_use, tokenizer,
                                     user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt0,
                                     verbose=True)

        assert 'Be kind' in prompt  # put into pre-conversation if no actual system prompt
        assert instruction in prompt
        assert history_to_use[0][0] in prompt
        assert history_to_use[0][1] in prompt


def test_partial_codeblock():
    json.dumps(invalid_json_str)

    # Example usages:
    example_1 = "```code block starts immediately"
    example_2 = "\n    ```code block after newline and spaces"
    example_3 = "<br>```code block after HTML line break"
    example_4 = "This is a regular text without a code block."

    assert has_starting_code_block(example_1)
    assert has_starting_code_block(example_2)
    assert has_starting_code_block(example_3)
    assert not has_starting_code_block(example_4)

    # Example usages:
    example_stream_1 = "```code block content here```more text"
    example_stream_2 = "```code block content with no end yet..."
    example_stream_3 = "```\ncode block content here\n```\nmore text"
    example_stream_4 = "```\ncode block content \nwith no end yet..."
    example_stream_5 = "\n ```\ncode block content here\n```\nmore text"
    example_stream_6 = "\n ```\ncode block content \nwith no end yet..."

    example_stream_7 = "more text"

    assert extract_code_block_content(example_stream_1) == "block content here"
    assert extract_code_block_content(example_stream_2) == "block content with no end yet..."
    assert extract_code_block_content(example_stream_3) == "code block content here"
    assert extract_code_block_content(example_stream_4) == "code block content \nwith no end yet..."
    assert extract_code_block_content(example_stream_5) == "code block content here"
    assert extract_code_block_content(example_stream_6) == "code block content \nwith no end yet..."
    assert extract_code_block_content(example_stream_7) == ""

    # Assuming the function extract_code_block_content is defined as previously described.

    # Test case 1: Empty string
    assert extract_code_block_content("") is '', "Test 1 Failed: Should return None for empty string"

    # Test case 2: No starting code block
    assert extract_code_block_content(
        "No code block here") is '', "Test 2 Failed: Should return None if there's no starting code block"

    # Test case 3: Code block at the start without ending
    assert extract_code_block_content(
        "```text\nStarting without end") == "Starting without end", "Test 3 Failed: Should return the content of code block starting at the beginning"

    # Test case 4: Code block at the end without starting
    assert extract_code_block_content(
        "Text before code block```text\nEnding without start") == "Ending without start", "Test 4 Failed: Should extract text following starting delimiter regardless of position"

    # Test case 5: Code block in the middle with proper closing
    assert extract_code_block_content(
        "Text before ```text\ncode block``` text after") == "code block", "Test 5 Failed: Should extract the code block in the middle"

    # Test case 6: Multiple code blocks, only extracts the first one
    assert extract_code_block_content(
        "```text\nFirst code block``` Text in between ```Second code block```") == "First code block", "Test 6 Failed: Should only extract the first code block"

    # Test case 7: Code block with only whitespace inside
    assert extract_code_block_content(
        "```   ```") == "", "Test 7 Failed: Should return an empty string for a code block with only whitespace"

    # Test case 8: Newline characters inside code block
    assert extract_code_block_content(
        "```\nLine 1\nLine 2\n```") == "Line 1\nLine 2", "Test 8 Failed: Should preserve newline characters within code block but not leading/trailing newlines due to .strip()"

    # Test case 9: Code block with special characters
    special_characters = "```text\nSpecial characters !@#$%^&*()```"
    assert extract_code_block_content(
        special_characters) == "Special characters !@#$%^&*()", "Test 9 Failed: Should correctly handle special characters"

    # Test case 10: No starting code block but with ending delimiter
    assert extract_code_block_content(
        "Text with ending code block delimiter```") is '', "Test 10 Failed: Should return None if there's no starting code block but with an ending delimiter"

    # Test cases
    assert looks_like_json('{ "key": "value" }'), "Failed: JSON object"
    assert looks_like_json('[1, 2, 3]'), "Failed: JSON array"
    assert looks_like_json(' "string" '), "Failed: JSON string"
    assert looks_like_json('null'), "Failed: JSON null"
    assert looks_like_json(' true '), "Failed: JSON true"
    assert looks_like_json('123'), "Failed: JSON number"
    assert not looks_like_json('Just a plain text'), "Failed: Not JSON"
    assert not looks_like_json('```code block```'), "Failed: Code block"

    # Test cases
    get_json_nofixup = functools.partial(get_json, fixup=False)
    assert get_json_nofixup(
        '{"key": "value"}') == '{"key": "value"}', "Failed: Valid JSON object should be returned as is."
    assert get_json_nofixup('[1, 2, 3]') == '[1, 2, 3]', "Failed: Valid JSON array should be returned as is."
    assert get_json_nofixup('```text\nSome code```') == 'Some code', "Failed: Code block content should be returned."
    assert get_json_nofixup(
        'Some random text') == invalid_json_str, "Failed: Random text should lead to 'invalid json' return."
    assert get_json_nofixup(
        '```{"key": "value in code block"}```') == '{"key": "value in code block"}', "Failed: JSON in code block should be correctly extracted and returned."
    assert get_json_nofixup(
        '```code\nmore code```') == 'more code', "Failed: Multi-line code block content should be returned."
    assert get_json_nofixup(
        '```\n{"key": "value"}\n```') == '{"key": "value"}', "Failed: JSON object in code block with new lines should be correctly extracted and returned."
    assert get_json_nofixup('') == invalid_json_str, "Failed: Empty string should lead to 'invalid json' return."
    assert get_json_nofixup(
        'True') == invalid_json_str, "Failed: Non-JSON 'True' value should lead to 'invalid json' return."
    assert get_json_nofixup(
        '{"incomplete": true,') == '{"incomplete": true,', "Failed: Incomplete JSON should still be considered as JSON and returned as is."

    answer = """Here is an example JSON that fits the provided schema:
```json
{
  "name": "John Doe",
  "age": 30,
  "skills": ["Java", "Python", "JavaScript"],
  "work history": [
    {
      "company": "ABC Corp",
      "duration": "2018-2020",
      "position": "Software Engineer"
    },
    {
      "company": "XYZ Inc",
      "position": "Senior Software Engineer",
      "duration": "2020-Present"
    }
  ]
}
```
Note that the `work history` array contains two objects, each with a `company`, `duration`, and `position` property. The `skills` array contains three string elements, each with a maximum length of 10 characters. The `name` and `age` properties are also present and are of the correct data types."""
    assert get_json_nofixup(answer) == """{
  "name": "John Doe",
  "age": 30,
  "skills": ["Java", "Python", "JavaScript"],
  "work history": [
    {
      "company": "ABC Corp",
      "duration": "2018-2020",
      "position": "Software Engineer"
    },
    {
      "company": "XYZ Inc",
      "position": "Senior Software Engineer",
      "duration": "2020-Present"
    }
  ]
}"""

    # JSON within a code block
    json_in_code_block = """
    Here is an example JSON:
    ```json
    {"key": "value"}
    ```
    """

    # Plain JSON response
    plain_json_response = '{"key": "value"}'

    # Invalid JSON or non-JSON response
    non_json_response = "This is just some text."

    # Tests
    assert get_json_nofixup(
        json_in_code_block).strip() == '{"key": "value"}', "Should extract and return JSON from a code block."
    assert get_json_nofixup(plain_json_response) == '{"key": "value"}', "Should return plain JSON as is."
    assert get_json_nofixup(
        non_json_response) == invalid_json_str, "Should return 'invalid json' for non-JSON response."

    # Test with the provided example
    stream_content = """ {\n \"name\": \"John Doe\",\n \"email\": \"john.doe@example.com\",\n \"jobTitle\": \"Software Developer\",\n \"department\": \"Technology\",\n \"hireDate\": \"2020-01-01\",\n \"employeeId\": 123456,\n \"manager\": {\n \"name\": \"Jane Smith\",\n \"email\": \"jane.smith@example.com\",\n \"jobTitle\": \"Senior Software Developer\"\n },\n \"skills\": [\n \"Java\",\n \"Python\",\n \"JavaScript\",\n \"React\",\n \"Spring\"\n ],\n \"education\": {\n \"degree\": \"Bachelor's Degree\",\n \"field\": \"Computer Science\",\n \"institution\": \"Example University\",\n \"graduationYear\": 2018\n },\n \"awards\": [\n {\n \"awardName\": \"Best Developer of the Year\",\n \"year\": 2021\n },\n {\n \"awardName\": \"Most Valuable Team Player\",\n \"year\": 2020\n }\n ],\n \"performanceRatings\": {\n \"communication\": 4.5,\n \"teamwork\": 4.8,\n \"creativity\": 4.2,\n \"problem-solving\": 4.6,\n \"technical skills\": 4.7\n }\n}\n```"""
    extracted_content = get_json_nofixup(stream_content)
    assert extracted_content == """{\n \"name\": \"John Doe\",\n \"email\": \"john.doe@example.com\",\n \"jobTitle\": \"Software Developer\",\n \"department\": \"Technology\",\n \"hireDate\": \"2020-01-01\",\n \"employeeId\": 123456,\n \"manager\": {\n \"name\": \"Jane Smith\",\n \"email\": \"jane.smith@example.com\",\n \"jobTitle\": \"Senior Software Developer\"\n },\n \"skills\": [\n \"Java\",\n \"Python\",\n \"JavaScript\",\n \"React\",\n \"Spring\"\n ],\n \"education\": {\n \"degree\": \"Bachelor's Degree\",\n \"field\": \"Computer Science\",\n \"institution\": \"Example University\",\n \"graduationYear\": 2018\n },\n \"awards\": [\n {\n \"awardName\": \"Best Developer of the Year\",\n \"year\": 2021\n },\n {\n \"awardName\": \"Most Valuable Team Player\",\n \"year\": 2020\n }\n ],\n \"performanceRatings\": {\n \"communication\": 4.5,\n \"teamwork\": 4.8,\n \"creativity\": 4.2,\n \"problem-solving\": 4.6,\n \"technical skills\": 4.7\n }\n}"""


def test_repair_json():
    a = """{
    "Supplementary Leverage Ratio": [7.0, 5.8, 5.7],
    "Liquidity Metrics": {
    "End of Period Liabilities and Equity": [2260, 2362, 2291],
    "Liquidity Coverage Ratio": [118, 115, 115],
    "Trading-Related Liabilities(7)": [84, 72, 72],
    "Total Available Liquidty Resources": [972, 994, 961],
    "Deposits Balance Sheet": [140, 166, 164],
    "Other Liabilities(7)": {},
    "LTD": {},
    "Equity": {
    "Book Value per share": [86.43, 92.16, 92.21],
    "Tangible Book Value per share": [73.67, 79.07, 79.16]
    }
    },
    "Capital and Balance Sheet ($ in B)": {
    "Risk-based Capital Metrics(1)": {
    "End of Period Assets": [2260, 2362, 2291],
    "CET1 Capital": [147, 150, 150],
    "Standardized RWAs": [1222, 1284, 1224],
    "Investments, net": {},
    "CET1 Capital Ratio - Standardized": [12.1, 11.7, 12.2],
    "Advanced RWAs": [1255, 1265, 1212],
    "Trading-Related Assets(5)": [670, 681, 659],
    "CET1 Capital Ratio - Advanced": [11.7, 11.8, 12.4],
    "Loans, net(6)": {},
    "Other(5)": [182, 210, 206]
    }
    }
    }
    
    Note: Totals may not sum due to rounding. LTD: Long-term debt. All information for 4Q21 is preliminary. All footnotes are presented on Slide 26."""

    from json_repair import repair_json

    for i in range(len(a)):
        text = a[:i]
        t0 = time.time()
        good_json_string = repair_json(text)
        if i > 50:
            assert len(good_json_string) > 5
        tdelta = time.time() - t0
        assert tdelta < 0.005, "Too slow: %s" % tdelta
        print("%s : %s : %s" % (i, tdelta, good_json_string))
        json.loads(good_json_string)
