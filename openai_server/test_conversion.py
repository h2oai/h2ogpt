import sys

sys.path.append('openai_server')
from openai_server.backend_utils import convert_messages_to_structure, structure_to_messages


def test_conversion():
    # Example usage
    messages = [
        {"role": "user", "content": "How does the weather look today?"},
        {"role": "assistant", "content": "The weather is sunny and warm."},
        {"role": "user", "content": "What about tomorrow?"},
        {"role": "assistant", "content": "It's expected to rain tomorrow."}
    ]

    converted_structure = convert_messages_to_structure(messages)
    assert converted_structure == (None, None, [('How does the weather look today?', 'The weather is sunny and warm.'),
                                                ('What about tomorrow?', "It's expected to rain tomorrow.")])

    messages = [{'role': 'user', 'content': 'What is your name?'},
                {'role': 'assistant', 'content': 'My name is Bob.'},
                {'role': 'user', 'content': 'What did I just ask?'},
                ]
    converted_structure = convert_messages_to_structure(messages)
    assert converted_structure == ('What did I just ask?', None, [('What is your name?', 'My name is Bob.')])

    messages = []
    converted_structure = convert_messages_to_structure(messages)
    assert converted_structure == (None, None, [])

    system_prompt = """You are a helpful assistant and have been created by H2O.ai. Never ever forget that your name is Liam Chen. 
    You are having a conversation with a user.\nThe user's name is Asghar. So you are talking to Asghar. 
    Keep your responses in short length to retain the person's attention. 
    If the conversation history is empty, start the conversation with just a greeting and inquire about how the person is doing.
    After the initial greeting, do not greet again, just focus on answering the user's questions directly.
    Don't say things like "I'm a computer program" or "I don't have feelings or experiences." I know that.
    """

    messages = [{"role": "system", "content": system_prompt},
                {"role": "assistant", "content": "Hello Asghar, how are you doing today?"},
                {"role": "user", "content": "who are you?"}
                ]
    converted_structure = convert_messages_to_structure(messages)
    assert converted_structure == ('who are you?',
                                   'You are a helpful assistant and have been created by H2O.ai. Never ever '
                                   'forget that your name is Liam Chen. \n'
                                   '    You are having a conversation with a user.\n'
                                   "The user's name is Asghar. So you are talking to Asghar. \n"
                                   "    Keep your responses in short length to retain the person's attention. \n"
                                   '    If the conversation history is empty, start the conversation with just a '
                                   'greeting and inquire about how the person is doing.\n'
                                   '    After the initial greeting, do not greet again, just focus on answering '
                                   "the user's questions directly.\n"
                                   '    Don\'t say things like "I\'m a computer program" or "I don\'t have '
                                   'feelings or experiences." I know that.\n'
                                   '    ',
                                   [(None, 'Hello Asghar, how are you doing today?')])

    messages = [{"role": "system", "content": system_prompt},
                {"role": "assistant", "content": "Hello Asghar, how are you doing today?"},
                {"role": "user", "content": "what is the sum of 4 plus 4?"},
                {"role": "assistant", "content": "The sum of 4+4 is 8."},
                {"role": "user", "content": "who are you?"}
                ]
    converted_structure = convert_messages_to_structure(messages)
    assert converted_structure == ('who are you?',
                                   'You are a helpful assistant and have been created by H2O.ai. Never ever '
                                   'forget that your name is Liam Chen. \n'
                                   '    You are having a conversation with a user.\n'
                                   "The user's name is Asghar. So you are talking to Asghar. \n"
                                   "    Keep your responses in short length to retain the person's attention. \n"
                                   '    If the conversation history is empty, start the conversation with just a '
                                   'greeting and inquire about how the person is doing.\n'
                                   '    After the initial greeting, do not greet again, just focus on answering '
                                   "the user's questions directly.\n"
                                   '    Don\'t say things like "I\'m a computer program" or "I don\'t have '
                                   'feelings or experiences." I know that.\n'
                                   '    ',
                                   [(None, 'Hello Asghar, how are you doing today?'),
                                    ('what is the sum of 4 plus 4?', 'The sum of 4+4 is 8.')])


def test_conversion2():
    # Basic conversion test
    messages = [
        {"role": "user", "content": "How does the weather look today?"},
        {"role": "assistant", "content": "The weather is sunny and warm."},
        {"role": "user", "content": "What about tomorrow?"},
        {"role": "assistant", "content": "It's expected to rain tomorrow."}
    ]
    converted_structure = convert_messages_to_structure(messages)
    assert converted_structure == (None, None,
                                   [('How does the weather look today?', 'The weather is sunny and warm.'),
                                    ('What about tomorrow?', "It's expected to rain tomorrow.")])

    # User asks a question after an initial Q&A
    messages = [
        {'role': 'user', 'content': 'What is your name?'},
        {'role': 'assistant', 'content': 'My name is Bob.'},
        {'role': 'user', 'content': 'What did I just ask?'},
    ]
    converted_structure = convert_messages_to_structure(messages)
    assert converted_structure == ('What did I just ask?', None, [('What is your name?', 'My name is Bob.')])

    # Empty messages list
    messages = []
    converted_structure = convert_messages_to_structure(messages)
    assert converted_structure == (None, None, [])

    # Only user messages
    messages = [{'role': 'user', 'content': 'Is it going to rain today?'}]
    converted_structure = convert_messages_to_structure(messages)
    assert converted_structure == ('Is it going to rain today?', None, [])

    # Only assistant messages
    messages = [{'role': 'assistant', 'content': 'Welcome to our service.'}]
    converted_structure = convert_messages_to_structure(messages)
    assert converted_structure == (None, None, [(None, 'Welcome to our service.')])

    # Starting with an assistant message
    messages = [
        {'role': 'assistant', 'content': 'First message from assistant.'},
        {'role': 'user', 'content': 'How can I help you?'}
    ]
    converted_structure = convert_messages_to_structure(messages)
    assert converted_structure == ('How can I help you?', None, [(None, 'First message from assistant.')])

    # Including a system message
    messages = [
        {'role': 'system', 'content': 'System initialization complete.'},
        {'role': 'user', 'content': 'What is the system status?'},
        {'role': 'assistant', 'content': 'System is operational.'}
    ]
    converted_structure = convert_messages_to_structure(messages)
    assert converted_structure == (
        None, 'System initialization complete.', [('What is the system status?', 'System is operational.')])

    # Mixed roles with no user message before an assistant message
    messages = [
        {'role': 'assistant', 'content': 'Unprompted advice.'},
        {'role': 'user', 'content': 'Thanks for the advice.'}
    ]
    converted_structure = convert_messages_to_structure(messages)
    assert converted_structure == ('Thanks for the advice.', None, [(None, 'Unprompted advice.')])

    # A longer conversation
    messages = [
        {'role': 'user', 'content': 'What time is it?'},
        {'role': 'assistant', 'content': 'It is 10 AM.'},
        {'role': 'user', 'content': 'Set an alarm for 11 AM.'},
        {'role': 'assistant', 'content': 'Alarm set for 11 AM.'},
        {'role': 'user', 'content': 'Cancel the alarm.'},
        {'role': 'assistant', 'content': 'Alarm canceled.'}
    ]
    converted_structure = convert_messages_to_structure(messages)
    assert converted_structure == (None, None, [
        ('What time is it?', 'It is 10 AM.'),
        ('Set an alarm for 11 AM.', 'Alarm set for 11 AM.'),
        ('Cancel the alarm.', 'Alarm canceled.')
    ])


def test_structure_to_messages():
    # First example
    messages_1 = [
        {"role": "user", "content": "How does the weather look today?"},
        {"role": "assistant", "content": "The weather is sunny and warm."},
        {"role": "user", "content": "What about tomorrow?"},
        {"role": "assistant", "content": "It's expected to rain tomorrow."}
    ]
    instruction_1, system_message_1, history_1 = convert_messages_to_structure(messages_1)
    reconstructed_messages_1 = structure_to_messages(instruction_1, system_message_1, history_1)
    assert reconstructed_messages_1 == messages_1

    # Second example
    messages_2 = [
        {"role": "user", "content": "What is your name?"},
        {"role": "assistant", "content": "My name is Bob."},
        {"role": "user", "content": "What did I just ask?"}
    ]
    instruction_2, system_message_2, history_2 = convert_messages_to_structure(messages_2)
    reconstructed_messages_2 = structure_to_messages(instruction_2, system_message_2, history_2)
    # Adjust for the last user message being moved to instruction
    messages_2[-1] = {"role": "user", "content": "What did I just ask?"}
    assert reconstructed_messages_2 == messages_2

    # Third example: empty messages
    messages_3 = []
    instruction_3, system_message_3, history_3 = convert_messages_to_structure(messages_3)
    reconstructed_messages_3 = structure_to_messages(instruction_3, system_message_3, history_3)
    assert reconstructed_messages_3 == messages_3

    # Fourth and fifth examples involve a system message, which is not directly handled in the same way by
    # the `structure_to_messages` function since it assumes the system message is part of the structure already.
    # You would need to ensure the system message is appropriately handled within the `structure_to_messages`
    # function or manually insert it into the test conditions here, depending on your implementation details.

    print("All tests passed.")


def test_structure_to_messages_with_system_message():
    # Setup example with a system message
    system_prompt = "System message content."
    messages_with_system = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 is 4."}
    ]

    instruction, system_message, history = convert_messages_to_structure(messages_with_system)
    reconstructed_messages = structure_to_messages(instruction, system_message, history)

    assert reconstructed_messages == messages_with_system, "Test with system message failed."

    print("All tests passed including those with a system message.")
