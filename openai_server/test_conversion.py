import sys
sys.path.append('openai_server')
from openai_server.backend import convert_messages_to_structure


def test_conversion():
    # Example usage
    messages = [
        {"role": "user", "content": "How does the weather look today?"},
        {"role": "assistant", "content": "The weather is sunny and warm."},
        {"role": "user", "content": "What about tomorrow?"},
        {"role": "assistant", "content": "It's expected to rain tomorrow."}
    ]

    converted_structure = convert_messages_to_structure(messages)
    print(converted_structure)

