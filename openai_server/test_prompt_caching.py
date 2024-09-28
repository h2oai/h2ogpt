import sys

import pytest
from typing import List, Dict

if 'src' not in sys.path:
    sys.path.append('src')

from src.gpt_langchain import H2OChatAnthropic3

# Assume the process_messages function is imported from the module where it's defined

process_messages = H2OChatAnthropic3.process_messages


def assert_cache_control_count(messages: List[Dict], expected_count: int):
    actual_count = sum(
        1 for msg in messages if msg["role"] == "user"
        for item in (msg["content"] if isinstance(msg["content"], list) else [msg["content"]])
        if isinstance(item, dict) and "cache_control" in item
    )
    assert actual_count == expected_count, f"Expected {expected_count} cache_control entries, but found {actual_count}"


def test_simple_string_messages():
    messages = [
        {"role": "user", "content": "Message 1"},
        {"role": "assistant", "content": "Response 1"},
        {"role": "user", "content": "Message 2"},
        {"role": "user", "content": "Message 3"},
        {"role": "user", "content": "Message 4"},
        {"role": "user", "content": "Message 5"},
    ]
    result = process_messages(messages)
    assert len(result) == 6
    assert_cache_control_count(result, 4)
    assert all("cache_control" in msg["content"][0] for msg in result[-4:] if msg["role"] == "user")
    assert "cache_control" not in result[0]["content"][0]


def test_mixed_content_types():
    messages = [
        {"role": "user", "content": "Text message"},
        {"role": "assistant", "content": "Response"},
        {"role": "user",
         "content": [{"type": "text", "text": "List item 1"}, {"type": "image", "image_url": "example.com/image.jpg"}]},
        {"role": "user", "content": "Another text message"},
    ]
    result = process_messages(messages)
    assert len(result) == 4
    assert_cache_control_count(result, 4)
    assert "cache_control" in result[-1]["content"][0]
    assert all("cache_control" in item for item in result[-2]["content"])
    assert "cache_control" in result[-3]["content"][0]


def test_max_cache_control_limit():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Item 1"}, {"type": "text", "text": "Item 2"}]},
        {"role": "user", "content": [{"type": "text", "text": "Item 3"}, {"type": "text", "text": "Item 4"}]},
        {"role": "user", "content": "Text message"},
    ]
    result = process_messages(messages, max_cache_controls=4)
    assert_cache_control_count(result, 4)
    assert "cache_control" in result[-1]["content"][0]
    assert all("cache_control" in item for item in result[-2]["content"])
    assert "cache_control" in result[-3]["content"][1]
    assert "cache_control" not in result[-3]["content"][0]


def test_fewer_messages_than_max():
    messages = [
        {"role": "user", "content": "Message 1"},
        {"role": "assistant", "content": "Response 1"},
        {"role": "user", "content": "Message 2"},
    ]
    result = process_messages(messages, max_cache_controls=4)
    assert_cache_control_count(result, 2)
    assert all("cache_control" in msg["content"][0] for msg in result if msg["role"] == "user")


def test_non_dict_items_in_list():
    messages = [
        {"role": "user", "content": ["Text item", {"type": "text", "text": "Dict item"}]},
        {"role": "user", "content": [{"type": "text", "text": "Another dict item"}, 123]},
    ]
    result = process_messages(messages)
    assert_cache_control_count(result, 2)
    assert "cache_control" in result[-1]["content"][0]
    assert "cache_control" in result[-2]["content"][1]
    assert isinstance(result[-2]["content"][0], str)
    assert isinstance(result[-1]["content"][1], int)


def test_unexpected_content_type():
    messages = [
        {"role": "user", "content": 123},
        {"role": "user", "content": "Text message"},
    ]
    result = process_messages(messages)
    assert len(result) == 2
    assert result[0]["content"] == 123
    assert "cache_control" in result[1]["content"][0]


def test_empty_list_content():
    messages = [
        {"role": "user", "content": []},
        {"role": "user", "content": "Text message"},
    ]
    result = process_messages(messages)
    assert len(result) == 2
    assert result[0]["content"] == []
    assert "cache_control" in result[1]["content"][0]


def test_preserve_message_order():
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Response"},
        {"role": "user", "content": "Second"},
        {"role": "user", "content": "Third"},
    ]
    result = process_messages(messages)
    assert [msg["content"] for msg in result if msg["role"] == "user"] == [
        [{"type": "text", "text": "First"}],
        [{"type": "text", "text": "Second", "cache_control": {"type": "ephemeral"}}],
        [{"type": "text", "text": "Third", "cache_control": {"type": "ephemeral"}}],
    ]


if __name__ == "__main__":
    pytest.main([__file__])
