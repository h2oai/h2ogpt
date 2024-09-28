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
    assert_cache_control_count(result, 3)
    assert all("cache_control" in msg["content"][0] for msg in result[-3:] if msg["role"] == "user")
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
    assert_cache_control_count(result, 3)
    assert "cache_control" in result[-1]["content"][0]
    assert all("cache_control" in item for item in result[-2]["content"])
    assert "cache_control" not in result[0]["content"][0]


def test_max_cache_control_limit():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Item 1"}, {"type": "text", "text": "Item 2"}]},
        {"role": "user", "content": [{"type": "text", "text": "Item 3"}, {"type": "text", "text": "Item 4"}]},
        {"role": "user", "content": "Text message"},
    ]
    result = process_messages(messages)
    assert_cache_control_count(result, 3)
    assert "cache_control" in result[-1]["content"][0]
    assert "cache_control" in result[-2]["content"][1]
    assert "cache_control" in result[-2]["content"][0]
    assert "cache_control" not in result[0]["content"][0]


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
        {"role": "assistant", "content": "Response 1"},
        {"role": "user", "content": "Second"},
        {"role": "assistant", "content": "Response 2"},
        {"role": "user", "content": "Third"},
        {"role": "user", "content": "Fourth"},
    ]
    result = process_messages(messages)
    user_messages = [msg["content"] for msg in result if msg["role"] == "user"]
    assert user_messages == [
        [{"type": "text", "text": "First"}],
        [{"type": "text", "text": "Second", "cache_control": {"type": "ephemeral"}}],
        [{"type": "text", "text": "Third", "cache_control": {"type": "ephemeral"}}],
        [{"type": "text", "text": "Fourth", "cache_control": {"type": "ephemeral"}}],
    ]
    assert len(result) == 6  # Ensure all messages are preserved
    assert [msg["role"] for msg in result] == ["user", "assistant", "user", "assistant", "user",
                                               "user"]  # Ensure order is preserved


if __name__ == "__main__":
    pytest.main([__file__])
