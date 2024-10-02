import re
import textwrap
from typing import List, Dict

markdown_mark = "---"


def chat_to_pretty_markdown(
        chat_history: List[Dict[str, str]],
        cute=False,
        assistant_name="Executor Agent",
        user_name="Coder Agent",
        dummy_name="Agent",
) -> str:
    markdown = ""
    for i, message in enumerate(chat_history):
        role = message["role"].capitalize()
        content = message["content"]

        if isinstance(content, list):
            # in case in image like structure
            content = "\n".join([x["text"] for x in content if x.get("type") == "text"])

        if not content or not content.strip():
            continue

        # Add a horizontal rule between messages (except before the first one)
        if i > 0:
            markdown += f"{markdown_mark}\n\n"

        # Add an emoji based on the role
        emoji = (
            "🧠"
            if role.lower() == "assistant"
            else "🤖"
            if role.lower() == "user"
            else "ℹ️"
        )
        real_role = (
            assistant_name
            if role.lower() == "assistant"
            else user_name
            if role.lower() == "user"
            else dummy_name
        )
        # If there is agent name mentioned, update the role and the emoji
        if 'name' in message and message['name']:
            # turns 'chat_agent' to 'Chat Agent'
            real_role = message['name']
            real_role = ' '.join(word.capitalize() for word in real_role.split('_'))

            if message['name'] == 'chat_agent':
                # put bubble emoji for chat agent
                emoji = "💬"
            if message['name'] == 'human_proxy_agent':
                # put human emoji for human proxy agent
                emoji = "👤"
            if message['name'] == 'code_writer_agent':
                # put code emoji for code writer agent
                emoji = "🤖"
            if message['name'] == 'code_executor_agent':
                # put code emoji for code executor agent
                emoji = "🧠"

        # Format the role
        if cute:
            markdown += f"### {emoji} {real_role}\n\n"
        else:
            markdown += f"### {real_role}\n\n"

        # Process the content
        lines = content.split("\n")
        in_code_block = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                markdown += line + "\n"
            elif in_code_block:
                # If we're in a code block, add the line as is
                markdown += line + "\n"
            else:
                # For non-code block content, wrap long lines
                wrapped_lines = wrap_long_lines(line)
                markdown += wrapped_lines + "\n"

        markdown += "\n"  # Add an extra newline for spacing between messages

    return markdown.strip()


def wrap_long_lines(line: str, max_width: int = 80) -> str:
    """Wrap long lines while preserving existing line breaks and indentation."""
    if len(line) <= max_width:
        return line

    words = line.split()
    wrapped_lines = []
    current_line = words[0]
    current_indent = len(line) - len(line.lstrip())
    indent = " " * current_indent

    for word in words[1:]:
        if len(current_line) + len(word) + 1 <= max_width:
            current_line += " " + word
        else:
            wrapped_lines.append(current_line)
            current_line = indent + word

    wrapped_lines.append(current_line)
    return "\n".join(wrapped_lines)


def chat_to_pretty_markdown_simple(
        chat_history,
        cute=False,
        assistant_name="Executor Agent",
        user_name="Coder Agent",
        dummy_name="Agent",
) -> str:
    # markdown = "# Chat History\n\n"
    markdown = ""
    for i, message in enumerate(chat_history):
        role = message["role"].capitalize()
        content = message["content"]

        if isinstance(content, list):
            # in case in image like structure
            content = "\n".join([x["text"] for x in content if x.get("type") == "text"])

        if not content or not content.strip():
            continue

        # Add a horizontal rule between messages (except before the first one)
        if i > 0:
            markdown += f"{markdown_mark}\n\n"

        # Add an emoji based on the role
        emoji = (
            "🧠"
            if role.lower() == "assistant"
            else "🤖"
            if role.lower() == "user"
            else "ℹ️"
        )
        real_role = (
            assistant_name
            if role.lower() == "assistant"
            else user_name
            if role.lower() == "user"
            else dummy_name
        )
        # If there is agent name mentioned, update the role and the emoji
        if 'name' in message and message['name']:
            # turns 'chat_agent' to 'Chat Agent'
            real_role = message['name']
            real_role = ' '.join(word.capitalize() for word in real_role.split('_'))

            if message['name'] == 'chat_agent':
                # put bubble emoji for chat agent
                emoji = "💬"
            if message['name'] == 'human_proxy_agent':
                # put human emoji for human proxy agent
                emoji = "👤"
            if message['name'] == 'code_writer_agent':
                # put code emoji for code writer agent
                emoji = "🤖"
            if message['name'] == 'code_executor_agent':
                # put code emoji for code executor agent
                emoji = "🧠"

        # Format the role
        if cute:
            markdown += f"### {emoji} {real_role}\n\n"
        else:
            markdown += f"### {real_role}\n\n"

        # Split content into code blocks and non-code blocks
        parts = re.split(r"(```[\s\S]*?```)", content)

        for part in parts:
            if part.startswith("```") and part.endswith("```"):
                # This is a code block, add it as-is
                markdown += part + "\n\n"
            else:
                # This is not a code block, wrap it
                wrapped_content = textwrap.wrap(part.strip(), width=80)
                markdown += "\n".join(wrapped_content) + "\n\n"

    return markdown.strip()
