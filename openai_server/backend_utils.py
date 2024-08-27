import json
import os
import typing
import uuid
from collections import defaultdict


def concatenate_messages(messages, role='assistant', sep='\n'):
    """
    # Function to concatenate back-to-back assistant messages
    :param messages:
    :return:
    """
    concatenated_messages = []
    temp_message = ""
    for message in messages:
        if message['role'] == role:
            temp_message += message['content'] + sep
        else:
            if temp_message:
                concatenated_messages.append({"role": role, "content": temp_message})
                temp_message = ""
            concatenated_messages.append(message)
    if temp_message:
        concatenated_messages.append({"role": role, "content": temp_message})
    return concatenated_messages


def concat_tool_messages(messages):
    if not messages:
        return []

    final_messages = []
    current_user_message = None
    tool_contents = []

    for message in messages:
        if message['role'] == 'user':
            if current_user_message:
                if tool_contents:
                    tool_info = ''.join(f'# Tool result:\n{content}\n' for content in tool_contents)
                    current_user_message['content'] = f"{tool_info}{current_user_message['content']}"
                    tool_contents = []
                final_messages.append(current_user_message)
            current_user_message = message.copy()
        elif message['role'] == 'tool':
            tool_contents.append(message['content'])
        else:
            if current_user_message:
                if tool_contents:
                    tool_info = ''.join(f'# Tool result:\n{content}\n' for content in tool_contents)
                    current_user_message['content'] = f"{tool_info}{current_user_message['content']}"
                    tool_contents = []
                final_messages.append(current_user_message)
                current_user_message = None
            final_messages.append(message)

    # Handle case where the last message(s) are tool messages
    if tool_contents:
        if current_user_message:
            tool_info = ''.join(f'# Tool result:\n{content}\n' for content in tool_contents)
            current_user_message['content'] = f"{tool_info}{current_user_message['content']}"
            final_messages.append(current_user_message)
        else:
            # If there's no current user message, append to the last user message
            for i in range(len(final_messages) - 1, -1, -1):
                if final_messages[i]['role'] == 'user':
                    tool_info = ''.join(f'# Tool result:\n{content}\n' for content in tool_contents)
                    final_messages[i]['content'] = f"{tool_info}{final_messages[i]['content']}"
                    break
    elif current_user_message:
        final_messages.append(current_user_message)

    return final_messages


def convert_messages_to_structure(messages):
    """
    Convert a list of messages with roles and content into a structured format.

    Parameters:
    messages (list of dicts): A list where each dict contains 'role' and 'content' keys.

    Returns:
    tuple: A tuple containing the instruction, system_message, history, and image_files.
    """

    # messages = concatenate_messages(messages, tole='assistant')
    messages = concat_tool_messages(messages)
    # messages = concatenate_messages(messages, role='tool')

    structure = {
        "instruction": None,
        "system_message": None,
        "history": [],
        "image_files": []
    }

    if not messages:
        return structure['instruction'], structure['system_message'], structure['history'], structure['image_files']

    # Remove None messages
    messages = [x for x in messages if x.get("content")]

    # remove pure tool parts
    # assume just part of tool processing, "tool" role will have results, put that as user context
    messages = [x for x in messages if not x.get('tool_calls')]

    last_user_message = None
    previous_role = None
    for message in messages:
        role = message.get("role")
        assert role, "Missing role"
        content = message.get("content")
        assert content, "Missing content"

        if previous_role == role and role != 'tool':
            raise ValueError("Consecutive messages with the same role are not allowed: %s %s\n\n%s" % (
                previous_role, role, messages))
        previous_role = role

        if role in ["function", "tool"]:
            continue
        elif role == "system" and structure["system_message"] is None:
            structure["system_message"] = content
        elif role == "user":
            if last_user_message is not None:
                structure["history"].append((last_user_message, None))
            last_user_message = handle_content(content, structure)
        elif role == "assistant":
            if last_user_message:
                structure["history"].append((last_user_message, handle_content(content, structure)))
                last_user_message = None
            else:
                structure["history"].append((None, handle_content(content, structure)))

    # Set the instruction to the last user message if the last message is from the user,
    # and do not include it in the history.
    if messages and messages[-1]["role"] == "user":
        structure["instruction"] = last_user_message
    else:
        if last_user_message:  # If there was a dangling last user message, add it to history
            structure["history"].append((last_user_message, None))

    return structure['instruction'], structure['system_message'], structure['history'], structure['image_files']


def handle_content(content, structure):
    """
    Handle content which can be text, a dict, or a list of dicts.

    Parameters:
    content: The content to handle.
    structure: The structure to update with image URLs.

    Returns:
    str: The text content.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        if content['type'] == 'text':
            return content['text']
        elif content['type'] == 'image_url':
            structure['image_files'].append(content['image_url']['url'])
            return None
    elif isinstance(content, list):
        text_content = []
        for item in content:
            if item['type'] == 'text':
                text_content.append(item['text'])
            elif item['type'] == 'image_url':
                structure['image_files'].append(item['image_url']['url'])
        return "\n".join(text_content)


def structure_to_messages(instruction, system_message, history, image_files):
    """
    Convert an instruction, system message, history, and image files back into a list of messages.
    Parameters:
    instruction (str): The last instruction from the user, if any.
    system_message (str): The initial system message, if any.
    history (list of tuples): A list of tuples, each containing a pair of user and assistant messages.
    image_files (list): A list of image URLs to be included in the most recent user message.
    Returns:
    list of dicts: A list where each dict contains 'role' and 'content' keys.
    """
    messages = []
    if image_files is None:
        image_files = []

    # Add the system message first if it exists.
    if system_message:
        messages.append({"role": "system", "content": system_message})

    # Loop through the history to add user and assistant messages.
    for user_message, assistant_message in history:
        if user_message:
            messages.append({"role": "user", "content": user_message})
        if assistant_message:
            messages.append({"role": "assistant", "content": assistant_message})

    # Add the final instruction as a user message, if present.
    if instruction:
        final_user_message = {"role": "user", "content": instruction}
        if image_files:
            final_user_message["content"] = [{
                "type": "text",
                "text": instruction
            }] + [{"type": "image_url", "image_url": {"url": url}} for url in image_files]
        messages.append(final_user_message)
    elif image_files:
        # If no instruction but images exist, add images to the most recent user message
        if messages and messages[-1]["role"] == "user":
            final_user_message = messages[-1]
            if isinstance(final_user_message["content"], str):
                final_user_message["content"] = [{"type": "text", "text": final_user_message["content"]}]
            for image_url in image_files:
                final_user_message["content"].append({"type": "image_url", "image_url": {"url": image_url}})
        else:
            final_user_message = {"role": "user", "content": []}
            for image_url in image_files:
                final_user_message["content"].append({"type": "image_url", "image_url": {"url": image_url}})
            messages.append(final_user_message)

    return messages


def convert_gen_kwargs(gen_kwargs):
    # max_tokens=16 for text completion by default
    gen_kwargs['max_new_tokens'] = gen_kwargs.pop('max_new_tokens', gen_kwargs.pop('max_tokens', 256))
    gen_kwargs['visible_models'] = gen_kwargs.pop('visible_models', gen_kwargs.pop('model', 0))
    gen_kwargs['top_p'] = gen_kwargs.get('top_p', 1.0)
    gen_kwargs['top_k'] = gen_kwargs.get('top_k', 1)
    gen_kwargs['seed'] = gen_kwargs.get('seed', 0)

    if gen_kwargs.get('do_sample') in [False, None]:
        # be more like OpenAI, only temperature, not do_sample, to control
        gen_kwargs['temperature'] = gen_kwargs.pop('temperature', 0.0)  # unlike OpenAI, default to not random
    # https://platform.openai.com/docs/api-reference/chat/create
    if gen_kwargs['temperature'] > 0.0:
        # let temperature control sampling
        gen_kwargs['do_sample'] = True
    elif gen_kwargs['top_p'] != 1.0:
        # let top_p control sampling
        gen_kwargs['do_sample'] = True
        if gen_kwargs.get('top_k') == 1 and gen_kwargs.get('temperature') == 0.0:
            print("Sampling with top_k=1 has no effect if top_k=1 and temperature=0")
    else:
        # no sampling, make consistent
        gen_kwargs['top_p'] = 1.0
        gen_kwargs['top_k'] = 1
    if gen_kwargs['seed'] is None:
        gen_kwargs['seed'] = 0

    if gen_kwargs.get('repetition_penalty', 1) == 1 and gen_kwargs.get('presence_penalty', 0.0) != 0.0:
        # then user using presence_penalty, convert to repetition_penalty for h2oGPT
        # presence_penalty=(repetition_penalty - 1.0) * 2.0 + 0.0,  # so good default
        gen_kwargs['repetition_penalty'] = 0.5 * (gen_kwargs['presence_penalty'] - 0.0) + 1.0

    if gen_kwargs.get('response_format') and hasattr(gen_kwargs.get('response_format'), 'type'):
        # pydantic ensures type and key
        # transcribe to h2oGPT way of just value
        gen_kwargs['response_format'] = gen_kwargs.get('response_format').type

    return gen_kwargs


def get_user_dir(authorization):
    base_path = os.getenv('H2OGPT_OPENAI_BASE_FILE_PATH', './openai_files/')
    user_dir = os.path.join(base_path, authorization.split(" ")[1])
    return user_dir


meta_ext = '.____meta______'


def run_upload_api(content, filename, purpose, authorization):
    user_dir = get_user_dir(authorization)

    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    file_id = str(uuid.uuid4())
    file_path = os.path.join(user_dir, file_id)
    file_path_meta = os.path.join(user_dir, file_id + meta_ext)

    with open(file_path, "wb") as f:
        f.write(content)

    file_stat = os.stat(file_path)
    response_dict = dict(id=file_id,
                         object="file",
                         bytes=file_stat.st_size,
                         created_at=int(file_stat.st_ctime),
                         filename=filename,
                         purpose=purpose
                         )

    with open(file_path_meta, "wt") as f:
        f.write(json.dumps(response_dict))
    return response_dict


def get_last_and_return_value(gen):
    last_value = None
    return_value = None
    try:
        while True:
            last_value = next(gen)
    except StopIteration as e:
        return_value = e.value
    return last_value, return_value


import xml.etree.ElementTree as ET
import re


def extract_xml_tags(xml_string):
    # Remove leading/trailing whitespace and newlines
    xml_string = xml_string.strip()

    # If the string is empty, return a special empty marker
    if not xml_string:
        return "[[EMPTY]]"

    # If the string doesn't contain any XML tags, return an unparseable marker
    if '<' not in xml_string or '>' not in xml_string:
        return "[[UNPARSEABLE]]" + xml_string

    try:
        # Try to parse the XML string
        if xml_string.startswith('<doc>') and xml_string.endswith('</doc>'):
            root = ET.fromstring(xml_string)
        else:
            # If there's no <doc> tag, wrap the content in a temporary root
            root = ET.fromstring(f"<root>{xml_string}</root>")

        # Create a list to store the extracted tags
        extracted_tags = []

        # Extract all child elements except 'text'
        for child in root:
            if child.tag not in ['text', 'doc']:
                # Convert the element to a string and remove any internal newlines
                tag_string = ET.tostring(child, encoding='unicode').strip()
                tag_string = re.sub(r'\s*\n\s*', ' ', tag_string)
                extracted_tags.append(tag_string)

        # Join the extracted tags with a single newline
        result = '\n'.join(extracted_tags)

        # Ensure there's a newline at the end, but only one
        result = result.rstrip() + '\n'

        return result

    except ET.ParseError:
        # If parsing fails, return the unparseable marker with the original string
        return "[[UNPARSEABLE]]" + xml_string


def generate_unique_filename(xml_output):
    # Check for the special empty marker
    if xml_output == "[[EMPTY]]":
        unique_id = str(uuid.uuid4())
        return f"unknown_{unique_id}_page_0.txt", f"unknown_{unique_id}", "0"

    # Check for the special unparseable marker
    if xml_output.startswith("[[UNPARSEABLE]]"):
        unique_id = str(uuid.uuid4())
        return f"unparseable_{unique_id}_page_0.txt", f"unparseable_{unique_id}", "0"

    # If xml_output is empty (shouldn't happen, but just in case), generate a filename with UUID
    if not xml_output.strip():
        unique_id = str(uuid.uuid4())
        return f"unknown_{unique_id}_page_0.txt", f"unknown_{unique_id}", "0"

    try:
        # Try to parse the XML string
        root = ET.fromstring(f"<root>{xml_output}</root>")
    except ET.ParseError:
        # If parsing fails, generate a filename with UUID
        unique_id = str(uuid.uuid4())
        return f"unparseable_{unique_id}_page_0.txt", f"unparseable_{unique_id}", "0"

    # Extract name and page
    name_elem = root.find('name')
    page_elem = root.find('page')

    # Use UUID if name is missing, '0' if page is missing
    name = name_elem.text.strip() if name_elem is not None else str(uuid.uuid4())
    page = page_elem.text.strip() if page_elem is not None else '0'

    # Remove file extension if present
    name = os.path.splitext(name)[0]

    # Clean the name: remove any characters that aren't alphanumeric, underscore, or hyphen
    clean_name = re.sub(r'[^\w\-]', '_', name)

    # Create the unique filename
    unique_filename = f"{clean_name}_page_{page}.txt"

    return unique_filename, clean_name, page


def deduplicate_filenames(filenames):
    seen = defaultdict(int)
    result = []
    needs_renumbering = set()

    # First pass: identify duplicates and mark for renumbering
    for filename in filenames:
        if seen[filename] > 0:
            needs_renumbering.add(filename)
        seen[filename] += 1

    # Reset the seen counter for the second pass
    seen = defaultdict(int)

    # Second pass: rename files
    for filename in filenames:
        base, ext = filename.rsplit('.', 1)
        if filename in needs_renumbering:
            new_filename = f"{base}_chunk_{seen[filename]}.{ext}"
        else:
            new_filename = filename

        seen[filename] += 1
        result.append(new_filename)

    return result
