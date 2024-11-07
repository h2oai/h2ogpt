import json
import os
import re
import uuid
from collections import defaultdict


def concatenate_messages(messages, role="assistant", sep="\n"):
    """
    # Function to concatenate back-to-back assistant messages
    :param messages:
    :return:
    """
    concatenated_messages = []
    temp_message = ""
    for message in messages:
        if message["role"] == role:
            temp_message += message["content"] + sep
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
        if message["role"] == "user":
            if current_user_message:
                if tool_contents:
                    tool_info = "".join(
                        f"# Tool result:\n{content}\n" for content in tool_contents
                    )
                    current_user_message[
                        "content"
                    ] = f"{tool_info}{current_user_message['content']}"
                    tool_contents = []
                final_messages.append(current_user_message)
            current_user_message = message.copy()
        elif message["role"] == "tool":
            tool_contents.append(message["content"])
        else:
            if current_user_message:
                if tool_contents:
                    tool_info = "".join(
                        f"# Tool result:\n{content}\n" for content in tool_contents
                    )
                    current_user_message[
                        "content"
                    ] = f"{tool_info}{current_user_message['content']}"
                    tool_contents = []
                final_messages.append(current_user_message)
                current_user_message = None
            final_messages.append(message)

    # Handle case where the last message(s) are tool messages
    if tool_contents:
        if current_user_message:
            tool_info = "".join(
                f"# Tool result:\n{content}\n" for content in tool_contents
            )
            current_user_message[
                "content"
            ] = f"{tool_info}{current_user_message['content']}"
            final_messages.append(current_user_message)
        else:
            # If there's no current user message, append to the last user message
            for i in range(len(final_messages) - 1, -1, -1):
                if final_messages[i]["role"] == "user":
                    tool_info = "".join(
                        f"# Tool result:\n{content}\n" for content in tool_contents
                    )
                    final_messages[i][
                        "content"
                    ] = f"{tool_info}{final_messages[i]['content']}"
                    break
    elif current_user_message:
        final_messages.append(current_user_message)

    return final_messages


def convert_messages_to_structure(
        messages,
        concat_tool=True,
        concat_assistant=False,
        concat_user=False
):
    """
    Convert a list of messages with roles and content into a structured format.

    Parameters:
    messages (list of dicts): A list where each dict contains 'role' and 'content' keys.

    Returns:
    tuple: A tuple containing the instruction, system_message, history, and image_files.
    """

    if concat_assistant:
        messages = concatenate_messages(messages, role='assistant')
    if concat_user:
        messages = concatenate_messages(messages, role='user')
    if concat_tool:
        messages = concat_tool_messages(messages)

    structure = {
        "instruction": None,
        "system_message": None,
        "history": [],
        "image_files": [],
    }

    if not messages:
        return (
            structure["instruction"],
            structure["system_message"],
            structure["history"],
            structure["image_files"],
        )

    # Remove None messages
    messages = [x for x in messages if x.get("content")]

    # remove pure tool parts
    # assume just part of tool processing, "tool" role will have results, put that as user context
    messages = [x for x in messages if not x.get("tool_calls")]

    last_user_message = None
    previous_role = None
    for message in messages:
        role = message.get("role")
        assert role, "Missing role"
        content = message.get("content")
        assert content, "Missing content"

        if previous_role == role and role != "tool":
            print(f"bad messages: {messages}")
            raise ValueError(
                "Consecutive messages with the same role are not allowed: %s %s"
                % (previous_role, role)
            )
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
                structure["history"].append(
                    (last_user_message, handle_content(content, structure))
                )
                last_user_message = None
            else:
                structure["history"].append((None, handle_content(content, structure)))

    # Set the instruction to the last user message if the last message is from the user,
    # and do not include it in the history.
    if messages and messages[-1]["role"] == "user":
        structure["instruction"] = last_user_message
    else:
        if (
                last_user_message
        ):  # If there was a dangling last user message, add it to history
            structure["history"].append((last_user_message, None))

    return (
        structure["instruction"],
        structure["system_message"],
        structure["history"],
        structure["image_files"],
    )


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
        if content["type"] == "text":
            return content["text"]
        elif content["type"] == "image_url":
            structure["image_files"].append(content["image_url"]["url"])
            return None
    elif isinstance(content, list):
        text_content = []
        for item in content:
            if item["type"] == "text":
                text_content.append(item["text"])
            elif item["type"] == "image_url":
                structure["image_files"].append(item["image_url"]["url"])
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
    if history:
        for user_message, assistant_message in history:
            if user_message:
                messages.append({"role": "user", "content": user_message})
            if assistant_message:
                messages.append({"role": "assistant", "content": assistant_message})

    # Add the final instruction as a user message, if present.
    if instruction:
        final_user_message = {"role": "user", "content": instruction}
        if image_files:
            final_user_message["content"] = [{"type": "text", "text": instruction}] + [
                {"type": "image_url", "image_url": {"url": url}} for url in image_files
            ]
        messages.append(final_user_message)
    elif image_files:
        # If no instruction but images exist, add images to the most recent user message
        if messages and messages[-1]["role"] == "user":
            final_user_message = messages[-1]
            if isinstance(final_user_message["content"], str):
                final_user_message["content"] = [
                    {"type": "text", "text": final_user_message["content"]}
                ]
            for image_url in image_files:
                final_user_message["content"].append(
                    {"type": "image_url", "image_url": {"url": image_url}}
                )
        else:
            final_user_message = {"role": "user", "content": []}
            for image_url in image_files:
                final_user_message["content"].append(
                    {"type": "image_url", "image_url": {"url": image_url}}
                )
            messages.append(final_user_message)

    return messages


def convert_gen_kwargs(gen_kwargs):
    gen_kwargs.update(dict(instruction=gen_kwargs['query']))
    if os.getenv('GRADIO_H2OGPT_H2OGPT_KEY'):
        gen_kwargs.update(dict(h2ogpt_key=os.getenv('GRADIO_H2OGPT_H2OGPT_KEY')))

    # max_tokens=16 for text completion by default
    gen_kwargs["max_new_tokens"] = gen_kwargs.pop(
        "max_new_tokens", gen_kwargs.pop("max_tokens", 256)
    )
    gen_kwargs["visible_models"] = gen_kwargs.pop(
        "visible_models", gen_kwargs.pop("model", 0)
    )
    gen_kwargs["top_p"] = gen_kwargs.get("top_p", 1.0)
    gen_kwargs["top_k"] = gen_kwargs.get("top_k", 1)
    gen_kwargs["seed"] = gen_kwargs.get("seed", 0)

    if gen_kwargs.get("do_sample") in [False, None]:
        # be more like OpenAI, only temperature, not do_sample, to control
        gen_kwargs["temperature"] = gen_kwargs.pop(
            "temperature", 0.0
        )  # unlike OpenAI, default to not random
    # https://platform.openai.com/docs/api-reference/chat/create
    if gen_kwargs["temperature"] > 0.0:
        # let temperature control sampling
        gen_kwargs["do_sample"] = True
    elif gen_kwargs["top_p"] != 1.0:
        # let top_p control sampling
        gen_kwargs["do_sample"] = True
        if gen_kwargs.get("top_k") == 1 and gen_kwargs.get("temperature") == 0.0:
            print("Sampling with top_k=1 has no effect if top_k=1 and temperature=0")
    else:
        # no sampling, make consistent
        gen_kwargs["top_p"] = 1.0
        gen_kwargs["top_k"] = 1
    if gen_kwargs["seed"] is None:
        gen_kwargs["seed"] = 0

    if (
            gen_kwargs.get("repetition_penalty", 1) == 1
            and gen_kwargs.get("presence_penalty", 0.0) != 0.0
    ):
        # then user using presence_penalty, convert to repetition_penalty for h2oGPT
        # presence_penalty=(repetition_penalty - 1.0) * 2.0 + 0.0,  # so good default
        gen_kwargs["repetition_penalty"] = (
                0.5 * (gen_kwargs["presence_penalty"] - 0.0) + 1.0
        )

    if gen_kwargs.get("response_format") and hasattr(
            gen_kwargs.get("response_format"), "type"
    ):
        # pydantic ensures type and key
        # transcribe to h2oGPT way of just value
        gen_kwargs["response_format"] = gen_kwargs.get("response_format").type

    return gen_kwargs


def get_user_dir(authorization):
    base_path = os.getenv("H2OGPT_OPENAI_BASE_FILE_PATH", "./openai_files/")
    user_dir = os.path.join(base_path, authorization.split(" ")[1])
    return user_dir


meta_ext = ".____meta______"


def run_upload_api(content, filename, purpose, authorization, created_at_orig=None):
    user_dir = get_user_dir(authorization)

    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    file_id = str(uuid.uuid4())
    file_path = os.path.join(user_dir, file_id)
    file_path_meta = os.path.join(user_dir, file_id + meta_ext)

    with open(file_path, "wb") as f:
        f.write(content)

    file_stat = os.stat(file_path)
    response_dict = dict(
        id=file_id,
        object="file",
        bytes=file_stat.st_size,
        created_at=int(file_stat.st_ctime) if not created_at_orig else created_at_orig,
        filename=filename,
        purpose=purpose,
    )

    with open(file_path_meta, "wt") as f:
        f.write(json.dumps(response_dict))
    return response_dict


def run_download_api(file_id, authorization):
    user_dir = get_user_dir(authorization)

    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    file_path = os.path.join(user_dir, file_id)
    file_path_meta = os.path.join(user_dir, file_id + meta_ext)

    with open(file_path, "rb") as f:
        content = f.read()

    with open(file_path_meta, "rt") as f:
        response_dict = json.loads(f.read())
    assert isinstance(response_dict, dict), "response_dict should be a dict"
    return response_dict, content


def run_download_api_all(agent_files, authorization, agent_work_dir):
    for file_id in agent_files:
        response_dict, content = run_download_api(file_id, authorization)
        filename = response_dict['filename']
        new_file = os.path.join(agent_work_dir, filename)
        with open(new_file, "wb") as f:
            f.write(content)


def extract_xml_tags(full_text, tags=['name', 'page']):
    results_dict = {k: None for k in tags}
    for tag in tags:
        pattern = fr'<{tag}>(.*?)</{tag}>'
        values = re.findall(pattern, full_text, re.DOTALL)
        if values:
            results_dict[tag] = values[0]
    return results_dict


def generate_unique_filename(name_page_dict):
    name = name_page_dict.get('name', 'unknown') or 'unknown'
    page = name_page_dict.get('page', '0') or '0'

    # Remove file extension if present
    name = os.path.splitext(name)[0]

    # Clean the name: remove any characters that aren't alphanumeric, underscore, or hyphen
    clean_name = re.sub(r"[^\w\-]", "_", name)

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
        base, ext = filename.rsplit(".", 1)
        if filename in needs_renumbering:
            new_filename = f"{base}_chunk_{seen[filename]}.{ext}"
        else:
            new_filename = filename

        seen[filename] += 1
        result.append(new_filename)

    return result
