def convert_messages_to_structure(messages):
    """
    Convert a list of messages with roles and content into a structured format.

    Parameters:
    messages (list of dicts): A list where each dict contains 'role' and 'content' keys.

    Variables:
    structure: dict: A dictionary with 'instruction', 'system_message', and 'history' keys.

    Returns
    """
    structure = {
        "instruction": None,
        "system_message": None,
        "history": []
    }

    if not messages:
        return structure['instruction'], structure['system_message'], structure['history']

    last_user_message = None
    for message in messages:
        role = message.get("role")
        assert role, "Missing role"
        content = message.get("content")
        assert content, "Missing content"

        if role == "function":
            raise NotImplementedError("role: function not implemented")
        elif role == "system" and structure["system_message"] is None:
            structure["system_message"] = content
        elif role == "user":
            if last_user_message is not None:
                structure["history"].append((last_user_message, None))
            last_user_message = content
        elif role == "assistant":
            if last_user_message:
                structure["history"].append((last_user_message, content))
                last_user_message = None
            else:
                structure["history"].append((None, content))

    # Set the instruction to the last user message if the last message is from the user,
    # and do not include it in the history.
    if messages and messages[-1]["role"] == "user":
        structure["instruction"] = last_user_message
    else:
        if last_user_message:  # If there was a dangling last user message, add it to history
            structure["history"].append((last_user_message, None))

    return structure['instruction'], structure['system_message'], structure['history']


def structure_to_messages(instruction, system_message, history):
    """
    Convert an instruction, system message, and history back into a list of messages.
    Parameters:
    instruction (str): The last instruction from the user, if any.
    system_message (str): The initial system message, if any.
    history (list of tuples): A list of tuples, each containing a pair of user and assistant messages.
    Returns:
    list of dicts: A list where each dict contains 'role' and 'content' keys.
    """
    messages = []

    # Add the system message first if it exists.
    if system_message:
        messages.append({"role": "system", "content": system_message})

    # Loop through the history to add user and assistant messages.
    for message1 in history:

        if len(message1) != 2:
            continue
        if len(message1) == 2 and (message1[0] is None or message1[1] is None):
            # then not really part of LLM, internal, so avoid
            continue

        user_message, assistant_message = message1

        if user_message:
            messages.append({"role": "user", "content": user_message})
        if assistant_message:
            messages.append({"role": "assistant", "content": assistant_message})

    # Add the final instruction as a user message, if present.
    if instruction:
        messages.append({"role": "user", "content": instruction})

    return messages
