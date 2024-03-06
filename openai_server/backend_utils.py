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

    # Immediately return the default structure if there are no messages
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
            # The first system message is considered as the system message
            structure["system_message"] = content
        elif role == "user":
            if last_user_message is not None:
                # The last user message becomes part of the history if not followed by an assistant response
                structure["history"].append((last_user_message, None))
            last_user_message = content
        elif role == "assistant":
            if last_user_message:
                structure["history"].append((last_user_message, content))
                last_user_message = None
            else:
                # Handle case where there's an assistant message without a preceding user message
                structure["history"].append((None, content))

    # Check if there are any messages before accessing the last one
    if messages and messages[-1]["role"] == "assistant":
        structure["instruction"] = None
        if last_user_message:  # If there was a dangling last user message, add it to history
            structure["history"].append((last_user_message, None))
    elif last_user_message:
        structure["instruction"] = last_user_message

    return structure['instruction'], structure['system_message'], structure['history']
