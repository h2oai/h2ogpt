import base64

from enums import unknown_prompt_type, template_prompt_type


def get_use_chat_template(tokenizer, prompt_type=None):
    if tokenizer is None:
        return False
    use_chat_template = prompt_type in [None, '', unknown_prompt_type, template_prompt_type] and \
                        has_chat_template(tokenizer)
    return use_chat_template


def has_chat_template(tokenizer):
    return (hasattr(tokenizer, 'chat_template') and
                         tokenizer.chat_template not in [None, ''] or
                         hasattr(tokenizer, 'default_chat_template') and
                         tokenizer.default_chat_template not in [None, '']
                         )


def get_chat_template(tokenizer):
    if tokenizer is None:
        return None
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template not in [None, '']:
        return tokenizer.chat_template
    if hasattr(tokenizer, 'default_chat_template') and tokenizer.default_chat_template not in [None, '']:
        return tokenizer.default_chat_template
    return None


def base64_encode_jinja_template(template_str):
    encoded_bytes = base64.b64encode(template_str.encode('utf-8'))
    encoded_str = encoded_bytes.decode('utf-8')
    return encoded_str


def base64_decode_jinja_template(encoded_str):
    if is_base64(encoded_str):
        decoded_bytes = base64.b64decode(encoded_str.encode('utf-8'))
        decoded_str = decoded_bytes.decode('utf-8')
        return decoded_str
    else:
        # just normal string, pass along
        return encoded_str


def is_base64(s):
    # Check if the length is a multiple of 4
    if len(s) % 4 != 0:
        return False

    # Check if the string contains only valid base64 characters
    try:
        # Try to decode the base64 string
        decoded = base64.b64decode(s, validate=True)
        # Check if the decoded bytes can be converted to a UTF-8 string
        decoded.decode('utf-8')
    except Exception:
        return False

    return True
