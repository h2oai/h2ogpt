from src.enums import unknown_prompt_type, template_prompt_type


def get_use_chat_template(tokenizer, prompt_type=None):
    if tokenizer is None:
        return False
    use_chat_template = prompt_type in [None, '', unknown_prompt_type, template_prompt_type] and \
                        (hasattr(tokenizer, 'chat_template') and
                         tokenizer.chat_template not in [None, ''] or
                         hasattr(tokenizer, 'default_chat_template') and
                         tokenizer.default_chat_template not in [None, '']
                         )
    return use_chat_template
