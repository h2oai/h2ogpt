"""
Copied from https://github.com/lm-sys/FastChat.
Later we will contribute our changes into it.
"""
import dataclasses
from enum import auto, IntEnum
from typing import List, Any, Dict
import math
from typing import List, Optional, Tuple, Union
import random
import numpy as np

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    StoppingCriteriaList,
    MaxLengthCriteria,
    BitsAndBytesConfig,
)



class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: List[str] = (("USER", "ASSISTANT"),)
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()

def get_conversation_template(model_path: str) -> Conversation:
    """Get the default conversation template."""
    if "aquila-v1" in model_path:
        return get_conv_template("aquila-v1")
    elif "aquila-chat" in model_path:
        return get_conv_template("aquila-chat")
    elif "aquila-legacy" in model_path:
        return get_conv_template("aquila-legacy")
    else:
        return get_conv_template("aquila")

# AquilaChat default template
# source: https://github.com/FlagAI-Open/FlagAI/blob/master/examples/Aquila/Aquila-chat/cyg_conversation.py
register_conv_template(
    Conversation(
        name="aquila-chat",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant", "System"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="###",
        sep2="",
        stop_str=["###", "</s>", "[UNK]"],
    )
)

register_conv_template(
    Conversation(
        name="aquila-legacy",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        roles=("### Human: ", "### Assistant: ", "System"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_TWO,
        sep="\n",
        sep2="</s>",
        stop_str=["</s>", "[UNK]"],
    )
)

register_conv_template(
    Conversation(
        name="aquila",
        system_message="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant", "System"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="###",
        sep2="</s>",
        stop_str=["</s>", "[UNK]"],
    )
)

register_conv_template(
    Conversation(
        name="aquila-v1",
        roles=("<|startofpiece|>", "<|endofpiece|>", ""),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_TWO,
        sep="",
        sep2="</s>",
        stop_str=["</s>", "<|endoftext|>"],
    )
)


if __name__ == "__main__":
    print("aquila template:")
    conv = get_conv_template("aquila")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())

    print("\n")

    print("aquila-chat template:")
    conv = get_conv_template("aquila-chat")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())

    print("\n")

    print("aquila-v1 template:")
    conv = get_conv_template("aquila-v1")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())

    print("\n")

    print("aquila-legacy template:")
    conv = get_conv_template("aquila-legacy")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())

    print("\n")

def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def covert_prompt_to_input_ids_with_history(text, history, tokenizer, max_token, convo_template="aquila-chat"):
    # aquila-chat as default
    conv = get_conv_template(convo_template)

    conv.append_message(conv.roles[1], None)
    conv.append_message(conv.roles[0], text)

    example = tokenizer.encode_plus(f"{conv.get_prompt()} ", None, max_length=None)['input_ids']

    while(len(history) > 0 and (len(example) < max_token)):
        tmp = history.pop()
        if tmp[0] == 'ASSISTANT':
            conv.append_message(conv.roles[1], tmp[1])
        else:
            conv.append_message(conv.roles[0], tmp[1])
        example = tokenizer.encode_plus(f"{conv.get_prompt()} ", None, max_length=None)['input_ids']

    if len(example) >= max_token:
        conv.messages.pop()
    conv.messages = conv.messages[::-1]
    print('model in:', conv.get_prompt())
    example = tokenizer.encode_plus(f"{conv.get_prompt()} ", None, max_length=None)['input_ids']

    return example

def predict(model, text, tokenizer=None,
            max_gen_len=200, top_p=0.95,
            seed=1234, topk=100,
            temperature=0.9, 
            sft=True, convo_template = "",
            device = "cuda",
            model_name="AquilaChat2-7B",
            history=[],
            **kwargs):

    vocab = tokenizer.get_vocab()

    id2word = {v:k for k, v in vocab.items()}

    
    template_map = {"AquilaChat2-7B": "aquila-v1",
                    "AquilaChat2-34B": "aquila-legacy",
                    "AquilaChat2-7B-16K": "aquila",
                    "AquilaChat2-34B-16K": "aquila"}
    if not convo_template:
        convo_template=template_map.get(model_name, "aquila-chat")

    set_random_seed(seed)
    if temperature == 0:
        topk = 1
        temperature = 1.0
    if sft:
        tokens = covert_prompt_to_input_ids_with_history(text, history=history, tokenizer=tokenizer, max_token=1000000, convo_template=convo_template)
        tokens = torch.tensor(tokens)[None,].to(device)
    else :
        tokens = tokenizer.encode_plus(text)["input_ids"]
        print(tokenizer.decode(tokens))
        tokens = torch.tensor(tokens)[None,].to(device)
    input_length = len(tokens[0])
    with torch.no_grad():

        # instantiate logits processors
        logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(1, eos_token_id=100007),
            ]
        )
        # instantiate logits processors
        logits_warper = LogitsProcessorList(
            [
                TopPLogitsWarper(top_p),
                TopKLogitsWarper(topk),
                TemperatureLogitsWarper(temperature),
                
            ]
        )

        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=input_length + max_gen_len)])
        out = model.sample(
                            tokens,
                            logits_processor=logits_processor,
                            logits_warper=logits_warper,
                            stopping_criteria=stopping_criteria,
                            return_dict_in_generate=True, 
                            output_scores=True,
                        )

        
        # print(out)
        out_ids = out["sequences"][0][input_length:].cpu().numpy()

        out_scores = out["scores"]

        out_scores = torch.cat(out_scores, dim=0)
        out_scores = torch.nn.functional.softmax(out_scores, dim=-1).cpu().numpy()

        probs = []
        for i in range(len(out_ids)):
            probs.append(float(out_scores[i][out_ids[i]]))

        # print(f"probs is {probs}")

        convert_tokens = []
        for t in out_ids:
            if t == 100006:
                convert_tokens.append("[CLS]")
            else :
                convert_tokens.append(id2word.get(t, "[unkonwn_token]"))

        out_text = tokenizer.decode(out_ids.tolist())
        

        out = out_text

    if "[UNK]" in out:
        special_index = out.index("[UNK]")
        out = out[:special_index]
        token_length = len(tokenizer.encode_plus(out)["input_ids"])
        convert_tokens = convert_tokens[:token_length]
        probs = probs[:token_length]

    if "</s>" in out:
        special_index = out.index("</s>")
        out = out[: special_index]
        token_length = len(tokenizer.encode_plus(out)["input_ids"])
        convert_tokens = convert_tokens[:token_length]
        probs = probs[:token_length]

    if len(out) > 0 and out[0] == " ":
        out = out[1:]

        convert_tokens = convert_tokens[1:]
        probs = probs[1:]

    # Update history
    history.insert(0, ('ASSISTANT', out))
    history.insert(0, ('USER', text))

    return out 
