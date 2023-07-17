import collections
from typing import Any, Dict, OrderedDict

H2OGPT_PARAMETERS_TO_CLIENT = collections.OrderedDict(
    instruction="instruction",
    iinput="input",
    context="system_pre_context",
    stream_output="stream_output",
    prompt_type="prompt_type",
    prompt_dict="prompt_dict",
    temperature="temperature",
    top_p="top_p",
    top_k="top_k",
    num_beams="beams",
    max_new_tokens="max_output_length",
    min_new_tokens="min_output_length",
    early_stopping="early_stopping",
    max_time="max_time",
    repetition_penalty="repetition_penalty",
    num_return_sequences="number_returns",
    do_sample="enable_sampler",
    chat="chat",
    instruction_nochat="instruction_nochat",
    iinput_nochat="input_context_for_instruction",
    langchain_mode="langchain_mode",
    langchain_action="langchain_action",
    langchain_agents="langchain_agents",
    top_k_docs="langchain_top_k_docs",
    chunk="langchain_enable_chunk",
    chunk_size="langchain_chunk_size",
    document_subset="langchain_document_subset",
    document_choice="langchain_document_choice",
)


def to_h2ogpt_params(client_params: Dict[str, Any]) -> OrderedDict[str, Any]:
    """Convert given params to the order of params in h2oGPT."""

    h2ogpt_params: OrderedDict[str, Any] = H2OGPT_PARAMETERS_TO_CLIENT.copy()
    for h2ogpt_param_name, client_param_name in h2ogpt_params.items():
        if client_param_name in client_params:
            h2ogpt_params[h2ogpt_param_name] = client_params[client_param_name]
    return h2ogpt_params
