import abc
import ast
import collections
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    OrderedDict,
    Union,
)

from h2ogpt_client._gradio_client import GradioClientWrapper
from h2ogpt_client._h2ogpt_enums import (
    DocumentSubset,
    LangChainAction,
    LangChainMode,
    PromptType,
)
from h2ogpt_client._models import Model

_H2OGPT_PARAMETERS_TO_CLIENT = collections.OrderedDict(
    instruction="instruction",
    iinput="input",
    context="system_pre_context",
    stream_output="stream_output",
    prompt_type="prompt_type",
    prompt_dict="prompt_dict",
    temperature="temperature",
    top_p="top_p",
    top_k="top_k",
    penalty_alpha="penalty_alpha",
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
    add_chat_history_to_context="add_chat_history_to_context",
    langchain_action="langchain_action",
    langchain_agents="langchain_agents",
    top_k_docs="langchain_top_k_docs",
    chunk="langchain_enable_chunk",
    chunk_size="langchain_chunk_size",
    document_subset="langchain_document_subset",
    document_choice="langchain_document_choice",
    document_source_substrings="langchain_document_source_substrings",
    document_source_substrings_op="langchain_document_source_substrings_op",
    document_content_substrings="langchain_document_content_substrings",
    document_content_substrings_op="langchain_document_content_substrings_op",
    pre_prompt_query="pre_prompt_query",
    prompt_query="prompt_query",
    pre_prompt_summary="pre_prompt_summary",
    prompt_summary="prompt_summary",
    hyde_llm_prompt="hyde_llm_prompt",
    system_prompt="system_prompt",
    image_audio_loaders="image_audio_loaders",
    pdf_loaders="pdf_loaders",
    url_loaders="url_loaders",
    jq_schema="jq_schema",
    visible_models="model",
    h2ogpt_key="h2ogpt_key",
    add_search_to_context="add_search_to_context",
    chat_conversation="chat_conversation",
    text_context_list="text_context_list",
    docs_ordering_type="docs_ordering_type",
    min_max_new_tokens="min_max_new_tokens",
    max_input_tokens="max_input_tokens",
    max_total_input_tokens="max_total_input_tokens",
    docs_token_handling="docs_token_handling",
    docs_joiner="docs_joiner",
    hyde_level="hyde_level",
    hyde_template="hyde_template",
    hyde_show_only_final="hyde_show_only_final",
    doc_json_mode="doc_json_mode",
    chatbot_role="chatbot_role",
    speaker="speaker",
    tts_language="tts_language",
    tts_speed="tts_speed",
)


def _to_h2ogpt_params(client_params: Dict[str, Any]) -> OrderedDict[str, Any]:
    """Convert given params to the order of params in h2oGPT."""

    h2ogpt_params: OrderedDict[str, Any] = collections.OrderedDict()
    for h2ogpt_param_name, client_param_name in _H2OGPT_PARAMETERS_TO_CLIENT.items():
        if client_param_name in client_params:
            h2ogpt_params[h2ogpt_param_name] = client_params[client_param_name]
    return h2ogpt_params


_DEFAULT_PARAMETERS: Dict[str, Any] = dict(
    instruction="",
    input="",
    system_pre_context="",
    stream_output=False,
    prompt_type=PromptType.plain.value,
    prompt_dict="",  # empty as prompt_type cannot be 'custom'
    temperature=0.1,
    top_p=1.0,
    top_k=40,
    penalty_alpha=0.0,
    beams=1.0,
    max_output_length=1024,
    min_output_length=0,
    early_stopping=False,
    max_time=360,
    repetition_penalty=1.07,
    number_returns=1,
    enable_sampler=False,
    chat=False,
    instruction_nochat="",
    input_context_for_instruction="",
    langchain_mode=LangChainMode.DISABLED.value,
    add_chat_history_to_context=False,  # relevant only for the UI
    langchain_action=LangChainAction.QUERY.value,
    langchain_agents=[],
    langchain_top_k_docs=4,  # langchain: number of document chunks
    langchain_enable_chunk=True,  # langchain: whether to chunk documents
    langchain_chunk_size=512,  # langchain: chunk size for document chunking
    langchain_document_subset=DocumentSubset.Relevant.name,
    langchain_document_choice=[],
    langchain_document_source_substrings=[],
    langchain_document_source_substrings_op='and',
    langchain_document_content_substrings=[],
    langchain_document_content_substrings_op='and',
    pre_prompt_query=[],
    prompt_query="",
    pre_prompt_summary="",
    prompt_summary="",
    hyde_llm_prompt="",
    system_prompt="",
    image_audio_loaders=[],
    pdf_loaders=[],
    url_loaders=[],
    jq_schema=".[]",
    model=None,
    h2ogpt_key=None,
    add_search_to_context=False,
    chat_conversation=None,
    text_context_list=[],
    docs_ordering_type="reverse_ucurve_sort",
    min_max_new_tokens=512,
    max_input_tokens=-1,
    max_total_input_tokens=-1,
    docs_token_handling="split_or_merge",
    docs_joiner="\n\n",
    hyde_level=0,
    hyde_template=None,
    hyde_show_only_final=None,
    doc_json_mode=False,
    chatbot_role="None",
    speaker="None",
    tts_language="autodetect",
    tts_speed=1.0,
)


class _Completion(abc.ABC):
    _API_NAME = "/submit_nochat_api"

    def __init__(self, client: GradioClientWrapper, parameters: OrderedDict[str, Any]):
        self._client = client
        self._parameters = dict(parameters)

    def _get_parameters(self, prompt: str) -> Dict[str, Any]:
        self._parameters["instruction_nochat"] = prompt
        return self._parameters

    @staticmethod
    def _get_reply(response: str) -> str:
        return ast.literal_eval(response)["response"]

    def _predict(self, prompt: str) -> str:
        response = self._client.predict(
            str(self._get_parameters(prompt)), api_name=self._API_NAME
        )
        return self._get_reply(response)

    def _predict_and_stream(self, prompt: str) -> Generator[str, None, None]:
        generator = self._client.predict_and_stream(
            str(self._get_parameters(prompt)), api_name=self._API_NAME
        )
        reply_size_so_far = 0
        for response in generator:
            current_reply = self._get_reply(response)
            new_reply_chunk = current_reply[reply_size_so_far:]
            if not new_reply_chunk:
                continue
            reply_size_so_far += len(new_reply_chunk)
            yield new_reply_chunk

    async def _submit(self, prompt: str) -> str:
        response = await self._client.submit(
            str(self._get_parameters(prompt)), api_name=self._API_NAME
        )
        return self._get_reply(response)

    async def _submit_and_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        generator = self._client.submit_and_stream(
            str(self._get_parameters(prompt)), api_name=self._API_NAME
        )
        reply_size_so_far = 0
        async for response in generator:
            current_reply = self._get_reply(response)
            new_reply_chunk = current_reply[reply_size_so_far:]
            if not new_reply_chunk:
                continue
            reply_size_so_far += len(new_reply_chunk)
            yield new_reply_chunk


class TextCompletionCreator:
    """Builder that can create text completions."""

    def __init__(self, client: GradioClientWrapper):
        self._client = client

    def create(
        self,
        model: Union[None, Model, str] = None,
        prompt_type: PromptType = PromptType.plain,
        input_context_for_instruction: str = "",
        enable_sampler=False,
        temperature: float = 0.1,
        top_p: float = 1.0,
        top_k: int = 40,
        penalty_alpha: float = 0.0,
        beams: float = 1.0,
        early_stopping: bool = False,
        min_output_length: int = 0,
        max_output_length: int = 1024,
        max_time: int = 360,
        repetition_penalty: float = 1.07,
        number_returns: int = 1,
        system_pre_context: str = "",
        langchain_mode: LangChainMode = LangChainMode.DISABLED,
        system_prompt: str = "",
        add_search_to_context: bool = False,
        text_context_list: List[str] = [],
        docs_ordering_type: str = "reverse_ucurve_sort",
        min_max_new_tokens: int = 512,
        max_input_tokens: int = -1,
        max_total_input_tokens: int = -1,
        docs_token_handling: str = "split_or_merge",
        docs_joiner: str = "\n\n",
        hyde_level: int = 0,
        hyde_template: Optional[str] = None,
        hyde_show_only_final: bool = False,
        doc_json_mode: bool = False,
        chatbot_role="None",
        speaker="None",
        tts_language="autodetect",
        tts_speed=1.0,
    ) -> "TextCompletion":
        """
        Creates a new text completion.

        :param model: model to be used, `None` means used the default model.
        :param prompt_type: type of the prompt
        :param input_context_for_instruction: input context for instruction
        :param enable_sampler: enable or disable the sampler, required for use of
                temperature, top_p, top_k
        :param temperature: What sampling temperature to use, between 0 and 3.
                Lower values will make it more focused and deterministic, but may lead
                to repeat. Higher values will make the output more creative, but may
                lead to hallucinations.
        :param top_p: cumulative probability of tokens to sample from
        :param top_k: number of tokens to sample from
        :param penalty_alpha: >0 and top_k>1 enable contrastive search (not all models support)
        :param beams: Number of searches for optimal overall probability.
                Higher values uses more GPU memory and compute.
        :param early_stopping: whether to stop early or not in beam search
        :param min_output_length: minimum output length
        :param max_output_length: maximum output length
        :param max_time: maximum time to search optimal output
        :param repetition_penalty: penalty for repetition
        :param number_returns:
        :param system_pre_context: directly pre-appended without prompt processing
        :param langchain_mode: LangChain mode
        :param system_prompt: Universal system prompt to override prompt_type's system
                              prompt
                              If pass 'None' or 'auto' or None, then automatic per-model value used
        :param add_search_to_context: Whether to add web search of query to context
        :param text_context_list: list of strings to use as context (up to allowed max_seq_len of model)
        :param docs_ordering_type: By default uses 'reverse_ucurve_sort' for optimal retrieval
        :param min_max_new_tokens: minimum value for max_new_tokens when auto-adjusting for content of prompt, docs, etc.
        :param max_input_tokens: Max input tokens to place into model context for each LLM call
                                 -1 means auto, fully fill context for query, and fill by original document chunk for summarization
                                 >=0 means use that to limit context filling to that many tokens
        :param max_total_input_tokens: like max_input_tokens but instead of per LLM call, applies across all LLM calls for single summarization/extraction action
        :param docs_token_handling: 'chunk' means fill context with top_k_docs (limited by max_input_tokens or model_max_len) chunks for query
                                                                         or top_k_docs original document chunks summarization
                                    None or 'split_or_merge' means same as 'chunk' for query, while for summarization merges documents to fill up to max_input_tokens or model_max_len tokens
        :param docs_joiner: string to join lists of text when doing split_or_merge.  None means '\n\n'
        :param hyde_level: HYDE level for HYDE approach (https://arxiv.org/abs/2212.10496)
                     0: No HYDE
                     1: Use non-document-based LLM response and original query for embedding query
                     2: Use document-based LLM response and original query for embedding query
                     3+: Continue iterations of embedding prior answer and getting new response
        :param hyde_template:
                     None, 'None', 'auto' uses internal value and enable
                     '{query}' is minimal template one can pass
        :param hyde_show_only_final: See h2oGPT server docs
        :param doc_json_mode: whether to give JSON to LLM and get JSON response back
        :param chatbot_role: See h2oGPT server docs
        :param speaker: See h2oGPT server docs
        :param tts_language: See h2oGPT server docs
        :param tts_speed: See h2oGPT server docs
        """
        args = locals().copy()
        args["prompt_type"] = prompt_type.value  # convert to serializable type
        args["langchain_mode"] = langchain_mode.value  # convert to serializable type
        params = _to_h2ogpt_params({**_DEFAULT_PARAMETERS, **args})
        params["instruction_nochat"] = None  # future prompt
        params["h2ogpt_key"] = self._client.h2ogpt_key
        return TextCompletion(self._client, params)


class TextCompletion(_Completion):
    """Text completion."""

    async def complete(
        self, prompt: str, enable_streaming: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Complete this text completion.

        :param prompt: text prompt to generate completion for
        :param enable_streaming: whether to enable or disable streaming the response
        :return: response from the model
        """
        if enable_streaming:
            params = self._get_parameters(prompt)
            params["stream_output"] = True
            return self._submit_and_stream(prompt)
        else:
            return await self._submit(prompt)

    def complete_sync(
        self, prompt: str, enable_streaming: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Complete this text completion synchronously.

        :param prompt: text prompt to generate completion for
        :param enable_streaming: whether to enable or disable streaming the response
        :return: response from the model
        """
        if enable_streaming:
            params = self._get_parameters(prompt)
            params["stream_output"] = True
            return self._predict_and_stream(prompt)
        else:
            return self._predict(prompt)


class ChatCompletionCreator:
    """Chat completion."""

    def __init__(self, client: GradioClientWrapper):
        self._client = client

    def create(
        self,
        model: Union[None, Model, str] = None,
        prompt_type: PromptType = PromptType.plain,
        input_context_for_instruction: str = "",
        enable_sampler=False,
        temperature: float = 0.1,
        top_p: float = 1.0,
        top_k: int = 40,
        penalty_alpha: float = 0.0,
        beams: float = 1.0,
        early_stopping: bool = False,
        min_output_length: int = 0,
        max_output_length: int = 1024,
        max_time: int = 360,
        repetition_penalty: float = 1.07,
        number_returns: int = 1,
        system_pre_context: str = "",
        langchain_mode: LangChainMode = LangChainMode.DISABLED,
        system_prompt: str = "",
        add_search_to_context: bool = False,
        text_context_list: List[str] = [],
        docs_ordering_type: str = "reverse_ucurve_sort",
        min_max_new_tokens: int = 512,
        max_input_tokens: int = -1,
        max_total_input_tokens: int = -1,
        docs_token_handling: str = "split_or_merge",
        docs_joiner: str = "\n\n",
        hyde_level: int = 0,
        hyde_template: Optional[str] = None,
        hyde_show_only_final: bool = False,
        doc_json_mode: bool = False,
        chatbot_role="None",
        speaker="None",
        tts_language="autodetect",
        tts_speed=1.0,
    ) -> "ChatCompletion":
        """
        Creates a new chat completion.

        :param model: model to be used, `None` means used the default model.
        :param prompt_type: type of the prompt
        :param input_context_for_instruction: input context for instruction
        :param enable_sampler: enable or disable the sampler, required for use of
                temperature, top_p, top_k
        :param temperature: What sampling temperature to use, between 0 and 3.
                Lower values will make it more focused and deterministic, but may lead
                to repeat. Higher values will make the output more creative, but may
                lead to hallucinations.
        :param top_p: cumulative probability of tokens to sample from
        :param top_k: number of tokens to sample from
        :param penalty_alpha: >0 and top_k>1 enable contrastive search (not all models support)
        :param beams: Number of searches for optimal overall probability.
                Higher values uses more GPU memory and compute.
        :param early_stopping: whether to stop early or not in beam search
        :param min_output_length: minimum output length
        :param max_output_length: maximum output length
        :param max_time: maximum time to search optimal output
        :param repetition_penalty: penalty for repetition
        :param number_returns:
        :param system_pre_context: directly pre-appended without prompt processing
        :param langchain_mode: LangChain mode
        :param system_prompt: Universal system prompt to override prompt_type's system
                              prompt
        :param add_search_to_context: Whether to add web search of query to context
        :param text_context_list: list of strings to use as context (up to allowed max_seq_len of model)
        :param docs_ordering_type: By default uses 'reverse_ucurve_sort' for optimal retrieval
        :param min_max_new_tokens: minimum value for max_new_tokens when auto-adjusting for content of prompt, docs, etc.
        :param max_input_tokens: Max input tokens to place into model context for each LLM call
                                 -1 means auto, fully fill context for query, and fill by original document chunk for summarization
                                 >=0 means use that to limit context filling to that many tokens
        :param max_total_input_tokens: like max_input_tokens but instead of per LLM call, applies across all LLM calls for single summarization/extraction action
        :param docs_token_handling: 'chunk' means fill context with top_k_docs (limited by max_input_tokens or model_max_len) chunks for query
                                                                         or top_k_docs original document chunks summarization
                                    None or 'split_or_merge' means same as 'chunk' for query, while for summarization merges documents to fill up to max_input_tokens or model_max_len tokens
        :param docs_joiner: string to join lists of text when doing split_or_merge.  None means '\n\n'
        :param hyde_level: HYDE level for HYDE approach (https://arxiv.org/abs/2212.10496)
                     0: No HYDE
                     1: Use non-document-based LLM response and original query for embedding query
                     2: Use document-based LLM response and original query for embedding query
                     3+: Continue iterations of embedding prior answer and getting new response
        :param hyde_template:
                     None, 'None', 'auto' uses internal value and enable
                     '{query}' is minimal template one can pass
        :param hyde_show_only_final: See h2oGPT server docs
        :param doc_json_mode: whether to give JSON to LLM and get JSON response back
        :param chatbot_role: See h2oGPT server docs
        :param speaker: See h2oGPT server docs
        :param tts_language: See h2oGPT server docs
        :param tts_speed: See h2oGPT server docs
        """
        args = locals().copy()
        args["prompt_type"] = prompt_type.value  # convert to serializable type
        args["langchain_mode"] = langchain_mode.value  # convert to serializable type
        params = _to_h2ogpt_params({**_DEFAULT_PARAMETERS, **args})
        params["instruction_nochat"] = None  # future prompts
        params["add_chat_history_to_context"] = True
        params["h2ogpt_key"] = self._client.h2ogpt_key
        params["chat_conversation"] = []  # chat history (FIXME: Only works if 1 model?)
        return ChatCompletion(self._client, params)


class ChatCompletion(_Completion):
    """Chat completion."""

    def _update_history(self, prompt: str, reply: str) -> None:
        self._parameters["chat_conversation"].append((prompt, reply))

    async def chat(self, prompt: str) -> Dict[str, str]:
        """
        Complete this chat completion.

        :param prompt: text prompt to generate completions for
        :returns chat reply
        """
        reply = await self._submit(prompt)
        self._update_history(prompt, reply)
        return {"user": prompt, "gpt": reply}

    def chat_sync(self, prompt: str) -> Dict[str, str]:
        """
        Complete this chat completion.

        :param prompt: text prompt to generate completions for
        :returns chat reply
        """
        reply = self._predict(prompt)
        self._update_history(prompt, reply)
        return {"user": prompt, "gpt": reply}

    def chat_history(self) -> List[Dict[str, str]]:
        """Returns the full chat history."""
        return [
            {"user": i[0], "gpt": i[1]} for i in self._parameters["chat_conversation"]
        ]
