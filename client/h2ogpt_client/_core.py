import ast
import asyncio
import typing
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union, ValuesView

import gradio_client  # type: ignore

from h2ogpt_client import _utils
from h2ogpt_client._h2ogpt_enums import (
    DocumentSubset,
    LangChainAction,
    LangChainMode,
    PromptType,
)


class Client:
    """h2oGPT Client."""

    def __init__(
        self,
        src: str,
        h2ogpt_key: Optional[str] = None,
        huggingface_token: Optional[str] = None,
    ):
        """
        Creates a GPT client.
        :param src: either the full URL to the hosted h2oGPT
            (e.g. "http://0.0.0.0:7860", "https://fc752f297207f01c32.gradio.live")
            or name of the Hugging Face Space to load, (e.g. "h2oai/h2ogpt-chatbot")
        :param h2ogpt_key: access key to connect with a h2oGPT server
        :param huggingface_token: Hugging Face token to use to access private Spaces
        """
        self._client = gradio_client.Client(
            src=src, hf_token=huggingface_token, serialize=False, verbose=False
        )
        self._h2ogpt_key = h2ogpt_key
        self._text_completion = TextCompletionCreator(self)
        self._chat_completion = ChatCompletionCreator(self)

    @property
    def text_completion(self) -> "TextCompletionCreator":
        """Text completion."""
        return self._text_completion

    @property
    def chat_completion(self) -> "ChatCompletionCreator":
        """Chat completion."""
        return self._chat_completion

    def _predict(self, *args, api_name: str) -> Any:
        return self._client.submit(*args, api_name=api_name).result()

    async def _predict_async(self, *args, api_name: str) -> Any:
        return await asyncio.wrap_future(self._client.submit(*args, api_name=api_name))


class TextCompletionCreator:
    """Builder that can create text completions."""

    def __init__(self, client: Client):
        self._client = client

    def create(
        self,
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
        add_chat_history_to_context: bool = False,
        langchain_mode: LangChainMode = LangChainMode.DISABLED,
        system_prompt: str = "",
        visible_models: Union[str, list] = [],
        add_search_to_context: bool = False,
        chat_conversation: typing.List[typing.Tuple[str, str]] = None,
        text_context_list: typing.List[str] = None,
        docs_ordering_type: str = None,
        min_max_new_tokens: int = None,
        max_input_tokens: int = None,
        docs_token_handling: str = None,
        docs_joiner: str = None,
    ) -> "TextCompletion":
        """
        Creates a new text completion.

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
        :param add_chat_history_to_context: Whether to add chat history to context
        :param system_prompt: Universal system prompt to override prompt_type's system
                              prompt
                              If pass 'None' or 'auto' or None, then automatic per-model value used
        :param visible_models: Single string of base model name, single integer of position of model, to get resopnse from
        :param add_search_to_context: Whether to add web search of query to context
        :param chat_conversation: list of tuples of (human, bot) form
        :param text_context_list: list of strings to use as context (up to allowed max_seq_len of model)
        :param docs_ordering_type: By default uses 'reverse_ucurve_sort' for optimal retrieval
        :param min_max_new_tokens: minimum value for max_new_tokens when auto-adjusting for content of prompt, docs, etc.
        :param max_input_tokens: Max input tokens to place into model context for each LLM call
                                 -1 means auto, fully fill context for query, and fill by original document chunk for summarization
                                 >=0 means use that to limit context filling to that many tokens
        :param docs_token_handling: 'chunk' means fill context with top_k_docs (limited by max_input_tokens or model_max_len) chunks for query
                                                                         or top_k_docs original document chunks summarization
                                    None or 'split_or_merge' means same as 'chunk' for query, while for summarization merges documents to fill up to max_input_tokens or model_max_len tokens
        :param docs_joiner: string to join lists of text when doing split_or_merge.  None means '\n\n'
        """
        params = _utils.to_h2ogpt_params(locals().copy())
        params["instruction"] = ""  # empty when chat_mode is False
        params["iinput"] = ""  # only chat_mode is True
        params["stream_output"] = False
        params["prompt_type"] = prompt_type.value  # convert to serializable type
        params["prompt_dict"] = ""  # empty as prompt_type cannot be 'custom'
        params["chat"] = False
        params["instruction_nochat"] = None  # future prompt
        params["langchain_mode"] = langchain_mode.value  # convert to serializable type
        params["add_chat_history_to_context"] = False  # relevant only for the UI
        params["langchain_action"] = LangChainAction.QUERY.value
        params["langchain_agents"] = []
        params["top_k_docs"] = 4  # langchain: number of document chunks
        params["chunk"] = True  # langchain: whether to chunk documents
        params["chunk_size"] = 512  # langchain: chunk size for document chunking
        params["document_subset"] = DocumentSubset.Relevant.name
        params["document_choice"] = []
        params["pre_prompt_query"] = ""
        params["prompt_query"] = ""
        params["pre_prompt_summary"] = ""
        params["prompt_summary"] = ""
        params["system_prompt"] = ""
        params["image_loaders"] = []
        params["pdf_loaders"] = []
        params["url_loaders"] = []
        params["jq_schema"] = '.[]'
        params["visible_models"] = visible_models
        params["h2ogpt_key"] = self._client._h2ogpt_key
        params["add_search_to_context"] = add_search_to_context
        params["chat_conversation"] = chat_conversation
        params["text_context_list"] = text_context_list
        params["docs_ordering_type"] = docs_ordering_type
        params["min_max_new_tokens"] = min_max_new_tokens
        params["max_input_tokens"] = max_input_tokens
        params["docs_token_handling"] = docs_token_handling
        params["docs_joiner"] = docs_joiner
        return TextCompletion(self._client, params)


class TextCompletion:
    """Text completion."""

    _API_NAME = "/submit_nochat_api"

    def __init__(self, client: Client, parameters: OrderedDict[str, Any]):
        self._client = client
        self._parameters = parameters

    def _get_parameters(self, prompt: str) -> OrderedDict[str, Any]:
        self._parameters["instruction_nochat"] = prompt
        return self._parameters

    @staticmethod
    def _get_reply(response: str) -> str:
        return ast.literal_eval(response)["response"]

    async def complete(self, prompt: str) -> str:
        """
        Complete this text completion.

        :param prompt: text prompt to generate completion for
        :return: response from the model
        """

        response = await self._client._predict_async(
            str(dict(self._get_parameters(prompt))), api_name=self._API_NAME
        )
        return self._get_reply(response)

    def complete_sync(self, prompt: str) -> str:
        """
        Complete this text completion synchronously.

        :param prompt: text prompt to generate completion for
        :return: response from the model
        """
        response = self._client._predict(
            str(dict(self._get_parameters(prompt))), api_name=self._API_NAME
        )
        return self._get_reply(response)


class ChatCompletionCreator:
    """Chat completion."""

    def __init__(self, client: Client):
        self._client = client

    def create(
        self,
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
        visible_models: Union[str, list] = [],
        add_search_to_context: bool= False,
        chat_conversation: typing.List[typing.Tuple[str, str]] = None,
        text_context_list: typing.List[str] = None,
        docs_ordering_type: str = None,
        min_max_new_tokens: int = None,
        max_input_tokens: int = None,
        docs_token_handling: str = None,
        docs_joiner: str = None,
    ) -> "ChatCompletion":
        """
        Creates a new chat completion.

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
        :param visible_models: Single string of base model name, single integer of position of model, to get resopnse from
        :param add_search_to_context: Whether to add web search of query to context
        :param chat_conversation: list of tuples of (human, bot) form
        :param text_context_list: list of strings to use as context (up to allowed max_seq_len of model)
        :param docs_ordering_type: By default uses 'reverse_ucurve_sort' for optimal retrieval
        :param min_max_new_tokens: minimum value for max_new_tokens when auto-adjusting for content of prompt, docs, etc.
        :param max_input_tokens: Max input tokens to place into model context for each LLM call
                                 -1 means auto, fully fill context for query, and fill by original document chunk for summarization
                                 >=0 means use that to limit context filling to that many tokens
        :param docs_token_handling: 'chunk' means fill context with top_k_docs (limited by max_input_tokens or model_max_len) chunks for query
                                                                         or top_k_docs original document chunks summarization
                                    None or 'split_or_merge' means same as 'chunk' for query, while for summarization merges documents to fill up to max_input_tokens or model_max_len tokens
        :param docs_joiner: string to join lists of text when doing split_or_merge.  None means '\n\n'
        """
        params = _utils.to_h2ogpt_params(locals().copy())
        params["instruction"] = None  # future prompts
        params["iinput"] = ""  # ??
        params["stream_output"] = False
        params["prompt_type"] = prompt_type.value  # convert to serializable type
        params["prompt_dict"] = ""  # empty as prompt_type cannot be 'custom'
        params["chat"] = True
        params["instruction_nochat"] = ""  # empty when chat_mode is True
        params["langchain_mode"] = langchain_mode.value  # convert to serializable type
        params["add_chat_history_to_context"] = False  # relevant only for the UI
        params["system_prompt"] = ""
        params["langchain_action"] = LangChainAction.QUERY.value
        params["langchain_agents"] = []
        params["top_k_docs"] = 4  # langchain: number of document chunks
        params["chunk"] = True  # langchain: whether to chunk documents
        params["chunk_size"] = 512  # langchain: chunk size for document chunking
        params["document_subset"] = DocumentSubset.Relevant.name
        params["document_choice"] = []
        params["pre_prompt_query"] = ""
        params["prompt_query"] = ""
        params["pre_prompt_summary"] = ""
        params["prompt_summary"] = ""
        params["system_prompt"] = ""
        params["image_loaders"] = []
        params["pdf_loaders"] = []
        params["url_loaders"] = []
        params["jq_schema"] = '.[]'
        params["visible_models"] = visible_models
        params["h2ogpt_key"] = self._client._h2ogpt_key
        params["add_search_to_context"] = add_search_to_context
        params["chat_conversation"] = chat_conversation
        params["text_context_list"] = text_context_list
        params["docs_ordering_type"] = docs_ordering_type
        params["min_max_new_tokens"] = min_max_new_tokens
        params["max_input_tokens"] = max_input_tokens
        params["docs_token_handling"] = docs_token_handling
        params["docs_joiner"] = docs_joiner
        params["chatbot"] = []  # chat history (FIXME: Only works if 1 model?)
        return ChatCompletion(self._client, params)


class ChatCompletion:
    """Chat completion."""

    _API_NAME = "/instruction_bot"

    def __init__(self, client: Client, parameters: OrderedDict[str, Any]):
        self._client = client
        self._parameters = parameters

    def _get_parameters(self, prompt: str) -> ValuesView:
        self._parameters["instruction"] = prompt
        self._parameters["chatbot"] += [[prompt, None]]
        return self._parameters.values()

    def _get_reply(self, response: Tuple[List[List[str]]]) -> Dict[str, str]:
        self._parameters["chatbot"][-1][1] = response[0][-1][1]
        return {"user": response[0][-1][0], "gpt": response[0][-1][1]}

    async def chat(self, prompt: str) -> Dict[str, str]:
        """
        Complete this chat completion.

        :param prompt: text prompt to generate completions for
        :returns chat reply
        """
        response = await self._client._predict_async(
            *self._get_parameters(prompt), api_name=self._API_NAME
        )
        return self._get_reply(response)

    def chat_sync(self, prompt: str) -> Dict[str, str]:
        """
        Complete this chat completion.

        :param prompt: text prompt to generate completions for
        :returns chat reply
        """
        response = self._client._predict(
            *self._get_parameters(prompt), api_name=self._API_NAME
        )
        return self._get_reply(response)

    def chat_history(self) -> List[Dict[str, str]]:
        """Returns the full chat history."""
        return [{"user": i[0], "gpt": i[1]} for i in self._parameters["chatbot"]]
