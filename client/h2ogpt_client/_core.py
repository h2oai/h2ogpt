import ast
import asyncio
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, ValuesView, Union

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

    def __init__(self, src: str, huggingface_token: Optional[str] = None):
        """
        Creates a GPT client.
        :param src: either the full URL to the hosted h2oGPT
            (e.g. "http://0.0.0.0:7860", "https://fc752f297207f01c32.gradio.live")
            or name of the Hugging Face Space to load, (e.g. "h2oai/h2ogpt-chatbot")
        :param huggingface_token: Hugging Face token to use to access private Spaces
        """
        self._client = gradio_client.Client(
            src=src, hf_token=huggingface_token, serialize=False, verbose=False
        )
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
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 40,
        beams: float = 1.0,
        early_stopping: bool = False,
        min_output_length: int = 0,
        max_output_length: int = 128,
        max_time: int = 180,
        repetition_penalty: float = 1.07,
        number_returns: int = 1,
        system_pre_context: str = "",
        add_chat_history_to_context: bool = False,
        langchain_mode: LangChainMode = LangChainMode.DISABLED,
        system_prompt: str = "",
        visible_models: Union[str, list] = [],
        h2ogpt_key: str = None,
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
        :param visible_models: Single string of base model name, single integer of position of model, to get resopnse from
        :param h2ogpt_key: Key for access to API on keyed endpoints
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
        params["image_loaders"] = None
        params["pdf_loaders"] = None
        params["url_loaders"] = None
        params["jq_schema"] = None
        params["visible_models"] = visible_models
        params["h2ogpt_key"] = h2ogpt_key
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
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 40,
        beams: float = 1.0,
        early_stopping: bool = False,
        min_output_length: int = 0,
        max_output_length: int = 128,
        max_time: int = 180,
        repetition_penalty: float = 1.07,
        number_returns: int = 1,
        system_pre_context: str = "",
        langchain_mode: LangChainMode = LangChainMode.DISABLED,
        system_prompt: str = "",
        visible_models: Union[str, list] = [],
        h2ogpt_key: str = None,
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
        :param h2ogpt_key: Key for access to API on keyed endpoints
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
        params["image_loaders"] = None
        params["pdf_loaders"] = None
        params["url_loaders"] = None
        params["jq_schema"] = None
        params["visible_models"] = visible_models
        params["h2ogpt_key"] = h2ogpt_key
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
