import ast
import asyncio
from typing import Any, Dict, List, Optional, OrderedDict, Union

import gradio_client  # type: ignore

from h2ogpt_client import _utils
from h2ogpt_client._h2ogpt_enums import (
    DocumentSubset,
    LangChainAction,
    LangChainMode,
    PromptType,
)
from h2ogpt_client._models import Model, Models


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
        self._models = Models(self)

    @property
    def text_completion(self) -> "TextCompletionCreator":
        """Text completion."""
        return self._text_completion

    @property
    def chat_completion(self) -> "ChatCompletionCreator":
        """Chat completion."""
        return self._chat_completion

    @property
    def models(self) -> "Models":
        """LL models"""
        return self._models

    def _predict(self, *args, api_name: str) -> Any:
        return self._client.submit(*args, api_name=api_name).result()

    async def _predict_async(self, *args, api_name: str) -> Any:
        return await asyncio.wrap_future(self._client.submit(*args, api_name=api_name))


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
    pre_prompt_query=[],
    prompt_query="",
    pre_prompt_summary="",
    prompt_summary="",
    system_prompt="",
    image_loaders=[],
    pdf_loaders=[],
    url_loaders=[],
    jq_schema=".[]",
    models=None,
    h2ogpt_key=None,
    add_search_to_context=False,
    chat_conversation=None,
    text_context_list=[],
    docs_ordering_type="reverse_ucurve_sort",
    min_max_new_tokens=256,
    max_input_tokens=-1,
    docs_token_handling="split_or_merge",
    docs_joiner="\n\n",
    hyde_level=0,
    hyde_template=None,
    doc_json_mode=False,
)


class TextCompletionCreator:
    """Builder that can create text completions."""

    def __init__(self, client: Client):
        self._client = client

    def create(
        self,
        models: Union[None, Model, str, List[Model], List[str]] = None,
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
        min_max_new_tokens: int = 256,
        max_input_tokens: int = -1,
        docs_token_handling: str = "split_or_merge",
        docs_joiner: str = "\n\n",
        hyde_level: int = 0,
        hyde_template: Optional[str] = None,
        doc_json_mode: bool = False,
    ) -> "TextCompletion":
        """
        Creates a new text completion.

        :param models: model(s) to be used, `None` means used the default model.
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
        :param doc_json_mode: whether to give JSON to LLM and get JSON response back
        """
        args = locals().copy()
        args["prompt_type"] = prompt_type.value  # convert to serializable type
        args["langchain_mode"] = langchain_mode.value  # convert to serializable type
        params = _utils.to_h2ogpt_params({**_DEFAULT_PARAMETERS, **args})
        params["instruction_nochat"] = None  # future prompt
        params["h2ogpt_key"] = self._client._h2ogpt_key
        return TextCompletion(self._client, params)


class TextCompletion:
    """Text completion."""

    _API_NAME = "/submit_nochat_api"

    def __init__(self, client: Client, parameters: OrderedDict[str, Any]):
        self._client = client
        self._parameters = dict(parameters)

    def _get_parameters(self, prompt: str) -> Dict[str, Any]:
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
            str(self._get_parameters(prompt)), api_name=self._API_NAME
        )
        return self._get_reply(response)

    def complete_sync(self, prompt: str) -> str:
        """
        Complete this text completion synchronously.

        :param prompt: text prompt to generate completion for
        :return: response from the model
        """
        response = self._client._predict(
            str(self._get_parameters(prompt)), api_name=self._API_NAME
        )
        return self._get_reply(response)


class ChatCompletionCreator:
    """Chat completion."""

    def __init__(self, client: Client):
        self._client = client

    def create(
        self,
        models: Union[None, Model, str, List[Model], List[str]] = None,
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
        min_max_new_tokens: int = 256,
        max_input_tokens: int = -1,
        docs_token_handling: str = "split_or_merge",
        docs_joiner: str = "\n\n",
        hyde_level: int = 0,
        hyde_template: Optional[str] = None,
        doc_json_mode: bool = False,
    ) -> "ChatCompletion":
        """
        Creates a new chat completion.

        :param models: model(s) to be used, `None` means used the default model.
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
        :param doc_json_mode: whether to give JSON to LLM and get JSON response back
        """
        args = locals().copy()
        args["prompt_type"] = prompt_type.value  # convert to serializable type
        args["langchain_mode"] = langchain_mode.value  # convert to serializable type
        params = _utils.to_h2ogpt_params({**_DEFAULT_PARAMETERS, **args})
        params["instruction_nochat"] = None  # future prompts
        params["add_chat_history_to_context"] = True
        params["h2ogpt_key"] = self._client._h2ogpt_key
        params["chat_conversation"] = []  # chat history (FIXME: Only works if 1 model?)
        return ChatCompletion(self._client, params)


class ChatCompletion:
    """Chat completion."""

    _API_NAME = "/submit_nochat_api"

    def __init__(self, client: Client, parameters: OrderedDict[str, Any]):
        self._client = client
        self._parameters = dict(parameters)

    def _get_parameters(self, prompt: str) -> Dict[str, Any]:
        self._parameters["instruction_nochat"] = prompt
        return self._parameters

    def _update_history_and_get_reply(
        self, prompt: str, response: str
    ) -> Dict[str, str]:
        reply = ast.literal_eval(response)["response"]
        self._parameters["chat_conversation"].append((prompt, reply))
        return {"user": prompt, "gpt": reply}

    async def chat(self, prompt: str) -> Dict[str, str]:
        """
        Complete this chat completion.

        :param prompt: text prompt to generate completions for
        :returns chat reply
        """
        response = await self._client._predict_async(
            str(self._get_parameters(prompt)), api_name=self._API_NAME
        )
        return self._update_history_and_get_reply(prompt, response)

    def chat_sync(self, prompt: str) -> Dict[str, str]:
        """
        Complete this chat completion.

        :param prompt: text prompt to generate completions for
        :returns chat reply
        """
        response = self._client._predict(
            str(self._get_parameters(prompt)), api_name=self._API_NAME
        )
        return self._update_history_and_get_reply(prompt, response)

    def chat_history(self) -> List[Dict[str, str]]:
        """Returns the full chat history."""
        return [
            {"user": i[0], "gpt": i[1]} for i in self._parameters["chat_conversation"]
        ]
