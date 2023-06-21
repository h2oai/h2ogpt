import asyncio
import collections
from typing import Any, Dict, List, Optional, OrderedDict, Tuple

import gradio_client  # type: ignore

from h2ogpt_client import enums


class Client:
    def __init__(self, server_url: str, huggingface_token: Optional[str] = None):
        self._client = gradio_client.Client(
            src=server_url, hf_token=huggingface_token, serialize=False, verbose=False
        )
        self._text_completion = TextCompletion(self)
        self._chat_completion = ChatCompletion(self)

    @property
    def text_completion(self) -> "TextCompletion":
        return self._text_completion

    @property
    def chat_completion(self) -> "ChatCompletion":
        return self._chat_completion

    def _predict(self, *args, api_name: str) -> Any:
        return self._client.submit(*args, api_name=api_name).result()

    async def _predict_async(self, *args, api_name: str) -> str:
        return await asyncio.wrap_future(self._client.submit(*args, api_name=api_name))


class TextCompletion:
    """Text completion"""

    def __init__(self, client: Client):
        self._client = client

    def create(
        self,
        prompt: str,
        prompt_type: enums.PromptType = enums.PromptType.plain,
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
        langchain_mode: enums.LangChainMode = enums.LangChainMode.DISABLED,
    ) -> str:
        """
        Creates a new text completion.

        :param prompt: text prompt to generate completions for
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
        :return: response from the model
        """
        # Not exposed parameters.
        instruction = ""  # empty when chat_mode is False
        input = ""  # only chat_mode is True
        stream_output = False
        prompt_dict = ""  # empty as prompt_type cannot be 'custom'
        chat_mode = False
        langchain_top_k_docs = 4  # number of document chunks; not public
        langchain_enable_chunk = True  # whether to chunk documents; not public
        langchain_chunk_size = 512  # chunk size for document chunking; not public
        langchain_document_choice = ["All"]

        return self._client._predict(
            instruction,
            input,
            system_pre_context,
            stream_output,
            prompt_type.value,
            prompt_dict,
            temperature,
            top_p,
            top_k,
            beams,
            max_output_length,
            min_output_length,
            early_stopping,
            max_time,
            repetition_penalty,
            number_returns,
            enable_sampler,
            chat_mode,
            prompt,
            input_context_for_instruction,
            langchain_mode.value,
            langchain_top_k_docs,
            langchain_enable_chunk,
            langchain_chunk_size,
            langchain_document_choice,
            api_name="/submit_nochat",
        )

    async def create_async(
        self,
        prompt: str,
        prompt_type: enums.PromptType = enums.PromptType.plain,
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
        langchain_mode: enums.LangChainMode = enums.LangChainMode.DISABLED,
    ) -> str:
        """
        Creates a new text completion asynchronously.

        :param prompt: text prompt to generate completions for
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
        :return: response from the model
        """
        # Not exposed parameters.
        instruction = ""  # empty when chat_mode is False
        input = ""  # only chat_mode is True
        stream_output = False
        prompt_dict = ""  # empty as prompt_type cannot be 'custom'
        chat_mode = False
        langchain_top_k_docs = 4  # number of document chunks; not public
        langchain_enable_chunk = True  # whether to chunk documents; not public
        langchain_chunk_size = 512  # chunk size for document chunking; not public
        langchain_document_choice = ["All"]  # not public

        return await self._client._predict_async(
            instruction,
            input,
            system_pre_context,
            stream_output,
            prompt_type.value,
            prompt_dict,
            temperature,
            top_p,
            top_k,
            beams,
            max_output_length,
            min_output_length,
            early_stopping,
            max_time,
            repetition_penalty,
            number_returns,
            enable_sampler,
            chat_mode,
            prompt,
            input_context_for_instruction,
            langchain_mode.value,
            langchain_top_k_docs,
            langchain_enable_chunk,
            langchain_chunk_size,
            langchain_document_choice,
            api_name="/submit_nochat",
        )


class ChatCompletion:
    """Chat completion"""

    def __init__(self, client: Client):
        self._client = client

    def create(
        self,
        prompt_type: enums.PromptType = enums.PromptType.plain,
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
        langchain_mode: enums.LangChainMode = enums.LangChainMode.DISABLED,
    ) -> "ChatContext":
        """
        Creates a new text completion asynchronously.

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
        :return: a chat context with given parameters
        """
        kwargs = collections.OrderedDict(
            instruction=None,  # future prompts
            input="",  # ??
            system_pre_context=system_pre_context,
            stream_output=False,
            prompt_type=prompt_type.value,
            prompt_dict="",  # empty as prompt_type cannot be 'custom'
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            beams=beams,
            max_output_length=max_output_length,
            min_output_length=min_output_length,
            early_stopping=early_stopping,
            max_time=max_time,
            repetition_penalty=repetition_penalty,
            number_returns=number_returns,
            enable_sampler=enable_sampler,
            chat_mode=True,
            instruction_nochat="",  # empty when chat_mode is True
            input_context_for_instruction=input_context_for_instruction,
            langchain_mode=langchain_mode.value,
            langchain_top_k_docs=4,  # number of document chunks; not public
            langchain_enable_chunk=True,  # whether to chunk documents; not public
            langchain_chunk_size=512,  # chunk size for document chunking; not public
            langchain_document_choice=["All"],  # not public
            chatbot=[],  # chat history
        )
        return ChatContext(self._client, kwargs)


class ChatContext:
    """ "Chat context"""

    def __init__(self, client: Client, kwargs: OrderedDict[str, Any]):
        self._client = client
        self._kwargs = kwargs

    def chat(self, prompt: str) -> Dict[str, str]:
        """
        Chat with the GPT.

        :param prompt: text prompt to generate completions for
        :returns chat reply
        """
        self._kwargs["instruction"] = prompt
        self._kwargs["chatbot"] += [[prompt, None]]
        response: Tuple[List[List[str]], str] = self._client._predict(
            *self._kwargs.values(), api_name="/instruction_bot"
        )
        self._kwargs["chatbot"][-1][1] = response[0][-1][1]
        return {"user": response[0][-1][0], "gpt": response[0][-1][1]}

    def chat_history(self) -> List[Dict[str, str]]:
        """Returns the full chat history."""
        return [{"user": i[0], "gpt": i[1]} for i in self._kwargs["chatbot"]]
