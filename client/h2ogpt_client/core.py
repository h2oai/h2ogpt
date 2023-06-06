import asyncio
from enum import Enum
from typing import Optional

import gradio_client  # type: ignore


class PromptType(Enum):
    DAI_FAQ = "dai_faq"
    HUMAN_BOT = "human_bot"
    HUMAN_BOT_ORIGINAL = "human_bot_orig"
    INSTRUCT = "instruct"
    INSTRUCT_SIMPLE = "instruct_simple"
    INSTRUCT_VICUNA = "instruct_vicuna"
    INSTRUCT_WITH_END = "instruct_with_end"
    OPEN_ASSISTANT = "open_assistant"
    PLAIN = "plain"
    PROMPT_ANSWER = "prompt_answer"
    QUALITY = "quality"
    SIMPLE_INSTRUCT = "simple_instruct"
    SUMMARIZE = "summarize"
    WIZARD_LM = "wizard_lm"
    WIZARD_MEGA = "wizard_mega"
    WIZARD_MEGA = "wizard_mega"


class LangChainMode(Enum):
    ALL = "All"
    CHAT_LLM = "ChatLLM"
    DISABLED = "Disabled"
    GITHUB_H2OGPT = "github h2oGPT"
    H2O_DAI_DOCS = "DriverlessAI docs"
    LLM = "LLM"
    MY_DATA = "MyData"
    USER_DATA = "UserData"
    WIKI = "wiki"
    WIKI_FULL = "wiki_full"


class Client:
    def __init__(self, server_url: str, huggingface_token: Optional[str] = None):
        self._client = gradio_client.Client(
            src=server_url, hf_token=huggingface_token, verbose=False
        )
        self._text_completion = TextCompletion(self)

    @property
    def text_completion(self) -> "TextCompletion":
        return self._text_completion

    def _predict(self, *args, api_name: str) -> str:
        return self._client.submit(*args, api_name=api_name).result()

    async def _predict_async(self, *args, api_name: str) -> str:
        return await asyncio.wrap_future(self._client.submit(*args, api_name=api_name))


class TextCompletion:
    def __init__(self, client: Client):
        self._client = client

    def create(
        self,
        prompt: str,
        prompt_type: PromptType = PromptType.PLAIN,
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
        input: str = "",
        system_pre_context: str = "",
        langchain_mode: LangChainMode = LangChainMode.DISABLED,
    ) -> str:
        """
        Creates a new text completion.

        :param prompt: The prompt to generate completions for.
        :param prompt_type: Type of the prompt.
        :param input_context_for_instruction: Input context for instruction.
        :param enable_sampler: Enable or disable tje sampler, required for use of
                temperature, top_p, top_k.
        :param temperature: What sampling temperature to use, between 0 and 3.
                Lower values will make it more focused and deterministic, but may lead
                to repeat. Higher values will make the output more creative, but may
                lead to hallucinations.
        :param top_p: Cumulative probability of tokens to sample from.
        :param top_k: Number of tokens to sample from.
        :param beams: Number of searches for optimal overall probability.
                Higher values uses more GPU memory and compute.
        :param early_stopping: Whether to stop early or not in beam search.
        :param min_output_length:
        :param max_output_length:
        :param max_time: Maximum time to search optimal output.
        :param repetition_penalty:
        :param number_returns:
        :param input:
        :param system_pre_context:
        :param langchain_mode:
        :return: response from the model
        """
        # Not exposed parameters.
        you = ""  # empty when chat_mode is False
        stream_output = False
        chat_mode = False

        return self._client._predict(
            you,
            input,
            system_pre_context,
            stream_output,
            prompt_type.value,
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
            api_name="/submit_nochat",
        )

    async def create_async(
        self,
        prompt: str,
        prompt_type: PromptType = PromptType.PLAIN,
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
        input: str = "",
        system_pre_context: str = "",
        langchain_mode: LangChainMode = LangChainMode.DISABLED,
    ) -> str:
        """
        Creates a new text completion.

        :param prompt: The prompt to generate completions for.
        :param prompt_type: Type of the prompt.
        :param input_context_for_instruction: Input context for instruction.
        :param enable_sampler: Enable or disable tje sampler, required for use of
                temperature, top_p, top_k.
        :param temperature: What sampling temperature to use, between 0 and 3.
                Lower values will make it more focused and deterministic, but may lead
                to repeat. Higher values will make the output more creative, but may
                lead to hallucinations.
        :param top_p: Cumulative probability of tokens to sample from.
        :param top_k: Number of tokens to sample from.
        :param beams: Number of searches for optimal overall probability.
                Higher values uses more GPU memory and compute.
        :param early_stopping: Whether to stop early or not in beam search.
        :param min_output_length:
        :param max_output_length:
        :param max_time: Maximum time to search optimal output.
        :param repetition_penalty:
        :param number_returns:
        :param input:
        :param system_pre_context:
        :param langchain_mode:
        :return: response from the model
        """
        # Not exposed parameters.
        you = ""  # empty when chat_mode is False
        stream_output = False
        chat_mode = False

        return await self._client._predict_async(
            you,
            input,
            system_pre_context,
            stream_output,
            prompt_type.value,
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
            api_name="/submit_nochat",
        )
