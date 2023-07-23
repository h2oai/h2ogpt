from functools import partial

from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, Dict, List, Optional
from pydantic import Field, root_validator
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator
from exllama.lora import ExLlamaLora
import os, glob


class H2OExLlamaTokenizer(ExLlamaTokenizer):
    def __call__(self, text, *args, **kwargs):
        return dict(input_ids=self.encode(text))


class H2OExLlamaGenerator(ExLlamaGenerator):
    def is_exlama(self):
        return True


class Exllama(LLM):
    client: Any  #: :meta private:
    model_path: str = None
    model: Any = None
    sanitize_bot_response: bool = False
    prompter: Any = None
    context: Any = ''
    iinput: Any = ''

    """The path to the GPTQ model folder."""
    exllama_cache: ExLlamaCache = None  #: :meta private:
    config: ExLlamaConfig = None  #: :meta private:
    generator: ExLlamaGenerator = None  #: :meta private:
    tokenizer: ExLlamaTokenizer = None  #: :meta private:

    ##Langchain parameters
    logfunc = print
    stop_sequences: Optional[List[str]] = Field("", description="Sequences that immediately will stop the generator.")
    streaming: Optional[bool] = Field(True, description="Whether to stream the results, token by token.")

    ##Generator parameters
    disallowed_tokens: Optional[List[int]] = Field(None, description="List of tokens to disallow during generation.")
    temperature: Optional[float] = Field(None, description="Temperature for sampling diversity.")
    top_k: Optional[int] = Field(None,
                                 description="Consider the most probable top_k samples, 0 to disable top_k sampling.")
    top_p: Optional[float] = Field(None,
                                   description="Consider tokens up to a cumulative probabiltiy of top_p, 0.0 to disable top_p sampling.")
    min_p: Optional[float] = Field(None, description="Do not consider tokens with probability less than this.")
    typical: Optional[float] = Field(None,
                                     description="Locally typical sampling threshold, 0.0 to disable typical sampling.")
    token_repetition_penalty_max: Optional[float] = Field(None,
                                                          description="Repetition penalty for most recent tokens.")
    token_repetition_penalty_sustain: Optional[int] = Field(None,
                                                            description="No. most recent tokens to repeat penalty for, -1 to apply to whole context.")
    token_repetition_penalty_decay: Optional[int] = Field(None,
                                                          description="Gradually decrease penalty over this many tokens.")
    beams: Optional[int] = Field(None, description="Number of beams for beam search.")
    beam_length: Optional[int] = Field(None, description="Length of beams for beam search.")

    ##Config overrides
    max_seq_len: Optional[int] = Field(2048,
                                       decription="Reduce to save memory. Can also be increased, ideally while also using compress_pos_emn and a compatible model/LoRA")
    compress_pos_emb: Optional[float] = Field(1.0,
                                              description="Amount of compression to apply to the positional embedding.")
    set_auto_map: Optional[str] = Field(None,
                                        description="Comma-separated list of VRAM (in GB) to use per GPU device for model layers, e.g. 20,7,7")
    gpu_peer_fix: Optional[bool] = Field(None, description="Prevent direct copies of data between GPUs")
    alpha_value: Optional[float] = Field(1.0, description="Rope context extension alpha")

    ##Tuning
    matmul_recons_thd: Optional[int] = Field(None)
    fused_mlp_thd: Optional[int] = Field(None)
    sdp_thd: Optional[int] = Field(None)
    fused_attn: Optional[bool] = Field(None)
    matmul_fused_remap: Optional[bool] = Field(None)
    rmsnorm_no_half2: Optional[bool] = Field(None)
    rope_no_half2: Optional[bool] = Field(None)
    matmul_no_half2: Optional[bool] = Field(None)
    silu_no_half2: Optional[bool] = Field(None)
    concurrent_streams: Optional[bool] = Field(None)

    ##Lora Parameters
    lora_path: Optional[str] = Field(None, description="Path to your lora.")

    @staticmethod
    def get_model_path_at(path):
        patterns = ["*.safetensors", "*.bin", "*.pt"]
        model_paths = []
        for pattern in patterns:
            full_pattern = os.path.join(path, pattern)
            model_paths = glob.glob(full_pattern)
            if model_paths:  # If there are any files matching the current pattern
                break  # Exit the loop as soon as we find a matching file
        if model_paths:  # If there are any files matching any of the patterns
            return model_paths[0]
        else:
            return None  # Return None if no matching files were found

    @staticmethod
    def configure_object(params, values, logfunc):
        obj_params = {k: values.get(k) for k in params}

        def apply_to(obj):
            for key, value in obj_params.items():
                if value:
                    if hasattr(obj, key):
                        setattr(obj, key, value)
                        logfunc(f"{key} {value}")
                    else:
                        raise AttributeError(f"{key} does not exist in {obj}")

        return apply_to

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        model_param_names = [
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "typical",
            "token_repetition_penalty_max",
            "token_repetition_penalty_sustain",
            "token_repetition_penalty_decay",
            "beams",
            "beam_length",
        ]

        config_param_names = [
            "max_seq_len",
            "compress_pos_emb",
            "gpu_peer_fix",
            "alpha_value"
        ]

        tuning_parameters = [
            "matmul_recons_thd",
            "fused_mlp_thd",
            "sdp_thd",
            "matmul_fused_remap",
            "rmsnorm_no_half2",
            "rope_no_half2",
            "matmul_no_half2",
            "silu_no_half2",
            "concurrent_streams",
            "fused_attn",
        ]

        ##Set logging function if verbose or set to empty lambda
        verbose = values['verbose']
        if not verbose:
            values['logfunc'] = lambda *args, **kwargs: None
        logfunc = values['logfunc']

        if values['model'] is None:
            model_path = values["model_path"]
            lora_path = values["lora_path"]

            tokenizer_path = os.path.join(model_path, "tokenizer.model")
            model_config_path = os.path.join(model_path, "config.json")
            model_path = Exllama.get_model_path_at(model_path)

            config = ExLlamaConfig(model_config_path)
            tokenizer = ExLlamaTokenizer(tokenizer_path)
            config.model_path = model_path

            configure_config = Exllama.configure_object(config_param_names, values, logfunc)
            configure_config(config)
            configure_tuning = Exllama.configure_object(tuning_parameters, values, logfunc)
            configure_tuning(config)

            ##Special parameter, set auto map, it's a function
            if values['set_auto_map']:
                config.set_auto_map(values['set_auto_map'])
                logfunc(f"set_auto_map {values['set_auto_map']}")

            model = ExLlama(config)
            exllama_cache = ExLlamaCache(model)
            generator = ExLlamaGenerator(model, tokenizer, exllama_cache)

            ##Load and apply lora to generator
            if lora_path is not None:
                lora_config_path = os.path.join(lora_path, "adapter_config.json")
                lora_path = Exllama.get_model_path_at(lora_path)
                lora = ExLlamaLora(model, lora_config_path, lora_path)
                generator.lora = lora
                logfunc(f"Loaded LORA @ {lora_path}")
        else:
            generator = values['model']
            exllama_cache = generator.cache
            model = generator.model
            config = model.config
            tokenizer = generator.tokenizer

        # Set if model existed before or not since generation-time parameters
        configure_model = Exllama.configure_object(model_param_names, values, logfunc)
        values["stop_sequences"] = [x.strip().lower() for x in values["stop_sequences"]]
        configure_model(generator.settings)

        setattr(generator.settings, "stop_sequences", values["stop_sequences"])
        logfunc(f"stop_sequences {values['stop_sequences']}")

        disallowed = values.get("disallowed_tokens")
        if disallowed:
            generator.disallow_tokens(disallowed)
            print(f"Disallowed Tokens: {generator.disallowed_tokens}")

        values["client"] = model
        values["generator"] = generator
        values["config"] = config
        values["tokenizer"] = tokenizer
        values["exllama_cache"] = exllama_cache

        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "Exllama"

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text."""
        return self.generator.tokenizer.num_tokens(text)

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        assert self.tokenizer is not None
        from h2oai_pipeline import H2OTextGenerationPipeline
        prompt, num_prompt_tokens = H2OTextGenerationPipeline.limit_prompt(prompt, self.tokenizer)

        # NOTE: TGI server does not add prompting, so must do here
        data_point = dict(context=self.context, instruction=prompt, input=self.iinput)
        prompt = self.prompter.generate_prompt(data_point)

        text = ''
        for text1 in self.stream(prompt=prompt, stop=stop, run_manager=run_manager):
            text = text1
        return text

    from enum import Enum

    class MatchStatus(Enum):
        EXACT_MATCH = 1
        PARTIAL_MATCH = 0
        NO_MATCH = 2

    def match_status(self, sequence: str, banned_sequences: List[str]):
        sequence = sequence.strip().lower()
        for banned_seq in banned_sequences:
            if banned_seq == sequence:
                return self.MatchStatus.EXACT_MATCH
            elif banned_seq.startswith(sequence):
                return self.MatchStatus.PARTIAL_MATCH
        return self.MatchStatus.NO_MATCH

    def stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        config = self.config
        generator = self.generator
        beam_search = (self.beams and self.beams >= 1 and self.beam_length and self.beam_length >= 1)

        ids = generator.tokenizer.encode(prompt)
        generator.gen_begin_reuse(ids)

        if beam_search:
            generator.begin_beam_search()
            token_getter = generator.beam_search
        else:
            generator.end_beam_search()
            token_getter = generator.gen_single_token

        last_newline_pos = 0
        seq_length = len(generator.tokenizer.decode(generator.sequence_actual[0]))
        response_start = seq_length
        cursor_head = response_start

        text_callback = None
        if run_manager:
            text_callback = partial(
                run_manager.on_llm_new_token, verbose=self.verbose
            )
        # parent handler of streamer expects to see prompt first else output="" and lose if prompt=None in prompter
        if text_callback:
            text_callback(prompt)
        text = ""
        while (generator.gen_num_tokens() <= (
                self.max_seq_len - 4)):  # Slight extra padding space as we seem to occassionally get a few more than 1-2 tokens
            # Fetch a token
            token = token_getter()

            # If it's the ending token replace it and end the generation.
            if token.item() == generator.tokenizer.eos_token_id:
                generator.replace_last_token(generator.tokenizer.newline_token_id)
                if beam_search:
                    generator.end_beam_search()
                return

            # Tokenize the string from the last new line, we can't just decode the last token due to how sentencepiece decodes.
            stuff = generator.tokenizer.decode(generator.sequence_actual[0][last_newline_pos:])
            cursor_tail = len(stuff)
            text_chunk = stuff[cursor_head:cursor_tail]
            cursor_head = cursor_tail

            # Append the generated chunk to our stream buffer
            text += text_chunk
            text = self.prompter.get_response(prompt + text, prompt=prompt,
                                              sanitize_bot_response=self.sanitize_bot_response)

            if token.item() == generator.tokenizer.newline_token_id:
                last_newline_pos = len(generator.sequence_actual[0])
                cursor_head = 0
                cursor_tail = 0

            # Check if the stream buffer is one of the stop sequences
            status = self.match_status(text, self.stop_sequences)

            if status == self.MatchStatus.EXACT_MATCH:
                # Encountered a stop, rewind our generator to before we hit the match and end generation.
                rewind_length = generator.tokenizer.encode(text).shape[-1]
                generator.gen_rewind(rewind_length)
                #gen = generator.tokenizer.decode(generator.sequence_actual[0][response_start:])
                if beam_search:
                    generator.end_beam_search()
                return
            elif status == self.MatchStatus.PARTIAL_MATCH:
                # Partially matched a stop, continue buffering but don't yield.
                continue
            elif status == self.MatchStatus.NO_MATCH:
                if text_callback:
                    text_callback(text_chunk)
                yield text  # Not a stop, yield the match buffer.

        return
