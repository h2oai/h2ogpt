import inspect
import os
from functools import partial
from typing import Dict, Any, Optional, List
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import root_validator
from langchain.llms import gpt4all
from dotenv import dotenv_values

from utils import FakeTokenizer


def get_model_tokenizer_gpt4all(base_model, **kwargs):
    # defaults (some of these are generation parameters, so need to be passed in at generation time)
    model_kwargs = dict(n_threads=os.cpu_count() // 2,
                        temp=kwargs.get('temperature', 0.2),
                        top_p=kwargs.get('top_p', 0.75),
                        top_k=kwargs.get('top_k', 40),
                        n_ctx=2048 - 256)
    env_gpt4all_file = ".env_gpt4all"
    model_kwargs.update(dotenv_values(env_gpt4all_file))
    # make int or float if can to satisfy types for class
    for k, v in model_kwargs.items():
        try:
            if float(v) == int(v):
                model_kwargs[k] = int(v)
            else:
                model_kwargs[k] = float(v)
        except:
            pass

    if base_model == "llama":
        if 'model_path_llama' not in model_kwargs:
            raise ValueError("No model_path_llama in %s" % env_gpt4all_file)
        model_path = model_kwargs.pop('model_path_llama')
        # FIXME: GPT4All version of llama doesn't handle new quantization, so use llama_cpp_python
        from llama_cpp import Llama
        # llama sets some things at init model time, not generation time
        func_names = list(inspect.signature(Llama.__init__).parameters)
        model_kwargs = {k: v for k, v in model_kwargs.items() if k in func_names}
        model_kwargs['n_ctx'] = int(model_kwargs['n_ctx'])
        model = Llama(model_path=model_path, **model_kwargs)
    elif base_model in "gpt4all_llama":
        if 'model_name_gpt4all_llama' not in model_kwargs and 'model_path_gpt4all_llama' not in model_kwargs:
            raise ValueError("No model_name_gpt4all_llama or model_path_gpt4all_llama in %s" % env_gpt4all_file)
        model_name = model_kwargs.pop('model_name_gpt4all_llama')
        model_type = 'llama'
        from gpt4all import GPT4All as GPT4AllModel
        model = GPT4AllModel(model_name=model_name, model_type=model_type)
    elif base_model in "gptj":
        if 'model_name_gptj' not in model_kwargs and 'model_path_gptj' not in model_kwargs:
            raise ValueError("No model_name_gpt4j or model_path_gpt4j in %s" % env_gpt4all_file)
        model_name = model_kwargs.pop('model_name_gptj')
        model_type = 'gptj'
        from gpt4all import GPT4All as GPT4AllModel
        model = GPT4AllModel(model_name=model_name, model_type=model_type)
    else:
        raise ValueError("No such base_model %s" % base_model)
    return model, FakeTokenizer(), 'cpu'


from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class H2OStreamingStdOutCallbackHandler(StreamingStdOutCallbackHandler):

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        # streaming to std already occurs without this
        # sys.stdout.write(token)
        # sys.stdout.flush()
        pass


def get_model_kwargs(env_kwargs, default_kwargs, cls, exclude_list=[]):
    # default from class
    model_kwargs = {k: v.default for k, v in dict(inspect.signature(cls).parameters).items() if k not in exclude_list}
    # from our defaults
    model_kwargs.update(default_kwargs)
    # from user defaults
    model_kwargs.update(env_kwargs)
    # ensure only valid keys
    func_names = list(inspect.signature(cls).parameters)
    model_kwargs = {k: v for k, v in model_kwargs.items() if k in func_names}
    return model_kwargs


def get_llm_gpt4all(model_name,
                    model=None,
                    max_new_tokens=256,
                    temperature=0.1,
                    repetition_penalty=1.0,
                    top_k=40,
                    top_p=0.7,
                    streaming=False,
                    callbacks=None,
                    prompter=None,
                    context='',
                    iinput='',
                    verbose=False,
                    ):
    assert prompter is not None
    env_gpt4all_file = ".env_gpt4all"
    env_kwargs = dotenv_values(env_gpt4all_file)
    max_tokens = env_kwargs.pop('max_tokens', 2048 - max_new_tokens)
    default_kwargs = dict(context_erase=0.5,
                          n_batch=1,
                          max_tokens=max_tokens,
                          n_predict=max_new_tokens,
                          repeat_last_n=64 if repetition_penalty != 1.0 else 0,
                          repeat_penalty=repetition_penalty,
                          temp=temperature,
                          temperature=temperature,
                          top_k=top_k,
                          top_p=top_p,
                          use_mlock=True,
                          verbose=verbose)
    if model_name == 'llama':
        cls = H2OLlamaCpp
        model_path = env_kwargs.pop('model_path_llama') if model is None else model
        model_kwargs = get_model_kwargs(env_kwargs, default_kwargs, cls, exclude_list=['lc_kwargs'])
        model_kwargs.update(dict(model_path=model_path, callbacks=callbacks, streaming=streaming,
                                 prompter=prompter, context=context, iinput=iinput))
        llm = cls(**model_kwargs)
        llm.client.verbose = verbose
    elif model_name == 'gpt4all_llama':
        cls = H2OGPT4All
        model_path = env_kwargs.pop('model_path_gpt4all_llama') if model is None else model
        model_kwargs = get_model_kwargs(env_kwargs, default_kwargs, cls, exclude_list=['lc_kwargs'])
        model_kwargs.update(
            dict(model=model_path, backend='llama', callbacks=callbacks, streaming=streaming,
                 prompter=prompter, context=context, iinput=iinput))
        llm = cls(**model_kwargs)
    elif model_name == 'gptj':
        cls = H2OGPT4All
        model_path = env_kwargs.pop('model_path_gptj') if model is None else model
        model_kwargs = get_model_kwargs(env_kwargs, default_kwargs, cls, exclude_list=['lc_kwargs'])
        model_kwargs.update(
            dict(model=model_path, backend='gptj', callbacks=callbacks, streaming=streaming,
                 prompter=prompter, context=context, iinput=iinput))
        llm = cls(**model_kwargs)
    else:
        raise RuntimeError("No such model_name %s" % model_name)
    return llm


class H2OGPT4All(gpt4all.GPT4All):
    model: Any
    prompter: Any
    context: Any = ''
    iinput: Any = ''
    """Path to the pre-trained GPT4All model file."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in the environment."""
        try:
            if isinstance(values["model"], str):
                from gpt4all import GPT4All as GPT4AllModel

                full_path = values["model"]
                model_path, delimiter, model_name = full_path.rpartition("/")
                model_path += delimiter

                values["client"] = GPT4AllModel(
                    model_name=model_name,
                    model_path=model_path or None,
                    model_type=values["backend"],
                    allow_download=False,
                )
                if values["n_threads"] is not None:
                    # set n_threads
                    values["client"].model.set_thread_count(values["n_threads"])
            else:
                values["client"] = values["model"]
            try:
                values["backend"] = values["client"].model_type
            except AttributeError:
                # The below is for compatibility with GPT4All Python bindings <= 0.2.3.
                values["backend"] = values["client"].model.model_type

        except ImportError:
            raise ValueError(
                "Could not import gpt4all python package. "
                "Please install it with `pip install gpt4all`."
            )
        return values

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs,
    ) -> str:
        # Roughly 4 chars per token if natural language
        n_ctx = 2048
        prompt = prompt[-self.max_tokens * 4:]

        # use instruct prompting
        data_point = dict(context=self.context, instruction=prompt, input=self.iinput)
        prompt = self.prompter.generate_prompt(data_point)

        verbose = False
        if verbose:
            print("_call prompt: %s" % prompt, flush=True)
        # FIXME: GPT4ALl doesn't support yield during generate, so cannot support streaming except via itself to stdout
        return super()._call(prompt, stop=stop, run_manager=run_manager)


from langchain.llms import LlamaCpp


class H2OLlamaCpp(LlamaCpp):
    model_path: Any
    prompter: Any
    context: Any
    iinput: Any
    """Path to the pre-trained GPT4All model file."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that llama-cpp-python library is installed."""
        if isinstance(values["model_path"], str):
            model_path = values["model_path"]
            model_param_names = [
                "lora_path",
                "lora_base",
                "n_ctx",
                "n_parts",
                "seed",
                "f16_kv",
                "logits_all",
                "vocab_only",
                "use_mlock",
                "n_threads",
                "n_batch",
                "use_mmap",
                "last_n_tokens_size",
            ]
            model_params = {k: values[k] for k in model_param_names}
            # For backwards compatibility, only include if non-null.
            if values["n_gpu_layers"] is not None:
                model_params["n_gpu_layers"] = values["n_gpu_layers"]

            try:
                from llama_cpp import Llama

                values["client"] = Llama(model_path, **model_params)
            except ImportError:
                raise ModuleNotFoundError(
                    "Could not import llama-cpp-python library. "
                    "Please install the llama-cpp-python library to "
                    "use this embedding model: pip install llama-cpp-python"
                )
            except Exception as e:
                raise ValueError(
                    f"Could not load Llama model from path: {model_path}. "
                    f"Received error {e}"
                )
        else:
            values["client"] = values["model_path"]
        return values

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs,
    ) -> str:
        verbose = False
        # tokenize twice, just to count tokens, since llama cpp python wrapper has no way to truncate
        # still have to avoid crazy sizes, else hit llama_tokenize: too many tokens -- might still hit, not fatal
        prompt = prompt[-self.n_ctx * 4:]
        prompt_tokens = self.client.tokenize(b" " + prompt.encode("utf-8"))
        num_prompt_tokens = len(prompt_tokens)
        if num_prompt_tokens > self.n_ctx:
            # conservative by using int()
            chars_per_token = int(len(prompt) / num_prompt_tokens)
            prompt = prompt[-self.n_ctx * chars_per_token:]
            if verbose:
                print("reducing tokens, assuming average of %s chars/token: %s" % chars_per_token, flush=True)
                prompt_tokens2 = self.client.tokenize(b" " + prompt.encode("utf-8"))
                num_prompt_tokens2 = len(prompt_tokens2)
                print("reduced tokens from %d -> %d" % (num_prompt_tokens, num_prompt_tokens2), flush=True)

        # use instruct prompting
        data_point = dict(context=self.context, instruction=prompt, input=self.iinput)
        prompt = self.prompter.generate_prompt(data_point)

        if verbose:
            print("_call prompt: %s" % prompt, flush=True)

        if self.streaming:
            text_callback = None
            if run_manager:
                text_callback = partial(
                    run_manager.on_llm_new_token, verbose=self.verbose
                )
            # parent handler of streamer expects to see prompt first else output="" and lose if prompt=None in prompter
            if text_callback:
                text_callback(prompt)
            text = ""
            for token in self.stream(prompt=prompt, stop=stop, run_manager=run_manager):
                text_chunk = token["choices"][0]["text"]
                # self.stream already calls text_callback
                # if text_callback:
                #    text_callback(text_chunk)
                text += text_chunk
            return text
        else:
            params = self._get_parameters(stop)
            params = {**params, **kwargs}
            result = self.client(prompt=prompt, **params)
            return result["choices"][0]["text"]
