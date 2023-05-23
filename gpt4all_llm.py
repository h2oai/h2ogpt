import inspect
import os
from typing import Dict, Any, Optional, List
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import root_validator
from langchain.llms import gpt4all
from dotenv import dotenv_values


class FakeTokenizer:

    def encode(self, x, *args, **kwargs):
        return dict(input_ids=[x])

    def decode(self, x, *args, **kwargs):
        return x

    def __call__(self, x, *args, **kwargs):
        return self.encode(x, *args, **kwargs)


def get_model_tokenizer_gpt4all(base_model, **kwargs):
        # defaults
        model_kwargs = dict(n_ctx=kwargs.get('max_new_tokens', 256),
                            n_threads=os.cpu_count() // 2,
                            temp=kwargs.get('temperature', 0.2),
                            top_p=kwargs.get('top_p', 0.75),
                            top_k=kwargs.get('top_k', 40))
        env_gpt4all_file = ".env_gpt4all"
        model_kwargs.update(dotenv_values(env_gpt4all_file))

        if base_model == "llama":
            if 'model_path_llama' not in model_kwargs:
                raise ValueError("No model_path_llama in %s" % env_gpt4all_file)
            model_path = model_kwargs.pop('model_path_llama')
            from pygpt4all import GPT4All as GPT4AllModel
        elif base_model == "gptj":
            if 'model_path_gptj' not in model_kwargs:
                raise ValueError("No model_path_gptj in %s" % env_gpt4all_file)
            model_path = model_kwargs.pop('model_path_gptj')
            from pygpt4all import GPT4All_J as GPT4AllModel
        else:
            raise ValueError("No such base_model %s" % base_model)
        func_names = list(inspect.signature(GPT4AllModel).parameters)
        model_kwargs = {k: v for k, v in model_kwargs.items() if k in func_names}
        model = GPT4AllModel(model_path, **model_kwargs)
        return model, FakeTokenizer(), 'cpu'


def get_llm_gpt4all(model_name, model=None, max_new_tokens=256):
    env_gpt4all_file = ".env_gpt4all"
    from dotenv import dotenv_values
    model_kwargs = dotenv_values(env_gpt4all_file)
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    callbacks = [StreamingStdOutCallbackHandler()]
    model_n_ctx = max_new_tokens
    if model_name == 'llama':
        from langchain.llms import LlamaCpp
        model_path = model_kwargs.pop('model_path_llama') if model is None else model
        llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
    else:
        model_path = model_kwargs.pop('model_path_gptj') if model is None else model
        from gpt4all_llm import H2OGPT4All
        llm = H2OGPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
    return llm


class H2OGPT4All(gpt4all.GPT4All):
    model: Any
    """Path to the pre-trained GPT4All model file."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in the environment."""
        try:
            if isinstance(values["model"], str):
                backend = values["backend"]
                if backend == "llama":
                    from pygpt4all import GPT4All as GPT4AllModel
                elif backend == "gptj":
                    from pygpt4all import GPT4All_J as GPT4AllModel
                else:
                    raise ValueError(f"Incorrect gpt4all backend {cls.backend}")

                model_kwargs = {
                    k: v
                    for k, v in values.items()
                    if k in H2OGPT4All._model_param_names(backend)
                }
                values["client"] = GPT4AllModel(
                    model_path=values["model"],
                    **model_kwargs,
                )
            else:
                values["client"] = values["model"]
        except ImportError:
            raise ValueError(
                "Could not import pygpt4all python package. "
                "Please install it with `pip install pygpt4all`."
            )
        return values

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        # Roughly 4 chars per token if natural language
        prompt = prompt[-self.n_ctx*4:]
        print("_call prompt: %s" % prompt, flush=True)
        return super()._call(prompt, stop=stop, run_manager=run_manager)
