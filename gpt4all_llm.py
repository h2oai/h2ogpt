import inspect
import os
import sys
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
    # defaults (some of these are generation parameters, so need to be passed in at generation time)
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
        from gpt4all import GPT4All as GPT4AllModel
    elif base_model == "gptj":
        if 'model_path_gptj' not in model_kwargs:
            raise ValueError("No model_path_gptj in %s" % env_gpt4all_file)
        model_path = model_kwargs.pop('model_path_gptj')
        from gpt4all import GPT4All as GPT4AllModel
    else:
        raise ValueError("No such base_model %s" % base_model)
    func_names = list(inspect.signature(GPT4AllModel).parameters)
    model_kwargs = {k: v for k, v in model_kwargs.items() if k in func_names}
    model = GPT4AllModel(model_path, **model_kwargs)
    return model, FakeTokenizer(), 'cpu'


from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class H2OStreamingStdOutCallbackHandler(StreamingStdOutCallbackHandler):

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        # streaming to std already occurs without this
        # sys.stdout.write(token)
        # sys.stdout.flush()
        pass


def get_llm_gpt4all(model_name, model=None,
                    max_new_tokens=256,
                    temperature=0.1,
                    repetition_penalty=1.0,
                    top_k=40,
                    top_p=0.7):
    env_gpt4all_file = ".env_gpt4all"
    model_kwargs = dotenv_values(env_gpt4all_file)
    callbacks = [H2OStreamingStdOutCallbackHandler()]
    n_ctx = model_kwargs.pop('n_ctx', 1024)
    default_params = {'context_erase': 0.5, 'n_batch': 1, 'n_ctx': n_ctx, 'n_predict': max_new_tokens,
                      'repeat_last_n': 64 if repetition_penalty != 1.0 else 0, 'repeat_penalty': repetition_penalty,
                      'temp': temperature, 'top_k': top_k, 'top_p': top_p}
    if model_name == 'llama':
        from langchain.llms import LlamaCpp
        model_path = model_kwargs.pop('model_path_llama') if model is None else model
        llm = LlamaCpp(model_path=model_path, n_ctx=n_ctx, callbacks=callbacks, verbose=False)
    else:
        model_path = model_kwargs.pop('model_path_gptj') if model is None else model
        llm = H2OGPT4All(model=model_path, backend='gptj', callbacks=callbacks,
                         verbose=False, **default_params,
                         )
    return llm


class H2OGPT4All(gpt4all.GPT4All):
    model: Any
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
            else:
                values["client"] = values["model"]
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
    ) -> str:
        # Roughly 4 chars per token if natural language
        prompt = prompt[-self.n_ctx * 4:]
        verbose = False
        if verbose:
            print("_call prompt: %s" % prompt, flush=True)
        return super()._call(prompt, stop=stop, run_manager=run_manager)
