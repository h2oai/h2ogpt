import inspect
import os
from typing import Dict, Any, Optional, List, Iterator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema.output import GenerationChunk
from langchain.llms import gpt4all
from pydantic.v1 import root_validator

from utils import FakeTokenizer, url_alive, download_simple, clear_torch_cache, n_gpus_global


def get_model_tokenizer_gpt4all(base_model, n_jobs=None, gpu_id=None, n_gpus=None, max_seq_len=None,
                                llamacpp_dict=None,
                                llamacpp_path=None):
    cvd = os.getenv('CUDA_VISIBLE_DEVICES')
    if gpu_id is not None and gpu_id != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    assert llamacpp_dict is not None
    # defaults (some of these are generation parameters, so need to be passed in at generation time)
    model_name = base_model.lower()
    llama_kwargs = dict(model_name=model_name,
                        model=None,
                        n_jobs=n_jobs,
                        n_gpus=n_gpus,
                        main_gpu=gpu_id if gpu_id not in [None, -1, '-1'] else 0,
                        inner_class=True,
                        max_seq_len=max_seq_len,
                        llamacpp_dict=llamacpp_dict,
                        llamacpp_path=llamacpp_path)
    model, tokenizer, redo, max_seq_len = get_llm_gpt4all(**llama_kwargs)
    if redo:
        del model
        del tokenizer
        clear_torch_cache()
        # auto max_seq_len
        llama_kwargs.update(dict(max_seq_len=max_seq_len))
        model, tokenizer, redo, max_seq_len = get_llm_gpt4all(**llama_kwargs)
    if cvd is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cvd
    else:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    return model, tokenizer, 'cpu' if n_gpus != 0 else 'cuda'


from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class H2OStreamingStdOutCallbackHandler(StreamingStdOutCallbackHandler):

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        # streaming to std already occurs without this
        # sys.stdout.write(token)
        # sys.stdout.flush()
        pass


def get_model_kwargs(llamacpp_dict, default_kwargs, cls, exclude_list=[]):
    # default from class
    model_kwargs = {k: v.default for k, v in dict(inspect.signature(cls).parameters).items() if k not in exclude_list}
    # from our defaults
    model_kwargs.update(default_kwargs)
    # from user defaults
    model_kwargs.update(llamacpp_dict)
    # ensure only valid keys
    func_names = list(inspect.signature(cls).parameters)
    model_kwargs = {k: v for k, v in model_kwargs.items() if k in func_names}
    # make int or float if can to satisfy types for class
    for k, v in model_kwargs.items():
        try:
            if float(v) == int(v):
                model_kwargs[k] = int(v)
            else:
                model_kwargs[k] = float(v)
        except:
            pass
    return model_kwargs


def get_gpt4all_default_kwargs(max_new_tokens=256,
                               temperature=0.1,
                               repetition_penalty=1.0,
                               top_k=40,
                               top_p=0.7,
                               n_jobs=None,
                               verbose=False,
                               max_seq_len=None,
                               main_gpu=0,
                               ):
    if n_jobs in [None, -1]:
        n_jobs = int(os.getenv('OMP_NUM_THREADS', str(os.cpu_count() // 2)))
    n_jobs = max(1, min(20, n_jobs))  # hurts beyond some point
    n_gpus = n_gpus_global
    max_seq_len_local = max_seq_len if max_seq_len is not None else 2048  # fake for auto mode
    default_kwargs = dict(context_erase=0.5,
                          n_batch=1,
                          max_tokens=max_seq_len_local - max_new_tokens,
                          n_predict=max_new_tokens,
                          repeat_last_n=64 if repetition_penalty != 1.0 else 0,
                          repeat_penalty=repetition_penalty,
                          temp=temperature,
                          temperature=temperature,
                          top_k=top_k,
                          top_p=top_p,
                          use_mlock=True,
                          n_ctx=max_seq_len_local,
                          n_threads=n_jobs,
                          main_gpu=main_gpu,
                          verbose=verbose)
    if n_gpus != 0:
        default_kwargs.update(dict(n_gpu_layers=100, f16_kv=True))
    return default_kwargs


def get_llm_gpt4all(model_name=None,
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
                    n_jobs=None,
                    n_gpus=None,
                    main_gpu=0,
                    verbose=False,
                    inner_class=False,
                    max_seq_len=None,
                    llamacpp_path=None,
                    llamacpp_dict=None,
                    ):
    redo = False
    if not inner_class:
        assert prompter is not None

    default_kwargs = \
        get_gpt4all_default_kwargs(max_new_tokens=max_new_tokens,
                                   temperature=temperature,
                                   repetition_penalty=repetition_penalty,
                                   top_k=top_k,
                                   top_p=top_p,
                                   n_jobs=n_jobs,
                                   verbose=verbose,
                                   max_seq_len=max_seq_len,
                                   main_gpu=main_gpu,
                                   )
    if model_name == 'llama':
        # FIXME: streaming not thread safe due to:
        # llama_cpp/utils.py:        sys.stdout = self.outnull_file
        # llama_cpp/utils.py:        sys.stdout = self.old_stdout
        cls = H2OLlamaCpp
        if model is None:
            llamacpp_dict = llamacpp_dict.copy()
            model_path = llamacpp_dict.pop('model_path_llama')
            llamacpp_path = os.getenv('LLAMACPP_PATH', llamacpp_path) or './'
            if os.path.isfile(os.path.basename(model_path)):
                # e.g. if offline but previously downloaded
                model_path = os.path.basename(model_path)
            elif os.path.isfile(os.path.join(llamacpp_path, os.path.basename(model_path))):
                # e.g. so don't have to point to full previously-downloaded path
                model_path = os.path.join(llamacpp_path, os.path.basename(model_path))
            elif url_alive(model_path):
                # online
                dest = os.path.join(llamacpp_path, os.path.basename(model_path)) if llamacpp_path else None
                if dest.endswith('?download=true'):
                    dest = dest.replace('?download=true', '')
                model_path = download_simple(model_path, dest=dest)
        else:
            model_path = model
        model_kwargs = get_model_kwargs(llamacpp_dict, default_kwargs, cls, exclude_list=['lc_kwargs'])
        model_kwargs.update(dict(model_path=model_path, callbacks=callbacks, streaming=streaming,
                                 prompter=prompter, context=context, iinput=iinput,
                                 n_gpus=n_gpus))

        # migration to  new langchain fix:
        odd_keys = ['model_kwargs', 'grammar_path', 'grammar']
        for key in odd_keys:
            model_kwargs.pop(key, None)

        llm = cls(**model_kwargs)
        llm.client.verbose = verbose
        inner_model = llm.client

        if max_seq_len is None:
            redo = True
            max_seq_len = llm.client.n_embd()
            print("Auto-detected LLaMa n_ctx=%s, will unload then reload with this setting." % max_seq_len)

        # with multiple GPUs, something goes wrong unless generation occurs early before other imports
        # CUDA error 704 at /tmp/pip-install-khkugdmy/llama-cpp-python_8c0a9782b7604a5aaf95ec79856eac97/vendor/llama.cpp/ggml-cuda.cu:6408: peer access is already enabled
        inner_model("Say exactly one word", max_tokens=1)
        inner_tokenizer = FakeTokenizer(tokenizer=llm.client, is_llama_cpp=True, model_max_length=max_seq_len)
    elif model_name == 'gpt4all_llama':
        # FIXME: streaming not thread safe due to:
        # gpt4all/pyllmodel.py:        sys.stdout = stream_processor
        # gpt4all/pyllmodel.py:        sys.stdout = old_stdout

        cls = H2OGPT4All
        if model is None:
            llamacpp_dict = llamacpp_dict.copy()
            model_path = llamacpp_dict.pop('model_name_gpt4all_llama')
            if url_alive(model_path):
                # online
                llamacpp_path = os.getenv('LLAMACPP_PATH', llamacpp_path) or './'
                dest = os.path.join(llamacpp_path, os.path.basename(model_path)) if llamacpp_path else None
                model_path = download_simple(model_path, dest=dest)
        else:
            model_path = model
        model_kwargs = get_model_kwargs(llamacpp_dict, default_kwargs, cls, exclude_list=['lc_kwargs'])
        model_kwargs.update(
            dict(model=model_path, backend='llama', callbacks=callbacks, streaming=streaming,
                 prompter=prompter, context=context, iinput=iinput))
        llm = cls(**model_kwargs)
        inner_model = llm.client
        inner_tokenizer = FakeTokenizer(model_max_length=max_seq_len)
    elif model_name == 'gptj':
        # FIXME: streaming not thread safe due to:
        # gpt4all/pyllmodel.py:        sys.stdout = stream_processor
        # gpt4all/pyllmodel.py:        sys.stdout = old_stdout

        cls = H2OGPT4All
        if model is None:
            llamacpp_dict = llamacpp_dict.copy()
            model_path = llamacpp_dict.pop('model_name_gptj') if model is None else model
            if url_alive(model_path):
                llamacpp_path = os.getenv('LLAMACPP_PATH', llamacpp_path) or './'
                dest = os.path.join(llamacpp_path, os.path.basename(model_path)) if llamacpp_path else None
                model_path = download_simple(model_path, dest=dest)
        else:
            model_path = model
        model_kwargs = get_model_kwargs(llamacpp_dict, default_kwargs, cls, exclude_list=['lc_kwargs'])
        model_kwargs.update(
            dict(model=model_path, backend='gptj', callbacks=callbacks, streaming=streaming,
                 prompter=prompter, context=context, iinput=iinput))
        llm = cls(**model_kwargs)
        inner_model = llm.client
        inner_tokenizer = FakeTokenizer(model_max_length=max_seq_len)
    else:
        raise RuntimeError("No such model_name %s" % model_name)
    if inner_class:
        return inner_model, inner_tokenizer, redo, max_seq_len
    else:
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
                    allow_download=True,
                )
                if values["n_threads"] is not None:
                    # set n_threads
                    values["client"].model.set_thread_count(values["n_threads"])
            else:
                values["client"] = values["model"]
                if values["n_threads"] is not None:
                    # set n_threads
                    values["client"].model.set_thread_count(values["n_threads"])
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

    # FIXME:  Unsure what uses
    # def get_token_ids(self, text: str) -> List[int]:
    #    return self.client.tokenize(b" " + text.encode("utf-8"))


from langchain.llms import LlamaCpp


class H2OLlamaCpp(LlamaCpp):
    """Path to the pre-trained GPT4All model file."""
    model_path: Any
    prompter: Any
    context: Any
    iinput: Any
    count_input_tokens: Any = 0
    prompts: Any = []
    count_output_tokens: Any = 0
    n_gpus: Any = -1

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
                try:
                    if values["n_gpus"] == 0:
                        from llama_cpp import Llama
                    else:
                        from llama_cpp_cuda import Llama
                except Exception as e:
                    print("Failed to listen to n_gpus: %s" % str(e), flush=True)
                    try:
                        from llama_cpp import Llama
                    except ImportError:
                        from llama_cpp_cuda import Llama

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

        inner_tokenizer = FakeTokenizer(tokenizer=self.client, is_llama_cpp=True, model_max_length=self.n_ctx)
        assert inner_tokenizer is not None
        from h2oai_pipeline import H2OTextGenerationPipeline
        prompt, num_prompt_tokens = H2OTextGenerationPipeline.limit_prompt(prompt, inner_tokenizer,
                                                                           max_prompt_length=self.n_ctx)

        # use instruct prompting
        data_point = dict(context=self.context, instruction=prompt, input=self.iinput)
        prompt = self.prompter.generate_prompt(data_point)
        self.count_input_tokens += self.get_num_tokens(prompt)
        self.prompts.append(prompt)
        if stop is None:
            stop = []
        stop.extend(self.prompter.stop_sequences)

        if verbose:
            print("_call prompt: %s" % prompt, flush=True)

        if self.streaming:
            # parent handler of streamer expects to see prompt first else output="" and lose if prompt=None in prompter
            text = ""
            for token in self.stream(input=prompt, stop=stop):
                # for token in self.stream(input=prompt, stop=stop, run_manager=run_manager):
                text_chunk = token  # ["choices"][0]["text"]
                text += text_chunk
            self.count_output_tokens += self.get_num_tokens(text)
            text = self.remove_stop_text(text, stop=stop)
            return text
        else:
            params = self._get_parameters(stop)
            params = {**params, **kwargs}
            result = self.client(prompt=prompt, **params)
            text = result["choices"][0]["text"]
            self.count_output_tokens += self.get_num_tokens(text)
            text = self.remove_stop_text(text, stop=stop)
            return text

    def remove_stop_text(self, text, stop=None):
        # remove stop sequences from the end of the generated text
        if stop is None:
            return text
        for stop_seq in stop:
            if stop_seq in text:
                text = text[:text.index(stop_seq)]
        return text

    def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        # parent expects only see actual new tokens, not prompt too
        total_text = ''
        for chunk in super()._stream(prompt, stop=stop, run_manager=run_manager, **kwargs):
            # remove stop sequences from the end of the generated text
            total_text += chunk.text
            got_stop = False
            if stop:
                for stop_seq in stop:
                    if stop_seq in total_text:
                        got_stop = True
            if not got_stop:
                yield chunk

    def get_token_ids(self, text: str) -> List[int]:
        return self.client.tokenize(b" " + text.encode("utf-8"))
