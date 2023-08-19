import ast
import asyncio
import copy
import functools
import glob
import inspect
import os
import pathlib
import pickle
import shutil
import subprocess
import tempfile
import time
import traceback
import types
import typing
import uuid
import zipfile
from collections import defaultdict
from datetime import datetime
from functools import reduce
from operator import concat
import filelock
import tabulate

from joblib import delayed
from langchain.callbacks import streaming_stdout
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.schema import LLMResult, Generation
from tqdm import tqdm

from enums import DocumentSubset, no_lora_str, model_token_mapping, source_prefix, source_postfix, non_query_commands, \
    LangChainAction, LangChainMode, DocumentChoice, LangChainTypes, font_size, head_acc, super_source_prefix, \
    super_source_postfix
from evaluate_params import gen_hyper
from gen import get_model, SEED
from prompter import non_hf_types, PromptType, Prompter
from utils import wrapped_partial, EThread, import_matplotlib, sanitize_filename, makedirs, get_url, flatten_list, \
    get_device, ProgressParallel, remove, hash_file, clear_torch_cache, NullContext, get_hf_server, FakeTokenizer, \
    have_libreoffice, have_arxiv, have_playwright, have_selenium, have_tesseract, have_pymupdf, set_openai, \
    get_list_or_str, have_pillow, only_selenium, only_playwright, only_unstructured_urls, get_sha, get_short_name, \
    get_accordion
from utils_langchain import StreamingGradioCallbackHandler

import_matplotlib()

import numpy as np
import pandas as pd
import requests
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
# , GCSDirectoryLoader, GCSFileLoader
# , OutlookMessageLoader # GPL3
# ImageCaptionLoader, # use our own wrapper
#  ReadTheDocsLoader,  # no special file, some path, so have to give as special option
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader, PythonLoader, TomlLoader, \
    UnstructuredURLLoader, UnstructuredHTMLLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader, \
    EverNoteLoader, UnstructuredEmailLoader, UnstructuredODTLoader, UnstructuredPowerPointLoader, \
    UnstructuredEPubLoader, UnstructuredImageLoader, UnstructuredRTFLoader, ArxivLoader, UnstructuredPDFLoader, \
    UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain import PromptTemplate, HuggingFaceTextGenInference
from langchain.vectorstores import Chroma


def get_db(sources, use_openai_embedding=False, db_type='faiss',
           persist_directory=None, load_db_if_exists=True,
           langchain_mode='notset',
           langchain_mode_paths={},
           langchain_mode_types={},
           collection_name=None,
           hf_embedding_model=None,
           migrate_embedding_model=False):
    if not sources:
        return None
    user_path = langchain_mode_paths.get(langchain_mode)
    if persist_directory is None:
        langchain_type = langchain_mode_types.get(langchain_mode, LangChainTypes.EITHER.value)
        persist_directory, langchain_type = get_persist_directory(langchain_mode, langchain_type=langchain_type)
        langchain_mode_types[langchain_mode] = langchain_type
    assert hf_embedding_model is not None

    # get freshly-determined embedding model
    embedding = get_embedding(use_openai_embedding, hf_embedding_model=hf_embedding_model)
    assert collection_name is not None or langchain_mode != 'notset'
    if collection_name is None:
        collection_name = langchain_mode.replace(' ', '_')

    # Create vector database
    if db_type == 'faiss':
        from langchain.vectorstores import FAISS
        db = FAISS.from_documents(sources, embedding)
    elif db_type == 'weaviate':
        import weaviate
        from weaviate.embedded import EmbeddedOptions
        from langchain.vectorstores import Weaviate

        if os.getenv('WEAVIATE_URL', None):
            client = _create_local_weaviate_client()
        else:
            client = weaviate.Client(
                embedded_options=EmbeddedOptions(persistence_data_path=persist_directory)
            )
        index_name = collection_name.capitalize()
        db = Weaviate.from_documents(documents=sources, embedding=embedding, client=client, by_text=False,
                                     index_name=index_name)
    elif db_type == 'chroma':
        assert persist_directory is not None
        # use_base already handled when making persist_directory, unless was passed into get_db()
        makedirs(persist_directory, exist_ok=True)

        # see if already actually have persistent db, and deal with possible changes in embedding
        db, use_openai_embedding, hf_embedding_model = \
            get_existing_db(None, persist_directory, load_db_if_exists, db_type,
                            use_openai_embedding,
                            langchain_mode, langchain_mode_paths, langchain_mode_types,
                            hf_embedding_model, migrate_embedding_model, verbose=False)
        if db is None:
            from chromadb.config import Settings
            client_settings = Settings(anonymized_telemetry=False,
                                       chroma_db_impl="duckdb+parquet",
                                       persist_directory=persist_directory)
            db = Chroma.from_documents(documents=sources,
                                       embedding=embedding,
                                       persist_directory=persist_directory,
                                       collection_name=collection_name,
                                       client_settings=client_settings)
            db.persist()
            clear_embedding(db)
            save_embed(db, use_openai_embedding, hf_embedding_model)
        else:
            # then just add
            # doesn't check or change embedding, just saves it in case not saved yet, after persisting
            db, num_new_sources, new_sources_metadata = add_to_db(db, sources, db_type=db_type,
                                                                  use_openai_embedding=use_openai_embedding,
                                                                  hf_embedding_model=hf_embedding_model)
    else:
        raise RuntimeError("No such db_type=%s" % db_type)

    # once here, db is not changing and embedding choices in calling functions does not matter
    return db


def _get_unique_sources_in_weaviate(db):
    batch_size = 100
    id_source_list = []
    result = db._client.data_object.get(class_name=db._index_name, limit=batch_size)

    while result['objects']:
        id_source_list += [(obj['id'], obj['properties']['source']) for obj in result['objects']]
        last_id = id_source_list[-1][0]
        result = db._client.data_object.get(class_name=db._index_name, limit=batch_size, after=last_id)

    unique_sources = {source for _, source in id_source_list}
    return unique_sources


def del_from_db(db, sources, db_type=None):
    if db_type == 'chroma':
        # sources should be list of x.metadata['source'] from document metadatas
        if isinstance(sources, str):
            sources = [sources]
        else:
            assert isinstance(sources, (list, tuple, types.GeneratorType))
        metadatas = set(sources)
        client_collection = db._client.get_collection(name=db._collection.name,
                                                      embedding_function=db._collection._embedding_function)
        for source in metadatas:
            meta = dict(source=source)
            try:
                client_collection.delete(where=meta)
            except KeyError:
                pass


def add_to_db(db, sources, db_type='faiss',
              avoid_dup_by_file=False,
              avoid_dup_by_content=True,
              use_openai_embedding=False,
              hf_embedding_model=None):
    assert hf_embedding_model is not None
    num_new_sources = len(sources)
    if not sources:
        return db, num_new_sources, []
    if db_type == 'faiss':
        db.add_documents(sources)
    elif db_type == 'weaviate':
        # FIXME: only control by file name, not hash yet
        if avoid_dup_by_file or avoid_dup_by_content:
            unique_sources = _get_unique_sources_in_weaviate(db)
            sources = [x for x in sources if x.metadata['source'] not in unique_sources]
        num_new_sources = len(sources)
        if num_new_sources == 0:
            return db, num_new_sources, []
        db.add_documents(documents=sources)
    elif db_type == 'chroma':
        collection = get_documents(db)
        # files we already have:
        metadata_files = set([x['source'] for x in collection['metadatas']])
        if avoid_dup_by_file:
            # Too weak in case file changed content, assume parent shouldn't pass true for this for now
            raise RuntimeError("Not desired code path")
        if avoid_dup_by_content:
            # look at hash, instead of page_content
            # migration: If no hash previously, avoid updating,
            #  since don't know if need to update and may be expensive to redo all unhashed files
            metadata_hash_ids = set(
                [x['hashid'] for x in collection['metadatas'] if 'hashid' in x and x['hashid'] not in ["None", None]])
            # avoid sources with same hash
            sources = [x for x in sources if x.metadata.get('hashid') not in metadata_hash_ids]
            num_nohash = len([x for x in sources if not x.metadata.get('hashid')])
            print("Found %s new sources (%d have no hash in original source,"
                  " so have to reprocess for migration to sources with hash)" % (len(sources), num_nohash), flush=True)
            # get new file names that match existing file names.  delete existing files we are overridding
            dup_metadata_files = set([x.metadata['source'] for x in sources if x.metadata['source'] in metadata_files])
            print("Removing %s duplicate files from db because ingesting those as new documents" % len(
                dup_metadata_files), flush=True)
            client_collection = db._client.get_collection(name=db._collection.name,
                                                          embedding_function=db._collection._embedding_function)
            for dup_file in dup_metadata_files:
                dup_file_meta = dict(source=dup_file)
                try:
                    client_collection.delete(where=dup_file_meta)
                except KeyError:
                    pass
        num_new_sources = len(sources)
        if num_new_sources == 0:
            return db, num_new_sources, []
        if hasattr(db, '_persist_directory'):
            print("Existing db, adding to %s" % db._persist_directory, flush=True)
            # chroma only
            name_path = os.path.basename(db._persist_directory)
            base_path = 'locks'
            base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
            file = os.path.join(base_path, "getdb_%s.lock" % name_path)
            context = filelock.FileLock
        else:
            file = None
            context = NullContext
        with context(file):
            # this is place where add to db, but others maybe accessing db, so lock access.
            # else see RuntimeError: Index seems to be corrupted or unsupported
            db.add_documents(documents=sources)
            db.persist()
            clear_embedding(db)
            # save here is for migration, in case old db directory without embedding saved
            save_embed(db, use_openai_embedding, hf_embedding_model)
    else:
        raise RuntimeError("No such db_type=%s" % db_type)

    new_sources_metadata = [x.metadata for x in sources]

    return db, num_new_sources, new_sources_metadata


def create_or_update_db(db_type, persist_directory, collection_name,
                        user_path, langchain_type,
                        sources, use_openai_embedding, add_if_exists, verbose,
                        hf_embedding_model, migrate_embedding_model):
    if not os.path.isdir(persist_directory) or not add_if_exists:
        if os.path.isdir(persist_directory):
            if verbose:
                print("Removing %s" % persist_directory, flush=True)
            remove(persist_directory)
        if verbose:
            print("Generating db", flush=True)
    if db_type == 'weaviate':
        import weaviate
        from weaviate.embedded import EmbeddedOptions

        if os.getenv('WEAVIATE_URL', None):
            client = _create_local_weaviate_client()
        else:
            client = weaviate.Client(
                embedded_options=EmbeddedOptions(persistence_data_path=persist_directory)
            )

        index_name = collection_name.replace(' ', '_').capitalize()
        if client.schema.exists(index_name) and not add_if_exists:
            client.schema.delete_class(index_name)
            if verbose:
                print("Removing %s" % index_name, flush=True)
    elif db_type == 'chroma':
        pass

    if not add_if_exists:
        if verbose:
            print("Generating db", flush=True)
    else:
        if verbose:
            print("Loading and updating db", flush=True)

    db = get_db(sources,
                use_openai_embedding=use_openai_embedding,
                db_type=db_type,
                persist_directory=persist_directory,
                langchain_mode=collection_name,
                langchain_mode_paths={collection_name: user_path},
                langchain_mode_types={collection_name: langchain_type},
                hf_embedding_model=hf_embedding_model,
                migrate_embedding_model=migrate_embedding_model)

    return db


def get_embedding(use_openai_embedding, hf_embedding_model=None):
    assert hf_embedding_model is not None
    # Get embedding model
    if use_openai_embedding:
        assert os.getenv("OPENAI_API_KEY") is not None, "Set ENV OPENAI_API_KEY"
        from langchain.embeddings import OpenAIEmbeddings
        embedding = OpenAIEmbeddings(disallowed_special=())
    else:
        # to ensure can fork without deadlock
        from langchain.embeddings import HuggingFaceEmbeddings

        device, torch_dtype, context_class = get_device_dtype()
        model_kwargs = dict(device=device)
        if 'instructor' in hf_embedding_model:
            encode_kwargs = {'normalize_embeddings': True}
            embedding = HuggingFaceInstructEmbeddings(model_name=hf_embedding_model,
                                                      model_kwargs=model_kwargs,
                                                      encode_kwargs=encode_kwargs)
        else:
            embedding = HuggingFaceEmbeddings(model_name=hf_embedding_model, model_kwargs=model_kwargs)
    return embedding


def get_answer_from_sources(chain, sources, question):
    return chain(
        {
            "input_documents": sources,
            "question": question,
        },
        return_only_outputs=True,
    )["output_text"]


"""Wrapper around Huggingface text generation inference API."""
from functools import partial
from typing import Any, Dict, List, Optional, Set, Iterable

from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain.llms.base import LLM


class GradioInference(LLM):
    """
    Gradio generation inference API.
    """
    inference_server_url: str = ""

    temperature: float = 0.8
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = None
    num_beams: Optional[int] = 1
    max_new_tokens: int = 512
    min_new_tokens: int = 1
    early_stopping: bool = False
    max_time: int = 180
    repetition_penalty: Optional[float] = None
    num_return_sequences: Optional[int] = 1
    do_sample: bool = False
    chat_client: bool = False

    return_full_text: bool = False
    stream_output: bool = False
    sanitize_bot_response: bool = False

    prompter: Any = None
    context: Any = ''
    iinput: Any = ''
    client: Any = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

        try:
            if values['client'] is None:
                import gradio_client
                values["client"] = gradio_client.Client(
                    values["inference_server_url"]
                )
        except ImportError:
            raise ImportError(
                "Could not import gradio_client python package. "
                "Please install it with `pip install gradio_client`."
            )
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "gradio_inference"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        # NOTE: prompt here has no prompt_type (e.g. human: bot:) prompt injection,
        # so server should get prompt_type or '', not plain
        # This is good, so gradio server can also handle stopping.py conditions
        # this is different than TGI server that uses prompter to inject prompt_type prompting
        stream_output = self.stream_output
        gr_client = self.client
        client_langchain_mode = 'Disabled'
        client_add_chat_history_to_context = True
        client_langchain_action = LangChainAction.QUERY.value
        client_langchain_agents = []
        top_k_docs = 1
        chunk = True
        chunk_size = 512
        client_kwargs = dict(instruction=prompt if self.chat_client else '',  # only for chat=True
                             iinput=self.iinput if self.chat_client else '',  # only for chat=True
                             context=self.context,
                             # streaming output is supported, loops over and outputs each generation in streaming mode
                             # but leave stream_output=False for simple input/output mode
                             stream_output=stream_output,
                             prompt_type=self.prompter.prompt_type,
                             prompt_dict='',

                             temperature=self.temperature,
                             top_p=self.top_p,
                             top_k=self.top_k,
                             num_beams=self.num_beams,
                             max_new_tokens=self.max_new_tokens,
                             min_new_tokens=self.min_new_tokens,
                             early_stopping=self.early_stopping,
                             max_time=self.max_time,
                             repetition_penalty=self.repetition_penalty,
                             num_return_sequences=self.num_return_sequences,
                             do_sample=self.do_sample,
                             chat=self.chat_client,

                             instruction_nochat=prompt if not self.chat_client else '',
                             iinput_nochat=self.iinput if not self.chat_client else '',
                             langchain_mode=client_langchain_mode,
                             add_chat_history_to_context=client_add_chat_history_to_context,
                             langchain_action=client_langchain_action,
                             langchain_agents=client_langchain_agents,
                             top_k_docs=top_k_docs,
                             chunk=chunk,
                             chunk_size=chunk_size,
                             document_subset=DocumentSubset.Relevant.name,
                             document_choice=[DocumentChoice.ALL.value],
                             )
        api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
        if not stream_output:
            res = gr_client.predict(str(dict(client_kwargs)), api_name=api_name)
            res_dict = ast.literal_eval(res)
            text = res_dict['response']
            return self.prompter.get_response(prompt + text, prompt=prompt,
                                              sanitize_bot_response=self.sanitize_bot_response)
        else:
            text_callback = None
            if run_manager:
                text_callback = partial(
                    run_manager.on_llm_new_token, verbose=self.verbose
                )

            job = gr_client.submit(str(dict(client_kwargs)), api_name=api_name)
            text0 = ''
            while not job.done():
                outputs_list = job.communicator.job.outputs
                if outputs_list:
                    res = job.communicator.job.outputs[-1]
                    res_dict = ast.literal_eval(res)
                    text = res_dict['response']
                    text = self.prompter.get_response(prompt + text, prompt=prompt,
                                                      sanitize_bot_response=self.sanitize_bot_response)
                    # FIXME: derive chunk from full for now
                    text_chunk = text[len(text0):]
                    # save old
                    text0 = text

                    if text_callback:
                        text_callback(text_chunk)

                time.sleep(0.01)

            # ensure get last output to avoid race
            res_all = job.outputs()
            if len(res_all) > 0:
                res = res_all[-1]
                res_dict = ast.literal_eval(res)
                text = res_dict['response']
                # FIXME: derive chunk from full for now
            else:
                # go with old if failure
                text = text0
            text_chunk = text[len(text0):]
            if text_callback:
                text_callback(text_chunk)
            return self.prompter.get_response(prompt + text, prompt=prompt,
                                              sanitize_bot_response=self.sanitize_bot_response)


class H2OHuggingFaceTextGenInference(HuggingFaceTextGenInference):
    max_new_tokens: int = 512
    do_sample: bool = False
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.95
    typical_p: Optional[float] = 0.95
    temperature: float = 0.8
    repetition_penalty: Optional[float] = None
    return_full_text: bool = False
    stop_sequences: List[str] = Field(default_factory=list)
    seed: Optional[int] = None
    inference_server_url: str = ""
    timeout: int = 300
    headers: dict = None
    stream_output: bool = False
    sanitize_bot_response: bool = False
    prompter: Any = None
    context: Any = ''
    iinput: Any = ''
    tokenizer: Any = None
    async_sem: Any = None
    count_input_tokens: Any = 0
    count_output_tokens: Any = 0

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        if stop is None:
            stop = self.stop_sequences.copy()
        else:
            stop += self.stop_sequences.copy()
        stop_tmp = stop.copy()
        stop = []
        [stop.append(x) for x in stop_tmp if x not in stop]

        # HF inference server needs control over input tokens
        assert self.tokenizer is not None
        from h2oai_pipeline import H2OTextGenerationPipeline
        prompt, num_prompt_tokens = H2OTextGenerationPipeline.limit_prompt(prompt, self.tokenizer)

        # NOTE: TGI server does not add prompting, so must do here
        data_point = dict(context=self.context, instruction=prompt, input=self.iinput)
        prompt = self.prompter.generate_prompt(data_point)
        self.count_input_tokens += self.get_num_tokens(prompt)

        gen_server_kwargs = dict(do_sample=self.do_sample,
                                 stop_sequences=stop,
                                 max_new_tokens=self.max_new_tokens,
                                 top_k=self.top_k,
                                 top_p=self.top_p,
                                 typical_p=self.typical_p,
                                 temperature=self.temperature,
                                 repetition_penalty=self.repetition_penalty,
                                 return_full_text=self.return_full_text,
                                 seed=self.seed,
                                 )
        gen_server_kwargs.update(kwargs)

        # lower bound because client is re-used if multi-threading
        self.client.timeout = max(300, self.timeout)

        if not self.stream_output:
            res = self.client.generate(
                prompt,
                **gen_server_kwargs,
            )
            if self.return_full_text:
                gen_text = res.generated_text[len(prompt):]
            else:
                gen_text = res.generated_text
            # remove stop sequences from the end of the generated text
            for stop_seq in stop:
                if stop_seq in gen_text:
                    gen_text = gen_text[:gen_text.index(stop_seq)]
            text = prompt + gen_text
            text = self.prompter.get_response(text, prompt=prompt,
                                              sanitize_bot_response=self.sanitize_bot_response)
        else:
            text_callback = None
            if run_manager:
                text_callback = partial(
                    run_manager.on_llm_new_token, verbose=self.verbose
                )
            # parent handler of streamer expects to see prompt first else output="" and lose if prompt=None in prompter
            if text_callback:
                text_callback(prompt)
            text = ""
            # Note: Streaming ignores return_full_text=True
            for response in self.client.generate_stream(prompt, **gen_server_kwargs):
                text_chunk = response.token.text
                text += text_chunk
                text = self.prompter.get_response(prompt + text, prompt=prompt,
                                                  sanitize_bot_response=self.sanitize_bot_response)
                # stream part
                is_stop = False
                for stop_seq in stop:
                    if stop_seq in text_chunk:
                        is_stop = True
                        break
                if is_stop:
                    break
                if not response.token.special:
                    if text_callback:
                        text_callback(text_chunk)
        self.count_output_tokens += self.get_num_tokens(text)
        return text

    async def _acall(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        # print("acall", flush=True)
        if stop is None:
            stop = self.stop_sequences.copy()
        else:
            stop += self.stop_sequences.copy()
        stop_tmp = stop.copy()
        stop = []
        [stop.append(x) for x in stop_tmp if x not in stop]

        # HF inference server needs control over input tokens
        assert self.tokenizer is not None
        from h2oai_pipeline import H2OTextGenerationPipeline
        prompt, num_prompt_tokens = H2OTextGenerationPipeline.limit_prompt(prompt, self.tokenizer)

        # NOTE: TGI server does not add prompting, so must do here
        data_point = dict(context=self.context, instruction=prompt, input=self.iinput)
        prompt = self.prompter.generate_prompt(data_point)

        gen_text = await super()._acall(prompt, stop=stop, run_manager=run_manager, **kwargs)

        # remove stop sequences from the end of the generated text
        for stop_seq in stop:
            if stop_seq in gen_text:
                gen_text = gen_text[:gen_text.index(stop_seq)]
        text = prompt + gen_text
        text = self.prompter.get_response(text, prompt=prompt,
                                          sanitize_bot_response=self.sanitize_bot_response)
        # print("acall done", flush=True)
        return text

    async def _agenerate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        generations = []
        new_arg_supported = inspect.signature(self._acall).parameters.get("run_manager")
        self.count_input_tokens += sum([self.get_num_tokens(prompt) for prompt in prompts])
        tasks = [
            asyncio.ensure_future(self._agenerate_one(prompt, stop=stop, run_manager=run_manager,
                                                      new_arg_supported=new_arg_supported, **kwargs))
            for prompt in prompts
        ]
        texts = await asyncio.gather(*tasks)
        self.count_output_tokens += sum([self.get_num_tokens(text) for text in texts])
        [generations.append([Generation(text=text)]) for text in texts]
        return LLMResult(generations=generations)

    async def _agenerate_one(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            new_arg_supported=None,
            **kwargs: Any,
    ) -> str:
        async with self.async_sem:  # semaphore limits num of simultaneous downloads
            return await self._acall(prompt, stop=stop, run_manager=run_manager, **kwargs) \
                if new_arg_supported else \
                await self._acall(prompt, stop=stop, **kwargs)

    def get_token_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
        # avoid base method that is not aware of how to properly tokenize (uses GPT2)
        # return _get_token_ids_default_method(text)


from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.llms import OpenAI, AzureOpenAI, Replicate
from langchain.llms.openai import _streaming_response_template, completion_with_retry, _update_response, \
    update_token_usage


class H2OOpenAI(OpenAI):
    """
    New class to handle vLLM's use of OpenAI, no vllm_chat supported, so only need here
    Handles prompting that OpenAI doesn't need, stopping as well
    """
    stop_sequences: Any = None
    sanitize_bot_response: bool = False
    prompter: Any = None
    context: Any = ''
    iinput: Any = ''
    tokenizer: Any = None

    @classmethod
    def _all_required_field_names(cls) -> Set:
        _all_required_field_names = super(OpenAI, cls)._all_required_field_names()
        _all_required_field_names.update(
            {'top_p', 'frequency_penalty', 'presence_penalty', 'stop_sequences', 'sanitize_bot_response', 'prompter',
             'tokenizer', 'logit_bias'})
        return _all_required_field_names

    def _generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
        stop_tmp = self.stop_sequences if not stop else self.stop_sequences + stop
        stop = []
        [stop.append(x) for x in stop_tmp if x not in stop]

        # HF inference server needs control over input tokens
        assert self.tokenizer is not None
        from h2oai_pipeline import H2OTextGenerationPipeline
        for prompti, prompt in enumerate(prompts):
            prompt, num_prompt_tokens = H2OTextGenerationPipeline.limit_prompt(prompt, self.tokenizer)
            # NOTE: OpenAI/vLLM server does not add prompting, so must do here
            data_point = dict(context=self.context, instruction=prompt, input=self.iinput)
            prompt = self.prompter.generate_prompt(data_point)
            prompts[prompti] = prompt

        params = self._invocation_params
        params = {**params, **kwargs}
        sub_prompts = self.get_sub_prompts(params, prompts, stop)
        choices = []
        token_usage: Dict[str, int] = {}
        # Get the token usage from the response.
        # Includes prompt, completion, and total tokens used.
        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        text = ''
        for _prompts in sub_prompts:
            if self.streaming:
                text_with_prompt = ""
                prompt = _prompts[0]
                if len(_prompts) > 1:
                    raise ValueError("Cannot stream results with multiple prompts.")
                params["stream"] = True
                response = _streaming_response_template()
                first = True
                for stream_resp in completion_with_retry(
                        self, prompt=_prompts, **params
                ):
                    if first:
                        stream_resp["choices"][0]["text"] = prompt + stream_resp["choices"][0]["text"]
                        first = False
                    text_chunk = stream_resp["choices"][0]["text"]
                    text_with_prompt += text_chunk
                    text = self.prompter.get_response(text_with_prompt, prompt=prompt,
                                                      sanitize_bot_response=self.sanitize_bot_response)
                    if run_manager:
                        run_manager.on_llm_new_token(
                            text_chunk,
                            verbose=self.verbose,
                            logprobs=stream_resp["choices"][0]["logprobs"],
                        )
                    _update_response(response, stream_resp)
                choices.extend(response["choices"])
            else:
                response = completion_with_retry(self, prompt=_prompts, **params)
                choices.extend(response["choices"])
            if not self.streaming:
                # Can't update token usage if streaming
                update_token_usage(_keys, response, token_usage)
        choices[0]['text'] = text
        return self.create_llm_result(choices, prompts, token_usage)


class H2OReplicate(Replicate):
    stop_sequences: Any = None
    sanitize_bot_response: bool = False
    prompter: Any = None
    context: Any = ''
    iinput: Any = ''
    tokenizer: Any = None

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """Call to replicate endpoint."""
        stop_tmp = self.stop_sequences if not stop else self.stop_sequences + stop
        stop = []
        [stop.append(x) for x in stop_tmp if x not in stop]

        # HF inference server needs control over input tokens
        assert self.tokenizer is not None
        from h2oai_pipeline import H2OTextGenerationPipeline
        prompt, num_prompt_tokens = H2OTextGenerationPipeline.limit_prompt(prompt, self.tokenizer)
        # Note Replicate handles the prompting of the specific model
        return super()._call(prompt, stop=stop, run_manager=run_manager, **kwargs)


class H2OChatOpenAI(ChatOpenAI):
    @classmethod
    def _all_required_field_names(cls) -> Set:
        _all_required_field_names = super(ChatOpenAI, cls)._all_required_field_names()
        _all_required_field_names.update({'top_p', 'frequency_penalty', 'presence_penalty', 'logit_bias'})
        return _all_required_field_names


class H2OAzureChatOpenAI(AzureChatOpenAI):
    @classmethod
    def _all_required_field_names(cls) -> Set:
        _all_required_field_names = super(AzureChatOpenAI, cls)._all_required_field_names()
        _all_required_field_names.update({'top_p', 'frequency_penalty', 'presence_penalty', 'logit_bias'})
        return _all_required_field_names


class H2OAzureOpenAI(AzureOpenAI):
    @classmethod
    def _all_required_field_names(cls) -> Set:
        _all_required_field_names = super(AzureOpenAI, cls)._all_required_field_names()
        _all_required_field_names.update({'top_p', 'frequency_penalty', 'presence_penalty', 'logit_bias'})
        return _all_required_field_names


def get_llm(use_openai_model=False,
            model_name=None,
            model=None,
            tokenizer=None,
            inference_server=None,
            langchain_only_model=None,
            stream_output=False,
            async_output=True,
            num_async=3,
            do_sample=False,
            temperature=0.1,
            top_k=40,
            top_p=0.7,
            num_beams=1,
            max_new_tokens=512,
            min_new_tokens=1,
            early_stopping=False,
            max_time=180,
            repetition_penalty=1.0,
            num_return_sequences=1,
            prompt_type=None,
            prompt_dict=None,
            prompter=None,
            context=None,
            iinput=None,
            sanitize_bot_response=False,
            system_prompt='',
            n_jobs=None,
            cli=False,
            llamacpp_dict=None,
            verbose=False,
            ):
    # currently all but h2oai_pipeline case return prompt + new text, but could change
    only_new_text = False

    if n_jobs is None:
        n_jobs = int(os.getenv('OMP_NUM_THREADS', str(os.cpu_count() // 2)))
    if inference_server is None:
        inference_server = ''
    if inference_server.startswith('replicate'):
        model_string = ':'.join(inference_server.split(':')[1:])
        gen_kwargs = dict(temperature=temperature if do_sample else 0,
                          max_length=max_new_tokens,  # langchain
                          max_new_tokens=max_new_tokens,  # replicate docs
                          top_p=top_p if do_sample else 1,
                          top_k=top_k,  # not always supported
                          repetition_penalty=repetition_penalty)
        if system_prompt:
            gen_kwargs.update(dict(system_prompt=system_prompt))
        elif prompter.system_prompt:
            gen_kwargs.update(dict(system_prompt=prompter.system_prompt))
        # replicate handles prompting, so avoid get_resopnse() filter
        prompter.prompt_type = 'plain'
        if stream_output:
            callbacks = [StreamingGradioCallbackHandler()]
            streamer = callbacks[0] if stream_output else None
            llm = H2OReplicate(
                streaming=True,
                callbacks=callbacks,
                model=model_string,
                input=gen_kwargs,
                stop=prompter.stop_sequences,
                stop_sequences=prompter.stop_sequences,
                sanitize_bot_response=sanitize_bot_response,
                prompter=prompter,
                context=context,
                iinput=iinput,
                tokenizer=tokenizer,
            )
        else:
            streamer = None
            llm = H2OReplicate(
                model=model_string,
                input=gen_kwargs,
                stop=prompter.stop_sequences,
                stop_sequences=prompter.stop_sequences,
                sanitize_bot_response=sanitize_bot_response,
                prompter=prompter,
                context=context,
                iinput=iinput,
                tokenizer=tokenizer,
            )
    elif use_openai_model or inference_server.startswith('openai') or inference_server.startswith('vllm'):
        if use_openai_model and model_name is None:
            model_name = "gpt-3.5-turbo"
        # FIXME: Will later import be ignored?  I think so, so should be fine
        openai, inf_type, deployment_name, base_url, api_version = set_openai(inference_server)
        kwargs_extra = {}
        if inf_type == 'openai_chat' or inf_type == 'vllm_chat':
            cls = H2OChatOpenAI
            # FIXME: Support context, iinput
        elif inf_type == 'openai_azure_chat':
            cls = H2OAzureChatOpenAI
            kwargs_extra.update(dict(openai_api_type='azure'))
            # FIXME: Support context, iinput
        elif inf_type == 'openai_azure':
            cls = H2OAzureOpenAI
            kwargs_extra.update(dict(openai_api_type='azure'))
            # FIXME: Support context, iinput
        else:
            cls = H2OOpenAI
            if inf_type == 'vllm':
                kwargs_extra.update(dict(stop_sequences=prompter.stop_sequences,
                                         sanitize_bot_response=sanitize_bot_response,
                                         prompter=prompter,
                                         context=context,
                                         iinput=iinput,
                                         tokenizer=tokenizer,
                                         openai_api_base=openai.api_base,
                                         client=None))
            else:
                assert inf_type == 'openai' or use_openai_model

        if deployment_name:
            kwargs_extra.update(dict(deployment_name=deployment_name))
        if api_version:
            kwargs_extra.update(dict(openai_api_version=api_version))
        elif openai.api_version:
            kwargs_extra.update(dict(openai_api_version=openai.api_version))
        elif inf_type in ['openai_azure', 'openai_azure_chat']:
            kwargs_extra.update(dict(openai_api_version="2023-05-15"))
        if base_url:
            kwargs_extra.update(dict(openai_api_base=base_url))
        else:
            kwargs_extra.update(dict(openai_api_base=openai.api_base))

        callbacks = [StreamingGradioCallbackHandler()]
        llm = cls(model_name=model_name,
                  temperature=temperature if do_sample else 0,
                  # FIXME: Need to count tokens and reduce max_new_tokens to fit like in generate.py
                  max_tokens=max_new_tokens,
                  top_p=top_p if do_sample else 1,
                  frequency_penalty=0,
                  presence_penalty=1.07 - repetition_penalty + 0.6,  # so good default
                  callbacks=callbacks if stream_output else None,
                  openai_api_key=openai.api_key,
                  logit_bias=None if inf_type == 'vllm' else {},
                  max_retries=2,
                  streaming=stream_output,
                  **kwargs_extra
                  )
        streamer = callbacks[0] if stream_output else None
        if inf_type in ['openai', 'openai_chat', 'openai_azure', 'openai_azure_chat']:
            prompt_type = inference_server
        else:
            # vllm goes here
            prompt_type = prompt_type or 'plain'
    elif inference_server:
        assert inference_server.startswith(
            'http'), "Malformed inference_server=%s.  Did you add http:// in front?" % inference_server

        from gradio_utils.grclient import GradioClient
        from text_generation import Client as HFClient
        if isinstance(model, GradioClient):
            gr_client = model
            hf_client = None
        else:
            gr_client = None
            hf_client = model
            assert isinstance(hf_client, HFClient)

        inference_server, headers = get_hf_server(inference_server)

        # quick sanity check to avoid long timeouts, just see if can reach server
        requests.get(inference_server, timeout=int(os.getenv('REQUEST_TIMEOUT_FAST', '10')))
        callbacks = [StreamingGradioCallbackHandler()]

        if gr_client:
            async_output = False  # FIXME: not implemented yet
            chat_client = False
            llm = GradioInference(
                inference_server_url=inference_server,
                return_full_text=False,

                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                early_stopping=early_stopping,
                max_time=max_time,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                chat_client=chat_client,

                callbacks=callbacks if stream_output else None,
                stream_output=stream_output,
                prompter=prompter,
                context=context,
                iinput=iinput,
                client=gr_client,
                sanitize_bot_response=sanitize_bot_response,
            )
        elif hf_client:
            # no need to pass original client, no state and fast, so can use same validate_environment from base class
            async_sem = asyncio.Semaphore(num_async) if async_output else NullContext()
            llm = H2OHuggingFaceTextGenInference(
                inference_server_url=inference_server,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                return_full_text=False,  # this only controls internal behavior, still returns processed text
                seed=SEED,

                stop_sequences=prompter.stop_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                # typical_p=top_p,
                callbacks=callbacks if stream_output else None,
                stream_output=stream_output,
                prompter=prompter,
                context=context,
                iinput=iinput,
                tokenizer=tokenizer,
                timeout=max_time,
                sanitize_bot_response=sanitize_bot_response,
                async_sem=async_sem,
            )
        else:
            raise RuntimeError("No defined client")
        streamer = callbacks[0] if stream_output else None
    elif model_name in non_hf_types:
        async_output = False  # FIXME: not implemented yet
        assert langchain_only_model
        if model_name == 'llama':
            callbacks = [StreamingGradioCallbackHandler()]
            streamer = callbacks[0] if stream_output else None
        else:
            # stream_output = False
            # doesn't stream properly as generator, but at least
            callbacks = [streaming_stdout.StreamingStdOutCallbackHandler()]
            streamer = None
        if prompter:
            prompt_type = prompter.prompt_type
        else:
            prompter = Prompter(prompt_type, prompt_dict, debug=False, chat=False, stream_output=stream_output)
            pass  # assume inputted prompt_type is correct
        from gpt4all_llm import get_llm_gpt4all
        max_max_tokens = tokenizer.model_max_length
        llm = get_llm_gpt4all(model_name,
                              model=model,
                              max_new_tokens=max_new_tokens,
                              temperature=temperature,
                              repetition_penalty=repetition_penalty,
                              top_k=top_k,
                              top_p=top_p,
                              callbacks=callbacks,
                              n_jobs=n_jobs,
                              verbose=verbose,
                              streaming=stream_output,
                              prompter=prompter,
                              context=context,
                              iinput=iinput,
                              max_seq_len=max_max_tokens,
                              llamacpp_dict=llamacpp_dict,
                              )
    elif hasattr(model, 'is_exlama') and model.is_exlama():
        async_output = False  # FIXME: not implemented yet
        assert langchain_only_model
        callbacks = [StreamingGradioCallbackHandler()]
        streamer = callbacks[0] if stream_output else None
        max_max_tokens = tokenizer.model_max_length

        from src.llm_exllama import Exllama
        llm = Exllama(streaming=stream_output,
                      model_path=None,
                      model=model,
                      lora_path=None,
                      temperature=temperature,
                      top_k=top_k,
                      top_p=top_p,
                      typical=.7,
                      beams=1,
                      # beam_length = 40,
                      stop_sequences=prompter.stop_sequences,
                      callbacks=callbacks,
                      verbose=verbose,
                      max_seq_len=max_max_tokens,
                      fused_attn=False,
                      # alpha_value = 1.0, #For use with any models
                      # compress_pos_emb = 4.0, #For use with superhot
                      # set_auto_map = "3, 2" #Gpu split, this will split 3gigs/2gigs
                      prompter=prompter,
                      context=context,
                      iinput=iinput,
                      )
    else:
        async_output = False  # FIXME: not implemented yet
        if model is None:
            # only used if didn't pass model in
            assert tokenizer is None
            prompt_type = 'human_bot'
            if model_name is None:
                model_name = 'h2oai/h2ogpt-oasst1-512-12b'
                # model_name = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
                # model_name = 'h2oai/h2ogpt-oasst1-512-20b'
            inference_server = ''
            model, tokenizer, device = get_model(load_8bit=True, base_model=model_name,
                                                 inference_server=inference_server, gpu_id=0)

        max_max_tokens = tokenizer.model_max_length
        only_new_text = True
        gen_kwargs = dict(do_sample=do_sample,
                          temperature=temperature,
                          top_k=top_k,
                          top_p=top_p,
                          num_beams=num_beams,
                          max_new_tokens=max_new_tokens,
                          min_new_tokens=min_new_tokens,
                          early_stopping=early_stopping,
                          max_time=max_time,
                          repetition_penalty=repetition_penalty,
                          num_return_sequences=num_return_sequences,
                          return_full_text=not only_new_text,
                          handle_long_generation=None)
        assert len(set(gen_hyper).difference(gen_kwargs.keys())) == 0

        if stream_output:
            skip_prompt = only_new_text
            from gen import H2OTextIteratorStreamer
            decoder_kwargs = {}
            streamer = H2OTextIteratorStreamer(tokenizer, skip_prompt=skip_prompt, block=False, **decoder_kwargs)
            gen_kwargs.update(dict(streamer=streamer))
        else:
            streamer = None

        from h2oai_pipeline import H2OTextGenerationPipeline
        pipe = H2OTextGenerationPipeline(model=model, use_prompter=True,
                                         prompter=prompter,
                                         context=context,
                                         iinput=iinput,
                                         prompt_type=prompt_type,
                                         prompt_dict=prompt_dict,
                                         sanitize_bot_response=sanitize_bot_response,
                                         chat=False, stream_output=stream_output,
                                         tokenizer=tokenizer,
                                         # leave some room for 1 paragraph, even if min_new_tokens=0
                                         max_input_tokens=max_max_tokens - max(min_new_tokens, 256),
                                         base_model=model_name,
                                         **gen_kwargs)
        # pipe.task = "text-generation"
        # below makes it listen only to our prompt removal,
        # not built in prompt removal that is less general and not specific for our model
        pipe.task = "text2text-generation"

        from langchain.llms import HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=pipe)
    return llm, model_name, streamer, prompt_type, async_output, only_new_text


def get_device_dtype():
    # torch.device("cuda") leads to cuda:x cuda:y mismatches for multi-GPU consistently
    import torch
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available else 0
    device = 'cpu' if n_gpus == 0 else 'cuda'
    # from utils import NullContext
    # context_class = NullContext if n_gpus > 1 or n_gpus == 0 else context_class
    context_class = torch.device
    torch_dtype = torch.float16 if device == 'cuda' else torch.float32
    return device, torch_dtype, context_class


def get_wiki_data(title, first_paragraph_only, text_limit=None, take_head=True):
    """
    Get wikipedia data from online
    :param title:
    :param first_paragraph_only:
    :param text_limit:
    :param take_head:
    :return:
    """
    filename = 'wiki_%s_%s_%s_%s.data' % (first_paragraph_only, title, text_limit, take_head)
    url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&titles={title}"
    if first_paragraph_only:
        url += "&exintro=1"
    import json
    if not os.path.isfile(filename):
        data = requests.get(url).json()
        json.dump(data, open(filename, 'wt'))
    else:
        data = json.load(open(filename, "rt"))
    page_content = list(data["query"]["pages"].values())[0]["extract"]
    if take_head is not None and text_limit is not None:
        page_content = page_content[:text_limit] if take_head else page_content[-text_limit:]
    title_url = str(title).replace(' ', '_')
    return Document(
        page_content=page_content,
        metadata={"source": f"https://en.wikipedia.org/wiki/{title_url}"},
    )


def get_wiki_sources(first_para=True, text_limit=None):
    """
    Get specific named sources from wikipedia
    :param first_para:
    :param text_limit:
    :return:
    """
    default_wiki_sources = ['Unix', 'Microsoft_Windows', 'Linux']
    wiki_sources = list(os.getenv('WIKI_SOURCES', default_wiki_sources))
    return [get_wiki_data(x, first_para, text_limit=text_limit) for x in wiki_sources]


def get_github_docs(repo_owner, repo_name):
    """
    Access github from specific repo
    :param repo_owner:
    :param repo_name:
    :return:
    """
    with tempfile.TemporaryDirectory() as d:
        subprocess.check_call(
            f"git clone --depth 1 https://github.com/{repo_owner}/{repo_name}.git .",
            cwd=d,
            shell=True,
        )
        git_sha = (
            subprocess.check_output("git rev-parse HEAD", shell=True, cwd=d)
            .decode("utf-8")
            .strip()
        )
        repo_path = pathlib.Path(d)
        markdown_files = list(repo_path.glob("*/*.md")) + list(
            repo_path.glob("*/*.mdx")
        )
        for markdown_file in markdown_files:
            with open(markdown_file, "r") as f:
                relative_path = markdown_file.relative_to(repo_path)
                github_url = f"https://github.com/{repo_owner}/{repo_name}/blob/{git_sha}/{relative_path}"
                yield Document(page_content=f.read(), metadata={"source": github_url})


def get_dai_pickle(dest="."):
    from huggingface_hub import hf_hub_download
    # True for case when locally already logged in with correct token, so don't have to set key
    token = os.getenv('HUGGING_FACE_HUB_TOKEN', True)
    path_to_zip_file = hf_hub_download('h2oai/dai_docs', 'dai_docs.pickle', token=token, repo_type='dataset')
    shutil.copy(path_to_zip_file, dest)


def get_dai_docs(from_hf=False, get_pickle=True):
    """
    Consume DAI documentation, or consume from public pickle
    :param from_hf: get DAI docs from HF, then generate pickle for later use by LangChain
    :param get_pickle: Avoid raw DAI docs, just get pickle directly from HF
    :return:
    """
    import pickle

    if get_pickle:
        get_dai_pickle()

    dai_store = 'dai_docs.pickle'
    dst = "working_dir_docs"
    if not os.path.isfile(dai_store):
        from create_data import setup_dai_docs
        dst = setup_dai_docs(dst=dst, from_hf=from_hf)

        import glob
        files = list(glob.glob(os.path.join(dst, '*rst'), recursive=True))

        basedir = os.path.abspath(os.getcwd())
        from create_data import rst_to_outputs
        new_outputs = rst_to_outputs(files)
        os.chdir(basedir)

        pickle.dump(new_outputs, open(dai_store, 'wb'))
    else:
        new_outputs = pickle.load(open(dai_store, 'rb'))

    sources = []
    for line, file in new_outputs:
        # gradio requires any linked file to be with app.py
        sym_src = os.path.abspath(os.path.join(dst, file))
        sym_dst = os.path.abspath(os.path.join(os.getcwd(), file))
        if os.path.lexists(sym_dst):
            os.remove(sym_dst)
        os.symlink(sym_src, sym_dst)
        itm = Document(page_content=line, metadata={"source": file})
        # NOTE: yield has issues when going into db, loses metadata
        # yield itm
        sources.append(itm)
    return sources


def get_supported_types():
    non_image_types0 = ["pdf", "txt", "csv", "toml", "py", "rst", "rtf",
                        "md",
                        "html", "mhtml", "htm",
                        "enex", "eml", "epub", "odt", "pptx", "ppt",
                        "zip", "urls",
                        ]
    # "msg",  GPL3

    video_types0 = ['WEBM',
                    'MPG', 'MP2', 'MPEG', 'MPE', '.PV',
                    'OGG',
                    'MP4', 'M4P', 'M4V',
                    'AVI', 'WMV',
                    'MOV', 'QT',
                    'FLV', 'SWF',
                    'AVCHD']
    video_types0 = [x.lower() for x in video_types0]
    if have_pillow:
        from PIL import Image
        exts = Image.registered_extensions()
        image_types0 = {ex for ex, f in exts.items() if f in Image.OPEN if ex not in video_types0 + non_image_types0}
        image_types0 = sorted(image_types0)
        image_types0 = [x[1:] if x.startswith('.') else x for x in image_types0]
    else:
        image_types0 = []
    return non_image_types0, image_types0, video_types0


non_image_types, image_types, video_types = get_supported_types()
set_image_types = set(image_types)

if have_libreoffice or True:
    # or True so it tries to load, e.g. on MAC/Windows, even if don't have libreoffice since works without that
    non_image_types.extend(["docx", "doc", "xls", "xlsx"])

file_types = non_image_types + image_types


def add_meta(docs1, file, headsize):
    if os.path.isfile(file):
        file_extension = pathlib.Path(file).suffix
        hashid = hash_file(file)
    else:
        file_extension = str(file)  # not file, just show full thing
        hashid = get_sha(file)
    doc_hash = str(uuid.uuid4())[:10]
    if not isinstance(docs1, (list, tuple, types.GeneratorType)):
        docs1 = [docs1]
    [x.metadata.update(dict(input_type=file_extension,
                            date=str(datetime.now()),
                            hashid=hashid,
                            doc_hash=doc_hash,
                            head=x.page_content[:headsize].strip())) for x in docs1]


def file_to_doc(file, base_path=None, verbose=False, fail_any_exception=False,
                chunk=True, chunk_size=512, n_jobs=-1,
                is_url=False, is_txt=False,
                enable_captions=True,
                captions_model=None,
                enable_ocr=False, enable_pdf_ocr='auto', caption_loader=None,
                headsize=50,
                db_type=None,
                selected_file_types=None):
    if selected_file_types is not None:
        set_image_types1 = set_image_types.intersection(set(selected_file_types))
    else:
        set_image_types1 = set_image_types

    assert db_type is not None
    chunk_sources = functools.partial(_chunk_sources, chunk=chunk, chunk_size=chunk_size, db_type=db_type)
    if file is None:
        if fail_any_exception:
            raise RuntimeError("Unexpected None file")
        else:
            return []
    doc1 = []  # in case no support, or disabled support
    if base_path is None and not is_txt and not is_url:
        # then assume want to persist but don't care which path used
        # can't be in base_path
        dir_name = os.path.dirname(file)
        base_name = os.path.basename(file)
        # if from gradio, will have its own temp uuid too, but that's ok
        base_name = sanitize_filename(base_name) + "_" + str(uuid.uuid4())[:10]
        base_path = os.path.join(dir_name, base_name)
    if is_url:
        file = file.strip()  # in case accidental spaces in front or at end
        file_lower = file.lower()
        case1 = file_lower.startswith('arxiv:') and len(file_lower.split('arxiv:')) == 2
        case2 = file_lower.startswith('https://arxiv.org/abs') and len(file_lower.split('https://arxiv.org/abs')) == 2
        case3 = file_lower.startswith('http://arxiv.org/abs') and len(file_lower.split('http://arxiv.org/abs')) == 2
        case4 = file_lower.startswith('arxiv.org/abs/') and len(file_lower.split('arxiv.org/abs/')) == 2
        if case1 or case2 or case3 or case4:
            if case1:
                query = file.lower().split('arxiv:')[1].strip()
            elif case2:
                query = file.lower().split('https://arxiv.org/abs/')[1].strip()
            elif case2:
                query = file.lower().split('http://arxiv.org/abs/')[1].strip()
            elif case3:
                query = file.lower().split('arxiv.org/abs/')[1].strip()
            else:
                raise RuntimeError("Unexpected arxiv error for %s" % file)
            if have_arxiv:
                docs1 = ArxivLoader(query=query, load_max_docs=20, load_all_available_meta=True).load()
                # ensure string, sometimes None
                [[x.metadata.update({k: str(v)}) for k, v in x.metadata.items()] for x in docs1]
                query_url = f"https://arxiv.org/abs/{query}"
                [x.metadata.update(
                    dict(source=x.metadata.get('entry_id', query_url), query=query_url,
                         input_type='arxiv', head=x.metadata.get('Title', ''), date=str(datetime.now))) for x in
                    docs1]
            else:
                docs1 = []
        else:
            if not (file.startswith("http://") or file.startswith("file://") or file.startswith("https://")):
                file = 'http://' + file
            docs1 = []
            do_unstructured = only_unstructured_urls or not (only_selenium or only_playwright)
            do_playwright = have_playwright and (only_playwright or not (only_selenium or only_unstructured_urls))
            do_selenium = have_selenium and (only_selenium or not (only_playwright or only_unstructured_urls))

            if do_unstructured:
                docs1 = UnstructuredURLLoader(urls=[file]).load()
                docs1 = [x for x in docs1 if x.page_content]
            if len(docs1) == 0 and do_playwright:
                # then something went wrong, try another loader:
                from langchain.document_loaders import PlaywrightURLLoader
                docs1 = PlaywrightURLLoader(urls=[file]).load()
                docs1 = [x for x in docs1 if x.page_content]
            if len(docs1) == 0 and do_selenium:
                # then something went wrong, try another loader:
                # but requires Chrome binary, else get: selenium.common.exceptions.WebDriverException:
                # Message: unknown error: cannot find Chrome binary
                from langchain.document_loaders import SeleniumURLLoader
                from selenium.common.exceptions import WebDriverException
                try:
                    docs1 = SeleniumURLLoader(urls=[file]).load()
                    docs1 = [x for x in docs1 if x.page_content]
                except WebDriverException as e:
                    print("No web driver: %s" % str(e), flush=True)
            [x.metadata.update(dict(input_type='url', date=str(datetime.now))) for x in docs1]
        add_meta(docs1, file, headsize)
        docs1 = clean_doc(docs1)
        doc1 = chunk_sources(docs1)
    elif is_txt:
        base_path = "user_paste"
        base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
        source_file = os.path.join(base_path, "_%s" % str(uuid.uuid4())[:10])
        with open(source_file, "wt") as f:
            f.write(file)
        metadata = dict(source=source_file, date=str(datetime.now()), input_type='pasted txt')
        doc1 = Document(page_content=file, metadata=metadata)
        add_meta(doc1, file, headsize)
        doc1 = clean_doc(doc1)
    elif file.lower().endswith('.html') or file.lower().endswith('.mhtml') or file.lower().endswith('.htm'):
        docs1 = UnstructuredHTMLLoader(file_path=file).load()
        add_meta(docs1, file, headsize)
        docs1 = clean_doc(docs1)
        doc1 = chunk_sources(docs1, language=Language.HTML)
    elif (file.lower().endswith('.docx') or file.lower().endswith('.doc')) and (have_libreoffice or True):
        docs1 = UnstructuredWordDocumentLoader(file_path=file).load()
        add_meta(docs1, file, headsize)
        doc1 = chunk_sources(docs1)
    elif (file.lower().endswith('.xlsx') or file.lower().endswith('.xls')) and (have_libreoffice or True):
        docs1 = UnstructuredExcelLoader(file_path=file).load()
        add_meta(docs1, file, headsize)
        doc1 = chunk_sources(docs1)
    elif file.lower().endswith('.odt'):
        docs1 = UnstructuredODTLoader(file_path=file).load()
        add_meta(docs1, file, headsize)
        doc1 = chunk_sources(docs1)
    elif file.lower().endswith('pptx') or file.lower().endswith('ppt'):
        docs1 = UnstructuredPowerPointLoader(file_path=file).load()
        add_meta(docs1, file, headsize)
        docs1 = clean_doc(docs1)
        doc1 = chunk_sources(docs1)
    elif file.lower().endswith('.txt'):
        # use UnstructuredFileLoader ?
        docs1 = TextLoader(file, encoding="utf8", autodetect_encoding=True).load()
        # makes just one, but big one
        doc1 = chunk_sources(docs1)
        doc1 = clean_doc(doc1)
        add_meta(doc1, file, headsize)
    elif file.lower().endswith('.rtf'):
        docs1 = UnstructuredRTFLoader(file).load()
        add_meta(docs1, file, headsize)
        doc1 = chunk_sources(docs1)
    elif file.lower().endswith('.md'):
        docs1 = UnstructuredMarkdownLoader(file).load()
        add_meta(docs1, file, headsize)
        docs1 = clean_doc(docs1)
        doc1 = chunk_sources(docs1, language=Language.MARKDOWN)
    elif file.lower().endswith('.enex'):
        docs1 = EverNoteLoader(file).load()
        add_meta(doc1, file, headsize)
        doc1 = chunk_sources(docs1)
    elif file.lower().endswith('.epub'):
        docs1 = UnstructuredEPubLoader(file).load()
        add_meta(docs1, file, headsize)
        doc1 = chunk_sources(docs1)
    elif any(file.lower().endswith(x) for x in set_image_types1):
        docs1 = []
        if have_tesseract and enable_ocr:
            # OCR, somewhat works, but not great
            docs1.extend(UnstructuredImageLoader(file).load())
            add_meta(docs1, file, headsize)
        if enable_captions:
            # BLIP
            if caption_loader is not None and not isinstance(caption_loader, (str, bool)):
                # assumes didn't fork into this process with joblib, else can deadlock
                caption_loader.set_image_paths([file])
                docs1c = caption_loader.load()
                add_meta(docs1c, file, headsize)
                docs1.extend(docs1c)
            else:
                from image_captions import H2OImageCaptionLoader
                caption_loader = H2OImageCaptionLoader(caption_gpu=caption_loader == 'gpu',
                                                       blip_model=captions_model,
                                                       blip_processor=captions_model)
                caption_loader.set_image_paths([file])
                docs1c = caption_loader.load()
                add_meta(docs1c, file, headsize)
                docs1.extend(docs1c)
            # caption didn't set source, so fix-up meta
            for doci in docs1:
                doci.metadata['source'] = doci.metadata['image_path']
                doci.metadata['hashid'] = hash_file(doci.metadata['source'])
            if docs1:
                doc1 = chunk_sources(docs1)
    elif file.lower().endswith('.msg'):
        raise RuntimeError("Not supported, GPL3 license")
        # docs1 = OutlookMessageLoader(file).load()
        # docs1[0].metadata['source'] = file
    elif file.lower().endswith('.eml'):
        try:
            docs1 = UnstructuredEmailLoader(file).load()
            add_meta(docs1, file, headsize)
            doc1 = chunk_sources(docs1)
        except ValueError as e:
            if 'text/html content not found in email' in str(e):
                pass
            else:
                raise
        doc1 = [x for x in doc1 if x.page_content]
        if len(doc1) == 0:
            # e.g. plain/text dict key exists, but not
            # doc1 = TextLoader(file, encoding="utf8").load()
            docs1 = UnstructuredEmailLoader(file, content_source="text/plain").load()
            docs1 = [x for x in docs1 if x.page_content]
            add_meta(docs1, file, headsize)
            doc1 = chunk_sources(docs1)
    # elif file.lower().endswith('.gcsdir'):
    #    doc1 = GCSDirectoryLoader(project_name, bucket, prefix).load()
    # elif file.lower().endswith('.gcsfile'):
    # doc1 = GCSFileLoader(project_name, bucket, blob).load()
    elif file.lower().endswith('.rst'):
        with open(file, "r") as f:
            doc1 = Document(page_content=f.read(), metadata={"source": file})
        add_meta(doc1, file, headsize)
        doc1 = chunk_sources(doc1, language=Language.RST)
    elif file.lower().endswith('.pdf'):
        pdf_class_name = os.getenv('PDF_CLASS_NAME', 'PyMuPDFParser')
        doc1 = []
        handled = False
        e = None
        if have_pymupdf and pdf_class_name == 'PyMuPDFParser':
            # GPL, only use if installed
            from langchain.document_loaders import PyMuPDFLoader
            # load() still chunks by pages, but every page has title at start to help
            try:
                doc1 = PyMuPDFLoader(file).load()
            except BaseException as e0:
                print("PyMuPDFLoader: %s" % str(e0), flush=True)
                e = e0
            # remove empty documents
            handled |= len(doc1) > 0
            doc1 = [x for x in doc1 if x.page_content]
            doc1 = clean_doc(doc1)
        if len(doc1) == 0:
            try:
                doc1 = UnstructuredPDFLoader(file).load()
            except BaseException as e0:
                print("UnstructuredPDFLoader: %s" % str(e0), flush=True)
                e = e0
            handled |= len(doc1) > 0
            # remove empty documents
            doc1 = [x for x in doc1 if x.page_content]
            # seems to not need cleaning in most cases
        if len(doc1) == 0:
            # open-source fallback
            # load() still chunks by pages, but every page has title at start to help
            try:
                doc1 = PyPDFLoader(file).load()
            except BaseException as e0:
                print("PyPDFLoader: %s" % str(e0), flush=True)
                e = e0
            handled |= len(doc1) > 0
            # remove empty documents
            doc1 = [x for x in doc1 if x.page_content]
            doc1 = clean_doc(doc1)
        if have_pymupdf and len(doc1) == 0:
            # GPL, only use if installed
            from langchain.document_loaders import PyMuPDFLoader
            # load() still chunks by pages, but every page has title at start to help
            try:
                doc1 = PyMuPDFLoader(file).load()
            except BaseException as e0:
                print("PyMuPDFLoader: %s" % str(e0), flush=True)
                e = e0
            handled |= len(doc1) > 0
            # remove empty documents
            doc1 = [x for x in doc1 if x.page_content]
            doc1 = clean_doc(doc1)

        # try treating as html as occurs when scraping websites
        if len(doc1) == 0:
            from bs4 import BeautifulSoup
            with open(file, "rt") as f:
                try:
                    is_html = bool(BeautifulSoup(f.read(), "html.parser").find())
                except:  # FIXME
                    is_html = False
            if is_html:
                file_url = 'file://' + file
                doc1 = UnstructuredURLLoader(urls=[file_url]).load()
                doc1 = [x for x in doc1 if x.page_content]
        if len(doc1) == 0 and enable_pdf_ocr == 'auto' or enable_pdf_ocr == 'on':
            # try OCR in end since slowest, but works on pure image pages well
            doc1 = UnstructuredPDFLoader(file, strategy='ocr_only').load()
            handled |= len(doc1) > 0
            # remove empty documents
            doc1 = [x for x in doc1 if x.page_content]
            # seems to not need cleaning in most cases
        # Some PDFs return nothing or junk from PDFMinerLoader
        if len(doc1) == 0:
            # if literally nothing, show failed to parse so user knows, since unlikely nothing in PDF at all.
            if handled:
                raise ValueError("%s had no valid text, but meta data was parsed" % file)
            else:
                raise ValueError("%s had no valid text and no meta data was parsed: %s" % (file, str(e)))
        add_meta(doc1, file, headsize)
        doc1 = chunk_sources(doc1)
    elif file.lower().endswith('.csv'):
        doc1 = CSVLoader(file).load()
        add_meta(doc1, file, headsize)
        if isinstance(doc1, list):
            # each row is a Document, identify
            [x.metadata.update(dict(chunk_id=chunk_id)) for chunk_id, x in enumerate(doc1)]
            if db_type == 'chroma':
                # then separate summarize list
                sdoc1 = clone_documents(doc1)
                [x.metadata.update(dict(chunk_id=-1)) for chunk_id, x in enumerate(sdoc1)]
                doc1 = sdoc1 + doc1
    elif file.lower().endswith('.py'):
        doc1 = PythonLoader(file).load()
        add_meta(doc1, file, headsize)
        doc1 = chunk_sources(doc1, language=Language.PYTHON)
    elif file.lower().endswith('.toml'):
        doc1 = TomlLoader(file).load()
        add_meta(doc1, file, headsize)
        doc1 = chunk_sources(doc1)
    elif file.lower().endswith('.urls'):
        with open(file, "r") as f:
            urls = f.readlines()
            # recurse
            doc1 = path_to_docs(None, url=urls, verbose=verbose, fail_any_exception=fail_any_exception, n_jobs=n_jobs,
                                db_type=db_type)
    elif file.lower().endswith('.zip'):
        with zipfile.ZipFile(file, 'r') as zip_ref:
            # don't put into temporary path, since want to keep references to docs inside zip
            # so just extract in path where
            zip_ref.extractall(base_path)
            # recurse
            doc1 = path_to_docs(base_path, verbose=verbose, fail_any_exception=fail_any_exception, n_jobs=n_jobs,
                                db_type=db_type)
    else:
        raise RuntimeError("No file handler for %s" % os.path.basename(file))

    # allow doc1 to be list or not.
    if not isinstance(doc1, list):
        # If not list, did not chunk yet, so chunk now
        docs = chunk_sources([doc1])
    elif isinstance(doc1, list) and len(doc1) == 1:
        # if list of length one, don't trust and chunk it, chunk_id's will still be correct if repeat
        docs = chunk_sources(doc1)
    else:
        docs = doc1

    assert isinstance(docs, list)
    return docs


def path_to_doc1(file, verbose=False, fail_any_exception=False, return_file=True,
                 chunk=True, chunk_size=512,
                 n_jobs=-1,
                 is_url=False, is_txt=False,
                 enable_captions=True,
                 captions_model=None,
                 enable_ocr=False, enable_pdf_ocr='auto', caption_loader=None,
                 db_type=None,
                 selected_file_types=None):
    assert db_type is not None
    if verbose:
        if is_url:
            print("Ingesting URL: %s" % file, flush=True)
        elif is_txt:
            print("Ingesting Text: %s" % file, flush=True)
        else:
            print("Ingesting file: %s" % file, flush=True)
    res = None
    try:
        # don't pass base_path=path, would infinitely recurse
        res = file_to_doc(file, base_path=None, verbose=verbose, fail_any_exception=fail_any_exception,
                          chunk=chunk, chunk_size=chunk_size,
                          n_jobs=n_jobs,
                          is_url=is_url, is_txt=is_txt,
                          enable_captions=enable_captions,
                          captions_model=captions_model,
                          enable_ocr=enable_ocr,
                          enable_pdf_ocr=enable_pdf_ocr,
                          caption_loader=caption_loader,
                          db_type=db_type,
                          selected_file_types=selected_file_types)
    except BaseException as e:
        print("Failed to ingest %s due to %s" % (file, traceback.format_exc()))
        if fail_any_exception:
            raise
        else:
            exception_doc = Document(
                page_content='',
                metadata={"source": file, "exception": '%s Exception: %s' % (file, str(e)),
                          "traceback": traceback.format_exc()})
            res = [exception_doc]
    if verbose:
        if is_url:
            print("DONE Ingesting URL: %s" % file, flush=True)
        elif is_txt:
            print("DONE Ingesting Text: %s" % file, flush=True)
        else:
            print("DONE Ingesting file: %s" % file, flush=True)
    if return_file:
        base_tmp = "temp_path_to_doc1"
        if not os.path.isdir(base_tmp):
            base_tmp = makedirs(base_tmp, exist_ok=True, tmp_ok=True, use_base=True)
        filename = os.path.join(base_tmp, str(uuid.uuid4()) + ".tmp.pickle")
        with open(filename, 'wb') as f:
            pickle.dump(res, f)
        return filename
    return res


def path_to_docs(path_or_paths, verbose=False, fail_any_exception=False, n_jobs=-1,
                 chunk=True, chunk_size=512,
                 url=None, text=None,
                 enable_captions=True,
                 captions_model=None,
                 caption_loader=None,
                 enable_ocr=False,
                 enable_pdf_ocr='auto',
                 existing_files=[],
                 existing_hash_ids={},
                 db_type=None,
                 selected_file_types=None,
                 ):
    if verbose:
        print("BEGIN Consuming path_or_paths=%s url=%s text=%s" % (path_or_paths, url, text), flush=True)
    if selected_file_types is not None:
        non_image_types1 = [x for x in non_image_types if x in selected_file_types]
        image_types1 = [x for x in image_types if x in selected_file_types]
    else:
        non_image_types1 = non_image_types.copy()
        image_types1 = image_types.copy()

    assert db_type is not None
    # path_or_paths could be str, list, tuple, generator
    globs_image_types = []
    globs_non_image_types = []
    if not path_or_paths and not url and not text:
        return []
    elif url:
        url = get_list_or_str(url)
        globs_non_image_types = url if isinstance(url, (list, tuple, types.GeneratorType)) else [url]
    elif text:
        globs_non_image_types = text if isinstance(text, (list, tuple, types.GeneratorType)) else [text]
    elif isinstance(path_or_paths, str) and os.path.isdir(path_or_paths):
        # single path, only consume allowed files
        path = path_or_paths
        # Below globs should match patterns in file_to_doc()
        [globs_image_types.extend(glob.glob(os.path.join(path, "./**/*.%s" % ftype), recursive=True))
         for ftype in image_types1]
        globs_image_types = [os.path.normpath(x) for x in globs_image_types]
        [globs_non_image_types.extend(glob.glob(os.path.join(path, "./**/*.%s" % ftype), recursive=True))
         for ftype in non_image_types1]
        globs_non_image_types = [os.path.normpath(x) for x in globs_non_image_types]
    else:
        if isinstance(path_or_paths, str):
            if os.path.isfile(path_or_paths) or os.path.isdir(path_or_paths):
                path_or_paths = [path_or_paths]
            else:
                # path was deleted etc.
                return []
        # list/tuple of files (consume what can, and exception those that selected but cannot consume so user knows)
        assert isinstance(path_or_paths, (list, tuple, types.GeneratorType)), \
            "Wrong type for path_or_paths: %s %s" % (path_or_paths, type(path_or_paths))
        # reform out of allowed types
        globs_image_types.extend(
            flatten_list([[os.path.normpath(x) for x in path_or_paths if x.endswith(y)] for y in image_types1]))
        # could do below:
        # globs_non_image_types = flatten_list([[x for x in path_or_paths if x.endswith(y)] for y in non_image_types1])
        # But instead, allow fail so can collect unsupported too
        set_globs_image_types = set(globs_image_types)
        globs_non_image_types.extend([os.path.normpath(x) for x in path_or_paths if x not in set_globs_image_types])

    # filter out any files to skip (e.g. if already processed them)
    # this is easy, but too aggressive in case a file changed, so parent probably passed existing_files=[]
    assert not existing_files, "DEV: assume not using this approach"
    if existing_files:
        set_skip_files = set(existing_files)
        globs_image_types = [x for x in globs_image_types if x not in set_skip_files]
        globs_non_image_types = [x for x in globs_non_image_types if x not in set_skip_files]
    if existing_hash_ids:
        # assume consistent with add_meta() use of hash_file(file)
        # also assume consistent with get_existing_hash_ids for dict creation
        # assume hashable values
        existing_hash_ids_set = set(existing_hash_ids.items())
        hash_ids_all_image = set({x: hash_file(x) for x in globs_image_types}.items())
        hash_ids_all_non_image = set({x: hash_file(x) for x in globs_non_image_types}.items())
        # don't use symmetric diff.  If file is gone, ignore and don't remove or something
        #  just consider existing files (key) having new hash or not (value)
        new_files_image = set(dict(hash_ids_all_image - existing_hash_ids_set).keys())
        new_files_non_image = set(dict(hash_ids_all_non_image - existing_hash_ids_set).keys())
        globs_image_types = [x for x in globs_image_types if x in new_files_image]
        globs_non_image_types = [x for x in globs_non_image_types if x in new_files_non_image]

    # could use generator, but messes up metadata handling in recursive case
    if caption_loader and not isinstance(caption_loader, (bool, str)) and \
            caption_loader.device != 'cpu' or \
            get_device() == 'cuda':
        # to avoid deadlocks, presume was preloaded and so can't fork due to cuda context
        n_jobs_image = 1
    else:
        n_jobs_image = n_jobs

    return_file = True  # local choice
    is_url = url is not None
    is_txt = text is not None
    kwargs = dict(verbose=verbose, fail_any_exception=fail_any_exception,
                  return_file=return_file,
                  chunk=chunk, chunk_size=chunk_size,
                  n_jobs=n_jobs,
                  is_url=is_url,
                  is_txt=is_txt,
                  enable_captions=enable_captions,
                  captions_model=captions_model,
                  caption_loader=caption_loader,
                  enable_ocr=enable_ocr,
                  enable_pdf_ocr=enable_pdf_ocr,
                  db_type=db_type,
                  selected_file_types=selected_file_types,
                  )

    if n_jobs != 1 and len(globs_non_image_types) > 1:
        # avoid nesting, e.g. upload 1 zip and then inside many files
        # harder to handle if upload many zips with many files, inner parallel one will be disabled by joblib
        documents = ProgressParallel(n_jobs=n_jobs, verbose=10 if verbose else 0, backend='multiprocessing')(
            delayed(path_to_doc1)(file, **kwargs) for file in globs_non_image_types
        )
    else:
        documents = [path_to_doc1(file, **kwargs) for file in tqdm(globs_non_image_types)]

    # do images separately since can't fork after cuda in parent, so can't be parallel
    if n_jobs_image != 1 and len(globs_image_types) > 1:
        # avoid nesting, e.g. upload 1 zip and then inside many files
        # harder to handle if upload many zips with many files, inner parallel one will be disabled by joblib
        image_documents = ProgressParallel(n_jobs=n_jobs, verbose=10 if verbose else 0, backend='multiprocessing')(
            delayed(path_to_doc1)(file, **kwargs) for file in globs_image_types
        )
    else:
        image_documents = [path_to_doc1(file, **kwargs) for file in tqdm(globs_image_types)]

    # add image docs in
    documents += image_documents

    if return_file:
        # then documents really are files
        files = documents.copy()
        documents = []
        for fil in files:
            with open(fil, 'rb') as f:
                documents.extend(pickle.load(f))
            # remove temp pickle
            remove(fil)
    else:
        documents = reduce(concat, documents)

    if verbose:
        print("END consuming path_or_paths=%s url=%s text=%s" % (path_or_paths, url, text), flush=True)
    return documents


def prep_langchain(persist_directory,
                   load_db_if_exists,
                   db_type, use_openai_embedding,
                   langchain_mode, langchain_mode_paths, langchain_mode_types,
                   hf_embedding_model,
                   migrate_embedding_model,
                   n_jobs=-1, kwargs_make_db={}):
    """
    do prep first time, involving downloads
    # FIXME: Add github caching then add here
    :return:
    """
    if os.getenv("HARD_ASSERTS"):
        assert langchain_mode not in ['MyData'], "Should not prep scratch/personal data"

    db_dir_exists = os.path.isdir(persist_directory)
    user_path = langchain_mode_paths.get(langchain_mode)

    if db_dir_exists and user_path is None:
        print("Prep: persist_directory=%s exists, using" % persist_directory, flush=True)
        db, use_openai_embedding, hf_embedding_model = \
            get_existing_db(None, persist_directory, load_db_if_exists,
                            db_type, use_openai_embedding,
                            langchain_mode, langchain_mode_paths, langchain_mode_types,
                            hf_embedding_model, migrate_embedding_model)
    else:
        if db_dir_exists and user_path is not None:
            print("Prep: persist_directory=%s exists, user_path=%s passed, adding any changed or new documents" % (
                persist_directory, user_path), flush=True)
        elif not db_dir_exists:
            print("Prep: persist_directory=%s does not exist, regenerating" % persist_directory, flush=True)
        db = None
        if langchain_mode in ['DriverlessAI docs']:
            # FIXME: Could also just use dai_docs.pickle directly and upload that
            get_dai_docs(from_hf=True)

        if langchain_mode in ['wiki']:
            get_wiki_sources(first_para=kwargs_make_db['first_para'], text_limit=kwargs_make_db['text_limit'])

        langchain_kwargs = kwargs_make_db.copy()
        langchain_kwargs.update(locals())
        db, num_new_sources, new_sources_metadata = make_db(**langchain_kwargs)

    return db


import posthog

posthog.disabled = True


class FakeConsumer(object):
    def __init__(self, *args, **kwargs):
        pass

    def run(self):
        pass

    def pause(self):
        pass

    def upload(self):
        pass

    def next(self):
        pass

    def request(self, batch):
        pass


posthog.Consumer = FakeConsumer


def check_update_chroma_embedding(db, use_openai_embedding,
                                  hf_embedding_model, migrate_embedding_model,
                                  langchain_mode, langchain_mode_paths, langchain_mode_types):
    changed_db = False
    if load_embed(db=db) not in [(True, use_openai_embedding, hf_embedding_model),
                                 (False, use_openai_embedding, hf_embedding_model)]:
        print("Detected new embedding, updating db: %s" % langchain_mode, flush=True)
        # handle embedding changes
        db_get = get_documents(db)
        sources = [Document(page_content=result[0], metadata=result[1] or {})
                   for result in zip(db_get['documents'], db_get['metadatas'])]
        # delete index, has to be redone
        persist_directory = db._persist_directory
        shutil.move(persist_directory, persist_directory + "_" + str(uuid.uuid4()) + ".bak")
        db_type = 'chroma'
        load_db_if_exists = False
        db = get_db(sources, use_openai_embedding=use_openai_embedding, db_type=db_type,
                    persist_directory=persist_directory, load_db_if_exists=load_db_if_exists,
                    langchain_mode=langchain_mode,
                    langchain_mode_paths=langchain_mode_paths,
                    langchain_mode_types=langchain_mode_types,
                    collection_name=None,
                    hf_embedding_model=hf_embedding_model,
                    migrate_embedding_model=migrate_embedding_model,
                    )
        changed_db = True
        print("Done updating db for new embedding: %s" % langchain_mode, flush=True)

    return db, changed_db


def get_existing_db(db, persist_directory,
                    load_db_if_exists, db_type, use_openai_embedding,
                    langchain_mode, langchain_mode_paths, langchain_mode_types,
                    hf_embedding_model,
                    migrate_embedding_model,
                    verbose=False, check_embedding=True, migrate_meta=True):
    if load_db_if_exists and db_type == 'chroma' and os.path.isdir(persist_directory) and os.path.isdir(
            os.path.join(persist_directory, 'index')):
        if db is None:
            if verbose:
                print("DO Loading db: %s" % langchain_mode, flush=True)
            got_embedding, use_openai_embedding0, hf_embedding_model0 = load_embed(persist_directory=persist_directory)
            if got_embedding:
                use_openai_embedding, hf_embedding_model = use_openai_embedding0, hf_embedding_model0
            embedding = get_embedding(use_openai_embedding, hf_embedding_model=hf_embedding_model)
            from chromadb.config import Settings
            client_settings = Settings(anonymized_telemetry=False,
                                       chroma_db_impl="duckdb+parquet",
                                       persist_directory=persist_directory)
            db = Chroma(persist_directory=persist_directory, embedding_function=embedding,
                        collection_name=langchain_mode.replace(' ', '_'),
                        client_settings=client_settings)
            if verbose:
                print("DONE Loading db: %s" % langchain_mode, flush=True)
        else:
            if not migrate_embedding_model:
                # OVERRIDE embedding choices if could load embedding info when not migrating
                got_embedding, use_openai_embedding, hf_embedding_model = load_embed(db=db)
            if verbose:
                print("USING already-loaded db: %s" % langchain_mode, flush=True)
        if migrate_meta and db is not None:
            db_documents, db_metadatas = get_docs_and_meta(db, top_k_docs=-1)
            [x.update(dict(chunk_id=x.get('chunk_id', 0))) for x in db_metadatas]

        if check_embedding:
            db_trial, changed_db = check_update_chroma_embedding(db, use_openai_embedding,
                                                                 hf_embedding_model,
                                                                 migrate_embedding_model,
                                                                 langchain_mode,
                                                                 langchain_mode_paths,
                                                                 langchain_mode_types)
            if changed_db:
                db = db_trial
                # only call persist if really changed db, else takes too long for large db
                if db is not None:
                    db.persist()
                    clear_embedding(db)
        save_embed(db, use_openai_embedding, hf_embedding_model)
        return db, use_openai_embedding, hf_embedding_model
    return db, use_openai_embedding, hf_embedding_model


def clear_embedding(db):
    if db is None:
        return
    # don't keep on GPU, wastes memory, push back onto CPU and only put back on GPU once again embed
    try:
        if hasattr(db._embedding_function.client, 'cpu'):
            db._embedding_function.client.cpu()
        clear_torch_cache()
    except RuntimeError as e:
        print("clear_embedding error: %s" % ''.join(traceback.format_tb(e.__traceback__)), flush=True)


def make_db(**langchain_kwargs):
    func_names = list(inspect.signature(_make_db).parameters)
    missing_kwargs = [x for x in func_names if x not in langchain_kwargs]
    defaults_db = {k: v.default for k, v in dict(inspect.signature(run_qa_db).parameters).items()}
    for k in missing_kwargs:
        if k in defaults_db:
            langchain_kwargs[k] = defaults_db[k]
    # final check for missing
    missing_kwargs = [x for x in func_names if x not in langchain_kwargs]
    assert not missing_kwargs, "Missing kwargs for make_db: %s" % missing_kwargs
    # only keep actual used
    langchain_kwargs = {k: v for k, v in langchain_kwargs.items() if k in func_names}
    return _make_db(**langchain_kwargs)


embed_file_string = "embed_%s.lock"


def save_embed(db, use_openai_embedding, hf_embedding_model):
    if hasattr(db, '_persist_directory'):
        name_path = os.path.basename(db._persist_directory)
        base_path = 'locks'
        base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
        with filelock.FileLock(os.path.join(base_path, embed_file_string % name_path)):
            embed_info_file = os.path.join(db._persist_directory, 'embed_info')
            with open(embed_info_file, 'wb') as f:
                pickle.dump((use_openai_embedding, hf_embedding_model), f)
    return use_openai_embedding, hf_embedding_model


def load_embed(db=None, persist_directory=None):
    if persist_directory is None:
        persist_directory = db._persist_directory
    embed_info_file = os.path.join(persist_directory, 'embed_info')
    if os.path.isfile(embed_info_file):
        name_path = os.path.basename(persist_directory)
        base_path = 'locks'
        base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
        with filelock.FileLock(os.path.join(base_path, embed_file_string % name_path)):
            with open(embed_info_file, 'rb') as f:
                use_openai_embedding, hf_embedding_model = pickle.load(f)
            got_embedding = True
    else:
        # migration, assume defaults
        use_openai_embedding, hf_embedding_model = False, "sentence-transformers/all-MiniLM-L6-v2"
        got_embedding = False
    return got_embedding, use_openai_embedding, hf_embedding_model


def get_persist_directory(langchain_mode, langchain_type=None, db1s=None, dbs=None):
    if langchain_mode in [LangChainMode.DISABLED.value, LangChainMode.LLM.value]:
        # not None so join works but will fail to find db
        return '', langchain_type

    userid = get_userid_direct(db1s)
    username = get_username_direct(db1s)

    # sanity for bad code
    assert userid != 'None'
    assert username != 'None'

    dirid = username or userid
    if langchain_type == LangChainTypes.SHARED.value and not dirid:
        dirid = './'  # just to avoid error
    if langchain_type == LangChainTypes.PERSONAL.value and not dirid:
        # e.g. from client when doing transient calls with MyData
        if db1s is None:
            # just trick to get filled locally
            db1s = {LangChainMode.MY_DATA.value: [None, None, None]}
        set_userid_direct(db1s, str(uuid.uuid4()), str(uuid.uuid4()))
        userid = get_userid_direct(db1s)
        username = get_username_direct(db1s)
        dirid = username or userid
        langchain_type = LangChainTypes.PERSONAL.value

    # deal with existing locations
    user_base_dir = os.getenv('USERS_BASE_DIR', 'users')
    persist_directory = os.path.join(user_base_dir, dirid, 'db_dir_%s' % langchain_mode)
    if userid and \
            (os.path.isdir(persist_directory) or
             db1s is not None and langchain_mode in db1s or
             langchain_type == LangChainTypes.PERSONAL.value):
        langchain_type = LangChainTypes.PERSONAL.value
        persist_directory = makedirs(persist_directory, use_base=True)
        return persist_directory, langchain_type

    persist_directory = 'db_dir_%s' % langchain_mode
    if (os.path.isdir(persist_directory) or
            dbs is not None and langchain_mode in dbs or
            langchain_type == LangChainTypes.SHARED.value):
        # ensure consistent
        langchain_type = LangChainTypes.SHARED.value
        persist_directory = makedirs(persist_directory, use_base=True)
        return persist_directory, langchain_type

    # dummy return for prep_langchain() or full personal space
    base_others = 'db_nonusers'
    persist_directory = os.path.join(base_others, 'db_dir_%s' % str(uuid.uuid4()))
    persist_directory = makedirs(persist_directory, use_base=True)
    langchain_type = LangChainTypes.PERSONAL.value
    return persist_directory, langchain_type


def _make_db(use_openai_embedding=False,
             hf_embedding_model=None,
             migrate_embedding_model=False,
             first_para=False, text_limit=None,
             chunk=True, chunk_size=512,
             langchain_mode=None,
             langchain_mode_paths=None,
             langchain_mode_types=None,
             db_type='faiss',
             load_db_if_exists=True,
             db=None,
             n_jobs=-1,
             verbose=False):
    assert hf_embedding_model is not None
    user_path = langchain_mode_paths.get(langchain_mode)
    langchain_type = langchain_mode_types.get(langchain_mode, LangChainTypes.EITHER.value)
    persist_directory, langchain_type = get_persist_directory(langchain_mode, langchain_type=langchain_type)
    langchain_mode_types[langchain_mode] = langchain_type
    # see if can get persistent chroma db
    db_trial, use_openai_embedding, hf_embedding_model = \
        get_existing_db(db, persist_directory, load_db_if_exists, db_type,
                        use_openai_embedding,
                        langchain_mode, langchain_mode_paths, langchain_mode_types,
                        hf_embedding_model, migrate_embedding_model, verbose=verbose)
    if db_trial is not None:
        db = db_trial

    sources = []
    if not db:
        chunk_sources = functools.partial(_chunk_sources, chunk=chunk, chunk_size=chunk_size, db_type=db_type)
        if langchain_mode in ['wiki_full']:
            from read_wiki_full import get_all_documents
            small_test = None
            print("Generating new wiki", flush=True)
            sources1 = get_all_documents(small_test=small_test, n_jobs=os.cpu_count() // 2)
            print("Got new wiki", flush=True)
            sources1 = chunk_sources(sources1, chunk=chunk)
            print("Chunked new wiki", flush=True)
            sources.extend(sources1)
        elif langchain_mode in ['wiki']:
            sources1 = get_wiki_sources(first_para=first_para, text_limit=text_limit)
            sources1 = chunk_sources(sources1, chunk=chunk)
            sources.extend(sources1)
        elif langchain_mode in ['github h2oGPT']:
            # sources = get_github_docs("dagster-io", "dagster")
            sources1 = get_github_docs("h2oai", "h2ogpt")
            # FIXME: always chunk for now
            sources1 = chunk_sources(sources1)
            sources.extend(sources1)
        elif langchain_mode in ['DriverlessAI docs']:
            sources1 = get_dai_docs(from_hf=True)
            # FIXME: DAI docs are already chunked well, should only chunk more if over limit
            sources1 = chunk_sources(sources1, chunk=False)
            sources.extend(sources1)
    if user_path:
        # UserData or custom, which has to be from user's disk
        if db is not None:
            # NOTE: Ignore file names for now, only go by hash ids
            # existing_files = get_existing_files(db)
            existing_files = []
            existing_hash_ids = get_existing_hash_ids(db)
        else:
            # pretend no existing files so won't filter
            existing_files = []
            existing_hash_ids = []
        # chunk internally for speed over multiple docs
        # FIXME: If first had old Hash=None and switch embeddings,
        #  then re-embed, and then hit here and reload so have hash, and then re-embed.
        sources1 = path_to_docs(user_path, n_jobs=n_jobs, chunk=chunk, chunk_size=chunk_size,
                                existing_files=existing_files, existing_hash_ids=existing_hash_ids,
                                db_type=db_type)
        new_metadata_sources = set([x.metadata['source'] for x in sources1])
        if new_metadata_sources:
            if os.getenv('NO_NEW_FILES') is not None:
                raise RuntimeError("Expected no new files! %s" % new_metadata_sources)
            print("Loaded %s new files as sources to add to %s" % (len(new_metadata_sources), langchain_mode),
                  flush=True)
            if verbose:
                print("Files added: %s" % '\n'.join(new_metadata_sources), flush=True)
        sources.extend(sources1)
        if len(sources) > 0 and os.getenv('NO_NEW_FILES') is not None:
            raise RuntimeError("Expected no new files! %s" % langchain_mode)
        if len(sources) == 0 and os.getenv('SHOULD_NEW_FILES') is not None:
            raise RuntimeError("Expected new files! %s" % langchain_mode)
        print("Loaded %s sources for potentially adding to %s" % (len(sources), langchain_mode), flush=True)

        # see if got sources
        if not sources:
            if verbose:
                if db is not None:
                    print("langchain_mode %s has no new sources, nothing to add to db" % langchain_mode, flush=True)
                else:
                    print("langchain_mode %s has no sources, not making new db" % langchain_mode, flush=True)
            return db, 0, []
        if verbose:
            if db is not None:
                print("Generating db", flush=True)
            else:
                print("Adding to db", flush=True)
    if not db:
        if sources:
            db = get_db(sources, use_openai_embedding=use_openai_embedding, db_type=db_type,
                        persist_directory=persist_directory,
                        langchain_mode=langchain_mode,
                        langchain_mode_paths=langchain_mode_paths,
                        langchain_mode_types=langchain_mode_types,
                        hf_embedding_model=hf_embedding_model,
                        migrate_embedding_model=migrate_embedding_model)
            if verbose:
                print("Generated db", flush=True)
        else:
            print("Did not generate db since no sources", flush=True)
        new_sources_metadata = [x.metadata for x in sources]
    elif user_path is not None:
        print("Existing db, potentially adding %s sources from user_path=%s" % (len(sources), user_path), flush=True)
        db, num_new_sources, new_sources_metadata = add_to_db(db, sources, db_type=db_type,
                                                              use_openai_embedding=use_openai_embedding,
                                                              hf_embedding_model=hf_embedding_model)
        print("Existing db, added %s new sources from user_path=%s" % (num_new_sources, user_path), flush=True)
    else:
        new_sources_metadata = [x.metadata for x in sources]

    return db, len(new_sources_metadata), new_sources_metadata


def get_metadatas(db):
    from langchain.vectorstores import FAISS
    if isinstance(db, FAISS):
        metadatas = [v.metadata for k, v in db.docstore._dict.items()]
    elif isinstance(db, Chroma):
        metadatas = get_documents(db)['metadatas']
    else:
        # FIXME: Hack due to https://github.com/weaviate/weaviate/issues/1947
        # seems no way to get all metadata, so need to avoid this approach for weaviate
        metadatas = [x.metadata for x in db.similarity_search("", k=10000)]
    return metadatas


def get_documents(db):
    if hasattr(db, '_persist_directory'):
        name_path = os.path.basename(db._persist_directory)
        base_path = 'locks'
        base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
        with filelock.FileLock(os.path.join(base_path, "getdb_%s.lock" % name_path)):
            # get segfaults and other errors when multiple threads access this
            return _get_documents(db)
    else:
        return _get_documents(db)


def _get_documents(db):
    from langchain.vectorstores import FAISS
    if isinstance(db, FAISS):
        documents = [v for k, v in db.docstore._dict.items()]
    elif isinstance(db, Chroma):
        documents = db.get()
    else:
        # FIXME: Hack due to https://github.com/weaviate/weaviate/issues/1947
        # seems no way to get all metadata, so need to avoid this approach for weaviate
        documents = [x for x in db.similarity_search("", k=10000)]
    return documents


def get_docs_and_meta(db, top_k_docs, filter_kwargs={}):
    if hasattr(db, '_persist_directory'):
        name_path = os.path.basename(db._persist_directory)
        base_path = 'locks'
        base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
        with filelock.FileLock(os.path.join(base_path, "getdb_%s.lock" % name_path)):
            return _get_docs_and_meta(db, top_k_docs, filter_kwargs=filter_kwargs)
    else:
        return _get_docs_and_meta(db, top_k_docs, filter_kwargs=filter_kwargs)


def _get_docs_and_meta(db, top_k_docs, filter_kwargs={}):
    from langchain.vectorstores import FAISS
    if isinstance(db, Chroma):
        db_get = db._collection.get(where=filter_kwargs.get('filter'))
        db_metadatas = db_get['metadatas']
        db_documents = db_get['documents']
    elif isinstance(db, FAISS):
        import itertools
        db_metadatas = get_metadatas(db)
        # FIXME: FAISS has no filter
        if top_k_docs == -1:
            db_documents = list(db.docstore._dict.values())
        else:
            # slice dict first
            db_documents = list(dict(itertools.islice(db.docstore._dict.items(), top_k_docs)).values())
    else:
        db_metadatas = get_metadatas(db)
        db_documents = get_documents(db)
    return db_documents, db_metadatas


def get_existing_files(db):
    metadatas = get_metadatas(db)
    metadata_sources = set([x['source'] for x in metadatas])
    return metadata_sources


def get_existing_hash_ids(db):
    metadatas = get_metadatas(db)
    # assume consistency, that any prior hashed source was single hashed file at the time among all source chunks
    metadata_hash_ids = {os.path.normpath(x['source']): x.get('hashid') for x in metadatas}
    return metadata_hash_ids


def run_qa_db(**kwargs):
    func_names = list(inspect.signature(_run_qa_db).parameters)
    # hard-coded defaults
    kwargs['answer_with_sources'] = kwargs.get('answer_with_sources', True)
    kwargs['show_rank'] = kwargs.get('show_rank', False)
    kwargs['show_accordions'] = kwargs.get('show_accordions', True)
    kwargs['top_k_docs_max_show'] = kwargs.get('top_k_docs_max_show', 10)
    kwargs['llamacpp_dict'] = {}  # shouldn't be required unless from test using _run_qa_db
    missing_kwargs = [x for x in func_names if x not in kwargs]
    assert not missing_kwargs, "Missing kwargs for run_qa_db: %s" % missing_kwargs
    # only keep actual used
    kwargs = {k: v for k, v in kwargs.items() if k in func_names}
    try:
        return _run_qa_db(**kwargs)
    finally:
        clear_torch_cache()


def _run_qa_db(query=None,
               iinput=None,
               context=None,
               use_openai_model=False, use_openai_embedding=False,
               first_para=False, text_limit=None, top_k_docs=4, chunk=True, chunk_size=512,
               langchain_mode_paths={},
               langchain_mode_types={},
               detect_user_path_changes_every_query=False,
               db_type=None,
               model_name=None, model=None, tokenizer=None, inference_server=None,
               langchain_only_model=False,
               hf_embedding_model=None,
               migrate_embedding_model=False,
               stream_output=False,
               async_output=True,
               num_async=3,
               prompter=None,
               prompt_type=None,
               prompt_dict=None,
               answer_with_sources=True,
               append_sources_to_answer=True,
               cut_distance=1.64,
               add_chat_history_to_context=True,
               system_prompt='',
               sanitize_bot_response=False,
               show_rank=False,
               show_accordions=True,
               top_k_docs_max_show=10,
               use_llm_if_no_docs=True,
               load_db_if_exists=False,
               db=None,
               do_sample=False,
               temperature=0.1,
               top_k=40,
               top_p=0.7,
               num_beams=1,
               max_new_tokens=512,
               min_new_tokens=1,
               early_stopping=False,
               max_time=180,
               repetition_penalty=1.0,
               num_return_sequences=1,
               langchain_mode=None,
               langchain_action=None,
               langchain_agents=None,
               document_subset=DocumentSubset.Relevant.name,
               document_choice=[DocumentChoice.ALL.value],
               pre_prompt_summary=None,
               prompt_summary=None,
               n_jobs=-1,
               llamacpp_dict=None,
               verbose=False,
               cli=False,
               reverse_docs=True,
               lora_weights='',
               auto_reduce_chunks=True,
               max_chunks=100,
               ):
    """

    :param query:
    :param use_openai_model:
    :param use_openai_embedding:
    :param first_para:
    :param text_limit:
    :param top_k_docs:
    :param chunk:
    :param chunk_size:
    :param langchain_mode_paths: dict of langchain_mode -> user path to glob recursively from
    :param db_type: 'faiss' for in-memory db or 'chroma' or 'weaviate' for persistent db
    :param model_name: model name, used to switch behaviors
    :param model: pre-initialized model, else will make new one
    :param tokenizer: pre-initialized tokenizer, else will make new one.  Required not None if model is not None
    :param answer_with_sources
    :return:
    """
    t_run = time.time()
    if stream_output:
        # threads and asyncio don't mix
        async_output = False
    if langchain_action in [LangChainAction.QUERY.value]:
        # only summarization supported
        async_output = False

    assert db_type is not None
    assert hf_embedding_model is not None
    assert langchain_mode_paths is not None
    assert langchain_mode_types is not None
    if model is not None:
        assert model_name is not None  # require so can make decisions
    assert query is not None
    assert prompter is not None or prompt_type is not None or model is None  # if model is None, then will generate
    if prompter is not None:
        prompt_type = prompter.prompt_type
        prompt_dict = prompter.prompt_dict
    if model is not None:
        assert prompt_type is not None
        if prompt_type == PromptType.custom.name:
            assert prompt_dict is not None  # should at least be {} or ''
        else:
            prompt_dict = ''
    assert len(set(gen_hyper).difference(inspect.signature(get_llm).parameters)) == 0
    # pass in context to LLM directly, since already has prompt_type structure
    # can't pass through langchain in get_chain() to LLM: https://github.com/hwchase17/langchain/issues/6638
    llm, model_name, streamer, prompt_type_out, async_output, only_new_text = \
        get_llm(use_openai_model=use_openai_model, model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                inference_server=inference_server,
                langchain_only_model=langchain_only_model,
                stream_output=stream_output,
                async_output=async_output,
                num_async=num_async,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                early_stopping=early_stopping,
                max_time=max_time,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                prompt_type=prompt_type,
                prompt_dict=prompt_dict,
                prompter=prompter,
                context=context if add_chat_history_to_context else '',
                iinput=iinput if add_chat_history_to_context else '',
                sanitize_bot_response=sanitize_bot_response,
                system_prompt=system_prompt,
                n_jobs=n_jobs,
                llamacpp_dict=llamacpp_dict,
                cli=cli,
                verbose=verbose,
                )
    # in case change, override original prompter
    if hasattr(llm, 'prompter'):
        prompter = llm.prompter

    use_docs_planned = False
    scores = []
    chain = None

    if isinstance(document_choice, str):
        # support string as well
        document_choice = [document_choice]

    func_names = list(inspect.signature(get_chain).parameters)
    sim_kwargs = {k: v for k, v in locals().items() if k in func_names}
    missing_kwargs = [x for x in func_names if x not in sim_kwargs]
    assert not missing_kwargs, "Missing: %s" % missing_kwargs
    docs, chain, scores, use_docs_planned, have_any_docs = get_chain(**sim_kwargs)
    if document_subset in non_query_commands:
        formatted_doc_chunks = '\n\n'.join([get_url(x) + '\n\n' + x.page_content for x in docs])
        if not formatted_doc_chunks and not use_llm_if_no_docs:
            yield "No sources", ''
            return
        # if no souces, outside gpt_langchain, LLM will be used with '' input
        yield formatted_doc_chunks, ''
        return
    if not use_llm_if_no_docs:
        if not docs and langchain_action in [LangChainAction.SUMMARIZE_MAP.value,
                                             LangChainAction.SUMMARIZE_ALL.value,
                                             LangChainAction.SUMMARIZE_REFINE.value]:
            ret = 'No relevant documents to summarize.' if have_any_docs else 'No documents to summarize.'
            extra = ''
            yield ret, extra
            return
        if not docs and langchain_mode not in [LangChainMode.DISABLED.value,
                                               LangChainMode.LLM.value]:
            ret = 'No relevant documents to query (for chatting with LLM, pick Resources->Collections->LLM).' if have_any_docs else 'No documents to query (for chatting with LLM, pick Resources->Collections->LLM).'
            extra = ''
            yield ret, extra
            return

    if chain is None and not langchain_only_model:
        # here if no docs at all and not HF type
        # can only return if HF type
        return

    # context stuff similar to used in evaluate()
    import torch
    device, torch_dtype, context_class = get_device_dtype()
    conditional_type = hasattr(llm, 'pipeline') and hasattr(llm.pipeline, 'model') and hasattr(llm.pipeline.model,
                                                                                               'conditional_type') and llm.pipeline.model.conditional_type
    with torch.no_grad():
        have_lora_weights = lora_weights not in [no_lora_str, '', None]
        context_class_cast = NullContext if device == 'cpu' or have_lora_weights else torch.autocast
        if conditional_type:
            # issues when casting to float16, can mess up t5 model, e.g. only when not streaming, or other odd behaviors
            context_class_cast = NullContext
        with context_class_cast(device):
            if stream_output and streamer:
                answer = None
                import queue
                bucket = queue.Queue()
                thread = EThread(target=chain, streamer=streamer, bucket=bucket)
                thread.start()
                outputs = ""
                try:
                    for new_text in streamer:
                        # print("new_text: %s" % new_text, flush=True)
                        if bucket.qsize() > 0 or thread.exc:
                            thread.join()
                        outputs += new_text
                        if prompter:  # and False:  # FIXME: pipeline can already use prompter
                            if conditional_type:
                                if prompter.botstr:
                                    prompt = prompter.botstr
                                    output_with_prompt = prompt + outputs
                                    only_new_text = False
                                else:
                                    prompt = None
                                    output_with_prompt = outputs
                                    only_new_text = True
                            else:
                                prompt = None  # FIXME
                                output_with_prompt = outputs
                                # don't specify only_new_text here, use get_llm() value
                            output1 = prompter.get_response(output_with_prompt, prompt=prompt,
                                                            only_new_text=only_new_text,
                                                            sanitize_bot_response=sanitize_bot_response)
                            yield output1, ''
                        else:
                            yield outputs, ''
                except BaseException:
                    # if any exception, raise that exception if was from thread, first
                    if thread.exc:
                        raise thread.exc
                    raise
                finally:
                    # in case no exception and didn't join with thread yet, then join
                    if not thread.exc:
                        answer = thread.join()
                        answer = answer['output_text']
                # in case raise StopIteration or broke queue loop in streamer, but still have exception
                if thread.exc:
                    raise thread.exc
            else:
                if async_output:
                    import asyncio
                    answer = asyncio.run(chain())
                else:
                    answer = chain()
                    answer = answer['output_text']

    t_run = time.time() - t_run
    if not use_docs_planned:
        ret = answer
        extra = ''
        yield ret, extra
    elif answer is not None:
        ret, extra = get_sources_answer(query, docs, answer, scores, show_rank,
                                        answer_with_sources,
                                        append_sources_to_answer,
                                        show_accordions=show_accordions,
                                        top_k_docs_max_show=top_k_docs_max_show,
                                        verbose=verbose,
                                        t_run=t_run,
                                        count_input_tokens=llm.count_input_tokens
                                        if hasattr(llm, 'count_input_tokens') else None,
                                        count_output_tokens=llm.count_output_tokens
                                        if hasattr(llm, 'count_output_tokens') else None)
        yield ret, extra
    return


def get_docs_with_score(query, k_db, filter_kwargs, db, db_type, verbose=False):
    # deal with bug in chroma where if (say) 234 doc chunks and ask for 233+ then fails due to reduction misbehavior
    docs_with_score = []
    if db_type == 'chroma':
        while True:
            try:
                docs_with_score = db.similarity_search_with_score(query, k=k_db, **filter_kwargs)
                break
            except (RuntimeError, AttributeError) as e:
                # AttributeError is for people with wrong version of langchain
                if verbose:
                    print("chroma bug: %s" % str(e), flush=True)
                if k_db == 1:
                    raise
                if k_db > 500:
                    k_db -= 200
                elif k_db > 100:
                    k_db -= 50
                elif k_db > 10:
                    k_db -= 5
                else:
                    k_db -= 1
                k_db = max(1, k_db)
    else:
        docs_with_score = db.similarity_search_with_score(query, k=k_db, **filter_kwargs)
    return docs_with_score


def get_chain(query=None,
              iinput=None,
              context=None,  # FIXME: https://github.com/hwchase17/langchain/issues/6638
              use_openai_model=False, use_openai_embedding=False,
              first_para=False, text_limit=None, top_k_docs=4, chunk=True, chunk_size=512,
              langchain_mode_paths=None,
              langchain_mode_types=None,
              detect_user_path_changes_every_query=False,
              db_type='faiss',
              model_name=None,
              inference_server='',
              langchain_only_model=False,
              hf_embedding_model=None,
              migrate_embedding_model=False,
              prompt_type=None,
              prompt_dict=None,
              cut_distance=1.1,
              add_chat_history_to_context=True,  # FIXME: https://github.com/hwchase17/langchain/issues/6638
              load_db_if_exists=False,
              db=None,
              langchain_mode=None,
              langchain_action=None,
              langchain_agents=None,
              document_subset=DocumentSubset.Relevant.name,
              document_choice=[DocumentChoice.ALL.value],
              pre_prompt_summary=None,
              prompt_summary=None,
              n_jobs=-1,
              # beyond run_db_query:
              llm=None,
              tokenizer=None,
              verbose=False,
              reverse_docs=True,
              stream_output=True,
              async_output=True,

              # local
              auto_reduce_chunks=True,
              max_chunks=100,
              ):
    if inference_server is None:
        inference_server = ''
    assert hf_embedding_model is not None
    assert langchain_agents is not None  # should be at least []
    # determine whether use of context out of docs is planned
    if not use_openai_model and prompt_type not in ['plain'] or langchain_only_model:
        if langchain_mode in ['Disabled', 'LLM']:
            use_docs_planned = False
        else:
            use_docs_planned = True
    else:
        use_docs_planned = True

    # https://github.com/hwchase17/langchain/issues/1946
    # FIXME: Seems to way to get size of chroma db to limit top_k_docs to avoid
    # Chroma collection MyData contains fewer than 4 elements.
    # type logger error
    if top_k_docs == -1:
        k_db = 1000 if db_type == 'chroma' else 100
    else:
        # top_k_docs=100 works ok too
        k_db = 1000 if db_type == 'chroma' else top_k_docs

    # FIXME: For All just go over all dbs instead of a separate db for All
    if not detect_user_path_changes_every_query and db is not None:
        # avoid looking at user_path during similarity search db handling,
        # if already have db and not updating from user_path every query
        # but if db is None, no db yet loaded (e.g. from prep), so allow user_path to be whatever it was
        if langchain_mode_paths is None:
            langchain_mode_paths = {}
        langchain_mode_paths = langchain_mode_paths.copy()
        langchain_mode_paths[langchain_mode] = None
    # once use_openai_embedding, hf_embedding_model passed in, possibly changed,
    # but that's ok as not used below or in calling functions
    db, num_new_sources, new_sources_metadata = make_db(use_openai_embedding=use_openai_embedding,
                                                        hf_embedding_model=hf_embedding_model,
                                                        migrate_embedding_model=migrate_embedding_model,
                                                        first_para=first_para, text_limit=text_limit,
                                                        chunk=chunk,
                                                        chunk_size=chunk_size,
                                                        langchain_mode=langchain_mode,
                                                        langchain_mode_paths=langchain_mode_paths,
                                                        langchain_mode_types=langchain_mode_types,
                                                        db_type=db_type,
                                                        load_db_if_exists=load_db_if_exists,
                                                        db=db,
                                                        n_jobs=n_jobs,
                                                        verbose=verbose)
    have_any_docs = db is not None
    if langchain_action == LangChainAction.QUERY.value:
        if iinput:
            query = "%s\n%s" % (query, iinput)

        if 'falcon' in model_name or 'Llama-2'.lower() in model_name.lower():
            extra = "According to only the information in the document sources provided within the context above, "
            prefix = "Pay attention and remember information below, which will help to answer the question or imperative after the context ends.\n"
        elif inference_server.startswith('openai'):
            extra = "According to (primarily) the information in the document sources provided within context above, "
            prefix = "Pay attention and remember information below, which will help to answer the question or imperative after the context ends.  If the answer cannot be primarily obtained from information within the context, then respond that the answer does not appear in the context of the documents.\n"
        else:
            extra = ""
            prefix = ""
        if langchain_mode in ['Disabled', 'LLM'] or not use_docs_planned:
            template_if_no_docs = template = """{context}{question}"""
        else:
            template = """%s
    \"\"\"
    {context}
    \"\"\"
    %s{question}""" % (prefix, extra)
            template_if_no_docs = """{context}{question}"""
    elif langchain_action in [LangChainAction.SUMMARIZE_ALL.value, LangChainAction.SUMMARIZE_MAP.value]:
        none = ['', '\n', None]

        if not pre_prompt_summary:
            pre_prompt_summary = """In order to write a concise single-paragraph or bulleted list summary, pay attention to the following text\n"""
        if not prompt_summary:
            if query in none and iinput in none:
                prompt_summary = "Using only the text above, write a condensed and concise summary of key results (preferably as bullet points):\n"
            elif query not in none:
                prompt_summary = "Focusing on %s, write a condensed and concise Summary:\n" % query
            elif iinput not in None:
                prompt_summary = iinput
            else:
                prompt_summary = "Focusing on %s, %s:\n" % (query, iinput)
        # don't auto reduce
        auto_reduce_chunks = False
        if langchain_action == LangChainAction.SUMMARIZE_MAP.value:
            fstring = '{text}'
        else:
            fstring = '{input_documents}'
        template = """%s:
\"\"\"
%s
\"\"\"\n%s""" % (pre_prompt_summary, fstring, prompt_summary)
        template_if_no_docs = "Exactly only say: There are no documents to summarize."
    elif langchain_action in [LangChainAction.SUMMARIZE_REFINE]:
        template = ''  # unused
        template_if_no_docs = ''  # unused
    else:
        raise RuntimeError("No such langchain_action=%s" % langchain_action)

    if not use_openai_model and prompt_type not in ['plain'] or langchain_only_model:
        use_template = True
    else:
        use_template = False

    query_action = langchain_action == LangChainAction.QUERY.value
    summarize_action = langchain_action in [LangChainAction.SUMMARIZE_MAP.value,
                                            LangChainAction.SUMMARIZE_ALL.value,
                                            LangChainAction.SUMMARIZE_REFINE.value]

    if hasattr(llm, 'pipeline') and hasattr(llm.pipeline, 'max_input_tokens'):
        max_input_tokens = llm.pipeline.max_input_tokens
    elif inference_server in ['openai', 'openai_azure']:
        max_tokens = llm.modelname_to_contextsize(model_name)
        # leave some room for 1 paragraph, even if min_new_tokens=0
        max_input_tokens = max_tokens - 256
    elif inference_server in ['openai_chat', 'openai_azure_chat']:
        max_tokens = model_token_mapping[model_name]
        # leave some room for 1 paragraph, even if min_new_tokens=0
        max_input_tokens = max_tokens - 256
    elif isinstance(tokenizer, FakeTokenizer):
        max_input_tokens = tokenizer.model_max_length - 256
    elif hasattr(tokenizer, 'model_max_length'):
        max_input_tokens = tokenizer.model_max_length - 256
    else:
        # leave some room for 1 paragraph, even if min_new_tokens=0
        max_input_tokens = 2048 - 256

    if db and use_docs_planned:
        base_path = 'locks'
        base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
        if hasattr(db, '_persist_directory'):
            name_path = "sim_%s.lock" % os.path.basename(db._persist_directory)
        else:
            name_path = "sim.lock"
        lock_file = os.path.join(base_path, name_path)

        if not isinstance(db, Chroma):
            # only chroma supports filtering
            filter_kwargs = {}
        else:
            assert document_choice is not None, "Document choice was None"
            if len(document_choice) >= 1 and document_choice[0] == DocumentChoice.ALL.value:
                filter_kwargs = {"filter": {"chunk_id": {"$gte": 0}}} if query_action else \
                    {"filter": {"chunk_id": {"$eq": -1}}}
            elif len(document_choice) >= 2:
                if document_choice[0] == DocumentChoice.ALL.value:
                    document_choice = document_choice[1:]
                or_filter = [{"source": {"$eq": x}, "chunk_id": {"$gte": 0}} if query_action else {"source": {"$eq": x},
                                                                                                   "chunk_id": {
                                                                                                       "$eq": -1}}
                             for x in document_choice]
                filter_kwargs = dict(filter={"$or": or_filter})
            elif len(document_choice) == 1:
                # degenerate UX bug in chroma
                one_filter = \
                    [{"source": {"$eq": x}, "chunk_id": {"$gte": 0}} if query_action else {"source": {"$eq": x},
                                                                                           "chunk_id": {
                                                                                               "$eq": -1}}
                     for x in document_choice][0]
                filter_kwargs = dict(filter=one_filter)
            else:
                # shouldn't reach
                filter_kwargs = {}
        if langchain_mode in [LangChainMode.LLM.value]:
            docs = []
            scores = []
        elif document_subset == DocumentSubset.TopKSources.name or query in [None, '', '\n']:
            db_documents, db_metadatas = get_docs_and_meta(db, top_k_docs, filter_kwargs=filter_kwargs)
            if top_k_docs == -1:
                top_k_docs = len(db_documents)
            # similar to langchain's chroma's _results_to_docs_and_scores
            docs_with_score = [(Document(page_content=result[0], metadata=result[1] or {}), 0)
                               for result in zip(db_documents, db_metadatas)]

            # order documents
            doc_hashes = [x.get('doc_hash', 'None') for x in db_metadatas]
            if query_action:
                doc_chunk_ids = [x.get('chunk_id', 0) for x in db_metadatas]
                docs_with_score = [x for hx, cx, x in
                                   sorted(zip(doc_hashes, doc_chunk_ids, docs_with_score), key=lambda x: (x[0], x[1]))
                                   if cx >= 0]
            else:
                assert summarize_action
                doc_chunk_ids = [x.get('chunk_id', -1) for x in db_metadatas]
                docs_with_score = [x for hx, cx, x in
                                   sorted(zip(doc_hashes, doc_chunk_ids, docs_with_score), key=lambda x: (x[0], x[1]))
                                   if cx == -1
                                   ]

            docs_with_score = docs_with_score[:top_k_docs]
            docs = [x[0] for x in docs_with_score]
            scores = [x[1] for x in docs_with_score]
            have_any_docs |= len(docs) > 0
        else:
            # FIXME: if langchain_action == LangChainAction.SUMMARIZE_MAP.value
            # if map_reduce, then no need to auto reduce chunks
            if top_k_docs == -1 or auto_reduce_chunks:
                top_k_docs_tokenize = 100
                with filelock.FileLock(lock_file):
                    docs_with_score = get_docs_with_score(query, k_db, filter_kwargs, db, db_type, verbose=verbose)[
                                      :top_k_docs_tokenize]
                if hasattr(llm, 'pipeline') and hasattr(llm.pipeline, 'tokenizer'):
                    # more accurate
                    tokens = [len(llm.pipeline.tokenizer(x[0].page_content)['input_ids']) for x in docs_with_score]
                    template_tokens = len(llm.pipeline.tokenizer(template)['input_ids'])
                elif inference_server in ['openai', 'openai_chat', 'openai_azure',
                                          'openai_azure_chat'] or use_openai_model:
                    tokens = [llm.get_num_tokens(x[0].page_content) for x in docs_with_score]
                    template_tokens = llm.get_num_tokens(template)
                elif isinstance(tokenizer, FakeTokenizer):
                    tokens = [tokenizer.num_tokens_from_string(x[0].page_content) for x in docs_with_score]
                    template_tokens = tokenizer.num_tokens_from_string(template)
                elif db_type in ['faiss', 'weaviate']:
                    # use ticktoken for faiss since embedding called differently
                    tokz = FakeTokenizer()
                    tokens = [tokz.num_tokens_from_string(x[0].page_content) for x in docs_with_score]
                    template_tokens = tokz.num_tokens_from_string(template)
                else:
                    # in case model is not our pipeline with HF tokenizer
                    tokens = [db._embedding_function.client.tokenize([x[0].page_content])['input_ids'].shape[1] for x in
                              docs_with_score]
                    template_tokens = db._embedding_function.client.tokenize([template])['input_ids'].shape[1]
                tokens_cumsum = np.cumsum(tokens)
                max_input_tokens -= template_tokens
                # FIXME: Doesn't account for query, == context, or new lines between contexts
                where_res = np.where(tokens_cumsum < max_input_tokens)[0]
                if where_res.shape[0] == 0:
                    # then no chunk can fit, still do first one
                    top_k_docs_trial = 1
                else:
                    top_k_docs_trial = 1 + where_res[-1]
                if 0 < top_k_docs_trial < max_chunks:
                    # avoid craziness
                    if top_k_docs == -1:
                        top_k_docs = top_k_docs_trial
                    else:
                        top_k_docs = min(top_k_docs, top_k_docs_trial)
                if top_k_docs == -1:
                    # if here, means 0 and just do best with 1 doc
                    print("Unexpected large chunks and can't add to context, will add 1 anyways", flush=True)
                    top_k_docs = 1
                docs_with_score = docs_with_score[:top_k_docs]
            else:
                with filelock.FileLock(lock_file):
                    docs_with_score = get_docs_with_score(query, k_db, filter_kwargs, db, db_type, verbose=verbose)[
                                      :top_k_docs]
            # put most relevant chunks closest to question,
            # esp. if truncation occurs will be "oldest" or "farthest from response" text that is truncated
            # BUT: for small models, e.g. 6_9 pythia, if sees some stuff related to h2oGPT first, it can connect that and not listen to rest
            if reverse_docs:
                docs_with_score.reverse()
            # cut off so no high distance docs/sources considered
            have_any_docs |= len(docs_with_score) > 0  # before cut
            docs = [x[0] for x in docs_with_score if x[1] < cut_distance]
            scores = [x[1] for x in docs_with_score if x[1] < cut_distance]
            if len(scores) > 0 and verbose:
                print("Distance: min: %s max: %s mean: %s median: %s" %
                      (scores[0], scores[-1], np.mean(scores), np.median(scores)), flush=True)
    else:
        docs = []
        scores = []

    if not docs and use_docs_planned and not langchain_only_model:
        # if HF type and have no docs, can bail out
        return docs, None, [], False, have_any_docs

    if document_subset in non_query_commands:
        # no LLM use
        return docs, None, [], False, have_any_docs

    common_words_file = "data/NGSL_1.2_stats.csv.zip"
    if os.path.isfile(common_words_file) and langchain_mode == LangChainAction.QUERY.value:
        df = pd.read_csv("data/NGSL_1.2_stats.csv.zip")
        import string
        reduced_query = query.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).strip()
        reduced_query_words = reduced_query.split(' ')
        set_common = set(df['Lemma'].values.tolist())
        num_common = len([x.lower() in set_common for x in reduced_query_words])
        frac_common = num_common / len(reduced_query) if reduced_query else 0
        # FIXME: report to user bad query that uses too many common words
        if verbose:
            print("frac_common: %s" % frac_common, flush=True)

    if len(docs) == 0:
        # avoid context == in prompt then
        use_docs_planned = False
        template = template_if_no_docs

    if langchain_action == LangChainAction.QUERY.value:
        if use_template:
            # instruct-like, rather than few-shot prompt_type='plain' as default
            # but then sources confuse the model with how inserted among rest of text, so avoid
            prompt = PromptTemplate(
                # input_variables=["summaries", "question"],
                input_variables=["context", "question"],
                template=template,
            )
            chain = load_qa_chain(llm, prompt=prompt, verbose=verbose)
        else:
            # only if use_openai_model = True, unused normally except in testing
            chain = load_qa_with_sources_chain(llm)
        if not use_docs_planned:
            chain_kwargs = dict(input_documents=[], question=query)
        else:
            chain_kwargs = dict(input_documents=docs, question=query)
        target = wrapped_partial(chain, chain_kwargs)
    elif langchain_action in [LangChainAction.SUMMARIZE_MAP.value,
                              LangChainAction.SUMMARIZE_REFINE,
                              LangChainAction.SUMMARIZE_ALL.value]:
        if async_output:
            return_intermediate_steps = False
        else:
            return_intermediate_steps = True
        from langchain.chains.summarize import load_summarize_chain
        if langchain_action == LangChainAction.SUMMARIZE_MAP.value:
            prompt = PromptTemplate(input_variables=["text"], template=template)
            chain = load_summarize_chain(llm, chain_type="map_reduce",
                                         map_prompt=prompt, combine_prompt=prompt,
                                         return_intermediate_steps=return_intermediate_steps,
                                         token_max=max_input_tokens, verbose=verbose)
            if async_output:
                chain_func = chain.arun
            else:
                chain_func = chain
            target = wrapped_partial(chain_func, {"input_documents": docs})  # , return_only_outputs=True)
        elif langchain_action == LangChainAction.SUMMARIZE_ALL.value:
            assert use_template
            prompt = PromptTemplate(input_variables=["text"], template=template)
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt,
                                         return_intermediate_steps=return_intermediate_steps, verbose=verbose)
            if async_output:
                chain_func = chain.arun
            else:
                chain_func = chain
            target = wrapped_partial(chain_func)
        elif langchain_action == LangChainAction.SUMMARIZE_REFINE.value:
            chain = load_summarize_chain(llm, chain_type="refine",
                                         return_intermediate_steps=return_intermediate_steps, verbose=verbose)
            if async_output:
                chain_func = chain.arun
            else:
                chain_func = chain
            target = wrapped_partial(chain_func)
        else:
            raise RuntimeError("No such langchain_action=%s" % langchain_action)
    else:
        raise RuntimeError("No such langchain_action=%s" % langchain_action)

    return docs, target, scores, use_docs_planned, have_any_docs


def get_sources_answer(query, docs, answer, scores, show_rank,
                       answer_with_sources, append_sources_to_answer,
                       show_accordions=True,
                       top_k_docs_max_show=10,
                       verbose=False,
                       t_run=None,
                       count_input_tokens=None, count_output_tokens=None):
    if verbose:
        print("query: %s" % query, flush=True)
        print("answer: %s" % answer, flush=True)

    if len(docs) == 0:
        extra = ''
        ret = answer + extra
        return ret, extra

    # link
    answer_sources = [(max(0.0, 1.5 - score) / 1.5,
                       get_url(doc, font_size=font_size),
                       get_accordion(doc, font_size=font_size, head_acc=head_acc)) for score, doc in
                      zip(scores, docs)]
    if not show_accordions:
        answer_sources_dict = defaultdict(list)
        [answer_sources_dict[url].append(score) for score, url in answer_sources]
        answers_dict = {}
        for url, scores_url in answer_sources_dict.items():
            answers_dict[url] = np.max(scores_url)
        answer_sources = [(score, url) for url, score in answers_dict.items()]
    answer_sources.sort(key=lambda x: x[0], reverse=True)
    if show_rank:
        # answer_sources = ['%d | %s' % (1 + rank, url) for rank, (score, url) in enumerate(answer_sources)]
        # sorted_sources_urls = "Sources [Rank | Link]:<br>" + "<br>".join(answer_sources)
        answer_sources = ['%s' % url for rank, (score, url) in enumerate(answer_sources)]
        answer_sources = answer_sources[:top_k_docs_max_show]
        sorted_sources_urls = "Ranked Sources:<br>" + "<br>".join(answer_sources)
    else:
        if show_accordions:
            answer_sources = ['<font size="%s"><li>%.2g | %s</li>%s</font>' % (font_size, score, url, accordion)
                              for score, url, accordion in answer_sources]
        else:
            answer_sources = ['<font size="%s"><li>%.2g | %s</li></font>' % (font_size, score, url)
                              for score, url in answer_sources]
        answer_sources = answer_sources[:top_k_docs_max_show]
        if show_accordions:
            sorted_sources_urls = f"<font size=\"{font_size}\">{source_prefix}<ul></font>" + "".join(answer_sources)
        else:
            sorted_sources_urls = f"<font size=\"{font_size}\">{source_prefix}<p><ul></font>" + "<p>".join(
                answer_sources)
        if verbose:
            if int(t_run):
                sorted_sources_urls += 'Total Time: %d [s]<p>' % t_run
            if count_input_tokens and count_output_tokens:
                sorted_sources_urls += 'Input Tokens: %s | Output Tokens: %d<p>' % (
                    count_input_tokens, count_output_tokens)
        sorted_sources_urls += f"<font size=\"{font_size}\"></ul></p>{source_postfix}</font>"
        title_overall = "Sources"
        sorted_sources_urls = f"""<details><summary><font size="{font_size}">{title_overall}</font></summary><font size="{font_size}">{sorted_sources_urls}</font></details>"""
        if os.getenv("HARD_ASSERTS"):
            assert sorted_sources_urls.startswith(super_source_prefix)
            assert sorted_sources_urls.endswith(super_source_postfix)

    if not answer.endswith('\n'):
        answer += '\n'

    if answer_with_sources:
        extra = '\n' + sorted_sources_urls
    else:
        extra = ''
    if append_sources_to_answer:
        ret = answer + extra
    else:
        ret = answer
    return ret, extra


def set_userid(db1s, requests_state1, get_userid_auth):
    db1 = db1s[LangChainMode.MY_DATA.value]
    assert db1 is not None and len(db1) == length_db1()
    if not db1[1]:
        db1[1] = get_userid_auth(requests_state1)
    if not db1[2]:
        username1 = None
        if 'username' in requests_state1:
            username1 = requests_state1['username']
        db1[2] = username1


def set_userid_direct(db1s, userid, username):
    db1 = db1s[LangChainMode.MY_DATA.value]
    db1[1] = userid
    db1[2] = username


def get_userid_direct(db1s):
    return db1s[LangChainMode.MY_DATA.value][1] if db1s is not None else ''


def get_username_direct(db1s):
    return db1s[LangChainMode.MY_DATA.value][2] if db1s is not None else ''


def get_dbid(db1):
    return db1[1]


def set_dbid(db1):
    # can only call this after function called so for specific user, not in gr.State() that occurs during app init
    assert db1 is not None and len(db1) == length_db1()
    if db1[1] is None:
        #  uuid in db is used as user ID
        db1[1] = str(uuid.uuid4())


def length_db1():
    # For MyData:
    # 0: db
    # 1: userid and dbid
    # 2: username

    # For others:
    # 0: db
    # 1: dbid
    # 2: None
    return 3


def get_any_db(db1s, langchain_mode, langchain_mode_paths, langchain_mode_types,
               dbs=None,
               load_db_if_exists=None, db_type=None,
               use_openai_embedding=None,
               hf_embedding_model=None, migrate_embedding_model=None,
               for_sources_list=False,
               verbose=False,
               ):
    if langchain_mode in [LangChainMode.DISABLED.value, LangChainMode.LLM.value]:
        return None
    elif for_sources_list and langchain_mode in [LangChainMode.WIKI_FULL.value]:
        # NOTE: avoid showing full wiki.  Takes about 30 seconds over about 90k entries, but not useful for now
        return None
    elif langchain_mode in db1s and len(db1s[langchain_mode]) > 1 and db1s[langchain_mode][0]:
        return db1s[langchain_mode][0]
    elif dbs is not None and langchain_mode in dbs and dbs[langchain_mode] is not None:
        return dbs[langchain_mode]
    else:
        db = None

    if db is None:
        langchain_type = langchain_mode_types.get(langchain_mode, LangChainTypes.EITHER.value)
        persist_directory, langchain_type = get_persist_directory(langchain_mode, db1s=db1s, dbs=dbs,
                                                                  langchain_type=langchain_type)
        langchain_mode_types[langchain_mode] = langchain_type
        # see if actually have on disk, don't try to switch embedding yet, since can't use return here
        migrate_embedding_model = False
        db, _, _ = \
            get_existing_db(db, persist_directory, load_db_if_exists, db_type,
                            use_openai_embedding,
                            langchain_mode, langchain_mode_paths, langchain_mode_types,
                            hf_embedding_model, migrate_embedding_model,
                            verbose=verbose)
        if db is not None:
            # if found db, then stuff into state, so don't have to reload again that takes time
            if langchain_type == LangChainTypes.PERSONAL.value:
                assert isinstance(db1s, dict), "db1s wrong type: %s" % type(db1s)
                db1 = db1s[langchain_mode] = [db, None, None]
                assert len(db1) == length_db1(), "Bad setup: %s" % len(db1)
                set_dbid(db1)
            else:
                assert isinstance(dbs, dict), "dbs wrong type: %s" % type(dbs)
                dbs[langchain_mode] = db

    return db


def get_sources(db1s, selection_docs_state1, requests_state1, langchain_mode,
                dbs=None, docs_state0=None,
                load_db_if_exists=None,
                db_type=None,
                use_openai_embedding=None,
                hf_embedding_model=None,
                migrate_embedding_model=None,
                verbose=False,
                get_userid_auth=None,
                ):
    for k in db1s:
        set_dbid(db1s[k])
    langchain_mode_paths = selection_docs_state1['langchain_mode_paths']
    langchain_mode_types = selection_docs_state1['langchain_mode_types']
    set_userid(db1s, requests_state1, get_userid_auth)
    db = get_any_db(db1s, langchain_mode, langchain_mode_paths, langchain_mode_types,
                    dbs=dbs,
                    load_db_if_exists=load_db_if_exists,
                    db_type=db_type,
                    use_openai_embedding=use_openai_embedding,
                    hf_embedding_model=hf_embedding_model,
                    migrate_embedding_model=migrate_embedding_model,
                    for_sources_list=True,
                    verbose=verbose,
                    )

    if langchain_mode in ['LLM'] or db is None:
        source_files_added = "NA"
        source_list = []
    elif langchain_mode in ['wiki_full']:
        source_files_added = "Not showing wiki_full, takes about 20 seconds and makes 4MB file." \
                             "  Ask jon.mckinney@h2o.ai for file if required."
        source_list = []
    elif db is not None:
        metadatas = get_metadatas(db)
        source_list = sorted(set([x['source'] for x in metadatas]))
        source_files_added = '\n'.join(source_list)
    else:
        source_list = []
        source_files_added = "None"
    sources_dir = "sources_dir"
    sources_dir = makedirs(sources_dir, exist_ok=True, tmp_ok=True, use_base=True)
    sources_file = os.path.join(sources_dir, 'sources_%s_%s' % (langchain_mode, str(uuid.uuid4())))
    with open(sources_file, "wt") as f:
        f.write(source_files_added)
    source_list = docs_state0 + source_list
    if 'All' in source_list:
        source_list.remove('All')
    return sources_file, source_list, db


def update_user_db(file, db1s, selection_docs_state1, requests_state1,
                   chunk, chunk_size, langchain_mode, dbs=None,
                   get_userid_auth=None,
                   **kwargs):
    kwargs.update(selection_docs_state1)
    set_userid(db1s, requests_state1, get_userid_auth)

    if file is None:
        raise RuntimeError("Don't use change, use input")

    try:
        return _update_user_db(file, db1s=db1s, chunk=chunk, chunk_size=chunk_size,
                               langchain_mode=langchain_mode, dbs=dbs,
                               **kwargs)
    except BaseException as e:
        print(traceback.format_exc(), flush=True)
        # gradio has issues if except, so fail semi-gracefully, else would hang forever in processing textbox
        ex_str = "Exception: %s" % str(e)
        source_files_added = """\
        <html>
          <body>
            <p>
               Sources: <br>
            </p>
               <div style="overflow-y: auto;height:400px">
               {0}
               </div>
          </body>
        </html>
        """.format(ex_str)
        doc_exception_text = str(e)
        return None, langchain_mode, source_files_added, doc_exception_text
    finally:
        clear_torch_cache()


def get_lock_file(db1, langchain_mode):
    db_id = get_dbid(db1)
    base_path = 'locks'
    base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
    lock_file = os.path.join(base_path, "db_%s_%s.lock" % (langchain_mode.replace(' ', '_'), db_id))
    return lock_file


def _update_user_db(file,
                    db1s=None,
                    chunk=None, chunk_size=None,
                    dbs=None, db_type=None,
                    langchain_mode='UserData',
                    langchain_modes=None,
                    langchain_mode_paths=None,
                    langchain_mode_types=None,
                    use_openai_embedding=None,
                    hf_embedding_model=None,
                    migrate_embedding_model=None,
                    caption_loader=None,
                    enable_captions=None,
                    captions_model=None,
                    enable_ocr=None,
                    enable_pdf_ocr=None,
                    verbose=None,
                    n_jobs=-1,
                    is_url=None, is_txt=None,
                    ):
    assert db1s is not None
    assert chunk is not None
    assert chunk_size is not None
    assert use_openai_embedding is not None
    assert hf_embedding_model is not None
    assert migrate_embedding_model is not None
    assert caption_loader is not None
    assert enable_captions is not None
    assert captions_model is not None
    assert enable_ocr is not None
    assert enable_pdf_ocr is not None
    assert verbose is not None

    if dbs is None:
        dbs = {}
    assert isinstance(dbs, dict), "Wrong type for dbs: %s" % str(type(dbs))
    # assert db_type in ['faiss', 'chroma'], "db_type %s not supported" % db_type
    # handle case of list of temp buffer
    if isinstance(file, str) and file.strip().startswith('['):
        try:
            file = ast.literal_eval(file.strip())
        except Exception as e:
            print("Tried to parse %s as list but failed: %s" % (file, str(e)), flush=True)
    if isinstance(file, list) and len(file) > 0 and hasattr(file[0], 'name'):
        file = [x.name for x in file]
    # handle single file of temp buffer
    if hasattr(file, 'name'):
        file = file.name
    if not isinstance(file, (list, tuple, typing.Generator)) and isinstance(file, str):
        file = [file]

    if langchain_mode == LangChainMode.DISABLED.value:
        return None, langchain_mode, get_source_files(), ""

    if langchain_mode in [LangChainMode.LLM.value]:
        # then switch to MyData, so langchain_mode also becomes way to select where upload goes
        # but default to mydata if nothing chosen, since safest
        if LangChainMode.MY_DATA.value in langchain_modes:
            langchain_mode = LangChainMode.MY_DATA.value

    if langchain_mode_paths is None:
        langchain_mode_paths = {}
    user_path = langchain_mode_paths.get(langchain_mode)
    # UserData or custom, which has to be from user's disk
    if user_path is not None:
        # move temp files from gradio upload to stable location
        for fili, fil in enumerate(file):
            if isinstance(fil, str) and os.path.isfile(fil):  # not url, text
                new_fil = os.path.normpath(os.path.join(user_path, os.path.basename(fil)))
                if os.path.normpath(os.path.abspath(fil)) != os.path.normpath(os.path.abspath(new_fil)):
                    if os.path.isfile(new_fil):
                        remove(new_fil)
                    try:
                        shutil.move(fil, new_fil)
                    except FileExistsError:
                        pass
                    file[fili] = new_fil

    if verbose:
        print("Adding %s" % file, flush=True)

    # FIXME: could avoid even parsing, let alone embedding, same old files if upload same file again
    # FIXME: but assume nominally user isn't uploading all files over again from UI

    sources = path_to_docs(file if not is_url and not is_txt else None,
                           verbose=verbose,
                           n_jobs=n_jobs,
                           chunk=chunk, chunk_size=chunk_size,
                           url=file if is_url else None,
                           text=file if is_txt else None,
                           enable_captions=enable_captions,
                           captions_model=captions_model,
                           enable_ocr=enable_ocr,
                           enable_pdf_ocr=enable_pdf_ocr,
                           caption_loader=caption_loader,
                           db_type=db_type,
                           )
    exceptions = [x for x in sources if x.metadata.get('exception')]
    exceptions_strs = [x.metadata['exception'] for x in exceptions]
    sources = [x for x in sources if 'exception' not in x.metadata]

    # below must at least come after langchain_mode is modified in case was LLM -> MyData,
    # so original langchain mode changed
    for k in db1s:
        set_dbid(db1s[k])
    db1 = get_db1(db1s, langchain_mode)

    lock_file = get_lock_file(db1s[LangChainMode.MY_DATA.value], langchain_mode)  # user-level lock, not db-level lock
    with filelock.FileLock(lock_file):
        if langchain_mode in db1s:
            if db1[0] is not None:
                # then add
                db, num_new_sources, new_sources_metadata = add_to_db(db1[0], sources, db_type=db_type,
                                                                      use_openai_embedding=use_openai_embedding,
                                                                      hf_embedding_model=hf_embedding_model)
            else:
                # in testing expect:
                # assert len(db1) == length_db1() and db1[1] is None, "Bad MyData db: %s" % db1
                # for production hit, when user gets clicky:
                assert len(db1) == length_db1(), "Bad %s db: %s" % (langchain_mode, db1)
                assert get_dbid(db1) is not None, "db hash was None, not allowed"
                # then create
                # if added has to original state and didn't change, then would be shared db for all users
                langchain_type = langchain_mode_types.get(langchain_mode, LangChainTypes.EITHER.value)
                persist_directory, langchain_type = get_persist_directory(langchain_mode, db1s=db1s, dbs=dbs,
                                                                          langchain_type=langchain_type)
                langchain_mode_types[langchain_mode] = langchain_type
                db = get_db(sources, use_openai_embedding=use_openai_embedding,
                            db_type=db_type,
                            persist_directory=persist_directory,
                            langchain_mode=langchain_mode,
                            langchain_mode_paths=langchain_mode_paths,
                            langchain_mode_types=langchain_mode_types,
                            hf_embedding_model=hf_embedding_model,
                            migrate_embedding_model=migrate_embedding_model)
            if db is not None:
                db1[0] = db
            source_files_added = get_source_files(db=db1[0], exceptions=exceptions)
            return None, langchain_mode, source_files_added, '\n'.join(exceptions_strs)
        else:
            langchain_type = langchain_mode_types.get(langchain_mode, LangChainTypes.EITHER.value)
            persist_directory, langchain_type = get_persist_directory(langchain_mode, db1s=db1s, dbs=dbs,
                                                                      langchain_type=langchain_type)
            langchain_mode_types[langchain_mode] = langchain_type
            if langchain_mode in dbs and dbs[langchain_mode] is not None:
                # then add
                db, num_new_sources, new_sources_metadata = add_to_db(dbs[langchain_mode], sources, db_type=db_type,
                                                                      use_openai_embedding=use_openai_embedding,
                                                                      hf_embedding_model=hf_embedding_model)
            else:
                # then create.  Or might just be that dbs is unfilled, then it will fill, then add
                db = get_db(sources, use_openai_embedding=use_openai_embedding,
                            db_type=db_type,
                            persist_directory=persist_directory,
                            langchain_mode=langchain_mode,
                            langchain_mode_paths=langchain_mode_paths,
                            langchain_mode_types=langchain_mode_types,
                            hf_embedding_model=hf_embedding_model,
                            migrate_embedding_model=migrate_embedding_model)
            dbs[langchain_mode] = db
            # NOTE we do not return db, because function call always same code path
            # return dbs[langchain_mode]
            # db in this code path is updated in place
            source_files_added = get_source_files(db=dbs[langchain_mode], exceptions=exceptions)
            return None, langchain_mode, source_files_added, '\n'.join(exceptions_strs)


def get_source_files_given_langchain_mode(db1s, selection_docs_state1, requests_state1, document_choice1,
                                          langchain_mode,
                                          dbs=None,
                                          load_db_if_exists=None,
                                          db_type=None,
                                          use_openai_embedding=None,
                                          hf_embedding_model=None,
                                          migrate_embedding_model=None,
                                          verbose=False,
                                          get_userid_auth=None,
                                          delete_sources=False):
    langchain_mode_paths = selection_docs_state1['langchain_mode_paths']
    langchain_mode_types = selection_docs_state1['langchain_mode_types']
    set_userid(db1s, requests_state1, get_userid_auth)
    db = get_any_db(db1s, langchain_mode, langchain_mode_paths, langchain_mode_types,
                    dbs=dbs,
                    load_db_if_exists=load_db_if_exists,
                    db_type=db_type,
                    use_openai_embedding=use_openai_embedding,
                    hf_embedding_model=hf_embedding_model,
                    migrate_embedding_model=migrate_embedding_model,
                    for_sources_list=True,
                    verbose=verbose,
                    )
    if delete_sources:
        del_from_db(db, document_choice1, db_type=db_type)

    if langchain_mode in ['LLM'] or db is None:
        return "Sources: N/A"
    return get_source_files(db=db, exceptions=None)


def get_source_files(db=None, exceptions=None, metadatas=None):
    if exceptions is None:
        exceptions = []

    # only should be one source, not confused
    # assert db is not None or metadatas is not None
    # clicky user
    if db is None and metadatas is None:
        return "No Sources at all"

    if metadatas is None:
        source_label = "Sources:"
        if db is not None:
            metadatas = get_metadatas(db)
        else:
            metadatas = []
        adding_new = False
    else:
        source_label = "New Sources:"
        adding_new = True

    # below automatically de-dups
    small_dict = {get_url(x['source'], from_str=True, short_name=True): get_short_name(x.get('head')) for x in
                  metadatas if x.get('page', 0) == 0}
    # if small_dict is empty dict, that's ok
    df = pd.DataFrame(small_dict.items(), columns=['source', 'head'])
    df.index = df.index + 1
    df.index.name = 'index'
    source_files_added = tabulate.tabulate(df, headers='keys', tablefmt='unsafehtml')

    if exceptions:
        exception_metadatas = [x.metadata for x in exceptions]
        small_dict = {get_url(x['source'], from_str=True, short_name=True): get_short_name(x.get('exception')) for x in
                      exception_metadatas}
        # if small_dict is empty dict, that's ok
        df = pd.DataFrame(small_dict.items(), columns=['source', 'exception'])
        df.index = df.index + 1
        df.index.name = 'index'
        exceptions_html = tabulate.tabulate(df, headers='keys', tablefmt='unsafehtml')
    else:
        exceptions_html = ''

    if metadatas and exceptions:
        source_files_added = """\
        <html>
          <body>
            <p>
               {0} <br>
            </p>
               <div style="overflow-y: auto;height:400px">
               {1}
               {2}
               </div>
          </body>
        </html>
        """.format(source_label, source_files_added, exceptions_html)
    elif metadatas:
        source_files_added = """\
        <html>
          <body>
            <p>
               {0} <br>
            </p>
               <div style="overflow-y: auto;height:400px">
               {1}
               </div>
          </body>
        </html>
        """.format(source_label, source_files_added)
    elif exceptions_html:
        source_files_added = """\
        <html>
          <body>
            <p>
               Exceptions: <br>
            </p>
               <div style="overflow-y: auto;height:400px">
               {0}
               </div>
          </body>
        </html>
        """.format(exceptions_html)
    else:
        if adding_new:
            source_files_added = "No New Sources"
        else:
            source_files_added = "No Sources"

    return source_files_added


def update_and_get_source_files_given_langchain_mode(db1s,
                                                     selection_docs_state,
                                                     requests_state,
                                                     langchain_mode, chunk, chunk_size,
                                                     dbs=None, first_para=None,
                                                     hf_embedding_model=None,
                                                     use_openai_embedding=None,
                                                     migrate_embedding_model=None,
                                                     text_limit=None,
                                                     db_type=None, load_db_if_exists=None,
                                                     n_jobs=None, verbose=None, get_userid_auth=None):
    set_userid(db1s, requests_state, get_userid_auth)
    assert hf_embedding_model is not None
    assert migrate_embedding_model is not None
    langchain_mode_paths = selection_docs_state['langchain_mode_paths']
    langchain_mode_types = selection_docs_state['langchain_mode_types']
    has_path = {k: v for k, v in langchain_mode_paths.items() if v}
    if langchain_mode in [LangChainMode.LLM.value, LangChainMode.MY_DATA.value]:
        # then assume user really meant UserData, to avoid extra clicks in UI,
        # since others can't be on disk, except custom user modes, which they should then select to query it
        if LangChainMode.USER_DATA.value in has_path:
            langchain_mode = LangChainMode.USER_DATA.value

    db = get_any_db(db1s, langchain_mode, langchain_mode_paths, langchain_mode_types,
                    dbs=dbs,
                    load_db_if_exists=load_db_if_exists,
                    db_type=db_type,
                    use_openai_embedding=use_openai_embedding,
                    hf_embedding_model=hf_embedding_model,
                    migrate_embedding_model=migrate_embedding_model,
                    for_sources_list=True,
                    verbose=verbose,
                    )

    # not designed for older way of using openai embeddings, why use_openai_embedding=False
    # use_openai_embedding, hf_embedding_model passed in and possible different values used,
    # but no longer used here or in calling functions so ok
    db, num_new_sources, new_sources_metadata = make_db(use_openai_embedding=False,
                                                        hf_embedding_model=hf_embedding_model,
                                                        migrate_embedding_model=migrate_embedding_model,
                                                        first_para=first_para, text_limit=text_limit,
                                                        chunk=chunk,
                                                        chunk_size=chunk_size,
                                                        langchain_mode=langchain_mode,
                                                        langchain_mode_paths=langchain_mode_paths,
                                                        langchain_mode_types=langchain_mode_types,
                                                        db_type=db_type,
                                                        load_db_if_exists=load_db_if_exists,
                                                        db=db,
                                                        n_jobs=n_jobs,
                                                        verbose=verbose)
    # during refreshing, might have "created" new db since not in dbs[] yet, so insert back just in case
    # so even if persisted, not kept up-to-date with dbs memory
    if langchain_mode in db1s:
        db1s[langchain_mode][0] = db
    else:
        dbs[langchain_mode] = db

    # return only new sources with text saying such
    return get_source_files(db=None, exceptions=None, metadatas=new_sources_metadata)


def get_db1(db1s, langchain_mode1):
    if langchain_mode1 in db1s:
        db1 = db1s[langchain_mode1]
    else:
        # indicates to code that not personal database
        db1 = [None] * length_db1()
    return db1


def clean_doc(docs1):
    if not isinstance(docs1, (list, tuple, types.GeneratorType)):
        docs1 = [docs1]
    for doci, doc in enumerate(docs1):
        docs1[doci].page_content = '\n'.join([x.strip() for x in doc.page_content.split("\n") if x.strip()])
    return docs1


def clone_documents(documents: Iterable[Document]) -> List[Document]:
    # first clone documents
    new_docs = []
    for doc in documents:
        new_doc = Document(page_content=doc.page_content, metadata=copy.deepcopy(doc.metadata))
        new_docs.append(new_doc)
    return new_docs


def _chunk_sources(sources, chunk=True, chunk_size=512, language=None, db_type=None):
    assert db_type is not None

    if not isinstance(sources, (list, tuple, types.GeneratorType)) and not callable(sources):
        # if just one document
        sources = [sources]
    if not chunk:
        [x.metadata.update(dict(chunk_id=0)) for chunk_id, x in enumerate(sources)]
        if db_type == 'chroma':
            # make copy so can have separate summarize case
            source_chunks = [Document(page_content=x.page_content,
                                      metadata=copy.deepcopy(x.metadata) or {})
                             for x in sources]
        else:
            source_chunks = sources  # just same thing
    else:
        if language and False:
            # Bug in langchain, keep separator=True not working
            # https://github.com/hwchase17/langchain/issues/2836
            # so avoid this for now
            keep_separator = True
            separators = RecursiveCharacterTextSplitter.get_separators_for_language(language)
        else:
            separators = ["\n\n", "\n", " ", ""]
            keep_separator = False
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, keep_separator=keep_separator,
                                                  separators=separators)
        source_chunks = splitter.split_documents(sources)

        # currently in order, but when pull from db won't be, so mark order and document by hash
        [x.metadata.update(dict(chunk_id=chunk_id)) for chunk_id, x in enumerate(source_chunks)]

    if db_type == 'chroma':
        # also keep original source for summarization and other tasks

        # assign chunk_id=-1 for original content
        # this assumes, as is currently true, that splitter makes new documents and list and metadata is deepcopy
        [x.metadata.update(dict(chunk_id=-1)) for chunk_id, x in enumerate(sources)]

        # in some cases sources is generator, so convert to list
        return list(sources) + source_chunks
    else:
        return source_chunks


def get_db_from_hf(dest=".", db_dir='db_dir_DriverlessAI_docs.zip'):
    from huggingface_hub import hf_hub_download
    # True for case when locally already logged in with correct token, so don't have to set key
    token = os.getenv('HUGGING_FACE_HUB_TOKEN', True)
    path_to_zip_file = hf_hub_download('h2oai/db_dirs', db_dir, token=token, repo_type='dataset')
    import zipfile
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        persist_directory = os.path.dirname(zip_ref.namelist()[0])
        remove(persist_directory)
        zip_ref.extractall(dest)
    return path_to_zip_file


# Note dir has space in some cases, while zip does not
some_db_zips = [['db_dir_DriverlessAI_docs.zip', 'db_dir_DriverlessAI docs', 'CC-BY-NC license'],
                ['db_dir_UserData.zip', 'db_dir_UserData', 'CC-BY license for ArXiv'],
                ['db_dir_github_h2oGPT.zip', 'db_dir_github h2oGPT', 'ApacheV2 license'],
                ['db_dir_wiki.zip', 'db_dir_wiki', 'CC-BY-SA Wikipedia license'],
                # ['db_dir_wiki_full.zip', 'db_dir_wiki_full.zip', '23GB, 05/04/2023 CC-BY-SA Wiki license'],
                ]

all_db_zips = some_db_zips + \
              [['db_dir_wiki_full.zip', 'db_dir_wiki_full.zip', '23GB, 05/04/2023 CC-BY-SA Wiki license'],
               ]


def get_some_dbs_from_hf(dest='.', db_zips=None):
    if db_zips is None:
        db_zips = some_db_zips
    for db_dir, dir_expected, license1 in db_zips:
        path_to_zip_file = get_db_from_hf(dest=dest, db_dir=db_dir)
        assert os.path.isfile(path_to_zip_file), "Missing zip in %s" % path_to_zip_file
        if dir_expected:
            assert os.path.isdir(os.path.join(dest, dir_expected)), "Missing path for %s" % dir_expected
            assert os.path.isdir(os.path.join(dest, dir_expected, 'index')), "Missing index in %s" % dir_expected


def _create_local_weaviate_client():
    WEAVIATE_URL = os.getenv('WEAVIATE_URL', "http://localhost:8080")
    WEAVIATE_USERNAME = os.getenv('WEAVIATE_USERNAME')
    WEAVIATE_PASSWORD = os.getenv('WEAVIATE_PASSWORD')
    WEAVIATE_SCOPE = os.getenv('WEAVIATE_SCOPE', "offline_access")

    resource_owner_config = None
    try:
        import weaviate
        from weaviate.embedded import EmbeddedOptions
        if WEAVIATE_USERNAME is not None and WEAVIATE_PASSWORD is not None:
            resource_owner_config = weaviate.AuthClientPassword(
                username=WEAVIATE_USERNAME,
                password=WEAVIATE_PASSWORD,
                scope=WEAVIATE_SCOPE
            )

        # if using remote server, don't choose persistent directory
        client = weaviate.Client(WEAVIATE_URL, auth_client_secret=resource_owner_config)
        return client
    except Exception as e:
        print(f"Failed to create Weaviate client: {e}")
        return None


if __name__ == '__main__':
    pass
