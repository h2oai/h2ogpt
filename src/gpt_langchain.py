import ast
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
import uuid
import zipfile
from collections import defaultdict
from datetime import datetime
from functools import reduce
from operator import concat
import filelock

from joblib import delayed
from langchain.callbacks import streaming_stdout
from langchain.embeddings import HuggingFaceInstructEmbeddings
from tqdm import tqdm

from enums import DocumentChoices, no_lora_str, model_token_mapping, source_prefix, source_postfix, non_query_commands, \
    LangChainAction, LangChainMode
from evaluate_params import gen_hyper
from gen import get_model, SEED
from prompter import non_hf_types, PromptType, Prompter
from utils import wrapped_partial, EThread, import_matplotlib, sanitize_filename, makedirs, get_url, flatten_list, \
    get_device, ProgressParallel, remove, hash_file, clear_torch_cache, NullContext, get_hf_server, FakeTokenizer, \
    have_libreoffice, have_arxiv, have_playwright, have_selenium, have_tesseract, have_pymupdf, set_openai
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
           persist_directory="db_dir", load_db_if_exists=True,
           langchain_mode='notset',
           collection_name=None,
           hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    if not sources:
        return None

    # get embedding model
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
                embedded_options=EmbeddedOptions()
            )
        index_name = collection_name.capitalize()
        db = Weaviate.from_documents(documents=sources, embedding=embedding, client=client, by_text=False,
                                     index_name=index_name)
    elif db_type == 'chroma':
        assert persist_directory is not None
        os.makedirs(persist_directory, exist_ok=True)

        # see if already actually have persistent db, and deal with possible changes in embedding
        db = get_existing_db(None, persist_directory, load_db_if_exists, db_type, use_openai_embedding, langchain_mode,
                             hf_embedding_model, verbose=False)
        if db is None:
            db = Chroma.from_documents(documents=sources,
                                       embedding=embedding,
                                       persist_directory=persist_directory,
                                       collection_name=collection_name,
                                       anonymized_telemetry=False)
            db.persist()
            clear_embedding(db)
            save_embed(db, use_openai_embedding, hf_embedding_model)
        else:
            # then just add
            db, num_new_sources, new_sources_metadata = add_to_db(db, sources, db_type=db_type,
                                                                  use_openai_embedding=use_openai_embedding,
                                                                  hf_embedding_model=hf_embedding_model)
    else:
        raise RuntimeError("No such db_type=%s" % db_type)

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
            sources = [x for x in sources if x.metadata['source'] not in metadata_files]
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
        db.add_documents(documents=sources)
        db.persist()
        clear_embedding(db)
        save_embed(db, use_openai_embedding, hf_embedding_model)
    else:
        raise RuntimeError("No such db_type=%s" % db_type)

    new_sources_metadata = [x.metadata for x in sources]

    return db, num_new_sources, new_sources_metadata


def create_or_update_db(db_type, persist_directory, collection_name,
                        sources, use_openai_embedding, add_if_exists, verbose, hf_embedding_model):
    if db_type == 'weaviate':
        import weaviate
        from weaviate.embedded import EmbeddedOptions

        if os.getenv('WEAVIATE_URL', None):
            client = _create_local_weaviate_client()
        else:
            client = weaviate.Client(
                embedded_options=EmbeddedOptions()
            )

        index_name = collection_name.replace(' ', '_').capitalize()
        if client.schema.exists(index_name) and not add_if_exists:
            client.schema.delete_class(index_name)
            if verbose:
                print("Removing %s" % index_name, flush=True)
    elif db_type == 'chroma':
        if not os.path.isdir(persist_directory) or not add_if_exists:
            if os.path.isdir(persist_directory):
                if verbose:
                    print("Removing %s" % persist_directory, flush=True)
                remove(persist_directory)
            if verbose:
                print("Generating db", flush=True)

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
                hf_embedding_model=hf_embedding_model)

    return db


def get_embedding(use_openai_embedding, hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
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
from typing import Any, Dict, List, Optional, Set

from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
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

    return_full_text: bool = True
    stream: bool = False
    sanitize_bot_response: bool = False

    prompter: Any = None
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
        stream_output = self.stream
        gr_client = self.client
        client_langchain_mode = 'Disabled'
        client_langchain_action = LangChainAction.QUERY.value
        client_langchain_agents = []
        top_k_docs = 1
        chunk = True
        chunk_size = 512
        client_kwargs = dict(instruction=prompt if self.chat_client else '',  # only for chat=True
                             iinput='',  # only for chat=True
                             context='',
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
                             iinput_nochat='',  # only for chat=False
                             langchain_mode=client_langchain_mode,
                             langchain_action=client_langchain_action,
                             langchain_agents=client_langchain_agents,
                             top_k_docs=top_k_docs,
                             chunk=chunk,
                             chunk_size=chunk_size,
                             document_subset=DocumentChoices.Relevant.name,
                             document_choice=[],
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
    stream: bool = False
    sanitize_bot_response: bool = False
    prompter: Any = None
    tokenizer: Any = None
    client: Any = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

        try:
            if values['client'] is None:
                import text_generation

                values["client"] = text_generation.Client(
                    values["inference_server_url"],
                    timeout=values["timeout"],
                    headers=values["headers"],
                )
        except ImportError:
            raise ImportError(
                "Could not import text_generation python package. "
                "Please install it with `pip install text_generation`."
            )
        return values

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        if stop is None:
            stop = self.stop_sequences
        else:
            stop += self.stop_sequences

        # HF inference server needs control over input tokens
        assert self.tokenizer is not None
        from h2oai_pipeline import H2OTextGenerationPipeline
        prompt, num_prompt_tokens = H2OTextGenerationPipeline.limit_prompt(prompt, self.tokenizer)

        # NOTE: TGI server does not add prompting, so must do here
        data_point = dict(context='', instruction=prompt, input='')
        prompt = self.prompter.generate_prompt(data_point)

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

        if not self.stream:
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
                    if stop_seq in response.token.text:
                        is_stop = True
                        break
                if is_stop:
                    break
                if not response.token.special:
                    if text_callback:
                        text_callback(response.token.text)
        return text


from langchain.chat_models import ChatOpenAI


class H2OChatOpenAI(ChatOpenAI):
    @classmethod
    def all_required_field_names(cls) -> Set:
        all_required_field_names = super(ChatOpenAI, cls).all_required_field_names()
        all_required_field_names.update({'top_p', 'frequency_penalty', 'presence_penalty'})
        return all_required_field_names


def get_llm(use_openai_model=False,
            model_name=None,
            model=None,
            tokenizer=None,
            inference_server=None,
            stream_output=False,
            do_sample=False,
            temperature=0.1,
            top_k=40,
            top_p=0.7,
            num_beams=1,
            max_new_tokens=256,
            min_new_tokens=1,
            early_stopping=False,
            max_time=180,
            repetition_penalty=1.0,
            num_return_sequences=1,
            prompt_type=None,
            prompt_dict=None,
            prompter=None,
            sanitize_bot_response=False,
            verbose=False,
            ):
    if use_openai_model or inference_server in ['openai', 'openai_chat', 'vllm', 'vllm_chat']:
        if use_openai_model and model_name is None:
            model_name = "gpt-3.5-turbo"
        openai, inf_type = set_openai(
            inference_server)  # FIXME: Will later import be ignored?  I think so, so should be fine
        if inference_server == 'openai_chat' or inf_type == 'vllm_chat':
            cls = H2OChatOpenAI
        else:
            from langchain.llms import OpenAI
            cls = OpenAI
        callbacks = [StreamingGradioCallbackHandler()]
        llm = cls(model_name=model_name,
                  temperature=temperature if do_sample else 0,
                  # FIXME: Need to count tokens and reduce max_new_tokens to fit like in generate.py
                  max_tokens=max_new_tokens,
                  top_p=top_p if do_sample else 1,
                  frequency_penalty=0,
                  presence_penalty=1.07 - repetition_penalty + 0.6,  # so good default
                  callbacks=callbacks if stream_output else None,
                  )
        streamer = callbacks[0] if stream_output else None
        if inference_server in ['openai', 'openai_chat']:
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
        assert prompter is not None
        terminate_response = prompter.terminate_response or []
        stop_sequences = list(set(terminate_response + [prompter.PreResponse]))
        stop_sequences = [x for x in stop_sequences if x]

        if gr_client:
            chat_client = False
            llm = GradioInference(
                inference_server_url=inference_server,
                return_full_text=True,

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
                stream=stream_output,
                prompter=prompter,
                client=gr_client,
                sanitize_bot_response=sanitize_bot_response,
            )
        elif hf_client:
            llm = H2OHuggingFaceTextGenInference(
                inference_server_url=inference_server,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                return_full_text=True,
                seed=SEED,

                stop_sequences=stop_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                # typical_p=top_p,
                callbacks=callbacks if stream_output else None,
                stream=stream_output,
                prompter=prompter,
                tokenizer=tokenizer,
                client=hf_client,
                timeout=max_time,
                sanitize_bot_response=sanitize_bot_response,
            )
        else:
            raise RuntimeError("No defined client")
        streamer = callbacks[0] if stream_output else None
    elif model_name in non_hf_types:
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
        llm = get_llm_gpt4all(model_name, model=model, max_new_tokens=max_new_tokens,
                              temperature=temperature,
                              repetition_penalty=repetition_penalty,
                              top_k=top_k,
                              top_p=top_p,
                              callbacks=callbacks,
                              verbose=verbose,
                              streaming=stream_output,
                              prompter=prompter,
                              )
    else:
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
                          return_full_text=True,
                          handle_long_generation=None)
        assert len(set(gen_hyper).difference(gen_kwargs.keys())) == 0

        if stream_output:
            skip_prompt = False
            from gen import H2OTextIteratorStreamer
            decoder_kwargs = {}
            streamer = H2OTextIteratorStreamer(tokenizer, skip_prompt=skip_prompt, block=False, **decoder_kwargs)
            gen_kwargs.update(dict(streamer=streamer))
        else:
            streamer = None

        from h2oai_pipeline import H2OTextGenerationPipeline
        pipe = H2OTextGenerationPipeline(model=model, use_prompter=True,
                                         prompter=prompter,
                                         prompt_type=prompt_type,
                                         prompt_dict=prompt_dict,
                                         sanitize_bot_response=sanitize_bot_response,
                                         chat=False, stream_output=stream_output,
                                         tokenizer=tokenizer,
                                         # leave some room for 1 paragraph, even if min_new_tokens=0
                                         max_input_tokens=max_max_tokens - max(min_new_tokens, 256),
                                         **gen_kwargs)
        # pipe.task = "text-generation"
        # below makes it listen only to our prompt removal,
        # not built in prompt removal that is less general and not specific for our model
        pipe.task = "text2text-generation"

        from langchain.llms import HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=pipe)
    return llm, model_name, streamer, prompt_type


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
    token = os.getenv('HUGGINGFACE_API_TOKEN', True)
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


image_types = ["png", "jpg", "jpeg"]
non_image_types = ["pdf", "txt", "csv", "toml", "py", "rst", "rtf",
                   "md",
                   "html", "mhtml",
                   "enex", "eml", "epub", "odt", "pptx", "ppt",
                   "zip", "urls",

                   ]
# "msg",  GPL3

if have_libreoffice or True:
    # or True so it tries to load, e.g. on MAC/Windows, even if don't have libreoffice since works without that
    non_image_types.extend(["docx", "doc", "xls", "xlsx"])

file_types = non_image_types + image_types


def add_meta(docs1, file):
    file_extension = pathlib.Path(file).suffix
    hashid = hash_file(file)
    doc_hash = str(uuid.uuid4())[:10]
    if not isinstance(docs1, (list, tuple, types.GeneratorType)):
        docs1 = [docs1]
    [x.metadata.update(dict(input_type=file_extension, date=str(datetime.now()), hashid=hashid, doc_hash=doc_hash)) for
     x in docs1]


def file_to_doc(file, base_path=None, verbose=False, fail_any_exception=False,
                chunk=True, chunk_size=512, n_jobs=-1,
                is_url=False, is_txt=False,
                enable_captions=True,
                captions_model=None,
                enable_ocr=False, caption_loader=None,
                headsize=50):
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
        if file.lower().startswith('arxiv:'):
            query = file.lower().split('arxiv:')
            if len(query) == 2 and have_arxiv:
                query = query[1]
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
            docs1 = UnstructuredURLLoader(urls=[file]).load()
            if len(docs1) == 0 and have_playwright:
                # then something went wrong, try another loader:
                from langchain.document_loaders import PlaywrightURLLoader
                docs1 = PlaywrightURLLoader(urls=[file]).load()
            if len(docs1) == 0 and have_selenium:
                # then something went wrong, try another loader:
                # but requires Chrome binary, else get: selenium.common.exceptions.WebDriverException: Message: unknown error: cannot find Chrome binary
                from langchain.document_loaders import SeleniumURLLoader
                from selenium.common.exceptions import WebDriverException
                try:
                    docs1 = SeleniumURLLoader(urls=[file]).load()
                except WebDriverException as e:
                    print("No web driver: %s" % str(e), flush=True)
            [x.metadata.update(dict(input_type='url', date=str(datetime.now))) for x in docs1]
        docs1 = clean_doc(docs1)
        doc1 = chunk_sources(docs1, chunk=chunk, chunk_size=chunk_size)
    elif is_txt:
        base_path = "user_paste"
        source_file = os.path.join(base_path, "_%s" % str(uuid.uuid4())[:10])
        makedirs(os.path.dirname(source_file), exist_ok=True)
        with open(source_file, "wt") as f:
            f.write(file)
        metadata = dict(source=source_file, date=str(datetime.now()), input_type='pasted txt')
        doc1 = Document(page_content=file, metadata=metadata)
        doc1 = clean_doc(doc1)
    elif file.lower().endswith('.html') or file.lower().endswith('.mhtml'):
        docs1 = UnstructuredHTMLLoader(file_path=file).load()
        add_meta(docs1, file)
        docs1 = clean_doc(docs1)
        doc1 = chunk_sources(docs1, chunk=chunk, chunk_size=chunk_size, language=Language.HTML)
    elif (file.lower().endswith('.docx') or file.lower().endswith('.doc')) and (have_libreoffice or True):
        docs1 = UnstructuredWordDocumentLoader(file_path=file).load()
        add_meta(docs1, file)
        doc1 = chunk_sources(docs1, chunk=chunk, chunk_size=chunk_size)
    elif (file.lower().endswith('.xlsx') or file.lower().endswith('.xls')) and (have_libreoffice or True):
        docs1 = UnstructuredExcelLoader(file_path=file).load()
        add_meta(docs1, file)
        doc1 = chunk_sources(docs1, chunk=chunk, chunk_size=chunk_size)
    elif file.lower().endswith('.odt'):
        docs1 = UnstructuredODTLoader(file_path=file).load()
        add_meta(docs1, file)
        doc1 = chunk_sources(docs1, chunk=chunk, chunk_size=chunk_size)
    elif file.lower().endswith('pptx') or file.lower().endswith('ppt'):
        docs1 = UnstructuredPowerPointLoader(file_path=file).load()
        add_meta(docs1, file)
        docs1 = clean_doc(docs1)
        doc1 = chunk_sources(docs1, chunk=chunk, chunk_size=chunk_size)
    elif file.lower().endswith('.txt'):
        # use UnstructuredFileLoader ?
        docs1 = TextLoader(file, encoding="utf8", autodetect_encoding=True).load()
        # makes just one, but big one
        doc1 = chunk_sources(docs1, chunk=chunk, chunk_size=chunk_size)
        doc1 = clean_doc(doc1)
        add_meta(doc1, file)
    elif file.lower().endswith('.rtf'):
        docs1 = UnstructuredRTFLoader(file).load()
        add_meta(docs1, file)
        doc1 = chunk_sources(docs1, chunk=chunk, chunk_size=chunk_size)
    elif file.lower().endswith('.md'):
        docs1 = UnstructuredMarkdownLoader(file).load()
        add_meta(docs1, file)
        docs1 = clean_doc(docs1)
        doc1 = chunk_sources(docs1, chunk=chunk, chunk_size=chunk_size, language=Language.MARKDOWN)
    elif file.lower().endswith('.enex'):
        docs1 = EverNoteLoader(file).load()
        add_meta(doc1, file)
        doc1 = chunk_sources(docs1, chunk=chunk, chunk_size=chunk_size)
    elif file.lower().endswith('.epub'):
        docs1 = UnstructuredEPubLoader(file).load()
        add_meta(docs1, file)
        doc1 = chunk_sources(docs1, chunk=chunk, chunk_size=chunk_size)
    elif file.lower().endswith('.jpeg') or file.lower().endswith('.jpg') or file.lower().endswith('.png'):
        docs1 = []
        if have_tesseract and enable_ocr:
            # OCR, somewhat works, but not great
            docs1.extend(UnstructuredImageLoader(file).load())
            add_meta(docs1, file)
        if enable_captions:
            # BLIP
            if caption_loader is not None and not isinstance(caption_loader, (str, bool)):
                # assumes didn't fork into this process with joblib, else can deadlock
                caption_loader.set_image_paths([file])
                docs1c = caption_loader.load()
                add_meta(docs1c, file)
                [x.metadata.update(dict(head=x.page_content[:headsize].strip())) for x in docs1c]
                docs1.extend(docs1c)
            else:
                from image_captions import H2OImageCaptionLoader
                caption_loader = H2OImageCaptionLoader(caption_gpu=caption_loader == 'gpu',
                                                       blip_model=captions_model,
                                                       blip_processor=captions_model)
                caption_loader.set_image_paths([file])
                docs1c = caption_loader.load()
                add_meta(docs1c, file)
                [x.metadata.update(dict(head=x.page_content[:headsize].strip())) for x in docs1c]
                docs1.extend(docs1c)
            for doci in docs1:
                doci.metadata['source'] = doci.metadata['image_path']
                doci.metadata['hash'] = hash_file(doci.metadata['source'])
            if docs1:
                doc1 = chunk_sources(docs1, chunk=chunk, chunk_size=chunk_size)
    elif file.lower().endswith('.msg'):
        raise RuntimeError("Not supported, GPL3 license")
        # docs1 = OutlookMessageLoader(file).load()
        # docs1[0].metadata['source'] = file
    elif file.lower().endswith('.eml'):
        try:
            docs1 = UnstructuredEmailLoader(file).load()
            add_meta(docs1, file)
            doc1 = chunk_sources(docs1, chunk=chunk, chunk_size=chunk_size)
        except ValueError as e:
            if 'text/html content not found in email' in str(e):
                # e.g. plain/text dict key exists, but not
                # doc1 = TextLoader(file, encoding="utf8").load()
                docs1 = UnstructuredEmailLoader(file, content_source="text/plain").load()
                add_meta(docs1, file)
                doc1 = chunk_sources(docs1, chunk=chunk, chunk_size=chunk_size)
            else:
                raise
    # elif file.lower().endswith('.gcsdir'):
    #    doc1 = GCSDirectoryLoader(project_name, bucket, prefix).load()
    # elif file.lower().endswith('.gcsfile'):
    # doc1 = GCSFileLoader(project_name, bucket, blob).load()
    elif file.lower().endswith('.rst'):
        with open(file, "r") as f:
            doc1 = Document(page_content=f.read(), metadata={"source": file})
        add_meta(doc1, file)
        doc1 = chunk_sources(doc1, chunk=chunk, chunk_size=chunk_size, language=Language.RST)
    elif file.lower().endswith('.pdf'):
        env_gpt4all_file = ".env_gpt4all"
        from dotenv import dotenv_values
        env_kwargs = dotenv_values(env_gpt4all_file)
        pdf_class_name = env_kwargs.get('PDF_CLASS_NAME', 'PyMuPDFParser')
        if have_pymupdf and pdf_class_name == 'PyMuPDFParser':
            # GPL, only use if installed
            from langchain.document_loaders import PyMuPDFLoader
            # load() still chunks by pages, but every page has title at start to help
            doc1 = PyMuPDFLoader(file).load()
            doc1 = clean_doc(doc1)
        elif pdf_class_name == 'UnstructuredPDFLoader':
            doc1 = UnstructuredPDFLoader(file).load()
            # seems to not need cleaning in most cases
        else:
            # open-source fallback
            # load() still chunks by pages, but every page has title at start to help
            doc1 = PyPDFLoader(file).load()
            doc1 = clean_doc(doc1)
        # Some PDFs return nothing or junk from PDFMinerLoader
        doc1 = chunk_sources(doc1, chunk=chunk, chunk_size=chunk_size)
        add_meta(doc1, file)
    elif file.lower().endswith('.csv'):
        doc1 = CSVLoader(file).load()
        add_meta(doc1, file)
    elif file.lower().endswith('.py'):
        doc1 = PythonLoader(file).load()
        add_meta(doc1, file)
        doc1 = chunk_sources(doc1, chunk=chunk, chunk_size=chunk_size, language=Language.PYTHON)
    elif file.lower().endswith('.toml'):
        doc1 = TomlLoader(file).load()
        add_meta(doc1, file)
    elif file.lower().endswith('.urls'):
        with open(file, "r") as f:
            docs1 = UnstructuredURLLoader(urls=f.readlines()).load()
        add_meta(docs1, file)
        doc1 = chunk_sources(docs1, chunk=chunk, chunk_size=chunk_size)
    elif file.lower().endswith('.zip'):
        with zipfile.ZipFile(file, 'r') as zip_ref:
            # don't put into temporary path, since want to keep references to docs inside zip
            # so just extract in path where
            zip_ref.extractall(base_path)
            # recurse
            doc1 = path_to_docs(base_path, verbose=verbose, fail_any_exception=fail_any_exception, n_jobs=n_jobs)
    else:
        raise RuntimeError("No file handler for %s" % os.path.basename(file))

    # allow doc1 to be list or not.  If not list, did not chunk yet, so chunk now
    # if list of length one, don't trust and chunk it
    if not isinstance(doc1, list):
        if chunk:
            docs = chunk_sources([doc1], chunk=chunk, chunk_size=chunk_size)
        else:
            docs = [doc1]
    elif isinstance(doc1, list) and len(doc1) == 1:
        if chunk:
            docs = chunk_sources(doc1, chunk=chunk, chunk_size=chunk_size)
        else:
            docs = doc1
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
                 enable_ocr=False, caption_loader=None):
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
                          caption_loader=caption_loader)
    except BaseException as e:
        print("Failed to ingest %s due to %s" % (file, traceback.format_exc()))
        if fail_any_exception:
            raise
        else:
            exception_doc = Document(
                page_content='',
                metadata={"source": file, "exception": '%s hit %s' % (file, str(e)),
                          "traceback": traceback.format_exc()})
            res = [exception_doc]
    if return_file:
        base_tmp = "temp_path_to_doc1"
        if not os.path.isdir(base_tmp):
            os.makedirs(base_tmp, exist_ok=True)
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
                 existing_files=[],
                 existing_hash_ids={},
                 ):
    # path_or_paths could be str, list, tuple, generator
    globs_image_types = []
    globs_non_image_types = []
    if not path_or_paths and not url and not text:
        return []
    elif url:
        globs_non_image_types = url if isinstance(url, (list, tuple, types.GeneratorType)) else [url]
    elif text:
        globs_non_image_types = text if isinstance(text, (list, tuple, types.GeneratorType)) else [text]
    elif isinstance(path_or_paths, str) and os.path.isdir(path_or_paths):
        # single path, only consume allowed files
        path = path_or_paths
        # Below globs should match patterns in file_to_doc()
        [globs_image_types.extend(glob.glob(os.path.join(path, "./**/*.%s" % ftype), recursive=True))
         for ftype in image_types]
        [globs_non_image_types.extend(glob.glob(os.path.join(path, "./**/*.%s" % ftype), recursive=True))
         for ftype in non_image_types]
    else:
        if isinstance(path_or_paths, str) and (os.path.isfile(path_or_paths) or os.path.isdir(path_or_paths)):
            path_or_paths = [path_or_paths]
        # list/tuple of files (consume what can, and exception those that selected but cannot consume so user knows)
        assert isinstance(path_or_paths, (list, tuple, types.GeneratorType)), "Wrong type for path_or_paths: %s" % type(
            path_or_paths)
        # reform out of allowed types
        globs_image_types.extend(flatten_list([[x for x in path_or_paths if x.endswith(y)] for y in image_types]))
        # could do below:
        # globs_non_image_types = flatten_list([[x for x in path_or_paths if x.endswith(y)] for y in non_image_types])
        # But instead, allow fail so can collect unsupported too
        set_globs_image_types = set(globs_image_types)
        globs_non_image_types.extend([x for x in path_or_paths if x not in set_globs_image_types])

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
            os.remove(fil)
    else:
        documents = reduce(concat, documents)
    return documents


def prep_langchain(persist_directory,
                   load_db_if_exists,
                   db_type, use_openai_embedding, langchain_mode, user_path,
                   hf_embedding_model, n_jobs=-1, kwargs_make_db={}):
    """
    do prep first time, involving downloads
    # FIXME: Add github caching then add here
    :return:
    """
    assert langchain_mode not in ['MyData'], "Should not prep scratch data"

    db_dir_exists = os.path.isdir(persist_directory)

    if db_dir_exists and user_path is None:
        print("Prep: persist_directory=%s exists, using" % persist_directory, flush=True)
        db = get_existing_db(None, persist_directory, load_db_if_exists, db_type, use_openai_embedding, langchain_mode,
                             hf_embedding_model)
    else:
        if db_dir_exists and user_path is not None:
            print("Prep: persist_directory=%s exists, user_path=%s passed, adding any changed or new documents" % (
                persist_directory, user_path), flush=True)
        elif not db_dir_exists:
            print("Prep: persist_directory=%s does not exist, regenerating" % persist_directory, flush=True)
        db = None
        if langchain_mode in ['All', 'DriverlessAI docs']:
            # FIXME: Could also just use dai_docs.pickle directly and upload that
            get_dai_docs(from_hf=True)

        if langchain_mode in ['All', 'wiki']:
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


def check_update_chroma_embedding(db, use_openai_embedding, hf_embedding_model, langchain_mode):
    changed_db = False
    if load_embed(db) != (use_openai_embedding, hf_embedding_model):
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
                    collection_name=None,
                    hf_embedding_model=hf_embedding_model)
        if False:
            # below doesn't work if db already in memory, so have to switch to new db as above
            # upsert does new embedding, but if index already in memory, complains about size mismatch etc.
            client_collection = db._client.get_collection(name=db._collection.name,
                                                          embedding_function=db._collection._embedding_function)
            client_collection.upsert(ids=db_get['ids'], metadatas=db_get['metadatas'], documents=db_get['documents'])
        changed_db = True
        print("Done updating db for new embedding: %s" % langchain_mode, flush=True)

    return db, changed_db


def get_existing_db(db, persist_directory, load_db_if_exists, db_type, use_openai_embedding, langchain_mode,
                    hf_embedding_model, verbose=False, check_embedding=True):
    if load_db_if_exists and db_type == 'chroma' and os.path.isdir(persist_directory) and os.path.isdir(
            os.path.join(persist_directory, 'index')):
        if db is None:
            if verbose:
                print("DO Loading db: %s" % langchain_mode, flush=True)
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
            if verbose:
                print("USING already-loaded db: %s" % langchain_mode, flush=True)
        if check_embedding:
            db_trial, changed_db = check_update_chroma_embedding(db, use_openai_embedding, hf_embedding_model,
                                                                 langchain_mode)
            if changed_db:
                db = db_trial
                # only call persist if really changed db, else takes too long for large db
                if db is not None:
                    db.persist()
                    clear_embedding(db)
        save_embed(db, use_openai_embedding, hf_embedding_model)
        return db
    return None


def clear_embedding(db):
    if db is None:
        return
    # don't keep on GPU, wastes memory, push back onto CPU and only put back on GPU once again embed
    db._embedding_function.client.cpu()
    clear_torch_cache()


def make_db(**langchain_kwargs):
    func_names = list(inspect.signature(_make_db).parameters)
    missing_kwargs = [x for x in func_names if x not in langchain_kwargs]
    defaults_db = {k: v.default for k, v in dict(inspect.signature(run_qa_db).parameters).items()}
    for k in missing_kwargs:
        if k in defaults_db:
            langchain_kwargs[k] = defaults_db[k]
    # final check for missing
    missing_kwargs = [x for x in func_names if x not in langchain_kwargs]
    assert not missing_kwargs, "Missing kwargs: %s" % missing_kwargs
    # only keep actual used
    langchain_kwargs = {k: v for k, v in langchain_kwargs.items() if k in func_names}
    return _make_db(**langchain_kwargs)


def save_embed(db, use_openai_embedding, hf_embedding_model):
    if db is not None:
        embed_info_file = os.path.join(db._persist_directory, 'embed_info')
        with open(embed_info_file, 'wb') as f:
            pickle.dump((use_openai_embedding, hf_embedding_model), f)
    return use_openai_embedding, hf_embedding_model


def load_embed(db):
    embed_info_file = os.path.join(db._persist_directory, 'embed_info')
    if os.path.isfile(embed_info_file):
        with open(embed_info_file, 'rb') as f:
            use_openai_embedding, hf_embedding_model = pickle.load(f)
    else:
        # migration, assume defaults
        use_openai_embedding, hf_embedding_model = False, "sentence-transformers/all-MiniLM-L6-v2"
    return use_openai_embedding, hf_embedding_model


def get_persist_directory(langchain_mode):
    return 'db_dir_%s' % langchain_mode  # single place, no special names for each case


def _make_db(use_openai_embedding=False,
             hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
             first_para=False, text_limit=None,
             chunk=True, chunk_size=512,
             langchain_mode=None,
             user_path=None,
             db_type='faiss',
             load_db_if_exists=True,
             db=None,
             n_jobs=-1,
             verbose=False):
    persist_directory = get_persist_directory(langchain_mode)
    # see if can get persistent chroma db
    db_trial = get_existing_db(db, persist_directory, load_db_if_exists, db_type, use_openai_embedding, langchain_mode,
                               hf_embedding_model, verbose=verbose)
    if db_trial is not None:
        db = db_trial

    sources = []
    if not db and langchain_mode not in ['MyData'] or \
            user_path is not None and \
            langchain_mode in ['UserData']:
        # Should not make MyData db this way, why avoided, only upload from UI
        assert langchain_mode not in ['MyData'], "Should not make MyData db this way"
        if verbose:
            if langchain_mode in ['UserData']:
                if user_path is not None:
                    print("Checking if changed or new sources in %s, and generating sources them" % user_path,
                          flush=True)
                elif db is None:
                    print("user_path not passed and no db, no sources", flush=True)
                else:
                    print("user_path not passed, using only existing db, no new sources", flush=True)
            else:
                print("Generating %s sources" % langchain_mode, flush=True)
        if langchain_mode in ['wiki_full', 'All', "'All'"]:
            from read_wiki_full import get_all_documents
            small_test = None
            print("Generating new wiki", flush=True)
            sources1 = get_all_documents(small_test=small_test, n_jobs=os.cpu_count() // 2)
            print("Got new wiki", flush=True)
            if chunk:
                sources1 = chunk_sources(sources1, chunk=chunk, chunk_size=chunk_size)
                print("Chunked new wiki", flush=True)
            sources.extend(sources1)
        if langchain_mode in ['wiki', 'All', "'All'"]:
            sources1 = get_wiki_sources(first_para=first_para, text_limit=text_limit)
            if chunk:
                sources1 = chunk_sources(sources1, chunk=chunk, chunk_size=chunk_size)
            sources.extend(sources1)
        if langchain_mode in ['github h2oGPT', 'All', "'All'"]:
            # sources = get_github_docs("dagster-io", "dagster")
            sources1 = get_github_docs("h2oai", "h2ogpt")
            # FIXME: always chunk for now
            sources1 = chunk_sources(sources1, chunk=chunk, chunk_size=chunk_size)
            sources.extend(sources1)
        if langchain_mode in ['DriverlessAI docs', 'All', "'All'"]:
            sources1 = get_dai_docs(from_hf=True)
            if chunk and False:  # FIXME: DAI docs are already chunked well, should only chunk more if over limit
                sources1 = chunk_sources(sources1, chunk=chunk, chunk_size=chunk_size)
            sources.extend(sources1)
        if langchain_mode in ['All', 'UserData']:
            if user_path:
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
                                        existing_files=existing_files, existing_hash_ids=existing_hash_ids)
                new_metadata_sources = set([x.metadata['source'] for x in sources1])
                if new_metadata_sources:
                    print("Loaded %s new files as sources to add to UserData" % len(new_metadata_sources), flush=True)
                    if verbose:
                        print("Files added: %s" % '\n'.join(new_metadata_sources), flush=True)
                sources.extend(sources1)
                print("Loaded %s sources for potentially adding to UserData" % len(sources), flush=True)
            else:
                print("Chose UserData but user_path is empty/None", flush=True)
        if False and langchain_mode in ['urls', 'All', "'All'"]:
            # from langchain.document_loaders import UnstructuredURLLoader
            # loader = UnstructuredURLLoader(urls=urls)
            urls = ["https://www.birdsongsf.com/who-we-are/"]
            from langchain.document_loaders import PlaywrightURLLoader
            loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])
            sources1 = loader.load()
            sources.extend(sources1)
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
                        persist_directory=persist_directory, langchain_mode=langchain_mode,
                        hf_embedding_model=hf_embedding_model)
            if verbose:
                print("Generated db", flush=True)
        else:
            print("Did not generate db since no sources", flush=True)
        new_sources_metadata = [x.metadata for x in sources]
    elif user_path is not None and langchain_mode in ['UserData']:
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
        makedirs(base_path)
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
        makedirs(base_path)
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
    metadata_hash_ids = {x['source']: x.get('hashid') for x in metadatas}
    return metadata_hash_ids


def run_qa_db(**kwargs):
    func_names = list(inspect.signature(_run_qa_db).parameters)
    # hard-coded defaults
    kwargs['answer_with_sources'] = True
    kwargs['show_rank'] = False
    missing_kwargs = [x for x in func_names if x not in kwargs]
    assert not missing_kwargs, "Missing kwargs: %s" % missing_kwargs
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
               user_path=None,
               detect_user_path_changes_every_query=False,
               db_type='faiss',
               model_name=None, model=None, tokenizer=None, inference_server=None,
               hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
               stream_output=False,
               prompter=None,
               prompt_type=None,
               prompt_dict=None,
               answer_with_sources=True,
               cut_distanct=1.1,
               sanitize_bot_response=False,
               show_rank=False,
               use_llm_if_no_docs=False,
               load_db_if_exists=False,
               db=None,
               do_sample=False,
               temperature=0.1,
               top_k=40,
               top_p=0.7,
               num_beams=1,
               max_new_tokens=256,
               min_new_tokens=1,
               early_stopping=False,
               max_time=180,
               repetition_penalty=1.0,
               num_return_sequences=1,
               langchain_mode=None,
               langchain_action=None,
               langchain_agents=None,
               document_subset=DocumentChoices.Relevant.name,
               document_choice=[],
               n_jobs=-1,
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
    :param user_path: user path to glob recursively from
    :param db_type: 'faiss' for in-memory db or 'chroma' or 'weaviate' for persistent db
    :param model_name: model name, used to switch behaviors
    :param model: pre-initialized model, else will make new one
    :param tokenizer: pre-initialized tokenizer, else will make new one.  Required not None if model is not None
    :param answer_with_sources
    :return:
    """
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
    llm, model_name, streamer, prompt_type_out = get_llm(use_openai_model=use_openai_model, model_name=model_name,
                                                         model=model,
                                                         tokenizer=tokenizer,
                                                         inference_server=inference_server,
                                                         stream_output=stream_output,
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
                                                         sanitize_bot_response=sanitize_bot_response,
                                                         verbose=verbose,
                                                         )

    use_context = False
    scores = []
    chain = None

    if isinstance(document_choice, str):
        # support string as well
        document_choice = [document_choice]

    func_names = list(inspect.signature(get_chain).parameters)
    sim_kwargs = {k: v for k, v in locals().items() if k in func_names}
    missing_kwargs = [x for x in func_names if x not in sim_kwargs]
    assert not missing_kwargs, "Missing: %s" % missing_kwargs
    docs, chain, scores, use_context, have_any_docs = get_chain(**sim_kwargs)
    if document_subset in non_query_commands:
        formatted_doc_chunks = '\n\n'.join([get_url(x) + '\n\n' + x.page_content for x in docs])
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
                                               LangChainMode.CHAT_LLM.value,
                                               LangChainMode.LLM.value]:
            ret = 'No relevant documents to query.' if have_any_docs else 'No documents to query.'
            extra = ''
            yield ret, extra
            return

    if chain is None and model_name not in non_hf_types:
        # here if no docs at all and not HF type
        # can only return if HF type
        return

    # context stuff similar to used in evaluate()
    import torch
    device, torch_dtype, context_class = get_device_dtype()
    with torch.no_grad():
        have_lora_weights = lora_weights not in [no_lora_str, '', None]
        context_class_cast = NullContext if device == 'cpu' or have_lora_weights else torch.autocast
        with context_class_cast(device):
            if stream_output and streamer:
                answer = None
                import queue
                bucket = queue.Queue()
                thread = EThread(target=chain, streamer=streamer, bucket=bucket)
                thread.start()
                outputs = ""
                prompt = None  # FIXME
                try:
                    for new_text in streamer:
                        # print("new_text: %s" % new_text, flush=True)
                        if bucket.qsize() > 0 or thread.exc:
                            thread.join()
                        outputs += new_text
                        if prompter:  # and False:  # FIXME: pipeline can already use prompter
                            output1 = prompter.get_response(outputs, prompt=prompt,
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
                # in case raise StopIteration or broke queue loop in streamer, but still have exception
                if thread.exc:
                    raise thread.exc
                # FIXME: answer is not string outputs from streamer.  How to get actual final output?
                # answer = outputs
            else:
                answer = chain()

    if not use_context:
        ret = answer['output_text']
        extra = ''
        yield ret, extra
    elif answer is not None:
        ret, extra = get_sources_answer(query, answer, scores, show_rank, answer_with_sources, verbose=verbose)
        yield ret, extra
    return


def get_chain(query=None,
              iinput=None,
              use_openai_model=False, use_openai_embedding=False,
              first_para=False, text_limit=None, top_k_docs=4, chunk=True, chunk_size=512,
              user_path=None,
              detect_user_path_changes_every_query=False,
              db_type='faiss',
              model_name=None,
              inference_server='',
              hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
              prompt_type=None,
              prompt_dict=None,
              cut_distanct=1.1,
              load_db_if_exists=False,
              db=None,
              langchain_mode=None,
              langchain_action=None,
              langchain_agents=None,
              document_subset=DocumentChoices.Relevant.name,
              document_choice=[],
              n_jobs=-1,
              # beyond run_db_query:
              llm=None,
              tokenizer=None,
              verbose=False,
              reverse_docs=True,

              # local
              auto_reduce_chunks=True,
              max_chunks=100,
              ):
    assert langchain_agents is not None  # should be at least []
    # determine whether use of context out of docs is planned
    if not use_openai_model and prompt_type not in ['plain'] or model_name in non_hf_types:
        if langchain_mode in ['Disabled', 'ChatLLM', 'LLM']:
            use_context = False
        else:
            use_context = True
    else:
        use_context = True

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
        user_path = None
    db, num_new_sources, new_sources_metadata = make_db(use_openai_embedding=use_openai_embedding,
                                                        hf_embedding_model=hf_embedding_model,
                                                        first_para=first_para, text_limit=text_limit,
                                                        chunk=chunk,
                                                        chunk_size=chunk_size,
                                                        langchain_mode=langchain_mode,
                                                        user_path=user_path,
                                                        db_type=db_type,
                                                        load_db_if_exists=load_db_if_exists,
                                                        db=db,
                                                        n_jobs=n_jobs,
                                                        verbose=verbose)
    have_any_docs = db is not None
    if langchain_action == LangChainAction.QUERY.value:
        if iinput:
            query = "%s\n%s" % (query, iinput)

        if 'falcon' in model_name:
            extra = "According to only the information in the document sources provided within the context above, "
            prefix = "Pay attention and remember information below, which will help to answer the question or imperative after the context ends."
        elif inference_server in ['openai', 'openai_chat']:
            extra = "According to (primarily) the information in the document sources provided within context above, "
            prefix = "Pay attention and remember information below, which will help to answer the question or imperative after the context ends.  If the answer cannot be primarily obtained from information within the context, then respond that the answer does not appear in the context of the documents."
        else:
            extra = ""
            prefix = ""
        if langchain_mode in ['Disabled', 'ChatLLM', 'LLM'] or not use_context:
            template_if_no_docs = template = """%s{context}{question}""" % prefix
        else:
            template = """%s
    \"\"\"
    {context}
    \"\"\"
    %s{question}""" % (prefix, extra)
            template_if_no_docs = """%s{context}%s{question}""" % (prefix, extra)
    elif langchain_action in [LangChainAction.SUMMARIZE_ALL.value, LangChainAction.SUMMARIZE_MAP.value]:
        none = ['', '\n', None]
        if query in none and iinput in none:
            prompt_summary = "Using only the text above, write a condensed and concise summary:\n"
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
        template = """In order to write a concise single-paragraph or bulleted list summary, pay attention to the following text:
\"\"\"
%s
\"\"\"\n%s""" % (fstring, prompt_summary)
        template_if_no_docs = "Exactly only say: There are no documents to summarize."
    elif langchain_action in [LangChainAction.SUMMARIZE_REFINE]:
        template = ''  # unused
        template_if_no_docs = ''  # unused
    else:
        raise RuntimeError("No such langchain_action=%s" % langchain_action)

    if not use_openai_model and prompt_type not in ['plain'] or model_name in non_hf_types:
        use_template = True
    else:
        use_template = False

    if db and use_context:
        base_path = 'locks'
        makedirs(base_path)
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
            if len(document_choice) >= 1 and document_choice[0] == DocumentChoices.All.name:
                filter_kwargs = {}
            elif len(document_choice) >= 2:
                if document_choice[0] == DocumentChoices.All.name:
                    # remove 'All'
                    document_choice = document_choice[1:]
                or_filter = [{"source": {"$eq": x}} for x in document_choice]
                filter_kwargs = dict(filter={"$or": or_filter})
            elif len(document_choice) == 1:
                # degenerate UX bug in chroma
                one_filter = [{"source": {"$eq": x}} for x in document_choice][0]
                filter_kwargs = dict(filter=one_filter)
            else:
                # shouldn't reach
                filter_kwargs = {}
        if langchain_mode in [LangChainMode.LLM.value, LangChainMode.CHAT_LLM.value]:
            docs = []
            scores = []
        elif document_subset == DocumentChoices.All.name or query in [None, '', '\n']:
            db_documents, db_metadatas = get_docs_and_meta(db, top_k_docs, filter_kwargs=filter_kwargs)
            # similar to langchain's chroma's _results_to_docs_and_scores
            docs_with_score = [(Document(page_content=result[0], metadata=result[1] or {}), 0)
                               for result in zip(db_documents, db_metadatas)]

            # order documents
            doc_hashes = [x.get('doc_hash', 'None') for x in db_metadatas]
            doc_chunk_ids = [x.get('chunk_id', 0) for x in db_metadatas]
            docs_with_score = [x for _, _, x in
                               sorted(zip(doc_hashes, doc_chunk_ids, docs_with_score), key=lambda x: (x[0], x[1]))
                               ]

            docs_with_score = docs_with_score[:top_k_docs]
            docs = [x[0] for x in docs_with_score]
            scores = [x[1] for x in docs_with_score]
            have_any_docs |= len(docs) > 0
        else:
            # FIXME: if langchain_action == LangChainAction.SUMMARIZE_MAP.value
            # if map_reduce, then no need to auto reduce chunks
            if top_k_docs == -1 or auto_reduce_chunks:
                # docs_with_score = db.similarity_search_with_score(query, k=k_db, **filter_kwargs)[:top_k_docs]
                top_k_docs_tokenize = 100
                with filelock.FileLock(lock_file):
                    docs_with_score = db.similarity_search_with_score(query, k=k_db, **filter_kwargs)[
                                      :top_k_docs_tokenize]
                if hasattr(llm, 'pipeline') and hasattr(llm.pipeline, 'tokenizer'):
                    # more accurate
                    tokens = [len(llm.pipeline.tokenizer(x[0].page_content)['input_ids']) for x in docs_with_score]
                    template_tokens = len(llm.pipeline.tokenizer(template)['input_ids'])
                elif inference_server in ['openai', 'openai_chat'] or use_openai_model or db_type in ['faiss',
                                                                                                      'weaviate']:
                    # use ticktoken for faiss since embedding called differently
                    tokens = [llm.get_num_tokens(x[0].page_content) for x in docs_with_score]
                    template_tokens = llm.get_num_tokens(template)
                elif isinstance(tokenizer, FakeTokenizer):
                    tokens = [tokenizer.num_tokens_from_string(x[0].page_content) for x in docs_with_score]
                    template_tokens = tokenizer.num_tokens_from_string(template)
                else:
                    # in case model is not our pipeline with HF tokenizer
                    tokens = [db._embedding_function.client.tokenize([x[0].page_content])['input_ids'].shape[1] for x in
                              docs_with_score]
                    template_tokens = db._embedding_function.client.tokenize([template])['input_ids'].shape[1]
                tokens_cumsum = np.cumsum(tokens)
                if hasattr(llm, 'pipeline') and hasattr(llm.pipeline, 'max_input_tokens'):
                    max_input_tokens = llm.pipeline.max_input_tokens
                elif inference_server in ['openai']:
                    max_tokens = llm.modelname_to_contextsize(model_name)
                    # leave some room for 1 paragraph, even if min_new_tokens=0
                    max_input_tokens = max_tokens - 256
                elif inference_server in ['openai_chat']:
                    max_tokens = model_token_mapping[model_name]
                    # leave some room for 1 paragraph, even if min_new_tokens=0
                    max_input_tokens = max_tokens - 256
                elif isinstance(tokenizer, FakeTokenizer):
                    max_input_tokens = tokenizer.model_max_length - 256
                else:
                    # leave some room for 1 paragraph, even if min_new_tokens=0
                    max_input_tokens = 2048 - 256
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
                    docs_with_score = db.similarity_search_with_score(query, k=k_db, **filter_kwargs)[:top_k_docs]
            # put most relevant chunks closest to question,
            # esp. if truncation occurs will be "oldest" or "farthest from response" text that is truncated
            # BUT: for small models, e.g. 6_9 pythia, if sees some stuff related to h2oGPT first, it can connect that and not listen to rest
            if reverse_docs:
                docs_with_score.reverse()
            # cut off so no high distance docs/sources considered
            have_any_docs |= len(docs_with_score) > 0  # before cut
            docs = [x[0] for x in docs_with_score if x[1] < cut_distanct]
            scores = [x[1] for x in docs_with_score if x[1] < cut_distanct]
            if len(scores) > 0 and verbose:
                print("Distance: min: %s max: %s mean: %s median: %s" %
                      (scores[0], scores[-1], np.mean(scores), np.median(scores)), flush=True)
    else:
        docs = []
        scores = []

    if not docs and use_context and model_name not in non_hf_types:
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
        use_context = False
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
            chain = load_qa_chain(llm, prompt=prompt)
        else:
            # only if use_openai_model = True, unused normally except in testing
            chain = load_qa_with_sources_chain(llm)
        if not use_context:
            chain_kwargs = dict(input_documents=[], question=query)
        else:
            chain_kwargs = dict(input_documents=docs, question=query)
        target = wrapped_partial(chain, chain_kwargs)
    elif langchain_action in [LangChainAction.SUMMARIZE_MAP.value,
                              LangChainAction.SUMMARIZE_REFINE,
                              LangChainAction.SUMMARIZE_ALL.value]:
        from langchain.chains.summarize import load_summarize_chain
        if langchain_action == LangChainAction.SUMMARIZE_MAP.value:
            prompt = PromptTemplate(input_variables=["text"], template=template)
            chain = load_summarize_chain(llm, chain_type="map_reduce",
                                         map_prompt=prompt, combine_prompt=prompt, return_intermediate_steps=True)
            target = wrapped_partial(chain, {"input_documents": docs})  # , return_only_outputs=True)
        elif langchain_action == LangChainAction.SUMMARIZE_ALL.value:
            assert use_template
            prompt = PromptTemplate(input_variables=["text"], template=template)
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt, return_intermediate_steps=True)
            target = wrapped_partial(chain)
        elif langchain_action == LangChainAction.SUMMARIZE_REFINE.value:
            chain = load_summarize_chain(llm, chain_type="refine", return_intermediate_steps=True)
            target = wrapped_partial(chain)
        else:
            raise RuntimeError("No such langchain_action=%s" % langchain_action)
    else:
        raise RuntimeError("No such langchain_action=%s" % langchain_action)

    return docs, target, scores, use_context, have_any_docs


def get_sources_answer(query, answer, scores, show_rank, answer_with_sources, verbose=False):
    if verbose:
        print("query: %s" % query, flush=True)
        print("answer: %s" % answer['output_text'], flush=True)

    if len(answer['input_documents']) == 0:
        extra = ''
        ret = answer['output_text'] + extra
        return ret, extra

    # link
    answer_sources = [(max(0.0, 1.5 - score) / 1.5, get_url(doc)) for score, doc in
                      zip(scores, answer['input_documents'])]
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
        sorted_sources_urls = "Ranked Sources:<br>" + "<br>".join(answer_sources)
    else:
        answer_sources = ['<li>%.2g | %s</li>' % (score, url) for score, url in answer_sources]
        sorted_sources_urls = f"{source_prefix}<p><ul>" + "<p>".join(answer_sources)
        sorted_sources_urls += f"</ul></p>{source_postfix}"

    if not answer['output_text'].endswith('\n'):
        answer['output_text'] += '\n'

    if answer_with_sources:
        extra = '\n' + sorted_sources_urls
    else:
        extra = ''
    ret = answer['output_text'] + extra
    return ret, extra


def clean_doc(docs1):
    if not isinstance(docs1, (list, tuple, types.GeneratorType)):
        docs1 = [docs1]
    for doci, doc in enumerate(docs1):
        docs1[doci].page_content = '\n'.join([x.strip() for x in doc.page_content.split("\n") if x.strip()])
    return docs1


def chunk_sources(sources, chunk=True, chunk_size=512, language=None):
    if not chunk:
        [x.metadata.update(dict(chunk_id=chunk_id)) for chunk_id, x in enumerate(sources)]
        return sources
    if not isinstance(sources, (list, tuple, types.GeneratorType)) and not callable(sources):
        # if just one document
        sources = [sources]
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

    return source_chunks


def get_db_from_hf(dest=".", db_dir='db_dir_DriverlessAI_docs.zip'):
    from huggingface_hub import hf_hub_download
    # True for case when locally already logged in with correct token, so don't have to set key
    token = os.getenv('HUGGINGFACE_API_TOKEN', True)
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
        if WEAVIATE_USERNAME is not None and WEAVIATE_PASSWORD is not None:
            resource_owner_config = weaviate.AuthClientPassword(
                username=WEAVIATE_USERNAME,
                password=WEAVIATE_PASSWORD,
                scope=WEAVIATE_SCOPE
            )

        client = weaviate.Client(WEAVIATE_URL, auth_client_secret=resource_owner_config)
        return client
    except Exception as e:
        print(f"Failed to create Weaviate client: {e}")
        return None


if __name__ == '__main__':
    pass
