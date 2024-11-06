import ast
import asyncio
import copy
import functools
import glob
import gzip
import importlib
import inspect
import json
import os
import pathlib
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
import typing
import urllib.error
import uuid
import zipfile
import tarfile
from collections import defaultdict
from datetime import datetime
from functools import reduce
from operator import concat
from random import randint
from urllib.parse import urlparse

import aiohttp
import filelock
import tabulate

from joblib import delayed
from langchain_anthropic.chat_models import _format_messages, _tools_in_params, _make_message_chunk_from_anthropic_event
from langchain_core.callbacks import streaming_stdout, AsyncCallbackManager, BaseCallbackHandler, BaseCallbackManager
from langchain.callbacks.base import Callbacks
from langchain_community.document_transformers import Html2TextTransformer, BeautifulSoupTransformer
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms.huggingface_pipeline import VALID_TASKS
from langchain_community.llms.utils import enforce_stop_tokens
from langchain.prompts.chat import ChatPromptValue
from langchain.schema import LLMResult, Generation, PromptValue
from langchain.schema.output import GenerationChunk
from langchain_core.globals import get_llm_cache
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.llms import aget_prompts, aupdate_cache
from langchain_core.load import dumpd
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, RunInfo, ChatGenerationChunk
from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools.json.tool import JsonSpec
from langchain_google_genai import ChatGoogleGenerativeAI

from gradio_utils.grclient import GradioClient
# from langchain_mistralai import ChatMistralAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_groq import ChatGroq
from pydantic.v1 import root_validator
from tqdm import tqdm

from db_utils import length_db1, set_dbid, set_userid, get_dbid, get_userid_direct, get_username_direct, \
    set_userid_direct
from image_utils import fix_image_file, get_image_types, get_image_file
from output_parser import H2OPythonMRKLOutputParser
from pandas_agent_langchain import create_csv_agent, create_pandas_dataframe_agent
from src.h2oai_pipeline import H2OTextGenerationPipeline
from src.langchain_openai_local import H2OBaseChatOpenAI, H2OBaseAzureChatOpenAI
from stopping import update_terminate_responses
from utils import wrapped_partial, EThread, import_matplotlib, sanitize_filename, makedirs, get_url, flatten_list, \
    get_device, ProgressParallel, remove, hash_file, clear_torch_cache, NullContext, get_hf_server, FakeTokenizer, \
    have_libreoffice, have_arxiv, have_playwright, have_selenium, have_tesseract, have_doctr, have_pymupdf, set_openai, \
    get_list_or_str, have_pillow, only_selenium, only_playwright, only_unstructured_urls, get_short_name, \
    get_accordion, have_jq, get_doc, get_source, get_token_count, reverse_ucurve_list, get_size, \
    get_test_name_core, download_simple, have_fiftyone, have_librosa, return_good_url, n_gpus_global, \
    get_accordion_named, hyde_titles, have_cv2, FullSet, create_relative_symlink, split_list, get_gradio_tmp, \
    merge_dict, get_docs_tokens, markdown_to_html, is_markdown, AsyncNullContext, url_prefixes_youtube, get_model_name, \
    dedup_list, have_pymupdf4llm
from enums import DocumentSubset, no_lora_str, model_token_mapping, source_prefix, source_postfix, non_query_commands, \
    LangChainAction, LangChainMode, DocumentChoice, LangChainTypes, font_size, head_acc, super_source_prefix, \
    super_source_postfix, langchain_modes_intrinsic, get_langchain_prompts, LangChainAgent, docs_joiner_default, \
    docs_ordering_types_default, langchain_modes_non_db, does_support_functiontools, doc_json_mode_system_prompt, \
    auto_choices, max_docs_public, max_chunks_per_doc_public, max_docs_public_api, max_chunks_per_doc_public_api, \
    does_support_json_mode, claude3imagetag, gpt4imagetag, geminiimagetag, \
    geminiimage_num_max, claude3image_num_max, gpt4image_num_max, llava_num_max, summary_prefix, extract_prefix, \
    noop_prompt_type, unknown_prompt_type, template_prompt_type, none, claude3_image_tokens, gemini_image_tokens, \
    gpt4_image_tokens, user_prompt_for_fake_system_prompt0, empty_prompt_type, \
    is_gradio_vision_model, is_json_model, anthropic_mapping, gemini15image_num_max, gemini15imagetag, \
    openai_supports_functiontools, openai_supports_parallel_functiontools, anthropic_prompt_caching
from evaluate_params import gen_hyper, gen_hyper0
from gen import SEED, get_limited_prompt, get_relaxed_max_new_tokens, get_model_retry, gradio_to_llm, \
    get_client_from_inference_server
from prompter import non_hf_types, PromptType, Prompter, get_vllm_extra_dict, system_docqa, system_summary
from h2o_serpapi import H2OSerpAPIWrapper
from utils_langchain import StreamingGradioCallbackHandler, _chunk_sources, _add_meta, add_parser, fix_json_meta, \
    load_general_summarization_chain, H2OHuggingFaceHubEmbeddings, make_sources_file, select_docs_with_score, \
    split_merge_docs, convert_to_genai_schema, PyMuPDF4LLMLoader

# to check imports
# find ./src -name '*.py' |  xargs awk '{ if (sub(/\\$/, "")) printf "%s ", $0; else print; }' |  grep 'from langchain\.' |  sed 's/^[ \t]*//' > go.py
# python go.py

import_matplotlib()

import numpy as np
import pandas as pd
import requests
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
# , GCSDirectoryLoader, GCSFileLoader
# , OutlookMessageLoader # GPL3
# ImageCaptionLoader, # use our own wrapper
#  ReadTheDocsLoader,  # no special file, some path, so have to give as special option
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, PythonLoader, TomlLoader, \
    UnstructuredURLLoader, UnstructuredHTMLLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader, \
    EverNoteLoader, UnstructuredEmailLoader, UnstructuredODTLoader, UnstructuredPowerPointLoader, \
    UnstructuredEPubLoader, UnstructuredImageLoader, UnstructuredRTFLoader, ArxivLoader, UnstructuredPDFLoader, \
    UnstructuredExcelLoader, JSONLoader, AsyncHtmlLoader, AsyncChromiumLoader
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter, TextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
# from langchain_community.llms import HuggingFaceTextGenInference, HuggingFacePipeline  # pycharm doesn't recognize parameters if use this
from langchain_community.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma


def get_context_cast():
    # chroma not autocasting right internally
    # return torch.autocast('cuda') if torch.cuda.is_available() else NullContext()
    return NullContext()


def get_db(sources, use_openai_embedding=False, db_type='faiss',
           persist_directory=None, load_db_if_exists=True,
           langchain_mode='notset',
           langchain_mode_paths={},
           langchain_mode_types={},
           collection_name=None,
           hf_embedding_model=None,
           migrate_embedding_model=False,
           n_jobs=-1,
           verbose=False):
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
        from langchain_community.vectorstores import FAISS
        db = FAISS.from_documents(sources, embedding)
    elif db_type == 'weaviate':
        import weaviate
        from weaviate.embedded import EmbeddedOptions
        from langchain_community.vectorstores import Weaviate

        if os.getenv('WEAVIATE_URL', None):
            client = _create_local_weaviate_client()
        else:
            client = weaviate.Client(
                embedded_options=EmbeddedOptions(persistence_data_path=persist_directory)
            )
        index_name = collection_name.capitalize()
        db = Weaviate.from_documents(documents=sources, embedding=embedding, client=client, by_text=False,
                                     index_name=index_name)
    elif db_type == 'qdrant':
        from langchain_community.vectorstores import Qdrant

        qdrant_options = _get_qdrant_options()

        if qdrant_options is not None:
            db = Qdrant.from_documents(documents=sources, embedding=embedding, collection_name=collection_name,
                                       **qdrant_options)
        else:
            db = Qdrant.from_documents(documents=sources, embedding=embedding, collection_name=collection_name,
                                       location=":memory:")

    elif db_type in ['chroma', 'chroma_old']:
        assert persist_directory, "persist_directory not filled"
        # use_base already handled when making persist_directory, unless was passed into get_db()
        makedirs(persist_directory, exist_ok=True)

        # see if already actually have persistent db, and deal with possible changes in embedding
        db, use_openai_embedding, hf_embedding_model = \
            get_existing_db(None, persist_directory, load_db_if_exists, db_type,
                            use_openai_embedding,
                            langchain_mode, langchain_mode_paths, langchain_mode_types,
                            hf_embedding_model, migrate_embedding_model,
                            verbose=False,
                            n_jobs=n_jobs)
        if db is None:
            import logging
            logging.getLogger("chromadb").setLevel(logging.ERROR)
            if db_type == 'chroma':
                from chromadb.config import Settings
                settings_extra_kwargs = dict(is_persistent=True)
            else:
                raise RuntimeError("Migration no longer supported")
            client_settings = Settings(anonymized_telemetry=False,
                                       persist_directory=persist_directory,
                                       **settings_extra_kwargs)
            if n_jobs in [None, -1]:
                n_jobs = int(os.getenv('OMP_NUM_THREADS', str(os.cpu_count() // 2)))
                num_threads = max(1, min(n_jobs, 8))
            else:
                num_threads = max(1, n_jobs)
            collection_metadata = {"hnsw:num_threads": num_threads}
            from_kwargs = dict(embedding=embedding,
                               persist_directory=persist_directory,
                               collection_name=collection_name,
                               client_settings=client_settings,
                               collection_metadata=collection_metadata)
            if db_type == 'chroma':
                import chromadb
                api = chromadb.PersistentClient(path=persist_directory)
                from_kwargs.update(dict(client=api))
                if hasattr(api, 'max_batch_size'):
                    max_batch_size = api.max_batch_size
                elif hasattr(api, '_producer') and hasattr(api._producer, 'max_batch_size'):
                    max_batch_size = api._producer.max_batch_size
                else:
                    max_batch_size = int(os.getenv('CHROMA_MAX_BATCH_SIZE', '100'))
                # limit embedding memory use
                max_batch_size = min(max_batch_size, int(os.getenv('CHROMA_MAX_BATCH_SIZE', '1024')))
                sources_batches = split_list(sources, max_batch_size)
                for sources_batch in sources_batches:
                    db = Chroma.from_documents(documents=sources_batch, **from_kwargs)
                    db.persist()
            else:
                raise RuntimeError("Migration no longer supported")
            clear_embedding(db)
            save_embed(db, use_openai_embedding, hf_embedding_model)
        else:
            # then just add
            # doesn't check or change embedding, just saves it in case not saved yet, after persisting
            db, num_new_sources, new_sources_metadata = add_to_db(db, sources, db_type=db_type,
                                                                  use_openai_embedding=use_openai_embedding,
                                                                  hf_embedding_model=hf_embedding_model,
                                                                  verbose=verbose)
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
    if hasattr(db, '_persist_directory'):
        print("Existing db, using %s" % db._persist_directory, flush=True)
        # chroma only
        lock_file = get_db_lock_file(db)
        context = filelock.FileLock
    else:
        lock_file = None
        context = NullContext
    if db_type in ['chroma', 'chroma_old'] and db is not None:
        with context(lock_file):
            # sources should be list of x.metadata['source'] from document metadatas
            if isinstance(sources, str):
                sources = [sources]
            else:
                assert isinstance(sources, (list, tuple, types.GeneratorType))
            api = db._client
            client_collection = api.get_collection(name=db._collection.name,
                                                   embedding_function=db._collection._embedding_function)
            if hasattr(api, 'max_batch_size'):
                max_batch_size = api.max_batch_size
            elif hasattr(client_collection, '_producer') and hasattr(client_collection._producer, 'max_batch_size'):
                max_batch_size = client_collection._producer.max_batch_size
            else:
                max_batch_size = int(os.getenv('CHROMA_MAX_BATCH_SIZE', '100'))
            max_batch_size = min(max_batch_size, int(os.getenv('CHROMA_MAX_BATCH_SIZE', '1024')))
            metadatas = list(set(sources))
            sources_batches = split_list(metadatas, max_batch_size)
            for sources_batch in sources_batches:
                for source in sources_batch:
                    meta = dict(source=source)
                    try:
                        client_collection.delete(where=meta)
                    except KeyError:
                        pass


def add_to_db(db, sources, db_type='faiss',
              avoid_dup_by_file=False,
              avoid_dup_by_content=True,
              use_openai_embedding=False,
              hf_embedding_model=None,
              verbose=False):
    assert hf_embedding_model is not None
    num_new_sources = len(sources)
    if not sources:
        return db, num_new_sources, []

    # don't do too large a batch so uses reasonable amount of memory
    max_max_batch_size = int(os.getenv('CHROMA_MAX_BATCH_SIZE', '4096'))

    if db_type == 'faiss':
        sources_batches = split_list(sources, max_max_batch_size)
        for sources_batch in sources_batches:
            db.add_documents(documents=sources_batch)
    elif db_type == 'weaviate':
        # FIXME: only control by file name, not hash yet
        if avoid_dup_by_file or avoid_dup_by_content:
            unique_sources = _get_unique_sources_in_weaviate(db)
            sources = [x for x in sources if x.metadata['source'] not in unique_sources]
        num_new_sources = len(sources)
        if num_new_sources == 0:
            return db, num_new_sources, []
        sources_batches = split_list(sources, max_max_batch_size)
        for sources_batch in sources_batches:
            db.add_documents(documents=sources_batch)
    elif db_type == 'qdrant':
        if avoid_dup_by_file or avoid_dup_by_content:
            unique_sources = _get_unique_sources_in_qdrant(db)
            sources = [x for x in sources if x.metadata['source'] not in unique_sources]
        num_new_sources = len(sources)
        if num_new_sources == 0:
            return db, num_new_sources, []
        sources_batches = split_list(sources, max_max_batch_size)
        for sources_batch in sources_batches:
            db.add_documents(documents=sources_batch)
    elif db_type in ['chroma', 'chroma_old']:
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
            lock_file = get_db_lock_file(db)
            context = filelock.FileLock
        else:
            lock_file = None
            context = NullContext
        with context(lock_file):
            # this is place where add to db, but others maybe accessing db, so lock access.
            # else see RuntimeError: Index seems to be corrupted or unsupported
            import chromadb
            api = chromadb.PersistentClient(path=db._persist_directory)
            if hasattr(api, 'max_batch_size'):
                max_batch_size = api.max_batch_size
            elif hasattr(api, '_producer') and hasattr(api._producer, 'max_batch_size'):
                max_batch_size = api._producer.max_batch_size
            else:
                # be conservative if not set
                max_batch_size = int(os.getenv('CHROMA_MAX_BATCH_SIZE', '100'))
            max_batch_size = min(max_batch_size, max_max_batch_size)
            if verbose:
                print("max_batch_size=%s" % max_batch_size, flush=True)
            sources_batches = split_list(sources, max_batch_size)
            for sources_batch in sources_batches:
                db.add_documents(documents=sources_batch)
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
                        hf_embedding_model, migrate_embedding_model,
                        n_jobs=-1):
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
    if db_type == 'qdrant':
        from qdrant_client import QdrantClient

        qdrant_options = _get_qdrant_options()

        if qdrant_options is not None:
            client = QdrantClient(**qdrant_options)
        else:
            client = QdrantClient(location=":memory:")

        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        if verbose:
            print("Removing %s" % collection_name, flush=True)
    elif db_type in ['chroma', 'chroma_old']:
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
                migrate_embedding_model=migrate_embedding_model,
                n_jobs=n_jobs,
                verbose=verbose,
                )

    return db


from langchain_community.embeddings import FakeEmbeddings


class H2OFakeEmbeddings(FakeEmbeddings):
    """Fake embedding model, but constant instead of random"""

    size: int
    """The size of the embedding vector."""

    def _get_embedding(self) -> typing.List[float]:
        return [1] * self.size

    def embed_documents(self, texts: typing.List[str]) -> typing.List[typing.List[float]]:
        return [self._get_embedding() for _ in texts]

    def embed_query(self, text: str) -> typing.List[float]:
        return self._get_embedding()


def get_embedding(use_openai_embedding, hf_embedding_model=None, preload=False, gpu_id=0):
    assert hf_embedding_model is not None
    # Get embedding model
    if use_openai_embedding:
        assert os.getenv("OPENAI_API_KEY") is not None, "Set ENV OPENAI_API_KEY"
        from langchain_community.embeddings import OpenAIEmbeddings
        embedding = OpenAIEmbeddings(disallowed_special=())
    elif hf_embedding_model == 'fake':
        embedding = H2OFakeEmbeddings(size=1)
    else:
        if isinstance(hf_embedding_model, str):
            pass
        elif isinstance(hf_embedding_model, dict):
            # embedding itself preloaded globally
            return hf_embedding_model['model']
        else:
            # object
            return hf_embedding_model

        if hf_embedding_model.startswith('tei:'):
            name = 'tei:'.join(hf_embedding_model.split('tei:')[1:])
            if not name.startswith('http'):
                name = 'http://' + name
            embedding = H2OHuggingFaceHubEmbeddings(model=name,
                                                    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN",
                                                                                            'foo'),
                                                    model_kwargs={"truncate": True})
        else:
            # to ensure can fork without deadlock
            from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings

            if isinstance(gpu_id, int) or gpu_id == 'auto':
                device, torch_dtype, context_class = get_device_dtype()
                model_kwargs = dict(device=device)
            else:
                # use gpu_id as device name
                model_kwargs = dict(device=gpu_id)
            if hf_embedding_model.startswith("BAAI/bge"):
                encode_kwargs = {'normalize_embeddings': True}
                if hf_embedding_model == "BAAI/bge-m3":
                    query_kwargs = dict(query_instruction="")
                else:
                    query_kwargs = dict()
                embedding = HuggingFaceBgeEmbeddings(model_name=hf_embedding_model,
                                                     model_kwargs=model_kwargs,
                                                     encode_kwargs=encode_kwargs,
                                                     **query_kwargs)
                embedding.client.eval()
            elif 'instructor' in hf_embedding_model:
                encode_kwargs = {'normalize_embeddings': True}
                embedding = HuggingFaceInstructEmbeddings(model_name=hf_embedding_model,
                                                          model_kwargs=model_kwargs,
                                                          encode_kwargs=encode_kwargs)
                embedding.client.eval()
            else:
                embedding = HuggingFaceEmbeddings(model_name=hf_embedding_model, model_kwargs=model_kwargs)
                embedding.client.eval()
            if gpu_id == 'auto':
                gpu_id = 0
            if preload and \
                    isinstance(gpu_id, int) and \
                    gpu_id >= 0 and \
                    hasattr(embedding.client, 'to') and \
                    get_device() == 'cuda':
                embedding.client = embedding.client.to('cuda:%d' % gpu_id)
            embedding.client.preload = preload
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
from typing import Any, Dict, List, Optional, Iterable

from pydantic import Field

from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain.llms.base import LLM


class H2Oagenerate:
    async def _agenerate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        if self.verbose:
            print("_agenerate H2O", flush=True)
        generations = []
        new_arg_supported = inspect.signature(self._acall).parameters.get("run_manager")
        if isinstance(self, GradioInference):
            pass
        else:
            self.count_input_tokens += sum([self.get_num_tokens(str(prompt)) for prompt in prompts])
            self.prompts.extend(prompts)
        tasks = [
            asyncio.ensure_future(self._agenerate_one(prompt, stop=stop, run_manager=run_manager,
                                                      new_arg_supported=new_arg_supported, **kwargs))
            for prompt in prompts
        ]
        texts = await asyncio.gather(*tasks)
        self.count_output_tokens += sum([self.get_num_tokens(text) for text in texts])
        [generations.append([Generation(text=text)]) for text in texts]
        if self.verbose:
            print("done _agenerate H2O", flush=True)
        return LLMResult(generations=generations)

    async def _agenerate_one(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            new_arg_supported=None,
            **kwargs: Any,
    ) -> str:
        async_sem = AsyncNullContext() if self.async_sem is None else self.async_sem
        async with async_sem:  # semaphore limits num of simultaneous downloads
            return await self._acall(prompt, stop=stop, run_manager=run_manager, **kwargs) \
                if new_arg_supported else \
                await self._acall(prompt, stop=stop, **kwargs)


class AGenerateStreamFirst:
    # from:
    # langchain_core/language_models/llms.py
    async def agenerate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            callbacks: Optional[typing.Union[Callbacks, List[Callbacks]]] = None,
            *,
            tags: Optional[typing.Union[List[str], List[List[str]]]] = None,
            metadata: Optional[typing.Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            run_name: Optional[typing.Union[str, List[str]]] = None,
            run_id: Optional[typing.Union[uuid.UUID, List[Optional[uuid.UUID]]]] = None,
            **kwargs: Any,
    ) -> LLMResult:
        # NOTE: overwrite of base class so can specify which messages will have callbacks
        callbacks_only_first = kwargs.get('stream', False) or \
                               kwargs.get('streaming', False) or \
                               hasattr(self, 'streaming') and self.streaming or \
                               hasattr(self, 'stream') and isinstance(self.stream, bool) and self.stream or \
                               hasattr(self, 'stream_output') and self.stream_output

        # Create callback managers
        if isinstance(callbacks, list) and (
                isinstance(callbacks[0], (list, BaseCallbackManager))
                or callbacks[0] is None
        ):
            # We've received a list of callbacks args to apply to each input
            assert len(callbacks) == len(prompts)
            assert tags is None or (
                    isinstance(tags, list) and len(tags) == len(prompts)
            )
            assert metadata is None or (
                    isinstance(metadata, list) and len(metadata) == len(prompts)
            )
            assert run_name is None or (
                    isinstance(run_name, list) and len(run_name) == len(prompts)
            )
            callbacks = typing.cast(List[Callbacks], callbacks)
            tags_list = typing.cast(List[Optional[List[str]]], tags or ([None] * len(prompts)))
            metadata_list = typing.cast(
                List[Optional[Dict[str, Any]]], metadata or ([{}] * len(prompts))
            )
            run_name_list = run_name or typing.cast(
                List[Optional[str]], ([None] * len(prompts))
            )
            callback_managers = [
                AsyncCallbackManager.configure(
                    callback,
                    self.callbacks,
                    self.verbose,
                    tag,
                    self.tags,
                    meta,
                    self.metadata,
                )
                for callback, tag, meta in zip(callbacks, tags_list, metadata_list)
            ]
        else:
            # We've received a single callbacks arg to apply to all inputs
            callback_managers = [
                                    AsyncCallbackManager.configure(
                                        typing.cast(Callbacks, callbacks),
                                        self.callbacks,
                                        self.verbose,
                                        typing.cast(List[str], tags),
                                        self.tags,
                                        typing.cast(Dict[str, Any], metadata),
                                        self.metadata,
                                    )
                                ] * len(prompts)
            run_name_list = [typing.cast(Optional[str], run_name)] * len(prompts)
        run_ids_list = self._get_run_ids_list(run_id, prompts)
        params = self.dict()
        params["stop"] = stop
        options = {"stop": stop}
        (
            existing_prompts,
            llm_string,
            missing_prompt_idxs,
            missing_prompts,
        ) = await aget_prompts(params, prompts, self.cache)

        # Verify whether the cache is set, and if the cache is set,
        # verify whether the cache is available.
        new_arg_supported = inspect.signature(self._agenerate).parameters.get(
            "run_manager"
        )
        if callbacks_only_first:
            for ii, run_manager in enumerate(callback_managers):
                if ii > 0:
                    run_manager.handlers = []
        if (self.cache is None and get_llm_cache() is None) or self.cache is False:
            run_managers = await asyncio.gather(
                *[
                    callback_manager.on_llm_start(
                        dumpd(self),
                        [prompt],
                        invocation_params=params,
                        options=options,
                        name=run_name,
                        batch_size=len(prompts),
                        run_id=run_id_,
                    )
                    for callback_manager, prompt, run_name, run_id_ in zip(
                        callback_managers, prompts, run_name_list, run_ids_list
                    )
                ]
            )
            run_managers = [r[0] for r in run_managers]  # type: ignore[misc]
            output = await self._agenerate_helper(
                prompts,
                stop,
                run_managers,  # type: ignore[arg-type]
                bool(new_arg_supported),
                **kwargs,  # type: ignore[arg-type]
            )
            return output
        if len(missing_prompts) > 0:
            run_managers = await asyncio.gather(
                *[
                    callback_managers[idx].on_llm_start(
                        dumpd(self),
                        [prompts[idx]],
                        invocation_params=params,
                        options=options,
                        name=run_name_list[idx],
                        batch_size=len(missing_prompts),
                    )
                    for idx in missing_prompt_idxs
                ]
            )
            run_managers = [r[0] for r in run_managers]  # type: ignore[misc]
            new_results = await self._agenerate_helper(
                missing_prompts,
                stop,
                run_managers,  # type: ignore[arg-type]
                bool(new_arg_supported),
                **kwargs,  # type: ignore[arg-type]
            )
            llm_output = await aupdate_cache(
                self.cache,
                existing_prompts,
                llm_string,
                missing_prompt_idxs,
                new_results,
                prompts,
            )
            run_info = (
                [RunInfo(run_id=run_manager.run_id) for run_manager in run_managers]  # type: ignore[attr-defined]
                if run_managers
                else None
            )
        else:
            llm_output = {}
            run_info = None
        generations = [existing_prompts[i] for i in range(len(prompts))]
        return LLMResult(generations=generations, llm_output=llm_output, run=run_info)


class ChatAGenerateStreamFirst:
    # from
    # langchain_core/language_models/chat_models.py
    async def agenerate(
            self,
            messages: List[List[BaseMessage]],
            stop: Optional[List[str]] = None,
            callbacks: Callbacks = None,
            *,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            run_name: Optional[str] = None,
            run_id: Optional[uuid.UUID] = None,
            **kwargs: Any,
    ) -> LLMResult:
        # NOTE: overwrite of base class so can specify which messages will have callbacks
        callbacks_only_first = kwargs.get('stream', False) or \
                               kwargs.get('streaming', False) or \
                               hasattr(self, 'streaming') and self.streaming or \
                               hasattr(self, 'stream') and isinstance(self.stream, bool) and self.stream or \
                               hasattr(self, 'stream_output') and self.stream_output
        if self.verbose:
            print("messages: %s" % len(messages))

        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {"stop": stop}

        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )

        run_managers = await callback_manager.on_chat_model_start(
            dumpd(self),
            messages,
            invocation_params=params,
            options=options,
            name=run_name,
            batch_size=len(messages),
            run_id=run_id,
        )
        if callbacks_only_first:
            for ii, run_manager in enumerate(run_managers):
                if ii > 0:
                    run_manager.handlers = []

        results = await asyncio.gather(
            *[
                self._agenerate_with_cache(
                    m,
                    stop=stop,
                    run_manager=run_managers[i] if run_managers else None,
                    **kwargs,
                )
                for i, m in enumerate(messages)
            ],
            return_exceptions=True,
        )
        exceptions = []
        for i, res in enumerate(results):
            if isinstance(res, BaseException):
                if run_managers:
                    await run_managers[i].on_llm_error(
                        res, response=LLMResult(generations=[])
                    )
                exceptions.append(res)
        if exceptions:
            if run_managers:
                await asyncio.gather(
                    *[
                        run_manager.on_llm_end(
                            LLMResult(
                                generations=[res.generations],  # type: ignore[list-item, union-attr]
                                llm_output=res.llm_output,  # type: ignore[list-item, union-attr]
                            )
                        )
                        for run_manager, res in zip(run_managers, results)
                        if not isinstance(res, Exception)
                    ]
                )
            raise exceptions[0]
        flattened_outputs = [
            LLMResult(generations=[res.generations], llm_output=res.llm_output)  # type: ignore[list-item, union-attr]
            for res in results
        ]
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])  # type: ignore[union-attr]
        generations = [res.generations for res in results]  # type: ignore[union-attr]
        output = LLMResult(generations=generations, llm_output=llm_output)  # type: ignore[arg-type]
        await asyncio.gather(
            *[
                run_manager.on_llm_end(flattened_output)
                for run_manager, flattened_output in zip(
                    run_managers, flattened_outputs
                )
            ]
        )
        if run_managers:
            output.run = [
                RunInfo(run_id=run_manager.run_id) for run_manager in run_managers
            ]
        return output


class GradioInference(AGenerateStreamFirst, H2Oagenerate, LLM):
    """
    Gradio generation inference API.
    """
    inference_server_url: str = ""

    temperature: float = 0.8
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = None
    penalty_alpha: Optional[float] = 0.0
    num_beams: Optional[int] = 1
    max_new_tokens: int = 512
    max_new_tokens0: int = 512
    min_new_tokens: int = 1
    early_stopping: bool = False
    max_time: int = 180
    repetition_penalty: Optional[float] = None
    num_return_sequences: Optional[int] = 1
    do_sample: bool = False
    seed: int = 0
    chat_client: bool = False

    return_full_text: bool = False
    stream_output: bool = False
    enable_caching: bool = False
    sanitize_bot_response: bool = False

    prompter: Any = None
    context: Any = ''
    iinput: Any = ''
    client: Any = None
    tokenizer: Any = None
    chat_template: Any = None

    add_chat_history_to_context: bool = True
    chat_conversation: Any = []
    all_docs_start_prompt: Any = None
    all_docs_finish_prompt: Any = None
    user_prompt_for_fake_system_prompt: Any = None
    json_object_prompt: Any = None
    json_object_prompt_simpler: Any = None
    json_code_prompt: Any = None
    json_code_prompt_if_no_schema: Any = None
    json_schema_instruction: Any = None
    json_preserve_system_prompt: bool = False
    json_object_post_prompt_reminder: Any = None
    json_code_post_prompt_reminder: Any = None
    json_code2_post_prompt_reminder: Any = None

    system_prompt: Any = None
    visible_models: Any = None
    h2ogpt_key: Any = None

    image_file: Any = None
    image_control: Any = None
    images_num_max: Any = None
    image_resolution: Any = None
    image_format: Any = None
    rotate_align_resize_image: Any = None
    video_frame_period: Any = None
    image_batch_image_prompt: Any = None
    image_batch_final_prompt: Any = None
    image_batch_stream: Any = None
    visible_vision_models: Any = None
    video_file: Any = None

    response_format: Any = None
    guided_json: Any = None
    guided_regex: Any = None
    guided_choice: Any = None
    guided_grammar: Any = None
    guided_whitespace_pattern: Any = None
    client_metadata: Any = ''

    async_sem: Any = None
    count_input_tokens: Any = 0
    prompts: Any = []
    count_output_tokens: Any = 0

    min_max_new_tokens: Any = 512
    max_input_tokens: Any = -1
    max_total_input_tokens: Any = -1

    doing_grounding: bool = False

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

        try:
            if values['client'] is None:
                from gradio_utils.grclient import GradioClient
                values["client"] = GradioClient(
                    values["inference_server_url"]
                ).setup()
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

    def setup_call(self, prompt):
        # NOTE: prompt here has no prompt_type (e.g. human: bot:) prompt injection,
        # so server should get prompt_type or '', not plain
        # This is good, so gradio server can also handle stopping.py conditions
        # this is different than TGI server that uses prompter to inject prompt_type prompting
        stream_output = self.stream_output
        enable_caching = self.enable_caching
        # don't double-up langchain behavior, already did langchain part
        client_langchain_mode = LangChainMode.LLM.value
        client_add_chat_history_to_context = self.add_chat_history_to_context
        # already did search part
        client_add_search_to_context = False
        # didn't do conversation part yet
        client_chat_conversation = self.chat_conversation
        client_langchain_action = LangChainAction.QUERY.value
        client_langchain_agents = []
        top_k_docs = 1
        chunk = True
        chunk_size = 512

        prompt_type = self.prompter.prompt_type
        if self.doing_grounding:
            # avoid double prompting from grounded then normal template
            prompt_type = noop_prompt_type
            # already did conversation as part of prompt
            client_chat_conversation = []
            self.context = ''
            self.iinput = ''
            self.system_prompt = ''

        client_kwargs = dict(instruction=prompt if self.chat_client else '',  # only for chat=True
                             iinput=self.iinput if self.chat_client else '',  # only for chat=True
                             # context shouldn't include conversation!
                             context=self.context,
                             # streaming output is supported, loops over and outputs each generation in streaming mode
                             # but leave stream_output=False for simple input/output mode
                             stream_output=stream_output,
                             enable_caching=enable_caching,
                             prompt_type=prompt_type,
                             prompt_dict='',
                             chat_template=self.chat_template,

                             temperature=self.temperature,
                             top_p=self.top_p,
                             top_k=self.top_k,
                             penalty_alpha=self.penalty_alpha,
                             num_beams=self.num_beams,
                             max_new_tokens=self.max_new_tokens,
                             min_new_tokens=self.min_new_tokens,
                             early_stopping=self.early_stopping,
                             max_time=self.max_time,
                             repetition_penalty=self.repetition_penalty,
                             num_return_sequences=self.num_return_sequences,
                             do_sample=self.do_sample,
                             seed=self.seed,
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
                             document_source_substrings=[],
                             document_source_substrings_op='and',
                             document_content_substrings=[],
                             document_content_substrings_op='and',
                             pre_prompt_query=None,  # no further langchain query
                             prompt_query=None,  # no further langchain query
                             pre_prompt_summary=None,  # no further langchain summary
                             prompt_summary=None,  # no further langchain summary
                             hyde_llm_prompt=None,  # no further HYDE
                             all_docs_start_prompt=self.all_docs_start_prompt,
                             all_docs_finish_prompt=self.all_docs_finish_prompt,
                             user_prompt_for_fake_system_prompt=self.user_prompt_for_fake_system_prompt,
                             json_object_prompt=self.json_object_prompt,
                             json_object_prompt_simpler=self.json_object_prompt_simpler,
                             json_code_prompt=self.json_code_prompt,
                             json_code_prompt_if_no_schema=self.json_code_prompt_if_no_schema,
                             json_schema_instruction=self.json_schema_instruction,
                             json_preserve_system_prompt=self.json_preserve_system_prompt,
                             json_object_post_prompt_reminder=self.json_object_post_prompt_reminder,
                             json_code_post_prompt_reminder=self.json_code_post_prompt_reminder,
                             json_code2_post_prompt_reminder=self.json_code2_post_prompt_reminder,
                             system_prompt=self.system_prompt,
                             image_audio_loaders=None,  # don't need to further do doc specific things
                             pdf_loaders=None,  # don't need to further do doc specific things
                             url_loaders=None,  # don't need to further do doc specific things
                             jq_schema=None,  # don't need to further do doc specific things
                             extract_frames=10,
                             llava_prompt=None,
                             visible_models=self.visible_models,
                             h2ogpt_key=self.h2ogpt_key,
                             add_search_to_context=client_add_search_to_context,
                             chat_conversation=client_chat_conversation,
                             text_context_list=None,
                             docs_ordering_type=None,
                             min_max_new_tokens=self.min_max_new_tokens,
                             max_input_tokens=self.max_input_tokens,
                             max_total_input_tokens=self.max_total_input_tokens,
                             docs_token_handling=None,
                             docs_joiner=None,
                             hyde_level=None,
                             hyde_template=None,
                             hyde_show_only_final=None,
                             doc_json_mode=None,
                             metadata_in_context=None,

                             image_file=self.image_file,
                             image_control=self.image_control,
                             images_num_max=self.images_num_max,
                             image_resolution=self.image_resolution,
                             image_format=self.image_format,
                             rotate_align_resize_image=self.rotate_align_resize_image,
                             video_frame_period=self.video_frame_period,
                             image_batch_image_prompt=self.image_batch_image_prompt,
                             image_batch_final_prompt=self.image_batch_final_prompt,
                             image_batch_stream=self.image_batch_stream,
                             visible_vision_models=self.visible_vision_models,
                             video_file=self.video_file,

                             response_format=self.response_format,
                             guided_json=self.guided_json,
                             guided_regex=self.guided_regex,
                             guided_choice=self.guided_choice,
                             guided_grammar=self.guided_grammar,
                             guided_whitespace_pattern=self.guided_whitespace_pattern,
                             client_metadata=self.client_metadata,
                             )
        api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
        # let inner gradio count input tokens
        # self.count_input_tokens += self.get_num_tokens(str(prompt))
        # self.prompts.append(prompt)

        return client_kwargs, api_name

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        if self.verbose:
            print("_call", flush=True)

        client_kwargs, api_name = self.setup_call(prompt)
        max_new_tokens = get_relaxed_max_new_tokens(prompt, tokenizer=self.tokenizer,
                                                    max_new_tokens=self.max_new_tokens,
                                                    max_new_tokens0=self.max_new_tokens0)
        client_kwargs.update(dict(max_new_tokens=max_new_tokens))

        # new client for each call
        client = self.client.clone()
        from gradio_utils.grclient import check_job

        res_dict = {}
        if not self.stream_output:
            res = client.predict(str(dict(client_kwargs)), api_name=api_name)
            res_dict = ast.literal_eval(res)
            GradioClient.check_error(res_dict)
            text = res_dict['response']
            ret = self.prompter.get_response(prompt + text, prompt=prompt,
                                             sanitize_bot_response=self.sanitize_bot_response)
            self.count_output_tokens += self.get_num_tokens(ret)
            if self.verbose:
                print("end _call", flush=True)
            self.use_gradio_return(res_dict, prompt)

            # ensure parent client is updated if remote server changed
            if client.server_hash != self.client.server_hash:
                self.client.refresh_client()

            return ret
        else:
            text_callback = None
            if run_manager:
                text_callback = partial(
                    run_manager.on_llm_new_token, verbose=self.verbose
                )

            job = client.submit(str(dict(client_kwargs)), api_name=api_name)
            text0 = ''
            t_start = time.time()
            while not job.done():
                if job.communicator.job.latest_status.code.name == 'FINISHED':
                    break
                e = check_job(job, timeout=0, raise_exception=False)
                if e is not None:
                    break
                if self.max_time is not None and time.time() - t_start > self.max_time:
                    if self.verbose:
                        print("Exceeded max_time=%s" % self.max_time, flush=True)
                    break
                outputs_list = job.outputs().copy()
                if outputs_list:
                    res = outputs_list[-1]
                    res_dict = ast.literal_eval(res)
                    GradioClient.check_error(res_dict)
                    text = res_dict['response']
                    text = self.prompter.get_response(prompt + text, prompt=prompt,
                                                      sanitize_bot_response=self.sanitize_bot_response)
                    # FIXME: derive chunk from full for now
                    text_chunk = text[len(text0):]
                    if not text_chunk:
                        # just need some sleep for threads to switch
                        time.sleep(0.001)
                        continue
                    # save old
                    text0 = text

                    if text_callback:
                        text_callback(text_chunk)

                time.sleep(0.005)

            # ensure get last output to avoid race
            res_all = job.outputs().copy()
            success = job.communicator.job.latest_status.success
            timeout = 0.1 if success else 10
            if len(res_all) > 0:
                # don't raise unless nochat API for now
                # set below to True for now, not self.chat_client, since not handling exception otherwise
                # in some return of strex
                check_job(job, timeout=timeout, raise_exception=True)

                res = res_all[-1]
                res_dict = ast.literal_eval(res)
                GradioClient.check_error(res_dict)
                text = res_dict['response']
                # FIXME: derive chunk from full for now
            else:
                # if got no answer at all, probably something bad, always raise exception
                # UI will still put exception in Chat History under chat exceptions
                check_job(job, timeout=timeout, raise_exception=True)
                # go with old if failure
                text = text0
            text_chunk = text[len(text0):]
            if text_callback:
                text_callback(text_chunk)
            ret = self.prompter.get_response(prompt + text, prompt=prompt,
                                             sanitize_bot_response=self.sanitize_bot_response)
            self.count_output_tokens += self.get_num_tokens(ret)
            if self.verbose:
                print("end _call", flush=True)
            self.use_gradio_return(res_dict, prompt)

            # ensure parent client is updated if remote server changed
            if client.server_hash != self.client.server_hash:
                with filelock.FileLock(os.path.join('locks', 'gradio_client.lock')):
                    self.client.refresh_client()

            return ret

    def use_gradio_return(self, res_dict, prompt_raw):
        self.count_input_tokens += res_dict.get('save_dict', {}).get('extra_dict', {}).get('num_prompt_tokens',
                                                                                           self.get_num_tokens(
                                                                                               str(prompt_raw)))
        self.prompts.append(res_dict.get('prompt_raw', prompt_raw))

    # copy-paste of streaming part of _call() with asyncio.sleep instead
    async def _acall(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        if self.verbose:
            print("_acall", flush=True)

        client_kwargs, api_name = self.setup_call(prompt)

        text_callback = None
        if run_manager:
            text_callback = partial(
                run_manager.on_llm_new_token, verbose=self.verbose
            )
        # new client for each acall
        client = self.client.clone()
        from gradio_utils.grclient import check_job

        res_dict = {}
        t_start = time.time()
        job = client.submit(str(dict(client_kwargs)), api_name=api_name)
        text0 = ''
        while not job.done():
            if job.communicator.job.latest_status.code.name == 'FINISHED':
                break
            e = job.future._exception
            if e is not None:
                break
            outputs_list = job.outputs().copy()
            if outputs_list:
                res = outputs_list[-1]
                res_dict = ast.literal_eval(res)
                GradioClient.check_error(res_dict)
                text = res_dict['response']
                text = self.prompter.get_response(prompt + text, prompt=prompt,
                                                  sanitize_bot_response=self.sanitize_bot_response)
                # FIXME: derive chunk from full for now
                text_chunk = text[len(text0):]
                if not text_chunk:
                    # just need some sleep for threads to switch
                    await asyncio.sleep(0.001)
                    continue
                # save old
                text0 = text

                if text_callback:
                    await text_callback(text_chunk)

            if self.max_time is not None and time.time() - t_start > self.max_time:
                if self.verbose:
                    print("Exceeded max_time=%s" % self.max_time, flush=True)
                break

            await asyncio.sleep(0.01)

        # ensure get last output to avoid race
        res_all = job.outputs().copy()
        success = job.communicator.job.latest_status.success
        timeout = 0.1 if success else 10
        if len(res_all) > 0:
            res = res_all[-1]
            res_dict = ast.literal_eval(res)
            GradioClient.check_error(res_dict)
            text = res_dict['response']
            # FIXME: derive chunk from full for now
            check_job(job, timeout=timeout, raise_exception=True)
        else:
            # go with old if failure
            text = text0
            check_job(job, timeout=timeout, raise_exception=True)

        text_chunk = text[len(text0):]
        if text_callback:
            await text_callback(text_chunk)
        ret = self.prompter.get_response(prompt + text, prompt=prompt,
                                         sanitize_bot_response=self.sanitize_bot_response)
        self.count_output_tokens += self.get_num_tokens(ret)
        if self.verbose:
            print("end _acall", flush=True)
        self.use_gradio_return(res_dict, prompt)
        # ensure parent client is updated if remote server changed
        if client.server_hash != self.client.server_hash:
            with filelock.FileLock(os.path.join('locks', 'gradio_client.lock')):
                self.client.refresh_client()
        return ret

    def get_token_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
        # avoid base method that is not aware of how to properly tokenize (uses GPT2)
        # return _get_token_ids_default_method(text)


class GradioLLaVaInference(GradioInference):
    """
    Gradio/LLaVa generation inference API.
    """
    image_file: Any = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

        try:
            if values['client'] is None:
                from gradio_utils.grclient import GradioClient
                values["client"] = GradioClient(
                    values["inference_server_url"], check_hash=False, serialize=True,
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
        return "gradio_llava_inference"

    def setup_call(self, prompt):

        stream_output = self.stream_output
        client_kwargs = dict(instruction=prompt,
                             stream_output=stream_output,
                             prompt_type=self.prompter.prompt_type,
                             prompt_dict='',
                             chat_template=self.chat_template,

                             temperature=self.temperature,
                             top_p=self.top_p,
                             top_k=self.top_k,
                             penalty_alpha=self.penalty_alpha,
                             max_new_tokens=self.max_new_tokens,
                             min_max_new_tokens=self.min_max_new_tokens,
                             min_new_tokens=self.min_new_tokens,
                             verbose=self.verbose,
                             )
        # NOTE: Don't handle self.context
        if not self.add_chat_history_to_context:
            self.chat_conversation = []

        if self.image_file is not None:
            self.image_file = self.image_file[:llava_num_max]
            self.count_input_tokens += 1500 * len(self.image_file)
        self.count_input_tokens += self.get_num_tokens(str(prompt))
        self.prompts.append(prompt)

        llava_kwargs = dict(file=self.image_file,
                            llava_model=self.inference_server_url,
                            # prompt=instruction,
                            prompt=prompt,  # prepared prompt with chat history etc.
                            chat_conversation=self.chat_conversation,
                            allow_prompt_auto=False,
                            image_model=self.visible_models,
                            temperature=client_kwargs['temperature'],
                            top_p=client_kwargs['top_p'],
                            max_new_tokens=client_kwargs['max_new_tokens'],
                            client=self.client,
                            max_time=self.max_time,
                            tokenizer=self.tokenizer,
                            )
        max_new_tokens = get_relaxed_max_new_tokens(prompt, tokenizer=self.tokenizer,
                                                    max_new_tokens=self.max_new_tokens,
                                                    max_new_tokens0=self.max_new_tokens0)
        client_kwargs.update(dict(max_new_tokens=get_relaxed_max_new_tokens(max_new_tokens)))

        return client_kwargs, llava_kwargs

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        if self.verbose:
            print("_call", flush=True)

        _, llava_kwargs = self.setup_call(prompt)

        if not self.stream_output:
            from vision.utils_vision import get_llava_response
            response, _ = get_llava_response(**llava_kwargs)
            self.count_output_tokens += self.get_num_tokens(response)
            if self.verbose:
                print("end _call", flush=True)
            return response
        else:
            text_callback = None
            if run_manager:
                text_callback = partial(
                    run_manager.on_llm_new_token, verbose=self.verbose
                )

            t_start = time.time()
            text0 = ''
            text = ''
            from vision.utils_vision import get_llava_stream
            for text in get_llava_stream(**llava_kwargs):

                # FIXME: derive chunk from full for now
                text_chunk = text[len(text0):]
                if not text_chunk:
                    # just need some sleep for threads to switch
                    time.sleep(0.001)
                    continue
                # save old
                text0 = text
                if text_callback:
                    text_callback(text_chunk)
                time.sleep(0.005)

                if self.max_time is not None and time.time() - t_start > self.max_time:
                    if self.verbose:
                        print("Exceeded max_time=%s" % self.max_time, flush=True)
                    break

            self.count_output_tokens += self.get_num_tokens(text)
            if self.verbose:
                print("end _call", flush=True)
            return text

    # copy-paste of streaming part of _call() with asyncio.sleep instead
    async def _acall(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        if self.verbose:
            print("_acall", flush=True)

        _, llava_kwargs = self.setup_call(prompt)

        text_callback = None
        if run_manager:
            text_callback = partial(
                run_manager.on_llm_new_token, verbose=self.verbose
            )

        t_start = time.time()
        text0 = ''
        text = ''
        from vision.utils_vision import get_llava_stream
        for text in get_llava_stream(**llava_kwargs):

            # FIXME: derive chunk from full for now
            text_chunk = text[len(text0):]
            if not text_chunk:
                # just need some sleep for threads to switch
                await asyncio.sleep(0.001)
                continue
            # save old
            text0 = text
            if text_callback:
                await text_callback(text_chunk)

            if self.max_time is not None and time.time() - t_start > self.max_time:
                if self.verbose:
                    print("Exceeded max_time=%s" % self.max_time, flush=True)
                break
            await asyncio.sleep(0.005)

        self.count_output_tokens += self.get_num_tokens(text)
        if self.verbose:
            print("end _acall", flush=True)
        return text


class SGlangInference(AGenerateStreamFirst, H2Oagenerate, LLM):
    """
    SGLang generation inference API.
    """
    inference_server: str = ""
    inference_server_url: str = ""
    temperature: float = 0.8
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = None
    penalty_alpha: Optional[float] = 0.0
    num_beams: Optional[int] = 1
    max_new_tokens: int = 512
    max_new_tokens0: int = 512
    min_new_tokens: int = 1
    early_stopping: bool = False
    max_time: int = 180
    repetition_penalty: Optional[float] = None
    do_sample: bool = False
    seed: int = 0

    stream_output: bool = False

    context: Any = ''
    tokenizer: Any = None

    chat_conversation: Any = []
    add_chat_history_to_context: bool = True
    user_prompt_for_fake_system_prompt: Any = None
    prompter: Any = None

    system_prompt: Any = None
    visible_models: Any = None
    h2ogpt_key: Any = None

    image_file: Any = None
    image_control: Any = None
    images_num_max: Any = None
    image_resolution: Any = None
    image_format: Any = None
    rotate_align_resize_image: Any = None
    video_frame_period: Any = None
    image_batch_image_prompt: Any = None
    image_batch_final_prompt: Any = None
    image_batch_stream: Any = None
    visible_vision_models: Any = None
    video_file: Any = None

    async_sem: Any = None
    count_input_tokens: Any = 0
    prompts: Any = []
    count_output_tokens: Any = 0

    # runtime
    assistant_role: Any = None
    user_role: Any = None
    conv_template_before_prompt: Any = None
    url: Any = None
    pload: Any = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""

        try:
            import llava
        except ImportError:
            raise ImportError(
                "Could not import llava python package. "
                "Please install it with `pip install https://h2o-release.s3.amazonaws.com/h2ogpt/llava-1.7.0.dev0-py3-none-any.whl`."
            )
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "sglang_inference"

    def get_token_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
        # avoid base method that is not aware of how to properly tokenize (uses GPT2)
        # return _get_token_ids_default_method(text)

    @staticmethod
    def get_conv_template(conv_template_name):
        # /home/jon/miniconda3/envs/h2ogpt/lib/python3.10/site-packages/llava/conversation.py
        conversation_module = importlib.import_module("llava.conversation")
        conv_template = copy.deepcopy(getattr(conversation_module, conv_template_name))
        return conv_template

    async def send_request(self, url, data, delay=0, timeout=None):
        if timeout is None:
            timeout = self.max_time
        await asyncio.sleep(delay)
        timeout_settings = aiohttp.ClientTimeout(total=timeout)  # Set the total timeout
        async_sem = AsyncNullContext() if self.async_sem is None else self.async_sem
        async with async_sem:  # semaphore limits num of simultaneous downloads
            async with aiohttp.ClientSession(timeout=timeout_settings) as session:
                async with session.post(url, json=data) as resp:
                    print("headers: %s" % resp.headers, flush=True)
                    if resp.headers['Content-Type'] == 'application/json':
                        output = await resp.json()
                    else:
                        output_text = await resp.text()
                        output = {"text": output_text}
                        if resp.status == 504:
                            print(f"504 Response received from {url}: {output}", flush=True)
                            raise TimeoutError(output_text)
        return output

    def setup_call(self, prompt):
        # NOTE: Don't handle self.context
        if not self.add_chat_history_to_context:
            self.chat_conversation = []

        if self.image_file is not None:
            self.image_file = self.image_file[:llava_num_max]
            self.count_input_tokens += 1500 * len(self.image_file)

        conv_template_name = self.inference_server.split(':')[1]
        conv_template = self.get_conv_template(conv_template_name)
        self.user_role = conv_template.roles[0]
        self.assistant_role = conv_template.roles[1]
        if self.system_prompt:
            if not conv_template.system:
                # assume means can't handle if didn't exist in template
                conv_template.append_message(role=self.user_role, message=self.user_prompt_for_fake_system_prompt)
                if self.system_prompt == 'auto':
                    self.system_prompt = 'You are a helpful assistant.' if not self.image_file else "You are helpful visual LLM assistant capable of understanding text and images."
                conv_template.append_message(role=self.assistant_role, message=self.system_prompt)
            else:
                our_system_prompt = False
                if our_system_prompt:
                    # FIXME: our own system prompt
                    if self.system_prompt == 'auto':
                        self.system_prompt = conv_template.system
                    if '<|im_start|>system\n' in conv_template.system:
                        conv_template.system = '<|im_start|>system\n' + self.system_prompt
                    elif conv_template.system == "":
                        conv_template.append_message(role="system", message=self.system_prompt)
        for message in self.chat_conversation:
            if isinstance(message[0], str) and message[0]:
                conv_template.append_message(role=self.user_role, message=message[0])
            if isinstance(message[1], str) and message[1]:
                conv_template.append_message(role=self.assistant_role, message=message[1])

        self.conv_template_before_prompt = copy.deepcopy(conv_template)

        prompt_with_image = f"<image>\n{prompt}"
        conv_template.append_message(role=self.user_role, message=prompt_with_image)
        conv_template.append_message(role=self.assistant_role, message=None)
        prompt_with_template = conv_template.get_prompt()
        if self.context:
            prompt_with_template = self.context + prompt_with_template
        self.prompts.append(prompt_with_template)
        self.count_input_tokens += self.get_num_tokens(str(prompt_with_template))
        presence_penalty = (self.repetition_penalty - 1.0) * 2.0 + 0.0  # so good default

        terminate_response = update_terminate_responses([], tokenizer=self.tokenizer)
        self.pload = {
            "text": prompt_with_template,
            "sampling_params": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "presence_penalty": presence_penalty,
                "frequency_penalty": 2,
                "stop": terminate_response,
            },
            "image_data": self.image_file[0],
            "stream": self.stream_output,
        }
        self.url = self.inference_server_url + "/generate"

    def do_many(self):
        # deal with all images
        # also contains prompt_tokens, completion_tokens, finish_reason, etc.
        return asyncio.run(self.get_many(self.url, self.pload))

    async def a_do_many(self):
        return await self.get_many(self.url, self.pload)

    def many_to_prompt(self, prompt, responses):
        if len(self.image_file) > 1:
            # now use all those in final prompt
            responses_context = '\n\n'.join(['# Image %d Answer\n\n%s\n\n' % (i, r['text']) for i, r in
                                             enumerate(responses)])
            prompt_with_responses = f"{responses_context}\n{prompt}"
            self.conv_template_before_prompt.append_message(role=self.user_role, message=prompt_with_responses)
            self.conv_template_before_prompt.append_message(role=self.assistant_role, message=None)
            prompt_with_template = self.conv_template_before_prompt.get_prompt()
            if self.context:
                prompt_with_template = self.context + prompt_with_template
            self.prompts.append(prompt_with_template)

            # update pload
            self.pload['text'] = prompt_with_template  # prompt now has response per image as single prompt
            self.pload.pop('image_data')  # no longer have images, just text

    def do_final(self):
        return requests.post(
            self.url,
            json=self.pload,
            stream=self.stream_output,
        )

    async def get_many(self, url, pload):
        pload_no_image = pload.copy()
        pload_no_image.pop('image_data')
        pload_no_image.pop('stream')  # so stays json not text stream

        responses = []
        for image_1 in self.image_file:
            pload_i = copy.deepcopy(pload_no_image)
            pload_i['image_data'] = image_1
            responses.append(self.send_request(url, pload_i))
        rets = await asyncio.gather(*responses)
        return rets

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        if self.verbose:
            print("_call", flush=True)

        self.setup_call(prompt)
        if len(self.image_file) > 1:
            responses = self.do_many()
            self.many_to_prompt(prompt, responses)
        response = self.do_final()

        if not self.stream_output:
            response = response.json()['text']
            self.count_output_tokens += self.get_num_tokens(response)
            if self.verbose:
                print("end _call", flush=True)
            return response
        else:
            text_callback = None
            if run_manager:
                text_callback = partial(
                    run_manager.on_llm_new_token, verbose=self.verbose
                )

            t_start = time.time()
            text = ''
            prev = 0
            for chunk in response.iter_lines(decode_unicode=False):
                chunk = chunk.decode("utf-8")
                if chunk and chunk.startswith("data:"):
                    if chunk == "data: [DONE]":
                        break
                    data = json.loads(chunk[5:].strip("\n"))
                    output = data["text"].strip()
                    text_chunk = output[prev:]
                    text += text_chunk
                    prev = len(output)

                    if not text_chunk:
                        # just need some sleep for threads to switch
                        time.sleep(0.001)
                        continue
                    if text_callback:
                        text_callback(text_chunk)
                time.sleep(0.005)

                if self.max_time is not None and time.time() - t_start > self.max_time:
                    if self.verbose:
                        print("Exceeded max_time=%s" % self.max_time, flush=True)
                    break

            self.count_output_tokens += self.get_num_tokens(text)
            if self.verbose:
                print("end _call", flush=True)
            return text

    # copy-paste of streaming part of _call() with asyncio.sleep instead
    async def _acall(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        if self.verbose:
            print("_call", flush=True)

        self.setup_call(prompt)
        if len(self.image_file) > 1:
            responses = await self.a_do_many()
            self.many_to_prompt(prompt, responses)
        response = self.do_final()

        if not self.stream_output:
            response = response.json()['text']
            self.count_output_tokens += self.get_num_tokens(response)
            if self.verbose:
                print("end _acall", flush=True)
            return response
        else:
            text_callback = None
            if run_manager:
                text_callback = partial(
                    run_manager.on_llm_new_token, verbose=self.verbose
                )

            t_start = time.time()
            text = ''
            prev = 0
            for chunk in response.iter_lines(decode_unicode=False):
                chunk = chunk.decode("utf-8")
                if chunk and chunk.startswith("data:"):
                    if chunk == "data: [DONE]":
                        break
                    data = json.loads(chunk[5:].strip("\n"))
                    output = data["text"].strip()
                    text_chunk = output[prev:]
                    text += text_chunk
                    prev = len(output)

                    if not text_chunk:
                        # just need some sleep for threads to switch
                        await asyncio.sleep(0.001)
                        continue
                    if text_callback:
                        text_callback(text_chunk)
                await asyncio.sleep(0.005)

                if self.max_time is not None and time.time() - t_start > self.max_time:
                    if self.verbose:
                        print("Exceeded max_time=%s" % self.max_time, flush=True)
                    break

            self.count_output_tokens += self.get_num_tokens(text)
            if self.verbose:
                print("end _acall", flush=True)
            return text


class H2OHuggingFaceTextGenInference(AGenerateStreamFirst, H2Oagenerate, HuggingFaceTextGenInference):
    max_new_tokens: int = 512
    do_sample: bool = False
    seed: int = 0
    top_p: Optional[float] = 0.99
    top_k: Optional[int] = None
    penalty_alpha: Optional[float] = 0.0
    typical_p: Optional[float] = 0.95
    temperature: float = 0.0
    repetition_penalty: Optional[float] = None
    return_full_text: bool = False
    stop_sequences: List[str] = Field(default_factory=list)
    inference_server_url: str = ""
    timeout: int = 300
    headers: dict = None
    stream_output: bool = False
    sanitize_bot_response: bool = False
    prompter: Any = None
    context: Any = ''
    iinput: Any = ''
    tokenizer: Any = None
    chat_conversation: Any = []
    user_prompt_for_fake_system_prompt: Any = None
    async_sem: Any = None
    count_input_tokens: Any = 0
    prompts: Any = []
    count_output_tokens: Any = 0

    base_model: Any = ''
    image_file: Any = None
    image_control: Any = None
    images_num_max: Any = None
    image_resolution: Any = None
    image_format: Any = None
    rotate_align_resize_image: Any = None
    video_frame_period: Any = None
    image_batch_image_prompt: Any = None
    image_batch_final_prompt: Any = None
    image_batch_stream: Any = None
    visible_vision_models: Any = None
    video_file: Any = None

    def prep_prompt(self, prompt, stop, kwargs):
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
        prompt = self.prompter.generate_prompt(data_point,
                                               chat_conversation=self.chat_conversation,
                                               user_prompt_for_fake_system_prompt=self.user_prompt_for_fake_system_prompt,
                                               image_file=self.image_file,
                                               )
        self.count_input_tokens += self.get_num_tokens(str(prompt))
        self.prompts.append(prompt)

        if self.image_file:
            prompt = ''.join([f'![]({x})!' for x in self.image_file]) + prompt
            self.count_input_tokens += 64 * len(self.image_file)

            data_point = dict(context=self.context, instruction=prompt, input=self.iinput)
            prompt = self.prompter.generate_prompt(data_point,
                                                   chat_conversation=self.chat_conversation,
                                                   user_prompt_for_fake_system_prompt=self.user_prompt_for_fake_system_prompt,
                                                   )

        gen_server_kwargs = dict(do_sample=self.do_sample,
                                 seed=self.seed,
                                 stop_sequences=stop,
                                 max_new_tokens=self.max_new_tokens,
                                 top_p=self.top_p,
                                 top_k=self.top_k,
                                 typical_p=self.typical_p,
                                 # penalty_alpha=self.penalty_alpha,
                                 temperature=self.temperature,
                                 repetition_penalty=self.repetition_penalty,
                                 return_full_text=self.return_full_text,
                                 )
        gen_server_kwargs.update(kwargs)

        return prompt, gen_server_kwargs, stop

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        prompt, gen_server_kwargs, stop = self.prep_prompt(prompt, stop, kwargs)

        # lower bound because client is re-used if multi-threading
        self.client.timeout = max(300, self.timeout)

        if not self.stream_output:
            res = self.client.generate(
                prompt,
                **gen_server_kwargs,
            )
            if self.return_full_text:
                assert not self.image_file, "Invalid use of image files with HF client"
                gen_text = res.generated_text[len(prompt):]
            else:
                gen_text = res.generated_text
            # remove stop sequences from the end of the generated text
            for stop_seq in stop:
                if stop_seq in gen_text:
                    gen_text = gen_text[:gen_text.index(stop_seq)]
            text = gen_text
        else:
            text_callback = None
            if run_manager:
                text_callback = partial(
                    run_manager.on_llm_new_token, verbose=self.verbose
                )
            text = ""
            # Note: Streaming ignores return_full_text=True
            for response in self.client.generate_stream(prompt, **gen_server_kwargs):
                text_chunk = response.token.text
                text += text_chunk
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
        if self.verbose:
            print("acall", flush=True)

        prompt, gen_server_kwargs, stop = self.prep_prompt(prompt, stop, kwargs)

        gen_text = await super()._acall(prompt, stop=stop, run_manager=run_manager, **kwargs)

        # remove stop sequences from the end of the generated text
        for stop_seq in stop:
            if stop_seq in gen_text:
                gen_text = gen_text[:gen_text.index(stop_seq)]
        text = gen_text
        if self.verbose:
            print("acall done", flush=True)
        return text

    def get_token_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
        # avoid base method that is not aware of how to properly tokenize (uses GPT2)
        # return _get_token_ids_default_method(text)


from langchain_community.chat_models import ChatAnthropic as ChatAnthropic2, ChatOpenAI
from langchain_anthropic import ChatAnthropic as ChatAnthropic3
from langchain_community.llms import OpenAI, AzureOpenAI, Replicate
from langchain_together import ChatTogether


class H2OTextGenOpenAI:
    def update_prompts_and_stops(self, prompts, stop, **kwargs):
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
            context_from_history = len(self.chat_conversation) > 0
            image_file = []  # FIXME: not supported, should use chat API for images via OpenAI API
            prompt = self.prompter.generate_prompt(data_point,
                                                   chat_conversation=self.chat_conversation,
                                                   user_prompt_for_fake_system_prompt=self.user_prompt_for_fake_system_prompt,
                                                   context_from_history=context_from_history,
                                                   image_file=image_file,
                                                   )
            prompts[prompti] = prompt

        kwargs = self.update_kwargs(prompts, kwargs)
        return prompts, stop, kwargs

    def update_kwargs(self, prompts, kwargs):
        # update kwargs per llm use, for when llm re-used for multiple prompts like summarization/extraction
        # relax max_new_tokens if can
        if self.max_new_tokens0 is not None and \
                self.max_new_tokens0 > self.max_tokens and \
                len(prompts) == 1 and \
                'max_tokens' not in kwargs:
            kwargs.update(dict(max_tokens=self.max_tokens_for_prompt(prompts[0])))
        return kwargs

    def max_tokens_for_prompt(self, prompt: str) -> int:
        # like super() OpenAI version but added limit
        num_tokens = self.get_num_tokens(str(prompt))
        if self.max_new_tokens0 is not None:
            return max(128, min(self.max_new_tokens0, self.tokenizer.model_max_length - num_tokens))
        else:
            return max(128, self.max_context_size - num_tokens)

    def count_out_tokens(self, rets):
        try:
            self.count_output_tokens += sum(
                [self.get_num_tokens(z) for z in flatten_list([[x.text for x in y] for y in rets.generations])])
        except Exception as e:
            if os.getenv('HARD_ASSERTS'):
                raise
            print("Failed to get total output tokens\n%s\n" % traceback.format_exc())

    def collect_llm_results(self, rets):
        generations = [x.generations[0] for x in rets]

        def reducer(accumulator, element):
            for key, value in element.items():
                accumulator[key] = accumulator.get(key, 0) + value
            return accumulator

        collection = [x.llm_output['token_usage'] for x in rets]
        token_usage = reduce(reducer, collection, {})

        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        self.count_output_tokens += token_usage.get('completion_tokens', 0)
        if self.count_output_tokens == 0:
            self.count_output_tokens += sum([self.get_num_tokens(x[0].text) for x in generations if len(x) > 0])
        return LLMResult(generations=generations, llm_output=llm_output)

    def _generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
        if self.verbose:
            print("Hit _generate", flush=True)
        prompts, stop, kwargs = self.update_prompts_and_stops(prompts, stop, **kwargs)
        self.count_input_tokens += sum([self.get_num_tokens(str(prompt)) for prompt in prompts])
        self.count_llm_calls += len(prompts)
        self.prompts.extend(prompts)
        if self.batch_size > 1:
            rets = super()._generate(prompts, stop=stop, run_manager=run_manager, **kwargs)
            self.count_out_tokens(rets)
        else:
            rets = []
            for sub_prompt in prompts:
                rets1 = super()._generate([sub_prompt], stop=stop, run_manager=run_manager, **kwargs)
                rets.append(rets1)
            rets = self.collect_llm_results(rets)  # counts output tokens already

        # handle fact that multi-character stops will only stop streaming once last matching character, then we get rest
        if stop is None:
            stop = []
        all_stops = stop.copy() if stop is not None else []
        for stop_seq in all_stops:
            if len(stop_seq) > 6:
                stop.append(stop_seq[:6])

        for gens in rets.generations:
            for genobj in gens:
                gen_text = genobj.text
                for stop_seq in stop:
                    if stop_seq in gen_text:
                        genobj.text = gen_text[:gen_text.index(stop_seq)]
        return rets

    def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> typing.Iterator[GenerationChunk]:
        kwargs = self.update_kwargs([prompt], kwargs)
        return super()._stream(prompt, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> typing.AsyncIterator[GenerationChunk]:
        kwargs = self.update_kwargs([prompt], kwargs)
        async for chunk in super()._astream(prompt, stop=stop, run_manager=run_manager, **kwargs):
            yield chunk

    async def _agenerate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
        prompts, stop, kwargs = self.update_prompts_and_stops(prompts, stop, **kwargs)
        self.count_input_tokens += sum([self.get_num_tokens(str(prompt)) for prompt in prompts])
        self.count_llm_calls += len(prompts)

        if self.streaming:
            self.prompts.extend(prompts)
            run_managers = [run_manager] * len(prompts)
            # only stream first if doing async
            run_managers = [x if i == 0 else None for i, x in enumerate(run_managers)]
            tasks = [
                asyncio.ensure_future(self._agenerate_one(prompt, stop=stop, run_manager=run_manager1, **kwargs))
                for run_manager1, prompt in zip(run_managers, prompts)]
            llm_results = await asyncio.gather(*tasks)
            return self.collect_llm_results(llm_results)
        elif self.batch_size > 1:
            rets = await super()._agenerate(prompts, stop=stop, run_manager=run_manager, **kwargs)
            self.count_out_tokens(rets)
            return rets
        else:
            self.prompts.extend(prompts)
            tasks = [
                asyncio.ensure_future(self._agenerate_one(prompt, stop=stop, run_manager=run_manager, **kwargs))
                for prompt in prompts]
            llm_results = await asyncio.gather(*tasks)
            return self.collect_llm_results(llm_results)

    async def _agenerate_one(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
        async_sem = AsyncNullContext() if self.async_sem is None else self.async_sem
        async with async_sem:  # semaphore limits num of simultaneous downloads
            prompts = [prompt]
            # update for each async call
            kwargs = self.update_kwargs(prompts, kwargs)
            return await super()._agenerate(prompts, stop=stop, run_manager=run_manager, **kwargs)

    def get_token_ids(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)
        else:
            # OpenAI uses tiktoken
            return super().get_token_ids(text)


class H2OOpenAI(H2OTextGenOpenAI, OpenAI):
    temperature: float = 0.0
    """
    New class to handle vLLM's use of OpenAI, no vllm_chat supported, so only need here
    Handles prompting that OpenAI doesn't need, stopping as well

    assume stop is used to keep out trailing text, and only generate new text,
    so don't use self.prompter.get_response as becomes too complex
    """
    stop_sequences: Any = None
    sanitize_bot_response: bool = False
    prompter: Any = None
    context: Any = ''
    iinput: Any = ''
    tokenizer: Any = None
    chat_conversation: Any = []
    user_prompt_for_fake_system_prompt: Any = None
    async_sem: Any = None
    count_input_tokens: Any = 0
    prompts: Any = []
    count_output_tokens: Any = 0
    max_new_tokens0: Any = None
    count_llm_calls: Any = 0


class H2OReplicate(Replicate):
    stop_sequences: Any = None
    sanitize_bot_response: bool = False
    prompter: Any = None
    context: Any = ''
    iinput: Any = ''
    tokenizer: Any = None
    chat_conversation: Any = []
    user_prompt_for_fake_system_prompt: Any = None
    prompts: Any = []

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
        # Note Replicate handles the prompting of the specific model, but not if history, so just do it all on our side
        data_point = dict(context=self.context, instruction=prompt, input=self.iinput)
        image_file = []
        prompt = self.prompter.generate_prompt(data_point,
                                               chat_conversation=self.chat_conversation,
                                               user_prompt_for_fake_system_prompt=self.user_prompt_for_fake_system_prompt,
                                               image_file=image_file,
                                               )

        response = super()._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
        return response


class ExtraChat:
    def get_token_ids(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            if isinstance(self.tokenizer, FakeTokenizer):
                return self.tokenizer.encode(text)['input_ids']
            else:
                ret = self.tokenizer.encode(text)
                if hasattr(ret, 'input_ids') and isinstance(ret, dict):
                    ret = ret['input_ids']
                return ret
        else:
            return FakeTokenizer().encode(text)['input_ids']

    def get_messages(self, prompts):
        from langchain.schema import AIMessage, SystemMessage, HumanMessage
        messages = []
        count_input_tokens_start = self.count_input_tokens
        if self.system_prompt:
            if isinstance(self, (H2OChatAnthropic2, H2OChatGoogle)) and not isinstance(self,
                                                                                       H2OChatAnthropic2Sys) or not self.prompter.can_handle_system_prompt:
                user_prompt_for_fake_system_prompt = self.user_prompt_for_fake_system_prompt or user_prompt_for_fake_system_prompt0
                self.chat_conversation = [[user_prompt_for_fake_system_prompt,
                                           self.system_prompt]] + self.chat_conversation
            else:
                self.count_input_tokens += self.get_num_tokens(str(self.system_prompt))
                messages.append(SystemMessage(content=self.system_prompt))
        img_base64 = None
        img_tag = None
        if self.chat_conversation:
            for messages1 in self.chat_conversation:
                if len(messages1) != 2:
                    continue
                if len(messages1) == 2 and (messages1[0] is None or messages1[1] is None):
                    # then not really part of LLM, internal, so avoid
                    continue
                if messages1[1] in [claude3imagetag, gpt4imagetag, geminiimagetag, gemini15imagetag]:
                    img_tag = messages1[1]
                    img_base64 = messages1[0]
                    continue
                if messages1[0]:
                    instruction = gradio_to_llm(messages1[0], bot=False)
                    messages.append(HumanMessage(content=instruction))
                    self.count_input_tokens += self.get_num_tokens(str(instruction))
                if messages1[1]:
                    output = gradio_to_llm(messages1[1], bot=True)
                    messages.append(AIMessage(content=output))
                    self.count_input_tokens += self.get_num_tokens(str(output))
        if isinstance(self, H2OChatGoogle) and img_base64 is not None:
            # Multiturn chat is not enabled for models/gemini-pro-vision
            messages = []
            self.count_input_tokens = count_input_tokens_start
        prompt_messages = []
        for prompt in prompts:
            if isinstance(prompt, ChatPromptValue):
                prompt_message = messages + prompt.messages
                assert img_base64 is None, "img_base64 was filled, unused"
            else:
                prompt_text = prompt.text if prompt.text is not None else ''
                if img_base64:
                    if isinstance(img_base64, str):
                        img_base64 = [img_base64]
                    assert isinstance(img_base64, list)
                    # https://docs.anthropic.com/claude/docs/vision
                    # https://python.langchain.com/docs/integrations/chat/anthropic
                    # could also be type "image" and add "source" with other details
                    # also valid for gpt-4-vision: https://community.openai.com/t/using-gpt-4-vision-preview-in-langchain/549393
                    # https://python.langchain.com/docs/integrations/chat/google_generative_ai
                    # https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_pro_vision_python.ipynb
                    content = []
                    num_images = 0
                    for img_base64_one in img_base64:
                        if img_tag in [geminiimagetag, gemini15imagetag]:
                            img_url = img_base64_one
                        else:
                            img_url = {
                                "url": img_base64_one,
                            }
                            # https://platform.openai.com/docs/guides/vision
                            if img_tag in [gpt4imagetag]:
                                img_url['detail'] = 'high'
                        content.append({
                            "type": "image_url",
                            "image_url": img_url,
                        })

                        # estimate cost, assuming usually use about 1kx1k
                        if img_tag in [claude3imagetag]:
                            # https://docs.anthropic.com/claude/docs/vision#image-costs
                            # for roughly 1kx1k image
                            self.count_input_tokens += claude3_image_tokens
                        if img_tag in [geminiimagetag, gemini15imagetag]:
                            # https://cloud.google.com/vertex-ai/generative-ai/pricing
                            # gemini gives $ cost per image, not by tokens, just estimate
                            # $0.0025 per image and $0.000125/1k tokens, 4 chars/token, so image like 20k chars or 5k tokens
                            self.count_input_tokens += gemini_image_tokens
                        if img_tag in [gpt4imagetag]:
                            # https://openai.com/pricing
                            # for 1kx1k costs $0.00765 while $10/M tokens, so image is like 765 tokens
                            self.count_input_tokens += gpt4_image_tokens

                        num_images += 1
                        hard_truncate = False
                        # do this elsewhere to allow API flexibility as well
                        if hard_truncate:
                            if img_tag in [geminiimagetag] and num_images >= geminiimage_num_max:
                                break
                            if img_tag in [gemini15imagetag] and num_images >= gemini15image_num_max:
                                break
                            if img_tag in [gpt4imagetag] and num_images >= gpt4image_num_max:
                                break
                            if img_tag in [claude3imagetag] and num_images >= claude3image_num_max:
                                break
                    # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/design-multimodal-prompts
                    # gemini recommends images come first before text
                    content.append({"type": "text", "text": prompt_text})
                    self.count_input_tokens += self.get_num_tokens(str(prompt_text))
                else:
                    content = prompt_text
                    self.count_input_tokens += self.get_num_tokens(str(prompt_text))
                prompt_message = HumanMessage(content=content)
                prompt_message = messages + [prompt_message]
            prompt_messages.append(prompt_message)
        if self.verbose:
            print('count_input_tokens for %s: %s' % (str(self.__class__.__name__), self.count_input_tokens), flush=True)
        return prompt_messages

    def get_num_tokens(self, text: str) -> int:
        tokens = self.get_token_ids(text)
        if isinstance(tokens, list):
            return len(tokens)
        elif isinstance(tokens, dict) and 'input_ids' in tokens:
            return len(tokens['input_ids'])
        else:
            return len(tokens)


class GenerateStream:
    def get_count_output_tokens(self, ret):
        if hasattr(ret, 'llm_output') and 'model_name' in ret.llm_output and ret.llm_output['model_name'] in ['o1-mini',
                                                                                                              'o1-preview']:
            usage_dict = ret.llm_output['token_usage']
            if 'completion_tokens' in usage_dict:
                self.count_output_tokens += usage_dict['completion_tokens']
            if 'completion_tokens_details' in usage_dict and 'reasoning_tokens' in usage_dict[
                'completion_tokens_details']:
                print("reasoning tokens for %s: %s" % (
                ret.llm_output['model_name'], usage_dict['completion_tokens_details']['reasoning_tokens']))

    def generate_prompt(
            self,
            prompts: List[PromptValue],
            stop: Optional[List[str]] = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> LLMResult:
        self.prompts.extend(prompts)
        prompt_messages = self.get_messages(prompts)
        if 'streaming' not in kwargs:
            kwargs['streaming'] = self.streaming
        # prompt_messages = [p.to_messages() for p in prompts]
        try:
            ret = self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)
            self.get_count_output_tokens(ret)
            return ret
        except Exception as e:
            if 'Internal server error' in str(e):
                print("Internal server error, retrying", flush=True)
                time.sleep(5)
                return self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)
            else:
                raise

    async def agenerate_prompt(
            self,
            prompts: List[PromptValue],
            stop: Optional[List[str]] = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> LLMResult:
        self.prompts.extend(prompts)
        prompt_messages = self.get_messages(prompts)
        # prompt_messages = [p.to_messages() for p in prompts]
        if 'streaming' not in kwargs:
            kwargs['streaming'] = self.streaming
        ret = await self.agenerate(
            prompt_messages, stop=stop, callbacks=callbacks, **kwargs
        )
        self.get_count_output_tokens(ret)
        return ret

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            stream: Optional[bool] = None,
            **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        have_tool = False
        if hasattr(self, 'model_kwargs') and 'tools' in self.model_kwargs:
            should_stream = False
            have_tool = True
        kwargs.pop('stream', None)
        kwargs.pop('streaming', None)
        # gemini specific:
        if hasattr(self, 'safety_settings'):
            # google
            kwargs['safety_settings'] = self.safety_settings
        # gemini specific:
        if hasattr(self, 'response_format') and self.response_format == 'json_object':
            kwargs['generation_config'] = dict(response_mime_type='application/json')
            if self.guided_json and isinstance(self.guided_json, dict) and self.model == 'models/gemini-1.5-pro-latest':
                # flash doesn't support, has to be part of prompt
                kwargs['generation_config'].update(dict(response_schema=convert_to_genai_schema(self.guided_json)))
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        else:
            ret = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            return self.tool_string_return(ret, have_tool)

    def tool_string_return(self, ret, have_tool):
        if have_tool and isinstance(ret.generations[0].text, list):
            # prior beta
            # overwrite
            if not ret.generations[0].text:
                ret.generations[0].text = json.dumps({})
            else:
                # -1 is last, to skip first thinking step for opus/sonnet
                # bug in claude with sonnet:
                result = ret.generations[0].text[-1]['input']
                if isinstance(result, dict) and len(result) == 1 and 'properties' in result:
                    result = result['properties']
                ret.generations[0].text = json.dumps(result)
        elif have_tool and isinstance(ret.generations[0].message.content, list):
            # new beta
            # overwrite
            if not ret.generations[0].message.content:
                ret.generations[0].text = json.dumps({})
            else:
                # -1 is last, to skip first thinking step for opus/sonnet
                # bug in claude with sonnet:
                if 'input' in ret.generations[0].message.content[-1] and ret.generations[0].message.content[-1][
                    'input']:
                    result = ret.generations[0].message.content[-1]['input']
                elif 'partial_json' in ret.generations[0].message.content[-1] and \
                        ret.generations[0].message.content[-1]['partial_json']:
                    result = json.loads(ret.generations[0].message.content[-1]['partial_json'])
                else:
                    result = {}
                if isinstance(result, dict) and len(result) == 1 and 'properties' in result:
                    result = result['properties']
                ret.generations[0].text = json.dumps(result)
        return ret

    async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            stream: Optional[bool] = None,
            **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        have_tool = False
        if hasattr(self, 'model_kwargs') and 'tools' in self.model_kwargs:
            should_stream = False
            have_tool = True
        kwargs.pop('stream', None)
        kwargs.pop('streaming', None)
        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)
        else:
            ret = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            return self.tool_string_return(ret, have_tool)


class GenerateNormal:
    def get_count_output_tokens(self, ret):
        if hasattr(ret, 'llm_output') and 'model_name' in ret.llm_output and ret.llm_output['model_name'] in ['o1-mini',
                                                                                                              'o1-preview']:
            usage_dict = ret.llm_output['token_usage']
            if 'completion_tokens' in usage_dict:
                self.count_output_tokens += usage_dict['completion_tokens']
            if 'completion_tokens_details' in usage_dict and 'reasoning_tokens' in usage_dict[
                'completion_tokens_details']:
                print("reasoning tokens for %s: %s" % (
                ret.llm_output['model_name'], usage_dict['completion_tokens_details']['reasoning_tokens']))

    def generate_prompt(
            self,
            prompts: List[PromptValue],
            stop: Optional[List[str]] = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> LLMResult:
        self.prompts.extend(prompts)
        prompt_messages = self.get_messages(prompts)
        # prompt_messages = [p.to_messages() for p in prompts]
        ret = self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)
        self.get_count_output_tokens(ret)
        return ret

    async def agenerate_prompt(
            self,
            prompts: List[PromptValue],
            stop: Optional[List[str]] = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> LLMResult:
        self.prompts.extend(prompts)
        prompt_messages = self.get_messages(prompts)
        # prompt_messages = [p.to_messages() for p in prompts]
        ret = await self.agenerate(
            prompt_messages, stop=stop, callbacks=callbacks, **kwargs
        )
        self.get_count_output_tokens(ret)
        return ret


class GenerateStream2:
    def count_out_tokens(self, rets):
        if rets is None:
            return
        try:
            self.count_output_tokens += sum(
                [self.get_num_tokens(z) for z in flatten_list([[x.text for x in y] for y in rets.generations])])
        except Exception as e:
            if os.getenv('HARD_ASSERTS'):
                raise
            print("Failed to get total output tokens\n%s\n" % traceback.format_exc())

    def pre_generate(self, prompts):
        if prompts and isinstance(prompts, list) and isinstance(prompts[0], str):
            self.prompts.extend(prompts)
        else:
            self.prompts.extend([x.text for x in prompts])
        self.count_input_tokens += sum([self.get_num_tokens(str(prompt)) for prompt in prompts])

    # slightly different from GenerateStream
    def generate_prompt(
            self,
            prompts: List[PromptValue],
            stop: Optional[List[str]] = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> LLMResult:
        prompt_messages = self.get_messages(prompts)
        self.pre_generate(prompts)
        # prompt_messages = [p.to_messages() for p in prompts]
        if self.stream_output:
            kwargs.update(dict(stream=True))
        if self.response_format == 'json_object':
            kwargs.update(dict(response_format=self.response_format))
        rets = None
        try:
            rets = self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)
        except Exception as e:
            t, v, tb = sys.exc_info()
            ex = ''.join(traceback.format_exception(t, v, tb))
            if 'assert generation is not None' in str(ex) or 'Input should be' in str(ex):
                # try one more time
                rets = self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)
            else:
                raise
        finally:
            self.count_out_tokens(rets)
        return rets

    async def agenerate_prompt(
            self,
            prompts: List[PromptValue],
            stop: Optional[List[str]] = None,
            callbacks: Callbacks = None,
            **kwargs: Any,
    ) -> LLMResult:
        prompt_messages = self.get_messages(prompts)
        self.pre_generate(prompts)
        # prompt_messages = [p.to_messages() for p in prompts]
        if self.stream_output:
            kwargs.update(dict(stream=True))
        if self.response_format == 'json_object':
            kwargs.update(dict(response_format=self.response_format))
        rets = None
        try:
            rets = await self.agenerate(
                prompt_messages, stop=stop, callbacks=callbacks, **kwargs
            )
        except Exception as e:
            t, v, tb = sys.exc_info()
            ex = ''.join(traceback.format_exception(t, v, tb))
            if 'assert generation is not None' in str(ex) or 'Input should be' in str(ex):
                # try one more time
                rets = await self.agenerate(
                    prompt_messages, stop=stop, callbacks=callbacks, **kwargs
                )
            else:
                raise
        finally:
            self.count_out_tokens(rets)
        return rets


class H2OChatOpenAI(ChatAGenerateStreamFirst, GenerateStream, ExtraChat, H2OBaseChatOpenAI, ChatOpenAI):
    temperature: float = 0.0
    tokenizer: Any = None
    system_prompt: Any = None
    chat_conversation: Any = []
    user_prompt_for_fake_system_prompt: Any = None
    prompts: Any = []
    count_input_tokens: Any = 0
    count_output_tokens: Any = 0
    prompter: Any = None

    # max_new_tokens0: Any = None  # FIXME: Doesn't seem to have same max_tokens == -1 for prompts==1

    def get_token_ids(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)
        else:
            # OpenAI uses tiktoken
            return super().get_token_ids(text)


class H2OAzureChatOpenAI(ChatAGenerateStreamFirst, GenerateNormal, ExtraChat, H2OBaseAzureChatOpenAI):
    temperature: float = 0.0
    system_prompt: Any = None
    chat_conversation: Any = []
    user_prompt_for_fake_system_prompt: Any = None
    prompts: Any = []
    count_input_tokens: Any = 0
    count_output_tokens: Any = 0
    prompter: Any = None
    tokenizer: Any = None

    def get_token_ids(self, text: str) -> List[int]:
        """Get the tokens present in the text with tiktoken package."""
        # tiktoken NOT supported for Python 3.7 or below
        if sys.version_info[1] <= 7:
            return super().get_token_ids(text)
        _, encoding_model = self._get_encoding_model()
        return encoding_model.encode(text)

    # max_new_tokens0: Any = None  # FIXME: Doesn't seem to have same max_tokens == -1 for prompts==1


class H2OChatAnthropic2(ChatAGenerateStreamFirst, GenerateNormal, ExtraChat, ChatAnthropic2):
    system_prompt: Any = None
    chat_conversation: Any = []
    user_prompt_for_fake_system_prompt: Any = None
    prompts: Any = []
    streaming: Any = True
    count_input_tokens: Any = 0
    count_output_tokens: Any = 0
    tokenizer: Any = None
    prompter: Any = None
    supports_caching: bool = False
    enable_caching: bool = False

    # max_new_tokens0: Any = None  # FIXME: Doesn't seem to have same max_tokens == -1 for prompts==1


class H2OChatAnthropic2Sys(H2OChatAnthropic2):
    pass


class H2OChatAnthropic3(ChatAGenerateStreamFirst, GenerateStream, ExtraChat, ChatAnthropic3):
    system_prompt: Any = None
    chat_conversation: Any = []
    user_prompt_for_fake_system_prompt: Any = None
    prompts: Any = []
    streaming: Any = True
    count_input_tokens: Any = 0
    count_output_tokens: Any = 0
    tokenizer: Any = None
    prompter: Any = None
    supports_caching: bool = False
    enable_caching: bool = False

    # max_new_tokens0: Any = None  # FIXME: Doesn't seem to have same max_tokens == -1 for prompts==1

    @staticmethod
    def process_messages(messages, max_cache_controls=3):
        processed_messages = []
        cache_control_count = 0

        for message in reversed(messages):
            if message["role"] == "user":
                if isinstance(message["content"], str):
                    content = [{
                        "type": "text",
                        "text": message["content"]
                    }]
                    if cache_control_count < max_cache_controls:
                        content[0]["cache_control"] = {"type": "ephemeral"}
                        cache_control_count += 1
                elif isinstance(message["content"], list):
                    content = []
                    for item in reversed(message["content"]):
                        if isinstance(item, dict):
                            item_copy = item.copy()
                            if cache_control_count < max_cache_controls:
                                item_copy["cache_control"] = {"type": "ephemeral"}
                                cache_control_count += 1
                            content.append(item_copy)
                        else:
                            content.append(item)
                    content.reverse()  # Restore original order within the message
                else:
                    content = message["content"]

                processed_messages.append({
                    "role": "user",
                    "content": content
                })
            else:
                processed_messages.append(message)

        return list(reversed(processed_messages))  # Reverse to restore original order

    def _get_request_payload(
            self,
            input_: LanguageModelInput,
            *,
            stop: Optional[List[str]] = None,
            **kwargs: Dict,
    ) -> Dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if hasattr(self, 'supports_caching') and self.supports_caching and \
                hasattr(self, 'enable_caching') and self.enable_caching:
            messages = payload['messages']
            system = payload.get('system', '')

            # fix system
            system_cached = [{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"}
            }] if system else ''

            # Process user and assistant messages
            messages_cached = self.process_messages(messages)

            # put messages and system back in
            payload['messages'] = messages_cached
            payload['system'] = system_cached
        # print('payload: %s' % payload, flush=True)
        return payload

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            *,
            stream_usage: Optional[bool] = None,
            **kwargs: Any,
    ) -> typing.Iterator[ChatGenerationChunk]:
        if stream_usage is None:
            stream_usage = self.stream_usage
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        stream = self._client.messages.create(**payload)
        coerce_content_to_string = not _tools_in_params(payload)
        for event in stream:
            if event.type == "message_start":
                usage = event.message.usage
                input_tokens = dict(usage).get('input_tokens', 0)
                cache_creation_input_tokens = dict(usage).get('cache_creation_input_tokens', 0)
                cache_read_input_tokens = dict(usage).get('cache_read_input_tokens', 0)
                # estimated cost effect, cache hits are roughly free compared to input or creation
                self.count_input_tokens += (cache_creation_input_tokens - cache_read_input_tokens)
                print(f"input_tokens: {input_tokens}")
                print(f"cache_creation_input_tokens: {cache_creation_input_tokens}")
                print(f"cache_read_input_tokens: {cache_read_input_tokens}")
            elif event.type == "message_delta":
                output_tokens = dict(event.usage).get('output_tokens', 0)
                self.count_output_tokens += output_tokens

            msg = _make_message_chunk_from_anthropic_event(
                event,
                stream_usage=stream_usage,
                coerce_content_to_string=coerce_content_to_string,
            )
            if msg is not None:
                chunk = ChatGenerationChunk(message=msg)
                if run_manager and isinstance(msg.content, str):
                    run_manager.on_llm_new_token(msg.content, chunk=chunk)
                yield chunk


class H2OChatAnthropic3Sys(H2OChatAnthropic3):
    pass


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream
)


class H2OChatGoogle(ChatAGenerateStreamFirst, GenerateStream, ExtraChat, ChatGoogleGenerativeAI):
    system_prompt: Any = None
    chat_conversation: Any = []
    user_prompt_for_fake_system_prompt: Any = None
    prompts: Any = []
    streaming: Any = False
    tokenizer: Any = None
    count_input_tokens: Any = 0
    count_output_tokens: Any = 0
    prompter: Any = None
    response_format: str = 'text'
    guided_json: dict | None = {}


class H2OChatMistralAI(ChatAGenerateStreamFirst, GenerateStream2, ExtraChat, ChatMistralAI):
    system_prompt: Any = None
    chat_conversation: Any = []
    user_prompt_for_fake_system_prompt: Any = None
    prompts: Any = []
    stream_output: bool = True
    tokenizer: Any = None
    count_input_tokens: Any = 0
    count_output_tokens: Any = 0
    response_format: Any = None
    prompter: Any = None

    # max_new_tokens0: Any = None  # FIXME: Doesn't seem to have same max_tokens == -1 for prompts==1


class H2OChatGroq(ChatAGenerateStreamFirst, GenerateStream2, ExtraChat, ChatGroq):
    system_prompt: Any = None
    chat_conversation: Any = []
    user_prompt_for_fake_system_prompt: Any = None
    prompts: Any = []
    tokenizer: Any = None
    stream_output: bool = True
    count_input_tokens: Any = 0
    count_output_tokens: Any = 0
    response_format: Any = None
    prompter: Any = None


class H2OAzureOpenAI(H2OTextGenOpenAI, AzureOpenAI):
    stop_sequences: Any = None
    sanitize_bot_response: bool = False
    prompter: Any = None
    context: Any = ''
    iinput: Any = ''
    tokenizer: Any = None
    async_sem: Any = None
    count_input_tokens: Any = 0
    prompts: Any = []
    count_output_tokens: Any = 0
    max_new_tokens0: Any = None
    count_llm_calls: Any = 0


class H2OHuggingFacePipeline(HuggingFacePipeline):
    prompts: Any = []
    count_input_tokens: Any = 0
    count_output_tokens: Any = 0

    def _generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
        self.count_input_tokens += sum([self.get_num_tokens(x) for x in prompts])
        rets = super()._generate(prompts, stop=stop, run_manager=run_manager, **kwargs)
        try:
            self.count_output_tokens += sum(
                [self.get_num_tokens(z) for z in flatten_list([[x.text for x in y] for y in rets.generations])])
        except Exception as e:
            if os.getenv('HARD_ASSERTS'):
                raise
            print("Failed to get total output tokens\n%s\n" % traceback.format_exc())
        return rets

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        self.count_input_tokens += self.get_num_tokens(str(prompt))
        self.prompts.append(prompt)
        response = self.pipeline(prompt, stop=stop)
        if self.pipeline.task == "text-generation":
            # Text generation return includes the starter text.
            text = response[0]["generated_text"][len(prompt):]
        elif self.pipeline.task == "text2text-generation":
            text = response[0]["generated_text"]
        elif self.pipeline.task == "summarization":
            text = response[0]["summary_text"]
        else:
            raise ValueError(
                f"Got invalid task {self.pipeline.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        if stop:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to huggingface_hub.
            text = enforce_stop_tokens(text, stop)
        self.count_output_tokens += self.get_num_tokens(text)
        return text

    def get_token_ids(self, text: str) -> List[int]:
        tokenizer = self.pipeline.tokenizer
        if tokenizer is not None:
            return tokenizer.encode(text)
        else:
            return FakeTokenizer().encode(text)['input_ids']


def get_llm(use_openai_model=False,
            model_name=None,
            model=None,
            tokenizer=None,
            inference_server=None,
            regenerate_clients=None,
            regenerate_gradio_clients=None,
            validate_clients=None,
            fail_if_invalid_client=None,
            langchain_only_model=None,
            load_awq='',
            stream_output=False,
            enable_caching=False,
            async_output=True,
            num_async=3,
            do_sample=False,
            seed=0,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            penalty_alpha=0.0,
            num_beams=1,
            max_new_tokens=512,
            max_new_tokens0=512,
            min_new_tokens=1,
            early_stopping=False,
            max_time=180,
            repetition_penalty=1.0,
            num_return_sequences=1,
            prompt_type=None,
            prompt_dict=None,
            chat_template=None,
            prompter=None,
            context=None,
            iinput=None,
            chat_conversation=None,
            add_chat_history_to_context=True,
            sanitize_bot_response=False,

            user_prompt_for_fake_system_prompt=None,
            json_object_prompt=None,
            json_object_prompt_simpler=None,
            json_code_prompt=None,
            json_code_prompt_if_no_schema=None,
            json_schema_instruction=None,

            system_prompt='',
            allow_chat_system_prompt=True,
            visible_models=0,
            h2ogpt_key=None,
            min_max_new_tokens=None,
            max_input_tokens=None,
            max_total_input_tokens=None,
            attention_sinks=None,
            sink_dict={},
            truncation_generation=None,

            langchain_agents=None,

            n_jobs=None,
            cli=False,
            llamacpp_path=None,
            llamacpp_dict=None,
            exllama_dict=None,
            verbose=False,

            image_file=None,
            image_control=None,
            images_num_max=None,
            image_resolution=None,
            image_format=None,
            rotate_align_resize_image=None,
            video_frame_period=None,
            image_batch_image_prompt=None,
            image_batch_final_prompt=None,
            image_batch_stream=None,
            visible_vision_models=None,
            video_file=None,

            document_choice=None,

            response_format=None,
            guided_json=None,
            guided_regex=None,
            guided_choice=None,
            guided_grammar=None,
            guided_whitespace_pattern=None,

            doing_grounding=False,
            json_vllm=False,

            query_action=True,
            summarize_action=False,
            stream_map=False,

            is_vision_model1=False,
            is_actually_vision_model1=False,

            ):
    # make all return only new text, so other uses work as expected, like summarization
    only_new_text = True
    gradio_server = False

    if chat_conversation is None:
        chat_conversation = []
    # shallow copy, so if add image entry doesn't affect outer one, only returned one
    chat_conversation = chat_conversation.copy()

    # in case prompter updated
    if prompter and prompter.system_prompt:
        system_prompt = prompter.system_prompt

    fake_for_tests = ['test_qa', 'test_make_add_db', 'test_many_text', 'test_chroma_filtering']
    if os.getenv('HARD_ASSERTS') and tokenizer is None and any([x in get_test_name_core() for x in fake_for_tests]):
        # allow certain tests to use fake one
        tokenizer = FakeTokenizer()
        max_input_tokens = 1024
        min_max_new_tokens = 512

    model_max_length = tokenizer.model_max_length
    if not attention_sinks:
        if max_input_tokens >= 0:
            max_input_tokens = min(model_max_length - min_max_new_tokens, max_input_tokens)
        else:
            max_input_tokens = model_max_length - min_max_new_tokens
    else:
        if max_input_tokens < 0:
            max_input_tokens = model_max_length

    streaming_callback = StreamingGradioCallbackHandler(max_time=max_time, verbose=verbose,
                                                        raise_stop=not stream_map or query_action)

    if n_jobs in [None, -1]:
        n_jobs = int(os.getenv('OMP_NUM_THREADS', str(os.cpu_count() // 2)))
    n_gpus = n_gpus_global
    if inference_server is None:
        inference_server = ''

    if inference_server.startswith('replicate'):
        async_output = False  # no real async
        model_string = ':'.join(inference_server.split(':')[1:])
        if 'meta/llama' in model_string:
            temperature = max(0.01, temperature if do_sample else 0)
        else:
            temperature = temperature if do_sample else 0
        gen_kwargs = dict(temperature=temperature,
                          seed=seed,
                          max_length=max_new_tokens,  # langchain
                          max_new_tokens=max_new_tokens,  # replicate docs
                          top_p=top_p if do_sample else 1.0,
                          top_k=top_k if do_sample else 1,  # not always supported
                          repetition_penalty=repetition_penalty)
        if system_prompt in auto_choices:
            if prompter.system_prompt:
                system_prompt = prompter.system_prompt
            else:
                system_prompt = ''
        if system_prompt:
            gen_kwargs.update(dict(system_prompt=system_prompt))

        # replicate handles prompting if no conversation, but in general has no chat API, so do all handling of prompting in h2oGPT
        if stream_output:
            callbacks = [streaming_callback]
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
                chat_conversation=chat_conversation,
                user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt,
                verbose=verbose,
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
                chat_conversation=chat_conversation,
                user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt,
                verbose=verbose,
            )
    elif use_openai_model or \
            inference_server.startswith('openai') or \
            inference_server.startswith('vllm') or \
            inference_server.startswith('sglang') and not (image_control or image_file):
        # supports async_output=True if chosen
        if use_openai_model and model_name is None:
            model_name = "gpt-3.5-turbo"
            inference_server = 'openai_chat'
        if not regenerate_clients and isinstance(model, dict):
            openai_client, openai_async_client, \
                inf_type, deployment_type, base_url, api_version, api_key = \
                model['client'], model['async_client'], model['inf_type'], \
                    model['deployment_type'], model['base_url'], model['api_version'], model['api_key']
        else:
            openai_client, openai_async_client, \
                inf_type, deployment_type, base_url, api_version, api_key = \
                set_openai(inference_server, model_name=model_name)

        if inf_type in ['vllm_chat', 'openai_chat', 'openai_azure_chat']:
            openai_client_completions = openai_client.chat.completions
            openai_async_client_completions = openai_async_client.chat.completions
        else:
            openai_client_completions = openai_client.completions
            openai_async_client_completions = openai_async_client.completions

        # Langchain oddly passes some things directly and rest via model_kwargs
        model_kwargs = dict(top_p=top_p if do_sample else 1.0,
                            frequency_penalty=0,
                            presence_penalty=(repetition_penalty - 1.0) * 2.0 + 0.0,  # so good default
                            )
        if not is_actually_vision_model1:
            model_kwargs.update(dict(logit_bias=None if inf_type == 'vllm' else {}))
        # if inference_server.startswith('vllm'):
        #    model_kwargs.update(dict(repetition_penalty=repetition_penalty))

        azure_kwargs = dict(openai_api_type='azure',
                            openai_api_key=api_key,
                            api_version=api_version,
                            deployment_name=deployment_type,
                            azure_endpoint=base_url,
                            )
        if langchain_agents is not None and \
                LangChainAgent.AUTOGPT.value in langchain_agents and \
                does_support_json_mode(inference_server, model_name, json_vllm=json_vllm):
            azure_kwargs.update(dict(response_format=dict(type="json_object")))

        kwargs_extra = {}

        if json_vllm:
            response_format_real = response_format if not (
                    guided_json or guided_regex or guided_choice or guided_grammar) else 'text'
            vllm_extra_dict = get_vllm_extra_dict(tokenizer,
                                                  stop_sequences=prompter.stop_sequences if prompter else [],
                                                  # repetition_penalty=repetition_penalty,  # could pass
                                                  response_format=response_format_real,
                                                  guided_json=guided_json,
                                                  guided_regex=guided_regex,
                                                  guided_choice=guided_choice,
                                                  guided_grammar=guided_grammar,
                                                  guided_whitespace_pattern=guided_whitespace_pattern,
                                                  )
        else:
            vllm_extra_dict = {}

        if guided_json and guided_json.get('properties', {}).get('type', '') == 'function':
            tools_openai = [
                guided_json['properties']
            ]
        elif guided_json:
            tools_openai = [
                {"type": "function",
                 "function": {
                     "name": "JSON",
                     "description": "Document, image, chat history conversion to strict JSON.",
                     "parameters": guided_json,
                 }
                 }
            ]
        else:
            tools_openai = []
        openai_model_supports_tools = model_name in openai_supports_functiontools + openai_supports_parallel_functiontools
        openai_model_supports_json = is_json_model(model_name, inference_server)
        if not json_vllm:
            openai_supports_json_or_tools = response_format == 'json_object' and openai_model_supports_json or openai_model_supports_tools and guided_json
        else:
            openai_supports_json_or_tools = False
        if inf_type == 'openai_chat' or inf_type == 'vllm_chat':
            kwargs_extra.update(dict(system_prompt=system_prompt,
                                     chat_conversation=chat_conversation,
                                     user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt))
            cls = H2OChatOpenAI
            # FIXME: Support context, iinput
            if inf_type == 'vllm_chat':
                if is_json_model(model_name, inference_server,
                                 json_vllm=json_vllm) and response_format == 'json_object':
                    # vllm without guided_json can't make json directly
                    if not json_vllm:
                        kwargs_extra.update(dict(response_format=dict(type=response_format if guided_json else 'text')))
                    else:
                        # for vllm 0.6.3+
                        kwargs_extra.update(dict(response_format=dict(type='text')))
                # async_output = False  # https://github.com/h2oai/h2ogpt/issues/928
                # async_sem = asyncio.Semaphore(num_async) if async_output else NullContext()
                kwargs_extra.update(dict(openai_api_key=api_key,
                                         # batch_size=1,
                                         client=openai_client_completions,
                                         async_client=openai_async_client_completions,
                                         # async_sem=async_sem,
                                         ))
                model_kwargs.update(vllm_extra_dict)
            else:
                if openai_supports_json_or_tools:
                    if openai_model_supports_tools and guided_json:
                        model_kwargs.update(dict(tools=tools_openai))
                        # Not vllm, guided_json not required
                        # ValueError: Error code: 400 - {'error': {'message': "'messages' must contain the word 'json' in some form, to use 'response_format' of type 'json_object'.", 'type': 'invalid_request_error', 'param': 'messages', 'code': None}}
                        kwargs_extra.update(dict(response_format=dict(type='text'), parallel_tool_calls=False))
                    else:
                        # Not vllm, guided_json not required
                        if not json_vllm:
                            kwargs_extra.update(dict(response_format=dict(type=response_format)))
                        else:
                            kwargs_extra.update(dict(response_format=dict(type='text')))
        elif inf_type == 'openai_azure_chat':
            cls = H2OAzureChatOpenAI
            if 'response_format' not in azure_kwargs and openai_supports_json_or_tools:
                # NOTE: not vllm, guided_json not required for json_object
                # overrides doc_json_mode if set
                if openai_model_supports_tools and guided_json:
                    model_kwargs.update(dict(tools=tools_openai))
                    # ValueError: Error code: 400 - {'error': {'message': "'messages' must contain the word 'json' in some form, to use 'response_format' of type 'json_object'.", 'type': 'invalid_request_error', 'param': 'messages', 'code': None}}
                    azure_kwargs.update(dict(response_format=dict(type='text')))  # , parallel_tool_calls=False))
                else:
                    azure_kwargs.update(dict(response_format=dict(type=response_format)))
            kwargs_extra.update(
                dict(system_prompt=system_prompt,
                     chat_conversation=chat_conversation,
                     user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt,
                     **azure_kwargs,
                     ))
            # FIXME: Support context, iinput
        elif inf_type == 'openai_azure':
            cls = H2OAzureOpenAI
            kwargs_extra.update(
                dict(**azure_kwargs,
                     ))
            kwargs_extra.update(model_kwargs)
            model_kwargs = {}
            # FIXME: Support context, iinput
        else:
            cls = H2OOpenAI
            if inf_type in ['vllm', 'sglang']:
                async_sem = asyncio.Semaphore(num_async) if async_output else AsyncNullContext()
                kwargs_extra.update(dict(stop_sequences=prompter.stop_sequences,
                                         sanitize_bot_response=sanitize_bot_response,
                                         context=context,
                                         iinput=iinput,
                                         chat_conversation=chat_conversation,
                                         user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt,
                                         openai_api_base=base_url,
                                         openai_api_key=api_key,
                                         batch_size=num_async if async_output else 1,
                                         client=openai_client_completions,
                                         async_client=openai_async_client_completions,
                                         async_sem=async_sem,
                                         max_new_tokens0=max_new_tokens0,
                                         ))
                kwargs_extra.update(model_kwargs)
                model_kwargs = {}
                model_kwargs.update(vllm_extra_dict)
            else:
                assert inf_type == 'openai' or use_openai_model, inf_type

        if is_actually_vision_model1:
            img_file = get_image_file(image_file, image_control, document_choice, base_model=model_name,
                                      images_num_max=images_num_max, image_resolution=image_resolution,
                                      image_format=image_format, convert=True, str_bytes=False)
            if img_file:
                # gpt4imagetag also applies to lmdeploy use of OpenAI via vllm_chat
                chat_conversation.append((img_file, gpt4imagetag))

        callbacks = [streaming_callback]
        model_kwargs.update(dict(seed=seed))

        if inf_type == 'vllm_chat':
            model_name = get_model_name(model_name, openai_client)

        gen_server_kwargs = dict(temperature=temperature if do_sample else 0.0,
                                 # FIXME: Need to count tokens and reduce max_new_tokens to fit like in generate.py
                                 max_tokens=max_new_tokens,
                                 )

        if model_name in ['o1-mini', 'o1-preview']:
            gen_server_kwargs['max_completion_tokens'] = gen_server_kwargs.pop('max_tokens')
            max_reasoning_tokens = int(os.getenv("MAX_REASONING_TOKENS", 25000))
            gen_server_kwargs['max_completion_tokens'] = max_reasoning_tokens + max(100, gen_server_kwargs[
                'max_completion_tokens'])
            gen_server_kwargs['temperature'] = 1.0
            model_kwargs.pop('presence_penalty', None)
            model_kwargs.pop('n', None)
            model_kwargs.pop('frequency_penalty', None)
            model_kwargs.pop('top_p', None)

        llm = cls(model_name=model_name,
                  model_kwargs=model_kwargs,
                  callbacks=callbacks if stream_output else None,
                  max_retries=6,
                  streaming=stream_output,
                  verbose=verbose,
                  request_timeout=max_time,
                  prompter=prompter,
                  tokenizer=tokenizer,
                  **gen_server_kwargs,
                  **kwargs_extra,
                  )
        streamer = callbacks[0] if stream_output else None
        if inf_type in ['openai', 'openai_chat', 'openai_azure', 'openai_azure_chat']:
            prompt_type = inference_server
        else:
            # vllm or non-image sglang goes here
            prompt_type = prompt_type or unknown_prompt_type
    elif inference_server.startswith('anthropic'):
        # no explicit JSON mode for anthropic
        # FIXME: Should use function calling
        # https://docs.anthropic.com/claude/docs/control-output-format

        if model_name in ["claude-2.0", "claude-2"]:
            cls = H2OChatAnthropic2
        elif model_name == "claude-2.1":
            # https://docs.anthropic.com/claude/docs/how-to-use-system-prompts
            cls = H2OChatAnthropic2Sys
        else:
            cls = H2OChatAnthropic3Sys

            if is_actually_vision_model1:
                img_file = get_image_file(image_file, image_control, document_choice, base_model=model_name,
                                          images_num_max=images_num_max, image_resolution=image_resolution,
                                          image_format=image_format, convert=True, str_bytes=False)
                if img_file:
                    chat_conversation.append((img_file, claude3imagetag))

        # Langchain oddly passes some things directly and rest via model_kwargs
        # NOTE: claude requires keys of properties to match pattern '^[a-zA-Z0-9_-]{1,64}$'
        # i.e. no spaces, while vLLM can handle spaces.
        if is_json_model(model_name, inference_server) and guided_json and response_format == 'json_object':
            # https://docs.anthropic.com/claude/docs/tool-use#specifying-tools
            model_kwargs = dict(tools=[
                {
                    "name": "JSON",
                    "description": "Document, image, chat history conversion to strict JSON.  This tool must be used, do not just answer the question from context.",
                    "input_schema": guided_json,
                }
            ], tool_choice={"type": "tool", "name": "JSON"})
        else:
            model_kwargs = {}
        kwargs_extra = {}
        kwargs_extra.update(dict(system_prompt=system_prompt, chat_conversation=chat_conversation,
                                 user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt))
        if not regenerate_clients and isinstance(model, dict):
            # FIXME: _AnthropicCommon ignores these and makes no client anyways
            kwargs_extra.update(dict(client=model['client'], async_client=model['async_client']))

        supports_caching = model_name in anthropic_prompt_caching
        if supports_caching:
            kwargs_extra.update(extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"})

        callbacks = [streaming_callback]
        llm = cls(model=model_name,
                  anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
                  max_tokens=max_new_tokens,
                  top_p=top_p if do_sample else 1.0,
                  top_k=top_k if do_sample else 1,
                  temperature=temperature if do_sample else 0,
                  # seed=seed,  # FIXME: Not supported yet
                  callbacks=callbacks if stream_output else None,
                  streaming=stream_output,
                  default_request_timeout=max_time,
                  model_kwargs=model_kwargs,
                  tokenizer=tokenizer,
                  prompter=prompter,
                  verbose=verbose,
                  supports_caching=supports_caching,
                  enable_caching=enable_caching,
                  **kwargs_extra
                  )
        streamer = callbacks[0] if stream_output else None
        prompt_type = inference_server
    elif inference_server.startswith('google'):
        # google doesn't have JSON mode but can use function calling so can give schema
        # https://ai.google.dev/tutorials/structured_data_extraction

        cls = H2OChatGoogle
        async_output = False  # client initialized inside event loop failures

        # Langchain oddly passes some things directly and rest via model_kwargs
        model_kwargs = dict()
        kwargs_extra = {}
        kwargs_extra.update(dict(system_prompt=system_prompt, chat_conversation=chat_conversation,
                                 user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt))
        if not regenerate_clients and isinstance(model, dict):
            kwargs_extra.update(dict(client=model['client'], async_client=model['async_client']))

        if is_actually_vision_model1:
            img_file = get_image_file(image_file, image_control, document_choice, base_model=model_name,
                                      images_num_max=images_num_max, image_resolution=image_resolution,
                                      image_format=image_format, convert=True, str_bytes=False)
            if img_file:
                tag = geminiimagetag if model_name == 'gemini-pro-vision' else gemini15imagetag
                chat_conversation.append((img_file, tag))
                # https://github.com/langchain-ai/langchain/issues/19115
                stream_output = False  # BUG IN GOOGLE/LANGCHAIN
            else:
                if '-vision' in model_name:
                    model_name = model_name.replace('-vision', '')

        # NOTE: assume want own control.  Too many false positives by Google.
        from google.generativeai.types import HarmCategory
        from google.generativeai.types import HarmBlockThreshold
        safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

        callbacks = [streaming_callback]
        llm = cls(model=model_name,
                  google_api_key=os.getenv('GOOGLE_API_KEY'),
                  top_p=top_p if do_sample else 1.0,
                  top_k=top_k if do_sample else 1,
                  temperature=temperature if do_sample else 0,
                  callbacks=callbacks if stream_output else None,
                  streaming=stream_output,
                  default_request_timeout=max_time,
                  max_output_tokens=max_new_tokens,
                  n=1,  # candidates
                  # seed=seed,  # FIXME: Not supported yet
                  model_kwargs=model_kwargs,
                  verbose=verbose,
                  tokenizer=tokenizer,
                  safety_settings=safety_settings,
                  response_format=response_format if response_format == 'json_object' else 'text',
                  guided_json=guided_json if response_format == 'json_object' else None,
                  prompter=prompter,
                  **kwargs_extra
                  )
        streamer = callbacks[0] if stream_output else None
        prompt_type = inference_server
    elif inference_server.startswith('mistralai'):
        cls = H2OChatMistralAI

        # Langchain oddly passes some things directly and rest via model_kwargs
        kwargs_extra = {}
        kwargs_extra.update(dict(system_prompt=system_prompt, chat_conversation=chat_conversation,
                                 user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt))
        if not regenerate_clients and isinstance(model, dict):
            kwargs_extra.update(dict(client=model['client'], async_client=model['async_client']))

        callbacks = [streaming_callback]
        # https://mistral.ai/news/mistral-large/

        if is_json_model(model_name, inference_server) and response_format == 'json_object':
            # https://docs.mistral.ai/platform/client/#json-mode
            # odd outputs for mistral-medium and mistral-tiny as of 04/02/2024
            # As if still since Feb 26, 2024 no updates for other models despite the bottom of https://mistral.ai/news/mistral-large/
            # Not vllm, guided_json not required
            kwargs_extra.update(dict(response_format=dict(type=response_format)))
        # Langchain oddly passes some things directly and rest via model_kwargs
        if does_support_functiontools(inference_server, model_name) and \
                is_json_model(model_name, inference_server) and guided_json and response_format == 'json_object':
            # https://docs.mistral.ai/capabilities/function_calling/
            model_kwargs = dict(tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "JSON",
                        "description": "Document, image, chat history conversion to strict JSON.",
                        "parameters": guided_json,
                    }
                }
            ],
                tool_choice='any',
                # if model_name not in ['mistral-medium', 'mistral-tiny', 'mistral-small-latest'] else 'auto'
            )
        else:
            model_kwargs = {}

        llm = cls(model=model_name,
                  mistral_api_key=os.getenv('MISTRAL_API_KEY'),
                  top_p=top_p if do_sample else 1.0,
                  top_k=top_k if do_sample else 1,
                  temperature=temperature if do_sample else 0,
                  callbacks=callbacks if stream_output else None,
                  streaming=stream_output,
                  stream=stream_output,
                  stream_output=stream_output,
                  default_request_timeout=max_time,
                  max_tokens=max_new_tokens,
                  safe_mode=False,
                  random_seed=seed,
                  verbose=verbose,
                  tokenizer=tokenizer,
                  prompter=prompter,
                  **model_kwargs,
                  **kwargs_extra,
                  llm_kwargs=dict(stream=True),
                  )
        streamer = callbacks[0] if stream_output else None
        prompt_type = inference_server
    elif inference_server.startswith('groq'):
        if len(inference_server.split(':')) == 2:
            groq_api_key = inference_server.split(':')[1]
            inference_server = inference_server.split(':')[0]
        else:
            groq_api_key = os.getenv('GROQ_API_KEY')
        cls = H2OChatGroq

        # Langchain oddly passes some things directly and rest via model_kwargs
        model_kwargs = dict()
        kwargs_extra = {}
        kwargs_extra.update(dict(system_prompt=system_prompt, chat_conversation=chat_conversation,
                                 user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt))
        if not regenerate_clients and isinstance(model, dict):
            kwargs_extra.update(dict(client=model['client'], async_client=model['async_client']))

        callbacks = [streaming_callback]
        llm = cls(model=model_name,
                  groq_api_key=groq_api_key,
                  temperature=temperature if do_sample else 0,
                  callbacks=callbacks if stream_output else None,
                  max_retries=2,
                  streaming=stream_output,
                  # stream=stream_output,
                  n=1,
                  max_tokens=max_new_tokens,
                  model_kwargs=dict(
                      top_p=top_p if do_sample else 1.0,
                      # seed=seed,  # FIXME: not supported yet
                      # top_k=top_k,
                  ),
                  tokenizer=tokenizer,
                  prompter=prompter,
                  **kwargs_extra,
                  )
        streamer = callbacks[0] if stream_output else None
        prompt_type = inference_server
    elif inference_server and inference_server.startswith('sagemaker'):
        async_output = False  # no real async
        callbacks = [streaming_callback]  # FIXME
        streamer = None
        async_output = False  # no real async

        endpoint_name = ':'.join(inference_server.split(':')[1:2])
        region_name = ':'.join(inference_server.split(':')[2:])

        from sagemaker import H2OSagemakerEndpoint, ChatContentHandler, BaseContentHandler
        if inference_server.startswith('sagemaker_chat'):
            content_handler = ChatContentHandler()
        else:
            content_handler = BaseContentHandler()
        model_kwargs = dict(temperature=temperature if do_sample else 1E-2,
                            return_full_text=False,
                            top_p=top_p if do_sample else 1.0,
                            max_new_tokens=max_new_tokens)
        llm = H2OSagemakerEndpoint(
            endpoint_name=endpoint_name,
            region_name=region_name,
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            model_kwargs=model_kwargs,
            content_handler=content_handler,
            endpoint_kwargs={'CustomAttributes': 'accept_eula=true'},
            tokenizer=tokenizer,  # for summarization and token counting
            verbose=verbose,
        )
    elif inference_server.startswith('sglang'):  # image mode
        callbacks = [streaming_callback]
        streamer = callbacks[0] if stream_output else None

        num_async = min(2, num_async)  # can't handle as much
        async_sem = asyncio.Semaphore(num_async) if async_output else AsyncNullContext()

        if is_actually_vision_model1:
            # https://github.com/sgl-project/sglang/issues/212#issuecomment-1973432493
            convert = True
            str_bytes = False
            img_file = get_image_file(image_file, image_control, document_choice, base_model=model_name,
                                      images_num_max=images_num_max, image_resolution=image_resolution,
                                      image_format=image_format, convert=convert, str_bytes=str_bytes)
        else:
            img_file = None

        inference_server_spit = inference_server.split(':')
        inference_server_url = ':'.join(inference_server_spit[2:])

        llm = SGlangInference(
            inference_server=inference_server,
            inference_server_url=inference_server_url,

            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            penalty_alpha=penalty_alpha,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            early_stopping=early_stopping,
            max_time=max_time,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            seed=seed,

            callbacks=callbacks if stream_output else None,
            stream_output=stream_output,

            prompter=prompter,
            context=context,

            tokenizer=tokenizer,
            system_prompt=system_prompt,
            chat_conversation=chat_conversation,
            user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt,
            add_chat_history_to_context=add_chat_history_to_context,

            async_sem=async_sem,
            verbose=verbose,

            image_file=img_file,  # we pass file name itself
        )
    elif inference_server:
        assert inference_server.startswith(
            'http'), "Malformed inference_server=%s.  Did you add http:// in front?" % inference_server

        from gradio_client import Client
        from gradio_utils.grclient import GradioClient
        from text_generation import Client as HFClient
        if isinstance(model, Client):
            gradio_server = True
            gr_client = model
            hf_client = None
        elif isinstance(model, GradioClient):
            gradio_server = True
            gr_client = model.clone()
            hf_client = None
        elif not regenerate_gradio_clients:
            gr_client = None
            hf_client = model
            assert isinstance(hf_client, HFClient)
        else:
            gr_client = None
            hf_client = None

        if regenerate_gradio_clients and gr_client:
            # regenerate or leave None for llava so created inside
            inference_server, gr_client, hf_client = get_client_from_inference_server(inference_server,
                                                                                      base_model=model_name,
                                                                                      validate_clients=validate_clients,
                                                                                      fail_if_invalid_client=fail_if_invalid_client,
                                                                                      verbose=verbose)
        inference_server, _, _, _ = get_hf_server(inference_server)

        # quick sanity check to avoid long timeouts, just see if can reach server
        requests.get(inference_server, timeout=int(os.getenv('REQUEST_TIMEOUT_FAST', '10')))
        callbacks = [streaming_callback]

        async_sem = asyncio.Semaphore(num_async) if async_output else AsyncNullContext()

        llava_direct_gradio = gr_client is not None and '/textbox_api_submit' in [x.api_name for x in
                                                                                  gr_client.endpoints]
        gradio_llava = is_gradio_vision_model(model_name) and llava_direct_gradio

        is_actually_vision_model2 = is_actually_vision_model1 if (hf_client or gradio_llava) else is_vision_model1

        if is_actually_vision_model2:
            # HF client uses markdown image url with bytes inside (or real url inside)
            if hf_client:
                convert = True
                str_bytes = False
            elif gradio_llava:
                convert = False
                str_bytes = True
            else:
                convert = True
                str_bytes = True
            # Gradio uses str_bytes=True
            img_file = get_image_file(image_file, image_control, document_choice, base_model=model_name,
                                      images_num_max=images_num_max, image_resolution=image_resolution,
                                      image_format=image_format, convert=convert, str_bytes=str_bytes)
        else:
            img_file = None

        if gradio_llava:
            llm = GradioLLaVaInference(
                inference_server_url=inference_server,

                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                penalty_alpha=penalty_alpha,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                early_stopping=early_stopping,
                max_time=max_time,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                seed=seed,

                callbacks=callbacks if stream_output else None,
                stream_output=stream_output,

                prompter=prompter,
                context=context,
                iinput=iinput,
                client=gr_client,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                chat_conversation=chat_conversation,
                user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt,
                add_chat_history_to_context=add_chat_history_to_context,
                # visible_models=visible_models,
                visible_models=model_name,
                h2ogpt_key=h2ogpt_key,
                min_max_new_tokens=min_max_new_tokens,
                max_input_tokens=max_input_tokens,
                max_total_input_tokens=max_total_input_tokens,
                async_sem=async_sem,
                verbose=verbose,

                image_file=img_file,  # we pass file name itself
            )
        elif gr_client:
            chat_client = False
            from vision.utils_vision import img_to_base64
            llm = GradioInference(
                inference_server_url=inference_server,
                return_full_text=False,

                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                penalty_alpha=penalty_alpha,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                early_stopping=early_stopping,
                max_time=max_time,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                seed=seed,
                chat_client=chat_client,

                callbacks=callbacks if stream_output else None,
                stream_output=stream_output,
                enable_caching=enable_caching,

                prompter=prompter,
                context=context,
                iinput=iinput,
                client=gr_client,
                sanitize_bot_response=sanitize_bot_response,
                tokenizer=tokenizer,
                chat_template=chat_template,

                user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt,
                json_object_prompt=json_object_prompt,
                json_object_prompt_simpler=json_object_prompt_simpler,
                json_code_prompt=json_code_prompt,
                json_code_prompt_if_no_schema=json_code_prompt_if_no_schema,
                json_schema_instruction=json_schema_instruction,

                system_prompt=system_prompt,
                chat_conversation=chat_conversation,
                add_chat_history_to_context=add_chat_history_to_context,
                visible_models=visible_models,
                h2ogpt_key=h2ogpt_key,
                min_max_new_tokens=min_max_new_tokens,
                max_input_tokens=max_input_tokens,
                max_total_input_tokens=max_total_input_tokens,
                async_sem=async_sem,
                verbose=verbose,

                image_file=img_file,
                image_control=None,  # already stuffed into image_file
                images_num_max=images_num_max,
                image_resolution=None,  # already changed
                image_format=None,  # already changed
                rotate_align_resize_image=None,  # already changed
                video_frame_period=None,  # already changed
                image_batch_image_prompt=image_batch_image_prompt,
                image_batch_final_prompt=image_batch_final_prompt,
                image_batch_stream=image_batch_stream,
                visible_vision_models=visible_vision_models,
                video_file=None,  # already handled in image list

                response_format=response_format,
                guided_json=guided_json,
                guided_regex=guided_regex,
                guided_choice=guided_choice,
                guided_grammar=guided_grammar,
                guided_whitespace_pattern=guided_whitespace_pattern,

                doing_grounding=doing_grounding,
            )
        elif hf_client:
            # no need to pass original client, no state and fast, so can use same validate_environment from base class
            # H2Oagenerate coming first in class makes these appear like unused inputs, but not case
            llm = H2OHuggingFaceTextGenInference(
                inference_server_url=inference_server,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                return_full_text=False,  # this only controls internal behavior, still returns processed text
                seed=seed,

                stop_sequences=prompter.stop_sequences,
                temperature=max(1e-2, temperature),
                top_k=top_k,
                top_p=min(max(1e-3, top_p), 1.0 - 1e-3),
                # typical_p=top_p,
                callbacks=callbacks if stream_output else None,
                stream_output=stream_output,
                prompter=prompter,
                context=context,
                iinput=iinput,
                tokenizer=tokenizer,
                chat_conversation=chat_conversation,
                user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt,
                timeout=max_time,
                sanitize_bot_response=sanitize_bot_response,
                async_sem=async_sem,
                verbose=verbose,

                base_model=model_name,
                image_file=img_file,
                image_control=None,  # already stuffed into image_file
                images_num_max=None,  # already set
                image_resolution=None,  # already changed
                image_format=None,  # already changed
                rotate_align_resize_image=None,  # already changed
                video_frame_period=None,  # already changed
                image_batch_image_prompt=image_batch_image_prompt,
                image_batch_final_prompt=image_batch_final_prompt,
                image_batch_stream=image_batch_stream,
                visible_vision_models=visible_vision_models,
                video_file=video_file,
            )
        else:
            raise RuntimeError("No defined client")
        streamer = callbacks[0] if stream_output else None
    elif model_name in non_hf_types:
        async_output = False  # FIXME: not implemented yet, and wouldn't make much sense as won't be faster
        assert langchain_only_model
        if model_name == 'llama':
            callbacks = [streaming_callback]
            streamer = callbacks[0] if stream_output else None
        else:
            # stream_output = False
            # doesn't stream properly as generator, but at least
            callbacks = [streaming_stdout.StreamingStdOutCallbackHandler()]
            streamer = None
        if prompter:
            prompt_type = prompter.prompt_type
        else:
            prompter = Prompter(prompt_type, prompt_dict, debug=False, stream_output=stream_output,
                                tokenizer=tokenizer, base_model=model_name)
            pass  # assume inputted prompt_type is correct
        from gpt4all_llm import get_llm_gpt4all
        llm = get_llm_gpt4all(model_name=model_name,
                              model=model,
                              max_new_tokens=max_new_tokens,
                              temperature=temperature,
                              seed=seed,
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
                              tokenizer=tokenizer,
                              chat_conversation=chat_conversation,
                              user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt,
                              max_seq_len=model_max_length,
                              llamacpp_path=llamacpp_path,
                              llamacpp_dict=llamacpp_dict,
                              n_gpus=n_gpus,
                              max_time=max_time,
                              )
    elif hasattr(model, 'is_exlama') and model.is_exlama():
        async_output = False  # FIXME: not implemented yet
        assert langchain_only_model
        callbacks = [streaming_callback]
        streamer = callbacks[0] if stream_output else None

        if exllama_dict is None:
            exllama_dict = {}

        from llm_exllama import Exllama
        llm = Exllama(streaming=stream_output,
                      model_path=None,
                      model=model,
                      lora_path=None,
                      temperature=temperature,
                      top_k=top_k,
                      top_p=top_p,
                      typical=.7,
                      beams=1,
                      beam_length=0,
                      # beam_length = 40,
                      stop_sequences=prompter.stop_sequences,
                      callbacks=callbacks,
                      verbose=verbose,
                      max_seq_len=model_max_length,
                      fused_attn=False,
                      **exllama_dict,
                      # alpha_value = 1.0, #For use with any models
                      # compress_pos_emb = 4.0, #For use with superhot
                      # set_auto_map = "3, 2" #Gpu split, this will split 3gigs/2gigs
                      prompter=prompter,
                      context=context,
                      iinput=iinput,
                      chat_conversation=chat_conversation,
                      user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt,
                      )
    else:
        if is_actually_vision_model1:
            convert = True
            str_bytes = False
            img_file = get_image_file(image_file, image_control, document_choice, base_model=model_name,
                                      images_num_max=images_num_max, image_resolution=image_resolution,
                                      image_format=image_format, convert=convert, str_bytes=str_bytes)
        else:
            img_file = None

        async_output = False  # FIXME: not implemented yet
        if model is None:
            # only used if didn't pass model in
            assert tokenizer is None or isinstance(tokenizer, FakeTokenizer)
            prompt_type = 'human_bot'
            if model_name is None:
                # model_name = 'h2oai/h2ogpt-oasst1-512-12b'
                # model_name = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
                # model_name = 'h2oai/h2ogpt-oasst1-512-20b'
                model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
                load_4bit = False
            else:
                load_4bit = True
            inference_server = ''
            model, tokenizer, device = get_model_retry(load_4bit=load_4bit, base_model=model_name,
                                                       inference_server=inference_server, gpu_id=0)

        gen_kwargs = dict(do_sample=do_sample,
                          seed=seed,
                          num_beams=num_beams,
                          max_new_tokens=max_new_tokens,
                          min_new_tokens=min_new_tokens,
                          early_stopping=early_stopping,
                          max_time=max_time,
                          repetition_penalty=repetition_penalty,
                          num_return_sequences=num_return_sequences,
                          return_full_text=not only_new_text,
                          handle_long_generation=None)
        if do_sample:
            gen_kwargs.update(dict(temperature=temperature,
                                   top_k=top_k,
                                   top_p=top_p,
                                   penalty_alpha=penalty_alpha))
            assert len(set(gen_hyper).difference(gen_kwargs.keys())) == 0
        else:
            gen_kwargs.update(dict(penalty_alpha=penalty_alpha))
            assert len(set(gen_hyper0).difference(gen_kwargs.keys())) == 0

        if attention_sinks:
            from transformers import SinkCache
            sink_dict['window_length'] = sink_dict.get('window_length', max_input_tokens)
            sink_dict['num_sink_tokens'] = sink_dict.get('num_sink_tokens', 4)
            cache = SinkCache(**sink_dict)
            gen_kwargs.update(dict(past_key_values=cache))

        if stream_output:
            skip_prompt = only_new_text
            from gen import H2OTextIteratorStreamer
            decoder_kwargs = {}
            streamer = H2OTextIteratorStreamer(tokenizer, skip_prompt=skip_prompt, block=False, **decoder_kwargs)
            gen_kwargs.update(dict(streamer=streamer))
        else:
            streamer = None

        from h2oai_pipeline import H2OTextGenerationPipeline

        if load_awq and hasattr(model, 'model'):
            # need this else get device on multiple devices cuda and cpu
            # e.g. AutoAWQForCausalLM
            model = model.model
        pipe = H2OTextGenerationPipeline(model=model,
                                         use_prompter=True,
                                         prompter=prompter,
                                         context=context,
                                         iinput=iinput,
                                         chat_conversation=chat_conversation,
                                         user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt,
                                         prompt_type=prompt_type,
                                         prompt_dict=prompt_dict,
                                         sanitize_bot_response=sanitize_bot_response,
                                         chat=False, stream_output=stream_output,
                                         tokenizer=tokenizer,
                                         max_input_tokens=max_input_tokens,
                                         base_model=model_name,
                                         verbose=verbose,
                                         truncation_generation=truncation_generation,
                                         image_file=img_file,
                                         image_control=image_control,
                                         images_num_max=images_num_max,
                                         image_resolution=image_resolution,
                                         image_format=image_format,
                                         rotate_align_resize_image=rotate_align_resize_image,
                                         video_frame_period=video_frame_period,
                                         image_batch_image_prompt=image_batch_image_prompt,
                                         image_batch_final_prompt=image_batch_final_prompt,
                                         image_batch_stream=image_batch_stream,
                                         visible_vision_models=visible_vision_models,
                                         video_file=video_file,
                                         **gen_kwargs)
        # pipe.task = "text-generation"
        # below makes it listen only to our prompt removal,
        # not built in prompt removal that is less general and not specific for our model
        # also works for Conditional generation: https://github.com/huggingface/transformers/issues/27870#issuecomment-1844775749
        # if img_file:
        #    pipe.task = 'image-to-text'
        # else:
        pipe.task = "text2text-generation"

        llm = H2OHuggingFacePipeline(pipeline=pipe)
    return llm, model_name, streamer, prompt_type, async_output, only_new_text, gradio_server


def get_device_dtype():
    # torch.device("cuda") leads to cuda:x cuda:y mismatches for multi-GPU consistently
    import torch
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
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
        page_content=str(page_content),
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
                yield Document(page_content=str(f.read()), metadata={"source": github_url})


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
        itm = Document(page_content=str(line), metadata={"source": file})
        # NOTE: yield has issues when going into db, loses metadata
        # yield itm
        sources.append(itm)
    return sources


def get_supported_types():
    non_image_types0 = ["pdf", "txt", "csv", "toml", "py", "rst", "xml", "rtf",
                        "md",
                        "html", "mhtml", "htm",
                        "enex", "eml", "epub", "odt", "pptx", "ppt",
                        "zip",
                        "gz",
                        "gzip",
                        "urls",
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
    image_types0 = get_image_types()
    return non_image_types0, image_types0, video_types0


non_image_types, image_types, video_types = get_supported_types()
set_image_types = set(image_types)

if have_libreoffice or True:
    # or True so it tries to load, e.g. on MAC/Windows, even if don't have libreoffice since works without that
    non_image_types.extend(["docx", "doc", "xls", "xlsx"])
if have_jq:
    non_image_types.extend(["json", "jsonl"])

if have_librosa:
    audio_types = ['aac', 'au', 'mp3', 'ogg', 'flac', 'm4a', 'wav', 'mp4', 'mpeg', 'mpg']
else:
    audio_types = []
set_audio_types = set(audio_types)

file_types = non_image_types + image_types + audio_types


def try_as_html(file):
    # try treating as html as occurs when scraping websites
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
    else:
        doc1 = []
    return doc1


def json_metadata_func(record: dict, metadata: dict) -> dict:
    # Define the metadata extraction function.

    if isinstance(record, dict):
        metadata["sender_name"] = record.get("sender_name")
        metadata["timestamp_ms"] = record.get("timestamp_ms")

    if "source" in metadata:
        metadata["source_json"] = metadata['source']
    if "seq_num" in metadata:
        metadata["seq_num_json"] = metadata['seq_num']

    return metadata


def get_num_pages(file):
    try:
        import fitz
        src = fitz.open(file)
        return len(src)
    except:
        return None


def get_each_page(file):
    import fitz

    pages = []
    src = fitz.open(file)
    for page in src:
        tar = fitz.open()  # output PDF for 1 page
        # copy over current page
        tar.insert_pdf(src, from_page=page.number, to_page=page.number)
        tmpdir = os.getenv('TMPDDIR', tempfile.mkdtemp())
        makedirs(tmpdir, exist_ok=True)
        page_file = os.path.join(tmpdir, f"{file}-page-{page.number}-{str(uuid.uuid4())}.pdf")
        makedirs(os.path.dirname(page_file), exist_ok=True)
        tar.save(page_file)
        tar.close()
        pages.append(page_file)
    return pages


class Crawler:
    # FIXME: Consider scrapy
    # https://www.scrapingbee.com/blog/crawling-python/
    # https://github.com/scrapy/scrapy
    # https://www.scrapingbee.com/blog/web-scraping-with-scrapy/

    def __init__(self, urls=[], deeper_only=True, depth=int(os.getenv('CRAWL_DEPTH', '1')), verbose=False):
        self.visited_urls = []
        self.urls_to_visit = urls.copy()
        self.starting_urls = urls.copy()
        self.deeper_only = deeper_only
        self.depth = depth
        self.verbose = verbose
        self.final_urls = []

    def download_url(self, url):
        return requests.get(url).text

    def get_linked_urls(self, url, html):
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        soup = BeautifulSoup(html, 'html.parser')
        for link in soup.find_all('a'):
            path = link.get('href')
            if path and path.startswith('/'):
                path = urljoin(url, path)
            yield path

    def add_url_to_visit(self, url):
        if url not in self.visited_urls and url not in self.urls_to_visit:
            if url in self.starting_urls:
                pass
            elif self.deeper_only and not any(url.startswith(x) for x in self.starting_urls):
                if self.verbose:
                    print("Skipped %s" % url, flush=True)
            else:
                self.urls_to_visit.append(url)
                if self.verbose:
                    print("Added %s" % url, flush=True)

    def crawl(self, url):
        html = self.download_url(url)
        for url in self.get_linked_urls(url, html):
            self.add_url_to_visit(url)

    def run(self):
        depth = 0
        while self.urls_to_visit:
            url = self.urls_to_visit.pop(0)
            if self.verbose:
                print(f'Crawling: {url}', flush=True)
            try:
                self.crawl(url)
            except Exception as e:
                if self.verbose:
                    print(f'Failed to crawl: {url}: {str(e)}', flush=True)
            finally:
                self.visited_urls.append(url)
                if depth >= self.depth:
                    if self.verbose:
                        print("Done crawling", flush=True)
                    break
                depth += 1
        self.final_urls = sorted(set(self.urls_to_visit + self.visited_urls))
        return self.final_urls


def file_to_doc(file,
                filei=0,
                base_path=None, verbose=False, fail_any_exception=False,
                chunk=True, chunk_size=512, n_jobs=-1,
                is_url=False, is_txt=False,

                # urls
                use_unstructured=True,
                use_playwright=False,
                use_selenium=False,
                use_scrapeplaywright=False,
                use_scrapehttp=False,

                # pdfs
                use_pymupdf='auto',
                use_unstructured_pdf='auto',
                use_pypdf='auto',
                enable_pdf_ocr='auto',
                try_pdf_as_html='auto',
                enable_pdf_doctr='auto',

                # images
                enable_ocr=False,
                enable_doctr=False,
                enable_pix2struct=False,
                enable_captions=True,
                enable_llava=True,
                enable_transcriptions=True,
                captions_model=None,
                llava_model=None,
                llava_prompt=None,

                asr_model=None,
                asr_gpu_id=0,

                model_loaders=None,

                # json
                jq_schema='.[]',
                extract_frames=10,

                headsize=50,  # see also H2OSerpAPIWrapper
                db_type=None,
                selected_file_types=None,

                is_public=False,
                from_ui=True,

                hf_embedding_model=None,
                use_openai_embedding=False,
                ):
    # SOME AUTODETECTION LOGIC FOR URL VS TEXT

    file_stripped = file.strip()  # in case accidental spaces in front or at end
    if file_stripped == '':
        raise ValueError("Refusing to accept empty data")
    file_lower = file_stripped.lower()
    case1_arxiv = file_lower.startswith('arxiv:') and len(file_lower.split('arxiv:')) == 2
    case2_arxiv = file_lower.startswith('https://arxiv.org/abs') and len(file_lower.split('https://arxiv.org/abs')) == 2
    case3_arxiv = file_lower.startswith('http://arxiv.org/abs') and len(file_lower.split('http://arxiv.org/abs')) == 2
    case4_arxiv = file_lower.startswith('arxiv.org/abs/') and len(file_lower.split('arxiv.org/abs/')) == 2

    is_arxiv = case1_arxiv or case2_arxiv or case3_arxiv or case4_arxiv
    is_youtube = any(
        file_lower.replace('http://', '').replace('https://', '').replace('www.', '').startswith(prefix) for prefix in
        url_prefixes_youtube)

    if is_url and is_txt:
        # decide which
        if ' ' in file_stripped:
            # can't have literal space in URL
            is_url = False
        elif is_arxiv or is_youtube:
            # force it
            is_txt = False
        else:
            file_test = return_good_url(file_stripped)
            if file_test is None:
                is_url = False
            else:
                is_txt = False

    assert isinstance(model_loaders, dict)
    if selected_file_types is not None:
        set_image_audio_types1 = set_image_types.intersection(set(selected_file_types))
        set_audio_types1 = set_audio_types.intersection(set(selected_file_types))
    else:
        set_image_audio_types1 = set_image_types
        set_audio_types1 = set_audio_types

    assert db_type is not None
    chunk_sources = functools.partial(_chunk_sources, chunk=chunk, chunk_size=chunk_size, db_type=db_type,
                                      hf_embedding_model=hf_embedding_model, use_openai_embedding=use_openai_embedding,
                                      verbose=verbose)
    add_meta = functools.partial(_add_meta, headsize=headsize, filei=filei)
    # FIXME: if zip, file index order will not be correct if other files involved
    path_to_docs_func = functools.partial(path_to_docs,
                                          verbose=verbose,
                                          fail_any_exception=fail_any_exception,
                                          n_jobs=n_jobs,
                                          chunk=chunk, chunk_size=chunk_size,
                                          # url=file if is_url else None,
                                          # text=file if is_txt else None,

                                          # urls
                                          use_unstructured=use_unstructured,
                                          use_playwright=use_playwright,
                                          use_selenium=use_selenium,
                                          use_scrapeplaywright=use_scrapeplaywright,
                                          use_scrapehttp=use_scrapehttp,

                                          # pdfs
                                          use_pymupdf=use_pymupdf,
                                          use_unstructured_pdf=use_unstructured_pdf,
                                          use_pypdf=use_pypdf,
                                          enable_pdf_ocr=enable_pdf_ocr,
                                          enable_pdf_doctr=enable_pdf_doctr,
                                          try_pdf_as_html=try_pdf_as_html,

                                          # images
                                          enable_ocr=enable_ocr,
                                          enable_doctr=enable_doctr,
                                          enable_pix2struct=enable_pix2struct,
                                          enable_captions=enable_captions,
                                          captions_model=captions_model,
                                          enable_llava=enable_llava,
                                          llava_model=llava_model,
                                          llava_prompt=llava_prompt,

                                          # audio
                                          enable_transcriptions=enable_transcriptions,
                                          asr_model=asr_model,

                                          caption_loader=model_loaders['caption'],
                                          doctr_loader=model_loaders['doctr'],
                                          pix2struct_loader=model_loaders['pix2struct'],
                                          asr_loader=model_loaders['asr'],

                                          # json
                                          jq_schema=jq_schema,
                                          # video
                                          extract_frames=extract_frames,

                                          db_type=db_type,

                                          is_public=is_public,
                                          from_ui=from_ui,

                                          hf_embedding_model=hf_embedding_model,
                                          use_openai_embedding=use_openai_embedding,
                                          )

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

    orig_url = None
    if is_url and any([file.strip().lower().endswith('.' + x) for x in file_types]):
        # then just download, so can use good parser, not always unstructured url parser
        base_path_url = os.path.join(get_gradio_tmp(), "urls_downloaded")
        base_path_url = makedirs(base_path_url, exist_ok=True, tmp_ok=True, use_base=True)
        source_file = os.path.join(base_path_url,
                                   "_%s_%s" % ("_" + str(uuid.uuid4())[:10], os.path.basename(urlparse(file).path)))
        try:
            download_simple(file, source_file, overwrite=True, verbose=verbose)
        except BaseException as e:
            print("Download simple failed: %s, trying other means" % str(e), flush=True)
        if os.path.isfile(source_file):
            orig_url = file
            is_url = False
            file = source_file

    can_do_audio_transcription = isinstance(file, str) and \
                                 any(file.lower().endswith(x) for x in set_audio_types1) and enable_transcriptions
    can_do_video_extraction = isinstance(file, str) and \
                              any([file.endswith(x) for x in video_types]) and extract_frames > 0 and have_fiftyone

    if is_url:
        if is_arxiv:
            if case1_arxiv:
                query = file.lower().split('arxiv:')[1].strip()
            elif case2_arxiv:
                query = file.lower().split('https://arxiv.org/abs/')[1].strip()
            elif case2_arxiv:
                query = file.lower().split('http://arxiv.org/abs/')[1].strip()
            elif case3_arxiv:
                query = file.lower().split('arxiv.org/abs/')[1].strip()
            else:
                raise RuntimeError("Unexpected arxiv error for %s" % file)
            if have_arxiv:
                trials = 3
                docs1 = []
                for trial in range(trials):
                    try:
                        docs1 = ArxivLoader(query=query, load_max_docs=20, load_all_available_meta=True).load()
                        break
                    except urllib.error.URLError:
                        pass
                if not docs1:
                    print("Failed to get arxiv %s" % query, flush=True)
                # ensure string, sometimes None
                [[x.metadata.update({k: str(v)}) for k, v in x.metadata.items()] for x in docs1]
                query_url = f"https://arxiv.org/abs/{query}"
                [x.metadata.update(
                    dict(source=x.metadata.get('entry_id', query_url), query=query_url,
                         input_type='arxiv', head=x.metadata.get('Title', ''), date=str(datetime.now))) for x in
                    docs1]
            else:
                docs1 = []
            add_meta(docs1, file, parser="is_url")
            docs1 = clean_doc(docs1)
            doc1.extend(chunk_sources(docs1))
        elif is_youtube and (enable_transcriptions or extract_frames > 0 and have_fiftyone):
            e = None
            handled = False
            docs1 = []
            files_out = []
            if enable_transcriptions:
                try:
                    if model_loaders['asr'] is not None and not isinstance(model_loaders['asr'], (str, bool)):
                        # assumes didn't fork into this process with joblib, else can deadlock
                        if verbose:
                            print("Reuse ASR", flush=True)
                        model_loaders['asr'].load_model()
                    else:
                        if verbose:
                            print("Fresh ASR", flush=True)
                        from audio_langchain import H2OAudioCaptionLoader
                        model_loaders['asr'] = H2OAudioCaptionLoader(asr_model=asr_model,
                                                                     asr_gpu=model_loaders['asr'] == 'gpu',
                                                                     gpu_id=asr_gpu_id,
                                                                     )
                    model_loaders['asr'].set_audio_paths([file])
                    docs1c = model_loaders['asr'].load(from_youtube=True)
                    files_out = model_loaders['asr'].files_out
                    docs1c = [x for x in docs1c if x.page_content]
                    add_meta(docs1c, file, parser='H2OAudioCaptionLoader: %s' % asr_model)
                    # caption didn't set source, so fix-up meta
                    hash_of_file = hash_file(file)
                    [doci.metadata.update(source=file, hashid=hash_of_file) for doci in docs1c]
                    for file_out_i, file_out in enumerate(files_out):
                        key = 'source_true_' + str(file_out_i)
                        [doci.metadata.update({key: file_out}) for doci in docs1c]
                    docs1.extend(docs1c)
                    doc1.extend(chunk_sources(docs1))
                    handled = True
                except BaseException as e0:
                    print("ASR: %s: %s" % (str(e0), traceback.print_exception(e0)), flush=True)
                    e = e0
                handled |= len(docs1) > 0
            if extract_frames > 0 and have_fiftyone:
                try:
                    from vision.extract_movie import extract_unique_frames
                    if not files_out or True:  # always do, seems makes audio m4a not with video when downloads
                        # have to directly download
                        export_dir = extract_unique_frames(urls=[file], extract_frames=extract_frames)
                        docs1c_files = path_to_docs_func(export_dir)
                    else:
                        # just use already-downloaded files
                        docs1c_files = []
                        for file_out in files_out:
                            export_dir = extract_unique_frames(file=file_out, extract_frames=extract_frames)
                            docs1c_files.extend(path_to_docs_func(export_dir))
                    if os.getenv('FRAMES_AS_SAME_DOC', '0') == '1':
                        add_meta(docs1c_files, file, parser='extract_frames from %s' % file)
                        hash_of_file = hash_file(file)
                        [doci.metadata.update(source=file, hashid=hash_of_file) for doci in docs1c_files]
                    else:
                        [x.metadata.update(dict(original_source=file)) for order_id, x in enumerate(docs1c_files)]
                    docs1c_files = chunk_sources(docs1c_files)
                    doc1.extend(docs1c_files)
                except BaseException as e0:
                    print("Extract YouTube Frames: %s" % str(e0), flush=True)
                    e = e0
                handled |= len(docs1) > 0
            if len(doc1) == 0:
                # if literally nothing, show failed to parse so user knows, since unlikely nothing in PDF at all.
                if handled:
                    raise ValueError("%s had no valid text, but meta data was parsed" % file)
                else:
                    raise ValueError("%s had no valid text and no meta data was parsed: %s" % (file, str(e)))
        else:
            if not (file.startswith("http://") or file.startswith("file://") or file.startswith("https://")):
                file = 'http://' + file
            url_depth = int(os.getenv('ALL_CRAWL_DEPTH', '0'))
            if url_depth > 0:
                final_urls = Crawler(urls=[file], verbose=verbose).run()
            else:
                final_urls = [file]
            docs1 = []
            do_unstructured = only_unstructured_urls or use_unstructured
            if only_selenium or only_playwright:
                do_unstructured = False
            do_playwright = have_playwright and (use_playwright or only_playwright)
            if only_unstructured_urls or only_selenium:
                do_playwright = False
            do_selenium = have_selenium and (use_selenium or only_selenium)
            if only_unstructured_urls or only_playwright:
                do_selenium = False
            if do_unstructured or use_unstructured:
                docs1a = UnstructuredURLLoader(urls=final_urls, headers=dict(ssl_verify="False")).load()
                docs1a = [x for x in docs1a if
                          x.page_content and x.page_content != '403 Forbidden' and not x.page_content.startswith(
                              'Access Denied')]
                add_parser(docs1a, 'UnstructuredURLLoader')
                docs1.extend(docs1a)
            if len(docs1) == 0 and have_playwright or do_playwright:
                # then something went wrong, try another loader:
                try:
                    from langchain_community.document_loaders import PlaywrightURLLoader
                    docs1a = asyncio.run(PlaywrightURLLoader(urls=final_urls).aload())
                    # docs1 = PlaywrightURLLoader(urls=[file]).load()
                    docs1a = [x for x in docs1a if
                              x.page_content and x.page_content != '403 Forbidden' and not x.page_content.startswith(
                                  'Access Denied')]
                    add_parser(docs1a, 'PlaywrightURLLoader')
                    docs1.extend(docs1a)
                except Exception as e0:
                    traceback.print_exc()
                    print("playwright failed: %s: %s" % (str(e0), traceback.print_exception(e0)), flush=True)
            if len(docs1) == 0 and have_selenium or do_selenium:
                # then something went wrong, try another loader:
                # but requires Chrome binary, else get: selenium.common.exceptions.WebDriverException:
                # Message: unknown error: cannot find Chrome binary
                from langchain_community.document_loaders import SeleniumURLLoader
                from selenium.common.exceptions import WebDriverException
                try:
                    docs1a = SeleniumURLLoader(urls=final_urls).load()
                    docs1a = [x for x in docs1a if
                              x.page_content and x.page_content != '403 Forbidden' and not x.page_content.startswith(
                                  'Access Denied')]
                    add_parser(docs1a, 'SeleniumURLLoader')
                    docs1.extend(docs1a)
                except WebDriverException as e:
                    print("No web driver: %s" % str(e), flush=True)
            if use_scrapehttp or use_scrapeplaywright:
                docs1a = []
                if url_depth > 0:
                    # then already did crawl over depth, just use
                    pass
                else:
                    final_urls = Crawler(urls=[file], verbose=verbose).run()
                if use_scrapehttp:
                    loader = AsyncHtmlLoader(final_urls, verify_ssl=False, requests_per_second=10,
                                             ignore_load_errors=True)
                    docs1a = loader.load()
                if use_scrapeplaywright:
                    loader = AsyncChromiumLoader(final_urls)
                    docs1a = loader.load()
                if os.getenv('HTML_TRANS', 'HTML2TEXT') == 'BS4':
                    bs_transformer = BeautifulSoupTransformer()
                    # Scrape text content tags such as <p>, <li>, <div>, and <a> tags from the HTML content:
                    # https://python.langchain.com/docs/use_cases/web_scraping#quickstart
                    tags_to_extract = ast.literal_eval(os.getenv('BS4_TAGS', '["span"]'))
                    docs1a = bs_transformer.transform_documents(docs1a, tags_to_extract=tags_to_extract)
                else:
                    html2text = Html2TextTransformer()
                    docs1a = html2text.transform_documents(docs1a)
                docs1.extend(docs1a)
            [x.metadata.update(dict(input_type='url', date=str(datetime.now))) for x in docs1]
            add_meta(docs1, file, parser="is_url")
            docs1 = clean_doc(docs1)
            doc1.extend(chunk_sources(docs1))
    elif is_txt:
        base_path = "user_paste"
        base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
        source_file = os.path.join(base_path, "_%s.txt" % str(uuid.uuid4())[:10])
        with open(source_file, "wt", encoding="utf-8") as f:
            f.write(file)
        metadata = dict(source=source_file, date=str(datetime.now()), input_type='pasted txt')
        doc1 = Document(page_content=str(file), metadata=metadata)
        add_meta(doc1, file, parser="f.write")
        # Bit odd to change if was original text
        # doc1 = clean_doc(doc1)
    elif file.lower().endswith('.html') or file.lower().endswith('.mhtml') or file.lower().endswith('.htm'):
        docs1 = UnstructuredHTMLLoader(file_path=file).load()
        add_meta(docs1, file, parser='UnstructuredHTMLLoader')
        docs1 = clean_doc(docs1)
        doc1 = chunk_sources(docs1, language=Language.HTML)
    elif (file.lower().endswith('.docx') or file.lower().endswith('.doc')) and (have_libreoffice or True):
        docs1 = UnstructuredWordDocumentLoader(file_path=file).load()
        add_meta(docs1, file, parser='UnstructuredWordDocumentLoader')
        docs1 = [x for x in docs1 if x.page_content]
        if not docs1:
            from langchain_community.document_loaders import Docx2txtLoader
            docs1 = Docx2txtLoader(file_path=file).load()
            docs1 = [x for x in docs1 if x.page_content]
            add_meta(docs1, file, parser='Docx2txtLoader')
        if os.getenv('H2OGPT_DOCX_EXTRACT_IMAGES', '1') == '1':
            try:
                # maybe images
                import docx2txt

                tmpdir = os.path.join(get_gradio_tmp(), str(uuid.uuid4()))
                makedirs(tmpdir, exist_ok=True)
                text = docx2txt.process(file, tmpdir)
                images = os.listdir(tmpdir)
                docs1.extend(path_to_docs_func([os.path.join(tmpdir, x) for x in images]))
            except Exception as e:
                print("docx images failure: %s" % str(e))

        doc1 = chunk_sources(docs1)
    elif (file.lower().endswith('.xlsx') or file.lower().endswith('.xls')) and (have_libreoffice or True):
        docs1 = UnstructuredExcelLoader(file_path=file).load()
        add_meta(docs1, file, parser='UnstructuredExcelLoader')
        doc1 = chunk_sources(docs1)
    elif file.lower().endswith('.odt'):
        docs1 = UnstructuredODTLoader(file_path=file).load()
        add_meta(docs1, file, parser='UnstructuredODTLoader')
        doc1 = chunk_sources(docs1)
    elif file.lower().endswith('pptx') or file.lower().endswith('ppt'):
        docs1 = UnstructuredPowerPointLoader(file_path=file).load()
        add_meta(docs1, file, parser='UnstructuredPowerPointLoader')
        docs1 = clean_doc(docs1)
        doc1 = chunk_sources(docs1)
    elif file.lower().endswith('.txt'):
        # use UnstructuredFileLoader ?
        docs1 = TextLoader(file, encoding="utf8", autodetect_encoding=True).load()
        # makes just one, but big one
        doc1 = chunk_sources(docs1)
        # Bit odd to change if was original text
        # doc1 = clean_doc(doc1)
        add_meta(doc1, file, parser='TextLoader')
    elif file.lower().endswith('.rtf'):
        docs1 = UnstructuredRTFLoader(file).load()
        add_meta(docs1, file, parser='UnstructuredRTFLoader')
        doc1 = chunk_sources(docs1)
    elif file.lower().endswith('.md'):
        docs1 = UnstructuredMarkdownLoader(file).load()
        add_meta(docs1, file, parser='UnstructuredMarkdownLoader')
        docs1 = clean_doc(docs1)
        doc1 = chunk_sources(docs1, language=Language.MARKDOWN)
    elif file.lower().endswith('.enex'):
        docs1 = EverNoteLoader(file).load()
        add_meta(doc1, file, parser='EverNoteLoader')
        doc1 = chunk_sources(docs1)
    elif file.lower().endswith('.epub'):
        docs1 = UnstructuredEPubLoader(file).load()
        add_meta(docs1, file, parser='UnstructuredEPubLoader')
        doc1 = chunk_sources(docs1)
    elif can_do_audio_transcription or can_do_video_extraction:
        handled = False
        e = None
        if can_do_audio_transcription:
            docs1c = []
            try:
                if model_loaders['asr'] is not None and not isinstance(model_loaders['asr'], (str, bool)):
                    # assumes didn't fork into this process with joblib, else can deadlock
                    if verbose:
                        print("Reuse ASR", flush=True)
                    model_loaders['asr'].load_model()
                else:
                    if verbose:
                        print("Fresh ASR", flush=True)
                    from audio_langchain import H2OAudioCaptionLoader
                    model_loaders['asr'] = H2OAudioCaptionLoader(asr_model=asr_model,
                                                                 asr_gpu=model_loaders['asr'] == 'gpu',
                                                                 gpu_id=asr_gpu_id,
                                                                 )
                model_loaders['asr'].set_audio_paths([file])
                docs1c = model_loaders['asr'].load(from_youtube=False)
                docs1c = [x for x in docs1c if x.page_content]
                add_meta(docs1c, file, parser='H2OAudioCaptionLoader: %s' % asr_model)
                files_out = model_loaders['asr'].files_out
                for file_out_i, file_out in enumerate(files_out):
                    key = 'source_true_' + str(file_out_i)
                    [doci.metadata.update({key: file_out}) for doci in docs1c]
                hash_of_file = hash_file(file)
                [doci.metadata.update(source=file, hashid=hash_of_file) for doci in docs1c]
                docs1c = chunk_sources(docs1c)
                # caption didn't set source, so fix-up meta
                doc1.extend(docs1c)
            except BaseException as e0:
                print("ASR2: %s" % str(e0), flush=True)
                e = e0
            handled |= len(docs1c) > 0

        if can_do_video_extraction:
            docs1c_files = []
            try:
                from vision.extract_movie import extract_unique_frames
                export_dir = extract_unique_frames(file=file, extract_frames=extract_frames)
                docs1c_files = path_to_docs_func(export_dir)
                if os.getenv('FRAMES_AS_SAME_DOC', '0') == '1':
                    add_meta(docs1c_files, file, parser='extract_frames from %s' % file)
                    hash_of_file = hash_file(file)
                    [doci.metadata.update(source=file, hashid=hash_of_file) for doci in docs1c_files]
                else:
                    [x.metadata.update(dict(original_source=file)) for order_id, x in enumerate(docs1c_files)]
                doc1.extend(docs1c_files)
            except BaseException as e0:
                print("Extract YouTube Frames: %s" % str(e0), flush=True)
                e = e0
            handled |= len(docs1c_files) > 0
        if len(doc1) == 0:
            # if literally nothing, show failed to parse so user knows, since unlikely nothing in PDF at all.
            if handled:
                raise ValueError("%s had no valid text, but meta data was parsed" % file)
            else:
                raise ValueError("%s had no valid text and no meta data was parsed: %s" % (file, str(e)))
    elif any(file.lower().endswith(x) for x in set_image_audio_types1):
        handled = False
        e = None
        docs1 = []
        if have_tesseract and enable_ocr:
            file_ocr = fix_image_file(file, do_align=True, do_rotate=True, do_pad=False)
            if verbose:
                print("BEGIN: Tesseract", flush=True)
            try:
                # OCR, somewhat works, but not great
                docs1a = UnstructuredImageLoader(file_ocr, strategy='ocr_only').load()
                # docs1a = UnstructuredImageLoader(file, strategy='hi_res').load()
                docs1a = [x for x in docs1a if x.page_content]
                add_meta(docs1a, file, parser='UnstructuredImageLoader', file_as_source=True)
                [doci.metadata.update(source_true=file_ocr) for doci in docs1a]
                docs1.extend(docs1a)
            except BaseException as e0:
                print("UnstructuredImageLoader: %s" % str(e0), flush=True)
                e = e0
            handled |= len(docs1) > 0
            if verbose:
                print("END: Tesseract", flush=True)
        if have_doctr and enable_doctr:
            file_doctr = fix_image_file(file, do_align=True, do_rotate=True, do_pad=True)
            if verbose:
                print("BEGIN: DocTR", flush=True)
            try:
                if model_loaders['doctr'] is not None and not isinstance(model_loaders['doctr'], (str, bool)):
                    if verbose:
                        print("Reuse DocTR", flush=True)
                    model_loaders['doctr'].load_model()
                else:
                    if verbose:
                        print("Fresh DocTR", flush=True)
                    from image_doctr import H2OOCRLoader
                    model_loaders['doctr'] = H2OOCRLoader(layout_aware=True)
                model_loaders['doctr'].set_document_paths([file_doctr])
                docs1c = model_loaders['doctr'].load()
                docs1c = [x for x in docs1c if x.page_content]
                add_meta(docs1c, file, parser='H2OOCRLoader: %s' % 'DocTR', file_as_source=True)
                # caption didn't set source, so fix-up meta
                hash_of_file = hash_file(file)
                [doci.metadata.update(source=file, source_true=file_doctr, hashid=hash_of_file) for doci in docs1c]
                docs1.extend(docs1c)
            except BaseException as e0:
                print("H2OOCRLoader: %s" % str(e0), flush=True)
                e = e0
            handled |= len(docs1) > 0
            if verbose:
                print("END: DocTR", flush=True)
        if enable_captions:
            file_caption = fix_image_file(file, do_align=False, do_rotate=False, do_pad=False)
            # Caption
            if verbose:
                print("BEGIN: Caption", flush=True)
            try:
                if model_loaders['caption'] is not None and not isinstance(model_loaders['caption'], (str, bool)):
                    # assumes didn't fork into this process with joblib, else can deadlock
                    if verbose:
                        print("Reuse Caption", flush=True)
                    model_loaders['caption'].load_model()
                else:
                    if verbose:
                        print("Fresh Caption", flush=True)
                    from image_captions import H2OImageCaptionLoader
                    model_loaders['caption'] = H2OImageCaptionLoader(caption_gpu=model_loaders['caption'] == 'gpu',
                                                                     caption_model=captions_model,
                                                                     caption_processor=captions_model)
                model_loaders['caption'].set_image_paths([file_caption])
                docs1c = model_loaders['caption'].load()
                docs1c = [x for x in docs1c if x.page_content]
                add_meta(docs1c, file, parser='H2OImageCaptionLoader: %s' % captions_model, file_as_source=True)
                # caption didn't set source, so fix-up meta
                hash_of_file = hash_file(file)
                [doci.metadata.update(source=file, source_true=file_caption, hashid=hash_of_file) for doci in docs1c]
                docs1.extend(docs1c)
            except BaseException as e0:
                print("H2OImageCaptionLoader: %s" % str(e0), flush=True)
                e = e0
            handled |= len(docs1) > 0

            if verbose:
                print("END: Caption", flush=True)
        if enable_pix2struct:
            file_pix = fix_image_file(file, do_align=True, do_rotate=True, do_pad=False)
            # PIX
            if verbose:
                print("BEGIN: Pix2Struct", flush=True)
            try:
                if model_loaders['pix2struct'] is not None and not isinstance(model_loaders['pix2struct'], (str, bool)):
                    if verbose:
                        print("Reuse pix2struct", flush=True)
                    model_loaders['pix2struct'].load_model()
                else:
                    if verbose:
                        print("Fresh pix2struct", flush=True)
                    from image_pix2struct import H2OPix2StructLoader
                    model_loaders['pix2struct'] = H2OPix2StructLoader()
                model_loaders['pix2struct'].set_image_paths([file_pix])
                docs1c = model_loaders['pix2struct'].load()
                docs1c = [x for x in docs1c if x.page_content]
                add_meta(docs1c, file, parser='H2OPix2StructLoader: %s' % model_loaders['pix2struct'],
                         file_as_source=True)
                # caption didn't set source, so fix-up meta
                hash_of_file = hash_file(file)
                [doci.metadata.update(source=file, source_true=file_pix, hashid=hash_of_file) for doci in docs1c]
                docs1.extend(docs1c)
            except BaseException as e0:
                print("H2OPix2StructLoader: %s" % str(e0), flush=True)
                e = e0
            handled |= len(docs1) > 0
            if verbose:
                print("END: Pix2Struct", flush=True)
        if llava_model and enable_llava and 'vllm' not in llava_model:
            file_llava = fix_image_file(file, do_align=True, do_rotate=True, do_pad=False)

            if llava_model.startswith('openai:'):
                if verbose:
                    print("BEGIN: OpenAI docAI", flush=True)
                try:
                    from openai import OpenAI
                    openai_client = OpenAI(base_url=os.getenv('H2OGPT_OPENAI_BASE_URL', 'https://api.openai.com'),
                                           api_key=os.getenv('H2OGPT_OPENAI_API_KEY', 'EMPTY'), timeout=60)
                    if llava_prompt in ['auto', None]:
                        llava_prompt = "Describe the image and what does the image say?"
                    from vision.utils_vision import img_to_base64
                    file_llava_url = img_to_base64(file_llava)
                    content = [{
                        'type': 'text',
                        'text': llava_prompt,
                    }, {
                        'type': 'image_url',
                        'image_url': {
                            'url':
                                file_llava_url,
                        },
                    }]
                    messages = [dict(role='system',
                                     content='You are a keen document vision model that can understand complex images and text and respond to queries or convert text inside images to text.'),
                                dict(role='user', content=content)]
                    stream_output = False
                    gen_server_kwargs = dict()
                    model_name = llava_model.split('openai:')[1]
                    responses = openai_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=stream_output,
                        **gen_server_kwargs,
                    )
                    if responses.choices is None and responses.model_extra:
                        raise RuntimeError("OpenAI Chat failed: %s" % responses.model_extra)
                    res = responses.choices[0].message.content
                    if not res:
                        raise RuntimeError("OpenAI Chat had no response")

                    metadata = dict(source=file, date=str(datetime.now()), input_type='OpenAI DocAI')
                    docs1c = [Document(page_content=res, metadata=metadata)]
                    docs1c = [x for x in docs1c if x.page_content]
                    add_meta(docs1c, file, parser='LLaVa: %s' % llava_model, file_as_source=True)
                    # caption didn't set source, so fix-up meta
                    hash_of_file = hash_file(file)
                    [doci.metadata.update(source=file, source_true=file_llava, hashid=hash_of_file,
                                          llava_prompt=llava_prompt or '') for doci in
                     docs1c]
                    docs1.extend(docs1c)
                except BaseException as e0:
                    print("LLaVa: %s: %s" % (str(e0), traceback.print_exception(e0)), flush=True)
                    e = e0
                handled |= len(docs1) > 0
                if verbose:
                    print("END: OpenAI docAI", flush=True)
            else:
                # LLaVa
                if verbose:
                    print("BEGIN: LLaVa", flush=True)
                try:
                    from vision.utils_vision import get_llava_response
                    res, llava_prompt = get_llava_response(file_llava, llava_model,
                                                           prompt=llava_prompt,
                                                           allow_prompt_auto=True,
                                                           max_time=60,  # not too much time for docQA
                                                           verbose=verbose,
                                                           )
                    metadata = dict(source=file, date=str(datetime.now()), input_type='LLaVa')
                    docs1c = [Document(page_content=res, metadata=metadata)]
                    docs1c = [x for x in docs1c if x.page_content]
                    add_meta(docs1c, file, parser='LLaVa: %s' % llava_model, file_as_source=True)
                    # caption didn't set source, so fix-up meta
                    hash_of_file = hash_file(file)
                    [doci.metadata.update(source=file, source_true=file_llava, hashid=hash_of_file,
                                          llava_prompt=llava_prompt or '') for doci in
                     docs1c]
                    docs1.extend(docs1c)
                except BaseException as e0:
                    print("LLaVa: %s: %s" % (str(e0), traceback.print_exception(e0)), flush=True)
                    e = e0
                handled |= len(docs1) > 0
                if verbose:
                    print("END: LLaVa", flush=True)

        doc1 = chunk_sources(docs1)
        if len(doc1) == 0:
            # if literally nothing, show failed to parse so user knows, since unlikely nothing in PDF at all.
            if handled:
                raise ValueError("%s had no valid text, but meta data was parsed" % file)
            else:
                raise ValueError("%s had no valid text and no meta data was parsed: %s" % (file, str(e)))
    elif file.lower().endswith('.msg'):
        raise RuntimeError("Not supported, GPL3 license")
        # docs1 = OutlookMessageLoader(file).load()
        # docs1[0].metadata['source'] = file
    elif file.lower().endswith('.eml'):
        try:
            docs1 = UnstructuredEmailLoader(file).load()
            add_meta(docs1, file, parser='UnstructuredEmailLoader')
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
            add_meta(docs1, file, parser='UnstructuredEmailLoader text/plain')
            doc1 = chunk_sources(docs1)
    # elif file.lower().endswith('.gcsdir'):
    #    doc1 = GCSDirectoryLoader(project_name, bucket, prefix).load()
    # elif file.lower().endswith('.gcsfile'):
    # doc1 = GCSFileLoader(project_name, bucket, blob).load()
    elif file.lower().endswith('.rst'):
        with open(file, "r") as f:
            doc1 = Document(page_content=str(f.read()), metadata={"source": file})
        add_meta(doc1, file, parser='f.read()')
        doc1 = chunk_sources(doc1, language=Language.RST)
    elif file.lower().endswith('.json'):
        # 10k rows, 100 columns-like parts 4 bytes each
        JSON_SIZE_LIMIT = int(os.getenv('JSON_SIZE_LIMIT', str(10 * 10 * 1024 * 10 * 4)))
        if os.path.getsize(file) > JSON_SIZE_LIMIT:
            raise ValueError(
                "JSON file sizes > %s not supported for naive parsing and embedding, requires Agents enabled" % JSON_SIZE_LIMIT)
        loader = JSONLoader(
            file_path=file,
            # jq_schema='.messages[].content',
            jq_schema=jq_schema,
            text_content=False,
            metadata_func=json_metadata_func)
        try:
            doc1 = loader.load()
            add_meta(doc1, file, parser='JSONLoader: %s' % jq_schema)
            fix_json_meta(doc1)
        except Exception as e:
            if os.getenv("TRYJSONASTEXT", '1') == '0':
                raise
            # revert to treating as text
            metadata = dict(source=file, date=str(datetime.now()), input_type='JSONAsText')
            with open(file, "r") as f:
                doc1 = Document(page_content=str(f.read()), metadata=metadata)
            add_meta(doc1, file, parser='JSONAsTextLoader: json failed with: %s' % str(e))
        doc1 = chunk_sources(doc1)
    elif file.lower().endswith('.jsonl'):
        loader = JSONLoader(
            file_path=file,
            # jq_schema='.messages[].content',
            jq_schema=jq_schema,
            json_lines=True,
            text_content=False,
            metadata_func=json_metadata_func)
        try:
            doc1 = loader.load()
            add_meta(doc1, file, parser='JSONLLoader: %s' % jq_schema)
            fix_json_meta(doc1)
        except Exception as e:
            if os.getenv("TRYJSONASTEXT", '1') == '0':
                raise
            # revert to treating as text
            metadata = dict(source=file, date=str(datetime.now()), input_type='JSONLAsText')
            with open(file, "r") as f:
                doc1 = Document(page_content=str(f.read()), metadata=metadata)
            add_meta(doc1, file, parser='JSONLAsTextLoader: jsonl failed with: %s' % str(e))
        doc1 = chunk_sources(doc1)
    elif file.lower().endswith('.pdf'):
        # migration
        if isinstance(use_pymupdf, bool):
            if use_pymupdf == False:
                use_pymupdf = 'off'
            if use_pymupdf == True:
                use_pymupdf = 'on'
        if isinstance(use_unstructured_pdf, bool):
            if use_unstructured_pdf == False:
                use_unstructured_pdf = 'off'
            if use_unstructured_pdf == True:
                use_unstructured_pdf = 'on'
        if isinstance(use_pypdf, bool):
            if use_pypdf == False:
                use_pypdf = 'off'
            if use_pypdf == True:
                use_pypdf = 'on'
        if isinstance(enable_pdf_ocr, bool):
            if enable_pdf_ocr == False:
                enable_pdf_ocr = 'off'
            if enable_pdf_ocr == True:
                enable_pdf_ocr = 'on'
        if isinstance(try_pdf_as_html, bool):
            if try_pdf_as_html == False:
                try_pdf_as_html = 'off'
            if try_pdf_as_html == True:
                try_pdf_as_html = 'on'

        num_pages = get_num_pages(file)

        doc1 = []
        tried_others = False
        handled = False
        did_pymupdf = False
        did_unstructured = False
        e = None
        if (have_pymupdf or have_pymupdf4llm) and (len(doc1) == 0 and use_pymupdf == 'auto' or use_pymupdf == 'on'):
            # GPL, only use if installed
            from langchain_community.document_loaders import PyMuPDFLoader
            # load() still chunks by pages, but every page has title at start to help
            try:
                if have_pymupdf4llm:
                    doc1a = PyMuPDF4LLMLoader(file).load()
                elif have_pymupdf:
                    doc1a = PyMuPDFLoader(file).load()
                else:
                    raise ValueError("no valid version of pymupdf")
                did_pymupdf = True
            except BaseException as e0:
                doc1a = []
                print("PyMuPDFLoader: %s" % str(e0), flush=True)
                e = e0
            # remove empty documents
            handled |= len(doc1a) > 0
            doc1a = [x for x in doc1a if x.page_content]
            doc1a = clean_doc(doc1a)
            add_parser(doc1a, 'PyMuPDFLoader')
            doc1.extend(doc1a)
        # PyPDF is first if PyMuPDF not installed
        if len(doc1) == 0 and use_pypdf == 'auto' or use_pypdf == 'on':
            tried_others = True
            # open-source fallback
            # load() still chunks by pages, but every page has title at start to help
            try:
                doc1a = PyPDFLoader(file).load()
            except BaseException as e0:
                doc1a = []
                print("PyPDFLoader: %s" % str(e0), flush=True)
                e = e0
            handled |= len(doc1a) > 0
            # remove empty documents
            doc1a = [x for x in doc1a if x.page_content]
            doc1a = clean_doc(doc1a)
            add_parser(doc1a, 'PyPDFLoader')
            doc1.extend(doc1a)
        # do OCR/tesseract if only 2 page and auto, since doctr superior and faster
        if (len(doc1) == 0 or num_pages is not None and num_pages < 2) and use_unstructured_pdf == 'auto' \
                or use_unstructured_pdf == 'on':
            tried_others = True
            try:
                doc1a = UnstructuredPDFLoader(file).load()
                did_unstructured = True
            except BaseException as e0:
                doc1a = []
                print("UnstructuredPDFLoader: %s" % str(e0), flush=True)
                e = e0
            handled |= len(doc1a) > 0
            # remove empty documents
            doc1a = [x for x in doc1a if x.page_content]
            add_parser(doc1a, 'UnstructuredPDFLoader')
            # seems to not need cleaning in most cases
            doc1.extend(doc1a)
        if not did_pymupdf and ((have_pymupdf and len(doc1) == 0) and tried_others):
            # try again in case only others used, but only if didn't already try (2nd part of and)
            # GPL, only use if installed
            from langchain_community.document_loaders import PyMuPDFLoader
            # load() still chunks by pages, but every page has title at start to help
            try:
                doc1a = PyMuPDFLoader(file).load()
            except BaseException as e0:
                doc1a = []
                print("PyMuPDFLoader: %s" % str(e0), flush=True)
                e = e0
            handled |= len(doc1a) > 0
            # remove empty documents
            doc1a = [x for x in doc1a if x.page_content]
            doc1a = clean_doc(doc1a)
            add_parser(doc1a, 'PyMuPDFLoader2')
            doc1.extend(doc1a)
        did_pdf_ocr = False
        if len(doc1) == 0 and (enable_pdf_ocr == 'auto' and enable_pdf_doctr != 'on') or enable_pdf_ocr == 'on':
            did_pdf_ocr = True
            # no did_unstructured condition here because here we do OCR, and before we did not
            # try OCR in end since slowest, but works on pure image pages well
            try:
                doc1a = UnstructuredPDFLoader(file, strategy='ocr_only').load()
            except BaseException as e0:
                doc1a = []
                print("UnstructuredPDFLoader: %s" % str(e0), flush=True)
                e = e0
            handled |= len(doc1a) > 0
            # remove empty documents
            doc1a = [x for x in doc1a if x.page_content]
            add_parser(doc1a, 'UnstructuredPDFLoader ocr_only')
            # seems to not need cleaning in most cases
            doc1.extend(doc1a)
        # Some PDFs return nothing or junk from PDFMinerLoader
        # if auto, do doctr pdf if not too many pages, else can be slow/expensive
        if (len(doc1) == 0 or num_pages is not None and num_pages < 5) and enable_pdf_doctr == 'auto' or \
                enable_pdf_doctr == 'on':
            if verbose:
                print("BEGIN: DocTR", flush=True)
            if model_loaders['doctr'] is not None and not isinstance(model_loaders['doctr'], (str, bool)):
                model_loaders['doctr'].load_model()
            else:
                from image_doctr import H2OOCRLoader
                model_loaders['doctr'] = H2OOCRLoader(layout_aware=True)
            # avoid having all pages in memory at same time, for large PDFs leads to system OOM
            try:
                pages = get_each_page(file)
                got_pages = True
            except Exception as e:
                # FIXME: protection for now, unsure how generally will work
                print("Exception in doctr page handling: %s" % str(e), flush=True)
                pages = [file]
                got_pages = False
            try:
                model_loaders['doctr'].set_document_paths(pages)
                doc1a = model_loaders['doctr'].load()
            except BaseException as e0:
                doc1a = []
                print("H2OOCRLoader: %s" % str(e0), flush=True)
                e = e0
            doc1a = [x for x in doc1a if x.page_content]
            add_meta(doc1a, file, parser='H2OOCRLoader: %s' % 'DocTR')
            handled |= len(doc1a) > 0
            if got_pages:
                for page in pages:
                    remove(page)
            # caption didn't set source, so fix-up meta
            hash_of_file = hash_file(file)
            [doci.metadata.update(source=file, hashid=hash_of_file) for doci in doc1a]
            doc1.extend(doc1a)
            if verbose:
                print("END: DocTR", flush=True)
        if try_pdf_as_html in ['auto', 'on']:
            doc1a = try_as_html(file)
            add_parser(doc1a, 'try_as_html')
            doc1.extend(doc1a)

        if len(doc1) == 0:
            # if literally nothing, show failed to parse so user knows, since unlikely nothing in PDF at all.
            if handled:
                raise ValueError("%s had no valid text, but meta data was parsed" % file)
            else:
                raise ValueError("%s had no valid text and no meta data was parsed: %s" % (file, str(e)))
        add_meta(doc1, file, parser='pdf')
        doc1 = chunk_sources(doc1)
    elif file.lower().endswith('.csv'):
        CSV_SIZE_LIMIT = int(os.getenv('CSV_SIZE_LIMIT', str(10 * 1024 * 10 * 4)))
        if os.path.getsize(file) > CSV_SIZE_LIMIT:
            raise ValueError(
                "CSV file sizes > %s not supported for naive parsing and embedding, requires Agents enabled" % CSV_SIZE_LIMIT)
        doc1 = CSVLoader(file).load()
        add_meta(doc1, file, parser='CSVLoader')
        if isinstance(doc1, list):
            # each row is a Document, identify
            [x.metadata.update(dict(chunk_id=chunk_id)) for chunk_id, x in enumerate(doc1)]
            if db_type in ['chroma', 'chroma_old']:
                # then separate summarize list
                sdoc1 = clone_documents(doc1)
                [x.metadata.update(dict(chunk_id=-1)) for chunk_id, x in enumerate(sdoc1)]
                doc1 = sdoc1 + doc1
    elif file.lower().endswith('.py'):
        doc1 = PythonLoader(file).load()
        add_meta(doc1, file, parser='PythonLoader')
        doc1 = chunk_sources(doc1, language=Language.PYTHON)
    elif file.lower().endswith('.toml'):
        doc1 = TomlLoader(file).load()
        add_meta(doc1, file, parser='TomlLoader')
        doc1 = chunk_sources(doc1)
    elif file.lower().endswith('.xml'):
        from langchain_community.document_loaders import UnstructuredXMLLoader
        loader = UnstructuredXMLLoader(file_path=file)
        doc1 = loader.load()
        add_meta(doc1, file, parser='UnstructuredXMLLoader')
    elif file.lower().endswith('.urls'):
        with open(file, "r") as f:
            urls = f.readlines()
            # recurse
            doc1 = path_to_docs_func(None, url=urls)
    elif file.lower().endswith('.zip'):
        with zipfile.ZipFile(file, 'r') as zip_ref:
            # don't put into temporary path, since want to keep references to docs inside zip
            # so just extract in path where
            zip_ref.extractall(base_path)
            # recurse
            doc1 = path_to_docs_func(base_path)
    elif file.lower().endswith('.tar.gz') or file.lower().endswith('.tgz'):
        with tarfile.open(file, 'r') as tar_ref:
            # don't put into temporary path, since want to keep references to docs inside tar.gz
            # so just extract in path where
            tar_ref.extractall(base_path)
            # recurse
            doc1 = path_to_docs_func(base_path)
    elif file.lower().endswith('.gz') or file.lower().endswith('.gzip'):
        if file.lower().endswith('.gz'):
            de_file = file.lower().replace('.gz', '')
        else:
            de_file = file.lower().replace('.gzip', '')
        with gzip.open(file, 'rb') as f_in:
            with open(de_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        # recurse
        doc1 = path_to_docs_func(de_file,
                                 filei=filei,  # single file, same file index as outside caller
                                 )

    else:
        raise RuntimeError("No file handler for %s" % os.path.basename(file))

    # allow doc1 to be list or not.
    if not isinstance(doc1, list):
        # If not list, did not chunk yet, so chunk now
        docs = chunk_sources([doc1])
    else:
        if len(doc1) == 1:
            # if list of length one, don't trust and chunk it, chunk_id's will still be correct if repeat
            docs = chunk_sources(doc1)
        elif doc1 and doc1[0].metadata.get('chunk_id') is None:
            if os.getenv('HARD_ASSERTS'):
                raise ValueError("Did not set chunk_id: %s" % str(doc1))
            docs = chunk_sources(doc1)
        else:
            docs = doc1

    assert isinstance(docs, list)

    if orig_url is not None:
        # go back to URL as source
        [doci.metadata.update(source=orig_url) for doci in doc1]

    if is_public:
        if len(docs) > max_chunks_per_doc_public and from_ui or \
                len(docs) > max_chunks_per_doc_public_api and not from_ui:
            raise ValueError("Public instance only allows up to"
                             " %s (%s from API) chunks "
                             "per document." % (max_chunks_per_doc_public, max_chunks_per_doc_public_api))

    return docs


def path_to_doc1(file,
                 filei=0,
                 verbose=False, fail_any_exception=False, return_file=True,
                 chunk=True, chunk_size=512,
                 n_jobs=-1,
                 is_url=False, is_txt=False,

                 # urls
                 use_unstructured=True,
                 use_playwright=False,
                 use_selenium=False,
                 use_scrapeplaywright=False,
                 use_scrapehttp=False,

                 # pdfs
                 use_pymupdf='auto',
                 use_unstructured_pdf='auto',
                 use_pypdf='auto',
                 enable_pdf_ocr='auto',
                 enable_pdf_doctr='auto',
                 try_pdf_as_html='auto',

                 # images
                 enable_ocr=False,
                 enable_doctr=False,
                 enable_pix2struct=False,
                 enable_captions=True,
                 enable_llava=True,
                 enable_transcriptions=True,
                 captions_model=None,
                 llava_model=None,
                 asr_model=None,

                 # json
                 jq_schema='.[]',
                 extract_frames=10,
                 llava_prompt=None,

                 model_loaders=None,

                 headsize=50,
                 db_type=None,
                 selected_file_types=None,

                 is_public=False,
                 from_ui=True,

                 hf_embedding_model=None,
                 use_openai_embedding=False,
                 ):
    assert db_type is not None
    if verbose:
        if is_url and is_txt:
            print("Ingesting URL or Text: %s" % file, flush=True)
        elif is_url:
            print("Ingesting URL: %s" % file, flush=True)
        elif is_txt:
            print("Ingesting Text: %s" % file, flush=True)
        else:
            print("Ingesting file: %s" % file, flush=True)
    res = None
    try:
        # don't pass base_path=path, would infinitely recurse
        res = file_to_doc(file,
                          filei=filei,
                          base_path=None,
                          verbose=verbose, fail_any_exception=fail_any_exception,
                          chunk=chunk, chunk_size=chunk_size,
                          n_jobs=n_jobs,
                          is_url=is_url, is_txt=is_txt,

                          # urls
                          use_unstructured=use_unstructured,
                          use_playwright=use_playwright,
                          use_selenium=use_selenium,
                          use_scrapeplaywright=use_scrapeplaywright,
                          use_scrapehttp=use_scrapehttp,

                          # pdfs
                          use_pymupdf=use_pymupdf,
                          use_unstructured_pdf=use_unstructured_pdf,
                          use_pypdf=use_pypdf,
                          enable_pdf_ocr=enable_pdf_ocr,
                          enable_pdf_doctr=enable_pdf_doctr,
                          try_pdf_as_html=try_pdf_as_html,

                          # images
                          enable_ocr=enable_ocr,
                          enable_doctr=enable_doctr,
                          enable_pix2struct=enable_pix2struct,
                          enable_captions=enable_captions,
                          enable_llava=enable_llava,
                          enable_transcriptions=enable_transcriptions,
                          captions_model=captions_model,
                          llava_model=llava_model,
                          llava_prompt=llava_prompt,
                          asr_model=asr_model,

                          model_loaders=model_loaders,

                          # json
                          jq_schema=jq_schema,

                          # video
                          extract_frames=extract_frames,

                          headsize=headsize,
                          db_type=db_type,
                          selected_file_types=selected_file_types,
                          is_public=is_public,
                          from_ui=from_ui,

                          hf_embedding_model=hf_embedding_model,
                          use_openai_embedding=use_openai_embedding,
                          )
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
        if is_url and is_txt:
            print("DONE Ingesting URL or Text: %s" % file, flush=True)
        elif is_url:
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


def path_to_docs(path_or_paths,
                 filei=None,
                 url=None, text=None,

                 verbose=False, fail_any_exception=False, n_jobs=-1,
                 chunk=True, chunk_size=512,

                 # urls
                 use_unstructured=True,
                 use_playwright=False,
                 use_selenium=False,
                 use_scrapeplaywright=False,
                 use_scrapehttp=False,

                 # pdfs
                 use_pymupdf='auto',
                 use_unstructured_pdf='auto',
                 use_pypdf='auto',
                 enable_pdf_ocr='auto',
                 enable_pdf_doctr='auto',
                 try_pdf_as_html='auto',

                 # images
                 enable_ocr=False,
                 enable_doctr=False,
                 enable_pix2struct=False,
                 enable_captions=True,
                 enable_llava=True,
                 enable_transcriptions=True,
                 captions_model=None,
                 llava_model=None,
                 llava_prompt=None,
                 asr_model=None,

                 caption_loader=None,
                 doctr_loader=None,
                 pix2struct_loader=None,
                 asr_loader=None,

                 # json
                 jq_schema='.[]',
                 # video
                 extract_frames=10,

                 db_type=None,
                 is_public=False,

                 existing_files=[],
                 existing_hash_ids={},
                 selected_file_types=None,

                 from_ui=True,

                 use_openai_embedding=False,
                 hf_embedding_model=None,
                 ):
    if verbose:
        print("BEGIN Consuming path_or_paths=%s url=%s text=%s" % (path_or_paths, url, text), flush=True)
    if selected_file_types is not None:
        non_image_audio_types1 = [x for x in non_image_types if x in selected_file_types]
        image_audio_types1 = [x for x in image_types + audio_types if x in selected_file_types]
    else:
        non_image_audio_types1 = non_image_types.copy()
        image_audio_types1 = image_types.copy() + audio_types.copy()

    assert db_type is not None
    # path_or_paths could be str, list, tuple, generator
    globs_image_audio_types = []
    globs_non_image_types = []
    if not path_or_paths and not url and not text:
        return []
    elif url:
        # ok if text too
        url = get_list_or_str(url)
        globs_non_image_types = url if isinstance(url, (list, tuple, types.GeneratorType)) else [url]
    elif text:
        globs_non_image_types = text if isinstance(text, (list, tuple, types.GeneratorType)) else [text]
    elif isinstance(path_or_paths, str) and os.path.isdir(path_or_paths):
        # single path, only consume allowed files
        path = path_or_paths
        # Below globs should match patterns in file_to_doc()
        [globs_image_audio_types.extend(glob.glob(os.path.join(path, "./**/*.%s" % ftype), recursive=True))
         for ftype in image_audio_types1]
        globs_image_audio_types = [os.path.normpath(x) for x in globs_image_audio_types]
        [globs_non_image_types.extend(glob.glob(os.path.join(path, "./**/*.%s" % ftype), recursive=True))
         for ftype in non_image_audio_types1]
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
        globs_image_audio_types.extend(
            flatten_list([[os.path.normpath(x) for x in path_or_paths if x.endswith(y)] for y in image_audio_types1]))
        # could do below:
        # globs_non_image_types = flatten_list([[x for x in path_or_paths if x.endswith(y)] for y in non_image_audio_types1])
        # But instead, allow fail so can collect unsupported too
        set_globs_image_audio_types = set(globs_image_audio_types)
        globs_non_image_types.extend(
            [os.path.normpath(x) for x in path_or_paths if x not in set_globs_image_audio_types])

    # filter out any files to skip (e.g. if already processed them)
    # this is easy, but too aggressive in case a file changed, so parent probably passed existing_files=[]
    assert not existing_files, "DEV: assume not using this approach"
    if existing_files:
        set_skip_files = set(existing_files)
        globs_image_audio_types = [x for x in globs_image_audio_types if x not in set_skip_files]
        globs_non_image_types = [x for x in globs_non_image_types if x not in set_skip_files]
    if existing_hash_ids:
        # assume consistent with add_meta() use of hash_file(file)
        # also assume consistent with get_existing_hash_ids for dict creation
        # assume hashable values
        existing_hash_ids_set = set(existing_hash_ids.items())
        hash_ids_all_image_audio = set({x: hash_file(x) for x in globs_image_audio_types}.items())
        hash_ids_all_non_image = set({x: hash_file(x) for x in globs_non_image_types}.items())
        # don't use symmetric diff.  If file is gone, ignore and don't remove or something
        #  just consider existing files (key) having new hash or not (value)
        new_files_image_audio = set(dict(hash_ids_all_image_audio - existing_hash_ids_set).keys())
        new_files_non_image = set(dict(hash_ids_all_non_image - existing_hash_ids_set).keys())
        globs_image_audio_types = [x for x in globs_image_audio_types if x in new_files_image_audio]
        globs_non_image_types = [x for x in globs_non_image_types if x in new_files_non_image]

    # could use generator, but messes up metadata handling in recursive case
    # FIXME: n_gpus=n_gpus?
    if caption_loader and not isinstance(caption_loader, (bool, str)) and caption_loader.device != 'cpu' or \
            get_device() == 'cuda' or \
            asr_loader and not isinstance(asr_loader, (bool, str)) and asr_loader.pipe.device != 'cpu' or \
            get_device() == 'cuda':
        # to avoid deadlocks, presume was preloaded and so can't fork due to cuda context
        # get_device() == 'cuda' because presume faster to process image from (temporarily) preloaded model
        n_jobs_image = 1
    else:
        n_jobs_image = n_jobs
    if enable_doctr or enable_pdf_doctr in [True, 'auto', 'on']:
        if doctr_loader and not isinstance(doctr_loader, (bool, str)) and doctr_loader.device != 'cpu':
            # can't fork cuda context
            n_jobs = 1

    return_file = True  # local choice
    is_url = url is not None
    is_txt = text is not None
    model_loaders = dict(caption=caption_loader,
                         doctr=doctr_loader,
                         pix2struct=pix2struct_loader,
                         asr=asr_loader)
    model_loaders0 = model_loaders.copy()
    kwargs = dict(verbose=verbose, fail_any_exception=fail_any_exception,
                  return_file=return_file,
                  chunk=chunk, chunk_size=chunk_size,
                  n_jobs=n_jobs,
                  is_url=is_url,
                  is_txt=is_txt,

                  # urls
                  use_unstructured=use_unstructured,
                  use_playwright=use_playwright,
                  use_selenium=use_selenium,
                  use_scrapeplaywright=use_scrapeplaywright,
                  use_scrapehttp=use_scrapehttp,

                  # pdfs
                  use_pymupdf=use_pymupdf,
                  use_unstructured_pdf=use_unstructured_pdf,
                  use_pypdf=use_pypdf,
                  enable_pdf_ocr=enable_pdf_ocr,
                  enable_pdf_doctr=enable_pdf_doctr,
                  try_pdf_as_html=try_pdf_as_html,

                  # images
                  enable_ocr=enable_ocr,
                  enable_doctr=enable_doctr,
                  enable_pix2struct=enable_pix2struct,
                  enable_captions=enable_captions,
                  enable_llava=enable_llava,
                  enable_transcriptions=enable_transcriptions,
                  captions_model=captions_model,
                  llava_model=llava_model,
                  llava_prompt=llava_prompt,
                  asr_model=asr_model,

                  model_loaders=model_loaders,

                  # json
                  jq_schema=jq_schema,
                  extract_frames=extract_frames,

                  db_type=db_type,
                  selected_file_types=selected_file_types,

                  is_public=is_public,
                  from_ui=from_ui,

                  hf_embedding_model=hf_embedding_model,
                  use_openai_embedding=use_openai_embedding,
                  )

    if is_public:
        n_docs = len(globs_non_image_types) + len(globs_image_audio_types)
        if n_docs > max_docs_public and from_ui or \
                n_docs > max_docs_public_api and not from_ui:
            raise ValueError(
                "Public instance only allows up to %d documents "
                "(including in zip) (%d for API) updated at a time." % (max_docs_public, max_docs_public_api))

    def no_tqdm(x):
        return x

    my_tqdm = no_tqdm if not verbose else tqdm
    filei0 = filei

    fork_lots_ok = kwargs['hf_embedding_model'] and \
                   'name' in kwargs['hf_embedding_model'] and \
                   kwargs['hf_embedding_model']['name'] and \
                   kwargs['hf_embedding_model']['name'].startswith('tei')
    if not fork_lots_ok:
        # else can hit OSError: [Errno 12] Cannot allocate memory
        n_jobs = max(1, min(8, n_jobs))

    if n_jobs != 1 and len(globs_non_image_types) > 1:
        kwargs['hf_embedding_model'] = None  # can't fork and use CUDA
        # avoid nesting, e.g. upload 1 zip and then inside many files
        # harder to handle if upload many zips with many files, inner parallel one will be disabled by joblib
        documents = ProgressParallel(n_jobs=n_jobs, verbose=10 if verbose else 0, backend='multiprocessing')(
            delayed(path_to_doc1)(file, filei=filei0 or filei, **kwargs) for filei, file in
            enumerate(globs_non_image_types)
        )
    else:
        documents = [path_to_doc1(file, filei=filei0 or filei, **kwargs) for filei, file in
                     enumerate(my_tqdm(globs_non_image_types))]

    # do images separately since can't fork after cuda in parent, so can't be parallel
    if n_jobs_image != 1 and len(globs_image_audio_types) > 1:
        # avoid nesting, e.g. upload 1 zip and then inside many files
        # harder to handle if upload many zips with many files, inner parallel one will be disabled by joblib
        image_documents = ProgressParallel(n_jobs=n_jobs, verbose=10 if verbose else 0, backend='multiprocessing')(
            delayed(path_to_doc1)(file, filei=filei0 or filei, **kwargs) for filei, file in
            enumerate(globs_image_audio_types)
        )
    else:
        image_documents = [path_to_doc1(file, filei=filei0 or filei, **kwargs) for filei, file in
                           enumerate(my_tqdm(globs_image_audio_types))]

    # unload loaders (image loaders, includes enable_pdf_doctr that uses same loader)
    for name, loader in model_loaders.items():
        loader0 = model_loaders0[name]
        real_model_initial = loader0 is not None and not isinstance(loader0, (str, bool))
        real_model_final = model_loaders[name] is not None and not isinstance(model_loaders[name], (str, bool))
        if not real_model_initial and real_model_final:
            # clear off GPU newly added model
            model_loaders[name].unload_model()

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
                   n_jobs=-1, embedding_gpu_id=0,
                   kwargs_make_db={},
                   verbose=False):
    """
    do prep first time, involving downloads
    # FIXME: Add github caching then add here
    :return:
    """
    if os.getenv("HARD_ASSERTS"):
        assert langchain_mode not in ['MyData'], "Should not prep scratch/personal data"

    if langchain_mode in langchain_modes_intrinsic:
        return None

    db_dir_exists = os.path.isdir(persist_directory)
    user_path = langchain_mode_paths.get(langchain_mode)

    if db_dir_exists and user_path is None:
        if verbose:
            print("Prep: persist_directory=%s exists, using" % persist_directory, flush=True)
        db, use_openai_embedding, hf_embedding_model = \
            get_existing_db(None, persist_directory, load_db_if_exists,
                            db_type, use_openai_embedding,
                            langchain_mode, langchain_mode_paths, langchain_mode_types,
                            hf_embedding_model, migrate_embedding_model,
                            n_jobs=n_jobs, embedding_gpu_id=embedding_gpu_id)
    else:
        if db_dir_exists and user_path is not None:
            if verbose:
                print("Prep: persist_directory=%s exists, user_path=%s passed, adding any changed or new documents" % (
                    persist_directory, user_path), flush=True)
        elif not db_dir_exists:
            if verbose:
                print("Prep: persist_directory=%s does not exist, regenerating" % persist_directory, flush=True)
        db = None
        if langchain_mode in ['DriverlessAI docs']:
            # FIXME: Could also just use dai_docs.pickle directly and upload that
            get_dai_docs(from_hf=True)

        if langchain_mode in ['wiki']:
            get_wiki_sources(first_para=kwargs_make_db['first_para'], text_limit=kwargs_make_db['text_limit'])

        langchain_kwargs = kwargs_make_db.copy()
        langchain_kwargs.update(locals().copy())
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


def get_hf_embedding_model_name(hf_embedding_model):
    if isinstance(hf_embedding_model, dict):
        # embedding itself preloaded globally
        hf_embedding_model = hf_embedding_model['name']
    return hf_embedding_model


def check_update_chroma_embedding(db,
                                  db_type,
                                  use_openai_embedding,
                                  hf_embedding_model, migrate_embedding_model,
                                  langchain_mode, langchain_mode_paths, langchain_mode_types,
                                  n_jobs=-1,
                                  verbose=False):
    changed_db = False
    embed_tuple = load_embed(db=db, use_openai_embedding=use_openai_embedding)

    # expect string comparison, if dict then model object with name and get name not dict or model
    hf_embedding_model = get_hf_embedding_model_name(hf_embedding_model)

    if embed_tuple not in [(True, use_openai_embedding, hf_embedding_model),
                           (False, use_openai_embedding, hf_embedding_model)]:
        print("Detected new embedding %s vs. %s %s, updating db: %s" % (
            use_openai_embedding, hf_embedding_model, embed_tuple, langchain_mode), flush=True)
        # handle embedding changes
        db_get = get_documents(db)
        sources = [Document(page_content=result[0], metadata=result[1] or {})
                   for result in zip(db_get['documents'], db_get['metadatas'])]
        # delete index, has to be redone
        persist_directory = db._persist_directory
        shutil.move(persist_directory, persist_directory + "_" + str(uuid.uuid4()) + ".bak")
        assert db_type in ['chroma', 'chroma_old']
        load_db_if_exists = False
        db = get_db(sources, use_openai_embedding=use_openai_embedding, db_type=db_type,
                    persist_directory=persist_directory, load_db_if_exists=load_db_if_exists,
                    langchain_mode=langchain_mode,
                    langchain_mode_paths=langchain_mode_paths,
                    langchain_mode_types=langchain_mode_types,
                    collection_name=None,
                    hf_embedding_model=hf_embedding_model,
                    migrate_embedding_model=migrate_embedding_model,
                    n_jobs=n_jobs,
                    verbose=verbose,
                    )
        changed_db = True
        print("Done updating db for new embedding: %s" % langchain_mode, flush=True)

    return db, changed_db


def migrate_meta_func(db, langchain_mode):
    changed_db = False
    if db is None:
        return db, changed_db

    if is_new_chroma_db(db):
        # when added new chroma db, already had chunk_id
        # so never need to migrate new db that does expensive db.get() because chunk_id always in new db
        return db, changed_db

    # full db.get() expensive, do faster trial with sim search
    # so can just check one doc as consistent or not
    docs1 = db.similarity_search("", k=1)
    if len(docs1) == 0:
        return db, changed_db
    doc1 = docs1[0]
    metadata1 = doc1.metadata
    if 'chunk_id' not in metadata1:
        print("Detected old metadata without chunk_id, adding additional information", flush=True)
        t0 = time.time()
        db_get = get_documents(db)
        # handle meta changes
        changed_db = True
        [x.update(dict(chunk_id=x.get('chunk_id', 0))) for x in db_get['metadatas']]
        client_collection = db._client.get_collection(name=db._collection.name,
                                                      embedding_function=db._collection._embedding_function)
        client_collection.update(ids=db_get['ids'], metadatas=db_get['metadatas'])
        if os.getenv('HARD_ASSERTS'):
            # check
            db_get = get_documents(db)
            assert 'chunk_id' in db_get['metadatas'][0], "Failed to add meta"
        print("Done updating db for new meta: %s in %s seconds" % (langchain_mode, time.time() - t0), flush=True)

    return db, changed_db


def get_existing_db(db, persist_directory,
                    load_db_if_exists, db_type, use_openai_embedding,
                    langchain_mode, langchain_mode_paths, langchain_mode_types,
                    hf_embedding_model,
                    migrate_embedding_model,
                    verbose=False, check_embedding=True, migrate_meta=True,
                    n_jobs=-1,
                    embedding_gpu_id=0):
    if load_db_if_exists and db_type in ['chroma', 'chroma_old'] and os.path.isdir(persist_directory):
        if os.path.isfile(os.path.join(persist_directory, 'chroma.sqlite3')):
            must_migrate = False
        elif os.path.isdir(os.path.join(persist_directory, 'index')):
            raise RuntimeError("Migration no longer supported")
        else:
            return db, use_openai_embedding, hf_embedding_model
        chroma_settings = dict(is_persistent=True)
        use_chromamigdb = False
        if must_migrate:
            raise ValueError(
                "Detected chromadb<0.4 database, not supported")

        if db is None:
            if verbose:
                print("DO Loading db: %s" % langchain_mode, flush=True)
            got_embedding, use_openai_embedding0, hf_embedding_model0 = load_embed(persist_directory=persist_directory,
                                                                                   use_openai_embedding=use_openai_embedding)
            if got_embedding and hf_embedding_model and 'name' in hf_embedding_model and hf_embedding_model0 == \
                    hf_embedding_model['name']:
                # already have
                embedding = hf_embedding_model['model']
            else:
                if got_embedding:
                    # doesn't match, must load new
                    use_openai_embedding, hf_embedding_model = use_openai_embedding0, hf_embedding_model0
                else:
                    if hf_embedding_model and 'name' in hf_embedding_model:
                        # if no embedding, use same as preloaded
                        hf_embedding_model = hf_embedding_model['name']
                embedding = get_embedding(use_openai_embedding, hf_embedding_model=hf_embedding_model,
                                          gpu_id=embedding_gpu_id)
            import logging
            logging.getLogger("chromadb").setLevel(logging.ERROR)
            if use_chromamigdb:
                raise RuntimeError("Migration no longer supported")
            else:
                from chromadb.config import Settings
                chroma_class = Chroma
                if os.path.isdir(persist_directory):
                    import chromadb
                    api_kwargs = dict(client=chromadb.PersistentClient(path=persist_directory))
                else:
                    api_kwargs = {}
            if not api_kwargs:
                client_settings = Settings(anonymized_telemetry=False,
                                           **chroma_settings,
                                           persist_directory=persist_directory)
                api_kwargs = dict(client_settings=client_settings)
            db = chroma_class(persist_directory=persist_directory, embedding_function=embedding,
                              collection_name=langchain_mode.replace(' ', '_'),
                              **api_kwargs)
            try:
                with get_context_cast():
                    db.similarity_search('')
            except BaseException as e:
                # migration when no embed_info
                if 'Dimensionality of (768) does not match index dimensionality (384)' in str(e) or \
                        'Embedding dimension 768 does not match collection dimensionality 384' in str(e) or \
                        'Embedding dimension 768 does not match collection dimensionality 1536' in str(e) or \
                        'Dimensionality of (1536) does not match index dimensionality (384)' in str(e):
                    hf_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
                    embedding = get_embedding(use_openai_embedding, hf_embedding_model=hf_embedding_model)
                    db = chroma_class(persist_directory=persist_directory, embedding_function=embedding,
                                      collection_name=langchain_mode.replace(' ', '_'),
                                      **api_kwargs)
                    # should work now, let fail if not
                    with get_context_cast():
                        db.similarity_search('')
                    save_embed(db, use_openai_embedding, hf_embedding_model)
                else:
                    raise

            if verbose:
                print("DONE Loading db: %s" % langchain_mode, flush=True)
        else:
            if not migrate_embedding_model:
                # OVERRIDE embedding choices if could load embedding info when not migrating
                got_embedding, use_openai_embedding, hf_embedding_model = load_embed(db=db,
                                                                                     use_openai_embedding=use_openai_embedding)
            if verbose:
                print("USING already-loaded db: %s" % langchain_mode, flush=True)
        if check_embedding:
            db_trial, changed_db = check_update_chroma_embedding(db,
                                                                 db_type,
                                                                 use_openai_embedding,
                                                                 hf_embedding_model,
                                                                 migrate_embedding_model,
                                                                 langchain_mode,
                                                                 langchain_mode_paths,
                                                                 langchain_mode_types,
                                                                 n_jobs=n_jobs,
                                                                 verbose=verbose)
            if changed_db:
                db = db_trial
                # only call persist if really changed db, else takes too long for large db
                if db is not None:
                    db.persist()
                    clear_embedding(db)
        save_embed(db, use_openai_embedding, hf_embedding_model)
        if migrate_meta:
            db_trial, changed_db = migrate_meta_func(db, langchain_mode)
            if changed_db:
                db = db_trial
        return db, use_openai_embedding, hf_embedding_model
    return db, use_openai_embedding, hf_embedding_model


def clear_embedding(db):
    if db is None:
        return
    # don't keep on GPU, wastes memory, push back onto CPU and only put back on GPU once again embed
    try:
        if hasattr(db._embedding_function, 'client') and hasattr(db._embedding_function.client, 'cpu'):
            # only push back to CPU if each db/user has own embedding model, else if shared share on GPU
            if hasattr(db._embedding_function.client, 'preload') and not db._embedding_function.client.preload:
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


embed_lock_name = 'embed.lock'


def get_embed_lock_file(db, persist_directory=None):
    if hasattr(db, '_persist_directory') or persist_directory:
        if persist_directory is None:
            persist_directory = db._persist_directory
        check_persist_directory(persist_directory)
        base_path = os.path.join('locks', persist_directory)
        base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
        lock_file = os.path.join(base_path, embed_lock_name)
        makedirs(os.path.dirname(lock_file))
        return lock_file
    return None


def save_embed(db, use_openai_embedding, hf_embedding_model):
    if hasattr(db, '_persist_directory'):
        persist_directory = db._persist_directory
        lock_file = get_embed_lock_file(db)
        with filelock.FileLock(lock_file):
            embed_info_file = os.path.join(persist_directory, 'embed_info')
            with open(embed_info_file, 'wb') as f:
                if isinstance(hf_embedding_model, str):
                    hf_embedding_model_save = hf_embedding_model
                elif hasattr(hf_embedding_model, 'model_name'):
                    hf_embedding_model_save = hf_embedding_model.model_name
                elif isinstance(hf_embedding_model, dict) and 'name' in hf_embedding_model:
                    hf_embedding_model_save = hf_embedding_model['name']
                elif isinstance(hf_embedding_model, dict) and 'name' in hf_embedding_model:
                    if os.getenv('HARD_ASSERTS'):
                        # unexpected in testing or normally
                        raise RuntimeError("HERE")
                    hf_embedding_model_save = 'BAAI/bge-large-en-v1.5'
                pickle.dump((use_openai_embedding, hf_embedding_model_save), f)
    return use_openai_embedding, hf_embedding_model


def load_embed(db=None, persist_directory=None, use_openai_embedding=False):
    if hasattr(db, 'embeddings') and hasattr(db.embeddings, 'model_name'):
        hf_embedding_model = db.embeddings.model_name if 'openai' not in db.embeddings.model_name.lower() else None
        use_openai_embedding = hf_embedding_model is None
        save_embed(db, use_openai_embedding, hf_embedding_model)
        return True, use_openai_embedding, hf_embedding_model
    if persist_directory is None:
        persist_directory = db._persist_directory
    embed_info_file = os.path.join(persist_directory, 'embed_info')
    if os.path.isfile(embed_info_file):
        lock_file = get_embed_lock_file(db, persist_directory=persist_directory)
        with filelock.FileLock(lock_file):
            with open(embed_info_file, 'rb') as f:
                try:
                    use_openai_embedding, hf_embedding_model = pickle.load(f)
                    if not isinstance(hf_embedding_model, str):
                        # work-around bug introduced here: https://github.com/h2oai/h2ogpt/commit/54c4414f1ce3b5b7c938def651c0f6af081c66de
                        hf_embedding_model = 'BAAI/bge-large-en-v1.5'
                        # fix file
                        save_embed(db, use_openai_embedding, hf_embedding_model)
                    got_embedding = True
                except EOFError:
                    use_openai_embedding, hf_embedding_model = False, 'BAAI/bge-large-en-v1.5'
                    got_embedding = False
                    if os.getenv('HARD_ASSERTS'):
                        # unexpected in testing or normally
                        raise
    else:
        # migration or not set yet, assume defaults
        if use_openai_embedding:
            use_openai_embedding, hf_embedding_model = True, ''
            got_embedding = False
        else:
            use_openai_embedding, hf_embedding_model = False, "sentence-transformers/all-MiniLM-L6-v2"
            got_embedding = False
    assert isinstance(hf_embedding_model, str)
    return got_embedding, use_openai_embedding, hf_embedding_model


def sanitize_path_segment(segment):
    # Replace invalid Windows filename characters with an underscore
    return re.sub(r'[<>:"/\\|?*]', '_', segment)


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
    makedirs(user_base_dir)
    user_base_dir = sanitize_path_segment(user_base_dir)
    dirid = sanitize_path_segment(dirid)
    persist_directory = os.path.join(user_base_dir, dirid, 'db_dir_%s' % langchain_mode)
    if userid and \
            (os.path.isdir(persist_directory) or
             db1s is not None and langchain_mode in db1s or
             langchain_type == LangChainTypes.PERSONAL.value):
        langchain_type = LangChainTypes.PERSONAL.value
        if userid:
            # then maybe logged in after added docs as non-logged-in user, try to preserve
            persist_directory0 = os.path.join(user_base_dir, userid, 'db_dir_%s' % langchain_mode)
            if userid != dirid and os.path.isdir(persist_directory0):
                # link new directory instead of making new directory
                try:
                    persist_directory1 = os.path.join(user_base_dir, dirid, 'db_dir_%s' % langchain_mode)
                    create_relative_symlink(persist_directory0, persist_directory1)
                except Exception as e:
                    print("Failed to soft link: %s %s :%s" % (userid, dirid, str(e)), flush=True)
                    persist_directory = makedirs(persist_directory, use_base=True)
                assert os.path.isdir(persist_directory)
            else:
                persist_directory = makedirs(persist_directory, use_base=True)
        else:
            persist_directory = makedirs(persist_directory, use_base=True)
        check_persist_directory(persist_directory)
        return persist_directory, langchain_type

    persist_directory = 'db_dir_%s' % langchain_mode
    if (os.path.isdir(persist_directory) or
            dbs is not None and langchain_mode in dbs or
            langchain_type == LangChainTypes.SHARED.value):
        # ensure consistent
        langchain_type = LangChainTypes.SHARED.value
        persist_directory = makedirs(persist_directory, use_base=True)
        check_persist_directory(persist_directory)
        return persist_directory, langchain_type

    # dummy return for prep_langchain() or full personal space
    base_others = 'db_nonusers'
    persist_directory = os.path.join(base_others, 'db_dir_%s' % str(uuid.uuid4()))
    persist_directory = makedirs(persist_directory, use_base=True)
    langchain_type = LangChainTypes.PERSONAL.value

    check_persist_directory(persist_directory)
    return persist_directory, langchain_type


def check_persist_directory(persist_directory):
    # deal with some cases when see intrinsic names being used as shared
    for langchain_mode in langchain_modes_intrinsic:
        if persist_directory == 'db_dir_%s' % langchain_mode:
            raise RuntimeError("Illegal access to %s" % persist_directory)


def _make_db(use_openai_embedding=False,
             hf_embedding_model=None,
             migrate_embedding_model=False,
             first_para=False, text_limit=None,
             chunk=True, chunk_size=512,

             # urls
             use_unstructured=True,
             use_playwright=False,
             use_selenium=False,
             use_scrapeplaywright=False,
             use_scrapehttp=False,

             # pdfs
             use_pymupdf='auto',
             use_unstructured_pdf='auto',
             use_pypdf='auto',
             enable_pdf_ocr='auto',
             enable_pdf_doctr='auto',
             try_pdf_as_html='auto',

             # images
             enable_ocr=False,
             enable_doctr=False,
             enable_pix2struct=False,
             enable_captions=True,
             enable_llava=True,
             enable_transcriptions=True,
             captions_model=None,
             caption_loader=None,
             llava_model=None,
             llava_prompt=None,
             doctr_loader=None,
             pix2struct_loader=None,
             asr_model=None,
             asr_loader=None,

             # json
             jq_schema='.[]',
             # video
             extract_frames=10,

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
                        hf_embedding_model, migrate_embedding_model, verbose=verbose,
                        n_jobs=n_jobs)
    if db_trial is not None:
        db = db_trial

    sources = []
    if db is None:
        chunk_sources = functools.partial(_chunk_sources, chunk=chunk, chunk_size=chunk_size, db_type=db_type,
                                          hf_embedding_model=hf_embedding_model,
                                          use_openai_embedding=use_openai_embedding, verbose=verbose)
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
            # full scan below, but only at start-up or when adding files from disk in UI, will be slow for large dbs
            # FIXME: Could have option to just add, not delete old ones
            existing_hash_ids = get_existing_hash_ids(db)
        else:
            # pretend no existing files so won't filter
            existing_files = []
            existing_hash_ids = []
        # chunk internally for speed over multiple docs
        # FIXME: If first had old Hash=None and switch embeddings,
        #  then re-embed, and then hit here and reload so have hash, and then re-embed.
        sources1 = path_to_docs(user_path, n_jobs=n_jobs, chunk=chunk, chunk_size=chunk_size,
                                # urls
                                use_unstructured=use_unstructured,
                                use_playwright=use_playwright,
                                use_selenium=use_selenium,
                                use_scrapeplaywright=use_scrapeplaywright,
                                use_scrapehttp=use_scrapehttp,

                                # pdfs
                                use_pymupdf=use_pymupdf,
                                use_unstructured_pdf=use_unstructured_pdf,
                                use_pypdf=use_pypdf,
                                enable_pdf_ocr=enable_pdf_ocr,
                                enable_pdf_doctr=enable_pdf_doctr,
                                try_pdf_as_html=try_pdf_as_html,

                                # images
                                enable_ocr=enable_ocr,
                                enable_doctr=enable_doctr,
                                enable_pix2struct=enable_pix2struct,
                                enable_captions=enable_captions,
                                enable_llava=enable_llava,
                                enable_transcriptions=enable_transcriptions,
                                captions_model=captions_model,
                                caption_loader=caption_loader,
                                llava_model=llava_model,
                                llava_prompt=llava_prompt,
                                doctr_loader=doctr_loader,
                                pix2struct_loader=pix2struct_loader,
                                asr_model=asr_model,
                                asr_loader=asr_loader,

                                # json
                                jq_schema=jq_schema,
                                extract_frames=extract_frames,

                                existing_files=existing_files, existing_hash_ids=existing_hash_ids,
                                db_type=db_type,

                                is_public=False,
                                from_ui=True,

                                hf_embedding_model=hf_embedding_model,
                                use_openai_embedding=use_openai_embedding,
                                )
        new_metadata_sources = set([x.metadata['source'] for x in sources1])
        if new_metadata_sources:
            new_metadata_sources_real = [x for x in new_metadata_sources if
                                         'rotated' not in x and 'pad_resized' not in x]
            if os.getenv('NO_NEW_FILES') is not None and new_metadata_sources_real:
                raise RuntimeError("Expected no new files1! %s" % new_metadata_sources_real)
            print("Loaded %s new files as sources to add to %s" % (len(new_metadata_sources), langchain_mode),
                  flush=True)
            if verbose:
                print("Files added: %s" % '\n'.join(new_metadata_sources), flush=True)
        sources.extend(sources1)
        if len(sources) > 0 and os.getenv('NO_NEW_FILES') is not None:
            raise RuntimeError("Expected no new files2! %s" % langchain_mode)
        if len(sources) == 0 and os.getenv('SHOULD_NEW_FILES') is not None:
            raise RuntimeError("Expected new files! %s" % langchain_mode)
        if verbose:
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
    if db is None:
        if sources:
            db = get_db(sources, use_openai_embedding=use_openai_embedding, db_type=db_type,
                        persist_directory=persist_directory,
                        langchain_mode=langchain_mode,
                        langchain_mode_paths=langchain_mode_paths,
                        langchain_mode_types=langchain_mode_types,
                        hf_embedding_model=hf_embedding_model,
                        migrate_embedding_model=migrate_embedding_model,
                        n_jobs=n_jobs,
                        verbose=verbose)
            if verbose:
                print("Generated db", flush=True)
        elif langchain_mode not in langchain_modes_intrinsic:
            print("Did not generate db for %s since no sources" % langchain_mode, flush=True)
        new_sources_metadata = [x.metadata for x in sources]
    elif user_path is not None:
        print("Existing db, potentially adding %s sources from user_path=%s" % (len(sources), user_path), flush=True)
        db, num_new_sources, new_sources_metadata = add_to_db(db, sources, db_type=db_type,
                                                              use_openai_embedding=use_openai_embedding,
                                                              hf_embedding_model=hf_embedding_model,
                                                              verbose=verbose)
        print("Existing db, added %s new sources from user_path=%s" % (num_new_sources, user_path), flush=True)
    else:
        new_sources_metadata = [x.metadata for x in sources]

    return db, len(new_sources_metadata), new_sources_metadata


def is_chroma_db(db):
    return isinstance(db, Chroma)


def is_new_chroma_db(db):
    if isinstance(db, Chroma):
        return True
    else:
        raise RuntimeError("Migration no longer supported")


def sim_search(db, query='', k=1000, with_score=False, filter_kwargs=None, chunk_id_filter=None,
               where_document_dict={},
               verbose=False):
    if is_chroma_db(db) and large_chroma_db(db) and chunk_id_filter is not None:
        # try to avoid filter if just doing chunk_id -1 or >= 0
        docs = _sim_search(db, query=query, k=k * 4, with_score=with_score, verbose=verbose)
        if with_score:
            if chunk_id_filter >= 0:
                docs = [x for x in docs if x[0].metadata.get('chunk_id', chunk_id_filter) >= chunk_id_filter]
            else:
                docs = [x for x in docs if x[0].metadata.get('chunk_id', chunk_id_filter) == chunk_id_filter]
        else:
            if chunk_id_filter >= 0:
                docs = [x for x in docs if x.metadata.get('chunk_id', chunk_id_filter) >= chunk_id_filter]
            else:
                docs = [x for x in docs if x.metadata.get('chunk_id', chunk_id_filter) == chunk_id_filter]
        if len(docs) < max(1, k // 4):
            # full search if failed to find enough
            docs = _sim_search(db, query=query, k=k, with_score=with_score, filter_kwargs=filter_kwargs,
                               where_document_dict=where_document_dict,
                               verbose=verbose)
        return docs
    else:
        return _sim_search(db, query=query, k=k, with_score=with_score, filter_kwargs=filter_kwargs,
                           where_document_dict=where_document_dict,
                           verbose=verbose)


def _sim_search(db, query='', k=1000, with_score=False, filter_kwargs=None,
                where_document_dict={},
                verbose=False):
    if k == -1:
        k = 1000
    if filter_kwargs is None:
        filter_kwargs = {}
    docs = []
    # avoid lock if fake embeddings or faiss etc., since no complex db
    lock_file = get_db_lock_file(db)
    lock_func = filelock.FileLock if hasattr(db, '_persist_directory') else NullContext
    # have query
    # for db=None too
    while True:
        try:
            if with_score:
                with get_context_cast():
                    with lock_func(lock_file):
                        docs = db.similarity_search_with_score(query, k=k, **filter_kwargs, **where_document_dict)
            else:
                with get_context_cast():
                    with lock_func(lock_file):
                        docs = db.similarity_search(query, k=k, **filter_kwargs, **where_document_dict)
            break
        except (RuntimeError, AttributeError) as e:
            # AttributeError is for people with wrong version of langchain
            if verbose:
                print("chroma bug: %s" % str(e), flush=True)
            if k == 1:
                raise
            if k > 500:
                k -= 200
            elif k > 100:
                k -= 50
            elif k > 10:
                k -= 5
            else:
                k -= 1
            k = max(1, k)
    return docs


def large_chroma_db(db):
    return get_size(db._persist_directory) >= 500 * 1024 ** 2


def get_metadatas(db, full_required=True, k_max=10000):
    lock_file = get_db_lock_file(db)
    lock_func = filelock.FileLock if hasattr(db, '_persist_directory') else NullContext

    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores import Qdrant
    if isinstance(db, FAISS):
        with lock_func(lock_file):
            metadatas = [v.metadata for k, v in db.docstore._dict.items()]
    elif is_chroma_db(db):
        if full_required or not (large_chroma_db(db) and is_new_chroma_db(db)):
            db_get = get_documents(db)
            documents = db_get['documents']
            if documents is None:
                documents = []
            metadatas = db_get['metadatas']
            if metadatas is None:
                if documents is not None:
                    metadatas = [{}] * len(documents)
                else:
                    metadatas = []
        else:
            # just use sim search, since too many
            docs1 = sim_search(db, k=k_max, with_score=False)
            metadatas = [x.metadata for x in docs1]
    elif isinstance(db, Qdrant):
        with lock_func(lock_file):
            points, _ = db.client.scroll(db.collection_name, limit=k_max, with_payload=True)
            metadatas = [point.payload["metadata"] for point in points]
    elif db is not None:
        # FIXME: Hack due to https://github.com/weaviate/weaviate/issues/1947
        # seems no way to get all metadata, so need to avoid this approach for weaviate
        with lock_func(lock_file):
            with get_context_cast():
                metadatas = [x.metadata for x in db.similarity_search("", k=k_max)]
    else:
        metadatas = []
    return metadatas


def get_db_lock_file(db, lock_type='getdb'):
    if hasattr(db, '_persist_directory'):
        persist_directory = db._persist_directory
        check_persist_directory(persist_directory)
        base_path = os.path.join('locks', persist_directory)
        base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
        lock_file = os.path.join(base_path, "%s.lock" % lock_type)
        makedirs(os.path.dirname(lock_file))  # ensure made
        return lock_file
    return None


def get_documents(db):
    lock_file = get_db_lock_file(db)
    lock_func = filelock.FileLock if hasattr(db, '_persist_directory') else NullContext

    # returns not just documents, but full dict of documents, metadatas, ids, embeddings
    # documents['documents] should be list of texts, not Document() type
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores import Qdrant
    if isinstance(db, FAISS):
        with lock_func(lock_file):
            documents = [v for k, v in db.docstore._dict.items()]
        documents = dict(documents=documents, metadatas=[{}] * len(documents), ids=[0] * len(documents))
    elif isinstance(db, Chroma):
        with lock_func(lock_file):
            documents = db.get()
        if documents is None:
            documents = dict(documents=[], metadatas=[], ids=[])
    elif isinstance(db, Qdrant):
        with lock_func(lock_file):
            points, next_id = db.client.scroll(db.collection_name, limit=10000, with_payload=True)
            documents, metadatas = [], []
            for point in points:
                documents.append(point.payload["page_content"])
                metadatas.append(point.payload["metadata"])
        documents = dict(documents=documents, metadatas=metadatas, ids=[0] * len(documents))
    else:
        # FIXME: Hack due to https://github.com/weaviate/weaviate/issues/1947
        # seems no way to get all metadata, so need to avoid this approach for weaviate
        with lock_func(lock_file):
            with get_context_cast():
                docs_from_search = [x for x in db.similarity_search("", k=10000)]
        # Don't filter out by content etc. here, might use get_metadatas too separately
        documents = [x.page_content for x in docs_from_search]
        metadatas = [x.metadata for x in docs_from_search]
        documents = dict(documents=documents, metadatas=metadatas, ids=[0] * len(documents))
    return documents


def get_docs_and_meta(db, top_k_docs, filter_kwargs={}, text_context_list=None, chunk_id_filter=None, k_max=1000):
    # db_documents should be list of texts
    # db_metadatas should be list of dicts
    db_documents = []
    db_metadatas = []

    if text_context_list:
        db_documents += [x.page_content if hasattr(x, 'page_content') else x for x in text_context_list]
        db_metadatas += [x.metadata if hasattr(x, 'metadata') else {} for x in text_context_list]

    lock_file = get_db_lock_file(db)
    lock_func = filelock.FileLock if hasattr(db, '_persist_directory') else NullContext

    from langchain_community.vectorstores import FAISS
    if isinstance(db, Chroma):
        if top_k_docs == -1:
            limit = k_max
        else:
            limit = max(top_k_docs, k_max)
        with lock_func(lock_file):
            db_get = db._collection.get(where=filter_kwargs.get('filter'), limit=limit)
        db_metadatas += db_get['metadatas']
        db_documents += db_get['documents']
    elif isinstance(db, FAISS):
        import itertools
        db_metadatas += get_metadatas(db)
        # FIXME: FAISS has no filter
        with lock_func(lock_file):
            if top_k_docs == -1:
                db_docs_faiss = list(db.docstore._dict.values())
            else:
                # slice dict first
                db_docs_faiss = list(dict(itertools.islice(db.docstore._dict.items(), top_k_docs)).values())
        db_docs_faiss = [x.page_content for x in db_docs_faiss]
        db_documents += db_docs_faiss
    elif db is not None:
        db_metadatas += get_metadatas(db)
        db_documents += get_documents(db)['documents']

    return db_documents, db_metadatas


def get_existing_files(db):
    # Note: Below full scan if used, but this function not used yet
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
    kwargs['sources_show_text_in_accordion'] = kwargs.get('sources_show_text_in_accordion', True)
    kwargs['hyde_show_intermediate_in_accordion'] = kwargs.get('hyde_show_intermediate_in_accordion', True)
    kwargs['map_reduce_show_intermediate_in_accordion'] = kwargs.get('map_reduce_show_intermediate_in_accordion', True)
    kwargs['show_link_in_sources'] = kwargs.get('show_link_in_sources', True)
    kwargs['top_k_docs_max_show'] = kwargs.get('top_k_docs_max_show', 10)
    kwargs['llamacpp_dict'] = {}  # shouldn't be required unless from test using _run_qa_db
    kwargs['exllama_dict'] = {}  # shouldn't be required unless from test using _run_qa_db
    kwargs['gptq_dict'] = {}  # shouldn't be required unless from test using _run_qa_db
    kwargs['sink_dict'] = {}  # shouldn't be required unless from test using _run_qa_db
    kwargs['hf_model_dict'] = {}  # shouldn't be required unless from test using _run_qa_db
    kwargs['force_seq2seq_type'] = False  # shouldn't be required unless from test using _run_qa_db
    kwargs['force_t5_type'] = False  # shouldn't be required unless from test using _run_qa_db
    kwargs['image_file'] = kwargs.get('image_file')
    kwargs['image_control'] = kwargs.get('image_control')
    kwargs['images_num_max'] = kwargs.get('images_num_max')
    kwargs['image_resolution'] = kwargs.get('image_resolution')
    kwargs['image_format'] = kwargs.get('image_format')
    kwargs['video_frame_period'] = kwargs.get('video_frame_period')
    kwargs['load_awq'] = kwargs.get('load_awq', '')

    kwargs['response_format'] = kwargs.get('response_format', 'text')
    kwargs['guided_json'] = kwargs.get('guided_json', None)
    kwargs['guided_regex'] = kwargs.get('guided_regex', None)
    kwargs['guided_choice'] = kwargs.get('guided_choice', None)
    kwargs['guided_grammar'] = kwargs.get('guided_grammar', None)
    kwargs['guided_whitespace_pattern'] = kwargs.get('guided_whitespace_pattern', None)
    kwargs['json_vllm'] = kwargs.get('json_vllm', False)

    kwargs['from_ui'] = kwargs.get('from_ui', True)
    kwargs['stream_map'] = kwargs.get('stream_map', False)

    kwargs['is_vision_model1'] = kwargs.get('is_vision_model1', False)
    kwargs['is_actually_vision_model1'] = kwargs.get('is_actually_vision_model1', False)

    missing_kwargs = [x for x in func_names if x not in kwargs]
    assert not missing_kwargs, "Missing kwargs for run_qa_db: %s" % missing_kwargs
    # only keep actual used
    kwargs = {k: v for k, v in kwargs.items() if k in func_names}
    try:
        return _run_qa_db(**kwargs)
    finally:
        if kwargs.get('cli', False):
            clear_torch_cache(allow_skip=True)


def _run_qa_db(query=None,
               iinput=None,
               context=None,
               use_openai_model=False, use_openai_embedding=False,
               first_para=False, text_limit=None, top_k_docs=4, chunk=True, chunk_size=512,
               langchain_instruct_mode=True,

               # urls
               use_unstructured=True,
               use_playwright=False,
               use_selenium=False,
               use_scrapeplaywright=False,
               use_scrapehttp=False,

               # pdfs
               use_pymupdf='auto',
               use_unstructured_pdf='auto',
               use_pypdf='auto',
               enable_pdf_ocr='auto',
               enable_pdf_doctr='auto',
               try_pdf_as_html='auto',

               # images
               enable_ocr=False,
               enable_doctr=False,
               enable_pix2struct=False,
               enable_captions=True,
               enable_llava=True,
               enable_transcriptions=True,
               captions_model=None,
               caption_loader=None,
               llava_model=None,
               llava_prompt=None,
               doctr_loader=None,
               pix2struct_loader=None,
               asr_model=None,
               asr_loader=None,

               # json
               jq_schema='.[]',
               extract_frames=10,

               langchain_mode_paths={},
               langchain_mode_types={},
               detect_user_path_changes_every_query=False,
               db_type=None,
               model_name=None, model=None, tokenizer=None, inference_server=None,
               langchain_only_model=False,
               load_awq='',
               hf_embedding_model=None,
               migrate_embedding_model=False,
               stream_output0=False,
               stream_output=False,
               enable_caching=False,
               async_output=True,
               num_async=3,
               prompter=None,
               prompt_type=None,
               prompt_dict=None,
               chat_template=None,
               answer_with_sources=True,
               append_sources_to_answer=False,
               append_sources_to_chat=True,
               cut_distance=1.64,
               add_chat_history_to_context=True,
               add_search_to_context=False,
               keep_sources_in_context=False,
               gradio_errors_to_chatbot=True,
               memory_restriction_level=0,
               system_prompt='',
               allow_chat_system_prompt=True,
               sanitize_bot_response=False,
               show_rank=False,
               sources_show_text_in_accordion=True,
               hyde_show_intermediate_in_accordion=True,
               map_reduce_show_intermediate_in_accordion=True,
               show_link_in_sources=True,
               top_k_docs_max_show=10,
               use_llm_if_no_docs=True,
               load_db_if_exists=False,
               db=None,
               do_sample=False,
               seed=0,
               temperature=0.1,
               top_p=0.7,
               top_k=40,
               penalty_alpha=0.0,
               num_beams=1,
               max_new_tokens=512,
               min_new_tokens=1,
               attention_sinks=False,
               sink_dict={},
               truncation_generation=False,
               early_stopping=False,
               regenerate_clients=None,
               regenerate_gradio_clients=None,
               validate_clients=None,
               fail_if_invalid_client=None,
               max_time=180,
               repetition_penalty=1.0,
               num_return_sequences=1,
               langchain_mode=None,
               langchain_action=None,
               langchain_agents=None,
               document_subset=DocumentSubset.Relevant.name,
               document_choice=[DocumentChoice.ALL.value],
               document_source_substrings=[],
               document_source_substrings_op='and',
               document_content_substrings=[],
               document_content_substrings_op='and',
               pre_prompt_query=None,
               prompt_query=None,
               pre_prompt_summary=None,
               prompt_summary=None,
               hyde_llm_prompt=None,
               all_docs_start_prompt=None,
               all_docs_finish_prompt=None,
               text_context_list=None,
               chat_conversation=None,

               user_prompt_for_fake_system_prompt=None,
               json_object_prompt=None,
               json_object_prompt_simpler=None,
               json_code_prompt=None,
               json_code_prompt_if_no_schema=None,
               json_schema_instruction=None,
               json_preserve_system_prompt=None,
               json_object_post_prompt_reminder=None,
               json_code_post_prompt_reminder=None,
               json_code2_post_prompt_reminder=None,

               visible_models=None,
               h2ogpt_key=None,
               docs_ordering_type=docs_ordering_types_default,
               min_max_new_tokens=512,
               max_input_tokens=-1,
               max_total_input_tokens=-1,
               docs_token_handling=None,
               docs_joiner=docs_joiner_default,
               hyde_level=0,
               hyde_template=None,
               hyde_show_only_final=None,
               doc_json_mode=False,
               metadata_in_context=[],

               n_jobs=-1,
               llamacpp_path=None,
               llamacpp_dict=None,
               exllama_dict=None,
               verbose=False,
               cli=False,
               lora_weights='',

               auto_reduce_chunks=True,
               max_chunks=100,
               headsize=50,

               image_file=None,
               image_control=None,
               images_num_max=None,
               image_resolution=None,
               image_format=None,
               rotate_align_resize_image=None,
               video_frame_period=None,
               image_batch_image_prompt=None,
               image_batch_final_prompt=None,
               image_batch_stream=None,
               visible_vision_models=None,
               video_file=None,

               response_format=None,
               guided_json=None,
               guided_regex=None,
               guided_choice=None,
               guided_grammar=None,
               guided_whitespace_pattern=None,
               client_metadata=None,

               json_vllm=False,

               from_ui=True,
               stream_map=False,

               is_vision_model1=False,
               is_actually_vision_model1=False,
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
    :param db_type: 'faiss' for in-memory
                    'chroma' (for chroma >= 0.4)
                    'chroma_old' (for chroma < 0.4)
                    'weaviate' for persisted on disk
                    'qdrant' for a Qdrant server or an in-memory instance
    :param model_name: model name, used to switch behaviors
    :param model: pre-initialized model, else will make new one
    :param tokenizer: pre-initialized tokenizer, else will make new one.  Required not None if model is not None
    :param answer_with_sources
    :return:
    """
    if client_metadata:
        print("RUNQADB START client_metadata: %s" % client_metadata, flush=True)
    t_run = time.time()
    if LangChainAgent.SMART.value in langchain_agents:
        # FIXME: support whatever model/user supports
        # right now doesn't support, just hangs for some reason
        async_output = False
    elif langchain_action in [LangChainAction.QUERY.value]:
        # only summarization supported
        async_output = False
    elif LangChainAgent.AUTOGPT.value in langchain_agents:
        async_output = False
    else:
        if stream_output0:
            # threads and asyncio don't mix
            # but if do asyncio inside thread, all fine
            # async_output = True
            pass
        else:
            # go back to not streaming for summarization/extraction to be parallel
            stream_output = stream_output0

    # avoid source stuff in response if not textual, e.g. json
    # also doesn't make much sense to get accordion stuff from API
    if response_format != 'text' or not from_ui:
        append_sources_to_answer = False
        hyde_show_intermediate_in_accordion = False
        map_reduce_show_intermediate_in_accordion = False

    # in case doing summarization/extraction, and docs originally limit, relax if each document or reduced response is smaller than max document size
    max_new_tokens0 = max_new_tokens

    # in case None, e.g. lazy client, then set based upon actual model
    pre_prompt_query, prompt_query, pre_prompt_summary, prompt_summary, hyde_llm_prompt = \
        get_langchain_prompts(pre_prompt_query, prompt_query,
                              pre_prompt_summary, prompt_summary, hyde_llm_prompt,
                              )

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

    query_action = langchain_action == LangChainAction.QUERY.value
    summarize_action = langchain_action in [LangChainAction.SUMMARIZE_MAP.value,
                                            LangChainAction.SUMMARIZE_ALL.value,
                                            LangChainAction.SUMMARIZE_REFINE.value,
                                            LangChainAction.EXTRACT.value]

    zero_shot_react_agent = any([x in langchain_agents for x in
                                 [LangChainAgent.SEARCH.value,
                                  LangChainAgent.CSV.value,
                                  LangChainAgent.PANDAS.value,
                                  ]]) and \
                            not does_support_functiontools(inference_server, model_name)
    if zero_shot_react_agent:
        if LangChainAgent.SEARCH.value in langchain_agents:
            answer_type = " bullet list"
        else:
            answer_type = ""
        system_prompt = """You are a zero shot react agent.
Consider to prompt of Question that was original query from the user.  Do not repeat "Question" as a prompt, that is only for the user.
Respond to prompt of Thought with a thought that may lead to a reasonable new action choice.
Respond to prompt of Action with an action to take out of the tools given, giving exactly single word for the tool name.
Respond to prompt of Action Input with an input to give the tool.
Consider to prompt of Observation that was response from the tool.
Repeat this Thought, Action, Action Input, Observation, Thought sequence several times with new and different thoughts and actions each time, do not repeat.
Once satisfied that the thoughts, responses are sufficient to answer the question, then respond to prompt of Thought with: I now know the final answer
Respond to prompt of Final Answer with your final well-structured%s answer to the original query.
""" % answer_type
        prompter.system_prompt = system_prompt

    if doc_json_mode:
        prompter.system_prompt = system_prompt = doc_json_mode_system_prompt

    doing_grounding = tokenizer is not None and \
                      hasattr(tokenizer, 'apply_grounded_generation_template') and \
                      prompt_type not in [empty_prompt_type, noop_prompt_type, template_prompt_type]

    # handle auto case
    if system_prompt == 'auto':
        changed = False
        if query_action and langchain_mode not in langchain_modes_non_db:
            system_prompt = system_docqa
            changed = True
        elif summarize_action:
            system_prompt = system_summary
            changed = True
        if changed and prompter:
            prompter.system_prompt = system_prompt
        if system_prompt == 'auto':
            if prompter:
                system_prompt = prompter.system_prompt
            if system_prompt == 'auto':
                # safest then to just avoid system prompt
                system_prompt = prompter.system_prompt = ""

    assert len(set(gen_hyper).difference(inspect.signature(get_llm).parameters)) == 0
    # pass in context to LLM directly, since already has prompt_type structure
    # can't pass through langchain in get_chain() to LLM: https://github.com/hwchase17/langchain/issues/6638
    llm_kwargs = dict(use_openai_model=use_openai_model, model_name=model_name,
                      model=model,
                      tokenizer=tokenizer,
                      inference_server=inference_server,
                      langchain_only_model=langchain_only_model,
                      load_awq=load_awq,
                      stream_output=stream_output,
                      enable_caching=enable_caching,
                      async_output=async_output,
                      num_async=num_async,
                      do_sample=do_sample,
                      seed=seed,
                      temperature=temperature,
                      top_k=top_k,
                      top_p=top_p,
                      penalty_alpha=penalty_alpha,
                      num_beams=num_beams,
                      max_new_tokens=max_new_tokens,
                      max_new_tokens0=max_new_tokens0,
                      min_new_tokens=min_new_tokens,
                      early_stopping=early_stopping,
                      max_time=max_time,
                      regenerate_clients=regenerate_clients,
                      regenerate_gradio_clients=regenerate_gradio_clients,
                      validate_clients=validate_clients,
                      fail_if_invalid_client=fail_if_invalid_client,
                      repetition_penalty=repetition_penalty,
                      num_return_sequences=num_return_sequences,
                      prompt_type=prompt_type,
                      prompt_dict=prompt_dict,
                      chat_template=chat_template,
                      prompter=prompter,
                      context=context,
                      iinput=iinput,
                      sanitize_bot_response=sanitize_bot_response,

                      user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt,
                      json_object_prompt=json_object_prompt,
                      json_object_prompt_simpler=json_object_prompt_simpler,
                      json_code_prompt=json_code_prompt,
                      json_code_prompt_if_no_schema=json_code_prompt_if_no_schema,
                      json_schema_instruction=json_schema_instruction,

                      system_prompt=system_prompt,
                      chat_conversation=chat_conversation,
                      add_chat_history_to_context=add_chat_history_to_context,
                      visible_models=visible_models,
                      h2ogpt_key=h2ogpt_key,
                      min_max_new_tokens=min_max_new_tokens,
                      max_input_tokens=max_input_tokens,
                      max_total_input_tokens=max_total_input_tokens,
                      n_jobs=n_jobs,
                      llamacpp_path=llamacpp_path,
                      llamacpp_dict=llamacpp_dict,
                      exllama_dict=exllama_dict,
                      cli=cli,
                      verbose=verbose,
                      attention_sinks=attention_sinks,
                      sink_dict=sink_dict,
                      truncation_generation=truncation_generation,
                      langchain_agents=langchain_agents,

                      image_file=image_file,
                      image_control=image_control,
                      images_num_max=images_num_max,
                      image_resolution=image_resolution,
                      image_format=image_format,
                      rotate_align_resize_image=rotate_align_resize_image,
                      video_frame_period=video_frame_period,
                      image_batch_image_prompt=image_batch_image_prompt,
                      image_batch_final_prompt=image_batch_final_prompt,
                      image_batch_stream=image_batch_stream,
                      visible_vision_models=visible_vision_models,
                      video_file=video_file,

                      document_choice=document_choice,

                      response_format=response_format,
                      guided_json=guided_json,
                      guided_regex=guided_regex,
                      guided_choice=guided_choice,
                      guided_grammar=guided_grammar,
                      guided_whitespace_pattern=guided_whitespace_pattern,

                      doing_grounding=doing_grounding,
                      json_vllm=json_vllm,

                      query_action=query_action,
                      summarize_action=summarize_action,
                      stream_map=stream_map,

                      is_vision_model1=is_vision_model1,
                      is_actually_vision_model1=is_actually_vision_model1,
                      )
    llm, model_name, streamer, prompt_type_out, async_output, only_new_text, gradio_server = \
        get_llm(**llm_kwargs)
    if LangChainAgent.SMART.value in langchain_agents:
        # get llm for exploration
        llm_kwargs_explore = llm_kwargs.copy()
        llm_kwargs_explore.update(dict(do_sample=True, temperature=0.5, seed=randint(0, 32000)))
        llm_explore, _, _, _, _, _, _ = get_llm(**llm_kwargs_explore)
    else:
        llm_explore = None

    # in case change, override original prompter
    if hasattr(llm, 'prompter'):
        prompter = llm.prompter
    if hasattr(llm, 'pipeline') and hasattr(llm.pipeline, 'prompter'):
        prompter = llm.pipeline.prompter

    if prompter is None:
        if prompt_type is None:
            prompt_type = prompt_type_out
        # get prompter
        chat = True  # FIXME?
        prompter = Prompter(prompt_type, prompt_dict, debug=False, stream_output=stream_output,
                            system_prompt=system_prompt, tokenizer=tokenizer, base_model=model_name)

    scores = []
    chain = None

    if query.startswith(summary_prefix) or query.startswith(extract_prefix):
        # avoid gradio_runner injection of user lead to query being filled
        query = ''
    if inference_server and \
            inference_server.startswith('anthropic') and \
            is_json_model(model_name, inference_server) and \
            guided_json and response_format == 'json_object':
        extra = '\n\nYou must use the `JSON` tool, even if the answer seems obvious.  Do not respond with thinking, you must use the tool right away.\n'
        if query_action:
            query += extra
        prompt_summary += extra

    # basic version of prompt without docs etc.
    data_point = dict(context=context, instruction=query, input=iinput)
    prompt_basic = prompter.generate_prompt(data_point)

    # default is to embed query directly without processing
    query_embedding = query

    # support string as well
    if isinstance(document_choice, str):
        document_choice = [document_choice]
    if isinstance(document_source_substrings, str):
        document_source_substrings = [document_source_substrings]
    if isinstance(document_content_substrings, str):
        document_content_substrings = [document_content_substrings]

    get_answer_kwargs = dict(sources_show_text_in_accordion=sources_show_text_in_accordion,
                             hyde_show_intermediate_in_accordion=hyde_show_intermediate_in_accordion,
                             map_reduce_show_intermediate_in_accordion=map_reduce_show_intermediate_in_accordion,
                             show_link_in_sources=show_link_in_sources,
                             docs_ordering_type=docs_ordering_type,
                             top_k_docs_max_show=top_k_docs_max_show,
                             verbose=verbose,
                             )

    # NOTE: only includes those things get_llm() and get_chain() do not change
    run_target_func = functools.partial(run_target,
                                        stream_output=stream_output,
                                        lora_weights=lora_weights, max_time=max_time,
                                        sanitize_bot_response=sanitize_bot_response,
                                        from_ui=from_ui,
                                        verbose=verbose,
                                        langchain_action=langchain_action,
                                        query_action=query_action,
                                        )

    run_target_func_hyde = functools.partial(run_target,
                                             stream_output=stream_output,
                                             lora_weights=lora_weights, max_time=max_time,
                                             sanitize_bot_response=sanitize_bot_response,
                                             allow_response_no_refs=False,
                                             from_ui=from_ui,
                                             verbose=verbose,
                                             langchain_action=langchain_action,
                                             query_action=query_action,
                                             for_hyde=True,
                                             )

    func_names = list(inspect.signature(get_chain).parameters)
    sim_kwargs = {k: v for k, v in locals().items() if k in func_names}
    missing_kwargs = [x for x in func_names if x not in sim_kwargs]
    assert not missing_kwargs, "Missing: %s" % missing_kwargs

    llm_answers = dict(response_raw='')
    if hyde_level is not None and hyde_level > 0 and query_action and document_subset not in non_query_commands:
        query_embedding, llm_answers = yield from run_hyde(**locals().copy())
        sim_kwargs['query_embedding'] = query_embedding

    docs, chain, scores, \
        num_docs_before_cut, \
        use_llm_if_no_docs, top_k_docs_max_show, \
        llm, model_name, streamer, prompt_type_out, async_output, only_new_text = \
        get_chain(**sim_kwargs)

    if document_subset in non_query_commands:
        formatted_doc_chunks = '\n\n'.join([get_url(x) + '\n\n' + x.page_content for x in docs])
        if not formatted_doc_chunks and not use_llm_if_no_docs:
            yield dict(prompt=prompt_basic, response="No sources", sources='', num_prompt_tokens=0,
                       llm_answers=llm_answers, response_no_refs='', sources_str='', prompt_raw=prompt_basic)
            return
        # if no sources, outside gpt_langchain, LLM will be used with '' input
        scores = [1] * len(docs)
        get_answer_args = tuple([query, docs, formatted_doc_chunks,
                                 llm_answers,
                                 scores, show_rank,
                                 answer_with_sources,
                                 append_sources_to_answer,
                                 append_sources_to_chat])
        get_answer_kwargs.update(dict(t_run=time.time() - t_run,
                                      count_input_tokens=0,
                                      count_output_tokens=0,
                                      ))
        ret, sources, ret_no_refs, sources_str = get_sources_answer(*get_answer_args, **get_answer_kwargs)
        if response_format in ['json_object', 'json_code']:
            ret = '{"response": "%s"}' % ret
        yield dict(prompt=prompt_basic, response=formatted_doc_chunks, sources=sources, num_prompt_tokens=0,
                   llm_answers=llm_answers, response_no_refs='', sources_str=sources_str, prompt_raw=prompt_basic)
        return
    if langchain_agents and not chain:
        ret = '%s not supported by this model' % langchain_agents[0]
        if response_format in ['json_object', 'json_code']:
            ret = '{"response": "%s"}' % ret
        sources = []
        yield dict(prompt=prompt_basic, response=ret, sources=sources, num_prompt_tokens=0, llm_answers=llm_answers,
                   response_no_refs=ret, sources_str='', prompt_raw=prompt_basic)
        return
    # if only images, then still can do valid summarization
    if langchain_mode not in langchain_modes_non_db and not (docs or image_file):
        if langchain_action in [LangChainAction.SUMMARIZE_MAP.value,
                                LangChainAction.SUMMARIZE_ALL.value,
                                LangChainAction.SUMMARIZE_REFINE.value]:
            ret = 'No relevant documents to summarize.' if query or num_docs_before_cut > 0 else 'No documents to summarize.'
        elif langchain_action in [LangChainAction.EXTRACT.value]:
            ret = ['No relevant documents to extract from.'] if query or num_docs_before_cut > 0 else [
                'No documents to extract from.']
        elif not use_llm_if_no_docs:
            ret = 'No relevant documents to query (for chatting with LLM, pick Resources->Collections->LLM).' if num_docs_before_cut else 'No documents to query (for chatting with LLM, pick Resources->Collections->LLM).'
        else:
            # if here then ok to continue using chain if exists.  E.g. use_llm_if_no_docs=True and doing query langchain_action
            ret = None
        if ret is not None:
            if response_format in ['json_object', 'json_code']:
                ret = '{"response": "%s"}' % ret
            sources = []
            yield dict(prompt=prompt_basic, response=ret, sources=sources, num_prompt_tokens=0, llm_answers=llm_answers,
                       response_no_refs=ret, sources_str='', prompt_raw=prompt_basic)
            return

    # NOTE: If chain=None, could return if HF type (i.e. not langchain_only_model), but makes code too complex
    # only return now if no chain at all, e.g. when only returning sources
    if chain is None:
        return

    answer = yield from run_target_func(query=query,
                                        chain=chain,
                                        llm=llm,
                                        streamer=streamer,
                                        prompter=prompter,
                                        llm_answers=llm_answers,
                                        llm_answers_key='llm_answer_final',
                                        async_output=async_output,
                                        only_new_text=only_new_text)

    get_answer_args = tuple([query, docs, answer,
                             llm_answers,
                             scores, show_rank,
                             answer_with_sources,
                             append_sources_to_answer,
                             append_sources_to_chat])
    get_answer_kwargs.update(dict(t_run=time.time() - t_run,
                                  count_input_tokens=llm.count_input_tokens
                                  if hasattr(llm, 'count_input_tokens') else None,
                                  count_output_tokens=llm.count_output_tokens
                                  if hasattr(llm, 'count_output_tokens') else None,
                                  ))

    # for final yield, get real prompt used
    if hasattr(llm, 'pipeline') and hasattr(llm.pipeline, 'prompts') and llm.pipeline.prompts:
        if isinstance(llm.pipeline.prompts, list) and len(llm.pipeline.prompts) == 1:
            prompt = str(llm.pipeline.prompts[0])
        else:
            prompt = str(dedup_list(llm.pipeline.prompts))
    elif hasattr(llm, 'prompts') and llm.prompts:
        if isinstance(llm.prompts, list) and len(llm.prompts) == 1:
            if hasattr(llm.prompts[0], 'text'):
                prompt = str(llm.prompts[0].text)
            else:
                prompt = str(llm.prompts[0])
        else:
            prompt = str(dedup_list(llm.prompts))
    elif hasattr(llm, 'prompter') and llm.prompter.prompt:
        prompt = llm.prompter.prompt
    else:
        prompt = prompt_basic
    if hasattr(llm, 'count_input_tokens') and llm.count_input_tokens != 0:
        num_prompt_tokens = llm.count_input_tokens
    else:
        num_prompt_tokens = get_token_count(prompt, tokenizer)
    if hasattr(llm, 'count_output_tokens') and llm.count_output_tokens != 0:
        ntokens = llm.count_output_tokens
    else:
        ntokens = None

    # ensure to close client
    # https://github.com/langchain-ai/langchain/issues/13509
    if regenerate_clients and \
            hasattr(llm, 'client') and \
            hasattr(llm.client, '_client') and \
            hasattr(llm.client._client, 'close'):
        llm.client._client.close()

    if len(docs) == 0:
        # if no docs, then no sources to cite
        ret, sources = answer, []
        # doesn't actually have docs, but name means got to end with that answer
        llm_answers['llm_answer_final'] = ret
        if verbose:
            print('response: %s' % ret)
        yield dict(prompt_raw=prompt, response=ret, sources=sources, num_prompt_tokens=num_prompt_tokens,
                   llm_answers=llm_answers, response_no_refs=ret, sources_str='', ntokens=ntokens)
    elif answer is not None:
        ret, sources, ret_no_refs, sources_str = get_sources_answer(*get_answer_args, **get_answer_kwargs)
        llm_answers['llm_answer_final'] = ret
        if verbose:
            print('response: %s' % ret)
        yield dict(prompt_raw=prompt, response=ret, sources=sources, num_prompt_tokens=num_prompt_tokens,
                   llm_answers=llm_answers, response_no_refs=ret_no_refs, sources_str=sources_str,
                   ntokens=ntokens)
    if client_metadata:
        print("RUNQADB FINISH client_metadata: %s" % client_metadata, flush=True)
    return


def run_target(query='',
               chain=None,
               llm=None,
               streamer=None,
               prompter=None,
               llm_answers=dict(responses_raw=''),
               llm_answers_key='llm_answer_final',
               query_action=True,
               langchain_action=None,
               for_hyde=False,
               async_output=False,
               only_new_text=True,
               # things below are fixed for entire _run_qa_db() call once hit get_llm() and so on
               stream_output=False,
               lora_weights='',
               max_time=0,
               sanitize_bot_response=False,
               allow_response_no_refs=True,
               from_ui=True,
               verbose=False):
    if not for_hyde and not query_action:
        if langchain_action == LangChainAction.EXTRACT.value:
            llm_answers_key = 'map_'
        else:
            llm_answers_key = 'map_reduce_'

    # context stuff similar to used in evaluate()
    import torch
    device, torch_dtype, context_class = get_device_dtype()
    conditional_type = hasattr(llm, 'pipeline') and hasattr(llm.pipeline, 'model') and hasattr(llm.pipeline.model,
                                                                                               'conditional_type') and llm.pipeline.model.conditional_type
    with torch.no_grad():
        have_lora_weights = lora_weights not in [no_lora_str, '', None]
        context_class_cast = NullContext if device == 'cpu' or have_lora_weights or device == 'mps' else torch.autocast
        if conditional_type:
            # issues when casting to float16, can mess up t5 model, e.g. only when not streaming, or other odd behaviors
            context_class_cast = NullContext
        with context_class_cast(device):
            if stream_output and streamer:
                count_map_reduces = 0
                answer = None
                import queue
                bucket = queue.Queue()
                thread = EThread(target=chain, streamer=streamer, bucket=bucket, async_output=async_output)
                thread.start()
                if not for_hyde and not query_action:
                    outputs = ""
                else:
                    outputs = ""
                output1_old = ''
                res_dict = dict(prompt=query, response='', sources='', num_prompt_tokens=0, llm_answers=llm_answers,
                                response_no_refs='', sources_str='', prompt_raw=query)
                try:
                    tgen0 = time.time()
                    for new_text in streamer:
                        # print("new_text: %s" % new_text, flush=True)
                        if bucket.qsize() > 0 or thread.exc:
                            thread.join()
                        if new_text is not None:
                            if new_text:
                                outputs += new_text
                                if prompter:  # and False:  # FIXME: pipeline can already use prompter
                                    if conditional_type:
                                        if prompter.botstr:
                                            prompt = prompter.botstr
                                            output_with_prompt = prompt + outputs
                                            only_new_text = False  # override llm return
                                        else:
                                            prompt = None
                                            output_with_prompt = outputs
                                            only_new_text = True  # override llm return
                                    else:
                                        prompt = None  # FIXME
                                        output_with_prompt = outputs
                                        # don't specify only_new_text here, use get_llm() value
                                    output1 = prompter.get_response(output_with_prompt, prompt=prompt,
                                                                    only_new_text=only_new_text,
                                                                    sanitize_bot_response=sanitize_bot_response)
                                else:
                                    output1 = outputs
                                # in-place change to this key so exposed outside this generator
                                if llm_answers_key in ['map_reduce_', 'map_']:
                                    llm_answers[llm_answers_key + '%s' % (1 + count_map_reduces)] = output1
                                    if not from_ui:
                                        response_prefix = ''
                                    elif llm_answers_key == 'map_reduce_':
                                        response_prefix = "Computing Summarization Step %d:\n------------------\n" % (
                                                1 + count_map_reduces)
                                    else:
                                        response_prefix = "Computing Extraction Step %d:\n------------------\n" % (
                                                1 + count_map_reduces)
                                else:
                                    response_prefix = ''
                                    llm_answers[llm_answers_key] = output1
                                res_dict = dict(prompt=query, response=response_prefix + output1,
                                                sources='', num_prompt_tokens=0,
                                                llm_answers=llm_answers,
                                                response_no_refs=output1 if allow_response_no_refs else '',
                                                sources_str='',
                                                prompt_raw=query)
                                if output1 != output1_old:
                                    yield res_dict
                                    output1_old = output1
                                if time.time() - tgen0 > max_time:
                                    if verbose:
                                        print("Took too long EThread for %s" % (time.time() - tgen0), flush=True)
                                    break
                        else:
                            # start fresh
                            outputs = ''
                            output1_old = ''
                            count_map_reduces += 1
                    # yield if anything left over as can happen (FIXME: Understand better)
                    yield res_dict
                except BaseException:
                    # if any exception, raise that exception if was from thread, first
                    if thread.exc:
                        raise thread.exc
                    raise
                finally:
                    # in case no exception and didn't join with thread yet, then join
                    if not thread.exc:
                        answer = thread.join()
                        if isinstance(answer, dict):
                            if 'output_text' in answer:
                                answer = answer['output_text']
                            elif 'output' in answer:
                                answer = answer['output']
                            elif 'resolution' in answer:
                                answer = answer['resolution']
                        answer_fix = functools.partial(prompter.get_response, prompt=None,
                                                       only_new_text=only_new_text,
                                                       sanitize_bot_response=sanitize_bot_response)
                        if isinstance(answer, str):
                            # ensure any changes to text are done
                            answer = answer_fix(answer)
                        elif isinstance(answer, list):
                            answer = [answer_fix(x) for x in answer]
                # in case raise StopIteration or broke queue loop in streamer, but still have exception
                if thread.exc:
                    raise thread.exc
            else:
                if async_output:
                    import asyncio
                    answer = asyncio.run(chain())
                else:
                    answer = chain()
                    if isinstance(answer, dict):
                        if 'output_text' in answer:
                            answer = answer['output_text']
                        elif 'output' in answer:
                            answer = answer['output']
                        elif 'resolution' in answer:
                            answer = answer['resolution']
                        answer_fix = functools.partial(prompter.get_response, prompt=None,
                                                       only_new_text=only_new_text,
                                                       sanitize_bot_response=sanitize_bot_response)
                        if isinstance(answer, str):
                            # ensure any changes to text are done
                            answer = answer_fix(answer)
                        elif isinstance(answer, list):
                            answer = [answer_fix(x) for x in answer]

    llm_answers[llm_answers_key] = answer
    if verbose:
        print("answer: %s" % answer, flush=True)
    return answer


def get_docs_with_score(query, k_db,
                        filter_kwargs,
                        filter_kwargs_backup,
                        db, db_type, text_context_list=None,
                        chunk_id_filter=None,
                        where_document_dict={},
                        verbose=False):
    docs_with_score = _get_docs_with_score(query, k_db,
                                           filter_kwargs,
                                           db, db_type,
                                           text_context_list=text_context_list,
                                           chunk_id_filter=chunk_id_filter,
                                           where_document_dict=where_document_dict,
                                           verbose=verbose)
    if len(docs_with_score) == 0 and filter_kwargs != filter_kwargs_backup:
        docs_with_score = _get_docs_with_score(query, k_db,
                                               filter_kwargs_backup,
                                               db, db_type,
                                               text_context_list=text_context_list,
                                               chunk_id_filter=chunk_id_filter,
                                               where_document_dict=where_document_dict,
                                               verbose=verbose)
    return docs_with_score


def _get_docs_with_score(query, k_db,
                         filter_kwargs,
                         db, db_type, text_context_list=None,
                         chunk_id_filter=None,
                         where_document_dict={},
                         verbose=False):
    docs_with_score = []

    if text_context_list:
        docs_with_score += [(x, x.metadata.get('score', 1.0)) for x in text_context_list]

    # deal with bug in chroma where if (say) 234 doc chunks and ask for 233+ then fails due to reduction misbehavior
    if hasattr(db, '_embedding_function') and isinstance(db._embedding_function, FakeEmbeddings):
        top_k_docs = -1
        # don't add text_context_list twice
        db_documents, db_metadatas = get_docs_and_meta(db, top_k_docs, filter_kwargs=filter_kwargs,
                                                       text_context_list=None)
        # sort by order given to parser (file_id) and any chunk_id if chunked
        doc_file_ids = [x.get('file_id', 0) for x in db_metadatas]
        doc_chunk_ids = [x.get('chunk_id', 0) for x in db_metadatas]
        docs_with_score_fake = [(Document(page_content=result[0], metadata=result[1] or {}), 1.0)
                                for result in zip(db_documents, db_metadatas)]
        docs_with_score_fake = [x for fx, cx, x in
                                sorted(zip(doc_file_ids, doc_chunk_ids, docs_with_score_fake),
                                       key=lambda x: (x[0], x[1]))
                                ]
        docs_with_score += docs_with_score_fake
    elif db is not None and db_type in ['chroma', 'chroma_old']:
        t0 = time.time()
        docs_with_score_chroma = sim_search(db, query=query, k=k_db, with_score=True,
                                            filter_kwargs=filter_kwargs,
                                            chunk_id_filter=chunk_id_filter,
                                            where_document_dict=where_document_dict,
                                            verbose=verbose)
        docs_with_score += docs_with_score_chroma
        if verbose:
            print("sim_search in %s" % (time.time() - t0), flush=True)
    elif db is not None:
        with get_context_cast():
            docs_with_score_other = db.similarity_search_with_score(query, k=k_db, **filter_kwargs)
        docs_with_score += docs_with_score_other

    # set in metadata original order of docs
    [x[0].metadata.update(orig_index=ii) for ii, x in enumerate(docs_with_score)]

    return docs_with_score


def get_single_document(document_choice, db, extension=None):
    if isinstance(document_choice, str):
        document_choice = [document_choice]
    if document_choice and document_choice[0] == DocumentChoice.ALL.value:
        document_choice.remove(DocumentChoice.ALL.value)
    if document_choice is None:
        return None

    if len(document_choice) > 0:
        # then choose what user gave, first if have to choose
        document_choice_agent = [x for x in document_choice if x.endswith(extension)]
    elif len(document_choice) == 0:
        # means user didn't choose, see if can auto-choose
        document_choice_agent = sorted(set([x['source'] for x in get_metadatas(db, k_max=1000) if
                                            extension is None or x['source'].endswith(extension)]))
    else:
        document_choice_agent = document_choice
    document_choice_agent = [x for x in document_choice_agent if x.endswith(extension)]
    if len(document_choice_agent) > 0:
        return document_choice_agent[0]
    else:
        return None


def run_hyde(*args, **kwargs):
    """
    :param hyde_level: HYDE level
                 0: No HYDE
                 1: Use non-document-based LLM response and original query for embedding query
                 2: Use document-based LLM response and original query for embedding query
                 3+: continue iterations of embedding prior answer and getting new response
    :param hyde_template: Use HYDE approach (https://arxiv.org/abs/2212.10496)
                 None, 'None', 'auto' uses internal value and enable
                 'off' means disable
                 '{query}' is minimal template one can pass

    """

    # get vars
    query = kwargs['query']
    sim_kwargs = kwargs['sim_kwargs']
    run_target_func = kwargs['run_target_func_hyde']
    prompter = kwargs['prompter']
    hyde_level = kwargs['hyde_level']
    hyde_llm_prompt = kwargs['hyde_llm_prompt']
    hyde_template = kwargs['hyde_template']
    hyde_show_only_final = kwargs['hyde_show_only_final']
    verbose = kwargs['verbose']
    show_rank = kwargs['show_rank']
    answer_with_sources = kwargs['answer_with_sources']
    get_answer_kwargs = kwargs['get_answer_kwargs']
    append_sources_to_answer = kwargs['append_sources_to_answer']
    append_sources_to_chat = kwargs['append_sources_to_chat']
    prompt_basic = kwargs['prompt_basic']
    docs_joiner = kwargs['docs_joiner']
    from_ui = kwargs['from_ui']

    # get llm answer
    auto_hyde = """%s {query}""" % escape_braces(hyde_llm_prompt)
    if hyde_template in auto_choices:
        hyde_template = auto_hyde
    elif isinstance(hyde_template, str):
        assert '{query}' in hyde_template, "Require at least {query} in HYDE template, but got: %s" % hyde_template
    else:
        raise TypeError("Bad Type hyde_template=%s" % hyde_template)

    hyde_higher_template = """{query}\n\n{answer}"""

    # default
    llm_answers = {}
    hyde_chain = sim_kwargs.copy()
    # no-doc chain first if done
    hyde_chain['query'] = hyde_template.format(query=query)
    hyde_chain['db'] = None
    hyde_chain['load_db_if_exists'] = False
    hyde_chain['text_context_list'] = []
    sources = []
    answers = []

    for hyde_level1 in range(hyde_level):
        if verbose:
            print("hyde_level1=%d embedding_query=%s" % (hyde_level1, hyde_chain['query']), flush=True)

        # run chain
        docs, chain, scores, \
            num_docs_before_cut, \
            use_llm_if_no_docs, top_k_docs_max_show, \
            llm, model_name, streamer, prompt_type_out, async_output, only_new_text = \
            get_chain(**hyde_chain)

        # get answer, updates llm_answers internally too
        llm_answers_key = 'llm_answers_hyde_level_%d' % hyde_level1
        # for LLM, query remains same each time
        if from_ui:
            response_prefix = "Computing HYDE %d/%d response:\n------------------\n" % (1 + hyde_level1, hyde_level) \
                if hyde_level1 < hyde_level else ''
        else:
            response_prefix = ''
        answer = ''
        for ret in run_target_func(query=query,
                                   chain=chain,
                                   llm=llm,
                                   streamer=streamer,
                                   prompter=prompter,
                                   llm_answers=llm_answers,
                                   llm_answers_key=llm_answers_key,
                                   async_output=async_output,
                                   only_new_text=only_new_text):
            response = response_prefix + ret['response']
            if not hyde_show_only_final:
                ret['response'], pre_answer = get_hyde_acc(ret['response'], llm_answers,
                                                           get_answer_kwargs['hyde_show_intermediate_in_accordion'],
                                                           get_answer_kwargs[
                                                               'map_reduce_show_intermediate_in_accordion'],
                                                           )
                if pre_answer:
                    response = pre_answer + response
                yield dict(prompt_raw=ret['prompt'], response=response, sources=ret['sources'],
                           num_prompt_tokens=ret['num_prompt_tokens'],
                           llm_answers=ret['llm_answers'],
                           # only give back no_refs if final
                           response_no_refs='' if hyde_level1 < hyde_level else response,
                           sources_str=ret['sources_str'])
            answer = ret['response']

        if answer:
            # give back what have so far with any sources (what above yield doesn't do)
            get_answer_args = tuple([query, docs, answer,
                                     llm_answers,
                                     scores, show_rank,
                                     answer_with_sources,
                                     append_sources_to_answer,
                                     append_sources_to_chat])
            ret, sources, ret_no_refs, sources_str = get_sources_answer(*get_answer_args, **get_answer_kwargs)
            # FIXME: Something odd, UI gets stuck and no more yields if pass these sources inside ret
            # https://github.com/gradio-app/gradio/issues/6100
            # print("ret: %s" % ret)
            # yield dict(prompt=prompt_basic, response=ret, sources=sources, num_prompt_tokens=0, llm_answers=llm_answers)
            # try yield after
            # print("answer: %s" % answer)
            if not hyde_show_only_final:
                yield dict(prompt_raw=prompt_basic, response=ret_no_refs, sources=sources, num_prompt_tokens=0,
                           llm_answers=llm_answers,
                           response_no_refs='' if hyde_level1 < hyde_level else ret_no_refs,
                           sources_str=sources_str)

            # update embedding query
            # use all answers, but use newer answers first, often shorter due to LLM RLHF not used to long docs inputted,
            # then add rest and will be truncated at end
            answers.append(answer)
            answers_reverse = docs_joiner.join(answers[::-1])
            hyde_chain['query_embedding'] = hyde_higher_template.format(query=query, answer=answers_reverse)
        # update hyde_chain with doc version from now on
        hyde_chain['db'] = kwargs['db']
        hyde_chain['text_context_list'] = kwargs['text_context_list']
        hyde_chain['load_db_if_exists'] = True

    return hyde_chain['query_embedding'], llm_answers


def get_chain(query=None,
              query_embedding=None,
              iinput=None,
              context=None,  # FIXME: https://github.com/hwchase17/langchain/issues/6638
              use_openai_model=False, use_openai_embedding=False,
              langchain_instruct_mode=True,
              first_para=False, text_limit=None, top_k_docs=4, chunk=True, chunk_size=512,

              # urls
              use_unstructured=True,
              use_playwright=False,
              use_selenium=False,
              use_scrapeplaywright=False,
              use_scrapehttp=False,

              # pdfs
              use_pymupdf='auto',
              use_unstructured_pdf='auto',
              use_pypdf='auto',
              enable_pdf_ocr='auto',
              enable_pdf_doctr='auto',
              try_pdf_as_html='auto',

              # images
              enable_ocr=False,
              enable_doctr=False,
              enable_pix2struct=False,
              enable_captions=True,
              enable_llava=True,
              enable_transcriptions=True,
              captions_model=None,
              caption_loader=None,
              doctr_loader=None,
              pix2struct_loader=None,
              llava_model=None,
              llava_prompt=None,
              asr_model=None,
              asr_loader=None,

              # json
              jq_schema='.[]',
              extract_frames=10,

              langchain_mode_paths=None,
              langchain_mode_types=None,
              detect_user_path_changes_every_query=False,
              db_type='faiss',
              model_name=None,
              inference_server='',
              max_new_tokens=None,
              langchain_only_model=False,
              hf_embedding_model=None,
              migrate_embedding_model=False,
              prompter=None,
              prompt_type=None,
              prompt_dict=None,
              system_prompt=None,
              allow_chat_system_prompt=None,
              cut_distance=1.1,
              add_chat_history_to_context=True,  # FIXME: https://github.com/hwchase17/langchain/issues/6638
              add_search_to_context=False,
              keep_sources_in_context=False,
              gradio_errors_to_chatbot=True,
              memory_restriction_level=0,
              top_k_docs_max_show=10,

              load_db_if_exists=False,
              db=None,
              langchain_mode=None,
              langchain_action=None,
              langchain_agents=None,
              document_subset=DocumentSubset.Relevant.name,
              document_choice=[DocumentChoice.ALL.value],
              document_source_substrings=[],
              document_source_substrings_op='and',
              document_content_substrings=[],
              document_content_substrings_op='and',

              pre_prompt_query=None,
              prompt_query=None,
              pre_prompt_summary=None,
              prompt_summary=None,
              hyde_llm_prompt=None,
              all_docs_start_prompt=None,
              all_docs_finish_prompt=None,
              text_context_list=None,
              chat_conversation=None,
              user_prompt_for_fake_system_prompt=None,

              n_jobs=-1,
              # beyond run_db_query:
              llm=None,
              llm_kwargs=None,
              llm_explore=None,
              streamer=None,
              prompt_type_out=None,
              only_new_text=None,
              tokenizer=None,
              verbose=False,
              docs_ordering_type=docs_ordering_types_default,
              min_max_new_tokens=512,
              max_input_tokens=-1,
              max_total_input_tokens=-1,
              attention_sinks=False,
              truncation_generation=False,
              docs_token_handling=None,
              docs_joiner=None,
              doc_json_mode=False,
              metadata_in_context=[],

              stream_output=True,
              async_output=True,
              gradio_server=False,

              hyde_level=None,

              # local
              auto_reduce_chunks=True,
              max_chunks=100,
              use_llm_if_no_docs=None,
              headsize=50,
              max_time=None,

              query_action=None,
              summarize_action=None,

              doing_grounding=False,
              image_file=[],
              ):
    if inference_server is None:
        inference_server = ''
    assert hf_embedding_model is not None
    assert langchain_agents is not None  # should be at least []
    if text_context_list is None:
        text_context_list = []

    # same code in get_limited_prompt, but needed for summarization/extraction since only query returns that
    if gradio_server or not inference_server:
        # can listen to truncation_generation
        pass
    else:
        # these don't support allowing going beyond total context
        truncation_generation = True
    # default nothing
    docs = []
    target = None
    scores = []
    num_docs_before_cut = 0

    if len(text_context_list) > 0:
        # turn into documents to make easy to manage and add meta
        # try to account for summarization vs. query
        chunk_id = 0 if query_action else -1
        text_context_list = [
            Document(page_content=x, metadata=dict(source='text_context_list', score=1.0, chunk_id=chunk_id)) for x
            in text_context_list if x]

    if add_search_to_context:
        params = {
            "engine": "duckduckgo",
            "gl": "us",
            "hl": "en",
        }
        search = H2OSerpAPIWrapper(params=params)
        # if doing search, allow more docs
        docs_search, top_k_docs = search.get_search_documents(query,
                                                              query_action=query_action,
                                                              chunk=chunk, chunk_size=chunk_size,
                                                              db_type=db_type,
                                                              headsize=headsize,
                                                              top_k_docs=top_k_docs)
        text_context_list = docs_search + text_context_list
        add_search_to_context &= len(docs_search) > 0
        top_k_docs_max_show = max(top_k_docs_max_show, len(docs_search))

    if LangChainAgent.AUTOGPT.value in langchain_agents:
        from langchain_experimental.autonomous_agents.autogpt.agent import AutoGPT
        from langchain.agents import load_tools

        search_tools1 = load_tools(["ddg-search"], llm=llm)
        search_tools2 = load_tools(["serpapi"], llm=llm, serpapi_api_key=os.environ.get('SERPAPI_API_KEY'))
        search_tools = search_tools1 + search_tools2

        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=chunk_size)
        wiki_tools = [WikipediaQueryRun(api_wrapper=api_wrapper)]

        # from langchain_community.tools.file_management.read import ReadFileTool
        # from langchain_community.tools.file_management.write import WriteFileTool
        # file_tools = [WriteFileTool(), ReadFileTool()]
        from langchain_community.tools import ShellTool
        shell_tool = ShellTool()
        shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
            "{", "{{"
        ).replace("}", "}}")
        shell_tools = [shell_tool]

        from langchain_community.agent_toolkits import FileManagementToolkit
        # from tempfile import TemporaryDirectory
        # working_directory = TemporaryDirectory().name
        working_directory = "autogpt_files"
        makedirs(working_directory)
        toolkit = FileManagementToolkit(
            root_dir=str(working_directory)
        )  # If you don't provide a root_dir, operations will default to the current working directory
        file_tools = toolkit.get_tools()

        from gradio_tools.tools import (
            ImageCaptioningTool,
            StableDiffusionPromptGeneratorTool,
            StableDiffusionTool,
            TextToVideoTool,
        )
        do_image_tools = False  # FIXME: times out and blocks everything
        if do_image_tools:
            image_tools = [
                StableDiffusionTool().langchain,
                ImageCaptioningTool().langchain,
                StableDiffusionPromptGeneratorTool().langchain,
                TextToVideoTool().langchain,
            ]
        else:
            image_tools = []

        from langchain_experimental.utilities import PythonREPL
        python_repl = PythonREPL()
        # You can create the tool to pass to an agent
        from langchain.agents import Tool
        repl_tool = Tool(
            name="python_repl",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
            func=python_repl.run,
        )

        requests_tools = load_tools(["requests_all"])

        from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
        if os.environ.get('WOLFRAM_ALPHA_APPID'):
            wolfram = WolframAlphaAPIWrapper()
            wolfram_tool = Tool(
                name="wolframalpha",
                description="WolframAlpha is an answer engine developed by Wolfram Research. It answers factual queries by computing answers from externally sourced data.",
                func=wolfram.run,
            )
        else:
            wolfram_tool = None

        from langchain_experimental.llm_symbolic_math.base import LLMSymbolicMathChain
        sympy_math = LLMSymbolicMathChain.from_llm(llm)
        sympy_tool = Tool(
            name="sympy",
            description="SymPy is a Python library for symbolic mathematics. It aims to become a full-featured computer algebra system (CAS) while keeping the code as simple as possible in order to be comprehensible and easily extensible.",
            func=sympy_math.run,
        )

        enable_semantictool = False  # FIXME: Hit Can't patch loop of type <class 'uvloop.Loop'>
        if enable_semantictool:
            # from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
            # semantic = SemanticScholarAPIWrapper()
            # So can pass API key as ENV: S2_API_KEY
            from utils_langchain import H2OSemanticScholarAPIWrapper
            semantic = H2OSemanticScholarAPIWrapper()
            scholar_tool = Tool(
                name="semantictool",
                description="Semantic Scholar is a searchable database that uses AI to search and discover academic papers. It's supported by the Allen Institute for AI and indexes over 200 million academic papers.",
                func=semantic.run,
            )
            scholar_tools = [scholar_tool]
        else:
            scholar_tools = []

        tools = ([]
                 + search_tools
                 + wiki_tools
                 + shell_tools
                 + file_tools
                 + [repl_tool]
                 + requests_tools
                 + scholar_tools
                 + image_tools
                 )
        if os.getenv('WOLFRAM_ALPHA_APPID'):
            tools.extend([wolfram_tool])
        else:
            tools.extend([sympy_tool])

        from langchain_community.docstore import InMemoryDocstore
        from langchain_community.vectorstores import FAISS

        # Define your embedding model
        embeddings_model = get_embedding(use_openai_embedding, hf_embedding_model=hf_embedding_model)
        # Initialize the vectorstore as empty
        import faiss

        embedding_size = len(embeddings_model.embed_documents(['prompt'])[0])
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

        agent = AutoGPT.from_llm_and_tools(
            ai_name="h2oAutoGPT",
            ai_role="General Search and Knowledge Assistant",
            tools=tools,
            llm=llm,
            memory=vectorstore.as_retriever(),
        )
        # Set verbose to be true
        agent.chain.verbose = True
        chain_kwargs = [query]
        chain_func = agent.run
        target = wrapped_partial(chain_func, chain_kwargs)

        docs = []
        scores = []
        num_docs_before_cut = 0
        use_llm_if_no_docs = True
        return docs, target, scores, num_docs_before_cut, use_llm_if_no_docs, top_k_docs_max_show, \
            llm, model_name, streamer, prompt_type_out, async_output, only_new_text

    if LangChainAgent.SMART.value in langchain_agents:
        # doesn't really work for non-OpenAI models unless larger
        # but allow for now any model
        if True:
            # FIXME: streams first llm if both same llm, but not final answer part
            # FIXME: If 2 llms, then no streaming from ideation_llm, only from 2nd llm
            from langchain_experimental.smart_llm import SmartLLMChain
            ideation_llm = llm_explore if llm_explore is not None else llm
            critique_resolution_llm = llm
            prompt = PromptTemplate.from_template(query)
            chain = SmartLLMChain(
                ideation_llm=ideation_llm,
                llm=critique_resolution_llm,
                n_ideas=3,
                verbose=verbose,
                prompt=prompt,
            )
            chain_kwargs = {}
            if async_output:
                chain_func = chain.arun
            else:
                chain_func = chain
            target = wrapped_partial(chain_func, chain_kwargs)

            docs = []
            scores = []
            num_docs_before_cut = 0
            use_llm_if_no_docs = True
        return docs, target, scores, num_docs_before_cut, use_llm_if_no_docs, top_k_docs_max_show, \
            llm, model_name, streamer, prompt_type_out, async_output, only_new_text

    from output_parser import H2OMRKLOutputParser
    if LangChainAgent.SEARCH.value in langchain_agents:
        output_parser = H2OMRKLOutputParser()
        from langchain.agents import load_tools, AgentType, initialize_agent
        tools = load_tools(["serpapi"], llm=llm, serpapi_api_key=os.environ.get('SERPAPI_API_KEY'))
        if does_support_functiontools(inference_server, model_name):
            agent_type = AgentType.OPENAI_FUNCTIONS
            agent_executor_kwargs = {"handle_parsing_errors": True, 'output_parser': output_parser}
        else:
            agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION
            agent_executor_kwargs = {'output_parser': output_parser}
        chain = initialize_agent(tools, llm, agent=agent_type,
                                 agent_executor_kwargs=agent_executor_kwargs,
                                 agent_kwargs=dict(output_parser=output_parser,
                                                   format_instructions=output_parser.get_format_instructions()),
                                 output_parser=output_parser,
                                 max_iterations=10,
                                 max_execution_time=max_time,
                                 verbose=True)
        chain_kwargs = dict(input=query)
        target = wrapped_partial(chain, chain_kwargs)

        docs = []
        scores = []
        num_docs_before_cut = 0
        use_llm_if_no_docs = True
        return docs, target, scores, num_docs_before_cut, use_llm_if_no_docs, top_k_docs_max_show, \
            llm, model_name, streamer, prompt_type_out, async_output, only_new_text

    if LangChainAgent.COLLECTION.value in langchain_agents:
        if db:
            from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit
            from langchain.agents import create_vectorstore_agent

            output_parser = H2OMRKLOutputParser()
            vectorstore_info = VectorStoreInfo(
                name=langchain_mode,
                description="DataBase of text from PDFs, Image Captions, or web URL content",
                vectorstore=db,
            )
            toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)
            chain = create_vectorstore_agent(llm=llm, toolkit=toolkit,
                                             agent_executor_kwargs=dict(output_parser=output_parser),
                                             verbose=True, max_execution_time=max_time)

            chain_kwargs = dict(input=query)
            target = wrapped_partial(chain, chain_kwargs)

            use_llm_if_no_docs = True
        return docs, target, scores, num_docs_before_cut, use_llm_if_no_docs, top_k_docs_max_show, \
            llm, model_name, streamer, prompt_type_out, async_output, only_new_text

    if LangChainAgent.PYTHON.value in langchain_agents:
        # non-thread safe things inside worker, but only after in fork, so ok
        if does_support_functiontools(inference_server, model_name):
            from langchain.agents import AgentType
            from langchain_experimental.agents.agent_toolkits import create_python_agent
            chain = create_python_agent(
                llm=llm,
                tool=PythonREPLTool(),
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                agent_executor_kwargs={"handle_parsing_errors": True, 'max_execution_time': max_time},
                max_execution_time=max_time,
            )

            chain_kwargs = dict(input=query)
            target = wrapped_partial(chain, chain_kwargs)

            use_llm_if_no_docs = True
        return docs, target, scores, num_docs_before_cut, use_llm_if_no_docs, top_k_docs_max_show, \
            llm, model_name, streamer, prompt_type_out, async_output, only_new_text

    prefix_functiontools_csv = """You are working with a pandas dataframe in Python.  The name of the dataframe is: df.  Assume every question is about the dataframe, for example Describe means to describe or summarize the dataframe contents using the python_repl_ast tool.  Action input requests the tool to use, and only use the action python_repl_ast with valid JSON."""
    prefix_react_csv = """You are working with a pandas dataframe in Python.  The name of the dataframe is: df.  Assume every question is about the dataframe, for example Describe means to describe or summarize the dataframe contents using the python_repl_ast tool.  For Action, only use python_repl_ast.  For Action input, specify the python interpreter code in pandas you want to perform."""

    if LangChainAgent.PANDAS.value in langchain_agents:
        document_choice = get_single_document(document_choice, db, extension='csv')
        if document_choice and does_support_functiontools(inference_server, model_name):
            from langchain.agents import AgentType
            df = pd.read_csv(document_choice)
            chain = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=verbose,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                max_execution_time=max_time,
                prefix=prefix_functiontools_csv,
                agent_executor_kwargs=dict(handle_parsing_errors=True),
            )

            chain_kwargs = dict(input=query)
            target = wrapped_partial(chain, chain_kwargs)

            docs = []
            scores = []
            num_docs_before_cut = 0
            use_llm_if_no_docs = True
        return docs, target, scores, num_docs_before_cut, use_llm_if_no_docs, top_k_docs_max_show, \
            llm, model_name, streamer, prompt_type_out, async_output, only_new_text

    if LangChainAgent.JSON.value in langchain_agents:
        document_choice = get_single_document(document_choice, db, extension='json')
        if document_choice and does_support_functiontools(inference_server, model_name):
            # with open('src/openai.yaml') as f:
            #    data = yaml.load(f, Loader=yaml.FullLoader)
            with open(document_choice[0], 'rt') as f:
                data = json.loads(f.read())
            json_spec = JsonSpec(dict_=data, max_value_length=4000)

            from langchain_community.agent_toolkits import JsonToolkit
            from langchain.agents import create_json_agent

            json_toolkit = JsonToolkit(spec=json_spec)
            chain = create_json_agent(
                llm=llm, toolkit=json_toolkit,
                verbose=verbose,
                max_execution_time=max_time,
                agent_executor_kwargs=dict(handle_parsing_errors=True),
            )

            chain_kwargs = dict(input=query)
            target = wrapped_partial(chain, chain_kwargs)

            docs = []
            scores = []
            num_docs_before_cut = 0
            use_llm_if_no_docs = True
        return docs, target, scores, num_docs_before_cut, use_llm_if_no_docs, top_k_docs_max_show, \
            llm, model_name, streamer, prompt_type_out, async_output, only_new_text

    if LangChainAgent.CSV.value in langchain_agents:
        document_choice = get_single_document(document_choice, db, extension='csv')
        if document_choice:
            if does_support_functiontools(inference_server, model_name):
                from langchain.agents import AgentType
                chain = create_csv_agent(
                    llm,
                    document_choice,
                    prefix=prefix_functiontools_csv,
                    verbose=verbose, max_execution_time=max_time,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    agent_executor_kwargs=dict(handle_parsing_errors=True),
                )
            else:
                output_parser = H2OPythonMRKLOutputParser()
                from langchain.agents import AgentType
                chain = create_csv_agent(
                    llm,
                    document_choice,
                    prefix=prefix_react_csv,
                    number_of_head_rows=1,
                    verbose=verbose, max_execution_time=max_time,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    output_parser=output_parser,
                    format_instructions=output_parser.get_format_instructions(),
                    agent_kwargs=dict(handle_parsing_errors=True,
                                      output_parser=output_parser,
                                      format_instructions=output_parser.get_format_instructions(),
                                      ),
                    agent_executor_kwargs=dict(handle_parsing_errors=True,
                                               output_parser=output_parser,
                                               format_instructions=output_parser.get_format_instructions(),
                                               ),
                )
            chain_kwargs = dict(input=query)
            target = wrapped_partial(chain, chain_kwargs)

            docs = []
            scores = []
            num_docs_before_cut = 0
            use_llm_if_no_docs = True
        return docs, target, scores, num_docs_before_cut, use_llm_if_no_docs, top_k_docs_max_show, \
            llm, model_name, streamer, prompt_type_out, async_output, only_new_text

    # https://github.com/hwchase17/langchain/issues/1946
    # FIXME: Seems to way to get size of chroma db to limit top_k_docs to avoid
    # Chroma collection MyData contains fewer than 4 elements.
    # type logger error
    if top_k_docs == -1:
        k_db = 1000 if db_type in ['chroma', 'chroma_old'] else 100
    else:
        # top_k_docs=100 works ok too
        k_db = 1000 if db_type in ['chroma', 'chroma_old'] else top_k_docs

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
                                                        chunk=chunk, chunk_size=chunk_size,

                                                        # urls
                                                        use_unstructured=use_unstructured,
                                                        use_playwright=use_playwright,
                                                        use_selenium=use_selenium,
                                                        use_scrapeplaywright=use_scrapeplaywright,
                                                        use_scrapehttp=use_scrapehttp,

                                                        # pdfs
                                                        use_pymupdf=use_pymupdf,
                                                        use_unstructured_pdf=use_unstructured_pdf,
                                                        use_pypdf=use_pypdf,
                                                        enable_pdf_ocr=enable_pdf_ocr,
                                                        enable_pdf_doctr=enable_pdf_doctr,
                                                        try_pdf_as_html=try_pdf_as_html,

                                                        # images
                                                        enable_ocr=enable_ocr,
                                                        enable_doctr=enable_doctr,
                                                        enable_pix2struct=enable_pix2struct,
                                                        enable_captions=enable_captions,
                                                        enable_llava=enable_llava,
                                                        enable_transcriptions=enable_transcriptions,
                                                        captions_model=captions_model,
                                                        caption_loader=caption_loader,
                                                        doctr_loader=doctr_loader,
                                                        pix2struct_loader=pix2struct_loader,
                                                        llava_model=llava_model,
                                                        llava_prompt=llava_prompt,
                                                        asr_model=asr_model,
                                                        asr_loader=asr_loader,

                                                        # json
                                                        jq_schema=jq_schema,
                                                        extract_frames=extract_frames,

                                                        langchain_mode=langchain_mode,
                                                        langchain_mode_paths=langchain_mode_paths,
                                                        langchain_mode_types=langchain_mode_types,
                                                        db_type=db_type,
                                                        load_db_if_exists=load_db_if_exists,
                                                        db=db,
                                                        n_jobs=n_jobs,
                                                        verbose=verbose)
    use_template = not use_openai_model and langchain_instruct_mode or langchain_only_model
    template, template_if_no_docs, auto_reduce_chunks, query = \
        get_template(query, iinput,
                     pre_prompt_query, prompt_query,
                     pre_prompt_summary, prompt_summary,
                     all_docs_start_prompt,
                     all_docs_finish_prompt,
                     langchain_action,
                     query_action,
                     summarize_action,
                     True,  # just to overestimate prompting
                     auto_reduce_chunks,
                     add_search_to_context,
                     system_prompt,
                     doc_json_mode,
                     model_name=model_name,
                     prompter=prompter)

    model_max_length = get_model_max_length(llm=llm, tokenizer=tokenizer, inference_server=inference_server,
                                            model_name=model_name)

    if not attention_sinks:
        # use min_max_new_tokens instead of max_new_tokens for max_new_tokens to get the largest input allowable
        #  else max_input_tokens interpreted as user input as smaller than possible and get over-restricted
        # but if summarization, this defines max tokens in each chunk, for same used max_new_tokens, so need to use original,
        #  e.g. first map may produce some output, larger than 256 tokens, and upon reduce includes that large output, which won't work for same large max_new_tokens -> max_input_tokens
        if query_action:
            max_new_tokens_used = min_max_new_tokens
        else:
            max_new_tokens_used = max_new_tokens
        max_input_tokens_default = get_max_input_tokens(llm=llm, tokenizer=tokenizer, inference_server=inference_server,
                                                        model_name=model_name, max_new_tokens=max_new_tokens_used)
        if max_input_tokens >= 0:
            max_input_tokens = min(max_input_tokens_default, max_input_tokens)
        else:
            max_input_tokens = max_input_tokens_default

        # don't let breach
        # max_new_tokens = model_max_length - max_input_tokens
        # min_max_new_tokens = min(min_max_new_tokens, max_new_tokens)

    else:
        if max_input_tokens < 0:
            max_input_tokens = model_max_length

    # GET FILTER

    if not is_chroma_db(db):
        # only chroma supports filtering
        chunk_id_filter = None
        filter_kwargs = {}
        filter_kwargs_backup = {}
        where_document_dict = {}
    else:
        where_document_dict = {}
        if document_content_substrings:
            if len(document_content_substrings) > 1:
                inner_list = [{'$contains': x} for x in document_content_substrings]
                if document_content_substrings_op == 'or':
                    where_document = {"$or": inner_list}
                else:
                    where_document = {"$and": inner_list}
            else:
                where_document = {'$contains': document_content_substrings[0]}
            where_document_dict = dict(where_document=where_document)
        import logging
        logging.getLogger("chromadb").setLevel(logging.ERROR)
        assert document_choice is not None, "Document choice was None"
        if isinstance(db, Chroma):
            filter_kwargs_backup = {}  # shouldn't ever need backup
            # chroma >= 0.4
            if len(document_choice) == 0 or len(document_choice) >= 1 and document_choice[
                0] == DocumentChoice.ALL.value:
                chunk_id_filter = 0 if query_action else -1
                filter_kwargs = {"filter": {"chunk_id": {"$gte": 0}}} if query_action else \
                    {"filter": {"chunk_id": {"$eq": -1}}}
            else:
                if document_choice[0] == DocumentChoice.ALL.value:
                    document_choice = document_choice[1:]
                if len(document_choice) == 0:
                    chunk_id_filter = None
                    filter_kwargs = {}
                elif len(document_choice) > 1:
                    chunk_id_filter = None
                    or_filter = [
                        {"$and": [dict(source={"$eq": x}), dict(chunk_id={"$gte": 0})]} if query_action else {
                            "$and": [dict(source={"$eq": x}), dict(chunk_id={"$eq": -1})]}
                        for x in document_choice]
                    filter_kwargs = dict(filter={"$or": or_filter})
                    or_filter_backup = [
                        dict(source={"$eq": x}) if query_action else dict(source={"$eq": x})
                        for x in document_choice]
                    filter_kwargs_backup = dict(filter={"$or": or_filter_backup})
                else:
                    chunk_id_filter = None
                    # still chromadb UX bug, have to do different thing for 1 vs. 2+ docs when doing filter
                    one_filter = \
                        [{"source": {"$eq": x}, "chunk_id": {"$gte": 0}} if query_action else {
                            "source": {"$eq": x},
                            "chunk_id": {
                                "$eq": -1}}
                         for x in document_choice][0]

                    filter_kwargs = dict(filter={"$and": [dict(source=one_filter['source']),
                                                          dict(chunk_id=one_filter['chunk_id'])]})
                    one_filter_backup = \
                        [{"source": {"$eq": x}, "chunk_id": {"$gte": 0}} if query_action else {
                            "source": {"$eq": x},
                            "chunk_id": {
                                "$eq": -1}}
                         for x in document_choice][0]

                    filter_kwargs_backup = dict(filter=dict(source=one_filter_backup['source']))
        else:
            # migration for chroma < 0.4
            if len(document_choice) == 0 or len(document_choice) >= 1 and document_choice[
                0] == DocumentChoice.ALL.value:
                chunk_id_filter = 0 if query_action else -1
                filter_kwargs = {"filter": {"chunk_id": {"$gte": 0}}} if query_action else \
                    {"filter": {"chunk_id": {"$eq": -1}}}
                filter_kwargs_backup = {"filter": {"chunk_id": {"$gte": 0}}}
            elif len(document_choice) >= 2:
                if document_choice[0] == DocumentChoice.ALL.value:
                    document_choice = document_choice[1:]
                chunk_id_filter = None
                or_filter = [
                    {"source": {"$eq": x}, "chunk_id": {"$gte": 0}} if query_action else {"source": {"$eq": x},
                                                                                          "chunk_id": {
                                                                                              "$eq": -1}}
                    for x in document_choice]
                filter_kwargs = dict(filter={"$or": or_filter})
                or_filter_backup = [
                    {"source": {"$eq": x}} if query_action else {"source": {"$eq": x}}
                    for x in document_choice]
                filter_kwargs_backup = dict(filter={"$or": or_filter_backup})
            elif len(document_choice) == 1:
                chunk_id_filter = None
                # degenerate UX bug in chroma
                one_filter = \
                    [{"source": {"$eq": x}, "chunk_id": {"$gte": 0}} if query_action else {"source": {"$eq": x},
                                                                                           "chunk_id": {
                                                                                               "$eq": -1}}
                     for x in document_choice][0]
                filter_kwargs = dict(filter=one_filter)
                one_filter_backup = \
                    [{"source": {"$eq": x}} if query_action else {"source": {"$eq": x}}
                     for x in document_choice][0]
                filter_kwargs_backup = dict(filter=one_filter_backup)
            else:
                chunk_id_filter = None
                # shouldn't reach
                filter_kwargs = {}
                filter_kwargs_backup = {}

    # GET DOCS

    if document_subset == DocumentSubset.TopKSources.name or query in [None, '', '\n']:
        db_documents, db_metadatas = get_docs_and_meta(db, top_k_docs, filter_kwargs=filter_kwargs,
                                                       text_context_list=text_context_list,
                                                       chunk_id_filter=chunk_id_filter)
        if len(db_documents) == 0 and filter_kwargs_backup != filter_kwargs:
            db_documents, db_metadatas = get_docs_and_meta(db, top_k_docs, filter_kwargs=filter_kwargs_backup,
                                                           text_context_list=text_context_list,
                                                           chunk_id_filter=chunk_id_filter)

        # similar to langchain's chroma's _results_to_docs_and_scores
        docs_with_score = [(Document(page_content=result[0], metadata=result[1] or {}), 0)
                           for result in zip(db_documents, db_metadatas)]
        # remove empty content, e.g. from exception version of document, so don't include empty stuff in summarization
        docs_with_score = [x for x in docs_with_score if x[0].page_content]
        # set in metadata original order of docs
        [x[0].metadata.update(orig_index=ii) for ii, x in enumerate(docs_with_score)]

        # order documents
        doc_hashes = [x.get('doc_hash', 'None') if x.get('doc_hash', 'None') is not None else 'None' for x in
                      db_metadatas]
        if query_action:
            doc_chunk_ids = [x.get('chunk_id', 0) if x.get('chunk_id', 0) is not None else 0 for x in db_metadatas]
            docs_with_score2 = [x for hx, cx, x in
                                sorted(zip(doc_hashes, doc_chunk_ids, docs_with_score), key=lambda x: (x[0], x[1]))
                                if cx >= 0]
        else:
            assert summarize_action
            doc_chunk_ids = [x.get('chunk_id', -1) if x.get('chunk_id', -1) is not None else -1 for x in db_metadatas]
            docs_with_score2 = [x for hx, cx, x in
                                sorted(zip(doc_hashes, doc_chunk_ids, docs_with_score), key=lambda x: (x[0], x[1]))
                                if cx == -1
                                ]
            if len(docs_with_score2) == 0 and len(docs_with_score) > 0:
                # old database without chunk_id, migration added 0 but didn't make -1 as that would be expensive
                # just do again and relax filter, let summarize operate on actual chunks if nothing else
                docs_with_score2 = [x for hx, cx, x in
                                    sorted(zip(doc_hashes, doc_chunk_ids, docs_with_score),
                                           key=lambda x: (x[0], x[1]))
                                    ]
        docs_with_score = docs_with_score2

        top_k_docs_sample = len(db_documents) if top_k_docs == -1 else top_k_docs
        docs_with_score = docs_with_score[:top_k_docs_sample]
        docs = [x[0] for x in docs_with_score]
        scores = [x[1] for x in docs_with_score]
    else:
        docs_with_score = get_docs_with_score(query_embedding, k_db,
                                              filter_kwargs,
                                              filter_kwargs_backup,
                                              db, db_type,
                                              text_context_list=text_context_list,
                                              chunk_id_filter=chunk_id_filter,
                                              where_document_dict=where_document_dict,
                                              verbose=verbose)
        if document_source_substrings:
            set_document_source_substrings = set(document_source_substrings)
            if document_source_substrings_op == 'or':
                docs_with_score = [x for x in docs_with_score if
                                   any(y in x[0].metadata.get('source') for y in set_document_source_substrings)]
            else:
                docs_with_score = [x for x in docs_with_score if
                                   all(y in x[0].metadata.get('source') for y in set_document_source_substrings)]

    if metadata_in_context is None:
        metadata_in_context = []
    if isinstance(metadata_in_context, str) and metadata_in_context == 'auto':
        if 'h2o-danube2' in model_name or 'h2o-danube3' in model_name or 'h2o-danube-' in model_name:
            # model not good enough to handle
            metadata_in_context = []
    if db is None and text_context_list:
        # not useful then, just mess
        metadata_in_context = []
    elif not metadata_in_context:
        metadata_in_context = []
    elif isinstance(metadata_in_context, str) and metadata_in_context not in ['all', 'auto']:
        metadata_in_context = ast.literal_eval(metadata_in_context)
        assert isinstance(metadata_in_context, list)

    if metadata_in_context == 'all':
        metadata_in_context_set = FullSet()
    elif metadata_in_context == 'auto':
        metadata_in_context_set = set(
            ['date', 'file_path', 'input_type', 'keywords', 'chunk_id', 'page', 'source', 'title'])
    else:
        assert isinstance(metadata_in_context, list)
        metadata_in_context_set = set(metadata_in_context)

    # SELECT PROMPT + DOCS

    tokenizer = get_tokenizer(db=db, llm=llm, tokenizer=tokenizer, inference_server=inference_server,
                              use_openai_model=use_openai_model,
                              db_type=db_type)

    get_limited_prompt_func = functools.partial(get_limited_prompt,
                                                prompter=prompter,
                                                base_model=model_name,
                                                inference_server=inference_server,
                                                prompt_type=prompt_type,
                                                prompt_dict=prompt_dict,
                                                max_new_tokens=max_new_tokens,
                                                system_prompt=system_prompt,
                                                allow_chat_system_prompt=allow_chat_system_prompt,
                                                context=context,
                                                chat_conversation=chat_conversation,
                                                user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt,
                                                keep_sources_in_context=keep_sources_in_context,
                                                gradio_errors_to_chatbot=gradio_errors_to_chatbot,
                                                model_max_length=model_max_length,
                                                memory_restriction_level=memory_restriction_level,
                                                langchain_mode=langchain_mode,
                                                add_chat_history_to_context=add_chat_history_to_context,
                                                min_max_new_tokens=min_max_new_tokens,
                                                max_input_tokens=max_input_tokens,
                                                truncation_generation=truncation_generation,
                                                gradio_server=gradio_server,
                                                attention_sinks=attention_sinks,
                                                hyde_level=hyde_level,
                                                doing_grounding=doing_grounding,
                                                image_file=image_file,
                                                )

    # NOTE: if map_reduce, then no need to auto reduce chunks
    if query_action and (top_k_docs == -1 or auto_reduce_chunks):
        top_k_docs_tokenize = 100
        docs_with_score = docs_with_score[:top_k_docs_tokenize]
        template_text = template_if_no_docs.format(context='', question='')

        # add metadata to documents and make new copy of docs with them to not contaminate originals
        if metadata_in_context and not doc_json_mode and not doing_grounding:
            docs_with_score = [(Document(page_content='Begin Document:\n\n' +
                                                      'Metadata:\n' +
                                                      '\n'.join(['%s = %s' % (k, v) for k, v in x.metadata.items() if
                                                                 v and k in metadata_in_context_set]) +
                                                      '\n\nDocument Contents:\n"""\n' +
                                                      x.page_content +
                                                      '\n"""\nEnd Document\n',
                                         metadata=copy.deepcopy(x.metadata) or {}), score)
                               for x, score in docs_with_score]

        # first docs_with_score are most important with highest score
        estimated_full_prompt, \
            query, iinput, context, \
            num_prompt_tokens, max_new_tokens, \
            num_prompt_tokens0, num_prompt_tokens_actual, \
            history_to_use_final, external_handle_chat_conversation, \
            top_k_docs_trial, one_doc_size, \
            truncation_generation, system_prompt, pre_prompt_query, prompt_query = \
            get_limited_prompt_func(query,
                                    iinput,
                                    tokenizer,
                                    template_text=template_text,
                                    text_context_list=[x[0].page_content for x in
                                                       docs_with_score],
                                    lang_pre_prompt=pre_prompt_query,
                                    lang_prompt=prompt_query,
                                    )
        # redo template in case pre_prompt and prompt changed
        template, template_if_no_docs, auto_reduce_chunks, query = \
            get_template(query, iinput,
                         pre_prompt_query, prompt_query,
                         pre_prompt_summary, prompt_summary,
                         all_docs_start_prompt,
                         all_docs_finish_prompt,
                         langchain_action,
                         query_action,
                         summarize_action,
                         True,  # just to overestimate prompting
                         auto_reduce_chunks,
                         add_search_to_context,
                         system_prompt,
                         doc_json_mode,
                         model_name=model_name,
                         prompter=prompter)

        # get updated llm
        actual_input_tokens = max(num_prompt_tokens, num_prompt_tokens0, num_prompt_tokens_actual)
        # see if can avoid dropping to min_max_new_tokens and use max_new_tokens
        max_new_tokens_possible = model_max_length - actual_input_tokens - 32
        max_new_tokens = max(min(max_new_tokens, max_new_tokens_possible), min_max_new_tokens)

        llm_kwargs.update(max_new_tokens=max_new_tokens,
                          max_input_tokens=max_input_tokens,
                          max_total_input_tokens=max_total_input_tokens,
                          context=context,
                          iinput=iinput,
                          system_prompt=system_prompt)
        if external_handle_chat_conversation or prompter.prompt_type in [template_prompt_type, unknown_prompt_type]:
            # should already have attribute, checking sanity
            assert hasattr(llm, 'chat_conversation') or isinstance(llm, H2OHuggingFacePipeline)
            llm_kwargs.update(chat_conversation=history_to_use_final)
        llm, model_name, streamer, prompt_type_out, async_output, only_new_text, gradio_server = \
            get_llm(**llm_kwargs)

        # avoid craziness
        top_k_docs_sample = len(docs_with_score) if top_k_docs == -1 else top_k_docs
        if 0 < top_k_docs_trial < max_chunks:
            # avoid craziness
            if top_k_docs == -1:
                top_k_docs_sample = top_k_docs_trial
            else:
                top_k_docs_sample = min(top_k_docs, top_k_docs_trial)
        elif top_k_docs_trial >= max_chunks:
            top_k_docs_sample = max_chunks
        docs_with_score = select_docs_with_score(docs_with_score, top_k_docs_sample, one_doc_size)
    else:
        # don't reduce, except listen to top_k_docs and max_total_input_tokens
        one_doc_size = None
        if max_total_input_tokens not in [None, -1]:
            # used to limit tokens for summarization, e.g. public instance, over all LLM calls allowed
            top_k_docs, one_doc_size, num_doc_tokens = \
                get_docs_tokens(tokenizer,
                                text_context_list=[x[0].page_content for x in docs_with_score],
                                max_input_tokens=max_total_input_tokens)
        # filter by top_k_docs and maybe one_doc_size
        docs_with_score = select_docs_with_score(docs_with_score, top_k_docs, one_doc_size)

    if summarize_action:
        if '{text}' in template:
            template_text = template.format(text='')
        elif '{input_documents}' in template:
            template_text = template.format(input_documents='')
        elif '{question}' in template:
            template_text = template.format(question='')
        else:
            template_text = ''

        # first docs_with_score are most important with highest score
        estimated_full_prompt, \
            _, iinput, context, \
            num_prompt_tokens1, max_new_tokens1, \
            num_prompt_tokens01, num_prompt_tokens_actual1, \
            history_to_use_final1, external_handle_chat_conversation1, \
            top_k_docs1, one_doc_size1, \
            truncation_generation1, system_prompt, pre_prompt_summary, prompt_summary = \
            get_limited_prompt_func(template_text,
                                    iinput,
                                    tokenizer,
                                    template_text=template_text,
                                    text_context_list=[],
                                    # nothing, just getting base amount for each call
                                    lang_pre_prompt=pre_prompt_summary,
                                    lang_prompt=prompt_summary,
                                    )
        # get template again in case pre_prompt and prompt changed
        template, template_if_no_docs, auto_reduce_chunks, query = \
            get_template(query, iinput,
                         pre_prompt_query, prompt_query,
                         pre_prompt_summary, prompt_summary,
                         all_docs_start_prompt,
                         all_docs_finish_prompt,
                         langchain_action,
                         query_action,
                         summarize_action,
                         True,  # just to overestimate prompting
                         auto_reduce_chunks,
                         add_search_to_context,
                         system_prompt,
                         doc_json_mode,
                         model_name=model_name,
                         prompter=prompter)

        # get updated llm, so includes chat_conversation etc.
        llm_kwargs.update(  # max_new_tokens=max_new_tokens,
            # max_input_tokens=max_input_tokens,
            # max_total_input_tokens=max_total_input_tokens,
            context=context,
            iinput=iinput,
            system_prompt=system_prompt)

        docs_with_score, max_doc_tokens = split_merge_docs(docs_with_score,
                                                           tokenizer,
                                                           max_input_tokens=max_input_tokens,
                                                           docs_token_handling=docs_token_handling,
                                                           joiner=docs_joiner if not doing_grounding else "Document xx",
                                                           # ensure splitting of docs accounts for rest of non-doc prompt
                                                           non_doc_prompt=estimated_full_prompt + template_text,
                                                           hf_embedding_model=hf_embedding_model,
                                                           verbose=verbose)
        # in case docs_with_score grew due to splitting, limit again by top_k_docs
        if top_k_docs > 0:
            docs_with_score = docs_with_score[:top_k_docs]
        # max_input_tokens used min_max_new_tokens as max_new_tokens, so need to assume filled up to that
        # but use actual largest token count
        if '{text}' in template:
            estimated_prompt_no_docs = template.format(text='')
        elif '{input_documents}' in template:
            estimated_prompt_no_docs = template.format(input_documents='')
        elif '{question}' in template:
            estimated_prompt_no_docs = template.format(question=query)
        else:
            estimated_prompt_no_docs = query
        data_point = dict(context=context, instruction=estimated_prompt_no_docs or ' ', input=iinput)
        prompt_basic = prompter.generate_prompt(data_point,
                                                chat_conversation=chat_conversation,
                                                user_prompt_for_fake_system_prompt=user_prompt_for_fake_system_prompt,
                                                )
        num_prompt_basic_tokens = get_token_count(prompt_basic, tokenizer)

        if truncation_generation:
            max_new_tokens = max(min_max_new_tokens,
                                 min(max_new_tokens, model_max_length - max_doc_tokens - num_prompt_basic_tokens))
            if os.getenv('HARD_ASSERTS') is not None:
                # imperfect calculation, so will see how testing does
                assert max_new_tokens >= min_max_new_tokens - 50, "%s %s" % (max_new_tokens, min_max_new_tokens)
        # get updated llm
        llm_kwargs.update(max_new_tokens=max_new_tokens, max_input_tokens=max_input_tokens)
        llm, model_name, streamer, prompt_type_out, async_output, only_new_text, gradio_server = \
            get_llm(**llm_kwargs)

    # now done with all docs and their sizes, re-order docs if required
    if query_action:
        # not relevant for summarization, including in chunk mode, so process docs in order for summarization or extraction
        # put most relevant chunks closest to question,
        # esp. if truncation occurs will be "oldest" or "farthest from response" text that is truncated
        # BUT: for small models, e.g. 6_9 pythia, if sees some stuff related to h2oGPT first, it can connect that and not listen to rest
        if docs_ordering_type in ['best_first']:
            pass
        elif docs_ordering_type in ['best_near_prompt', 'reverse_sort']:
            docs_with_score.reverse()
        elif docs_ordering_type in ['', None, 'reverse_ucurve_sort']:
            docs_with_score = reverse_ucurve_list(docs_with_score)
        else:
            raise ValueError("No such docs_ordering_type=%s" % docs_ordering_type)

    # cut off so no high distance docs/sources considered
    # NOTE: If no query, then distance set was 0 and nothing will be cut
    num_docs_before_cut = len(docs_with_score)
    docs = [x[0] for x in docs_with_score if x[1] < cut_distance]
    scores = [x[1] for x in docs_with_score if x[1] < cut_distance]
    if len(scores) > 0 and verbose:
        print("Distance: min: %s max: %s mean: %s median: %s" %
              (scores[0], scores[-1], np.mean(scores), np.median(scores)), flush=True)

    # if HF type and have no docs, could bail out, but makes code too complex

    if document_subset in non_query_commands:
        # no LLM use at all, just sources
        return docs, None, [], num_docs_before_cut, use_llm_if_no_docs, top_k_docs_max_show, \
            llm, model_name, streamer, prompt_type_out, async_output, only_new_text

    # FIXME: WIP
    common_words_file = "data/NGSL_1.2_stats.csv.zip"
    if False and os.path.isfile(common_words_file) and langchain_action == LangChainAction.QUERY.value:
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

    got_any_docs = len(docs) > 0
    # update template in case situation changed or did get docs
    # then no new documents from database or not used, redo template
    # got template earlier as estimate of template token size, here is final used version
    template, template_if_no_docs, auto_reduce_chunks, query = \
        get_template(query, iinput,
                     pre_prompt_query, prompt_query,
                     pre_prompt_summary, prompt_summary,
                     all_docs_start_prompt,
                     all_docs_finish_prompt,
                     langchain_action,
                     query_action,
                     summarize_action,
                     got_any_docs,
                     auto_reduce_chunks,
                     add_search_to_context,
                     system_prompt,
                     doc_json_mode,
                     model_name=model_name,
                     prompter=prompter)

    if doc_json_mode:
        # make copy so don't change originals
        if metadata_in_context and not doing_grounding:
            docs = [Document(page_content=json.dumps(merge_dict(dict(ID=xi, content=x.page_content),
                                                                {k: v for k, v in x.metadata.items() if
                                                                 v and k in metadata_in_context_set})),
                             metadata=copy.deepcopy(x.metadata) or {})
                    for xi, x in enumerate(docs)]
        else:
            docs = [Document(page_content=json.dumps(dict(ID=xi, content=x.page_content)),
                             metadata=copy.deepcopy(x.metadata) or {})
                    for xi, x in enumerate(docs)]

    if langchain_action == LangChainAction.QUERY.value:
        if doing_grounding:
            # https://huggingface.co/CohereForAI/c4ai-command-r-v01
            prompt = PromptTemplate(
                # input_variables=["summaries", "question"],
                input_variables=["context", "question"],
                template='{context}{question}',  # ignored
            )
            chain = load_qa_chain(llm, prompt=prompt, verbose=verbose)
            from openai_server.backend_utils import structure_to_messages

            while True:
                conversation = structure_to_messages(query,
                                                     system_prompt if system_prompt not in [None, '', 'auto'] else None,
                                                     chat_conversation,
                                                     image_file)
                documents = [merge_dict(dict(text=x.page_content),
                                        {k: v for k, v in x.metadata.items() if
                                         v and k in metadata_in_context_set}) for x in docs]
                query_with_docs = tokenizer.apply_grounded_generation_template(
                    conversation,
                    documents=documents,
                    citation_mode="accurate",  # or "fast"
                    tokenize=False,
                    add_generation_prompt=True,
                )
                grounded_tokens = len(tokenizer.encode(query_with_docs))
                if grounded_tokens > max_input_tokens and len(docs) > 0:
                    if docs_ordering_type in ['best_first']:
                        docs.pop()
                    elif docs_ordering_type in ['best_near_prompt', 'reverse_sort']:
                        docs.pop(0)
                    elif docs_ordering_type in ['', None, 'reverse_ucurve_sort']:
                        del docs[len(docs) // 2]
                    else:
                        raise ValueError("No such docs_ordering_type=%s" % docs_ordering_type)

                elif grounded_tokens > max_input_tokens and len(chat_conversation) > 0:
                    chat_conversation = []
                elif grounded_tokens > max_input_tokens and system_prompt:
                    system_prompt = ''
                else:
                    if grounded_tokens > max_input_tokens:
                        print("Failed to fit grounded tokens: %s %s" % (grounded_tokens, max_input_tokens))
                    break

            chain_kwargs = dict(input_documents=[], question=query_with_docs)
        else:
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
                # unused normally except in testing
                assert use_openai_model or prompt_type in [empty_prompt_type, noop_prompt_type,
                                                           unknown_prompt_type], "Unexpected to use few-shot template for %s %s" % (
                    model_name, prompt_type)
                chain = load_qa_with_sources_chain(llm)
            chain_kwargs = dict(input_documents=docs, question=query)
        target = wrapped_partial(chain, chain_kwargs)
    elif summarize_action:
        if async_output:
            return_intermediate_steps = False
        else:
            return_intermediate_steps = True
        if langchain_action == LangChainAction.SUMMARIZE_MAP.value:
            prompt = PromptTemplate(input_variables=["text"], template=template)
            # token_max is per llm call
            chain = load_general_summarization_chain(llm, chain_type="map_reduce",
                                                     map_prompt=prompt, combine_prompt=prompt,
                                                     return_intermediate_steps=return_intermediate_steps,
                                                     token_max=max_input_tokens, verbose=verbose)
            if async_output:
                chain_func = chain.arun
            else:
                chain_func = chain
            target = wrapped_partial(chain_func, dict(input_documents=docs,
                                                      token_max=max_input_tokens))  # , return_only_outputs=True)
        elif langchain_action == LangChainAction.SUMMARIZE_ALL.value:
            assert use_template
            prompt = PromptTemplate(input_variables=["text"], template=template)
            chain = load_general_summarization_chain(llm, chain_type="stuff", prompt=prompt,
                                                     return_intermediate_steps=return_intermediate_steps,
                                                     verbose=verbose)
            if async_output:
                chain_func = chain.arun
            else:
                chain_func = chain
            target = wrapped_partial(chain_func)
        elif langchain_action == LangChainAction.SUMMARIZE_REFINE.value:
            chain = load_general_summarization_chain(llm, chain_type="refine",
                                                     return_intermediate_steps=return_intermediate_steps,
                                                     verbose=verbose)
            if async_output:
                chain_func = chain.arun
            else:
                chain_func = chain
            target = wrapped_partial(chain_func)
        elif langchain_action == LangChainAction.EXTRACT.value:
            prompt = PromptTemplate(input_variables=["text"], template=template)
            chain = load_general_summarization_chain(llm, chain_type="map",
                                                     map_prompt=prompt, combine_prompt=prompt,
                                                     return_intermediate_steps=return_intermediate_steps,
                                                     token_max=max_input_tokens, verbose=verbose)
            if async_output:
                chain_func = chain.arun
            else:
                chain_func = chain
            target = wrapped_partial(chain_func, dict(input_documents=docs,
                                                      token_max=max_input_tokens))  # , return_only_outputs=True)
        else:
            raise RuntimeError("No such langchain_action=%s" % langchain_action)
    else:
        raise RuntimeError("No such langchain_action=%s" % langchain_action)

    return docs, target, scores, num_docs_before_cut, use_llm_if_no_docs, top_k_docs_max_show, \
        llm, model_name, streamer, prompt_type_out, async_output, only_new_text


def get_model_max_length(llm=None, tokenizer=None, inference_server=None, model_name=None):
    if hasattr(tokenizer, 'model_max_length'):
        return tokenizer.model_max_length
    elif inference_server in ['openai', 'openai_azure']:
        return llm.modelname_to_contextsize(model_name)
    elif inference_server in ['openai_chat', 'openai_azure_chat']:
        return model_token_mapping[model_name]
    elif isinstance(tokenizer, FakeTokenizer):
        # GGML
        return tokenizer.model_max_length
    else:
        return 2048


def get_max_input_tokens(llm=None, tokenizer=None, inference_server=None, model_name=None, max_new_tokens=None):
    model_max_length = get_model_max_length(llm=llm, tokenizer=tokenizer, inference_server=inference_server,
                                            model_name=model_name)

    if any([inference_server.startswith(x) for x in
            ['openai', 'openai_azure', 'openai_chat', 'openai_azure_chat', 'vllm']]):
        # openai can't handle tokens + max_new_tokens > max_tokens even if never generate those tokens
        # and vllm uses OpenAI API with same limits
        max_input_tokens = model_max_length - max_new_tokens
    elif isinstance(tokenizer, FakeTokenizer):
        # don't trust that fake tokenizer (e.g. GGUF/GGML) will make lots of tokens normally, allow more input
        max_input_tokens = model_max_length - min(256, max_new_tokens)
    else:
        # trust that maybe model will make so many tokens, so limit input
        max_input_tokens = model_max_length - max_new_tokens

    return max_input_tokens


def get_tokenizer(db=None, llm=None, tokenizer=None, inference_server=None, use_openai_model=False,
                  db_type='chroma'):
    if hasattr(llm, 'pipeline') and hasattr(llm.pipeline, 'tokenizer'):
        # more accurate
        return llm.pipeline.tokenizer
    elif hasattr(llm, 'tokenizer') and llm.tokenizer is not None:
        # e.g. TGI client mode etc.
        return llm.tokenizer
    elif inference_server and any([inference_server.startswith(x) for x in ['openai', 'openai_chat', 'openai_azure',
                                                                            'openai_azure_chat']]) and \
            tokenizer is not None:
        return tokenizer
    elif isinstance(tokenizer, FakeTokenizer):
        return tokenizer
    elif use_openai_model:
        return FakeTokenizer(is_openai=True)
    elif (hasattr(db, '_embedding_function') and
          hasattr(db._embedding_function, 'client') and
          hasattr(db._embedding_function.client, 'tokenize')):
        # in case model is not our pipeline with HF tokenizer
        return db._embedding_function.client.tokenize
    else:
        # backup method
        if os.getenv('HARD_ASSERTS'):
            assert db_type in ['faiss', 'weaviate', 'qdrant']
        # use tiktoken for faiss since embedding called differently
        return FakeTokenizer()


def escape_braces(text):
    if not isinstance(text, str):
        return text
    """Escapes braces in the text for safe formatting."""
    return text.replace("{", "{{").replace("}", "}}")


def get_template(query, iinput,
                 pre_prompt_query, prompt_query,
                 pre_prompt_summary, prompt_summary,
                 all_docs_start_prompt,
                 all_docs_finish_prompt,
                 langchain_action,
                 query_action,
                 summarize_action,
                 got_any_docs,
                 auto_reduce_chunks,
                 add_search_to_context,
                 system_prompt,
                 doc_json_mode,
                 model_name=None,
                 prompter=None):
    # Escape braces in the inputs that will be used in the format strings
    query_esc = escape_braces(query)
    iinput = escape_braces(iinput)
    prompt_summary = escape_braces(prompt_summary)
    prompt_query = escape_braces(prompt_query)
    pre_prompt_query = escape_braces(pre_prompt_query)
    pre_prompt_summary = escape_braces(pre_prompt_summary)

    if all_docs_start_prompt in ['auto', None] or all_docs_finish_prompt in ['auto', None]:
        if 'h2o-danube2' not in model_name and 'h2o-danube3' not in model_name and 'h2o-danube-' not in model_name:
            # NOTE: enabled generally for now, seems to help generally
            triple_quotes_start = """
<all_documents>
"""
            triple_quotes_finish = """
</all_documents>
"""
        else:
            triple_quotes_start = triple_quotes_finish = """
\"\"\"
"""
    else:
        triple_quotes_start = all_docs_start_prompt
        triple_quotes_finish = all_docs_finish_prompt

    if got_any_docs and add_search_to_context:
        # modify prompts, assumes patterns like in predefined prompts.  If user customizes, then they'd need to account for that.
        prompt_query = prompt_query.replace('information in the document sources',
                                            'information in the document and web search sources (and their source dates and website source)')
        prompt_summary = prompt_summary.replace('information in the document sources',
                                                'information in the document and web search sources (and their source dates and website source)')
    elif got_any_docs and not add_search_to_context:
        pass
    elif not got_any_docs and add_search_to_context:
        # modify prompts, assumes patterns like in predefined prompts.  If user customizes, then they'd need to account for that.
        prompt_query = prompt_query.replace('information in the document sources',
                                            'information in the web search sources (and their source dates and website source)')
        prompt_summary = prompt_summary.replace('information in the document sources',
                                                'information in the web search sources (and their source dates and website source)')

    if doc_json_mode:
        triple_quotes_start = triple_quotes_finish = '\n\n'
        question_fstring = """{{"question": "{question}".  Respond absolutely only in valid JSON.}}"""
        if got_any_docs:
            if query_action:
                system_prompt += '\n' + prompt_query
            if summarize_action:
                system_prompt += '\n' + prompt_summary
        prompt_query = pre_prompt_query = prompt_summary = pre_prompt_summary = ''

    else:
        question_fstring = """{question}"""

    if langchain_action == LangChainAction.QUERY.value:
        if iinput:
            query = "%s\n%s" % (query, iinput)
        if not got_any_docs:
            template_if_no_docs = template = """{context}%s""" % question_fstring
        else:
            fstring = "{context}"
            if prompter and prompter.prompt_type == 'docsgpt':
                sys_context = "\nSystem Instructions: %s" % system_prompt if system_prompt else "\n"
                template = """%s%s%s%s%s%s""" % (
                    question_fstring, "\n### Context\n", fstring, sys_context, '\n', '')
                sys_context_no_docs = '\n### Context%s' % sys_context if system_prompt else ''
                # {context} will be empty string, so ok that no new line surrounding it
                template_if_no_docs = """%s%s%s%s%s""" % (question_fstring, sys_context_no_docs, '', fstring, '')
            else:
                template = """%s%s%s%s%s\n%s""" % (
                    pre_prompt_query, triple_quotes_start, fstring, triple_quotes_finish, prompt_query,
                    question_fstring)
                if doc_json_mode:
                    template_if_no_docs = """{context}{{"question": {question}}}"""
                else:
                    template_if_no_docs = """{context}{question}"""
    elif langchain_action in [LangChainAction.SUMMARIZE_ALL.value, LangChainAction.SUMMARIZE_MAP.value,
                              LangChainAction.EXTRACT.value]:
        # modify prompt_summary if user passes query or iinput
        if query not in none and iinput not in none:
            prompt_summary = "Focusing on %s, %s, %s" % (query_esc, iinput, prompt_summary)
        elif query not in none:
            prompt_summary = "Focusing on %s, %s" % (query_esc, prompt_summary)
        # don't auto reduce
        auto_reduce_chunks = False
        if langchain_action in [LangChainAction.SUMMARIZE_MAP.value, LangChainAction.EXTRACT.value]:
            fstring = '{text}'
        else:
            fstring = '{input_documents}'
        # triple_quotes includes \n before """ and after """
        template = """%s%s%s%s%s\n""" % (
            pre_prompt_summary, triple_quotes_start, fstring, triple_quotes_finish, prompt_summary)
        template_if_no_docs = "Exactly only say: There are no documents to summarize/extract from."
    elif langchain_action in [LangChainAction.SUMMARIZE_REFINE]:
        template = ''  # unused
        template_if_no_docs = ''  # unused
    else:
        raise RuntimeError("No such langchain_action=%s" % langchain_action)

    return template, template_if_no_docs, auto_reduce_chunks, query


def get_hyde_acc(answer, llm_answers, hyde_show_intermediate_in_accordion, map_reduce_show_intermediate_in_accordion):
    if not isinstance(answer, str):
        return answer, None
    pre_answer = ''
    count = 0
    all_count = len(llm_answers)
    do_acc = hyde_show_intermediate_in_accordion and 'llm_answers_hyde_level_' in str(list(llm_answers.keys()))
    do_acc |= map_reduce_show_intermediate_in_accordion and 'map_reduce_' in str(list(llm_answers.keys()))
    do_acc |= map_reduce_show_intermediate_in_accordion and 'map_' in str(list(llm_answers.keys()))
    if llm_answers and do_acc:
        for title, content in llm_answers.items():
            if title == 'response_raw':
                count += 1
                continue
            if count + 1 == all_count:
                # skip one just generating or just generated.  Either not ready yet or final answer not in accordion
                count += 1
                continue
            # improve title for UI
            if 'llm_answers_hyde_level_0' == title:
                title = hyde_titles(0)
            elif 'llm_answers_hyde_level_1' == title:
                title = hyde_titles(1)
            elif 'llm_answers_hyde_level_2' == title:
                title = hyde_titles(2)
            elif 'llm_answers_hyde_level_3' == title:
                title = hyde_titles(3)
            elif 'llm_answers_hyde_level_4' == title:
                title = hyde_titles(4)
            elif 'map_reduce_' in title:
                title = 'Summarize Step %s' % title.split('map_reduce_')[1]
            elif 'map_' in title:
                title = 'Extraction Step %s' % title.split('map_')[1]
            pre_answer += get_accordion_named(content, title, font_size=3)
            count += 1

    if pre_answer and is_markdown(answer):
        answer = markdown_to_html(answer)

    return answer, pre_answer


def get_sources_answer(query, docs, answer,
                       llm_answers,
                       scores, show_rank,
                       answer_with_sources,
                       append_sources_to_answer,
                       append_sources_to_chat,
                       sources_show_text_in_accordion=True,
                       hyde_show_intermediate_in_accordion=True,
                       map_reduce_show_intermediate_in_accordion=True,
                       show_link_in_sources=True,
                       docs_ordering_type=None,
                       top_k_docs_max_show=10,
                       verbose=False,
                       t_run=None,
                       count_input_tokens=None, count_output_tokens=None):
    if verbose:
        print("query: %s" % query, flush=True)
        print("answer: %s" % answer, flush=True)

    answer, pre_answer = get_hyde_acc(answer, llm_answers, hyde_show_intermediate_in_accordion,
                                      map_reduce_show_intermediate_in_accordion)
    if pre_answer:
        pre_answer = pre_answer + '<br>'
        answer_with_acc = pre_answer + answer
    else:
        # e.g. extract goes here, list not str
        answer_with_acc = answer

    if len(docs) == 0:
        sources = []
        return answer_with_acc, sources, answer, ''

    sources = [dict(score=score, content=get_doc(x), source=get_source(x), orig_index=x.metadata.get('orig_index', 0))
               for score, x in zip(scores, docs)]
    if docs_ordering_type in ['best_first']:
        sources = sources[:top_k_docs_max_show]
    else:
        # sources as usually most important near prompt that comes last in sources list
        sources = sources[-top_k_docs_max_show:]
    if answer_with_sources == -1:
        sources_str = [str(x) for x in sources]
        sources_str = '\n'.join(sources_str)
        if append_sources_to_answer:
            ret = answer_with_acc + '\n\n' + sources_str
        else:
            ret = answer_with_acc
        return ret, sources, answer, sources_str

    # link
    answer_sources = [(max(0.0, 1.5 - score) / 1.5,
                       get_url(doc, font_size=font_size),
                       get_accordion(doc, font_size=font_size, head_acc=head_acc)) for score, doc in
                      zip(scores, docs)]
    if not sources_show_text_in_accordion:
        answer_sources_dict = defaultdict(list)
        [answer_sources_dict[url].append((score, accordion)) for score, url, accordion in answer_sources]
        answers_dict = {}
        for url, key in answer_sources_dict.items():
            scores_url = [x[0] for x in key]
            accordions = [x[1] for x in key]
            answers_dict[url] = (np.max(scores_url), accordions[0] if accordions else '')
        answer_sources = [(score, url, accordion) for url, (score, accordion) in answers_dict.items()]
    answer_sources.sort(key=lambda x: x[0], reverse=True)
    if show_rank:
        # answer_sources = ['%d | %s' % (1 + rank, url) for rank, (score, url) in enumerate(answer_sources)]
        # sorted_sources_urls = "Sources [Rank | Link]:<br>" + "<br>".join(answer_sources)
        answer_sources = ['%s' % url for rank, (score, url, _) in enumerate(answer_sources)]
        if docs_ordering_type in ['best_first']:
            answer_sources = answer_sources[:top_k_docs_max_show]
        else:
            answer_sources = answer_sources[-top_k_docs_max_show:]
        sorted_sources_urls = "Ranked Sources:<br>" + "<br>".join(answer_sources)
    else:
        if sources_show_text_in_accordion:
            if show_link_in_sources:
                answer_sources = ['<font size="%s"><li>%.2g | %s</li>%s</font>' % (font_size, score, url, accordion)
                                  for score, url, accordion in answer_sources]
            else:
                answer_sources = ['<font size="%s"><li>%.2g</li>%s</font>' % (font_size, score, accordion)
                                  for score, url, accordion in answer_sources]
        else:
            if show_link_in_sources:
                answer_sources = ['<font size="%s"><li>%.2g | %s</li></font>' % (font_size, score, url)
                                  for score, url, accordion in answer_sources]
            else:
                answer_sources = ['<font size="%s"><li>%.2g</li></font>' % (font_size, score)
                                  for score, url, accordion in answer_sources]
        if docs_ordering_type in ['best_first']:
            answer_sources = answer_sources[:top_k_docs_max_show]
        else:
            answer_sources = answer_sources[-top_k_docs_max_show:]
        if sources_show_text_in_accordion:
            sorted_sources_urls = f"<font size=\"{font_size}\">{source_prefix}<ul></font>" + "".join(answer_sources)
        else:
            sorted_sources_urls = f"<font size=\"{font_size}\">{source_prefix}<p><ul></font>" + "<p>".join(
                answer_sources)
        if verbose or True:
            if t_run is not None and int(t_run) > 0:
                sorted_sources_urls += 'Total Time: %d [s]<br>' % t_run
            if count_input_tokens and count_output_tokens:
                sorted_sources_urls += 'Input Tokens: %s | Output Tokens: %d<br>' % (
                    count_input_tokens, count_output_tokens)
        sorted_sources_urls += "Total document chunks used: %s<br>" % len(docs)
        sorted_sources_urls += f"<font size=\"{font_size}\"></ul></p>{source_postfix}</font>"
        title_overall = "Sources"
        sorted_sources_urls = f"""<details><summary><font size="{font_size}">{title_overall}</font></summary><font size="{font_size}">{sorted_sources_urls}</font></details>"""
        if os.getenv("HARD_ASSERTS"):
            assert sorted_sources_urls.startswith(super_source_prefix)
            assert sorted_sources_urls.endswith(super_source_postfix)

    if isinstance(answer, str) and not answer.endswith('\n'):
        answer += '\n'
    if isinstance(answer_with_acc, str) and not answer_with_acc.endswith('\n'):
        answer_with_acc += '\n'

    answer_no_refs = answer
    if answer_with_sources:
        sources_str = '\n' + sorted_sources_urls
    else:
        sources_str = ''
    if isinstance(answer_with_acc, str) and append_sources_to_answer:
        ret = answer_with_acc + sources_str
    else:
        ret = answer_with_acc
    return ret, sources, answer_no_refs, sources_str


def get_any_db(db1s, langchain_mode, langchain_mode_paths, langchain_mode_types,
               dbs=None,
               load_db_if_exists=None, db_type=None,
               use_openai_embedding=None,
               hf_embedding_model=None, migrate_embedding_model=None,
               for_sources_list=False,
               verbose=False,
               n_jobs=-1,
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
                            verbose=verbose, n_jobs=n_jobs)
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
                n_jobs=-1,
                ):
    db = None
    if langchain_mode in ['LLM', 'Disabled']:
        source_files_added = "NA"
        source_list = []
        num_chunks = 0
        num_sources_str = str(0)
    elif langchain_mode in ['wiki_full']:
        source_files_added = "Not showing wiki_full, takes about 20 seconds and makes 4MB file." \
                             "  Ask jon.mckinney@h2o.ai for file if required."
        source_list = []
        num_chunks = 0
        num_sources_str = str(0)
    else:
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
                        n_jobs=n_jobs,
                        )
        if db is not None:
            metadatas = get_metadatas(db, full_required=False)
            metadatas_sources = [x['source'] for x in metadatas if not x.get('exception', '')]
            exception_metadatas_sources = [x['source'] for x in metadatas if x.get('exception', '')]
            source_list = sorted(set(metadatas_sources))
            source_files_added = '\n'.join(source_list)
            num_chunks = len(metadatas_sources)
            num_sources_str = ">=%d" % len(source_list)
            if is_chroma_db(db):
                num_chunks_real = db._collection.count()  # includes exceptions
                num_chunks_real -= len(exception_metadatas_sources)  # exclude exceptions
                if num_chunks_real == num_chunks:
                    num_sources_str = "=%d" % len(source_list)
                else:
                    num_chunks = num_chunks_real
        else:
            source_list = []
            source_files_added = "None"
            num_chunks = 0
            num_sources_str = str(0)
    sources_file = make_sources_file(langchain_mode, source_files_added)
    source_list = docs_state0 + source_list
    if DocumentChoice.ALL.value in source_list:
        source_list.remove(DocumentChoice.ALL.value)
    return sources_file, source_list, num_chunks, num_sources_str, db


def update_user_db(file, db1s, selection_docs_state1, requests_state1,
                   langchain_mode=None,
                   get_userid_auth=None,
                   **kwargs):
    if file is None:
        raise RuntimeError("Don't use change, use input")

    # can't do below, langchain_mode can change, e.g. LLM -> MyData and UI will reflect
    # if langchain_mode in ['LLM', 'Disabled']:
    #    return None, langchain_mode, "", "", None, None

    kwargs.update(selection_docs_state1)
    set_userid(db1s, requests_state1, get_userid_auth)

    try:
        return _update_user_db(file, db1s=db1s,
                               langchain_mode=langchain_mode,
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
        return None, langchain_mode, source_files_added, doc_exception_text, None, None
    finally:
        clear_torch_cache(allow_skip=True)


def get_lock_file(db1, langchain_mode):
    db_id = get_dbid(db1)
    base_path = 'locks'
    base_path = makedirs(base_path, exist_ok=True, tmp_ok=True, use_base=True)
    # don't allow db_id to be '' or None, would be bug and lock up everything
    if not db_id:
        if os.getenv('HARD_ASSERTS'):
            raise ValueError("Invalid access for langchain_mode=%s" % langchain_mode)
        db_id = str(uuid.uuid4())
    lock_file = os.path.join(base_path, "db_%s_%s.lock" % (langchain_mode.replace(' ', '_').replace('/', '_'), db_id))
    makedirs(os.path.dirname(lock_file))  # ensure really made
    return lock_file


def _update_user_db(file,
                    db1s=None,
                    langchain_mode='UserData',
                    chunk=None, chunk_size=None,

                    # urls
                    use_unstructured=True,
                    use_playwright=False,
                    use_selenium=False,
                    use_scrapeplaywright=False,
                    use_scrapehttp=False,

                    # pdfs
                    use_pymupdf='auto',
                    use_unstructured_pdf='auto',
                    use_pypdf='auto',
                    enable_pdf_ocr='auto',
                    enable_pdf_doctr='auto',
                    try_pdf_as_html='auto',

                    # images
                    enable_ocr=False,
                    enable_doctr=False,
                    enable_pix2struct=False,
                    enable_captions=True,
                    enable_llava=True,
                    enable_transcriptions=True,
                    captions_model=None,
                    caption_loader=None,
                    doctr_loader=None,
                    pix2struct_loader=None,
                    llava_model=None,
                    llava_prompt=None,
                    asr_model=None,
                    asr_loader=None,

                    # json
                    jq_schema='.[]',
                    extract_frames=10,

                    dbs=None, db_type=None,
                    langchain_modes=None,
                    langchain_mode_paths=None,
                    langchain_mode_types=None,
                    use_openai_embedding=None,
                    hf_embedding_model=None,
                    migrate_embedding_model=None,
                    verbose=None,
                    n_jobs=-1,
                    is_url=None, is_txt=None,
                    is_public=False,
                    from_ui=False,

                    gradio_upload_to_chatbot_num_max=None,

                    allow_upload_to_my_data=None,
                    allow_upload_to_user_data=None,

                    function_server: bool = False,
                    function_server_port: int = None,
                    function_api_key: str = 'EMPTY',
                    ):
    assert db1s is not None
    assert chunk is not None
    assert chunk_size is not None
    assert use_openai_embedding is not None
    assert hf_embedding_model is not None
    assert migrate_embedding_model is not None
    assert caption_loader is not None
    assert asr_loader is not None
    assert doctr_loader is not None
    assert enable_captions is not None
    assert enable_transcriptions is not None
    assert captions_model is not None
    assert asr_model is not None
    assert enable_ocr is not None
    assert enable_doctr is not None
    assert enable_pdf_ocr is not None
    assert enable_pdf_doctr is not None
    assert enable_pix2struct is not None
    assert enable_llava is not None
    assert verbose is not None
    assert gradio_upload_to_chatbot_num_max is not None
    assert allow_upload_to_my_data is not None
    assert allow_upload_to_user_data is not None

    if dbs is None:
        dbs = {}
    assert isinstance(dbs, dict), "Wrong type for dbs: %s" % str(type(dbs))

    if langchain_mode is not None:
        in_scratch_db = langchain_mode in db1s
        in_user_db = dbs is not None and langchain_mode in dbs
        if in_scratch_db and not allow_upload_to_my_data:
            raise ValueError("Not allowed to upload to scratch/personal space")
        elif in_user_db and not allow_upload_to_user_data:
            raise ValueError("Not allowed to upload to shared space")
        elif langchain_mode_types and langchain_mode in langchain_mode_types and langchain_mode_types[
            langchain_mode] in [LangChainTypes.SHARED.value] and not allow_upload_to_user_data:
            raise ValueError("Not allowed to upload to shared space")
        elif langchain_mode_types and langchain_mode in langchain_mode_types and langchain_mode_types[
            langchain_mode] in [LangChainTypes.PERSONAL.value] and not allow_upload_to_my_data:
            raise ValueError("Not allowed to upload to scratch/personal space")

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

    if is_public:
        if len(file) > max_docs_public and from_ui or \
                len(file) > max_docs_public_api and not from_ui:
            raise ValueError("Public instance only allows up to"
                             " %d (%d from API) documents updated at a time." % (max_docs_public, max_docs_public_api))

    add_text_called = is_txt and is_url is False
    # Upload file button = add_file has is_txt=is_url=None
    # Ingest button = add_url is True and add_text may be True or False (that is used for add_button for any text, url, or file)
    if not add_text_called:
        # is_url will do extra checks of if good url, want to avoid if just text
        # if file, also want to avoid if possible extra checks
        if is_url is None and file:
            # assume add_button action if not set
            is_url = True
        if isinstance(file, str) and os.path.isfile(file):
            is_url = False
        if isinstance(file, list) and len(file) > 0 and all(os.path.isfile(x) for x in file):
            is_url = False

    if langchain_mode == LangChainMode.DISABLED.value:
        return None, langchain_mode, get_source_files(), "", None, {}

    if langchain_mode in [LangChainMode.LLM.value]:
        # then switch to MyData, so langchain_mode also becomes way to select where upload goes
        # but default to mydata if nothing chosen, since safest
        if LangChainMode.MY_DATA.value in langchain_modes:
            langchain_mode = LangChainMode.MY_DATA.value
        elif len(langchain_modes) >= 1:
            langchain_mode = langchain_modes[0]
        else:
            return None, langchain_mode, get_source_files(), "", None, {}

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
                        if os.path.dirname(new_fil):
                            makedirs(os.path.dirname(new_fil))
                        shutil.move(fil, new_fil)
                    except FileExistsError:
                        pass
                    file[fili] = new_fil

    if verbose:
        print("Adding %s" % file, flush=True)

    # FIXME: could avoid even parsing, let alone embedding, same old files if upload same file again
    # FIXME: but assume nominally user isn't uploading all files over again from UI

    # expect string comparison, if dict then model object with name and get name not dict or model
    hf_embedding_model_str = get_hf_embedding_model_name(hf_embedding_model)
    if not is_url and is_txt and hf_embedding_model_str == 'fake':
        # avoid parallel if fake embedding since assume trivial ingestion
        n_jobs = 1

    complex_kwargs = dict(
        captions_model=captions_model,
        caption_loader=caption_loader,
        doctr_loader=doctr_loader,
        pix2struct_loader=pix2struct_loader,
        llava_model=llava_model,
        asr_model=asr_model,
        asr_loader=asr_loader,

        hf_embedding_model=hf_embedding_model,
    )
    simple_kwargs = dict(
        url=file if is_url else None,
        text=file if is_txt else None,

        # images
        enable_ocr=enable_ocr,
        enable_doctr=enable_doctr,
        enable_pix2struct=enable_pix2struct,
        enable_captions=enable_captions,
        enable_llava=enable_llava,
        enable_transcriptions=enable_transcriptions,
        llava_prompt=llava_prompt,

        # urls
        use_unstructured=use_unstructured,
        use_playwright=use_playwright,
        use_selenium=use_selenium,
        use_scrapeplaywright=use_scrapeplaywright,
        use_scrapehttp=use_scrapehttp,

        # pdfs
        use_pymupdf=use_pymupdf,
        use_unstructured_pdf=use_unstructured_pdf,
        use_pypdf=use_pypdf,
        enable_pdf_ocr=enable_pdf_ocr,
        enable_pdf_doctr=enable_pdf_doctr,
        try_pdf_as_html=try_pdf_as_html,

        # json
        jq_schema=jq_schema,
        extract_frames=extract_frames,

        db_type=db_type,

        is_public=is_public,
        from_ui=from_ui,

        use_openai_embedding=use_openai_embedding,
        verbose=verbose,
        fail_any_exception=False,
        n_jobs=n_jobs,
        chunk=chunk, chunk_size=chunk_size,
    )

    args = (file if not is_url and not is_txt else None,)

    if function_server:
        from function_client import call_function_server
        sources = call_function_server('0.0.0.0', function_server_port, 'path_to_docs', (file,), simple_kwargs,
                                       use_disk=True, use_pickle=True,
                                       function_api_key=function_api_key,
                                       verbose=verbose)
    else:
        sources = path_to_docs(*args,
                               **simple_kwargs,
                               **complex_kwargs,
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
    lock_func = filelock.FileLock if db1[0] and hasattr(db1[0], '_persist_directory') else NullContext
    with lock_func(lock_file):
        if langchain_mode in db1s:
            if db1[0] is not None:
                # then add
                db, num_new_sources, new_sources_metadata = add_to_db(db1[0], sources, db_type=db_type,
                                                                      use_openai_embedding=use_openai_embedding,
                                                                      hf_embedding_model=hf_embedding_model,
                                                                      verbose=verbose)
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
                            migrate_embedding_model=migrate_embedding_model,
                            n_jobs=n_jobs,
                            verbose=verbose)
            if db is not None:
                db1[0] = db
            source_files_added = get_source_files(db=db1[0], exceptions=exceptions)
            if len(sources) > 0:
                sources_last = os.path.basename(sources[-1].metadata.get('source', 'Unknown Source'))
                all_sources_last_dict = get_all_sources_last_dict(sources, gradio_upload_to_chatbot_num_max)
            else:
                sources_last = None
                all_sources_last_dict = {}
            return None, langchain_mode, source_files_added, '\n'.join(
                exceptions_strs), sources_last, all_sources_last_dict
        else:
            langchain_type = langchain_mode_types.get(langchain_mode, LangChainTypes.EITHER.value)
            persist_directory, langchain_type = get_persist_directory(langchain_mode, db1s=db1s, dbs=dbs,
                                                                      langchain_type=langchain_type)
            langchain_mode_types[langchain_mode] = langchain_type
            if not persist_directory:
                raise ValueError("Switch to valid Collection, not %s" % langchain_mode)
            elif langchain_mode in dbs and dbs[langchain_mode] is not None:
                # then add
                db, num_new_sources, new_sources_metadata = add_to_db(dbs[langchain_mode], sources, db_type=db_type,
                                                                      use_openai_embedding=use_openai_embedding,
                                                                      hf_embedding_model=hf_embedding_model,
                                                                      verbose=verbose)
            else:
                # then create.  Or might just be that dbs is unfilled, then it will fill, then add
                db = get_db(sources, use_openai_embedding=use_openai_embedding,
                            db_type=db_type,
                            persist_directory=persist_directory,
                            langchain_mode=langchain_mode,
                            langchain_mode_paths=langchain_mode_paths,
                            langchain_mode_types=langchain_mode_types,
                            hf_embedding_model=hf_embedding_model,
                            migrate_embedding_model=migrate_embedding_model,
                            n_jobs=n_jobs,
                            verbose=verbose)
            dbs[langchain_mode] = db
            # NOTE we do not return db, because function call always same code path
            # return dbs[langchain_mode]
            # db in this code path is updated in place
            source_files_added = get_source_files(db=dbs[langchain_mode], exceptions=exceptions)
            if len(sources) > 0:
                sources_last = os.path.basename(sources[-1].metadata.get('source', 'Unknown Source'))
                all_sources_last_dict = get_all_sources_last_dict(sources, gradio_upload_to_chatbot_num_max)
            else:
                sources_last = None
                all_sources_last_dict = {}
            return None, langchain_mode, source_files_added, '\n'.join(
                exceptions_strs), sources_last, all_sources_last_dict


def get_all_sources_last_dict(sources, gradio_upload_to_chatbot_num_max):
    valid_sources = [x for x in sources if
                     x.metadata.get('source', '') and x.page_content and x.metadata.get('chunk_id', -1) == -1]
    # FIXME: Choose longest output if multiple?

    # only what can be shown in gradio
    allowed_types = image_types + audio_types
    valid_sources = [x for x in valid_sources if any(x.metadata['source'].endswith(y) for y in allowed_types)]

    all_sources_last_dict = {x.metadata['source']: x.page_content
                             for x in valid_sources[:gradio_upload_to_chatbot_num_max]}
    return all_sources_last_dict


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
                                          delete_sources=False,
                                          n_jobs=-1):
    if langchain_mode in ['LLM', 'Disabled']:
        return "Sources: N/A"

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
                    n_jobs=n_jobs,
                    )
    if delete_sources:
        del_from_db(db, document_choice1, db_type=db_type)

    if db is None:
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
            metadatas = get_metadatas(db, full_required=False)
        else:
            metadatas = []
        adding_new = False
    else:
        source_label = "New Sources:"
        adding_new = True

    # below automatically de-dups
    # non-exception cases only
    small_dict = dict()
    for page_0 in [1, 0]:
        small_dict.update({get_url(x['source'], from_str=True, short_name=True): get_short_name(x.get('head')) for x in
                           metadatas if x.get('page', 0) in [page_0] and not x.get('exception', '')})
    # if small_dict is empty dict, that's ok
    df = pd.DataFrame(small_dict.items(), columns=['source', 'head'])
    df.index = df.index + 1
    df.index.name = 'index'
    source_files_added = tabulate.tabulate(df, headers='keys', tablefmt='unsafehtml')

    no_exception_metadatas = [x for x in metadatas if not x.get('exception')]

    if not exceptions:
        # auto-get exceptions
        exception_metadatas = [x for x in metadatas if x.get('exception')]
    else:
        exception_metadatas = [x.metadata for x in exceptions]

    if exception_metadatas:
        small_dict = {get_url(x['source'], from_str=True, short_name=True): get_short_name(x.get('exception')) for x in
                      exception_metadatas}
        # if small_dict is empty dict, that's ok
        df = pd.DataFrame(small_dict.items(), columns=['source', 'exception'])
        df.index = df.index + 1
        df.index.name = 'index'
        exceptions_html = tabulate.tabulate(df, headers='keys', tablefmt='unsafehtml')
    else:
        exceptions_html = ''

    if no_exception_metadatas and exception_metadatas:
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
    elif no_exception_metadatas:
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

                                                     # urls
                                                     use_unstructured=True,
                                                     use_playwright=False,
                                                     use_selenium=False,
                                                     use_scrapeplaywright=False,
                                                     use_scrapehttp=False,

                                                     # pdfs
                                                     use_pymupdf='auto',
                                                     use_unstructured_pdf='auto',
                                                     use_pypdf='auto',
                                                     enable_pdf_ocr='auto',
                                                     enable_pdf_doctr='auto',
                                                     try_pdf_as_html='auto',

                                                     # images
                                                     enable_ocr=False,
                                                     enable_doctr=False,
                                                     enable_pix2struct=False,
                                                     enable_captions=True,
                                                     enable_llava=True,
                                                     enable_transcriptions=True,
                                                     captions_model=None,
                                                     caption_loader=None,
                                                     doctr_loader=None,
                                                     pix2struct_loader=None,
                                                     llava_model=None,
                                                     llava_prompt=None,
                                                     asr_model=None,
                                                     asr_loader=None,

                                                     # json
                                                     jq_schema='.[]',
                                                     extract_frames=10,

                                                     dbs=None, first_para=None,
                                                     hf_embedding_model=None,
                                                     use_openai_embedding=None,
                                                     migrate_embedding_model=None,
                                                     text_limit=None,
                                                     db_type=None, load_db_if_exists=None,
                                                     n_jobs=None, verbose=None, get_userid_auth=None):
    if langchain_mode in ['LLM', 'Disabled']:
        return get_source_files(db=None, exceptions=None, metadatas=None)

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
                    n_jobs=n_jobs,
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

                                                        # urls
                                                        use_unstructured=use_unstructured,
                                                        use_playwright=use_playwright,
                                                        use_selenium=use_selenium,
                                                        use_scrapeplaywright=use_scrapeplaywright,
                                                        use_scrapehttp=use_scrapehttp,

                                                        # pdfs
                                                        use_pymupdf=use_pymupdf,
                                                        use_unstructured_pdf=use_unstructured_pdf,
                                                        use_pypdf=use_pypdf,
                                                        enable_pdf_ocr=enable_pdf_ocr,
                                                        enable_pdf_doctr=enable_pdf_doctr,
                                                        try_pdf_as_html=try_pdf_as_html,

                                                        # images
                                                        enable_ocr=enable_ocr,
                                                        enable_doctr=enable_doctr,
                                                        enable_pix2struct=enable_pix2struct,
                                                        enable_captions=enable_captions,
                                                        enable_llava=enable_llava,
                                                        enable_transcriptions=enable_transcriptions,
                                                        captions_model=captions_model,
                                                        caption_loader=caption_loader,
                                                        doctr_loader=doctr_loader,
                                                        pix2struct_loader=pix2struct_loader,
                                                        llava_model=llava_model,
                                                        llava_prompt=llava_prompt,
                                                        asr_model=asr_model,
                                                        asr_loader=asr_loader,

                                                        # json
                                                        jq_schema=jq_schema,
                                                        extract_frames=extract_frames,

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
            assert os.path.isfile(
                os.path.join(dest, dir_expected, 'chroma.sqlite3')), "Missing db in %s" % dir_expected


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


def _get_qdrant_options():
    env_vars = os.environ.keys()

    qdrant_env_vars = [var for var in env_vars if var.startswith("QDRANT_")]

    if len(qdrant_env_vars) == 0:
        return None

    options = {
        "url": os.getenv("QDRANT_URL", None),
        "host": os.getenv("QDRANT_HOST", None),
        "port": int(os.getenv("QDRANT_PORT", 6333)),
        "grpc_port": int(os.getenv("QDRANT_GRPC_PORT", 6334)),
        "prefer_grpc": bool(os.getenv("QDRANT_PREFER_GRPC", False)),
        "https": bool(os.getenv("QDRANT_HTTPS", None)),
        "api_key": os.getenv("QDRANT_API_KEY", None),
        "prefix": os.getenv("QDRANT_PREFIX", None),
        "timeout": float(os.getenv("QDRANT_TIMEOUT", None)),
        "path": os.getenv("QDRANT_PATH", None),
    }

    return options


def _get_unique_sources_in_qdrant(db):
    from langchain_community.vectorstores import Qdrant
    import grpc

    if not isinstance(db, Qdrant):
        raise ValueError("db must be an instance of langchain.vectorstores.Qdrant")

    sources = []
    next_offset = None
    stop_scrolling = False
    scroll_size = 1000

    while not stop_scrolling:
        records, next_offset = db.client.scroll(
            collection_name=db.collection_name,
            limit=scroll_size,
            offset=next_offset,
            with_payload=True,
        )

        # Qdrant client supports a REST and GPRC interface. So we need to handle the response differently
        stop_scrolling = next_offset is None or (
                isinstance(next_offset, grpc.PointId)
                and next_offset.num == 0
                and next_offset.uuid == ""
        )

        sources.extend(records)
    unique_sources = {source.payload["metadata"]["source"] for source in sources}
    return unique_sources


if __name__ == '__main__':
    pass
