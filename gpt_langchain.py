import glob
import inspect
import os
import pathlib
import pickle
import queue
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
import zipfile
from collections import defaultdict
from datetime import datetime
from functools import reduce
from operator import concat

from joblib import Parallel, delayed
from tqdm import tqdm

from prompter import non_hf_types
from utils import wrapped_partial, EThread, import_matplotlib, sanitize_filename, makedirs, get_url, flatten_list, \
    get_device, ProgressParallel, remove

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
    UnstructuredEPubLoader, UnstructuredImageLoader, UnstructuredRTFLoader, ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain import PromptTemplate
from langchain.vectorstores import Chroma


def get_db(sources, use_openai_embedding=False, db_type='faiss', persist_directory="db_dir", langchain_mode='notset',
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

        # TODO: add support for connecting via docker compose
        client = weaviate.Client(
            embedded_options=EmbeddedOptions()
        )
        index_name = collection_name.capitalize()
        db = Weaviate.from_documents(documents=sources, embedding=embedding, client=client, by_text=False,
                                     index_name=index_name)

    elif db_type == 'chroma':
        assert persist_directory is not None
        os.makedirs(persist_directory, exist_ok=True)
        db = Chroma.from_documents(documents=sources,
                                   embedding=embedding,
                                   persist_directory=persist_directory,
                                   collection_name=collection_name,
                                   anonymized_telemetry=False)
        db.persist()
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


def add_to_db(db, sources, db_type='faiss', avoid_dup=True):
    if not sources:
        return db
    if db_type == 'faiss':
        db.add_documents(sources)
    elif db_type == 'weaviate':
        if avoid_dup:
            unique_sources = _get_unique_sources_in_weaviate(db)
            sources = [x for x in sources if x.metadata['source'] not in unique_sources]
        if len(sources) == 0:
            return db
        db.add_documents(documents=sources)

    elif db_type == 'chroma':
        if avoid_dup:
            collection = db.get()
            metadata_sources = set([x['source'] for x in collection['metadatas']])
            sources = [x for x in sources if x.metadata['source'] not in metadata_sources]
        if len(sources) == 0:
            return db
        db.add_documents(documents=sources)
        db.persist()
    else:
        raise RuntimeError("No such db_type=%s" % db_type)

    return db


def create_or_update_db(db_type, persist_directory, collection_name,
                        sources, use_openai_embedding, add_if_exists, verbose, hf_embedding_model):
    if db_type == 'weaviate':
        import weaviate
        from weaviate.embedded import EmbeddedOptions

        # TODO: add support for connecting via docker compose
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
        embedding = OpenAIEmbeddings()
    else:
        # to ensure can fork without deadlock
        from langchain.embeddings import HuggingFaceEmbeddings

        device, torch_dtype, context_class = get_device_dtype()
        model_kwargs = dict(device=device)
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


def get_llm(use_openai_model=False, model_name=None, model=None,
            tokenizer=None, stream_output=False,
            max_new_tokens=256,
            temperature=0.1,
            repetition_penalty=1.0,
            top_k=40,
            top_p=0.7,
            prompt_type=None,
            prompter=None,
            verbose=False,
            ):
    if use_openai_model:
        from langchain.llms import OpenAI
        llm = OpenAI(temperature=0)
        model_name = 'openai'
        streamer = None
        prompt_type = 'plain'
    elif model_name in non_hf_types:
        from gpt4all_llm import get_llm_gpt4all
        llm = get_llm_gpt4all(model_name, model=model, max_new_tokens=max_new_tokens,
                              temperature=temperature,
                              repetition_penalty=repetition_penalty,
                              top_k=top_k,
                              top_p=top_p,
                              verbose=verbose,
                              )
        streamer = None
        prompt_type = 'plain'
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if model is None:
            # only used if didn't pass model in
            assert model_name is None
            assert tokenizer is None
            prompt_type = 'human_bot'
            model_name = 'h2oai/h2ogpt-oasst1-512-12b'
            # model_name = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
            # model_name = 'h2oai/h2ogpt-oasst1-512-20b'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            device, torch_dtype, context_class = get_device_dtype()

            with context_class(device):
                load_8bit = True
                # FIXME: for now not to spread across hetero GPUs
                # device_map={"": 0} if load_8bit and device == 'cuda' else "auto"
                device_map = {"": 0} if device == 'cuda' else "auto"
                model = AutoModelForCausalLM.from_pretrained(model_name,
                                                             device_map=device_map,
                                                             torch_dtype=torch_dtype,
                                                             load_in_8bit=load_8bit)

        max_max_tokens = tokenizer.model_max_length
        gen_kwargs = dict(max_new_tokens=max_new_tokens,
                          return_full_text=True,
                          early_stopping=False,
                          handle_long_generation='hole')

        if stream_output:
            skip_prompt = False
            from generate import H2OTextIteratorStreamer
            decoder_kwargs = {}
            streamer = H2OTextIteratorStreamer(tokenizer, skip_prompt=skip_prompt, block=False, **decoder_kwargs)
            gen_kwargs.update(dict(streamer=streamer))
        else:
            streamer = None

        from h2oai_pipeline import H2OTextGenerationPipeline
        pipe = H2OTextGenerationPipeline(model=model, use_prompter=True,
                                         prompter=prompter,
                                         prompt_type=prompt_type,
                                         sanitize_bot_response=True,
                                         chat=False, stream_output=stream_output,
                                         tokenizer=tokenizer,
                                         max_input_tokens=max_max_tokens - max_new_tokens,
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
        page_content = page_content[:text_limit] if take_head else page_content[:-text_limit]
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


import distutils.spawn

have_tesseract = distutils.spawn.find_executable("tesseract")
have_libreoffice = distutils.spawn.find_executable("libreoffice")

import pkg_resources

try:
    assert pkg_resources.get_distribution('arxiv') is not None
    assert pkg_resources.get_distribution('pymupdf') is not None
    have_arxiv = True
except (pkg_resources.DistributionNotFound, AssertionError):
    have_arxiv = False

image_types = ["png", "jpg", "jpeg"]
non_image_types = ["pdf", "txt", "csv", "toml", "py", "rst", "rtf",
                   "md", "html",
                   "enex", "eml", "epub", "odt", "pptx", "ppt",
                   "zip", "urls",
                   ]
# "msg",  GPL3

if have_libreoffice:
    non_image_types.extend(["docx", "doc"])

file_types = non_image_types + image_types


def add_meta(docs1, file):
    file_extension = pathlib.Path(file).suffix
    if not isinstance(docs1, list):
        docs1 = [docs1]
    [x.metadata.update(dict(input_type=file_extension, date=str(datetime.now))) for x in docs1]


def file_to_doc(file, base_path=None, verbose=False, fail_any_exception=False, chunk=True, chunk_size=512,
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
            docs1 = UnstructuredURLLoader(urls=[file]).load()
            [x.metadata.update(dict(input_type='url', date=str(datetime.now))) for x in docs1]
        doc1 = chunk_sources(docs1, chunk_size=chunk_size)
    elif is_txt:
        base_path = "user_paste"
        source_file = os.path.join(base_path, "_%s" % str(uuid.uuid4())[:10])
        makedirs(os.path.dirname(source_file), exist_ok=True)
        with open(source_file, "wt") as f:
            f.write(file)
        metadata = dict(source=source_file, date=str(datetime.now()), input_type='pasted txt')
        doc1 = Document(page_content=file, metadata=metadata)
    elif file.lower().endswith('.html') or file.lower().endswith('.mhtml'):
        docs1 = UnstructuredHTMLLoader(file_path=file).load()
        add_meta(docs1, file)
        doc1 = chunk_sources(docs1, chunk_size=chunk_size)
    elif (file.lower().endswith('.docx') or file.lower().endswith('.doc')) and have_libreoffice:
        docs1 = UnstructuredWordDocumentLoader(file_path=file).load()
        add_meta(docs1, file)
        doc1 = chunk_sources(docs1, chunk_size=chunk_size)
    elif file.lower().endswith('.odt'):
        docs1 = UnstructuredODTLoader(file_path=file).load()
        add_meta(docs1, file)
        doc1 = chunk_sources(docs1, chunk_size=chunk_size)
    elif file.lower().endswith('pptx') or file.lower().endswith('ppt'):
        docs1 = UnstructuredPowerPointLoader(file_path=file).load()
        add_meta(docs1, file)
        doc1 = chunk_sources(docs1, chunk_size=chunk_size)
    elif file.lower().endswith('.txt'):
        # use UnstructuredFileLoader ?
        docs1 = TextLoader(file, encoding="utf8", autodetect_encoding=True).load()
        # makes just one, but big one
        doc1 = chunk_sources(docs1, chunk_size=chunk_size)
        add_meta(doc1, file)
    elif file.lower().endswith('.rtf'):
        docs1 = UnstructuredRTFLoader(file).load()
        add_meta(docs1, file)
        doc1 = chunk_sources(docs1, chunk_size=chunk_size)
    elif file.lower().endswith('.md'):
        docs1 = UnstructuredMarkdownLoader(file).load()
        add_meta(docs1, file)
        doc1 = chunk_sources(docs1, chunk_size=chunk_size)
    elif file.lower().endswith('.enex'):
        docs1 = EverNoteLoader(file).load()
        add_meta(doc1, file)
        doc1 = chunk_sources(docs1, chunk_size=chunk_size)
    elif file.lower().endswith('.epub'):
        docs1 = UnstructuredEPubLoader(file).load()
        add_meta(docs1, file)
        doc1 = chunk_sources(docs1, chunk_size=chunk_size)
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
            if docs1:
                doc1 = chunk_sources(docs1, chunk_size=chunk_size)
    elif file.lower().endswith('.msg'):
        raise RuntimeError("Not supported, GPL3 license")
        # docs1 = OutlookMessageLoader(file).load()
        # docs1[0].metadata['source'] = file
    elif file.lower().endswith('.eml'):
        try:
            docs1 = UnstructuredEmailLoader(file).load()
            add_meta(docs1, file)
            doc1 = chunk_sources(docs1, chunk_size=chunk_size)
        except ValueError as e:
            if 'text/html content not found in email' in str(e):
                # e.g. plain/text dict key exists, but not
                # doc1 = TextLoader(file, encoding="utf8").load()
                docs1 = UnstructuredEmailLoader(file, content_source="text/plain").load()
                add_meta(docs1, file)
                doc1 = chunk_sources(docs1, chunk_size=chunk_size)
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
    elif file.lower().endswith('.pdf'):
        # Some PDFs return nothing or junk from PDFMinerLoader
        # e.g. Beyond fine-tuning_ Classifying high resolution mammograms using function-preserving transformations _ Elsevier Enhanced Reader.pdf
        doc1 = PyPDFLoader(file).load_and_split()
        add_meta(doc1, file)
    elif file.lower().endswith('.csv'):
        doc1 = CSVLoader(file).load()
        add_meta(doc1, file)
    elif file.lower().endswith('.py'):
        doc1 = PythonLoader(file).load()
        add_meta(doc1, file)
    elif file.lower().endswith('.toml'):
        doc1 = TomlLoader(file).load()
        add_meta(doc1, file)
    elif file.lower().endswith('.urls'):
        with open(file, "r") as f:
            docs1 = UnstructuredURLLoader(urls=f.readlines()).load()
        add_meta(docs1, file)
        doc1 = chunk_sources(docs1, chunk_size=chunk_size)
    elif file.lower().endswith('.zip'):
        with zipfile.ZipFile(file, 'r') as zip_ref:
            # don't put into temporary path, since want to keep references to docs inside zip
            # so just extract in path where
            zip_ref.extractall(base_path)
            # recurse
            doc1 = path_to_docs(base_path, verbose=verbose, fail_any_exception=fail_any_exception)
    else:
        raise RuntimeError("No file handler for %s" % os.path.basename(file))

    # allow doc1 to be list or not.  If not list, did not chunk yet, so chunk now
    # if list of length one, don't trust and chunk it
    if not isinstance(doc1, list):
        if chunk:
            docs = chunk_sources([doc1], chunk_size=chunk_size)
        else:
            docs = [doc1]
    elif isinstance(doc1, list) and len(doc1) == 1:
        if chunk:
            docs = chunk_sources(doc1, chunk_size=chunk_size)
        else:
            docs = doc1
    else:
        docs = doc1

    assert isinstance(docs, list)
    return docs


def path_to_doc1(file, verbose=False, fail_any_exception=False, return_file=True, chunk=True, chunk_size=512,
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
                metadata={"source": file, "exception": str(e), "traceback": traceback.format_exc()})
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
                 ):
    globs_image_types = []
    globs_non_image_types = []
    if not path_or_paths and not url and not text:
        return []
    elif url:
        globs_non_image_types = [url]
    elif text:
        globs_non_image_types = [text]
    elif isinstance(path_or_paths, str):
        # single path, only consume allowed files
        path = path_or_paths
        # Below globs should match patterns in file_to_doc()
        [globs_image_types.extend(glob.glob(os.path.join(path, "./**/*.%s" % ftype), recursive=True))
         for ftype in image_types]
        [globs_non_image_types.extend(glob.glob(os.path.join(path, "./**/*.%s" % ftype), recursive=True))
         for ftype in non_image_types]
    else:
        # list/tuple of files (consume what can, and exception those that selected but cannot consume so user knows)
        assert isinstance(path_or_paths, (list, tuple)), "Wrong type for path_or_paths: %s" % type(path_or_paths)
        # reform out of allowed types
        globs_image_types.extend(flatten_list([[x for x in path_or_paths if x.endswith(y)] for y in image_types]))
        # could do below:
        # globs_non_image_types = flatten_list([[x for x in path_or_paths if x.endswith(y)] for y in non_image_types])
        # But instead, allow fail so can collect unsupported too
        set_globs_image_types = set(globs_image_types)
        globs_non_image_types.extend([x for x in path_or_paths if x not in set_globs_image_types])
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


def prep_langchain(persist_directory, load_db_if_exists, db_type, use_openai_embedding, langchain_mode, user_path,
                   hf_embedding_model, n_jobs=-1, kwargs_make_db={}):
    """
    do prep first time, involving downloads
    # FIXME: Add github caching then add here
    :return:
    """
    assert langchain_mode not in ['MyData'], "Should not prep scratch data"

    if os.path.isdir(persist_directory):
        print("Prep: persist_directory=%s exists, using" % persist_directory, flush=True)
        db = get_existing_db(persist_directory, load_db_if_exists, db_type, use_openai_embedding, langchain_mode,
                             hf_embedding_model)
    else:
        print("Prep: persist_directory=%s does not exist, regenerating" % persist_directory, flush=True)
        db = None
        if langchain_mode in ['All', 'DriverlessAI docs']:
            # FIXME: Could also just use dai_docs.pickle directly and upload that
            get_dai_docs(from_hf=True)

        if langchain_mode in ['All', 'wiki']:
            get_wiki_sources(first_para=kwargs_make_db['first_para'], text_limit=kwargs_make_db['text_limit'])

        langchain_kwargs = kwargs_make_db.copy()
        langchain_kwargs.update(locals())
        db = make_db(**langchain_kwargs)

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


def get_existing_db(persist_directory, load_db_if_exists, db_type, use_openai_embedding, langchain_mode,
                    hf_embedding_model):
    if load_db_if_exists and db_type == 'chroma' and os.path.isdir(persist_directory) and os.path.isdir(
            os.path.join(persist_directory, 'index')):
        print("DO Loading db: %s" % langchain_mode, flush=True)
        embedding = get_embedding(use_openai_embedding, hf_embedding_model=hf_embedding_model)
        from chromadb.config import Settings
        client_settings = Settings(anonymized_telemetry=False,
                                   chroma_db_impl="duckdb+parquet",
                                   persist_directory=persist_directory, )
        db = Chroma(persist_directory=persist_directory, embedding_function=embedding,
                    collection_name=langchain_mode.replace(' ', '_'),
                    client_settings=client_settings)
        print("DONE Loading db: %s" % langchain_mode, flush=True)
        return db
    return None


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


def _make_db(use_openai_embedding=False,
             hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
             first_para=False, text_limit=None, chunk=False, chunk_size=1024,
             langchain_mode=None,
             user_path=None,
             db_type='faiss',
             load_db_if_exists=False,
             db=None,
             n_jobs=-1,
             verbose=False):
    persist_directory = 'db_dir_%s' % langchain_mode  # single place, no special names for each case
    if not db and load_db_if_exists and db_type == 'chroma' and os.path.isdir(persist_directory) and os.path.isdir(
            os.path.join(persist_directory, 'index')):
        assert langchain_mode not in ['MyData'], "Should not load MyData db this way"
        print("Loading db", flush=True)
        embedding = get_embedding(use_openai_embedding, hf_embedding_model=hf_embedding_model)
        db = Chroma(persist_directory=persist_directory, embedding_function=embedding,
                    collection_name=langchain_mode.replace(' ', '_'))
    elif not db and langchain_mode not in ['MyData']:
        # Should not make MyData db this way, why avoided, only upload from UI
        sources = []
        if verbose:
            print("Generating sources", flush=True)
        if langchain_mode in ['wiki_full', 'All', "'All'"]:
            from read_wiki_full import get_all_documents
            small_test = None
            print("Generating new wiki", flush=True)
            sources1 = get_all_documents(small_test=small_test, n_jobs=os.cpu_count() // 2)
            print("Got new wiki", flush=True)
            if chunk:
                sources1 = chunk_sources(sources1, chunk_size=chunk_size)
                print("Chunked new wiki", flush=True)
            sources.extend(sources1)
        if langchain_mode in ['wiki', 'All', "'All'"]:
            sources1 = get_wiki_sources(first_para=first_para, text_limit=text_limit)
            if chunk:
                sources1 = chunk_sources(sources1, chunk_size=chunk_size)
            sources.extend(sources1)
        if langchain_mode in ['github h2oGPT', 'All', "'All'"]:
            # sources = get_github_docs("dagster-io", "dagster")
            sources1 = get_github_docs("h2oai", "h2ogpt")
            # FIXME: always chunk for now
            sources1 = chunk_sources(sources1, chunk_size=chunk_size)
            sources.extend(sources1)
        if langchain_mode in ['DriverlessAI docs', 'All', "'All'"]:
            sources1 = get_dai_docs(from_hf=True)
            if chunk and False:  # FIXME: DAI docs are already chunked well, should only chunk more if over limit
                sources1 = chunk_sources(sources1, chunk_size=chunk_size)
            sources.extend(sources1)
        if langchain_mode in ['All', 'UserData']:
            if user_path:
                # chunk internally for speed over multiple docs
                sources1 = path_to_docs(user_path, n_jobs=n_jobs, chunk=chunk, chunk_size=chunk_size)
                sources.extend(sources1)
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
                print("langchain_mode %s has no sources, not making db" % langchain_mode, flush=True)
            return None
        if verbose:
            print("Generating db", flush=True)
        db = get_db(sources, use_openai_embedding=use_openai_embedding, db_type=db_type,
                    persist_directory=persist_directory, langchain_mode=langchain_mode,
                    hf_embedding_model=hf_embedding_model)
        if verbose:
            print("Generated db", flush=True)
    return db


source_prefix = "Sources [Score | Link]:"
source_postfix = "End Sources<p>"


def run_qa_db(**kwargs):
    func_names = list(inspect.signature(_run_qa_db).parameters)
    # hard-coded defaults
    kwargs['answer_with_sources'] = True
    kwargs['sanitize_bot_response'] = True
    kwargs['show_rank'] = False
    missing_kwargs = [x for x in func_names if x not in kwargs]
    assert not missing_kwargs, "Missing kwargs: %s" % missing_kwargs
    # only keep actual used
    kwargs = {k: v for k, v in kwargs.items() if k in func_names}
    return _run_qa_db(**kwargs)


def _run_qa_db(query=None,
               use_openai_model=False, use_openai_embedding=False,
               first_para=False, text_limit=None, k=4, chunk=False, chunk_size=1024,
               user_path=None,
               db_type='faiss',
               model_name=None, model=None, tokenizer=None,
               hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
               stream_output=False,
               prompter=None,
               prompt_type=None,
               answer_with_sources=True,
               cut_distanct=1.1,
               sanitize_bot_response=True,
               show_rank=False,
               load_db_if_exists=False,
               db=None,
               max_new_tokens=256,
               temperature=0.1,
               repetition_penalty=1.0,
               top_k=40,
               top_p=0.7,
               langchain_mode=None,
               document_choice=['All'],
               n_jobs=-1,
               verbose=False,
               cli=False):
    """

    :param query:
    :param use_openai_model:
    :param use_openai_embedding:
    :param first_para:
    :param text_limit:
    :param k:
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
    assert query is not None
    assert prompter is not None or prompt_type is not None or model is None  # if model is None, then will generate
    if prompter is not None:
        prompt_type = prompter.prompt_type
    if model is not None:
        assert prompt_type is not None
    llm, model_name, streamer, prompt_type_out = get_llm(use_openai_model=use_openai_model, model_name=model_name,
                                                         model=model, tokenizer=tokenizer,
                                                         stream_output=stream_output,
                                                         max_new_tokens=max_new_tokens,
                                                         temperature=temperature,
                                                         repetition_penalty=repetition_penalty,
                                                         top_k=top_k,
                                                         top_p=top_p,
                                                         prompt_type=prompt_type,
                                                         prompter=prompter,
                                                         verbose=verbose,
                                                         )

    if model_name in non_hf_types:
        # FIXME: for now, streams to stdout/stderr currently
        stream_output = False

    use_context = False
    scores = []
    chain = None

    func_names = list(inspect.signature(get_similarity_chain).parameters)
    sim_kwargs = {k: v for k, v in locals().items() if k in func_names}
    missing_kwargs = [x for x in func_names if x not in sim_kwargs]
    assert not missing_kwargs, "Missing: %s" % missing_kwargs
    docs, chain, scores, use_context = get_similarity_chain(**sim_kwargs)
    if len(document_choice) > 0 and document_choice[0] == 'Only':
        formatted_doc_chunks = '\n\n'.join([get_url(x) + '\n\n' + x.page_content for x in docs])
        yield formatted_doc_chunks, ''
        return
    if chain is None and model_name not in non_hf_types:
        # can only return if HF type
        return

    if stream_output:
        answer = None
        assert streamer is not None
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


def get_similarity_chain(query=None,
                         use_openai_model=False, use_openai_embedding=False,
                         first_para=False, text_limit=None, k=4, chunk=False, chunk_size=1024,
                         user_path=None,
                         db_type='faiss',
                         model_name=None,
                         hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                         prompt_type=None,
                         cut_distanct=1.1,
                         load_db_if_exists=False,
                         db=None,
                         langchain_mode=None,
                         document_choice=['All'],
                         n_jobs=-1,
                         # beyond run_db_query:
                         llm=None,
                         verbose=False,
                         ):
    # determine whether use of context out of docs is planned
    if not use_openai_model and prompt_type not in ['plain'] or model_name in non_hf_types:
        if langchain_mode in ['Disabled', 'ChatLLM', 'LLM']:
            use_context = False
        else:
            use_context = True
    else:
        use_context = True

    # https://github.com/hwchase17/langchain/issues/1946
    # FIXME: Seems to way to get size of chroma db to limit k to avoid
    # Chroma collection MyData contains fewer than 4 elements.
    # type logger error
    k_db = 1000 if db_type == 'chroma' else k  # k=100 works ok too for

    # FIXME: For All just go over all dbs instead of a separate db for All
    db = make_db(use_openai_embedding=use_openai_embedding,
                 hf_embedding_model=hf_embedding_model,
                 first_para=first_para, text_limit=text_limit, chunk=chunk, chunk_size=chunk_size,
                 langchain_mode=langchain_mode,
                 user_path=user_path,
                 db_type=db_type,
                 load_db_if_exists=load_db_if_exists,
                 db=db,
                 n_jobs=n_jobs,
                 verbose=verbose)

    if db and use_context:
        if isinstance(document_choice, str):
            # support string as well
            document_choice = [document_choice]
        if not isinstance(db, Chroma) or \
                len(document_choice) == 0 or \
                len(document_choice) <= 1 and document_choice[0] == 'All':
            # treat empty list as All for now, not 'None'
            filter_kwargs = {}
        elif len(document_choice) > 0 and document_choice[0] == 'Only':
            # Only means All docs, but only will return sources, not LLM response
            filter_kwargs = {}
        else:
            if len(document_choice) >= 2:
                or_filter = [{"source": {"$eq": x}} for x in document_choice]
                filter_kwargs = dict(filter={"$or": or_filter})
            elif len(document_choice) > 0:
                one_filter = [{"source": {"$eq": x}} for x in document_choice][0]
                filter_kwargs = dict(filter=one_filter)
            else:
                filter_kwargs = {}
            if len(document_choice) == 1 and document_choice[0] == 'None':
                k_db = 1
                k = 0
        docs_with_score = db.similarity_search_with_score(query, k=k_db, **filter_kwargs)[:k]
        # cut off so no high distance docs/sources considered
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
        return docs, None, [], False

    if len(document_choice) > 0 and document_choice[0] == 'Only':
        # no LLM use
        return docs, None, [], False

    common_words_file = "data/NGSL_1.2_stats.csv.zip"
    if os.path.isfile(common_words_file):
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

    if not use_openai_model and prompt_type not in ['plain'] or model_name in non_hf_types:
        # instruct-like, rather than few-shot prompt_type='plain' as default
        # but then sources confuse the model with how inserted among rest of text, so avoid
        prefix = ""
        if langchain_mode in ['Disabled', 'ChatLLM', 'LLM'] or not use_context:
            template = """%s{context}{question}""" % prefix
        else:
            template = """%s
==
{context}
==
{question}""" % prefix
        prompt = PromptTemplate(
            # input_variables=["summaries", "question"],
            input_variables=["context", "question"],
            template=template,
        )
        chain = load_qa_chain(llm, prompt=prompt)
    else:
        chain = load_qa_with_sources_chain(llm)

    if not use_context:
        chain_kwargs = dict(input_documents=[], question=query)
    else:
        chain_kwargs = dict(input_documents=docs, question=query)

    target = wrapped_partial(chain, chain_kwargs)
    return docs, target, scores, use_context


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


def chunk_sources(sources, chunk_size=1024):
    source_chunks = []
    # Below for known separator
    # splitter = CharacterTextSplitter(separator=" ", chunk_size=chunk_size, chunk_overlap=0)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    for source in sources:
        # print(source.metadata['source'], flush=True)
        for chunky in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunky, metadata=source.metadata))
    return source_chunks


def get_db_from_hf(dest=".", db_dir='db_dir_DriverlessAI_docs.zip'):
    from huggingface_hub import hf_hub_download
    # True for case when locally already logged in with correct token, so don't have to set key
    token = os.getenv('HUGGINGFACE_API_TOKEN', True)
    path_to_zip_file = hf_hub_download('h2oai/db_dirs', db_dir, token=token, repo_type='dataset')
    import zipfile
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
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


if __name__ == '__main__':
    pass
