import glob
import inspect
import os
import pathlib
import subprocess
import tempfile
import traceback
from collections import defaultdict

from utils import wrapped_partial, EThread, import_matplotlib

import_matplotlib()

import numpy as np
import pandas as pd
import requests
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader, PythonLoader, TomlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain import PromptTemplate
from langchain.vectorstores import Chroma


def get_db(sources, use_openai_embedding=False, db_type='faiss', persist_directory="db_dir", langchain_mode='notset'):
    # get embedding model
    embedding = get_embedding(use_openai_embedding)

    # Create vector database
    if db_type == 'faiss':
        db = FAISS.from_documents(sources, embedding)
    elif db_type == 'chroma':
        os.makedirs(persist_directory, exist_ok=True)
        db = Chroma.from_documents(documents=sources, embedding=embedding, persist_directory=persist_directory,
                                   collection_name=langchain_mode.replace(' ', '_'),
                                   anonymized_telemetry=False)
        db.persist()
        # FIXME: below just proves can load persistent dir, regenerates its embedding files, so a bit wasteful
        db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    else:
        raise RuntimeError("No such db_type=%s" % db_type)

    return db


def add_to_db(db, sources, db_type='faiss'):
    if db_type == 'faiss':
        db = db.add_documents(sources)
    elif db_type == 'chroma':
        db = db.add_documents(documents=sources)
        db.persist()
    else:
        raise RuntimeError("No such db_type=%s" % db_type)

    return db


def get_embedding(use_openai_embedding):
    # Get embedding model
    if use_openai_embedding:
        assert os.getenv("OPENAI_API_KEY") is not None, "Set ENV OPENAI_API_KEY"
        from langchain.embeddings import OpenAIEmbeddings
        embedding = OpenAIEmbeddings()
    else:
        from langchain.embeddings import HuggingFaceEmbeddings

        model_name = "sentence-transformers/all-MiniLM-L6-v2"  # good, gets authors
        device, torch_dtype, context_class = get_device_dtype()
        model_kwargs = dict(device=device)
        embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
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
            ):
    if use_openai_model:
        from langchain.llms import OpenAI
        llm = OpenAI(temperature=0)
        model_name = 'openai'
        streamer = None
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if model is None:
            # only used if didn't pass model in
            assert model_name is None
            assert tokenizer is None
            model_name = 'h2oai/h2ogpt-oasst1-512-12b'
            # model_name = 'h2oai/h2ogpt-oig-oasst1-512-6.9b'
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

        gen_kwargs = dict(max_new_tokens=max_new_tokens, return_full_text=True, early_stopping=False)
        if stream_output:
            skip_prompt = False
            from generate import H2OTextIteratorStreamer
            decoder_kwargs = {}
            streamer = H2OTextIteratorStreamer(tokenizer, skip_prompt=skip_prompt, block=False, **decoder_kwargs)
            gen_kwargs.update(dict(streamer=streamer))
        else:
            streamer = None

        if 'h2ogpt' in model_name:
            from h2oai_pipeline import H2OTextGenerationPipeline
            pipe = H2OTextGenerationPipeline(model=model, tokenizer=tokenizer, **gen_kwargs)
            # pipe.task = "text-generation"
            # below makes it listen only to our prompt removal, not built in prompt removal that is less general and not specific for our model
            pipe.task = "text2text-generation"
        else:
            # only for non-instruct tuned cases when ok with just normal next token prediction
            from transformers import pipeline
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **gen_kwargs)

        from langchain.llms import HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=pipe)
    return llm, model_name, streamer


def get_llm_prompt(model_name):
    if 'h2ogpt' in model_name:
        template = """<human>: {question}
<bot>: """
    else:
        template = """
                {question}
                """

    prompt = PromptTemplate(
        input_variables=["question"],
        template=template,
    )
    return prompt


def get_llm_chain(llm, model_name):
    from langchain import LLMChain

    prompt = get_llm_prompt(model_name)

    chain = LLMChain(
        llm=llm,
        verbose=True,
        prompt=prompt
    )
    return chain


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


def get_dai_docs(from_hf=False):
    """
    Consume DAI documentation
    :param from_hf:
    :return:
    """
    import pickle

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


def file_to_doc(file):
    if file.endswith('.txt'):
        return TextLoader(file, encoding="utf8").load()
    elif file.endswith('.md') or file.endswith('.rst'):
        with open(file, "r") as f:
            return Document(page_content=f.read(), metadata={"source": file})
    elif file.endswith('.pdf'):
        # return PDFMinerLoader(file).load()  # fails with ypeError: expected str, bytes or os.PathLike object, not BufferedReader
        return PyPDFLoader(file).load_and_split()
    elif file.endswith('.csv'):
        return CSVLoader(file).load()
    elif file.endswith('.py'):
        return PythonLoader(file).load()
    elif file.endswith('.toml'):
        return TomlLoader(file).load()
    else:
        raise RuntimeError("No file handler for %s" % file)


def path_to_docs(path, verbose=False, fail_any_exception=False):
    globs = glob.glob(os.path.join(path, "./**/*.txt"), recursive=True) + \
            glob.glob(os.path.join(path, "./**/*.md"), recursive=True) + \
            glob.glob(os.path.join(path, "./**/*.rst"), recursive=True) + \
            glob.glob(os.path.join(path, "./**/*.pdf"), recursive=True) + \
            glob.glob(os.path.join(path, "./**/*.csv"), recursive=True) + \
            glob.glob(os.path.join(path, "./**/*.py"), recursive=True) + \
            glob.glob(os.path.join(path, "./**/*.toml"), recursive=True)
    for file in globs:
        if verbose:
            print("Ingesting file: %s" % file, flush=True)
        res = None
        try:
            res = file_to_doc(file)
        except BaseException:
            print("Failed to ingest %s due to %s" % (file, traceback.format_exc()))
            if fail_any_exception:
                raise
        if res:
            if isinstance(res, list):
                for x in res:
                    yield x
            else:
                yield res


def test_qa_wiki_openai():
    return run_qa_wiki(use_openai_model=True)


def test_qa_wiki_stuff_hf():
    # NOTE: total context length makes things fail when n_sources * text_limit >~ 2048
    return run_qa_wiki(use_openai_model=False, text_limit=256, chain_type='stuff')


def test_qa_wiki_map_reduce_hf():
    return run_qa_wiki(use_openai_model=False, text_limit=None, chain_type='map_reduce')


def run_qa_wiki(use_openai_model=False, first_para=True, text_limit=None, chain_type='stuff'):
    sources = get_wiki_sources(first_para=first_para, text_limit=text_limit)
    llm, model_name, streamer = get_llm(use_openai_model=use_openai_model)
    chain = load_qa_with_sources_chain(llm, chain_type=chain_type)

    question = "What are the main differences between Linux and Windows?"
    answer = get_answer_from_sources(chain, sources, question)
    print(answer)


def test_qa_wiki_db_openai():
    return run_qa_db(use_openai_model=True, use_openai_embedding=True, text_limit=None, wiki=True)


def test_qa_wiki_db_hf():
    # if don't chunk, still need to limit
    # but this case can handle at least more documents, by picking top k
    # FIXME: but spitting out garbage answer right now, all fragmented, or just 1-word answer
    return run_qa_db(use_openai_model=False, use_openai_embedding=False, text_limit=256, wiki=True)


def test_qa_wiki_db_chunk_hf():
    return run_qa_db(use_openai_model=False, use_openai_embedding=False, text_limit=256, chunk=True, chunk_size=256,
                     wiki=True)


def test_qa_wiki_db_chunk_openai():
    # don't need 256, just seeing how compares to hf
    return run_qa_db(use_openai_model=True, use_openai_embedding=True, text_limit=256, chunk=True, chunk_size=256,
                     wiki=True)


def test_qa_github_db_chunk_openai():
    # don't need 256, just seeing how compares to hf
    query = "what is a software defined asset"
    return run_qa_db(query=query, use_openai_model=True, use_openai_embedding=True, text_limit=256, chunk=True,
                     chunk_size=256, github=True)


def test_qa_daidocs_db_chunk_hf():
    # FIXME: doesn't work well with non-instruct-tuned Cerebras
    query = "Which config.toml enables pytorch for NLP?"
    return run_qa_db(query=query, use_openai_model=False, use_openai_embedding=False, text_limit=None, chunk=True,
                     chunk_size=128, wiki=False,
                     dai_rst=True)


def test_qa_daidocs_db_chunk_hf_faiss():
    query = "Which config.toml enables pytorch for NLP?"
    # chunk_size is chars for each of k=4 chunks
    return run_qa_db(query=query, use_openai_model=False, use_openai_embedding=False, text_limit=None, chunk=True,
                     chunk_size=128 * 1,  # characters, and if k=4, then 4*4*128 = 2048 chars ~ 512 tokens
                     wiki=False,
                     dai_rst=True,
                     db_type='faiss',
                     )


def test_qa_daidocs_db_chunk_hf_chroma():
    query = "Which config.toml enables pytorch for NLP?"
    # chunk_size is chars for each of k=4 chunks
    return run_qa_db(query=query, use_openai_model=False, use_openai_embedding=False, text_limit=None, chunk=True,
                     chunk_size=128 * 1,  # characters, and if k=4, then 4*4*128 = 2048 chars ~ 512 tokens
                     wiki=False,
                     dai_rst=True,
                     db_type='chroma',
                     )


def test_qa_daidocs_db_chunk_openai():
    query = "Which config.toml enables pytorch for NLP?"
    return run_qa_db(query=query, use_openai_model=True, use_openai_embedding=True, text_limit=256, chunk=True,
                     chunk_size=256, wiki=False, dai_rst=True)


def test_qa_daidocs_db_chunk_openaiembedding_hfmodel():
    query = "Which config.toml enables pytorch for NLP?"
    return run_qa_db(query=query, use_openai_model=False, use_openai_embedding=True, text_limit=None, chunk=True,
                     chunk_size=128, wiki=False, dai_rst=True)


def prep_langchain(persist_directory, load_db_if_exists, db_type, use_openai_embedding, langchain_mode, user_path):
    """
    do prep first time, involving downloads
    # FIXME: Add github caching then add here
    # FIXME: Once go FAISS->Chroma, can avoid this prep step
    :return:
    """

    if os.path.isdir(persist_directory):
        db = get_existing_db(persist_directory, load_db_if_exists, db_type, use_openai_embedding, langchain_mode)
    else:
        db = None
        if langchain_mode in ['All', 'DriverlessAI docs']:
            # FIXME: Could also just use dai_docs.pickle directly and upload that
            get_dai_docs(from_hf=True)

        if langchain_mode in ['All', 'wiki']:
            text_limit = None
            for first_para in [True, False]:
                get_wiki_sources(first_para=first_para, text_limit=text_limit)

        langchain_kwargs = get_db_kwargs(langchain_mode)
        langchain_kwargs.update(locals())
        db = make_db(**langchain_kwargs)

    return db


def get_db_kwargs(langchain_mode):
    return dict(chunk=True,  # chunking with small chunk_size hurts accuracy esp. if k small
                # chunk = False  # chunking with small chunk_size hurts accuracy esp. if k small
                chunk_size=128 * 4,  # FIXME
                first_para=False,
                text_limit=None,
                # db_type = 'faiss',
                db_type='chroma',
                sanitize_bot_response=True,
                langchain_mode=langchain_mode)


def get_existing_db(persist_directory, load_db_if_exists, db_type, use_openai_embedding, langchain_mode):
    if load_db_if_exists and db_type == 'chroma' and os.path.isdir(persist_directory) and os.path.isdir(
            os.path.join(persist_directory, 'index')):
        print("DO Loading db", flush=True)
        embedding = get_embedding(use_openai_embedding)
        db = Chroma(persist_directory=persist_directory, embedding_function=embedding,
                    collection_name=langchain_mode.replace(' ', '_'))
        print("DONE Loading db", flush=True)
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
             first_para=False, text_limit=None, chunk=False, chunk_size=1024,
             langchain_mode=None,
             user_path=None,
             db_type='faiss',
             load_db_if_exists=False,
             db=None):
    persist_directory = 'db_dir_%s' % langchain_mode  # single place, no special names for each case
    if not db and load_db_if_exists and db_type == 'chroma' and os.path.isdir(persist_directory) and os.path.isdir(
            os.path.join(persist_directory, 'index')):
        print("Loading db", flush=True)
        embedding = get_embedding(use_openai_embedding)
        db = Chroma(persist_directory=persist_directory, embedding_function=embedding,
                    collection_name=langchain_mode.replace(' ', '_'))
    elif not db:
        sources = []
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
        if user_path and langchain_mode in ['All', 'UserData']:
            sources1 = path_to_docs(user_path)
            if chunk:
                sources1 = chunk_sources(sources1, chunk_size=chunk_size)
            sources.extend(sources1)
        if False and langchain_mode in ['urls', 'All', "'All'"]:
            # from langchain.document_loaders import UnstructuredURLLoader
            # loader = UnstructuredURLLoader(urls=urls)
            urls = ["https://www.birdsongsf.com/who-we-are/"]
            from langchain.document_loaders import PlaywrightURLLoader
            loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])
            sources1 = loader.load()
            sources.extend(sources1)
        if not sources:
            print("langchain_mode %s has no sources, not making db" % langchain_mode, flush=True)
            return None
        print("Generating db", flush=True)
        db = get_db(sources, use_openai_embedding=use_openai_embedding, db_type=db_type,
                    persist_directory=persist_directory, langchain_mode=langchain_mode)
    return db


source_prefix = "Sources [Score | Link]:"
source_postfix = "End Sources<p>"


def run_qa_db(query=None,
              use_openai_model=False, use_openai_embedding=False,
              first_para=False, text_limit=None, k=4, chunk=False, chunk_size=1024,
              wiki=False, github=False, dai_rst=False, urls=False, wiki_full=True, all=None,
              user_path=None, split_method='chunk',
              db_type='faiss',
              model_name=None, model=None, tokenizer=None,
              stream_output=False,
              prompter=None,
              answer_with_sources=True,
              cut_distanct=1.1,
              sanitize_bot_response=True,
              do_yield=False,
              show_rank=False,
              load_db_if_exists=False,
              persist_directory_base='db_dir',
              limit_wiki_full=5000000,
              min_views=1000,
              db=None,
              max_new_tokens=256,
              langchain_mode=None):
    """

    :param query:
    :param use_openai_model:
    :param use_openai_embedding:
    :param first_para:
    :param text_limit:
    :param k:
    :param chunk:
    :param chunk_size:
    :param wiki: bool if using wiki
    :param github: bool if using github
    :param dai_rst: bool if using dai RST files
    :param user_path: user path to glob recursively from
    :param split_method: split method for PDF inputs
    :param db_type: 'faiss' for in-memory db or 'chroma' for persistent db
    :param model_name: model name, used to switch behaviors
    :param model: pre-initialized model, else will make new one
    :param tokenizer: pre-initialized tokenizer, else will make new one.  Required not None if model is not None
    :param answer_with_sources
    :return:
    """

    # FIXME: For All just go over all dbs instead of a separate db for All
    db = make_db(**locals())
    llm, model_name, streamer = get_llm(use_openai_model=use_openai_model, model_name=model_name,
                                        model=model, tokenizer=tokenizer,
                                        stream_output=stream_output,
                                        max_new_tokens=max_new_tokens)

    if not use_openai_model and 'h2ogpt' in model_name:
        # instruct-like, rather than few-shot prompt_type='plain' as default
        # but then sources confuse the model with how inserted among rest of text, so avoid
        prefix = "The following text contains Content from chunks of text extracted from source documentation.  Please give a natural language concise answer to any question using the Content text fragments information provided."
        prefix = ""
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
        # chain = load_qa_with_sources_chain(llm, prompt=prompt)
        chain = load_qa_chain(llm, prompt=prompt)
    else:
        chain = load_qa_with_sources_chain(llm)

    if query is None:
        query = "What are the main differences between Linux and Windows?"
    # https://github.com/hwchase17/langchain/issues/1946
    k_db = 1000 if db_type == 'chroma' else k  # k=100 works ok too for

    docs_with_score = db.similarity_search_with_score(query, k=k_db)[:k]

    # cut off so no high distance docs/sources considered
    docs = [x[0] for x in docs_with_score if x[1] < cut_distanct]
    scores = [x[1] for x in docs_with_score if x[1] < cut_distanct]
    if not docs:
        return None
    print("Distance: min: %s max: %s mean: %s median: %s" %
          (scores[0], scores[-1], np.mean(scores), np.median(scores)), flush=True)

    common_words_file = "data/NGSL_1.2_stats.csv.zip"
    if os.path.isfile(common_words_file):
        df = pd.read_csv("data/NGSL_1.2_stats.csv.zip")
        import string
        reduced_query = query.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).strip()
        reduced_query_words = reduced_query.split(' ')
        set_common = set(df['Lemma'].values.tolist())
        num_common = len([x.lower() in set_common for x in reduced_query_words])
        frac_common = num_common / len(reduced_query)
        # FIXME: report to user bad query that uses too many common words
        print("frac_common: %s" % frac_common, flush=True)

    chain_kwargs = dict(input_documents=docs, question=query)
    if stream_output:
        answer = None
        assert streamer is not None
        target = wrapped_partial(chain, chain_kwargs)
        import queue
        bucket = queue.Queue()
        thread = EThread(target=target, streamer=streamer, bucket=bucket)
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
                    yield output1
                else:
                    yield outputs
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
        answer = chain(chain_kwargs)

    if answer is not None:
        print("query: %s" % query, flush=True)
        print("answer: %s" % answer['output_text'], flush=True)
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
            ret = answer['output_text'] + '\n' + sorted_sources_urls
        else:
            ret = answer['output_text']

        if stream_output or do_yield:
            # just yield more, not all
            yield ret
            return
        else:
            return ret


def get_url(x):
    if x.metadata['source'].startswith('http://') or x.metadata['source'].startswith('https://'):
        return """<a href="%s" target="_blank"  rel="noopener noreferrer">%s</a>""" % (
            x.metadata['source'], x.metadata['source'])
    else:
        return """<a href="file/%s" target="_blank"  rel="noopener noreferrer">%s</a>""" % (
            x.metadata['source'], x.metadata['source'])


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


def get_db_from_hf():
    from huggingface_hub import hf_hub_download
    # True for case when locally already logged in with correct token, so don't have to set key
    token = os.getenv('HUGGINGFACE_API_TOKEN', True)
    path_to_zip_file = hf_hub_download('h2oai/dai_docs', 'db_dirs.zip', token=token, repo_type='dataset')
    path = '.'
    import zipfile
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(path)


if __name__ == '__main__':
    pass
