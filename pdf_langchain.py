import glob
import os
import pathlib
import subprocess
import tempfile
from abc import ABC
from collections import defaultdict
from typing import Optional, List, Mapping, Any

from utils import wrapped_partial, EThread, import_matplotlib

import_matplotlib()

import numpy as np
import pandas as pd
import requests
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.docstore.document import Document
from langchain.llms.base import LLM
from langchain import PromptTemplate

# FIXME:
# from langchain.vectorstores import Milvus

from langchain.vectorstores import Chroma

# https://python.langchain.com/en/latest/modules/models/llms/examples/llm_caching.html
# from langchain.cache import InMemoryCache
# langchain.llm_cache = InMemoryCache()

try:
    raise ValueError("Disabled, too greedy even if change model etc.")
    import langchain
    from langchain.cache import SQLiteCache

    langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
    print("Caching", flush=True)
except Exception as e:
    print("NO caching: %s" % str(e), flush=True)


def get_db(sources, use_openai_embedding=False, db_type='faiss', persist_directory="db_dir"):
    # get embedding model
    embedding = get_embedding(use_openai_embedding)

    # Create vector database
    if db_type == 'faiss':
        db = FAISS.from_documents(sources, embedding)
    elif db_type == 'chroma':
        os.makedirs(persist_directory, exist_ok=True)
        db = Chroma.from_documents(documents=sources, embedding=embedding, persist_directory=persist_directory)
        db.persist()
        db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    else:
        raise RuntimeError("No such db_type=%s" % db_type)

    return db


def pdf_to_sources(pdf_filename=None, split_method='chunk'):
    if split_method == 'page':
        # Simple method - Split by pages
        loader = PyPDFLoader(pdf_filename)
        pages = loader.load_and_split()
        print(pages[0])

        # SKIP TO STEP 2 IF YOU'RE USING THIS METHOD
        chunks = pages
    elif split_method == 'chunk':
        # Advanced method - Split by chunk

        # Step 1: Convert PDF to text
        raise RuntimeError("textract requires old six, avoid")
        import textract
        doc = textract.process(pdf_filename)

        # Step 2: Save to .txt and reopen (helps prevent issues)
        txt_filename = pdf_filename.replace('.pdf', '.txt')
        with open(txt_filename, 'w') as f:
            f.write(doc.decode('utf-8'))

        with open(txt_filename, 'r') as f:
            text = f.read()

        # Step 3: Create function to count tokens
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        def count_tokens(textin: str) -> int:
            return len(tokenizer.encode(textin))

        # Step 4: Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=512,
            chunk_overlap=24,
            length_function=count_tokens,
        )

        chunks = text_splitter.create_documents([text])

        # show_counts(chunks, count_tokens)
    else:
        raise RuntimeError("No such split_method=%s" % split_method)

    # Result is many LangChain 'Documents' around 500 tokens or less (Recursive splitter sometimes allows more tokens to retain context)
    type(chunks[0])
    return chunks


def get_embedding(use_openai_embedding):
    # Get embedding model
    if use_openai_embedding:
        assert os.getenv("OPENAI_API_KEY") is not None, "Set ENV OPENAI_API_KEY"
        from langchain.embeddings import OpenAIEmbeddings
        embedding = OpenAIEmbeddings()
    else:
        from langchain.embeddings import HuggingFaceEmbeddings

        # model_name = "sentence-transformers/all-mpnet-base-v2"  # poor
        model_name = "sentence-transformers/all-MiniLM-L6-v2"  # good, gets authors
        # model_name = "sentence-transformers/all-MiniLM-L12-v2"  # 12 layers FAILS OOM I think
        # model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
        # model_name = 'cerebras/Cerebras-GPT-2.7B' # OOM
        # model_name = 'microsoft/deberta-v3-base'  # microsoft/mdeberta-v3-base for multilinguial
        load_8bit = False
        device, torch_dtype, context_class = get_device_dtype()
        model_kwargs = dict(device=device)  # , torch_dtype=torch_dtype, load_in_8bit=load_8bit)
        embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

        # for some models need to fix tokenizer
        if model_name in ['cerebras/Cerebras-GPT-2.7B']:
            embedding.client.tokenizer.pad_token = embedding.client.tokenizer.eos_token

        # also see:
        # https://www.sbert.net/docs/pretrained-models/msmarco-v3.html
        # https://www.sbert.net/examples/applications/semantic-search/README.html
        # https://towardsdatascience.com/bert-for-measuring-text-similarity-eec91c6bf9e1
        # https://discuss.huggingface.co/t/get-word-embeddings-from-transformer-model/6929/2
    return embedding


def get_answer_from_sources(chain, sources, question):
    return chain(
        {
            "input_documents": sources,
            "question": question,
        },
        return_only_outputs=True,
    )["output_text"]


def get_llm(use_openai_model=False, model_name=None, model=None, tokenizer=None, stream_output=False):
    if use_openai_model:
        from langchain.llms import OpenAI
        llm = OpenAI(temperature=0)
        model_name = 'openai'
        streamer = None
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if model is None:
            assert model_name is None
            assert tokenizer is None
            # model_name = "cerebras/Cerebras-GPT-2.7B"
            # model_name = "cerebras/Cerebras-GPT-13B"
            # model_name = "cerebras/Cerebras-GPT-6.7B"
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

        gen_kwargs = dict(max_new_tokens=256, return_full_text=True, early_stopping=False)
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


def get_answer_from_db(db, query, sources=False, chat_history='', use_openai_model=False, k=4, chat=True,
                       use_chain_ret=False):
    # Check similarity search is working
    # docs = db.similarity_search(query, k=k)
    # print(docs[0])

    # get LLM
    llm, model_name, streamer = get_llm(use_openai_model=use_openai_model)

    # Create QA chain to integrate similarity search with user queries (answer query from knowledge base)
    if use_openai_model:
        if sources:
            chain = load_qa_with_sources_chain(llm, chain_type="stuff")
        else:
            chain = load_qa_chain(llm, chain_type="stuff")
    else:
        # make custom llm prompt aware
        chain = get_llm_chain(llm, model_name)
        # WIP OPTIONAL
        if False:
            from langchain import SerpAPIWrapper
            from langchain.agents import Tool
            from langchain.agents import initialize_agent

            serpapi = SerpAPIWrapper(serpapi_api_key='...')
            tools = [
                Tool(
                    name="Search",
                    func=serpapi.run,
                    description="useful for when you need to get a weather forecast"
                )
            ]

            agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
            res = agent.run(input="What is the weather forecast for Poznan, Poland")
            print(res)

    docs = db.similarity_search(query, k=k)

    if chat:
        chain.run(input_documents=docs, question=query)
        # Create conversation chain that uses our vectordb as retriever, this also allows for chat history management
        qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever())

        # [x.page_content for x in docs if 'Illia' in x.page_content]
        result = qa({"question": query, "chat_history": chat_history})
        answer = result['answer']
    elif use_chain_ret:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever(),
                                               return_source_documents=True)
        llm_response = qa_chain(query)
        answer = llm_response['result']
        sources = ''
        for source in llm_response["source_documents"]:
            if 'source' in source.metadata:
                sources += source.metadata['source']
        if sources:
            sources = "\nSources\n" + sources
        else:
            sources = "\n No Sources\n"
        answer += sources
    else:
        answer = chain(
            {
                "input_documents": docs,
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]

    return answer


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


class H2OChatBotLLM(LLM, ABC):
    # FIXME: WIP, use gradio_client not requests
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = requests.post(
            "http://0.0.0.0:7860/prompt",
            json={
                "prompt": prompt,
                "temperature": 0,
                "max_new_tokens": 256,
                "stop": stop + ["Observation:"]
            }
        )
        response.raise_for_status()
        return response.json()["response"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {

        }


def show_counts(chunks, count_tokens):
    # Quick data visualization to ensure chunking was successful

    # Create a list of token counts
    token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

    # Create a DataFrame from the token counts
    df = pd.DataFrame({'Token Count': token_counts})

    # Create a histogram of the token count distribution
    df.hist(bins=40, )

    # Show the plot
    import matplotlib.pyplot as plt
    plt.show()


def get_wiki_data(title, first_paragraph_only, text_limit=None, take_head=True):
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
    return Document(
        page_content=page_content,
        metadata={"source": f"https://en.wikipedia.org/wiki/{title}"},
    )


def get_wiki_sources(first_para=True, text_limit=None):
    return [
        get_wiki_data("Barclays", first_para, text_limit=text_limit),
        get_wiki_data("Birdsong_(restaurant)", first_para, text_limit=text_limit),
        get_wiki_data("Unix", first_para, text_limit=text_limit),
        get_wiki_data("Microsoft_Windows", first_para, text_limit=text_limit),
        get_wiki_data("Linux", first_para, text_limit=text_limit),
        #get_wiki_data("Seinfeld", first_para, text_limit=text_limit),
    ]


def get_github_docs(repo_owner, repo_name):
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
        #yield itm
        sources.append(itm)
    return sources, dst


def get_rst_docs(path):
    md_files = glob.glob(os.path.join(path, "./**/*.md"), recursive=True)
    rst_files = glob.glob(os.path.join(path, "./**/*.rst"), recursive=True)
    for file in md_files + rst_files:
        with open(file, "r") as f:
            yield Document(page_content=f.read(), metadata={"source": file})


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
                     chunk_size=128*1,  # characters, and if k=4, then 4*4*128 = 2048 chars ~ 512 tokens
                     wiki=False,
                     dai_rst=True,
                     db_type='faiss',
                     )


def test_qa_daidocs_db_chunk_hf_chroma():
    query = "Which config.toml enables pytorch for NLP?"
    # chunk_size is chars for each of k=4 chunks
    return run_qa_db(query=query, use_openai_model=False, use_openai_embedding=False, text_limit=None, chunk=True,
                     chunk_size=128*1,  # characters, and if k=4, then 4*4*128 = 2048 chars ~ 512 tokens
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


def prep_langchain():
    """
    do prep first time, involving downloads
    # FIXME: Add github caching then add here
    # FIXME: Once go FAISS->Chroma, can avoid this prep step
    :return:
    """

    # FIXME: Could also just use dai_docs.pickle directly and upload that
    get_dai_docs(from_hf=True)

    text_limit = None
    for first_para in [True, False]:
        get_wiki_sources(first_para=first_para, text_limit=text_limit)


def run_qa_db(query=None,
              use_openai_model=False, use_openai_embedding=False,
              first_para=True, text_limit=None, k=4, chunk=False, chunk_size=1024,
              wiki=False, github=False, dai_rst=False, all=None,
              pdf_filename=None, split_method='chunk',
              texts_folder=None,
              db_type='faiss',
              model_name=None, model=None, tokenizer=None,
              stream_output=False,
              prompter=None,
              answer_with_sources=True,
              cut_distanct=1.3,
              sanitize_bot_response=True,
              do_yield=False,
              show_rank=False):
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
    :param pdf_filename: PDF filename
    :param split_method: split method for PDF inputs
    :param texts_folder:
    :param db_type: 'faiss' for in-memory db or 'chroma' for persistent db
    :param model_name: model name, used to switch behaviors
    :param model: pre-initialized model, else will make new one
    :param tokenizer: pre-initialized tokenizer, else will make new one.  Required not None if model is not None
    :param answer_with_sources
    :return:
    """
    # see https://dagster.io/blog/chatgpt-langchain
    sources = []
    if wiki or all:
        sources1 = get_wiki_sources(first_para=first_para, text_limit=text_limit)
        if chunk:
            sources1 = chunk_sources(sources1, chunk_size=chunk_size)
        sources.extend(sources1)
    if github or all:
        # sources = get_github_docs("dagster-io", "dagster")
        sources1 = get_github_docs("h2oai", "h2ogpt")
        # FIXME: always chunk for now
        sources1 = chunk_sources(sources1, chunk_size=chunk_size)
        sources.extend(sources1)
    if dai_rst or all:
        #home = os.path.expanduser('~')
        #sources = get_rst_docs(os.path.join(home, "h2oai.superclean/docs/"))
        sources1, dst = get_dai_docs(from_hf=True)
        if chunk and False:  # FIXME: DAI docs are already chunked well, should only chunk more if over limit
            sources1 = chunk_sources(sources1, chunk_size=chunk_size)
        sources.extend(sources1)
    if pdf_filename:
        sources1 = pdf_to_sources(pdf_filename=pdf_filename, split_method=split_method)
        if chunk:
            sources1 = chunk_sources(sources1, chunk_size=chunk_size)
        sources.extend(sources1)
    if texts_folder or all:
        # FIXME: Can be any loader types
        loader = DirectoryLoader(texts_folder, glob="./*.txt", loader_cls=TextLoader)
        sources1 = loader.load()
        if chunk:
            sources1 = chunk_sources(sources1, chunk_size=chunk_size)
        sources.extend(sources1)
    if False and all:
        #from langchain.document_loaders import UnstructuredURLLoader
        #loader = UnstructuredURLLoader(urls=urls)
        urls = ["https://www.birdsongsf.com/who-we-are/"]
        from langchain.document_loaders import PlaywrightURLLoader
        loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])
        sources1 = loader.load()
        sources.extend(sources1)
    assert sources, "No sources"

    llm, model_name, streamer = get_llm(use_openai_model=use_openai_model, model_name=model_name, model=model,
                                        tokenizer=tokenizer, stream_output=stream_output)

    db = get_db(sources, use_openai_embedding=use_openai_embedding, db_type=db_type)
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
            #input_variables=["summaries", "question"],
            input_variables=["context", "question"],
            template=template,
        )
        #chain = load_qa_with_sources_chain(llm, prompt=prompt)
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

    chain_kwargs = dict(input_documents=docs, question=query)
    if stream_output:
        answer = None
        assert streamer is not None
        from generate import generate_with_exceptions
        #target = wrapped_partial(generate_with_exceptions, chain, chain_kwargs)
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
                if prompter:# and False:  # FIXME: pipeline can already use prompter
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
        #answer = outputs
    else:
        answer = chain(chain_kwargs)

    if answer is not None:
        print("query: %s" % query, flush=True)
        print("answer: %s" % answer['output_text'], flush=True)
        # link
        answer_sources = [(max(0.0, 1.5 - score)/1.5, get_url(doc)) for score, doc in zip(scores, answer['input_documents'])]
        answer_sources_dict = defaultdict(list)
        [answer_sources_dict[url].append(score) for score, url in answer_sources]
        answers_dict = {}
        for url, scores_url in answer_sources_dict.items():
            answers_dict[url] = np.max(scores_url)
        answer_sources = [(score, url) for url, score in answers_dict.items()]
        answer_sources.sort(key=lambda x: x[0], reverse=True)
        if show_rank:
            #answer_sources = ['%d | %s' % (1 + rank, url) for rank, (score, url) in enumerate(answer_sources)]
            #sorted_sources_urls = "Sources [Rank | Link]:<br>" + "<br>".join(answer_sources)
            answer_sources = ['%s' % url for rank, (score, url) in enumerate(answer_sources)]
            sorted_sources_urls = "Ranked Sources:<br>" + "<br>".join(answer_sources)
        else:
            answer_sources = ['%.2g | %s' % (score, url) for score, url in answer_sources]
            sorted_sources_urls = "Sources [Score | Link]:<br>" + "<br>".join(answer_sources)

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
    return """<a href="file/%s" target="_blank"  rel="noopener noreferrer">%s</a>""" % (x.metadata['source'], x.metadata['source'])


def chunk_sources(sources, chunk_size=1024):
    # allows handling full docs if passed first_para=False
    # NLTK and SPACY can be used instead
    if False:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        # doesn't preserve source
        sources = text_splitter.split_documents(sources)
    else:
        source_chunks = []
        # Below for known separator
        #splitter = CharacterTextSplitter(separator=" ", chunk_size=chunk_size, chunk_overlap=0)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        for source in sources:
            # print(source.metadata['source'], flush=True)
            for chunky in splitter.split_text(source.page_content):
                source_chunks.append(Document(page_content=chunky, metadata=source.metadata))
        sources = source_chunks
    return sources


def test_demo_openai():
    return run_demo(use_openai_model=True, use_openai_embedding=True)


def test_demo_hf():
    return run_demo(use_openai_model=False, use_openai_embedding=False)


def run_demo(use_openai_model=False, use_openai_embedding=False, chat=True, use_chain_ret=False, db_type='faiss'):
    # quick test
    pdf_filename = '1706.03762.pdf'
    if not os.path.isfile(pdf_filename):
        if os.path.isfile('1706.03762'):
            os.remove('1706.03762')
        os.system("wget --user-agent TryToStopMeFromUsingWgetNow https://arxiv.org/pdf/1706.03762")
        os.rename('1706.03762', '1706.03762.pdf')
    sources = pdf_to_sources(pdf_filename=pdf_filename, split_method='chunk')
    db = get_db(sources, use_openai_embedding=use_openai_embedding, db_type=db_type)
    query = "Who created transformers?"
    answer = get_answer_from_db(db, query, chat_history='', use_openai_model=use_openai_model, k=4,
                                chat=chat, use_chain_ret=use_chain_ret)
    print(answer)


def test_demo2_hf():
    return run_demo(use_openai_model=False, use_openai_embedding=False, chat=False, use_chain_ret=True,
                    db_type='chroma')


if __name__ == '__main__':
    test_demo_hf()
