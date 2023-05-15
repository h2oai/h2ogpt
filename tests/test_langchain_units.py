import os
import tempfile

import pytest
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from gpt_langchain import run_qa_db, get_wiki_sources, get_llm, get_answer_from_sources, get_dai_pickle, get_db_from_hf, \
    db_zips, get_some_dbs_from_hf

have_openai_key = os.environ.get('OPENAI_API_KEY') is not None


@pytest.mark.skipif(not have_openai_key, reason="requires OpenAI key to run")
def test_qa_wiki_openai():
    return run_qa_wiki(use_openai_model=True)


def test_qa_wiki_stuff_hf():
    # NOTE: total context length makes things fail when n_sources * text_limit >~ 2048
    return run_qa_wiki(use_openai_model=False, text_limit=256, chain_type='stuff', prompt_type='human_bot')


@pytest.mark.xfail(strict=False,
                   reason="Too long context, improve prompt for map_reduce.  Until then hit: The size of tensor a (2048) must match the size of tensor b (2125) at non-singleton dimension 3")
def test_qa_wiki_map_reduce_hf():
    return run_qa_wiki(use_openai_model=False, text_limit=None, chain_type='map_reduce', prompt_type='human_bot')


def run_qa_wiki(use_openai_model=False, first_para=True, text_limit=None, chain_type='stuff', prompt_type=None):
    sources = get_wiki_sources(first_para=first_para, text_limit=text_limit)
    llm, model_name, streamer = get_llm(use_openai_model=use_openai_model, prompt_type=prompt_type)
    chain = load_qa_with_sources_chain(llm, chain_type=chain_type)

    question = "What are the main differences between Linux and Windows?"
    answer = get_answer_from_sources(chain, sources, question)
    print(answer)


@pytest.mark.skipif(not have_openai_key, reason="requires OpenAI key to run")
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


@pytest.mark.skipif(not have_openai_key, reason="requires OpenAI key to run")
def test_qa_wiki_db_chunk_openai():
    # don't need 256, just seeing how compares to hf
    return run_qa_db(use_openai_model=True, use_openai_embedding=True, text_limit=256, chunk=True, chunk_size=256,
                     wiki=True)


@pytest.mark.skipif(not have_openai_key, reason="requires OpenAI key to run")
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


@pytest.mark.skipif(not have_openai_key, reason="requires OpenAI key to run")
def test_qa_daidocs_db_chunk_openai():
    query = "Which config.toml enables pytorch for NLP?"
    return run_qa_db(query=query, use_openai_model=True, use_openai_embedding=True, text_limit=256, chunk=True,
                     chunk_size=256, wiki=False, dai_rst=True)


@pytest.mark.skipif(not have_openai_key, reason="requires OpenAI key to run")
def test_qa_daidocs_db_chunk_openaiembedding_hfmodel():
    query = "Which config.toml enables pytorch for NLP?"
    return run_qa_db(query=query, use_openai_model=False, use_openai_embedding=True, text_limit=None, chunk=True,
                     chunk_size=128, wiki=False, dai_rst=True)


def test_get_dai_pickle():
    with tempfile.TemporaryDirectory() as tmpdirname:
        get_dai_pickle(dest=tmpdirname)
        assert os.path.isfile(os.path.join(tmpdirname, 'dai_docs.pickle'))


def test_get_dai_db_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        get_some_dbs_from_hf(tmpdirname)


if __name__ == '__main__':
    pass
