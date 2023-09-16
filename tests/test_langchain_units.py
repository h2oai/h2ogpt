import gzip
import io
import json
import os
import shutil
import tempfile
import pytest

from tests.utils import wrap_test_forked, kill_weaviate
from src.enums import DocumentSubset, LangChainAction, LangChainMode, LangChainTypes
from src.gpt_langchain import get_persist_directory
from src.utils import zip_data, download_simple, get_ngpus_vis, get_mem_gpus, have_faiss, remove, get_kwargs

have_openai_key = os.environ.get('OPENAI_API_KEY') is not None
have_replicate_key = os.environ.get('REPLICATE_API_TOKEN') is not None

have_gpus = get_ngpus_vis() > 0

mem_gpus = get_mem_gpus()

# FIXME:
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

db_types = ['chroma', 'weaviate']
db_types_full = ['chroma', 'weaviate', 'faiss']


@pytest.mark.skipif(not have_openai_key, reason="requires OpenAI key to run")
@wrap_test_forked
def test_qa_wiki_openai():
    return run_qa_wiki_fork(use_openai_model=True)


@pytest.mark.need_gpu
@wrap_test_forked
def test_qa_wiki_stuff_hf():
    # NOTE: total context length makes things fail when n_sources * text_limit >~ 2048
    return run_qa_wiki_fork(use_openai_model=False, text_limit=256, chain_type='stuff', prompt_type='human_bot')


@pytest.mark.xfail(strict=False,
                   reason="Too long context, improve prompt for map_reduce.  Until then hit: The size of tensor a (2048) must match the size of tensor b (2125) at non-singleton dimension 3")
@wrap_test_forked
def test_qa_wiki_map_reduce_hf():
    return run_qa_wiki_fork(use_openai_model=False, text_limit=None, chain_type='map_reduce', prompt_type='human_bot')


def run_qa_wiki_fork(*args, **kwargs):
    # disable fork to avoid
    # RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
    # because some other tests use cuda in parent
    # from tests.utils import call_subprocess_onetask
    # return call_subprocess_onetask(run_qa_wiki, args=args, kwargs=kwargs)
    return run_qa_wiki(*args, **kwargs)


def run_qa_wiki(use_openai_model=False, first_para=True, text_limit=None, chain_type='stuff', prompt_type=None):
    from src.gpt_langchain import get_wiki_sources, get_llm
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain

    sources = get_wiki_sources(first_para=first_para, text_limit=text_limit)
    llm, model_name, streamer, prompt_type_out, async_output, only_new_text = \
        get_llm(use_openai_model=use_openai_model, prompt_type=prompt_type, llamacpp_dict={})
    chain = load_qa_with_sources_chain(llm, chain_type=chain_type)

    question = "What are the main differences between Linux and Windows?"
    from src.gpt_langchain import get_answer_from_sources
    answer = get_answer_from_sources(chain, sources, question)
    print(answer)


def check_ret(ret):
    """
    check generator
    :param ret:
    :return:
    """
    rets = []
    for ret1 in ret:
        rets.append(ret1)
        print(ret1)
    assert rets


@pytest.mark.skipif(not have_openai_key, reason="requires OpenAI key to run")
@wrap_test_forked
def test_qa_wiki_db_openai():
    from src.gpt_langchain import _run_qa_db
    query = "What are the main differences between Linux and Windows?"
    langchain_mode = 'wiki'
    ret = _run_qa_db(query=query, use_openai_model=True, use_openai_embedding=True, text_limit=None,
                     hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                     db_type='faiss',
                     langchain_mode_types=dict(langchain_mode=LangChainTypes.SHARED.value),
                     langchain_mode=langchain_mode,
                     langchain_action=LangChainAction.QUERY.value, langchain_agents=[], llamacpp_dict={})
    check_ret(ret)


@pytest.mark.need_gpu
@wrap_test_forked
def test_qa_wiki_db_hf():
    from src.gpt_langchain import _run_qa_db
    # if don't chunk, still need to limit
    # but this case can handle at least more documents, by picking top k
    # FIXME: but spitting out garbage answer right now, all fragmented, or just 1-word answer
    query = "What are the main differences between Linux and Windows?"
    langchain_mode = 'wiki'
    ret = _run_qa_db(query=query, use_openai_model=False, use_openai_embedding=False, text_limit=256,
                     hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                     db_type='faiss',
                     langchain_mode_types=dict(langchain_mode=LangChainTypes.SHARED.value),
                     langchain_mode=langchain_mode,
                     langchain_action=LangChainAction.QUERY.value,
                     langchain_agents=[], llamacpp_dict={})
    check_ret(ret)


@pytest.mark.need_gpu
@wrap_test_forked
def test_qa_wiki_db_chunk_hf():
    from src.gpt_langchain import _run_qa_db
    query = "What are the main differences between Linux and Windows?"
    langchain_mode = 'wiki'
    ret = _run_qa_db(query=query, use_openai_model=False, use_openai_embedding=False, text_limit=256, chunk=True,
                     chunk_size=256,
                     hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                     db_type='faiss',
                     langchain_mode_types=dict(langchain_mode=LangChainTypes.SHARED.value),
                     langchain_mode=langchain_mode,
                     langchain_action=LangChainAction.QUERY.value,
                     langchain_agents=[], llamacpp_dict={})
    check_ret(ret)


@pytest.mark.skipif(not have_openai_key, reason="requires OpenAI key to run")
@wrap_test_forked
def test_qa_wiki_db_chunk_openai():
    from src.gpt_langchain import _run_qa_db
    # don't need 256, just seeing how compares to hf
    query = "What are the main differences between Linux and Windows?"
    langchain_mode = 'wiki'
    ret = _run_qa_db(query=query, use_openai_model=True, use_openai_embedding=True, text_limit=256, chunk=True,
                     chunk_size=256,
                     hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                     db_type='faiss',
                     langchain_mode_types=dict(langchain_mode=LangChainTypes.SHARED.value),
                     langchain_mode=langchain_mode,
                     langchain_action=LangChainAction.QUERY.value,
                     langchain_agents=[], llamacpp_dict={})
    check_ret(ret)


@pytest.mark.skipif(not have_openai_key, reason="requires OpenAI key to run")
@wrap_test_forked
def test_qa_github_db_chunk_openai():
    from src.gpt_langchain import _run_qa_db
    # don't need 256, just seeing how compares to hf
    query = "what is a software defined asset"
    langchain_mode = 'github h2oGPT'
    ret = _run_qa_db(query=query, use_openai_model=True, use_openai_embedding=True, text_limit=256, chunk=True,
                     chunk_size=256,
                     hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                     db_type='faiss',
                     langchain_mode_types=dict(langchain_mode=LangChainTypes.SHARED.value),
                     langchain_mode=langchain_mode,
                     langchain_action=LangChainAction.QUERY.value,
                     langchain_agents=[], llamacpp_dict={})
    check_ret(ret)


@pytest.mark.need_gpu
@wrap_test_forked
def test_qa_daidocs_db_chunk_hf():
    from src.gpt_langchain import _run_qa_db
    # FIXME: doesn't work well with non-instruct-tuned Cerebras
    query = "Which config.toml enables pytorch for NLP?"
    langchain_mode = 'DriverlessAI docs'
    ret = _run_qa_db(query=query, use_openai_model=False, use_openai_embedding=False, text_limit=None, chunk=True,
                     chunk_size=128,
                     hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                     db_type='faiss',
                     langchain_mode_types=dict(langchain_mode=LangChainTypes.SHARED.value),
                     langchain_mode=langchain_mode,
                     langchain_action=LangChainAction.QUERY.value,
                     langchain_agents=[], llamacpp_dict={})
    check_ret(ret)


@pytest.mark.skipif(not have_faiss, reason="requires FAISS")
@wrap_test_forked
def test_qa_daidocs_db_chunk_hf_faiss():
    from src.gpt_langchain import _run_qa_db
    query = "Which config.toml enables pytorch for NLP?"
    # chunk_size is chars for each of k=4 chunks
    langchain_mode = 'DriverlessAI docs'
    ret = _run_qa_db(query=query, use_openai_model=False, use_openai_embedding=False, text_limit=None, chunk=True,
                     chunk_size=128 * 1,  # characters, and if k=4, then 4*4*128 = 2048 chars ~ 512 tokens
                     langchain_mode_types=dict(langchain_mode=LangChainTypes.SHARED.value),
                     langchain_mode=langchain_mode,
                     langchain_action=LangChainAction.QUERY.value,
                     langchain_agents=[],
                     llamacpp_dict={},
                     db_type='faiss',
                     hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                     )
    check_ret(ret)


@pytest.mark.need_gpu
@pytest.mark.parametrize("db_type", db_types)
@pytest.mark.parametrize("top_k_docs", [-1, 3])
@wrap_test_forked
def test_qa_daidocs_db_chunk_hf_dbs(db_type, top_k_docs):
    kill_weaviate(db_type)
    langchain_mode = 'DriverlessAI docs'
    langchain_action = LangChainAction.QUERY.value
    langchain_agents = []
    persist_directory, langchain_type = get_persist_directory(langchain_mode,
                                                              langchain_type=LangChainTypes.SHARED.value)
    assert langchain_type == LangChainTypes.SHARED.value
    remove(persist_directory)
    from src.gpt_langchain import _run_qa_db
    query = "Which config.toml enables pytorch for NLP?"
    # chunk_size is chars for each of k=4 chunks
    if top_k_docs == -1:
        # else OOMs on generation immediately when generation starts, even though only 1600 tokens and 256 new tokens
        model_name = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    else:
        model_name = None
    ret = _run_qa_db(query=query, use_openai_model=False, use_openai_embedding=False, text_limit=None, chunk=True,
                     chunk_size=128 * 1,  # characters, and if k=4, then 4*4*128 = 2048 chars ~ 512 tokens
                     langchain_mode=langchain_mode,
                     langchain_action=langchain_action,
                     langchain_agents=langchain_agents,
                     hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                     db_type=db_type,
                     top_k_docs=top_k_docs,
                     model_name=model_name,
                     llamacpp_dict={},
                     )
    check_ret(ret)


@pytest.mark.need_gpu
@pytest.mark.parametrize("db_type", ['chroma'])
@wrap_test_forked
def test_qa_daidocs_db_chunk_hf_dbs_switch_embedding(db_type):
    # need to get model externally, so don't OOM
    from src.gen import get_model
    base_model = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    prompt_type = 'human_bot'
    all_kwargs = dict(load_8bit=False,
                      load_4bit=False,
                      low_bit_mode=1,
                      load_half=True,
                      load_gptq=False,
                      load_exllama=False,
                      use_safetensors=False,
                      revision=None,
                      use_gpu_id=True,
                      base_model=base_model,
                      tokenizer_base_model=base_model,
                      inference_server='',
                      lora_weights='',
                      gpu_id=0,
                      n_jobs=1,

                      reward_type=False,
                      local_files_only=False,
                      resume_download=True,
                      use_auth_token=False,
                      trust_remote_code=True,
                      offload_folder=None,
                      rope_scaling=None,
                      max_seq_len=None,
                      compile_model=True,
                      llamacpp_dict={},

                      verbose=False)
    model, tokenizer, device = get_model(reward_type=False,
                                         **get_kwargs(get_model, exclude_names=['reward_type'], **all_kwargs))

    langchain_mode = 'DriverlessAI docs'
    langchain_action = LangChainAction.QUERY.value
    langchain_agents = []
    persist_directory, langchain_type = get_persist_directory(langchain_mode,
                                                              langchain_type=LangChainTypes.SHARED.value)
    assert langchain_type == LangChainTypes.SHARED.value
    remove(persist_directory)
    from src.gpt_langchain import _run_qa_db
    query = "Which config.toml enables pytorch for NLP?"
    # chunk_size is chars for each of k=4 chunks
    ret = _run_qa_db(query=query, use_openai_model=False, use_openai_embedding=False,
                     hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                     migrate_embedding_model=True,
                     model=model,
                     tokenizer=tokenizer,
                     model_name=base_model,
                     prompt_type=prompt_type,
                     text_limit=None, chunk=True,
                     chunk_size=128 * 1,  # characters, and if k=4, then 4*4*128 = 2048 chars ~ 512 tokens
                     langchain_mode=langchain_mode,
                     langchain_action=langchain_action,
                     langchain_agents=langchain_agents,
                     db_type=db_type,
                     llamacpp_dict={},
                     )
    check_ret(ret)

    query = "Which config.toml enables pytorch for NLP?"
    # chunk_size is chars for each of k=4 chunks
    ret = _run_qa_db(query=query, use_openai_model=False, use_openai_embedding=False,
                     hf_embedding_model='hkunlp/instructor-large',
                     migrate_embedding_model=True,
                     model=model,
                     tokenizer=tokenizer,
                     model_name=base_model,
                     prompt_type=prompt_type,
                     text_limit=None, chunk=True,
                     chunk_size=128 * 1,  # characters, and if k=4, then 4*4*128 = 2048 chars ~ 512 tokens
                     langchain_mode=langchain_mode,
                     langchain_action=langchain_action,
                     langchain_agents=langchain_agents,
                     db_type=db_type,
                     llamacpp_dict={},
                     )
    check_ret(ret)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_qa_wiki_db_chunk_hf_dbs_llama(db_type):
    kill_weaviate(db_type)
    from src.gpt4all_llm import get_model_tokenizer_gpt4all
    model_name = 'llama'
    model, tokenizer, device = get_model_tokenizer_gpt4all(model_name,
                                                           n_jobs=8,
                                                           max_seq_len=512,
                                                           llamacpp_dict=dict(
                                                               model_path_llama='llama-2-7b-chat.ggmlv3.q8_0.bin',
                                                               n_gpu_layers=100,
                                                               use_mlock=True,
                                                               n_batch=1024))

    from src.gpt_langchain import _run_qa_db
    query = "What are the main differences between Linux and Windows?"
    # chunk_size is chars for each of k=4 chunks
    langchain_mode = 'wiki'
    ret = _run_qa_db(query=query, use_openai_model=False, use_openai_embedding=False, text_limit=None, chunk=True,
                     chunk_size=128 * 1,  # characters, and if k=4, then 4*4*128 = 2048 chars ~ 512 tokens
                     hf_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                     langchain_mode_types=dict(langchain_mode=LangChainTypes.SHARED.value),
                     langchain_mode=langchain_mode,
                     langchain_action=LangChainAction.QUERY.value,
                     langchain_agents=[],
                     db_type=db_type,
                     prompt_type='llama2',
                     langchain_only_model=True,
                     model_name=model_name, model=model, tokenizer=tokenizer,
                     llamacpp_dict=dict(n_gpu_layers=100, use_mlock=True, n_batch=1024),
                     )
    check_ret(ret)


@pytest.mark.skipif(not have_openai_key, reason="requires OpenAI key to run")
@wrap_test_forked
def test_qa_daidocs_db_chunk_openai():
    from src.gpt_langchain import _run_qa_db
    query = "Which config.toml enables pytorch for NLP?"
    langchain_mode = 'DriverlessAI docs'
    ret = _run_qa_db(query=query, use_openai_model=True, use_openai_embedding=True, text_limit=256, chunk=True,
                     db_type='faiss',
                     hf_embedding_model="",
                     chunk_size=256,
                     langchain_mode_types=dict(langchain_mode=LangChainTypes.SHARED.value),
                     langchain_mode=langchain_mode,
                     langchain_action=LangChainAction.QUERY.value,
                     langchain_agents=[], llamacpp_dict={})
    check_ret(ret)


@pytest.mark.skipif(not have_openai_key, reason="requires OpenAI key to run")
@wrap_test_forked
def test_qa_daidocs_db_chunk_openaiembedding_hfmodel():
    from src.gpt_langchain import _run_qa_db
    query = "Which config.toml enables pytorch for NLP?"
    langchain_mode = 'DriverlessAI docs'
    ret = _run_qa_db(query=query, use_openai_model=False, use_openai_embedding=True, text_limit=None, chunk=True,
                     chunk_size=128,
                     hf_embedding_model="",
                     db_type='faiss',
                     langchain_mode_types=dict(langchain_mode=LangChainTypes.SHARED.value),
                     langchain_mode=langchain_mode,
                     langchain_action=LangChainAction.QUERY.value,
                     langchain_agents=[], llamacpp_dict={})
    check_ret(ret)


@pytest.mark.need_tokens
@wrap_test_forked
def test_get_dai_pickle():
    from src.gpt_langchain import get_dai_pickle
    with tempfile.TemporaryDirectory() as tmpdirname:
        get_dai_pickle(dest=tmpdirname)
        assert os.path.isfile(os.path.join(tmpdirname, 'dai_docs.pickle'))


@pytest.mark.need_tokens
@wrap_test_forked
def test_get_dai_db_dir():
    from src.gpt_langchain import get_some_dbs_from_hf
    with tempfile.TemporaryDirectory() as tmpdirname:
        get_some_dbs_from_hf(tmpdirname)


# repeat is to check if first case really deletes, else assert will fail if accumulates wrongly
@pytest.mark.parametrize("repeat", [0, 1])
@pytest.mark.parametrize("db_type", db_types_full)
@wrap_test_forked
def test_make_add_db(repeat, db_type):
    kill_weaviate(db_type)
    from src.gpt_langchain import get_source_files, get_source_files_given_langchain_mode, get_any_db, update_user_db, \
        get_sources, update_and_get_source_files_given_langchain_mode
    from src.make_db import make_db_main
    from src.gpt_langchain import path_to_docs
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            with tempfile.TemporaryDirectory() as tmp_persist_directory_my:
                with tempfile.TemporaryDirectory() as tmp_user_path_my:
                    msg1 = "Hello World"
                    test_file1 = os.path.join(tmp_user_path, 'test.txt')
                    with open(test_file1, "wt") as f:
                        f.write(msg1)
                    chunk = True
                    chunk_size = 512
                    langchain_mode = 'UserData'
                    db, collection_name = make_db_main(persist_directory=tmp_persist_directory,
                                                       user_path=tmp_user_path,
                                                       add_if_exists=False,
                                                       collection_name=langchain_mode,
                                                       fail_any_exception=True, db_type=db_type)
                    assert db is not None
                    docs = db.similarity_search("World")
                    assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
                    assert docs[0].page_content == msg1
                    assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)

                    test_file1my = os.path.join(tmp_user_path_my, 'test.txt')
                    with open(test_file1my, "wt") as f:
                        f.write(msg1)
                    dbmy, collection_namemy = make_db_main(persist_directory=tmp_persist_directory_my,
                                                           user_path=tmp_user_path_my,
                                                           add_if_exists=False,
                                                           collection_name='MyData',
                                                           fail_any_exception=True, db_type=db_type)
                    db1 = {LangChainMode.MY_DATA.value: [dbmy, 'foouuid', 'foousername']}
                    assert dbmy is not None
                    docs1 = dbmy.similarity_search("World")
                    assert len(docs1) == 1 + (1 if db_type == 'chroma' else 0)
                    assert docs1[0].page_content == msg1
                    assert os.path.normpath(docs1[0].metadata['source']) == os.path.normpath(test_file1my)

                    # some db testing for gradio UI/client
                    get_source_files(db=db)
                    get_source_files(db=dbmy)
                    selection_docs_state1 = dict(langchain_modes=[langchain_mode], langchain_mode_paths={},
                                                 langchain_mode_types={})
                    requests_state1 = dict()
                    get_source_files_given_langchain_mode(db1, selection_docs_state1, requests_state1, None,
                                                          langchain_mode, dbs={langchain_mode: db})
                    get_source_files_given_langchain_mode(db1, selection_docs_state1, requests_state1, None,
                                                          langchain_mode='MyData', dbs={})
                    get_any_db(db1, langchain_mode='UserData',
                               langchain_mode_paths=selection_docs_state1['langchain_mode_paths'],
                               langchain_mode_types=selection_docs_state1['langchain_mode_types'],
                               dbs={langchain_mode: db})
                    get_any_db(db1, langchain_mode='MyData',
                               langchain_mode_paths=selection_docs_state1['langchain_mode_paths'],
                               langchain_mode_types=selection_docs_state1['langchain_mode_types'],
                               dbs={})

                    msg1up = "Beefy Chicken"
                    test_file2 = os.path.join(tmp_user_path, 'test2.txt')
                    with open(test_file2, "wt") as f:
                        f.write(msg1up)
                    test_file2_my = os.path.join(tmp_user_path_my, 'test2my.txt')
                    with open(test_file2_my, "wt") as f:
                        f.write(msg1up)
                    kwargs = dict(use_openai_embedding=False,
                                  hf_embedding_model='hkunlp/instructor-large',
                                  migrate_embedding_model=True,
                                  auto_migrate_db=False,
                                  caption_loader=False,
                                  enable_captions=False,
                                  enable_doctr=False,
                                  enable_pix2struct=False,
                                  captions_model="Salesforce/blip-image-captioning-base",
                                  enable_ocr=False,
                                  enable_pdf_ocr='auto',
                                  enable_pdf_doctr=False,
                                  verbose=False,
                                  is_url=False, is_txt=False)
                    langchain_mode2 = 'MyData'
                    selection_docs_state2 = dict(langchain_modes=[langchain_mode2],
                                                 langchain_mode_paths={},
                                                 langchain_mode_types={})
                    requests_state2 = dict()
                    z1, z2, source_files_added, exceptions, last_file = update_user_db(test_file2_my, db1,
                                                                            selection_docs_state2,
                                                                            requests_state2,
                                                                            langchain_mode2,
                                                                            chunk=chunk,
                                                                            chunk_size=chunk_size,
                                                                            dbs={}, db_type=db_type,
                                                                            **kwargs)
                    assert z1 is None
                    assert 'MyData' == z2
                    assert 'test2my' in str(source_files_added)
                    assert len(exceptions) == 0

                    langchain_mode = 'UserData'
                    selection_docs_state1 = dict(langchain_modes=[langchain_mode],
                                                 langchain_mode_paths={langchain_mode: tmp_user_path},
                                                 langchain_mode_types={langchain_mode: LangChainTypes.SHARED.value})
                    z1, z2, source_files_added, exceptions, last_file = update_user_db(test_file2, db1,
                                                                            selection_docs_state1,
                                                                            requests_state1,
                                                                            langchain_mode,
                                                                            chunk=chunk,
                                                                            chunk_size=chunk_size,
                                                                            dbs={langchain_mode: db},
                                                                            db_type=db_type,
                                                                            **kwargs)
                    assert 'test2' in str(source_files_added)
                    assert langchain_mode == z2
                    assert z1 is None
                    docs_state0 = [x.name for x in list(DocumentSubset)]
                    get_sources(db1, selection_docs_state1, {}, langchain_mode, dbs={langchain_mode: db},
                                docs_state0=docs_state0)
                    get_sources(db1, selection_docs_state1, {}, 'MyData', dbs={}, docs_state0=docs_state0)
                    selection_docs_state1['langchain_mode_paths'] = {langchain_mode: tmp_user_path}
                    kwargs2 = dict(first_para=False,
                                   text_limit=None, chunk=chunk, chunk_size=chunk_size,
                                   db_type=db_type,
                                   hf_embedding_model=kwargs['hf_embedding_model'],
                                   migrate_embedding_model=kwargs['migrate_embedding_model'],
                                   auto_migrate_db=kwargs['auto_migrate_db'],
                                   load_db_if_exists=True,
                                   n_jobs=-1, verbose=False)
                    update_and_get_source_files_given_langchain_mode(db1,
                                                                     selection_docs_state1, requests_state1,
                                                                     langchain_mode, dbs={langchain_mode: db},
                                                                     **kwargs2)
                    update_and_get_source_files_given_langchain_mode(db1,
                                                                     selection_docs_state2, requests_state2,
                                                                     'MyData', dbs={}, **kwargs2)

                    assert path_to_docs(test_file2_my, db_type=db_type)[0].metadata['source'] == test_file2_my
                    extra = 1 if db_type == 'chroma' else 0
                    assert os.path.normpath(
                        path_to_docs(os.path.dirname(test_file2_my), db_type=db_type)[1 + extra].metadata[
                            'source']) == os.path.normpath(
                        os.path.abspath(test_file2_my))
                    assert path_to_docs([test_file1, test_file2, test_file2_my], db_type=db_type)[0].metadata[
                               'source'] == test_file1

                    assert path_to_docs(None, url='arxiv:1706.03762', db_type=db_type)[0].metadata[
                               'source'] == 'http://arxiv.org/abs/2002.05202v1'
                    assert path_to_docs(None, url='http://h2o.ai', db_type=db_type)[0].metadata[
                               'source'] == 'http://h2o.ai'

                    assert 'user_paste' in path_to_docs(None,
                                                        text='Yufuu is a wonderful place and you should really visit because there is lots of sun.',
                                                        db_type=db_type)[0].metadata['source']

                if db_type == 'faiss':
                    # doesn't persist
                    return

                # now add using new source path, to original persisted
                with tempfile.TemporaryDirectory() as tmp_user_path3:
                    msg2 = "Jill ran up the hill"
                    test_file2 = os.path.join(tmp_user_path3, 'test2.txt')
                    with open(test_file2, "wt") as f:
                        f.write(msg2)
                    db, collection_name = make_db_main(persist_directory=tmp_persist_directory,
                                                       user_path=tmp_user_path3,
                                                       add_if_exists=True,
                                                       fail_any_exception=True, db_type=db_type,
                                                       collection_name=collection_name)
                    assert db is not None
                    docs = db.similarity_search("World")
                    assert len(docs) == 3 + (1 if db_type == 'chroma' else 0)
                    assert docs[0].page_content == msg1
                    assert docs[1 + extra].page_content in [msg2, msg1up]
                    assert docs[2 + extra].page_content in [msg2, msg1up]
                    assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)

                    docs = db.similarity_search("Jill")
                    assert len(docs) == 3 + (1 if db_type == 'chroma' else 0)
                    assert docs[0].page_content == msg2
                    assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file2)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_zip_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            msg1 = "Hello World"
            test_file1 = os.path.join(tmp_user_path, 'test.txt')
            with open(test_file1, "wt") as f:
                f.write(msg1)
            zip_file = './tmpdata/data.zip'
            zip_data(tmp_user_path, zip_file=zip_file, fail_any_exception=True)
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type,
                                               add_if_exists=False)
            assert db is not None
            docs = db.similarity_search("World")
            assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
            assert docs[0].page_content == msg1
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_url_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        url = 'https://h2o.ai/company/team/leadership-team/'
        db, collection_name = make_db_main(persist_directory=tmp_persist_directory, url=url, fail_any_exception=True,
                                           db_type=db_type)
        assert db is not None
        docs = db.similarity_search("list founding team of h2o.ai")
        assert len(docs) == 4
        assert 'Sri Ambati' in docs[0].page_content


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_urls_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        urls = ['https://h2o.ai/company/team/leadership-team/',
                'https://arxiv.org/abs/1706.03762',
                'https://github.com/h2oai/h2ogpt',
                'https://h2o.ai'
                ]

        db, collection_name = make_db_main(persist_directory=tmp_persist_directory, url=urls,
                                           fail_any_exception=True,
                                           db_type=db_type)
        assert db is not None
        if db_type == 'chroma':
            assert len(db.get()['documents']) > 100
        docs = db.similarity_search("list founding team of h2o.ai")
        assert len(docs) == 4
        assert 'Sri Ambati' in docs[0].page_content


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_urls_file_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            urls = ['https://h2o.ai/company/team/leadership-team/',
                    'https://arxiv.org/abs/1706.03762',
                    'https://github.com/h2oai/h2ogpt',
                    'https://h2o.ai'
                    ]
            with open(os.path.join(tmp_user_path, 'list.urls'), 'wt') as f:
                f.write('\n'.join(urls))

            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, url=urls,
                                               user_path=tmp_user_path,
                                               fail_any_exception=True,
                                               db_type=db_type)
            assert db is not None
            if db_type == 'chroma':
                assert len(db.get()['documents']) > 100
            docs = db.similarity_search("list founding team of h2o.ai")
            assert len(docs) == 4
            assert 'Sri Ambati' in docs[0].page_content


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_html_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            html_content = """
<!DOCTYPE html>
<html>
<body>

<h1>Yugu is a wonderful place</h1>

<p>Animals love to run in the world of Yugu.  They play all day long in the alien sun.</p>

</body>
</html>
"""
            test_file1 = os.path.join(tmp_user_path, 'test.html')
            with open(test_file1, "wt") as f:
                f.write(html_content)
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type,
                                               add_if_exists=False)
            assert db is not None
            docs = db.similarity_search("Yugu")
            assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
            assert 'Yugu' in docs[0].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_docx_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            url = 'https://calibre-ebook.com/downloads/demos/demo.docx'
            test_file1 = os.path.join(tmp_user_path, 'demo.docx')
            download_simple(url, dest=test_file1)
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type)
            assert db is not None
            docs = db.similarity_search("What is calibre DOCX plugin do?")
            assert len(docs) == 4
            assert 'calibre' in docs[0].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_xls_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            test_file1 = os.path.join(tmp_user_path, 'example.xlsx')
            shutil.copy('data/example.xlsx', tmp_user_path)
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type)
            assert db is not None
            docs = db.similarity_search("What is Profit?")
            assert len(docs) == 4
            assert '16604.000' in docs[0].page_content or 'Small Business' in docs[
                0].page_content or 'United States of America' in docs[0].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_md_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            test_file1 = 'README.md'
            if not os.path.isfile(test_file1):
                # see if ran from tests directory
                test_file1 = '../README.md'
                test_file1 = os.path.abspath(test_file1)
            shutil.copy(test_file1, tmp_user_path)
            test_file1 = os.path.join(tmp_user_path, os.path.basename(test_file1))
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type)
            assert db is not None
            docs = db.similarity_search("What is h2oGPT?")
            assert len(docs) == 4
            assert 'Query and summarize your documents' in docs[1].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_rst_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            url = 'https://gist.githubusercontent.com/javiertejero/4585196/raw/21786e2145c0cc0a202ffc4f257f99c26985eaea/README.rst'
            test_file1 = os.path.join(tmp_user_path, 'demo.rst')
            download_simple(url, dest=test_file1)
            test_file1 = os.path.join(tmp_user_path, os.path.basename(test_file1))
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type)
            assert db is not None
            docs = db.similarity_search("Font Faces - Emphasis and Examples")
            assert len(docs) == 4
            assert 'Within paragraphs, inline markup' in docs[0].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_xml_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            url = 'https://gist.githubusercontent.com/theresajayne/1409545/raw/a8b46e7799805e86f4339172c9778fa55afb0f30/gistfile1.txt'
            test_file1 = os.path.join(tmp_user_path, 'demo.xml')
            download_simple(url, dest=test_file1)
            test_file1 = os.path.join(tmp_user_path, os.path.basename(test_file1))
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type)
            assert db is not None
            docs = db.similarity_search("Entrance Hall")
            assert len(docs) == 4 if db_type == 'chroma' else 3
            assert 'Ensuite Bathroom' in docs[0].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_eml_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            url = 'https://raw.githubusercontent.com/FlexConfirmMail/Thunderbird/master/sample.eml'
            test_file1 = os.path.join(tmp_user_path, 'sample.eml')
            download_simple(url, dest=test_file1)
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type,
                                               add_if_exists=False)
            assert db is not None
            docs = db.similarity_search("What is subject?")
            assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
            assert 'testtest' in docs[0].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_simple_eml_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            html_content = """
Date: Sun, 1 Apr 2012 14:25:25 -0600
From: file@fyicenter.com
Subject: Welcome
To: someone@somewhere.com

Dear Friend,

Welcome to file.fyicenter.com!

Sincerely,
FYIcenter.com Team"""
            test_file1 = os.path.join(tmp_user_path, 'test.eml')
            with open(test_file1, "wt") as f:
                f.write(html_content)
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type,
                                               add_if_exists=False)
            assert db is not None
            docs = db.similarity_search("Subject")
            assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
            assert 'Welcome' in docs[0].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_odt_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            url = 'https://github.com/owncloud/example-files/raw/master/Documents/Example.odt'
            test_file1 = os.path.join(tmp_user_path, 'sample.odt')
            download_simple(url, dest=test_file1)
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type)
            assert db is not None
            docs = db.similarity_search("What is ownCloud?")
            assert len(docs) == 4
            assert 'ownCloud' in docs[0].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_pptx_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            url = 'https://www.unm.edu/~unmvclib/powerpoint/pptexamples.ppt'
            test_file1 = os.path.join(tmp_user_path, 'sample.pptx')
            download_simple(url, dest=test_file1)
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type,
                                               add_if_exists=False)
            assert db is not None
            docs = db.similarity_search("Suggestions")
            assert len(docs) == 4
            assert 'Presentation' in docs[0].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_pdf_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            url = 'https://www.africau.edu/images/default/sample.pdf'
            test_file1 = os.path.join(tmp_user_path, 'sample.pdf')
            download_simple(url, dest=test_file1)
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type,
                                               add_if_exists=False)
            assert db is not None
            docs = db.similarity_search("Suggestions")
            assert len(docs) == 3 + (1 if db_type == 'chroma' else 0)
            assert 'And more text. And more text.' in docs[0].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


@pytest.mark.parametrize("enable_pdf_doctr", [False, True])
@pytest.mark.parametrize("enable_pdf_ocr", ['auto', 'on', 'off'])
@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_image_pdf_add(db_type, enable_pdf_ocr, enable_pdf_doctr):
    if enable_pdf_ocr == 'off' and not enable_pdf_doctr:
        return
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            test_file1 = os.path.join('tests', 'CityofTshwaneWater.pdf')
            shutil.copy(test_file1, tmp_user_path)
            test_file1 = os.path.join(tmp_user_path, 'CityofTshwaneWater.pdf')
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type,
                                               enable_pdf_ocr=enable_pdf_ocr,
                                               enable_pdf_doctr=enable_pdf_doctr,
                                               add_if_exists=False)
            assert db is not None
            docs = db.similarity_search("List Tshwane's concerns about water.")
            assert len(docs) == 4
            assert 'we appeal to residents that do have water to please use it sparingly.' in docs[
                1].page_content or 'OFFICE OF THE MMC FOR UTILITIES AND REGIONAL OPERATIONS' in docs[1].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_simple_pptx_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            url = 'https://www.suu.edu/webservices/styleguide/example-files/example.pptx'
            test_file1 = os.path.join(tmp_user_path, 'sample.pptx')
            download_simple(url, dest=test_file1)
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type,
                                               add_if_exists=False)
            assert db is not None
            docs = db.similarity_search("Example")
            assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
            assert 'Powerpoint' in docs[0].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_epub_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            url = 'https://contentserver.adobe.com/store/books/GeographyofBliss_oneChapter.epub'
            test_file1 = os.path.join(tmp_user_path, 'sample.epub')
            download_simple(url, dest=test_file1)
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type,
                                               add_if_exists=False)
            assert db is not None
            docs = db.similarity_search("Grump")
            assert len(docs) == 4
            assert 'happy' in docs[0].page_content or 'happiness' in docs[0].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


@pytest.mark.skip(reason="Not supported, GPL3, and msg-extractor code fails too often")
@pytest.mark.xfail(strict=False,
                   reason="fails with AttributeError: 'Message' object has no attribute '_MSGFile__stringEncoding'. Did you mean: '_MSGFile__overrideEncoding'? even though can use online converter to .eml fine.")
@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_msg_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            url = 'http://file.fyicenter.com/b/sample.msg'
            test_file1 = os.path.join(tmp_user_path, 'sample.msg')
            download_simple(url, dest=test_file1)
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type)
            assert db is not None
            docs = db.similarity_search("Grump")
            assert len(docs) == 4 + (1 if db_type == 'chroma' else 0)
            assert 'Happy' in docs[0].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


os.system('cd tests ; unzip -o driverslicense.jpeg.zip')


@pytest.mark.parametrize("file", ['data/pexels-evg-kowalievska-1170986_small.jpg',
                                  'data/Sample-Invoice-printable.png',
                                  'tests/driverslicense.jpeg.zip',
                                  'tests/driverslicense.jpeg'])
@pytest.mark.parametrize("db_type", db_types)
@pytest.mark.parametrize("enable_pix2struct", [False, True])
@pytest.mark.parametrize("enable_doctr", [False, True])
@pytest.mark.parametrize("enable_ocr", [False, True])
@pytest.mark.parametrize("enable_captions", [False, True])
@pytest.mark.parametrize("pre_load_caption_model", [False, True])
@pytest.mark.parametrize("caption_gpu", [False, True])
@pytest.mark.parametrize("captions_model", [None, 'Salesforce/blip2-flan-t5-xl'])
@wrap_test_forked
def test_png_add(captions_model, caption_gpu, pre_load_caption_model, enable_captions,
                 enable_doctr, enable_pix2struct, enable_ocr, db_type, file):
    if not have_gpus and caption_gpu:
        # if have no GPUs, don't enable caption on GPU
        return
    if not caption_gpu and captions_model == 'Salesforce/blip2-flan-t5-xl':
        # RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
        return
    if not enable_captions and pre_load_caption_model:
        # nothing to preload if not enabling captions
        return
    if captions_model == 'Salesforce/blip2-flan-t5-xl' and not (have_gpus and mem_gpus[0] > 20 * 1024 ** 3):
        # requires GPUs and enough memory to run
        return
    if not (enable_ocr or enable_doctr or enable_pix2struct or enable_captions):
        # nothing enabled for images
        return
    # FIXME (too many permutations):
    if enable_pix2struct and (pre_load_caption_model or enable_captions or enable_ocr or enable_doctr or captions_model or caption_gpu):
        return
    if enable_pix2struct and 'kowalievska' in file:
        # FIXME: Not good for this
        return
    kill_weaviate(db_type)
    return run_png_add(captions_model=captions_model, caption_gpu=caption_gpu,
                       pre_load_caption_model=pre_load_caption_model,
                       enable_captions=enable_captions,
                       enable_ocr=enable_ocr,
                       enable_doctr=enable_doctr,
                       enable_pix2struct=enable_pix2struct,
                       db_type=db_type,
                       file=file)


def run_png_add(captions_model=None, caption_gpu=False,
                pre_load_caption_model=False,
                enable_captions=True,
                enable_ocr=False,
                enable_doctr=False,
                enable_pix2struct=False,
                db_type='chroma',
                file='data/pexels-evg-kowalievska-1170986_small.jpg'):
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            test_file1 = file
            if not os.path.isfile(test_file1):
                # see if ran from tests directory
                test_file1 = os.path.join('../', file)
                assert os.path.isfile(test_file1)
            test_file1 = os.path.abspath(test_file1)
            shutil.copy(test_file1, tmp_user_path)
            test_file1 = os.path.join(tmp_user_path, os.path.basename(test_file1))
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True,
                                               enable_ocr=enable_ocr,
                                               enable_pdf_ocr='auto',
                                               enable_pdf_doctr=False,
                                               caption_gpu=caption_gpu,
                                               pre_load_caption_model=pre_load_caption_model,
                                               captions_model=captions_model,
                                               enable_captions=enable_captions,
                                               enable_doctr=enable_doctr,
                                               enable_pix2struct=enable_pix2struct,
                                               db_type=db_type,
                                               add_if_exists=False,
                                               fail_if_no_sources=False)
            if (enable_captions or enable_pix2struct) and not enable_doctr and not enable_ocr:
                if 'kowalievska' in file:
                    docs = db.similarity_search("cat")
                    assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
                    assert 'a cat sitting on a window' in docs[0].page_content
                    check_source(docs, test_file1)
                elif 'Sample-Invoice-printable' in file:
                    docs = db.similarity_search("invoice")
                    assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
                    # weak test
                    assert 'plumbing' in docs[0].page_content.lower() or 'invoice' in docs[0].page_content.lower()
                    check_source(docs, test_file1)
                else:
                    docs = db.similarity_search("license")
                    assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
                    check_content_captions(docs, captions_model, enable_pix2struct)
                    check_source(docs, test_file1)
            elif not (enable_captions or enable_pix2struct) and not enable_doctr and enable_ocr:
                if 'kowalievska' in file:
                    assert db is None
                elif 'Sample-Invoice-printable' in file:
                    # weak test
                    assert db is not None
                else:
                    docs = db.similarity_search("license")
                    assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
                    check_content_ocr(docs)
                    check_source(docs, test_file1)
            elif not (enable_captions or enable_pix2struct) and enable_doctr and not enable_ocr:
                if 'kowalievska' in file:
                    assert db is None
                elif 'Sample-Invoice-printable' in file:
                    # weak test
                    assert db is not None
                else:
                    docs = db.similarity_search("license")
                    assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
                    check_content_doctr(docs)
                    check_source(docs, test_file1)
            elif not (enable_captions or enable_pix2struct) and enable_doctr and enable_ocr:
                if 'kowalievska' in file:
                    assert db is None
                elif 'Sample-Invoice-printable' in file:
                    # weak test
                    assert db is not None
                else:
                    docs = db.similarity_search("license")
                    assert len(docs) == 2 + (2 if db_type == 'chroma' else 0)
                    check_content_doctr(docs)
                    check_content_ocr(docs)
                    check_source(docs, test_file1)
            elif (enable_captions or enable_pix2struct) and not enable_doctr and enable_ocr:
                if 'kowalievska' in file:
                    docs = db.similarity_search("cat")
                    assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
                    assert 'a cat sitting on a window' in docs[0].page_content
                    check_source(docs, test_file1)
                elif 'Sample-Invoice-printable' in file:
                    # weak test
                    assert db is not None
                else:
                    docs = db.similarity_search("license")
                    assert len(docs) == 2 + (2 if db_type == 'chroma' else 0)
                    check_content_ocr(docs)
                    check_content_captions(docs, captions_model, enable_pix2struct)
                    check_source(docs, test_file1)
            elif (enable_captions or enable_pix2struct) and enable_doctr and not enable_ocr:
                if 'kowalievska' in file:
                    docs = db.similarity_search("cat")
                    assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
                    assert 'a cat sitting on a window' in docs[0].page_content
                    check_source(docs, test_file1)
                elif 'Sample-Invoice-printable' in file:
                    # weak test
                    assert db is not None
                else:
                    docs = db.similarity_search("license")
                    assert len(docs) == 2 + (2 if db_type == 'chroma' else 0)
                    check_content_doctr(docs)
                    check_content_captions(docs, captions_model, enable_pix2struct)
                    check_source(docs, test_file1)
            elif (enable_captions or enable_pix2struct) and enable_doctr and enable_ocr:
                if 'kowalievska' in file:
                    docs = db.similarity_search("cat")
                    assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
                    assert 'a cat sitting on a window' in docs[0].page_content
                    check_source(docs, test_file1)
                elif 'Sample-Invoice-printable' in file:
                    # weak test
                    assert db is not None
                else:
                    if db_type == 'chroma':
                        assert len(db.get()['documents']) == 6
                    docs = db.similarity_search("license")
                    # because search can't find DRIVERLICENSE from DocTR one
                    assert len(docs) == 2 + (2 if db_type == 'chroma' else 1)
                    check_content_ocr(docs)
                    #check_content_doctr(docs)
                    check_content_captions(docs, captions_model, enable_pix2struct)
                    check_source(docs, test_file1)
            else:
                raise NotImplementedError()


def check_content_captions(docs, caption_model, enable_pix2struct):
    assert any(['license' in docs[ix].page_content for ix in range(len(docs))])
    if caption_model is not None and 'blip2' in caption_model:
        str_expected = """california driver license with a woman's face on it california driver license"""
    elif enable_pix2struct:
        str_expected = """California license"""
    else:
        str_expected = """a california driver's license with a picture of a woman's face and a picture of a man's face"""
    assert any([str_expected in docs[ix].page_content for ix in range(len(docs))])


def check_content_doctr(docs):
    assert any(['DRIVERLICENSE' in docs[ix].page_content for ix in range(len(docs))])
    assert any(['California' in docs[ix].page_content for ix in range(len(docs))])
    assert any(['EXP 08/31/2014' in docs[ix].page_content for ix in range(len(docs))])
    assert any(['VETERAN' in docs[ix].page_content for ix in range(len(docs))])


def check_content_ocr(docs):
    # hi_res
    #assert any(['Californias' in docs[ix].page_content for ix in range(len(docs))])
    # ocr_only
    assert any(['DRIVER LICENSE A' in docs[ix].page_content for ix in range(len(docs))])


def check_source(docs, test_file1):
    if test_file1.endswith('.zip'):
        # when zip, adds dir etc.:
        # AssertionError: assert '/tmp/tmp63h5dxxv/driverslicense.jpeg.zip_d7d5f561-6/driverslicense.jpeg' == '/tmp/tmp63h5dxxv/driverslicense.jpeg.zip'
        assert os.path.basename(os.path.normpath(test_file1)) in os.path.normpath(docs[0].metadata['source'])
    else:
        assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_simple_rtf_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            rtf_content = """
{\rtf1\mac\deff2 {\fonttbl{\f0\fswiss Chicago;}{\f2\froman New York;}{\f3\fswiss Geneva;}{\f4\fmodern Monaco;}{\f11\fnil Cairo;}{\f13\fnil Zapf Dingbats;}{\f16\fnil Palatino;}{\f18\fnil Zapf Chancery;}{\f20\froman Times;}{\f21\fswiss Helvetica;}
{\f22\fmodern Courier;}{\f23\ftech Symbol;}{\f24\fnil Mobile;}{\f100\fnil FoxFont;}{\f107\fnil MathMeteor;}{\f164\fnil Futura;}{\f1024\fnil American Heritage;}{\f2001\fnil Arial;}{\f2005\fnil Courier New;}{\f2010\fnil Times New Roman;}
{\f2011\fnil Wingdings;}{\f2515\fnil MT Extra;}{\f3409\fnil FoxPrint;}{\f11132\fnil InsigniaLQmono;}{\f11133\fnil InsigniaLQprop;}{\f14974\fnil LB Helvetica Black;}{\f14976\fnil L Helvetica Light;}}{\colortbl\red0\green0\blue0;\red0\green0\blue255;
\red0\green255\blue255;\red0\green255\blue0;\red255\green0\blue255;\red255\green0\blue0;\red255\green255\blue0;\red255\green255\blue255;}{\stylesheet{\f4\fs18 \sbasedon222\snext0 Normal;}}{\info{\title samplepostscript.msw}{\author 
Computer Science Department}}\widowctrl\ftnbj \sectd \sbknone\linemod0\linex0\cols1\endnhere \pard\plain \qc \f4\fs18 {\plain \b\f21 Sample Rich Text Format Document\par 
}\pard {\plain \f20 \par 
}\pard \ri-80\sl-720\keep\keepn\absw570 {\caps\f20\fs92\dn6 T}{\plain \f20 \par 
}\pard \qj {\plain \f20 his is a sample rich text format (RTF), document. This document was created using Microsoft Word and then printing the document to a RTF file. It illustrates the very basic text formatting effects that can be achieved using RTF. 
\par 
\par 
}\pard \qj\li1440\ri1440\box\brdrs \shading1000 {\plain \f20 RTF }{\plain \b\f20 contains codes for producing advanced editing effects. Such as this indented, boxed, grayed background, entirely boldfaced paragraph.\par 
}\pard \qj {\plain \f20 \par 
Microsoft  Word developed RTF for document transportability and gives a user access to the complete set of the effects that can be achieved using RTF. \par 
}}
"""
            test_file1 = os.path.join(tmp_user_path, 'test.rtf')
            with open(test_file1, "wt") as f:
                f.write(rtf_content)
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type,
                                               add_if_exists=False)
            assert db is not None
            docs = db.similarity_search("How was this document created?")
            assert len(docs) == 4
            assert 'Microsoft' in docs[1].page_content
            assert os.path.normpath(docs[1].metadata['source']) == os.path.normpath(test_file1)


# Windows is not supported with EmbeddedDB. Please upvote the feature request if you want this: https://github.com/weaviate/weaviate-python-client/issues/239
@pytest.mark.parametrize("db_type", ['chroma'])
@wrap_test_forked
def test_url_more_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        url = 'https://edition.cnn.com/2023/08/19/europe/ukraine-f-16s-counteroffensive-intl/index.html'
        db, collection_name = make_db_main(persist_directory=tmp_persist_directory, url=url, fail_any_exception=True,
                                           db_type=db_type)
        assert db is not None
        docs = db.similarity_search("Ukraine")
        assert len(docs) == 4
        assert 'Ukraine' in docs[0].page_content


json_data = {
    "quiz": {
        "sport": {
            "q1": {
                "question": "Which one is correct team name in NBA?",
                "options": [
                    "New York Bulls",
                    "Los Angeles Kings",
                    "Golden State Warriros",
                    "Huston Rocket"
                ],
                "answer": "Huston Rocket"
            }
        },
        "maths": {
            "q1": {
                "question": "5 + 7 = ?",
                "options": [
                    "10",
                    "11",
                    "12",
                    "13"
                ],
                "answer": "12"
            },
            "q2": {
                "question": "12 - 8 = ?",
                "options": [
                    "1",
                    "2",
                    "3",
                    "4"
                ],
                "answer": "4"
            }
        }
    }
}


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_json_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            # too slow:
            # eval_filename = 'ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json'
            # url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/%s" % eval_filename
            test_file1 = os.path.join(tmp_user_path, 'sample.json')
            # download_simple(url, dest=test_file1)

            with open(test_file1, 'wt') as f:
                f.write(json.dumps(json_data))

            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type,
                                               add_if_exists=False)
            assert db is not None
            docs = db.similarity_search("NBA")
            assert len(docs) == 2 if db_type == 'chroma' else 1
            assert 'Bulls' in docs[0].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_jsonl_gz_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            # url = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_spam.messages.jsonl.gz"
            test_file1 = os.path.join(tmp_user_path, 'sample.jsonl.gz')
            # download_simple(url, dest=test_file1)

            with gzip.open(test_file1, 'wb') as f:
                f.write(json.dumps(json_data).encode())

            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type,
                                               add_if_exists=False)
            assert db is not None
            docs = db.similarity_search("NBA")
            assert len(docs) == 2 if db_type == 'chroma' else 1
            assert 'Bulls' in docs[0].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1).replace('.gz', '')


@wrap_test_forked
def test_url_more_subunit():
    url = 'https://edition.cnn.com/2023/08/19/europe/ukraine-f-16s-counteroffensive-intl/index.html'
    from langchain.document_loaders import UnstructuredURLLoader
    docs1 = UnstructuredURLLoader(urls=[url]).load()
    docs1 = [x for x in docs1 if x.page_content]
    assert len(docs1) > 0

    from langchain.document_loaders import PlaywrightURLLoader
    docs1 = PlaywrightURLLoader(urls=[url]).load()
    docs1 = [x for x in docs1 if x.page_content]
    assert len(docs1) > 0

    from langchain.document_loaders import SeleniumURLLoader
    docs1 = SeleniumURLLoader(urls=[url]).load()
    docs1 = [x for x in docs1 if x.page_content]
    assert len(docs1) > 0


if __name__ == '__main__':
    pass
