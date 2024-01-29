import gzip
import io
import json
import os
import shutil
import tempfile
import uuid

import pytest

from src.gen import get_model_retry
from tests.test_client_calls import texts_helium1, texts_helium2, texts_helium3, texts_helium4, texts_helium5, \
    texts_simple
from tests.utils import wrap_test_forked, kill_weaviate, make_user_path_test
from src.enums import DocumentSubset, LangChainAction, LangChainMode, LangChainTypes, DocumentChoice, \
    docs_joiner_default, docs_token_handling_default, db_types, db_types_full
from src.gpt_langchain import get_persist_directory, get_db, get_documents, length_db1, _run_qa_db, split_merge_docs
from src.utils import zip_data, download_simple, get_ngpus_vis, get_mem_gpus, have_faiss, remove, get_kwargs, \
    FakeTokenizer, get_token_count, flatten_list, tar_data

have_openai_key = os.environ.get('OPENAI_API_KEY') is not None
have_replicate_key = os.environ.get('REPLICATE_API_TOKEN') is not None

have_gpus = get_ngpus_vis() > 0

mem_gpus = get_mem_gpus()

# FIXME:
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


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
    llm, model_name, streamer, prompt_type_out, async_output, only_new_text, gradio_server = \
        get_llm(use_openai_model=use_openai_model, prompt_type=prompt_type, llamacpp_dict={},
                exllama_dict={})
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
    return rets


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
    kill_weaviate(db_type)


def get_test_model(base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b',
                   tokenizer_base_model='',
                   prompt_type='human_bot',
                   inference_server='',
                   max_seq_len=None,
                   regenerate_clients=True):
    # need to get model externally, so don't OOM
    from src.gen import get_model
    all_kwargs = dict(load_8bit=False,
                      load_4bit=False,
                      low_bit_mode=1,
                      load_half=True,
                      load_gptq='',
                      use_autogptq=False,
                      load_awq='',
                      load_exllama=False,
                      use_safetensors=False,
                      revision=None,
                      use_gpu_id=True,
                      base_model=base_model,
                      tokenizer_base_model=tokenizer_base_model,
                      inference_server=inference_server,
                      regenerate_clients=regenerate_clients,
                      lora_weights='',
                      gpu_id=0,
                      n_jobs=1,
                      n_gpus=None,

                      reward_type=False,
                      local_files_only=False,
                      resume_download=True,
                      use_auth_token=False,
                      trust_remote_code=True,
                      offload_folder=None,
                      rope_scaling=None,
                      max_seq_len=max_seq_len,
                      compile_model=True,
                      llamacpp_dict={},
                      exllama_dict={},
                      gptq_dict={},
                      attention_sinks=False,
                      sink_dict={},
                      truncation_generation=False,
                      hf_model_dict={},
                      use_flash_attention_2=False,
                      llamacpp_path='llamacpp_path',

                      verbose=False)
    model, tokenizer, device = get_model_retry(reward_type=False,
                                               **get_kwargs(get_model, exclude_names=['reward_type'], **all_kwargs))
    return model, tokenizer, base_model, prompt_type


@pytest.mark.need_gpu
@pytest.mark.parametrize("db_type", ['chroma'])
@wrap_test_forked
def test_qa_daidocs_db_chunk_hf_dbs_switch_embedding(db_type):
    model, tokenizer, base_model, prompt_type = get_test_model()

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
                                                               model_path_llama='https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf?download=true',
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
    kill_weaviate(db_type)


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
                                  doctr_loader=False,
                                  asr_loader=False,
                                  enable_captions=False,
                                  enable_doctr=False,
                                  enable_pix2struct=False,
                                  enable_llava=False,
                                  enable_transcriptions=False,
                                  captions_model="Salesforce/blip-image-captioning-base",
                                  llava_model=None,
                                  llava_prompt=None,
                                  asr_model='openai/whisper-medium',
                                  enable_ocr=False,
                                  enable_pdf_ocr='auto',
                                  enable_pdf_doctr=False,
                                  gradio_upload_to_chatbot_num_max=1,
                                  verbose=False,
                                  is_url=False, is_txt=False,
                                  allow_upload_to_my_data=True,
                                  allow_upload_to_user_data=True,
                                  )
                    langchain_mode2 = 'MyData'
                    selection_docs_state2 = dict(langchain_modes=[langchain_mode2],
                                                 langchain_mode_paths={},
                                                 langchain_mode_types={})
                    requests_state2 = dict()
                    z1, z2, source_files_added, exceptions, last_file, last_dict = update_user_db(test_file2_my, db1,
                                                                                                  selection_docs_state2,
                                                                                                  requests_state2,
                                                                                                  langchain_mode2,
                                                                                                  chunk=chunk,
                                                                                                  chunk_size=chunk_size,
                                                                                                  dbs={},
                                                                                                  db_type=db_type,
                                                                                                  **kwargs)
                    assert z1 is None
                    assert 'MyData' == z2
                    assert 'test2my' in str(source_files_added)
                    assert len(exceptions) == 0

                    langchain_mode = 'UserData'
                    selection_docs_state1 = dict(langchain_modes=[langchain_mode],
                                                 langchain_mode_paths={langchain_mode: tmp_user_path},
                                                 langchain_mode_types={langchain_mode: LangChainTypes.SHARED.value})
                    z1, z2, source_files_added, exceptions, last_file, last_dict = update_user_db(test_file2, db1,
                                                                                                  selection_docs_state1,
                                                                                                  requests_state1,
                                                                                                  langchain_mode,
                                                                                                  chunk=chunk,
                                                                                                  chunk_size=chunk_size,
                                                                                                  dbs={
                                                                                                      langchain_mode: db},
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
                               'source'] == 'http://arxiv.org/abs/1706.03762v7'
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
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


@pytest.mark.parametrize("db_type", db_types)
@pytest.mark.parametrize("tar_type", ["tar.gz", "tgz"])
@wrap_test_forked
def test_tar_add(db_type, tar_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            msg1 = "Hello World"
            test_file1 = os.path.join(tmp_user_path, 'test.txt')
            with open(test_file1, "wt") as f:
                f.write(msg1)
            tar_file = f'./tmpdata/data.{tar_type}'
            tar_data(tmp_user_path, tar_file=tar_file, fail_any_exception=True)
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type,
                                               add_if_exists=False)
            assert db is not None
            docs = db.similarity_search("World")
            assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
            assert docs[0].page_content == msg1
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


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
            assert '16185' in docs[0].page_content or \
                   'Small Business' in docs[0].page_content or \
                   'United States of America' in docs[0].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)
    kill_weaviate(db_type)


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
            assert 'Query and summarize your documents' in docs[1].page_content or 'document Q/A' in docs[
                1].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


@pytest.mark.parametrize("use_pypdf", ['auto', 'on', 'off'])
@pytest.mark.parametrize("use_unstructured_pdf", ['auto', 'on', 'off'])
@pytest.mark.parametrize("use_pymupdf", ['auto', 'on', 'off'])
@pytest.mark.parametrize("enable_pdf_doctr", ['auto', 'on', 'off'])
@pytest.mark.parametrize("enable_pdf_ocr", ['auto', 'on', 'off'])
@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_pdf_add(db_type, enable_pdf_ocr, enable_pdf_doctr, use_pymupdf, use_unstructured_pdf, use_pypdf):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            if True:
                if False:
                    url = 'https://www.africau.edu/images/default/sample.pdf'
                    test_file1 = os.path.join(tmp_user_path, 'sample.pdf')
                    download_simple(url, dest=test_file1)
                else:
                    test_file1 = os.path.join(tmp_user_path, 'sample2.pdf')
                    shutil.copy(os.path.join('tests', 'sample.pdf'), test_file1)
            else:
                if False:
                    name = 'CityofTshwaneWater.pdf'
                    location = "tests"
                else:
                    name = '555_593.pdf'
                    location = '/home/jon/Downloads/'

                test_file1 = os.path.join(location, name)
                shutil.copy(test_file1, tmp_user_path)
                test_file1 = os.path.join(tmp_user_path, name)

            default_mode = use_pymupdf in ['auto', 'on'] and \
                           use_pypdf in ['auto'] and \
                           use_unstructured_pdf in ['auto'] and \
                           enable_pdf_doctr in ['off', 'auto'] and \
                           enable_pdf_ocr in ['off', 'auto']
            no_doc_mode = use_pymupdf in ['off'] and \
                          use_pypdf in ['off'] and \
                          use_unstructured_pdf in ['off'] and \
                          enable_pdf_doctr in ['off'] and \
                          enable_pdf_ocr in ['off', 'auto']

            try:
                db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                                   fail_any_exception=True, db_type=db_type,
                                                   use_pymupdf=use_pymupdf,
                                                   enable_pdf_ocr=enable_pdf_ocr,
                                                   enable_pdf_doctr=enable_pdf_doctr,
                                                   use_unstructured_pdf=use_unstructured_pdf,
                                                   use_pypdf=use_pypdf,
                                                   add_if_exists=False)
            except Exception as e:
                if 'had no valid text and no meta data was parsed' in str(
                        e) or 'had no valid text, but meta data was parsed' in str(e):
                    if no_doc_mode:
                        return
                    else:
                        raise
                raise

            assert db is not None
            docs = db.similarity_search("Suggestions")
            if default_mode:
                assert len(docs) == 3 + (1 if db_type == 'chroma' else 0) or len(docs) == 4  # weaviate madness
            else:
                # ocr etc. end up with different pages, overly complex to test exact count
                assert len(docs) >= 2
            assert 'And more text. And more text.' in docs[0].page_content
            if db_type == 'weaviate':
                assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1) or os.path.basename(
                    docs[0].metadata['source']) == os.path.basename(test_file1)
            else:
                assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)
    kill_weaviate(db_type)


@pytest.mark.parametrize("use_pypdf", ['auto', 'on', 'off'])
@pytest.mark.parametrize("use_unstructured_pdf", ['auto', 'on', 'off'])
@pytest.mark.parametrize("use_pymupdf", ['auto', 'on', 'off'])
@pytest.mark.parametrize("enable_pdf_doctr", ['auto', 'on', 'off'])
@pytest.mark.parametrize("enable_pdf_ocr", ['auto', 'on', 'off'])
@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_image_pdf_add(db_type, enable_pdf_ocr, enable_pdf_doctr, use_pymupdf, use_unstructured_pdf, use_pypdf):
    if enable_pdf_ocr == 'off' and not enable_pdf_doctr:
        return
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            name = 'CityofTshwaneWater.pdf'
            location = "tests"
            test_file1 = os.path.join(location, name)
            shutil.copy(test_file1, tmp_user_path)
            test_file1 = os.path.join(tmp_user_path, name)

            str_test = [db_type, enable_pdf_ocr, enable_pdf_doctr, use_pymupdf, use_unstructured_pdf, use_pypdf]
            str_test = [str(x) for x in str_test]
            str_test = '-'.join(str_test)

            default_mode = use_pymupdf in ['auto', 'on'] and \
                           use_pypdf in ['off', 'auto'] and \
                           use_unstructured_pdf in ['auto'] and \
                           enable_pdf_doctr in ['off', 'auto'] and \
                           enable_pdf_ocr in ['off', 'auto']
            no_doc_mode = use_pymupdf in ['off'] and \
                          use_pypdf in ['off'] and \
                          use_unstructured_pdf in ['off'] and \
                          enable_pdf_doctr in ['off'] and \
                          enable_pdf_ocr in ['off', 'auto']
            no_docs = ['off-off-auto-off-auto', 'off-off-on-off-on', 'off-off-auto-off-off', 'off-off-off-off-auto',
                       'off-off-on-off-off', 'off-off-on-off-auto', 'off-off-auto-off-on', 'off-off-off-off-on',

                       ]
            no_doc_mode |= any([x in str_test for x in no_docs])

            try:
                db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                                   fail_any_exception=True, db_type=db_type,
                                                   use_pymupdf=use_pymupdf,
                                                   enable_pdf_ocr=enable_pdf_ocr,
                                                   enable_pdf_doctr=enable_pdf_doctr,
                                                   use_unstructured_pdf=use_unstructured_pdf,
                                                   use_pypdf=use_pypdf,
                                                   add_if_exists=False)
            except Exception as e:
                if 'had no valid text and no meta data was parsed' in str(
                        e) or 'had no valid text, but meta data was parsed' in str(e):
                    if no_doc_mode:
                        return
                    else:
                        raise
                raise

            if default_mode:
                assert db is not None
                docs = db.similarity_search("List Tshwane's concerns about water.")
                assert len(docs) == 4
                assert 'we appeal to residents that do have water to please use it sparingly.' in docs[
                    1].page_content or 'OFFICE OF THE MMC FOR UTILITIES AND REGIONAL' in docs[1].page_content
            else:

                assert db is not None
                docs = db.similarity_search("List Tshwane's concerns about water.")
                assert len(docs) >= 2
                assert docs[0].page_content
                assert docs[1].page_content
            if db_type == 'weaviate':
                assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1) or os.path.basename(
                    docs[0].metadata['source']) == os.path.basename(test_file1)
            else:
                assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


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
@pytest.mark.parametrize("pre_load_image_audio_models", [False, True])
@pytest.mark.parametrize("caption_gpu", [False, True])
@pytest.mark.parametrize("captions_model", [None, 'Salesforce/blip2-flan-t5-xl'])
@wrap_test_forked
@pytest.mark.parallel10
def test_png_add(captions_model, caption_gpu, pre_load_image_audio_models, enable_captions,
                 enable_doctr, enable_pix2struct, enable_ocr, db_type, file):
    if not have_gpus and caption_gpu:
        # if have no GPUs, don't enable caption on GPU
        return
    if not caption_gpu and captions_model == 'Salesforce/blip2-flan-t5-xl':
        # RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
        return
    if not enable_captions and pre_load_image_audio_models:
        # nothing to preload if not enabling captions
        return
    if captions_model == 'Salesforce/blip2-flan-t5-xl' and not (have_gpus and mem_gpus[0] > 20 * 1024 ** 3):
        # requires GPUs and enough memory to run
        return
    if not (enable_ocr or enable_doctr or enable_pix2struct or enable_captions):
        # nothing enabled for images
        return
    # FIXME (too many permutations):
    if enable_pix2struct and (
            pre_load_image_audio_models or enable_captions or enable_ocr or enable_doctr or captions_model or caption_gpu):
        return
    if enable_pix2struct and 'kowalievska' in file:
        # FIXME: Not good for this
        return
    kill_weaviate(db_type)
    try:
        return run_png_add(captions_model=captions_model, caption_gpu=caption_gpu,
                           pre_load_image_audio_models=pre_load_image_audio_models,
                           enable_captions=enable_captions,
                           enable_ocr=enable_ocr,
                           enable_doctr=enable_doctr,
                           enable_pix2struct=enable_pix2struct,
                           db_type=db_type,
                           file=file)
    except Exception as e:
        if not enable_captions and 'data/pexels-evg-kowalievska-1170986_small.jpg' in file and 'had no valid text and no meta data was parsed' in str(
                e):
            pass
        else:
            raise
    kill_weaviate(db_type)


def run_png_add(captions_model=None, caption_gpu=False,
                pre_load_image_audio_models=False,
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
                                               pre_load_image_audio_models=pre_load_image_audio_models,
                                               captions_model=captions_model,
                                               enable_captions=enable_captions,
                                               enable_doctr=enable_doctr,
                                               enable_pix2struct=enable_pix2struct,
                                               db_type=db_type,
                                               add_if_exists=False,
                                               fail_if_no_sources=False)
            if (enable_captions or enable_pix2struct) and not enable_doctr and not enable_ocr:
                if 'kowalievska' in file:
                    docs = db.similarity_search("cat", k=10)
                    assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
                    assert 'a cat sitting on a window' in docs[0].page_content
                    check_source(docs, test_file1)
                elif 'Sample-Invoice-printable' in file:
                    docs = db.similarity_search("invoice", k=10)
                    assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
                    # weak test
                    assert 'plumbing' in docs[0].page_content.lower() or 'invoice' in docs[0].page_content.lower()
                    check_source(docs, test_file1)
                else:
                    docs = db.similarity_search("license", k=10)
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
                    docs = db.similarity_search("license", k=10)
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
                    docs = db.similarity_search("license", k=10)
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
                    docs = db.similarity_search("license", k=10)
                    assert len(docs) == 2 + (2 if db_type == 'chroma' else 0)
                    check_content_doctr(docs)
                    check_content_ocr(docs)
                    check_source(docs, test_file1)
            elif (enable_captions or enable_pix2struct) and not enable_doctr and enable_ocr:
                if 'kowalievska' in file:
                    docs = db.similarity_search("cat", k=10)
                    assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
                    assert 'a cat sitting on a window' in docs[0].page_content
                    check_source(docs, test_file1)
                elif 'Sample-Invoice-printable' in file:
                    # weak test
                    assert db is not None
                else:
                    docs = db.similarity_search("license", k=10)
                    assert len(docs) == 2 + (2 if db_type == 'chroma' else 0)
                    check_content_ocr(docs)
                    check_content_captions(docs, captions_model, enable_pix2struct)
                    check_source(docs, test_file1)
            elif (enable_captions or enable_pix2struct) and enable_doctr and not enable_ocr:
                if 'kowalievska' in file:
                    docs = db.similarity_search("cat", k=10)
                    assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
                    assert 'a cat sitting on a window' in docs[0].page_content
                    check_source(docs, test_file1)
                elif 'Sample-Invoice-printable' in file:
                    # weak test
                    assert db is not None
                else:
                    docs = db.similarity_search("license", k=10)
                    assert len(docs) == 2 + (2 if db_type == 'chroma' else 0)
                    check_content_doctr(docs)
                    check_content_captions(docs, captions_model, enable_pix2struct)
                    check_source(docs, test_file1)
            elif (enable_captions or enable_pix2struct) and enable_doctr and enable_ocr:
                if 'kowalievska' in file:
                    docs = db.similarity_search("cat", k=10)
                    assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
                    assert 'a cat sitting on a window' in docs[0].page_content
                    check_source(docs, test_file1)
                elif 'Sample-Invoice-printable' in file:
                    # weak test
                    assert db is not None
                else:
                    if db_type == 'chroma':
                        assert len(db.get()['documents']) == 6
                    docs = db.similarity_search("license", k=10)
                    # because search can't find DRIVERLICENSE from DocTR one
                    assert len(docs) == 4 + (2 if db_type == 'chroma' else 1)
                    check_content_ocr(docs)
                    # check_content_doctr(docs)
                    check_content_captions(docs, captions_model, enable_pix2struct)
                    check_source(docs, test_file1)
            else:
                raise NotImplementedError()


def check_content_captions(docs, captions_model, enable_pix2struct):
    assert any(['license' in docs[ix].page_content.lower() for ix in range(len(docs))])
    if captions_model is not None and 'blip2' in captions_model:
        str_expected = """california driver license with a woman's face on it california driver license"""
    elif enable_pix2struct:
        str_expected = """california license"""
    else:
        str_expected = """a california driver's license with a picture of a woman's face and a picture of a man's face"""
    assert any([str_expected in docs[ix].page_content.lower() for ix in range(len(docs))])


def check_content_doctr(docs):
    assert any(['DRIVER LICENSE' in docs[ix].page_content for ix in range(len(docs))])
    assert any(['California' in docs[ix].page_content for ix in range(len(docs))])
    assert any(['ExP08/31/2014' in docs[ix].page_content for ix in range(len(docs))])
    assert any(['VETERAN' in docs[ix].page_content for ix in range(len(docs))])


def check_content_ocr(docs):
    # hi_res
    # assert any(['Californias' in docs[ix].page_content for ix in range(len(docs))])
    # ocr_only
    assert any(['DRIVER LICENSE' in docs[ix].page_content for ix in range(len(docs))])


def check_source(docs, test_file1):
    if test_file1.endswith('.zip'):
        # when zip, adds dir etc.:
        # AssertionError: assert '/tmp/tmp63h5dxxv/driverslicense.jpeg.zip_d7d5f561-6/driverslicense.jpeg' == '/tmp/tmp63h5dxxv/driverslicense.jpeg.zip'
        assert os.path.basename(os.path.normpath(test_file1)) in os.path.normpath(docs[0].metadata['source'])
    else:
        assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)


@pytest.mark.parametrize("image_file", ['./models/anthropic.png', 'data/pexels-evg-kowalievska-1170986_small.jpg'])
@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_llava_add(image_file, db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            file = os.path.basename(image_file)
            test_file1 = os.path.join(tmp_user_path, file)
            shutil.copy(image_file, test_file1)

            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type,
                                               add_if_exists=False,
                                               enable_llava=True,
                                               llava_model=os.getenv('H2OGPT_LLAVA_MODEL', 'http://192.168.1.46:7861'),
                                               llava_prompt=None,
                                               enable_doctr=False,
                                               enable_captions=False,
                                               enable_ocr=False,
                                               enable_transcriptions=False,
                                               enable_pdf_ocr=False,
                                               enable_pdf_doctr=False,
                                               enable_pix2struct=False,
                                               )
            assert db is not None
            if 'anthropic' in image_file:
                docs = db.similarity_search("circle")
                assert len(docs) == 2 if db_type == 'chroma' else 1
                assert 'letter "A"' in docs[0].page_content
            else:
                docs = db.similarity_search("cat")
                assert len(docs) == 2 if db_type == 'chroma' else 1
                assert 'cat' in docs[0].page_content
                assert 'birds' in docs[0].page_content or 'outdoors' in docs[0].page_content or 'outside' in docs[
                    0].page_content
            assert os.path.normpath(docs[0].metadata['source']) == os.path.normpath(test_file1)
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


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
    kill_weaviate(db_type)


@wrap_test_forked
def test_url_more_subunit():
    url = 'https://edition.cnn.com/2023/08/19/europe/ukraine-f-16s-counteroffensive-intl/index.html'
    from langchain.document_loaders import UnstructuredURLLoader
    docs1 = UnstructuredURLLoader(urls=[url]).load()
    docs1 = [x for x in docs1 if x.page_content]
    assert len(docs1) > 0

    # Playwright and Selenium fails on cnn url
    url_easy = 'https://github.com/h2oai/h2ogpt'

    from langchain.document_loaders import PlaywrightURLLoader
    docs1 = PlaywrightURLLoader(urls=[url_easy]).load()
    docs1 = [x for x in docs1 if x.page_content]
    assert len(docs1) > 0

    from langchain.document_loaders import SeleniumURLLoader
    docs1 = SeleniumURLLoader(urls=[url_easy]).load()
    docs1 = [x for x in docs1 if x.page_content]
    assert len(docs1) > 0


@wrap_test_forked
@pytest.mark.parametrize("db_type", db_types_full)
@pytest.mark.parametrize("num", [1000, 100000])
def test_many_text(db_type, num):
    from langchain.docstore.document import Document

    sources = [Document(page_content=str(i)) for i in range(0, num)]
    hf_embedding_model = "fake"
    # hf_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    # hf_embedding_model = 'hkunlp/instructor-large'
    db = get_db(sources, db_type=db_type, langchain_mode='ManyTextData', hf_embedding_model=hf_embedding_model)
    documents = get_documents(db)['documents']
    assert len(documents) == num


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_youtube_audio_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            url = 'https://www.youtube.com/watch?v=cwjs1WAG9CM'
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, url=url,
                                               fail_any_exception=True, db_type=db_type,
                                               add_if_exists=False,
                                               extract_frames=0)
            assert db is not None
            docs = db.similarity_search("Example")
            assert len(docs) == 3 + (1 if db_type == 'chroma' else 0) or len(docs) == 4
            assert 'structured output' in docs[0].page_content
            assert url in docs[0].metadata['source']
    kill_weaviate(db_type)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_youtube_full_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            url = 'https://www.youtube.com/shorts/JjdqlglRxrU'
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, url=url,
                                               fail_any_exception=True, db_type=db_type,
                                               add_if_exists=False)
            assert db is not None
            docs = db.similarity_search("cat")
            assert len(docs) >= 2
            assert 'couch' in str([x.page_content for x in docs])
            assert url in docs[0].metadata['source'] or url in docs[0].metadata['original_source']
            docs = db.similarity_search("cat", 100)
            assert 'So I heard if you give a cat an egg' in str([x.page_content for x in docs])
    kill_weaviate(db_type)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_mp3_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            test_file1 = os.path.join(tmp_user_path, 'sample.mp3.zip')
            shutil.copy('tests/porsche.mp3.zip', test_file1)
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type)
            assert db is not None
            docs = db.similarity_search("Porsche")
            assert len(docs) == 1 + (1 if db_type == 'chroma' else 0)
            assert 'Porsche Macan' in docs[0].page_content
            assert 'porsche.mp3' in os.path.normpath(docs[0].metadata['source'])
    kill_weaviate(db_type)


@pytest.mark.parametrize("db_type", db_types)
@wrap_test_forked
def test_mp4_add(db_type):
    kill_weaviate(db_type)
    from src.make_db import make_db_main
    with tempfile.TemporaryDirectory() as tmp_persist_directory:
        with tempfile.TemporaryDirectory() as tmp_user_path:
            url = 'https://h2o-release.s3.amazonaws.com/h2ogpt/iG_jeMeUPBnUO6sx.mp4'
            test_file1 = os.path.join(tmp_user_path, 'demo.mp4')
            download_simple(url, dest=test_file1)
            db, collection_name = make_db_main(persist_directory=tmp_persist_directory, user_path=tmp_user_path,
                                               fail_any_exception=True, db_type=db_type,
                                               enable_captions=True)
            assert db is not None
            docs = db.similarity_search("Gemini")
            assert len(docs) >= 3
            assert 'Gemini' in str([x.page_content for x in docs])
            assert 'demo.mp4' in os.path.normpath(docs[0].metadata['source'])
            docs = db.similarity_search("AI", 100)
            assert 'fun birthday party' in str([x.page_content for x in docs])
            assert 'Gemini tries to design' in str([x.page_content for x in docs])
            assert 'H2OAudioCaptionLoader' in str([x.metadata for x in docs])
            assert 'H2OImageCaptionLoader' in str([x.metadata for x in docs])
            assert '.jpg' in str([x.metadata for x in docs])
    kill_weaviate(db_type)


@wrap_test_forked
def test_chroma_filtering():
    # get test model so don't have to reload it each time
    model, tokenizer, base_model, prompt_type = get_test_model()

    # generic settings true for all cases
    requests_state1 = {'username': 'foo'}
    verbose1 = True
    max_raw_chunks = None
    api = False
    n_jobs = -1
    db_type1 = 'chroma'
    load_db_if_exists1 = True
    use_openai_embedding1 = False
    migrate_embedding_model_or_db1 = False
    auto_migrate_db1 = False

    def get_userid_auth_fake(requests_state1, auth_filename=None, auth_access=None, guest_name=None, **kwargs):
        return str(uuid.uuid4())

    other_kwargs = dict(load_db_if_exists1=load_db_if_exists1,
                        db_type1=db_type1,
                        use_openai_embedding1=use_openai_embedding1,
                        migrate_embedding_model_or_db1=migrate_embedding_model_or_db1,
                        auto_migrate_db1=auto_migrate_db1,
                        verbose1=verbose1,
                        get_userid_auth1=get_userid_auth_fake,
                        max_raw_chunks=max_raw_chunks,
                        api=api,
                        n_jobs=n_jobs,
                        enforce_h2ogpt_api_key=False,
                        enforce_h2ogpt_ui_key=False,
                        )
    mydata_mode1 = LangChainMode.MY_DATA.value
    from src.make_db import make_db_main

    for chroma_new in [False, True]:
        print("chroma_new: %s" % chroma_new, flush=True)
        if chroma_new:
            # fresh, so chroma >= 0.4
            user_path = make_user_path_test()
            from langchain.vectorstores import Chroma
            db, collection_name = make_db_main(user_path=user_path)
            assert isinstance(db, Chroma)

            hf_embedding_model = 'hkunlp/instructor-xl'
            langchain_mode1 = collection_name
            query = 'What is h2oGPT?'
        else:
            # old, was with chroma < 0.4
            # has no user_path
            db, collection_name = make_db_main(download_some=True)
            from src.gpt_langchain import ChromaMig
            assert isinstance(db, ChromaMig)
            assert ChromaMig.__name__ in str(db)
            query = 'What is whisper?'

            hf_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            langchain_mode1 = collection_name

        db1s = {langchain_mode1: [None] * length_db1(), mydata_mode1: [None] * length_db1()}

        dbs1 = {langchain_mode1: db}
        langchain_modes = [langchain_mode1]
        langchain_mode_paths = dict(langchain_mode1=None)
        langchain_mode_types = dict(langchain_modes='shared')
        selection_docs_state1 = dict(langchain_modes=langchain_modes,
                                     langchain_mode_paths=langchain_mode_paths,
                                     langchain_mode_types=langchain_mode_types)

        run_db_kwargs = dict(query=query,
                             db=db,
                             use_openai_model=False, use_openai_embedding=False, text_limit=None,
                             hf_embedding_model=hf_embedding_model,
                             db_type=db_type1,
                             langchain_mode_paths=langchain_mode_paths,
                             langchain_mode_types=langchain_mode_types,
                             langchain_mode=langchain_mode1,
                             langchain_agents=[],
                             llamacpp_dict={},

                             model=model,
                             tokenizer=tokenizer,
                             model_name=base_model,
                             prompt_type=prompt_type,

                             top_k_docs=10,  # 4 leaves out docs for test in some cases, so use 10
                             cut_distance=1.8,  # default leaves out some docs in some cases
                             )

        # GET_CHAIN etc.
        for answer_with_sources in [-1, True]:
            print("answer_with_sources: %s" % answer_with_sources, flush=True)
            # mimic nochat-API or chat-UI
            append_sources_to_answer = answer_with_sources != -1
            for doc_choice in ['All', 1, 2]:
                if doc_choice == 'All':
                    document_choice = [DocumentChoice.ALL.value]
                else:
                    docs = [x['source'] for x in db.get()['metadatas']]
                    if doc_choice == 1:
                        document_choice = docs[:doc_choice]
                    else:
                        # ensure don't get dup
                        docs = sorted(set(docs))
                        document_choice = docs[:doc_choice]
                print("doc_choice: %s" % doc_choice, flush=True)
                for langchain_action in [LangChainAction.QUERY.value, LangChainAction.SUMMARIZE_MAP.value]:
                    print("langchain_action: %s" % langchain_action, flush=True)
                    for document_subset in [DocumentSubset.Relevant.name, DocumentSubset.TopKSources.name,
                                            DocumentSubset.RelSources.name]:
                        print("document_subset: %s" % document_subset, flush=True)

                        ret = _run_qa_db(**run_db_kwargs,
                                         langchain_action=langchain_action,
                                         document_subset=document_subset,
                                         document_choice=document_choice,
                                         answer_with_sources=answer_with_sources,
                                         append_sources_to_answer=append_sources_to_answer,
                                         )
                        rets = check_ret(ret)
                        rets1 = rets[0]
                        if chroma_new:
                            if answer_with_sources == -1:
                                assert len(rets1) >= 7 and (
                                        'h2oGPT' in rets1['response'] or 'H2O GPT' in rets1['response'] or 'H2O.ai' in
                                        rets1['response'])
                            else:
                                assert len(rets1) >= 7 and (
                                        'h2oGPT' in rets1['response'] or 'H2O GPT' in rets1['response'] or 'H2O.ai' in
                                        rets1['response'])
                                if document_subset == DocumentSubset.Relevant.name:
                                    assert 'h2oGPT' in str(rets1['sources'])
                        else:
                            if answer_with_sources == -1:
                                assert len(rets1) >= 7 and (
                                        'whisper' in rets1['response'].lower() or
                                        'phase' in rets1['response'].lower() or
                                        'generate' in rets1['response'].lower() or
                                        'statistic' in rets1['response'].lower() or
                                        'a chat bot that' in rets1['response'].lower() or
                                        'non-centrality parameter' in rets1['response'].lower() or
                                        '.pdf' in rets1['response'].lower() or
                                        'gravitational' in rets1['response'].lower() or
                                        'answer to the question'  in rets1['response'].lower()
                                )
                            else:
                                assert len(rets1) >= 7 and (
                                        'whisper' in rets1['response'].lower() or
                                        'phase' in rets1['response'].lower() or
                                        'generate' in rets1['response'].lower() or
                                        'statistic' in rets1['response'].lower() or
                                        '.pdf' in rets1['response'].lower())
                                if document_subset == DocumentSubset.Relevant.name:
                                    assert 'whisper' in str(rets1['sources']) or 'unbiased' in str(rets1[
                                        'sources']) or 'approximate' in str(rets1['sources'])
                        if answer_with_sources == -1:
                            if document_subset == DocumentSubset.Relevant.name:
                                assert 'score' in rets1['sources'][0] and 'content' in rets1['sources'][
                                    0] and 'source' in rets1['sources'][0]
                                if doc_choice in [1, 2]:
                                    if langchain_action == 'Summarize':
                                        assert len(set(flatten_list([x['source'].split(docs_joiner_default) for x in
                                                                     rets1['sources']]))) >= doc_choice
                                    else:
                                        assert len(set([x['source'] for x in rets1['sources']])) == doc_choice
                                else:
                                    assert len(set([x['source'] for x in rets1['sources']])) >= 1
                            elif document_subset == DocumentSubset.RelSources.name:
                                if doc_choice in [1, 2]:
                                    assert len(set([x['source'] for x in rets1['sources']])) <= doc_choice
                                else:
                                    if langchain_action == 'Summarize':
                                        assert len(set(flatten_list(
                                            [x['source'].split(docs_joiner_default) for x in rets1['sources']]))) >= 2
                                    else:
                                        assert len(set([x['source'] for x in rets1['sources']])) >= 2
                            else:
                                # TopK may just be 1 doc because of many chunks from that doc
                                # if top_k_docs=-1 might get more
                                assert len(set([x['source'] for x in rets1['sources']])) >= 1

        # SHOW DOC
        single_document_choice1 = [x['source'] for x in db.get()['metadatas']][0]
        text_context_list1 = []
        pdf_height = 800
        h2ogpt_key1 = ''
        for view_raw_text_checkbox1 in [True, False]:
            print("view_raw_text_checkbox1: %s" % view_raw_text_checkbox1, flush=True)
            from src.gradio_runner import show_doc
            show_ret = show_doc(db1s, selection_docs_state1, requests_state1,
                                langchain_mode1,
                                single_document_choice1,
                                view_raw_text_checkbox1,
                                text_context_list1,
                                pdf_height,
                                h2ogpt_key1,
                                dbs1=dbs1,
                                hf_embedding_model1=hf_embedding_model,
                                **other_kwargs
                                )
            assert len(show_ret) == 8
            if chroma_new:
                assert1 = show_ret[4]['value'] is not None and 'README.md' in show_ret[4]['value']
                assert2 = show_ret[3]['value'] is not None and 'h2oGPT' in show_ret[3]['value']
                assert assert1 or assert2
            else:
                assert1 = show_ret[4]['value'] is not None and single_document_choice1 in show_ret[4]['value']
                assert2 = show_ret[3]['value'] is not None and single_document_choice1 in show_ret[3]['value']
                assert assert1 or assert2


@pytest.mark.parametrize("data_kind", [
    'simple',
    'helium1',
    'helium2',
    'helium3',
    'helium4',
    'helium5',
])
@wrap_test_forked
def test_merge_docs(data_kind):
    model_max_length = 4096
    max_input_tokens = 1024
    docs_joiner = docs_joiner_default
    docs_token_handling = docs_token_handling_default
    tokenizer = FakeTokenizer(model_max_length=model_max_length)

    from langchain.docstore.document import Document
    if data_kind == 'simple':
        texts = texts_simple
    elif data_kind == 'helium1':
        texts = texts_helium1
    elif data_kind == 'helium2':
        texts = texts_helium2
    elif data_kind == 'helium3':
        texts = texts_helium3
    elif data_kind == 'helium4':
        texts = texts_helium4
    elif data_kind == 'helium5':
        texts = texts_helium5
    else:
        raise RuntimeError("BAD")

    docs_with_score = [(Document(page_content=page_content, metadata={"source": "%d" % pi}), 1.0) for pi, page_content
                       in enumerate(texts)]

    docs_with_score_new, max_docs_tokens = (
        split_merge_docs(docs_with_score, tokenizer=tokenizer, max_input_tokens=max_input_tokens,
                         docs_token_handling=docs_token_handling, joiner=docs_joiner, verbose=True))

    text_context_list = [x[0].page_content for x in docs_with_score_new]
    tokens = [get_token_count(x + docs_joiner, tokenizer) for x in text_context_list]
    print(tokens)

    if data_kind == 'simple':
        assert len(docs_with_score_new) == 1
        assert all([x < max_input_tokens for x in tokens])
    elif data_kind == 'helium1':
        assert len(docs_with_score_new) == 4
        assert all([x < max_input_tokens for x in tokens])
    elif data_kind == 'helium2':
        assert len(docs_with_score_new) == 8
        assert all([x < max_input_tokens for x in tokens])
    elif data_kind == 'helium3':
        assert len(docs_with_score_new) == 5
        assert all([x < max_input_tokens for x in tokens])
    elif data_kind == 'helium4':
        assert len(docs_with_score_new) == 5
        assert all([x < max_input_tokens for x in tokens])
    elif data_kind == 'helium5':
        assert len(docs_with_score_new) == 3
        assert all([x < max_input_tokens for x in tokens])


@wrap_test_forked
def test_crawl():
    from src.gpt_langchain import Crawler
    final_urls = Crawler(urls=['https://github.com/h2oai/h2ogpt'], verbose=True).run()
    assert 'https://github.com/h2oai/h2ogpt/blob/main/docs/README_GPU.md' in final_urls
    print(final_urls)


if __name__ == '__main__':
    pass
