import os

from gpt_langchain import path_to_docs, chunk_sources, get_db


def glob_to_db(chunk=True, chunk_size=512):
    glob_path = os.getenv('GLOB_PATH')
    assert glob_path, """Must set GLOB_PATH env
    e.g. GLOB_PATH='glob_dir' python make_db.py
    where 'glob_dir' contains files to glob recursively
"""
    sources1 = path_to_docs(glob_path)
    if chunk:
        sources1 = chunk_sources(sources1, chunk_size=chunk_size)
    return sources1


if __name__ == "__main__":
    print("Getting sources", flush=True)
    sources = glob_to_db()
    print("Generating db", flush=True)
    use_openai_embedding = os.getenv('use_openai_embedding', False)
    persist_directory = os.getenv('persist_directory', 'db_dir_glob')
    db = get_db(sources, use_openai_embedding=use_openai_embedding,
                db_type='chroma',
                persist_directory=persist_directory,
                langchain_mode='glob')
    print("DONE", flush=True)
