import fire

from gpt_langchain import path_to_docs, chunk_sources, get_db


def glob_to_db(glob_path, chunk=True, chunk_size=512, verbose=False, fail_any_exception=False):
    sources1 = path_to_docs(glob_path, verbose=verbose, fail_any_exception=fail_any_exception)
    if chunk:
        sources1 = chunk_sources(sources1, chunk_size=chunk_size)
    return sources1


def main(use_openai_embedding: bool = False,
         persist_directory: str = 'db_dir_glob',
         glob_path: str = 'glob_dir',
         verbose: bool = False,
         chunk: bool = True,
         chunk_size: int = 512,
         fail_any_exception: bool = False,
         ):
    if verbose:
        print("Getting sources", flush=True)
    sources = glob_to_db(glob_path, chunk=chunk, chunk_size=chunk_size, verbose=verbose,
                         fail_any_exception=fail_any_exception)
    if verbose:
        print("Generating db", flush=True)
    db = get_db(sources,
                use_openai_embedding=use_openai_embedding,
                db_type='chroma',
                persist_directory=persist_directory,
                langchain_mode='glob')
    assert db is not None
    if verbose:
        print("DONE", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
