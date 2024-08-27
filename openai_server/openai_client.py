import contextlib
import os
import shutil
import tempfile
import time
import base64
import mimetypes
from pathlib import Path
from pydantic import BaseModel

from .chat_history_render import chat_to_pretty_markdown


class MyReturnType(BaseModel):
    class Config:
        extra = "allow"


def get_files_from_ids(usage, client):
    if not hasattr(usage, "file_ids"):
        return []

    file_ids = usage.file_ids

    list_response = client.files.list().data
    assert isinstance(list_response, list)
    response_dict = {
        item.id: {key: value for key, value in dict(item).items() if key != "id"}
        for item in list_response
    }

    temp_dir = tempfile.mkdtemp()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    files = []
    for file_id in file_ids:
        test_filename = os.path.join(
            temp_dir, os.path.basename(response_dict[file_id]["filename"])
        )
        content = client.files.content(file_id).content
        with open(test_filename, "wb") as f:
            f.write(content)
        files.append(test_filename)

    return files


def file_to_base64(file_path, file_path_to_use=None):
    # Detect the file's MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = "unknown"

    # Read the file and encode it in base64
    with open(file_path, "rb") as file:
        encoded_file = base64.b64encode(file.read()).decode("utf-8")

    # Construct the data URL
    data_url = f"data:{mime_type};base64,{encoded_file}"
    if file_path_to_use is None:
        file_path_to_use = file_path
    return {file_path_to_use: data_url}


def clean_text_string(input_string):
    lines = input_string.split("\n")
    cleaned_lines = [line for line in lines if line.strip() and line.strip() != "-"]
    return "\n".join(cleaned_lines)


def run_openai_client(
    client,
    ReturnType=None,
    convert_to_pdf=None,
    use_agent=False,
    base64_encode_agent_files=True,
    **query_kwargs,
):
    """
    Bsed upon test in h2oGPT OSS:
    https://github.com/h2oai/h2ogpt/blob/ee3995865c85bf74f3644a4ebd007971c809de11/openai_server/test_openai_server.py#L189-L320
    """
    if ReturnType is None:
        ReturnType = MyReturnType

    # pick correct prompt
    langchain_mode = query_kwargs.get("langchain_mode", "LLM")
    if langchain_mode == "LLM":
        prompt = query_kwargs["instruction"]
    elif langchain_mode == "Query":
        prompt = query_kwargs["prompt_query"]
    else:
        prompt = query_kwargs["prompt_summary"]

    model = query_kwargs["visible_models"]
    stream_output = query_kwargs["stream_output"]
    text_context_list = query_kwargs["text_context_list"]
    image_files = query_kwargs["image_files"]

    messages = []

    if query_kwargs.get("chat_history"):
        for chat in query_kwargs["chat_history"]:
            if chat[0] and chat[1]:
                messages.append(
                    {
                        "role": "user",
                        "content": chat[0],
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": chat[1],
                    }
                )
    # NOTE: Could pass image_files through extra_body instead.
    if query_kwargs.get("image_files"):
        messages_images = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]
        for image_file in image_files:
            messages_images[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        # URL accessible from web or markdown + base64 type url
                        "url": image_file,
                        "detail": "high",
                    },
                }
            )
        messages.extend(messages_images)
    else:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )

    if use_agent:
        extra_body = dict(
            use_agent=use_agent,
            agent_type="auto",
            autogen_stop_docker_executor=False,
            autogen_run_code_in_docker=False,
            autogen_max_consecutive_auto_reply=10,
            autogen_max_turns=None,
            autogen_timeout=120,
            autogen_cache_seed=None,
            autogen_venv_dir=None,
            agent_verbose=True,
            text_context_list=text_context_list,
        )
        # agent needs room, else keep hitting continue
        hyper_kwargs = dict(
            temperature=query_kwargs["temperature"],
            max_tokens=4096,
        )
    else:
        extra_body = {}
        hyper_kwargs = dict(
            temperature=query_kwargs["temperature"],
            max_tokens=query_kwargs["max_new_tokens"],
        )

    time_to_first_token = None
    t0 = time.time()

    responses = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream_output,
        **hyper_kwargs,
        extra_body=extra_body,
    )

    if not stream_output:
        usage = responses.usage
        if responses.choices:
            response = responses.choices[-1].message.content
        else:
            response = ""
        yield ReturnType(reply=response)
        time_to_first_token = time.time() - t0
    else:
        response = ""
        usages = []
        for chunk in responses:
            if chunk.usage is not None:
                usages.append(chunk.usage)
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    response += delta
                    # ensure if h2oGPTe wants full or delta, looks like delta from gradio code, except at very end?
                    yield ReturnType(reply=delta)
                    if time_to_first_token is None:
                        time_to_first_token = time.time() - t0
        assert len(usages) == 1, 'Missing usage"'
        usage = usages[0]

    tf = time.time()

    # Get files
    file_names = get_files_from_ids(usage, client) if use_agent else []

    # See if can make text in case no extension
    for file_i, file in enumerate(file_names):
        file_path = Path(file)
        suffix = file_path.suffix.lower()
        if not suffix and not is_binary(file):  # No suffix and not binary
            new_file = file_path.with_suffix(".txt")
            try:
                file_path.rename(new_file)  # Rename the file, overwriting if necessary
                file_names[file_i] = str(new_file)
            except OSError as e:
                print(f"Error renaming {file} to {new_file}: {e}")

    if base64_encode_agent_files:
        files = [file_to_base64(x) for x in file_names]
        files = update_file_names(files)
    else:
        files = file_names

    def local_convert_to_pdf(x, *args, **kwargs):
        try:
            return convert_to_pdf(x, *args, **kwargs)
        except Exception as e1:
            print(f"Error converting {x} to PDF: {e1}")
            return None

    # make PDF versions of the files
    pdf_file_names = [
        local_convert_to_pdf(Path(x), correct_image=False) for x in file_names
    ]
    pdf_file_names = [str(x) for x in pdf_file_names if x is not None]

    if base64_encode_agent_files:
        files_pdf = [file_to_base64(x, y) for x, y in zip(pdf_file_names, file_names)]
        files_pdf = update_file_names(files_pdf)

        # clean-up
        [remove(x) for x in file_names if os.path.isfile(x)]
        [remove(x) for x in pdf_file_names if os.path.isfile(x)]
    else:
        files_pdf = pdf_file_names

    # Get usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    if hasattr(usage, "cost") and usage.cost:
        usage_no_caching = usage.cost["usage_excluding_cached_inference"]
        assert model in usage_no_caching, "Missing model %s in %s" % (
            model,
            usage_no_caching,
        )
        input_tokens += usage_no_caching[model]["prompt_tokens"]
        output_tokens += usage_no_caching[model]["completion_tokens"]

    # Get final answer
    response_intermediate = response
    if hasattr(usage, "summary"):
        response = usage.summary
        if not response:
            split1 = response_intermediate.split(
                "code_writer_agent(tocode_executor_agent):"
            )
            if split1 and split1[-1]:
                split2 = split1[-1].split("code_executor_agent(tocode_writer_agent):")
                if split2 and split1[0]:
                    response = split2[0]
                    response = clean_text_string(response)
        if not response:
            response = "The task is complete"

    # Get internal chat history
    chat_history = usage.chat_history if hasattr(usage, "chat_history") else None
    chat_history_md = chat_to_pretty_markdown(chat_history)

    # estimate tokens per second
    tokens_per_second = output_tokens / (tf - t0 + 1e-6)

    t_taken_s = time.time() - t0
    t_taken = "%.4f" % t_taken_s
    if not (response or response_intermediate or files or chat_history):
        raise TimeoutError(
            f"No output from Agent with LLM {model} after {t_taken} seconds."
        )

    # final yield
    yield ReturnType(
        reply=response_intermediate,
        reply_final=response,
        prompt_raw=prompt,  # meaningless, so use original
        actual_llm=model,
        text_context_list=text_context_list,  # WIP
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tokens_per_second=tokens_per_second,
        time_to_first_token=time_to_first_token,
        trial=0,  # WIP
        vision_visible_model=model,  # WIP
        vision_batch_input_tokens=0,  # WIP
        vision_batch_output_tokens=0,  # WIP
        vision_batch_tokens_per_second=0.0,  # WIP
        files=files,
        chat_history=chat_history,
        chat_history_md=chat_history_md,
        files_pdf=files_pdf,
    )


def is_binary(filename):
    """
    Check if a file is binary or text using a quick check.

    Args:
        filename (str): The path to the file.

    Returns:
        bool: True if the file is binary, False otherwise.
    """

    try:
        with open(filename, "rb") as f:
            chunk = f.read(1024)  # Read the first 1KB of the file for a quick check
            if b"\0" in chunk:  # Null byte found, indicating binary content
                return True
            # Try decoding the chunk as UTF-8
            try:
                chunk.decode("utf-8")
            except UnicodeDecodeError:
                return True  # Decoding failed, likely a binary file
    except Exception as e:
        raise RuntimeError(f"Error reading file: {e}")

    return False  # No null bytes and successful UTF-8 decoding, likely a text file


def update_file_names(file_list):
    def process_item(item):
        if isinstance(item, str):
            return os.path.basename(item)
        elif isinstance(item, dict):
            old_key = list(item.keys())[0]
            return {os.path.basename(old_key): item[old_key]}
        else:
            raise ValueError(f"Unsupported item type: {type(item)}")

    return [process_item(item) for item in file_list]


def shutil_rmtree(*args, **kwargs):
    path = args[0]
    assert not os.path.samefile(
        path, "/"
    ), "Should not be trying to remove entire root directory: %s" % str(path)
    assert not os.path.samefile(
        path, "./"
    ), "Should not be trying to remove entire local directory: %s" % str(path)
    return shutil.rmtree(*args, **kwargs)


def remove(path: str):
    try:
        if path is not None and os.path.exists(path):
            if os.path.isdir(path):
                shutil_rmtree(path, ignore_errors=True)
            else:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(path)
    except BaseException as e:
        print(f"Error removing {path}: {e}")
        pass
