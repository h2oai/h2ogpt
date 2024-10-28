import ast
import contextlib
import gc
import os
import shutil
import tempfile
import threading
import traceback
import time
import base64
import mimetypes
import uuid
from enum import Enum
from pathlib import Path
from collections import defaultdict

import numpy as np
from pydantic import BaseModel

from .chat_history_render import chat_to_pretty_markdown

# control convert_to_pdf as expensive use of cores
num_convert_threads = max(min(10, os.cpu_count() or 1), 1)
convert_sem = threading.Semaphore(num_convert_threads)


class MyReturnType(BaseModel):
    class Config:
        extra = "allow"


# Local copy of minimal version from h2oGPT server
class LangChainAction(Enum):
    """LangChain action"""

    QUERY = "Query"
    SUMMARIZE_MAP = "Summarize"
    EXTRACT = "Extract"


def get_files_from_ids(usage=None, client=None, file_ids=None, work_dir=None):
    if usage is None and file_ids:
        pass
    elif hasattr(usage, "file_ids"):
        file_ids = usage.file_ids
    else:
        return []

    response_dict = {
        file_id: dict(client.files.retrieve(file_id)) for file_id in file_ids
    }

    # sort file_ids by server ctime, so first is newest
    file_ids = list(
        reversed(sorted(file_ids, key=lambda x: response_dict[x]["created_at"]))
    )

    if work_dir is None:
        temp_dir = tempfile.mkdtemp()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        work_dir = temp_dir

    files = []
    for file_id in file_ids:
        new_filename = os.path.join(
            work_dir, os.path.basename(response_dict[file_id]["filename"])
        )
        if os.path.exists(new_filename):
            # FIXME: small chance different with same name
            pass
        else:
            content = client.files.content(file_id).content
            with open(new_filename, "wb") as f:
                f.write(content)
        files.append(new_filename)

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
    cleaned_lines = [
        line for line in lines if line and line.strip() and line.strip() != "-"
    ]
    return "\n".join(cleaned_lines)


def local_convert_to_pdf(convert_to_pdf, x, files_already_pdf, *args, **kwargs):
    if x in files_already_pdf:
        return x
    try:
        with convert_sem:
            return convert_to_pdf(x, *args, **kwargs)
    except Exception as e1:
        print(f"Error converting {x} to PDF: {e1}")
        return None


def group_files_by_base_name(file_names):
    grouped_files = defaultdict(list)
    for file in file_names:
        base_name = Path(file).stem
        grouped_files[base_name].append(file)
    return grouped_files


def group_and_prioritize_files(file_names):
    grouped_files = group_files_by_base_name(file_names)

    prioritized_files = []
    for base_name, files in grouped_files.items():
        preferred_file = select_preferred_file(files)
        # Put the preferred file first, then add all other files
        prioritized_group = [preferred_file] + [f for f in files if f != preferred_file]
        prioritized_files.extend(prioritized_group)

    return prioritized_files


def select_preferred_file(files):
    # Preference order: PDF, PNG, SVG, others
    for ext in [".pdf", ".png", ".svg"]:
        for file in files:
            if file.lower().endswith(ext):
                return file
    # If no preferred format found, return the first file
    return files[0]


def get_pdf_files(file_names, convert_to_pdf):
    # Group files by base name
    prioritized_files = group_and_prioritize_files(file_names)

    # Filter out binary files with text-like extensions
    # e.g. .txt but giant binary, then libreoffice will take too long to convert
    selected_files = [
        file
        for file in prioritized_files
        if not (is_binary(file) and Path(file).suffix.lower() in TEXT_EXTENSIONS)
    ]

    # Filter out audio files
    audio_exts = [
        ".mp3",
        ".wav",
        ".flac",
        ".ogg",
        ".m4a",
        ".aac",
        ".wma",
        ".aiff",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".mpga",
        ".webm",
    ]

    exclude_exts = audio_exts + [".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar"]

    selected_files = [
        file
        for file in selected_files
        if not any(file.lower().endswith(ext) for ext in exclude_exts)
    ]

    # 5MB limit to avoid long conversions
    selected_files = [
        f for f in selected_files if os.path.getsize(f) <= 5 * 1024 * 1024
    ]

    # Convert files to PDF
    pdf_file_names = []
    pdf_base_names = set()
    errors = []

    def process_file(file, pdf_base_names, convert_to_pdf):
        file_path = Path(file)
        base_name = file_path.stem
        ext_name = file_path.suffix.lower()

        if file_path.suffix.lower() == ".pdf":
            pdf_base_names.add(base_name)
            return str(file_path), base_name, None

        if base_name in pdf_base_names:
            new_pdf_name = f"{base_name}{ext_name}.pdf"
        else:
            new_pdf_name = f"{base_name}.pdf"
            pdf_base_names.add(base_name)

        new_pdf_path = file_path.with_name(new_pdf_name)
        new_dir = os.path.dirname(new_pdf_path)
        temp_file = file_path.with_suffix(f".{uuid.uuid4()}{file_path.suffix}")

        try:
            if not os.path.exists(new_dir):
                os.makedirs(new_dir, exist_ok=True)
            shutil.copy(file_path, temp_file)
            converted_pdf = local_convert_to_pdf(
                convert_to_pdf,
                temp_file,
                set(),
                correct_image=False,
            )
            if converted_pdf:
                shutil.move(converted_pdf, str(new_pdf_path))
                return str(new_pdf_path), base_name, None
        except Exception as e:
            return None, None, f"Error converting {file} to PDF: {e}"
        finally:
            if os.path.isfile(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"Error removing temp file {temp_file}: {e}")

        return None, None, f"Failed to process {file}"

    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

    # Set timeouts
    timeout_seconds = 3 * 60
    timeout_seconds_per_file = 30

    t0 = time.time()

    with ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(process_file, file, pdf_base_names, convert_to_pdf): file
            for file in selected_files
        }

        while future_to_file:
            # Re-check remaining time for the overall timeout
            remaining_time = timeout_seconds - (time.time() - t0)
            if remaining_time <= 0:
                errors.append(f"Overall timeout of {timeout_seconds} seconds reached.")
                break

            # Check the futures as they complete or timeout
            try:
                for future in as_completed(future_to_file, timeout=remaining_time):
                    file = future_to_file[future]  # Get the corresponding file
                    try:
                        # Wait for the result of each future with a per-file timeout
                        result, base_name, error = future.result(
                            timeout=timeout_seconds_per_file
                        )

                        # Only pop the future after successful completion
                        future_to_file.pop(future)

                        if error:
                            errors.append(f"Error processing {file}: {error}")
                        elif result:
                            pdf_file_names.append(result)
                            pdf_base_names.add(base_name)
                    except TimeoutError:
                        errors.append(
                            f"Timeout error processing {file}: operation took longer than {timeout_seconds_per_file} seconds"
                        )
                    except Exception as exc:
                        errors.append(f"Unexpected error processing {file}: {exc}")
                        # We still want to pop the future on failure
                        future_to_file.pop(future)
            except TimeoutError:
                errors.append(
                    f"Timeout error processing {file}: operation took longer than {timeout_seconds_per_file} seconds"
                )
            except Exception as exc:
                errors.append(f"Unexpected error processing {file}: {exc}")

            # If all futures are processed or timeout reached, break
            if time.time() - t0 > timeout_seconds:
                errors.append(
                    f"Overall timeout of {timeout_seconds} seconds reached.  {len(future_to_file)} files remaining."
                )
                break

    if errors:
        print(errors)

    return pdf_file_names


def completion_with_backoff(
    get_client,
    model,
    messages,
    stream_output,
    hyper_kwargs,
    extra_body,
    timeout,
    time_to_first_token_max,
    ReturnType=None,
    use_agent=False,
    add_extra_endofturn=False,
    max_chars_per_turn=1024 * 4,
):
    t0_outer = time.time()
    ntrials = 3
    trial = 0
    while True:
        t0 = time.time()
        responses = None
        client = None
        time_to_first_token = None
        response = ""
        usage = None
        file_names = []
        try:
            client = get_client()
            responses = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream_output,
                **hyper_kwargs,
                extra_body=extra_body,
                timeout=timeout,
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
                            if use_agent and add_extra_endofturn:
                                splits = response.split("ENDOFTURN")
                                if splits and len(splits[-1]) > max_chars_per_turn:
                                    # force end of turn for UI purposes
                                    delta = "\n\nENDOFTURN\n\n"
                                    response += delta
                                    yield ReturnType(reply=delta)
                    time.sleep(0.005)
                    if (
                        time_to_first_token is None
                        and time.time() - t0 > time_to_first_token_max
                    ):
                        raise TimeoutError(
                            f"LLM {model} timed out without any response after {time_to_first_token_max} seconds, for total {time.time() - t0_outer} seconds.."
                        )
                    if time.time() - t0 > timeout:
                        print("Timed out, but had response: %s" % response, flush=True)
                        raise TimeoutError(
                            f"LLM {model} timed out after {time.time() - t0} seconds, for total {time.time() - t0_outer} seconds."
                        )
                assert len(usages) == 1, 'Missing usage"'
                usage = usages[0]

            # Get files
            file_names = (
                get_files_from_ids(usage=usage, client=client) if use_agent else []
            )
            return (
                response,
                usage,
                file_names,
                time_to_first_token or (time.time() - t0),
                None,
                None,
            )
        except (GeneratorExit, StopIteration):
            # caller is trying to cancel
            print(f"Caller initiated GeneratorExit in completion_with_backoff.")
            raise
        except Exception as e:
            error_ex = traceback.format_exc()
            error_e = str(e)
            if trial == ntrials - 1 or "Output contains sensitive information" in str(
                e
            ):
                print(
                    f"{model} hit final error in completion_with_backoff: {e}. Retrying trial {trial}."
                )
                if os.getenv("HARD_ASSERTS"):
                    raise
                # Note: response can be partial
                return (
                    response,
                    usage,
                    file_names,
                    time_to_first_token or (time.time() - t0),
                    error_e,
                    error_ex,
                )
            else:
                if trial == 0:
                    time.sleep(1)
                elif trial == 1:
                    time.sleep(5)
                else:
                    time.sleep(20)
                trial += 1
                print(
                    f"{model} hit error in completion_with_backoff: {e}. Retrying trial {trial}."
                )
        finally:
            if responses is not None:
                try:
                    responses.close()
                    del responses
                    gc.collect()
                except Exception as e:
                    print("Failed to close OpenAI response: %s" % str(e), flush=True)
            if client is not None:
                try:
                    client.close()
                    del client
                    gc.collect()
                except Exception as e:
                    print("Failed to close OpenAI client: %s" % str(e), flush=True)


def run_openai_client(
    get_client=None,
    ReturnType=None,
    convert_to_pdf=None,
    use_agent=False,
    agent_accuracy="standard",
    autogen_max_turns=80,
    agent_chat_history=[],
    agent_files=[],
    agent_venv_dir=None,
    agent_work_dir=None,
    base64_encode_agent_files=True,
    cute=False,
    time_to_first_token_max=None,
    **query_kwargs,
):
    """
    Bsed upon test in h2oGPT OSS:
    https://github.com/h2oai/h2ogpt/blob/ee3995865c85bf74f3644a4ebd007971c809de11/openai_server/test_openai_server.py#L189-L320
    """
    if ReturnType is None:
        ReturnType = MyReturnType

    # pick correct prompt
    # langchain_mode = query_kwargs.get("langchain_mode", "LLM")
    langchain_action = query_kwargs.get("langchain_action", "Query")
    # prompt will be "" for langchain_action = 'Summarize'
    prompt = query_kwargs["instruction"]
    model = query_kwargs["visible_models"]
    stream_output = query_kwargs["stream_output"]
    max_time = query_kwargs["max_time"]
    time_to_first_token_max = time_to_first_token_max or max_time
    text_context_list = query_kwargs["text_context_list"]
    chat_conversation = query_kwargs["chat_conversation"]
    image_files = query_kwargs["image_file"]
    system_message = query_kwargs["system_prompt"]

    from h2ogpte_core.backend_utils import structure_to_messages

    if use_agent:
        chat_conversation = None  # don't include high-level history yet

        file_ids = []
        if agent_files:
            client = get_client()
            for file_path in agent_files:
                with open(file_path, "rb") as file:
                    ret = client.files.create(
                        file=file,
                        purpose="assistants",
                    )
                    file_id = ret.id
                    file_ids.append(file_id)
                    assert ret.bytes > 0

        extra_body = dict(
            use_agent=use_agent,
            agent_type="auto",
            agent_accuracy=agent_accuracy,
            autogen_stop_docker_executor=False,
            autogen_run_code_in_docker=False,
            autogen_max_consecutive_auto_reply=80,
            autogen_max_turns=autogen_max_turns,
            autogen_timeout=240,
            autogen_cache_seed=None,
            work_dir=agent_work_dir,
            venv_dir=agent_venv_dir,
            agent_verbose=True,
            text_context_list=text_context_list,
            agent_chat_history=agent_chat_history,
            agent_files=file_ids,
            client_metadata=query_kwargs.get("client_metadata", ""),
        )
        # agent needs room, else keep hitting continue
        hyper_kwargs = dict(
            temperature=query_kwargs["temperature"],
            seed=query_kwargs["seed"],
            max_tokens=8192 if "claude-3-5-sonnet" in model else 4096,
        )
    else:
        extra_body = query_kwargs.copy()
        from h2ogpte_core.src.evaluate_params import eval_func_param_names

        extra_body = {k: v for k, v in extra_body.items() if k in eval_func_param_names}
        hyper_kwargs = dict(
            temperature=query_kwargs["temperature"],
            top_p=query_kwargs["top_p"],
            seed=query_kwargs["seed"],
            max_tokens=query_kwargs["max_new_tokens"],
        )
        extra_body = {k: v for k, v in extra_body.items() if k not in hyper_kwargs}
        # remove things that go through OpenAI API messages
        keys_in_api = [
            "visible_models",
            "image_file",
            "chat_conversation",
            "system_prompt",
            "instruction",
            "stream_output",
        ]
        for key in keys_in_api:
            extra_body.pop(key, None)
        # translate
        if "response_format" in extra_body:
            extra_body["response_format"] = dict(type=extra_body["response_format"])

    time_to_first_token = None
    t0 = time.time()

    messages = structure_to_messages(
        prompt, system_message, chat_conversation, image_files
    )

    timeout = 5 * max_time if use_agent else max_time
    (
        response,
        usage,
        file_names,
        time_to_first_token,
        error_e,
        error_ex,
    ) = yield from completion_with_backoff(
        get_client,
        model,
        messages,
        stream_output,
        hyper_kwargs,
        extra_body,
        timeout,
        time_to_first_token_max,
        ReturnType=ReturnType,
        use_agent=use_agent,
    )

    # in case streaming had deletions not yet accounted for, recover at least final answer,
    # e.g. for JSON {} then {}{"response": "yes"}
    if hasattr(usage, "response"):
        response = usage.response

    tf = time.time()

    # See if we can make text in case of no extension
    for file_i, file in enumerate(file_names):
        file_path = Path(file)
        suffix = file_path.suffix.lower()

        # If no suffix and not binary, rename to ".txt"
        if not suffix and not is_binary(file):
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

    # Process files and get PDF file names
    pdf_file_names = get_pdf_files(files, convert_to_pdf)

    if base64_encode_agent_files:
        files_pdf = [file_to_base64(x, y) for x, y in zip(pdf_file_names, file_names)]
        files_pdf = update_file_names(files_pdf)

        # clean-up
        [remove(x) for x in file_names if os.path.isfile(x)]
        [remove(x) for x in pdf_file_names if os.path.isfile(x)]
    else:
        files_pdf = pdf_file_names

    # Get usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    if hasattr(usage, "cost") and usage.cost:
        usage_no_caching = usage.cost["usage_excluding_cached_inference"]
        assert model in usage_no_caching, "Missing model %s in %s" % (
            model,
            usage_no_caching,
        )
        input_tokens += usage_no_caching[model]["prompt_tokens"]
        output_tokens += usage_no_caching[model]["completion_tokens"]

    # Get internal chat history
    chat_history = (
        usage.chat_history
        if hasattr(usage, "chat_history")
        else [{"role": "assistant", "content": response}]
    )
    chat_history_md = (
        chat_to_pretty_markdown(chat_history, cute=cute) if chat_history else ""
    )

    agent_work_dir = usage.agent_work_dir if hasattr(usage, "agent_work_dir") else None
    agent_venv_dir = usage.agent_venv_dir if hasattr(usage, "agent_venv_dir") else None

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
    elif "ENDOFTURN" in response:
        # show last turn as final response
        split_responses = response.split("ENDOFTURN")
        if len(split_responses) > 1:
            response = split_responses[-1]
        if not response:
            response = "The task completed"

    # estimate tokens per second
    tokens_per_second = output_tokens / (tf - t0 + 1e-6)

    t_taken_s = time.time() - t0
    t_taken = "%.4f" % t_taken_s
    if use_agent:
        if not (response or response_intermediate or files or chat_history):
            msg = f"No output from Agent with LLM {model} after {t_taken} seconds."
            if error_e:
                raise ValueError("Error: " + error_e + "\n" + msg)
            else:
                raise TimeoutError(msg)
    else:
        if not (response or response_intermediate):
            msg = f"No response from LLM {model} after {t_taken} seconds."
            if error_e:
                raise ValueError("Error: " + error_e + "\n" + msg)
            else:
                raise TimeoutError(msg)

    # extract other usages:
    sources = usage.sources if hasattr(usage, "sources") else []
    prompt_raw = usage.prompt_raw if hasattr(usage, "prompt_raw") else ""
    save_dict = usage.save_dict if hasattr(usage, "save_dict") else {}
    if not use_agent:
        if not hasattr(usage, "sources"):
            print("missing sources from usage: %s" % usage)
        if not hasattr(usage, "prompt_raw"):
            print("missing prompt_raw from usage: %s" % usage)
        if not hasattr(usage, "save_dict"):
            print("missing save_dict from usage: %s" % usage)
    extra_dict = save_dict.get("extra_dict", {})
    texts_out = [x["content"] for x in sources] if not use_agent else text_context_list
    t_taken_s = time.time() - t0
    t_taken = "%.4f" % t_taken_s

    if langchain_action != LangChainAction.EXTRACT.value:
        response = response.strip() if response else ""
        response_intermediate = response_intermediate.strip()
    else:
        response = [r.strip() if r else "" for r in ast.literal_eval(response)]
        response_intermediate = [
            r.strip() if r else "" for r in ast.literal_eval(response_intermediate)
        ]

    try:
        actual_llm = save_dict["display_name"]
    except Exception as e:
        actual_llm = model
        print(f"Unable to access save_dict to get actual_llm: {str(e)}")

    reply = response_intermediate if use_agent else response

    if not reply:
        error_e = (
            error_ex
        ) = f"No final response from LLM {actual_llm} after {t_taken} seconds\nError:{error_e}."
    if "error" in save_dict and not prompt_raw:
        msg = f"Error from LLM {actual_llm}: {save_dict['error']}"
        if os.getenv("HARD_ASSERTS"):
            if error_e:
                raise ValueError("Error: " + error_e + "\n" + msg)
            else:
                raise ValueError(msg)
    if not use_agent:
        if not (prompt_raw or extra_dict):
            msg = "LLM response failed to return final metadata."
            if os.getenv("HARD_ASSERTS"):
                if error_e:
                    raise ValueError("Error: " + error_e + "\n" + msg)
                else:
                    raise ValueError(msg)
    else:
        prompt_raw = prompt

    try:
        input_tokens = extra_dict["num_prompt_tokens"]
        output_tokens = extra_dict["ntokens"]
        vision_visible_model = extra_dict.get("batch_vision_visible_model")
        vision_batch_input_tokens = extra_dict.get("batch_num_prompt_tokens", 0)
        vision_batch_output_tokens = extra_dict.get("batch_ntokens", 0)
        tokens_per_second = np.round(extra_dict["tokens_persecond"], decimals=3)
        vision_batch_tokens_per_second = extra_dict.get("batch_tokens_persecond", 0)
        if vision_batch_tokens_per_second:
            vision_batch_tokens_per_second = np.round(
                vision_batch_tokens_per_second, decimals=3
            )
    except:
        vision_visible_model = model
        vision_batch_input_tokens = 0
        vision_batch_output_tokens = 0
        vision_batch_tokens_per_second = 0
        if not use_agent and os.getenv("HARD_ASSERTS"):
            raise

    if use_agent and not response and reply:
        # show streamed output then, to avoid confusion with whether had response
        response = reply

    if error_e or error_ex:
        delta_error = f"\n\n**Partial Error:**\n\n {error_e}"
        if use_agent:
            yield ReturnType(reply="\nENDOFTURN\n" + delta_error)
            response = delta_error
        else:
            yield ReturnType(reply=delta_error)
            response += delta_error

    # final yield
    yield ReturnType(
        reply=reply,
        reply_final=response,
        prompt_raw=prompt_raw,
        actual_llm=actual_llm,
        text_context_list=texts_out,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tokens_per_second=tokens_per_second,
        time_to_first_token=time_to_first_token or (time.time() - t0),
        trial=0,  # Not required, OpenAI has retries
        vision_visible_model=vision_visible_model,
        vision_batch_input_tokens=vision_batch_input_tokens,
        vision_batch_output_tokens=vision_batch_output_tokens,
        vision_batch_tokens_per_second=vision_batch_tokens_per_second,
        agent_work_dir=agent_work_dir,
        agent_venv_dir=agent_venv_dir,
        files=files,
        files_pdf=files_pdf,
        chat_history=chat_history,
        chat_history_md=chat_history_md,
    )


# List of common text file extensions
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".html",
    ".xml",
    ".json",
    ".yaml",
    ".yml",
    ".log",
}


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
        print(f"Error reading file: {e}")
        return True

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
