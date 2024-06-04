import os
import pickle

import requests
import json


def execute_function_on_server(host: str, port: int, function_name: str, args: tuple, kwargs: dict, use_disk: bool,
                               use_pickle: bool, function_api_key: str):
    url = f"http://{host}:{port}/execute_function/"
    payload = {
        "function_name": function_name,
        "args": args,
        "kwargs": kwargs,
        "use_disk": use_disk,
        "use_pickle": use_pickle,
    }
    headers = {
        "Authorization": f"Bearer {function_api_key}"
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.json()["detail"]}


def read_result_from_disk(file_path: str, use_pickle: bool, verbose=False):
    if verbose:
        print(f"Size of {file_path} is {os.path.getsize(file_path)}")
    try:
        if use_pickle:
            with open(file_path, "rb") as f:
                result = pickle.load(f)
        else:
            with open(file_path, "r") as f:
                result = json.load(f)
    except Exception as e:
        raise IOError(f"Error reading file {file_path}: {e}")
    finally:
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")
    return result


def call_function_server(host, port, function_name, args, kwargs, use_disk=False, use_pickle=False,
                         function_api_key='EMPTY', verbose=False):
    execute_result = execute_function_on_server(host, port, function_name, args, kwargs, use_disk, use_pickle,
                                                function_api_key)
    if "error" in execute_result:
        raise RuntimeError(execute_result['error'])
    else:
        if use_disk:
            file_path = execute_result["file_path"]
            result_from_disk = read_result_from_disk(file_path, use_pickle, verbose=verbose)
            return result_from_disk
        else:
            return execute_result["result"]


def get_data_h2ogpt(file, file_type, file_path, verbose=False):
    """
    Simple function for Open Web UI
    """
    function_server_host = os.getenv('H2OGPT_FUNCTION_SERVER_HOST', '0.0.0.0')
    function_server_port = int(os.getenv('H2OGPT_FUNCTION_SERVER_PORT', '5003'))
    function_api_key = os.getenv('H2OGPT_FUNCTION_SERVER_API_KEY', 'EMPTY')

    # could set other things:
    # https://github.com/h2oai/h2ogpt/blob/d2fa3d7ce507e8fb141c78ff92a83a8e27cf8b31/src/gpt_langchain.py#L9498
    simple_kwargs = {}
    function_name = 'path_to_docs'
    use_disk = False
    use_pickle = True
    sources = call_function_server(function_server_host,
                                   function_server_port,
                                   function_name,
                                   (file_path,),
                                   simple_kwargs,
                                   use_disk=use_disk, use_pickle=use_pickle,
                                   function_api_key=function_api_key,
                                   verbose=verbose)
    return sources
