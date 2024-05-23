import os
import pickle

import requests
import json


def execute_function_on_server(host: str, port: int, function_name: str, args: tuple, kwargs: dict, use_disk: bool,
                               use_pickle: bool):
    url = f"http://{host}:{port}/execute_function/"
    payload = {
        "function_name": function_name,
        "args": args,
        "kwargs": kwargs,
        "use_disk": use_disk,
        "use_pickle": use_pickle,
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.json()["detail"]}


def read_result_from_disk(file_path: str, use_pickle: bool):
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


def function_client(host, port, function_name, args, kwargs, use_disk, use_pickle):
    execute_result = execute_function_on_server(host, port, function_name, args, kwargs, use_disk, use_pickle)
    if "error" in execute_result:
        print(f"Error: {execute_result['error']}")
    else:
        if use_disk:
            file_path = execute_result["file_path"]
            print(f"Result saved at: {file_path}")
            result_from_disk = read_result_from_disk(file_path, use_pickle)
            print("Result read from disk:", result_from_disk)
        else:
            print("Result received directly:", execute_result["result"])
