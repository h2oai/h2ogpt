import inspect
import os
import sys
import requests
from PIL import Image


def get_have_internet():
    try:
        response = requests.get("http://www.google.com", timeout=5)
        # If the request was successful, status code will be 200
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.ConnectionError:
        return False


def is_image_file(filename):
    try:
        with Image.open(filename) as img:
            img.verify()  # Verify that it's an image
        return True
    except (IOError, SyntaxError):
        return False


def identify_image_files(file_list):
    image_files = []
    non_image_files = []

    for filename in file_list:
        if os.path.isfile(filename):  # Ensure the file exists
            if is_image_file(filename):
                image_files.append(filename)
            else:
                non_image_files.append(filename)
        else:
            print(f"Warning: '{filename}' is not a valid file path.")

    return image_files, non_image_files


def in_pycharm():
    return os.getenv("PYCHARM_HOSTED") is not None


def filter_kwargs(func, kwargs):
    # Get the parameter list of the function
    sig = inspect.signature(func)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return valid_kwargs


def set_python_path():
    # Get the current working directory
    current_dir = os.getcwd()
    current_dir = os.path.abspath(current_dir)

    # Retrieve the existing PYTHONPATH, if it exists, and append the current directory
    pythonpath = os.environ.get('PYTHONPATH', '')
    new_pythonpath = current_dir if not pythonpath else pythonpath + os.pathsep + current_dir

    # Update the PYTHONPATH environment variable
    os.environ['PYTHONPATH'] = new_pythonpath

    # Also, ensure sys.path is updated
    if current_dir not in sys.path:
        sys.path.append(current_dir)
