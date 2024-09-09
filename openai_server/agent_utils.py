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


def current_datetime():
    from datetime import datetime
    import tzlocal

    # Get the local time zone
    local_timezone = tzlocal.get_localzone()

    # Get the current time in the local time zone
    now = datetime.now(local_timezone)

    # Format the date, time, and time zone
    formatted_date_time = now.strftime("%A, %B %d, %Y - %I:%M %p %Z")

    # Print the formatted date, time, and time zone
    return "For current user query: Current Date, Time, and Local Time Zone: %s. Note some APIs may have data from different time zones, so may reflect a different date." % formatted_date_time

def merge_group_chat_messages(a, b):
    """
    Helps to merge chat messages from two different sources.
    Mostly messages from Group Chat Managers.
    """
    # Create a copy of b to avoid modifying the original list
    merged_list = b.copy()
    
    # Convert b into a set of contents for faster lookup
    b_contents = {item['content'] for item in b}
    
    # Iterate through the list a
    for i, item_a in enumerate(a):
        content_a = item_a['content']
        
        # If the content is not in b, insert it at the correct position
        if content_a not in b_contents:
            # Find the position in b where this content should be inserted
            # Insert right after the content of the previous item in list a (if it exists)
            if i > 0:
                prev_content = a[i - 1]['content']
                # Find the index of the previous content in the merged list
                for j, item_b in enumerate(merged_list):
                    if item_b['content'] == prev_content:
                        merged_list.insert(j + 1, item_a)
                        break
            else:
                # If it's the first item in a, just append it to the beginning
                merged_list.insert(0, item_a)
            
            # Update the b_contents set
            b_contents.add(content_a)
    
    return merged_list
