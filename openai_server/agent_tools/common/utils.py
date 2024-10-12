import os
import shutil
import uuid
from urllib.parse import urlparse

import requests


def is_url_valid_and_alive(url, timeout=5):
    try:
        # Check if the URL is valid
        result = urlparse(url)
        if all([result.scheme, result.netloc]):
            # Try to send a GET request to the URL
            response = requests.get(url, timeout=timeout)
            # If the status code is less than 400, consider it alive
            return response.status_code < 400
        else:
            return False
    except requests.exceptions.RequestException:
        return False


def filename_is_url(filename):
    if filename and (filename.startswith('http://') or filename.startswith('https://') or filename.startswith('www.')):
        if is_url_valid_and_alive(filename):
            return True
    return False


def download_simple(url, dest=None, overwrite=False, verbose=False):
    if dest is None:
        dest = os.path.basename(url)
    base_path = os.path.dirname(dest)
    if base_path:  # else local path
        os.makedirs(base_path, exist_ok=True)
        dest = os.path.join(base_path, os.path.basename(dest))

    if os.path.isfile(dest):
        if not overwrite:
            if verbose:
                print("Already have %s from url %s, delete file if invalid" % (dest, str(url)), flush=True)
            return dest
        else:
            os.remove(dest)

    if verbose:
        print("BEGIN get url %s" % str(url), flush=True)
    if url.startswith("file://"):
        from requests_file import FileAdapter
        s = requests.Session()
        s.mount('file://', FileAdapter())
        url_data = s.get(url, stream=True)
    else:
        url_data = requests.get(url, stream=True)
    if verbose:
        print("GOT url %s" % str(url), flush=True)

    if url_data.status_code != requests.codes.ok:
        msg = "Cannot get url %s, code: %s, reason: %s" % (
            str(url),
            str(url_data.status_code),
            str(url_data.reason),
        )
        raise requests.exceptions.RequestException(msg)
    url_data.raw.decode_content = True

    uuid_tmp = str(uuid.uuid4())[:6]
    dest_tmp = dest + "_dl_" + uuid_tmp + ".tmp"

    # Sizes in bytes.
    block_size = 1024
    with open(dest_tmp, "wb") as file:
        for data in url_data.iter_content(block_size):
            file.write(data)

    try:
        shutil.move(dest_tmp, dest)
    except (shutil.Error, FileExistsError):
        pass

    if verbose:
        print("DONE url %s" % str(url), flush=True)
    return dest
