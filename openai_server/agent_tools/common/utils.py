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
