import os
import sys
import importlib.util


def pytest_itemcollected(item):
    item.name = item.name + os.getenv("PYTEST_TEST_NAME", "")


def pytest_sessionstart(session):
    if not os.getenv("IS_PR_BUILD", None):
        return
    try:
        sys.path.append(os.path.dirname(importlib.util.find_spec("h2ogpt").origin))
    except AttributeError:
        pass
