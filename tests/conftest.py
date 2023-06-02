import os


def pytest_itemcollected(item):
    item.name = item.name + os.getenv("PYTEST_TEST_NAME", "")
