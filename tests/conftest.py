import os

def pytest_itemcollected(item):
    item.name = os.getenv("PYTEST_TEST_NAME", "") + item.name
