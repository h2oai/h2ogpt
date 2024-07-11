import os
import sys

root_path = os.path.dirname((__file__))
root_path = os.path.abspath(os.path.abspath(os.path.join(root_path, "..")))
if root_path not in sys.path:
    sys.path.append(root_path)
