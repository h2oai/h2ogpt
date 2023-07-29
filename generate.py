import os
import sys

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.gen import main
from src.utils import H2O_Fire


def entrypoint_main():
    H2O_Fire(main)


if __name__ == "__main__":
    entrypoint_main()
