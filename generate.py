import os
import sys
import fire

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.gen import main


def entrypoint_main():
    fire.Fire(main)


if __name__ == "__main__":
    entrypoint_main()
