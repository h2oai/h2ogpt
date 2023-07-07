import fire
from src.gen import main


def entrypoint_main():
    fire.Fire(main)


if __name__ == "__main__":
    entrypoint_main()
