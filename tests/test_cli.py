import pytest

from tests.utils import wrap_test_forked, get_llama
from src.enums import DocumentChoices


@wrap_test_forked
def test_cli(monkeypatch):
    query = "What is the Earth?"
    monkeypatch.setattr('builtins.input', lambda _: query)

    from src.gen import main
    all_generations = main(base_model='gptj', cli=True, cli_loop=False, score_model='None')

    assert len(all_generations) == 1
    assert "The Earth is a planet in our solar system" in all_generations[0]


@wrap_test_forked
def test_cli_langchain(monkeypatch):
    from tests.utils import make_user_path_test
    user_path = make_user_path_test()

    query = "What is the cat doing?"
    monkeypatch.setattr('builtins.input', lambda _: query)

    from src.gen import main
    all_generations = main(base_model='gptj', cli=True, cli_loop=False, score_model='None',
                           langchain_mode='UserData',
                           user_path=user_path,
                           visible_langchain_modes=['UserData', 'MyData'],
                           document_subset=DocumentChoices.Relevant.name,
                           verbose=True)

    print(all_generations)
    assert len(all_generations) == 1
    assert "pexels-evg-kowalievska-1170986_small.jpg" in all_generations[0]
    assert "looking out the window" in all_generations[0] or \
           "staring out the window at the city skyline" in all_generations[0] or \
           "what the cat is doing" in all_generations[0] or \
           "question about a cat" in all_generations[0]


@pytest.mark.need_tokens
@wrap_test_forked
def test_cli_langchain_llamacpp(monkeypatch):
    prompt_type = get_llama()

    from tests.utils import make_user_path_test
    user_path = make_user_path_test()

    query = "What is the cat doing?"
    monkeypatch.setattr('builtins.input', lambda _: query)

    from src.gen import main
    all_generations = main(base_model='llama', cli=True, cli_loop=False, score_model='None',
                           langchain_mode='UserData',
                           prompt_type=prompt_type,
                           user_path=user_path,
                           visible_langchain_modes=['UserData', 'MyData'],
                           document_subset=DocumentChoices.Relevant.name,
                           verbose=True)

    print(all_generations)
    assert len(all_generations) == 1
    assert "pexels-evg-kowalievska-1170986_small.jpg" in all_generations[0]
    assert "The cat is sitting on a window seat and looking out the window" in all_generations[0] or \
           "staring out the window at the city skyline" in all_generations[0] or \
           "The cat is likely relaxing and enjoying" in all_generations[0] or \
           "The cat is sitting on a window seat and looking out" in all_generations[0] or \
           "cat in the image is" in all_generations[0]


@pytest.mark.need_tokens
@wrap_test_forked
def test_cli_llamacpp(monkeypatch):
    prompt_type = get_llama()

    query = "Who are you?"
    monkeypatch.setattr('builtins.input', lambda _: query)

    from src.gen import main
    all_generations = main(base_model='llama', cli=True, cli_loop=False, score_model='None',
                           langchain_mode='Disabled',
                           prompt_type=prompt_type,
                           user_path=None,
                           visible_langchain_modes=[],
                           document_subset=DocumentChoices.Relevant.name,
                           verbose=True)

    print(all_generations)
    assert len(all_generations) == 1
    assert "I'm a software engineer with a passion for building scalable" in all_generations[0] or \
           "how can I assist" in all_generations[0] or \
           "am a virtual assistant" in all_generations[0]


@wrap_test_forked
def test_cli_h2ogpt(monkeypatch):
    query = "What is the Earth?"
    monkeypatch.setattr('builtins.input', lambda _: query)

    from src.gen import main
    all_generations = main(base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b', cli=True, cli_loop=False, score_model='None')

    assert len(all_generations) == 1
    assert "The Earth is a planet in the Solar System." in all_generations[0] or \
           "The Earth is the third planet" in all_generations[0]


@wrap_test_forked
def test_cli_langchain_h2ogpt(monkeypatch):
    from tests.utils import make_user_path_test
    user_path = make_user_path_test()

    query = "What is the cat doing?"
    monkeypatch.setattr('builtins.input', lambda _: query)

    from src.gen import main
    all_generations = main(base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b',
                           cli=True, cli_loop=False, score_model='None',
                           langchain_mode='UserData',
                           user_path=user_path,
                           visible_langchain_modes=['UserData', 'MyData'],
                           document_subset=DocumentChoices.Relevant.name,
                           verbose=True)

    print(all_generations)
    assert len(all_generations) == 1
    assert "pexels-evg-kowalievska-1170986_small.jpg" in all_generations[0]
    assert "looking out the window" in all_generations[0] or "staring out the window at the city skyline" in \
           all_generations[0]
