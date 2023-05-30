import pytest

from tests.utils import wrap_test_forked, get_llama


@wrap_test_forked
def test_cli(monkeypatch):
    query = "What is the Earth?"
    monkeypatch.setattr('builtins.input', lambda _: query)

    from generate import main
    all_generations = main(base_model='gptj', cli=True, cli_loop=False, score_model='None')

    assert len(all_generations) == 1
    assert "The Earth is a planet in our solar system that orbits around the Sun." in all_generations[0]


@pytest.mark.xfail(strict=False, reason="GPT4All produces no output if input has new lines etc."
                                        "  See FAQ.md, outside h2oGPT same thing even for single line inputs.")
@wrap_test_forked
def test_cli_langchain(monkeypatch):
    from tests.utils import make_user_path_test
    user_path = make_user_path_test()

    query = "What is the cat doing?"
    monkeypatch.setattr('builtins.input', lambda _: query)

    from generate import main
    all_generations = main(base_model='gptj', cli=True, cli_loop=False, score_model='None',
                           langchain_mode='UserData',
                           user_path=user_path,
                           visible_langchain_modes=['UserData', 'MyData'],
                           document_choice=['All'],
                           verbose=True)

    print(all_generations)
    assert len(all_generations) == 1
    assert "pexels-evg-kowalievska-1170986_small.jpg" in all_generations[0]
    assert "looking out the window" in all_generations[0]


@wrap_test_forked
def test_cli_langchain_llamacpp(monkeypatch):
    prompt_type = get_llama()

    from tests.utils import make_user_path_test
    user_path = make_user_path_test()

    query = "What is the cat doing?"
    monkeypatch.setattr('builtins.input', lambda _: query)

    from generate import main
    all_generations = main(base_model='llama', cli=True, cli_loop=False, score_model='None',
                           langchain_mode='UserData',
                           prompt_type=prompt_type,
                           user_path=user_path,
                           visible_langchain_modes=['UserData', 'MyData'],
                           document_choice=['All'],
                           verbose=True)

    print(all_generations)
    assert len(all_generations) == 1
    assert "pexels-evg-kowalievska-1170986_small.jpg" in all_generations[0]
    assert "It is sitting on a window and looking out at the view" in all_generations[0]


@wrap_test_forked
def test_cli_h2ogpt(monkeypatch):
    query = "What is the Earth?"
    monkeypatch.setattr('builtins.input', lambda _: query)

    from generate import main
    all_generations = main(base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b', cli=True, cli_loop=False, score_model='None')

    assert len(all_generations) == 1
    assert "The Earth is a planet in the Solar System." in all_generations[0]


@wrap_test_forked
def test_cli_langchain_h2ogpt(monkeypatch):
    from tests.utils import make_user_path_test
    user_path = make_user_path_test()

    query = "What is the cat doing?"
    monkeypatch.setattr('builtins.input', lambda _: query)

    from generate import main
    all_generations = main(base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b',
                           cli=True, cli_loop=False, score_model='None',
                           langchain_mode='UserData',
                           user_path=user_path,
                           visible_langchain_modes=['UserData', 'MyData'],
                           document_choice=['All'],
                           verbose=True)

    print(all_generations)
    assert len(all_generations) == 1
    assert "pexels-evg-kowalievska-1170986_small.jpg" in all_generations[0]
    assert "looking out the window" in all_generations[0]
