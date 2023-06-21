from tests.utils import wrap_test_forked


@wrap_test_forked
def test_transformers():
    import transformers
    assert transformers is not None
