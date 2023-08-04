from src.utils import get_list_or_str
from tests.utils import wrap_test_forked


@wrap_test_forked
def test_get_list_or_str():
    assert get_list_or_str(['foo', 'bar']) == ['foo', 'bar']
    assert get_list_or_str('foo') == 'foo'
    assert get_list_or_str("['foo', 'bar']") == ['foo', 'bar']
