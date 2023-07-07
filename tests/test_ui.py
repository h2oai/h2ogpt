from tests.utils import wrap_test_forked


@wrap_test_forked
def test_newline_replace():

    text0 = """You can use the `sorted()` function to merge two sorted lists in Python. The `sorted()` function takes a list as an argument and returns a new sorted list. Here’s an example of how you can use it to merge two sorted lists:

```python
list1 = [1, 2, 3, 4, 5]
list2 = [6, 7, 8, 9, 10]
merged_list = sorted(list1 + list2)<br>print(merged_list)
```

The output of this code is:
```
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

As you can see, the `sorted()` function has merged the two sorted lists into a single sorted list."""

    from src.gradio_runner import fix_text_for_gradio
    fixed = fix_text_for_gradio(text0, fix_new_lines=True)

    expected = """You can use the `sorted()` function to merge two sorted lists in Python. The `sorted()` function takes a list as an argument and returns a new sorted list. Here’s an example of how you can use it to merge two sorted lists:<br><br>```python
list1 = [1, 2, 3, 4, 5]
list2 = [6, 7, 8, 9, 10]
merged_list = sorted(list1 + list2)<br>print(merged_list)
```<br><br>The output of this code is:<br>```
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```<br><br>As you can see, the `sorted()` function has merged the two sorted lists into a single sorted list."""
    assert fixed == expected
