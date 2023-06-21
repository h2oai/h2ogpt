import os

import nltk
import pytest

from tests.utils import wrap_test_forked


def nltkTokenize(text):
    words = nltk.word_tokenize(text)
    return words


import re

WORD = re.compile(r'\w+')


def regTokenize(text):
    words = WORD.findall(text)
    return words


import time


@pytest.mark.skipif(not os.getenv('MEASURE'), reason="For checking token length for various methods: MEASURE=1 pytest -s -v tests/test_tokenizer.py")
@wrap_test_forked
def test_tokenizer1():
    prompt = """Here is an example of how to write a Python program to generate the Fibonacci sequence:
    
    
    
    
    def fib(n):
        a, b = 0, 1
        if n == 0 or n == 1:
            return a
        for i in range(n-2):
            a, b = b, a+b
        return b
    
    for i in range(10):
        print(fib(i))
    This program defines a function called fib that takes an integer n as input and returns the nth Fibonacci number. The function uses two variables a and b to keep track of the current and previous Fibonacci numbers.
    
    The first two lines of the function check if n is either 0 or 1, in which case the function returns 0 or 1 respectively. If n is greater than 1, the function iterates over the range of integers from 2 to n-1, adding the previous two Fibonacci numbers to get the current Fibonacci number. Finally, the function returns the last Fibonacci number calculated.
    
    In the main part of the program, we use a for loop to call the fib function with different"""

    prompt = os.getenv('PROMPT', prompt)
    run_tokenizer1(prompt)


def run_tokenizer1(prompt):
    from transformers import AutoTokenizer

    t = AutoTokenizer.from_pretrained("distilgpt2")
    llm_tokenizer = AutoTokenizer.from_pretrained('h2oai/h2ogpt-oig-oasst1-512-6_9b')

    from InstructorEmbedding import INSTRUCTOR
    emb = INSTRUCTOR('hkunlp/instructor-large')

    t0 = time.time()
    a = len(regTokenize(prompt))
    print("Regexp Tokenizer", a, time.time() - t0)

    t0 = time.time()
    a = len(nltkTokenize(prompt))
    print("NLTK Tokenizer", a, time.time() - t0)

    t0 = time.time()
    a = len(t(prompt)['input_ids'])
    print("Slow Tokenizer", a, time.time() - t0)

    t0 = time.time()
    a = len(llm_tokenizer(prompt)['input_ids'])
    print("Fast Tokenizer LLM", a, time.time() - t0)

    t0 = time.time()
    a = emb.tokenize([prompt])['input_ids'].shape[1]
    print("Instruct Embedding", a, time.time() - t0)


if __name__ == '__main__':
    test_tokenizer1()

