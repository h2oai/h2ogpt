import unittest
import time

from iterators import TimeoutIterator


def iter_simple():
    yield 1
    yield 2


def iter_with_sleep():
    yield 1
    time.sleep(0.6)
    yield 2
    time.sleep(0.4)
    yield 3


def iter_with_exception():
    yield 1
    yield 2
    raise Exception
    yield 3


class TestTimeoutIterator(unittest.TestCase):

    def test_normal_iteration(self):
        i = iter_simple()
        it = TimeoutIterator(i)

        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), 2)

        self.assertRaises(StopIteration, next, it)
        self.assertRaises(StopIteration, next, it)

    def test_normal_iteration_for_loop(self):
        i = iter_simple()
        it = TimeoutIterator(i)
        iterResults = []
        for x in it:
            iterResults.append(x)
        self.assertEqual(iterResults, [1, 2])

    def test_timeout_block(self):
        i = iter_with_sleep()
        it = TimeoutIterator(i)
        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), 2)
        self.assertEqual(next(it), 3)
        self.assertRaises(StopIteration, next, it)
        self.assertRaises(StopIteration, next, it)

    def test_timeout_block_for_loop(self):
        i = iter_with_sleep()
        it = TimeoutIterator(i)
        iterResults = []
        for x in it:
            iterResults.append(x)
        self.assertEqual(iterResults, [1, 2, 3])

    def test_fixed_timeout(self):
        i = iter_with_sleep()
        it = TimeoutIterator(i, timeout=0.5)
        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), it.get_sentinel())

        self.assertEqual(next(it), 2)
        self.assertEqual(next(it), 3)
        self.assertRaises(StopIteration, next, it)

    def test_fixed_timeout_for_loop(self):
        i = iter_with_sleep()
        it = TimeoutIterator(i, timeout=0.5)
        iterResults = []
        for x in it:
            iterResults.append(x)
        self.assertEqual(iterResults, [1, it.get_sentinel(), 2, 3])

    def test_timeout_update(self):
        i = iter_with_sleep()
        it = TimeoutIterator(i, timeout=0.5)
        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), it.get_sentinel())

        it.set_timeout(0.3)
        self.assertEqual(next(it), 2)
        self.assertEqual(next(it), it.get_sentinel())

        self.assertEqual(next(it), 3)
        self.assertRaises(StopIteration, next, it)

    def test_custom_sentinel(self):
        i = iter_with_sleep()
        it = TimeoutIterator(i, timeout=0.5, sentinel="END")
        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), "END")

        self.assertEqual(next(it), 2)
        self.assertEqual(next(it), 3)
        self.assertRaises(StopIteration, next, it)

    def test_feature_timeout_reset(self):
        i = iter_with_sleep()
        it = TimeoutIterator(i, timeout=0.5, reset_on_next=True)
        self.assertEqual(next(it), 1)  # timeout gets reset after first iteration
        self.assertEqual(next(it), 2)
        self.assertEqual(next(it), 3)
        self.assertRaises(StopIteration, next, it)

    def test_function_set_reset_on_next(self):
        i = iter_with_sleep()
        it = TimeoutIterator(i, timeout=0.35, reset_on_next=False)
        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), it.get_sentinel())
        it.set_reset_on_next(True)
        self.assertEqual(next(it), 2)
        self.assertEqual(next(it), 3)
        self.assertRaises(StopIteration, next, it)

    def test_iterator_raises_exception(self):
        i = iter_with_exception()
        it = TimeoutIterator(i, timeout=0.5, sentinel="END")
        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), 2)
        self.assertRaises(Exception, next, it)
        self.assertRaises(StopIteration, next, it)

    def test_interrupt_thread(self):
        i = iter_with_sleep()
        it = TimeoutIterator(i, timeout=0.5, sentinel="END")
        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), it.get_sentinel())
        it.interrupt()
        self.assertEqual(next(it), 2)
        self.assertRaises(StopIteration, next, it)
