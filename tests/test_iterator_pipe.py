import unittest
from iterators import IteratorPipe


class TestQueueToIterator(unittest.TestCase):

    def test_normal_iteration(self):
        it = IteratorPipe()

        it.put(1)
        it.put(2)
        it.put(3)
        it.close()  # stop iteration

        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), 2)
        self.assertEqual(next(it), 3)
        self.assertRaises(StopIteration, next, it)

    def test_normal_custom_sentinel(self):
        sentinel = object()
        it = IteratorPipe(sentinel=sentinel)

        it.put(1)
        it.put(2)
        it.put(3)
        it.put(sentinel)  # stop iteration

        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), 2)
        self.assertEqual(next(it), 3)
        self.assertRaises(StopIteration, next, it)

    def test_multiple_close(self):
        sentinel = object()
        it = IteratorPipe(sentinel=sentinel)

        it.put(1)
        it.put(2)
        it.put(3)
        it.close()  # stop iteration
        it.close()  # stop iteration
        it.close()  # stop iteration

        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), 2)
        self.assertEqual(next(it), 3)
        self.assertRaises(StopIteration, next, it)

    def test_multiple_next_after_close(self):
        sentinel = object()
        it = IteratorPipe(sentinel=sentinel)

        it.put(1)
        it.put(2)
        it.put(3)
        it.close()  # stop iteration

        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), 2)
        self.assertEqual(next(it), 3)
        self.assertRaises(StopIteration, next, it)
        self.assertRaises(StopIteration, next, it)
        self.assertRaises(StopIteration, next, it)

    def test_put_after_close(self):
        sentinel = object()
        it = IteratorPipe(sentinel=sentinel)

        self.assertTrue(it.put(1))
        it.close()  # stop iteration

        self.assertFalse(it.put(2))
        it.close()  # stop iteration

        self.assertFalse(it.put(3))
        it.close()  # stop iteration

        self.assertEqual(next(it), 1)
        self.assertRaises(StopIteration, next, it)
        self.assertRaises(StopIteration, next, it)
