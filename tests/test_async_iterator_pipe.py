import unittest
import asyncio
from iterators import AsyncIteratorPipe


class TestTimeoutIterator(unittest.TestCase):

    def test_normal_iteration(self):

        async def _(self):
            it = AsyncIteratorPipe()

            await it.put(1)
            await it.put(2)
            await it.put(3)
            await it.close()  # stop iteration

            self.assertEqual(await it.__anext__(), 1)
            self.assertEqual(await it.__anext__(), 2)
            self.assertEqual(await it.__anext__(), 3)

            with self.assertRaises(StopAsyncIteration):
                await it.__anext__()

        asyncio.get_event_loop().run_until_complete(_(self))

    def test_multiple_next_after_exception(self):

        async def _(self):
            it = AsyncIteratorPipe()

            await it.put(1)
            await it.put(2)
            await it.put(3)
            await it.close()  # stop iteration

            self.assertEqual(await it.__anext__(), 1)
            self.assertEqual(await it.__anext__(), 2)
            self.assertEqual(await it.__anext__(), 3)

            with self.assertRaises(StopAsyncIteration):
                await it.__anext__()

            with self.assertRaises(StopAsyncIteration):
                await it.__anext__()

        asyncio.get_event_loop().run_until_complete(_(self))


    def test_multiple_close(self):

        async def _(self):
            it = AsyncIteratorPipe()

            await it.put(1)
            await it.put(2)
            await it.put(3)
            await it.close()  # stop iteration
            await it.close()  # stop iteration
            await it.close()  # stop iteration

            self.assertEqual(await it.__anext__(), 1)
            self.assertEqual(await it.__anext__(), 2)
            self.assertEqual(await it.__anext__(), 3)

            with self.assertRaises(StopAsyncIteration):
                await it.__anext__()

        asyncio.get_event_loop().run_until_complete(_(self))


    def test_put_after_close(self):

        async def _(self):
            it = AsyncIteratorPipe()

            self.assertTrue(await it.put(1))
            await it.close()  # stop iteration

            self.assertFalse(await it.put(2))
            await it.close()  # stop iteration

            self.assertFalse(await it.put(3))
            await it.close()  # stop iteration

            self.assertEqual(await it.__anext__(), 1)

            with self.assertRaises(StopAsyncIteration):
                await it.__anext__()

        asyncio.get_event_loop().run_until_complete(_(self))

    def test_normal_iteration_via_for_loop(self):

        async def _(self):
            it = AsyncIteratorPipe()
            await it.put(1)
            await it.put(2)
            await it.put(3)
            await it.close()

            iter_results = []
            async for x in it:
                iter_results.append(x)
            self.assertEqual(iter_results, [1,2,3])

            iter_results = []
            async for x in it:
                iter_results.append(x)
            self.assertEqual(iter_results, [])

        asyncio.get_event_loop().run_until_complete(_(self))
