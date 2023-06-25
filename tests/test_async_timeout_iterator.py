import unittest
import asyncio

from iterators import AsyncTimeoutIterator


async def iter_simple():
    yield 1
    yield 2


async def iter_with_sleep():
    yield 1
    await asyncio.sleep(0.6)
    yield 2
    await asyncio.sleep(0.4)
    yield 3


async def iter_with_exception():
    yield 1
    yield 2
    raise Exception
    yield 3


class TestTimeoutIterator(unittest.TestCase):

    def test_normal_iteration(self):

        async def _(self):

            i = iter_simple()
            it = AsyncTimeoutIterator(i)

            self.assertEqual(await it.__anext__(), 1)
            self.assertEqual(await it.__anext__(), 2)

            with self.assertRaises(StopAsyncIteration):
                await it.__anext__()
            with self.assertRaises(StopAsyncIteration):
                await it.__anext__()

        asyncio.get_event_loop().run_until_complete(_(self))

    def test_normal_iteration_for_loop(self):

        async def _(self):

            i = iter_simple()
            it = AsyncTimeoutIterator(i)
            iterResults = []
            async for x in it: 
                iterResults.append(x)        
            self.assertEqual(iterResults, [1,2])

        asyncio.get_event_loop().run_until_complete(_(self))

    def test_timeout_block(self):

        async def _(self):
            i = iter_with_sleep()
            it = AsyncTimeoutIterator(i)
            self.assertEqual(await it.__anext__(), 1)
            self.assertEqual(await it.__anext__(), 2)
            self.assertEqual(await it.__anext__(), 3)

            with self.assertRaises(StopAsyncIteration):
                await it.__anext__()
            with self.assertRaises(StopAsyncIteration):
                            await it.__anext__()

        asyncio.get_event_loop().run_until_complete(_(self))

    def test_timeout_block_for_loop(self):

        async def _(self):
            i = iter_with_sleep()
            it = AsyncTimeoutIterator(i)
            iterResults = []
            async for x in it: 
                iterResults.append(x)        
            self.assertEqual(iterResults, [1,2,3])

        asyncio.get_event_loop().run_until_complete(_(self))

    def test_fixed_timeout(self):

        async def _(self):
            i = iter_with_sleep()
            it = AsyncTimeoutIterator(i, timeout=0.5)

            self.assertEqual(await it.__anext__(), 1)
            self.assertEqual(await it.__anext__(), it.get_sentinel())
            self.assertEqual(await it.__anext__(), 2)
            self.assertEqual(await it.__anext__(), 3)
            with self.assertRaises(StopAsyncIteration):
                await it.__anext__()
        asyncio.get_event_loop().run_until_complete(_(self))
        
    def test_fixed_timeout(self):

        async def _(self):
            i = iter_with_sleep()
            it = AsyncTimeoutIterator(i, timeout=0.5)
            iterResults = []
            async for x in it: 
                iterResults.append(x)        
            self.assertEqual(iterResults, [1,it.get_sentinel(),2,3])
                
        asyncio.get_event_loop().run_until_complete(_(self))

    def test_timeout_update(self):
        async def _(self):
            i = iter_with_sleep()
            it = AsyncTimeoutIterator(i, timeout=0.5)

            self.assertEqual(await it.__anext__(), 1)
            self.assertEqual(await it.__anext__(), it.get_sentinel())

            it.set_timeout(0.3)
            self.assertEqual(await it.__anext__(), 2)
            self.assertEqual(await it.__anext__(), it.get_sentinel())

            self.assertEqual(await it.__anext__(), 3)

            with self.assertRaises(StopAsyncIteration):
                await it.__anext__()

        asyncio.get_event_loop().run_until_complete(_(self))

    def test_custom_sentinel(self):
        async def _(self):
            i = iter_with_sleep()
            it = AsyncTimeoutIterator(i, timeout=0.5, sentinel="END")
            self.assertEqual(await it.__anext__(), 1)
            self.assertEqual(await it.__anext__(), "END")

            self.assertEqual(await it.__anext__(), 2)
            self.assertEqual(await it.__anext__(), 3)

            with self.assertRaises(StopAsyncIteration):
                await it.__anext__()

        asyncio.get_event_loop().run_until_complete(_(self))

    def test_feature_timeout_reset(self):
        async def _(self):
            i = iter_with_sleep()
            it = AsyncTimeoutIterator(i, timeout=0.5, reset_on_next=True)

            self.assertEqual(await it.__anext__(), 1) # timeout gets reset after first iteration
            self.assertEqual(await it.__anext__(), 2)
            self.assertEqual(await it.__anext__(), 3)

            with self.assertRaises(StopAsyncIteration):
                await it.__anext__()

        asyncio.get_event_loop().run_until_complete(_(self))

    def test_function_set_reset_on_next(self):
        async def _(self):
            i = iter_with_sleep()
            it = AsyncTimeoutIterator(i, timeout=0.35, reset_on_next=False)

            self.assertEqual(await it.__anext__(), 1)
            self.assertEqual(await it.__anext__(), it.get_sentinel())
            it.set_reset_on_next(True)
            self.assertEqual(await it.__anext__(), 2)
            self.assertEqual(await it.__anext__(), 3)

            with self.assertRaises(StopAsyncIteration):
                await it.__anext__()

        asyncio.get_event_loop().run_until_complete(_(self))

    def test_iterator_raises_exception(self):
        async def _(self):
            i = iter_with_exception()
            it = AsyncTimeoutIterator(i, timeout=0.5, sentinel="END")
            self.assertEqual(await it.__anext__(), 1)
            self.assertEqual(await it.__anext__(), 2)

            with self.assertRaises(Exception):
                await it.__anext__()
            with self.assertRaises(StopAsyncIteration):
                await it.__anext__()

        asyncio.get_event_loop().run_until_complete(_(self))


    def test_interrupt_thread(self):
        async def _(self):
            i = iter_with_sleep()
            it = AsyncTimeoutIterator(i, timeout=0.5, sentinel="END")
            self.assertEqual(await it.__anext__(), 1)
            self.assertEqual(await it.__anext__(), it.get_sentinel())
            it.interrupt()
            self.assertEqual(await it.__anext__(), 2)

            with self.assertRaises(StopAsyncIteration):
                await it.__anext__()

        asyncio.get_event_loop().run_until_complete(_(self))
