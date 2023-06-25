import queue
import asyncio


class IteratorPipe:
    """
    Iterator Pipe creates an iterator that can be fed in data from another block of code or thread of execution
    """

    def __init__(self, sentinel=object()):
        self._q = queue.Queue()
        self._sentinel = sentinel
        self._sentinel_pushed = False
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration

        data = self._q.get(block=True)
        if data is self._sentinel:
            self._closed = True
            raise StopIteration

        return data

    def put(self, data) -> bool:
        """
        Pushes next item to Iterator and returns True
        If iterator has been closed via close(), doesn't push anything and returns False
        """
        if self._sentinel_pushed:
            return False

        self._q.put(data)
        return True

    def close(self):
        """
        Close is idempotent. Calling close multiple times is safe
        Iterator will raise StopIteration only after all elements pushed before close have been iterated
        """
        # make close idempotent
        if not self._sentinel_pushed:
            self._sentinel_pushed = True
        self._q.put(self._sentinel)


class AsyncIteratorPipe:

    def __init__(self, sentinel=object()):
        self._q = asyncio.Queue()
        self._sentinel = sentinel
        self._sentinel_pushed = False
        self._closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration

        data = await self._q.get()
        if data is self._sentinel:
            self._closed = True
            raise StopAsyncIteration

        return data

    async def put(self, data) -> bool:
        """
        Pushes next item to Iterator and returns True
        If iterator has been closed via close(), doesn't push anything and returns False
        """
        if self._sentinel_pushed:
            return False

        await self._q.put(data)
        return True

    async def close(self):
        """
        Close is idempotent. Calling close multiple times is safe
        Iterator will raise StopIteration only after all elements pushed before close have been iterated
        """
        # make close idempotent
        if not self._sentinel_pushed:
            self._sentinel_pushed = True
            await self._q.put(self._sentinel)
