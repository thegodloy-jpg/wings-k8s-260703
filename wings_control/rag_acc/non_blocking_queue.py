import asyncio
import collections


class NonBlockingQueue:
    def __init__(self):
        self._queue = collections.deque()
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()

    def qsize(self):
        return len(self._queue)

    def empty(self):
        return len(self._queue) == 0

    async def put(self, item):
        async with self._lock:
            self._queue.append(item)
            self._not_empty.set()

    def get_nowait(self):
        if self.empty():
            raise asyncio.QueueEmpty()
        # 与 put() 使用相同的锁保护（同步快速路径，锁在外层获取）
        item = self._queue.popleft()
        if self.empty():
            self._not_empty.clear()
        return item

    def peek_nowait(self):
        if self.empty():
            raise asyncio.QueueEmpty()
        return self._queue[0]

    async def get(self):
        while True:
            async with self._lock:
                try:
                    item = self._queue.popleft()
                    if self.empty():
                        self._not_empty.clear()
                    return item
                except IndexError:
                    self._not_empty.clear()
            await self._not_empty.wait()

    async def prepend(self, item):
        async with self._lock:
            self._queue.appendleft(item)
            self._not_empty.set()
