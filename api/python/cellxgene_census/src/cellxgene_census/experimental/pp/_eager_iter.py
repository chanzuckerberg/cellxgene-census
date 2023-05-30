from concurrent import futures
from typing import Iterator, Optional, TypeVar

_T = TypeVar("_T")


class EagerIterator(Iterator[_T]):
    def __init__(
        self,
        iterator: Iterator[_T],
        pool: Optional[futures.Executor] = None,
    ):
        super().__init__()
        self.iterator = iterator
        self._pool = pool or futures.ThreadPoolExecutor()
        self._own_pool = pool is None
        self._future = self._pool.submit(self.iterator.__next__)

    def __next__(self) -> _T:
        try:
            res = self._future.result()
            self._future = self._pool.submit(self.iterator.__next__)
            return res
        except StopIteration:
            self._cleanup()
            raise

    def _cleanup(self) -> None:
        if self._own_pool:
            self._pool.shutdown()

    def __del__(self) -> None:
        # Ensure the threadpool is cleaned up in the case where the
        # iterator is not exhausted. For more information on __del__:
        # https://docs.python.org/3/reference/datamodel.html#object.__del__
        self._cleanup()
        super_del = getattr(super(), "__del__", lambda: None)
        super_del()
