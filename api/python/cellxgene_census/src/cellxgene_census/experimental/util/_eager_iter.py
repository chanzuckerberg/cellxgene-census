import logging
import threading
from collections import deque
from concurrent import futures
from concurrent.futures import Future
from typing import Deque, Iterator, Optional, TypeVar

util_logger = logging.getLogger("cellxgene_census.experimental.util")

_T = TypeVar("_T")


class _EagerIterator(Iterator[_T]):
    def __init__(
        self,
        iterator: Iterator[_T],
        pool: Optional[futures.Executor] = None,
    ):
        super().__init__()
        self.iterator = iterator
        self._pool = pool or futures.ThreadPoolExecutor()
        self._own_pool = pool is None
        self._future: Optional[Future[_T]] = None
        self._begin_next()

    def _begin_next(self) -> None:
        self._future = self._pool.submit(self.iterator.__next__)
        util_logger.debug("Fetching next iterator element, eagerly")

    def __next__(self) -> _T:
        try:
            assert self._future
            res = self._future.result()
            self._begin_next()
            return res
        except StopIteration:
            self._cleanup()
            raise

    def _cleanup(self) -> None:
        util_logger.debug("Cleaning up eager iterator")
        if self._own_pool:
            self._pool.shutdown()

    def __del__(self) -> None:
        # Ensure the threadpool is cleaned up in the case where the
        # iterator is not exhausted. For more information on __del__:
        # https://docs.python.org/3/reference/datamodel.html#object.__del__
        self._cleanup()
        super_del = getattr(super(), "__del__", lambda: None)
        super_del()


class _EagerBufferedIterator(Iterator[_T]):
    def __init__(
        self,
        iterator: Iterator[_T],
        max_pending: int = 1,
        pool: Optional[futures.Executor] = None,
    ):
        super().__init__()
        self.iterator = iterator
        self.max_pending = max_pending
        self._pool = pool or futures.ThreadPoolExecutor()
        self._own_pool = pool is None
        self._pending_results: Deque[futures.Future[_T]] = deque()
        self._lock = threading.Lock()
        self._begin_next()

    def __next__(self) -> _T:
        try:
            res = self._pending_results[0].result()
            self._pending_results.popleft()
            self._begin_next()
            return res
        except StopIteration:
            self._cleanup()
            raise

    def _begin_next(self) -> None:
        def _fut_done(fut: futures.Future[_T]) -> None:
            util_logger.debug("Finished fetching next iterator element, eagerly")
            if fut.exception() is None:
                self._begin_next()

        with self._lock:
            not_running = len(self._pending_results) == 0 or self._pending_results[-1].done()
            if len(self._pending_results) < self.max_pending and not_running:
                _future = self._pool.submit(self.iterator.__next__)
                util_logger.debug("Fetching next iterator element, eagerly")
                _future.add_done_callback(_fut_done)
                self._pending_results.append(_future)
            assert len(self._pending_results) <= self.max_pending

    def _cleanup(self) -> None:
        util_logger.debug("Cleaning up eager iterator")
        if self._own_pool:
            self._pool.shutdown()

    def __del__(self) -> None:
        # Ensure the threadpool is cleaned up in the case where the
        # iterator is not exhausted. For more information on __del__:
        # https://docs.python.org/3/reference/datamodel.html#object.__del__
        self._cleanup()
        super_del = getattr(super(), "__del__", lambda: None)
        super_del()
