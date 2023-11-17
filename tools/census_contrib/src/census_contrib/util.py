from __future__ import annotations

import logging
import math
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Iterator, Optional, TypeVar, cast

import tiledbsoma as soma


def get_logger() -> logging.Logger:
    return logging.getLogger("census_contrib")


DEFAULT_READ_BUFFER_SIZE = 4 * 1024**3
MAX_NNZ_GOAL = DEFAULT_READ_BUFFER_SIZE // 8  # sizeof(int64) - worst case size


def blocksize(n_features: int, nnz_goal: int = MAX_NNZ_GOAL) -> int:
    """
    Given an nnz goal, and n_features, return step size for a blockwise iterator.
    """
    nnz_goal = max(nnz_goal, MAX_NNZ_GOAL)
    return cast(int, 2 ** round(math.log2((nnz_goal) / n_features)))


def soma_context() -> soma.options.SOMATileDBContext:
    """Return soma context with default config"""
    return soma.options.SOMATileDBContext(
        tiledb_config={
            "py.init_buffer_bytes": DEFAULT_READ_BUFFER_SIZE + 10 * 1024,
            "soma.init_buffer_bytes": DEFAULT_READ_BUFFER_SIZE + 10 * 1024,
        }
    )


_T = TypeVar("_T")


class EagerIterator(Iterator[_T]):
    def __init__(
        self,
        iterator: Iterator[_T],
        pool: Optional[ThreadPoolExecutor] = None,
    ):
        super().__init__()
        self.iterator = iterator
        self._pool = pool or ThreadPoolExecutor()
        self._own_pool = pool is None
        self._future: Optional[Future[_T]] = None
        self._fetch_next()

    def _fetch_next(self) -> None:
        self._future = self._pool.submit(self.iterator.__next__)

    def __next__(self) -> _T:
        try:
            assert self._future
            res = self._future.result()
            self._fetch_next()
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
