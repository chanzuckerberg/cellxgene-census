import concurrent.futures
import logging
import multiprocessing
import os
from typing import Iterator, Optional, TypeVar, cast

from ..build_state import CensusBuildArgs
from ..util import process_init


def cpu_count() -> int:
    """Sign, os.cpu_count() returns None if "undetermined" number of CPUs"""
    cpu_count = os.cpu_count()
    if os.cpu_count() is None:
        return 1
    return cast(int, cpu_count)


def _mp_config_checks() -> bool:
    # We rely on the pool configuration being correct. Failure to do this will
    # lead to strange errors on some OS (eg., Linux defaults to fork). Rather
    # than chase those bugs, assert correct configuration.
    assert multiprocessing.get_start_method(True) == "spawn"

    return True


def create_process_pool_executor(
    args: CensusBuildArgs, max_workers: Optional[int] = None
) -> concurrent.futures.ProcessPoolExecutor:
    assert _mp_config_checks()
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=args.config.max_workers if max_workers is None else max_workers,
        initializer=process_init,
        initargs=(args,),
    )


def create_thread_pool_executor(
    args: CensusBuildArgs, max_workers: Optional[int] = None
) -> concurrent.futures.ThreadPoolExecutor:
    assert _mp_config_checks()
    return concurrent.futures.ThreadPoolExecutor(
        max_workers=args.config.max_workers if max_workers is None else max_workers,
    )


def log_on_broken_process_pool(ppe: concurrent.futures.ProcessPoolExecutor) -> None:
    """
    There are a number of conditions where the Process Pool can be broken,
    such that it will hang in a shutdown. This will cause the context __exit__
    to hang indefinitely, as it calls ProcessPoolExecutor.shutdown with
    `wait=True`.

    An example condition which can cause a deadlock is an OOM, where a the
    repear kills a process.

    This routine is used to detect the condition and log the error, so a
    human has a chance of detecting/diagnosing.

    Caution: uses ProcessPoolExecutor internal API, as this state is not
    otherwise visible.
    """

    if ppe._broken:
        logging.critical(f"Process pool broken and may fail or hang: {ppe._broken}")

    return


# TODO: when the builder is updated to cellxgene_census 1.3+, we can pull
# this out of the experimental.util package. Until then, it is hard to keep
# it DRY.

_T = TypeVar("_T")


class EagerIterator(Iterator[_T]):
    def __init__(
        self,
        iterator: Iterator[_T],
        pool: Optional[concurrent.futures.Executor] = None,
    ):
        super().__init__()
        self.iterator = iterator
        self._pool = pool or concurrent.futures.ThreadPoolExecutor()
        self._own_pool = pool is None
        self._future: Optional[concurrent.futures.Future[_T]] = None
        self.fetch_next()

    def fetch_next(self) -> None:
        self._future = self._pool.submit(self.iterator.__next__)
        logging.debug("Fetching next iterator element, eagerly")

    def __next__(self) -> _T:
        try:
            assert self._future
            res = self._future.result()
            self.fetch_next()
            return res
        except StopIteration:
            self._cleanup()
            raise

    def _cleanup(self) -> None:
        logging.debug("Cleaning up eager iterator")
        if self._own_pool:
            self._pool.shutdown()

    def __del__(self) -> None:
        # Ensure the threadpool is cleaned up in the case where the
        # iterator is not exhausted. For more information on __del__:
        # https://docs.python.org/3/reference/datamodel.html#object.__del__
        self._cleanup()
        super_del = getattr(super(), "__del__", lambda: None)
        super_del()
