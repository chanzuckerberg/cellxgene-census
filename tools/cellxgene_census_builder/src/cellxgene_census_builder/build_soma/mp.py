import contextlib
import logging
import multiprocessing
import queue
import threading
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from types import TracebackType
from typing import Any, Callable, Generic, Iterable, Iterator, Mapping, Optional, ParamSpec, TypeVar, Union

import attrs

from ..build_state import CensusBuildArgs
from ..util import cpu_count, process_init


def _mp_config_checks() -> bool:
    # We rely on the pool configuration being correct. Failure to do this will
    # lead to strange errors on some OS (eg., Linux defaults to fork). Rather
    # than chase those bugs, assert correct configuration.
    assert multiprocessing.get_start_method(True) == "spawn"

    return True


def n_workers_from_memory_budget(args: CensusBuildArgs, per_worker_budget: int) -> int:
    n_workers: int = int(args.config.memory_budget // per_worker_budget)
    return min(max(1, n_workers), cpu_count() + 2)


def create_process_pool_executor(args: CensusBuildArgs, max_workers: Optional[int] = None) -> ProcessPoolExecutor:
    assert _mp_config_checks()
    logging.debug(f"create_process_pool_executor [max_workers={max_workers}]")
    return ProcessPoolExecutor(max_workers=max_workers, initializer=process_init, initargs=(args,))


def create_thread_pool_executor(max_workers: Optional[int] = None) -> ThreadPoolExecutor:
    assert _mp_config_checks()
    logging.debug(f"create_thread_pool_executor [max_workers={max_workers}]")
    return ThreadPoolExecutor(max_workers=max_workers)


def log_on_broken_process_pool(ppe: Union[ProcessPoolExecutor, "ResourcePoolProcessExecutor"]) -> None:
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
_P = ParamSpec("_P")


class EagerIterator(Iterator[_T]):
    def __init__(
        self,
        iterator: Iterator[_T],
        pool: Optional[Executor] = None,
    ):
        super().__init__()
        self.iterator = iterator
        self._pool = pool or ThreadPoolExecutor()
        self._own_pool = pool is None
        self._future: Optional[Future[_T]] = None
        self._fetch_next()

    def _fetch_next(self) -> None:
        self._future = self._pool.submit(self.iterator.__next__)
        logging.debug("EagerIterator: fetching next iterator element, eagerly")

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
        logging.debug("EagerIterator: cleaning up eager iterator")
        if self._own_pool:
            self._pool.shutdown()

    def __del__(self) -> None:
        # Ensure the threadpool is cleaned up in the case where the
        # iterator is not exhausted. For more information on __del__:
        # https://docs.python.org/3/reference/datamodel.html#object.__del__
        self._cleanup()
        super_del = getattr(super(), "__del__", lambda: None)
        super_del()


@attrs.define
class _WorkItem(Generic[_T]):
    resources: int
    future: Future[_T]
    fn: Callable[..., _T]
    args: Iterable[Any]
    kwargs: Mapping[str, Any]


def _work_item_done(scheduler: "_SchedulerThread", wi: _WorkItem[_T], future: Future[_T]) -> None:
    """Callback when Future is cancelled or completes."""
    assert not future.running() and future.done()
    if future.cancelled():
        wi.future.cancel()
    else:
        exc = future.exception()
        if exc is None:
            wi.future.set_result(future.result())
        else:
            wi.future.set_exception(exc)

    scheduler._release_resouces(wi)


class _SchedulerThread(threading.Thread):
    def __init__(self, executor: "ResourcePoolProcessExecutor", max_resources: int):
        super().__init__(name="ResourcePoolProcessExecutor_scheduler")
        self.executor = executor
        self.max_resources: int = max_resources

        self.resources_in_use: int = 0

        self.shutdown_requested: bool = False

        self.mutex_lock: threading.Lock = threading.Lock()
        self.wakeup: threading.Condition = threading.Condition(self.mutex_lock)

    def shutdown(self) -> None:
        self.shutdown_requested = True
        with self.wakeup:
            self.wakeup.notify()

    def run(self) -> None:
        while True:
            with self.wakeup:
                while not self.shutdown_requested and self.executor.pending_work.qsize() == 0:
                    self.wakeup.wait()

                if self.shutdown_requested:
                    logging.debug("ResourcePoolProcessExecutor: shutdown request received by scheduler")
                    return

                # there is work, so peek to see how much. We may need to wait for resources to be available.
                #
                # TODO: this always schedules in LIFO order. There are opportunties for better
                # packing algos, which would lead to higher resource utilization.
                #
                wi: _WorkItem[Any] = self.executor.pending_work.queue[0]
                while (
                    not self.shutdown_requested
                    and (self.resources_in_use > 0)
                    and (wi.resources + self.resources_in_use > self.max_resources)
                ):
                    self.wakeup.wait()

                if self.shutdown_requested:
                    logging.debug("ResourcePoolProcessExecutor: shutdown request received by scheduler")  # type: ignore[unreachable]
                    return

                logging.debug(
                    "ResourcePoolProcessExecutor: adding work item to process pool "
                    f"[free={self.max_resources-self.resources_in_use}, "
                    f"utilization={self.resources_in_use/self.max_resources:0.3f}]"
                )
                wi = self.executor.pending_work.get(block=False)
                self.resources_in_use += wi.resources
                ftr: Future[Any] = self.executor.process_pool.submit(wi.fn, *wi.args, **wi.kwargs)
                ftr.add_done_callback(partial(_work_item_done, self, wi))

    def _release_resouces(self, wi: _WorkItem[Any]) -> None:
        with self.wakeup:
            self.resources_in_use -= wi.resources
            self.wakeup.notify()


class ResourcePoolProcessExecutor(contextlib.AbstractContextManager["ResourcePoolProcessExecutor"]):
    def __init__(self, max_resources: int, *args: Any, **kwargs: Any):
        super().__init__()
        self.pending_work: queue.Queue[_WorkItem[Any]] = queue.Queue()
        self.process_pool: ProcessPoolExecutor = ProcessPoolExecutor(*args, **kwargs)
        self.scheduler = _SchedulerThread(self, max_resources)
        self.scheduler.start()

    @property
    def _broken(self) -> bool:
        return self.process_pool._broken

    def submit(self, resources: int, fn: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> Future[_T]:
        f = Future[_T]()
        w = _WorkItem[_T](resources=resources, future=f, fn=fn, args=args, kwargs=kwargs)
        self.pending_work.put(w)
        with self.scheduler.wakeup:
            self.scheduler.wakeup.notify()
        return f

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.scheduler.shutdown()
        while not self.pending_work.empty():
            self.pending_work.get(block=False).future.cancel()
        self.process_pool.shutdown(wait, cancel_futures=cancel_futures)

    def __exit__(
        self, exc_type: Optional[type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        self.shutdown(wait=True)
        return None


def create_resource_pool_executor(
    args: CensusBuildArgs, max_resources: Optional[int] = None, max_workers: Optional[int] = None
) -> ResourcePoolProcessExecutor:
    assert _mp_config_checks()
    if max_resources is None:
        max_resources = args.config.memory_budget
    if max_workers is None:
        max_workers = cpu_count() + 2
    logging.debug(f"create_resource_pool_executor [max_workers={max_workers}, max_resources={max_resources}]")
    return ResourcePoolProcessExecutor(
        max_resources=max_resources, max_workers=max_workers, initializer=process_init, initargs=(args,)
    )
