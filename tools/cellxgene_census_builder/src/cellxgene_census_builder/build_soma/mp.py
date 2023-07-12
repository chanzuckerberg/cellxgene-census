import contextlib
import logging
import multiprocessing
import threading
from collections import deque
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from types import TracebackType
from typing import Any, Callable, Generic, Iterable, Iterator, Literal, Mapping, Optional, ParamSpec, TypeVar, Union

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
    if max_workers is None:
        max_workers = cpu_count() + 2
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


_MightBeWork = Union[bool, _WorkItem[Any]]
_SchedulerMethod = Literal["best-fit", "first-fit"]


def _work_item_done(scheduler: "_Scheduler", wi: _WorkItem[_T], future: Future[_T]) -> None:
    """Callback when Future is cancelled or completes."""
    assert not future.running() and future.done()
    scheduler._debug_msg(f"done - {id(wi):#x} {future}")
    if future.cancelled():
        wi.future.cancel()
    else:
        exc = future.exception()
        if exc is None:
            wi.future.set_result(future.result())
        else:
            wi.future.set_exception(exc)

    scheduler._release_resouces(wi)


class _Scheduler(threading.Thread):
    def __init__(self, executor: "ResourcePoolProcessExecutor", max_resources: int):
        super().__init__(name="ResourcePoolProcessExecutor_scheduler")
        self.executor = executor
        self.max_resources: int = max_resources

        self.resources_in_use: int = 0
        self._pending_work: deque[_WorkItem[Any]] = deque()

        self.shutdown_requested: bool = False

        self._condition: threading.Condition = threading.Condition()

    def shutdown(self) -> None:
        self.shutdown_requested = True
        with self._condition:
            while len(self._pending_work):
                wi = self._pending_work.popleft()
                wi.future.cancel()
            self._condition.notify()

    def submit(self, wi: _WorkItem[_T]) -> Future[_T]:
        f = Future[_T]()
        with self._condition:
            self._pending_work.append(wi)
            self._condition.notify()
        return f

    def _get_work(self, method: _SchedulerMethod = "best-fit") -> _MightBeWork:
        """
        Get next work item to schedule.

        IMPORTANT: caller MUST own scheduler _condition lock to call this.

        Return values:
        * True - shutdown requested
        * False - no work, please wait
        * _WorkItem - some work to schedule
        """

        def _get_best_work(method: _SchedulerMethod) -> int | None:
            """Return index of "best" work item to scheudle, or None if work is unavailable."""
            if method == "first-fit":
                # first fit: return first work item that fits in the available resources
                for i in range(len(self._pending_work)):
                    if self._pending_work[i].resources + self.resources_in_use <= self.max_resources:
                        return i
                return None
            elif method == "best-fit":
                # Best fit: return the largest resource consumer that fits in available space
                max_available_resources = self.max_resources - self.resources_in_use
                candidate_work = filter(
                    lambda v: v[1].resources <= max_available_resources, enumerate(self._pending_work)
                )
                return max(candidate_work, key=lambda v: v[1].resources, default=(None,))[0]
            else:
                raise NotImplementedError("Unknown scheduler method")

        if self.executor.process_pool._broken:
            logging.critical(f"Process pool broken and may fail or hang: {self.executor.process_pool._broken}")
            self.shutdown_requested = True

        if self.shutdown_requested:
            return True

        if len(self._pending_work):
            if (i := _get_best_work(method)) is not None:
                # pop ith item from the deque
                self._pending_work.rotate(-i)
                wi = self._pending_work.popleft()
                self._pending_work.rotate(i)
                return wi

            if self.resources_in_use == 0:
                # We always want at least one job, regardless of cost
                return self._pending_work.popleft()

        return False

    def run(self) -> None:
        while True:
            with self._condition:
                work: _MightBeWork
                while not (work := self._get_work()):
                    self._condition.wait()

                if self.shutdown_requested:
                    assert work is True
                    self._debug_msg("shutdown request received by scheduler")
                    return

                assert isinstance(work, _WorkItem)
                self._debug_msg(f"adding work item {id(work):#x} [resources={work.resources}] to process pool")
                self.resources_in_use += work.resources
                try:
                    ftr: Future[Any] = self.executor.process_pool.submit(work.fn, *work.args, **work.kwargs)
                    ftr.add_done_callback(partial(_work_item_done, self, work))
                    del ftr
                except BrokenProcessPool as bpe:
                    logging.critical(f"Process pool broken and may fail or hang: {bpe.args}")
                    work.future.cancel()
                    self.shutdow_requested = True

                del work  # don't hold onto references

    def _release_resouces(self, wi: _WorkItem[Any]) -> None:
        with self._condition:
            self.resources_in_use -= wi.resources
            self._debug_msg(f"release resources ({wi.resources}) for {id(wi):#x}")
            self._condition.notify()

    def _debug_msg(self, msg: str) -> None:
        logging.debug(
            f"ResourcePoolProcessExecutor: {msg} "
            f"[free={self.max_resources-self.resources_in_use}, "
            f"in_use={self.resources_in_use} "
            f"unsched={len(self._pending_work)} "
            f"sched={len(self.executor.process_pool._pending_work_items)}]"
        )


class ResourcePoolProcessExecutor(contextlib.AbstractContextManager["ResourcePoolProcessExecutor"]):
    """
    Provides a ProcessPoolExecutor-like API, scheduling based upon static "resource" reservation
    requests. A "resource" is any shared capacity or resource, expressed as an integer
    value. Class holds a queue of "work items", scheduling them into an actual ProcessPoolExecutor
    when sufficient resources are available.

    Primarily use case is managing finite memory, by throttling submissions until memory is
    available.
    """

    def __init__(self, max_resources: int, *args: Any, **kwargs: Any):
        super().__init__()
        logging.debug(f"ResourcePoolProcessExecutor: starting process pool with args ({args} {kwargs})")
        self.process_pool: ProcessPoolExecutor = ProcessPoolExecutor(*args, **kwargs)
        self.scheduler = _Scheduler(self, max_resources)
        self.scheduler.start()

    @property
    def _broken(self) -> bool:
        return self.process_pool._broken

    def submit(self, resources: int, fn: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> Future[_T]:
        f = Future[_T]()
        self.scheduler.submit(_WorkItem[_T](resources=resources, future=f, fn=fn, args=args, kwargs=kwargs))
        return f

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.scheduler.shutdown()
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
