import contextlib
import logging
import multiprocessing
import threading
import weakref
from collections import deque
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from types import TracebackType
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    ParamSpec,
    TypeVar,
    Union,
)

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
    """Trivial helper to estimate appropriate number of fixed-memory-budget workers from total memory available"""
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
# EagerIterator out of the experimental.util package. Until then, it is hard to keep
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


class _Scheduler(threading.Thread):
    def __init__(self, executor: "ResourcePoolProcessExecutor", max_resources: int):
        super().__init__(name="ResourcePoolProcessExecutor_scheduler")

        def _weakref_collected(_: weakref.ReferenceType[Any], scheduler: "_Scheduler" = self) -> None:
            scheduler.shutdown()

        assert isinstance(executor.process_pool, multiprocessing.pool.Pool)
        self.executor_ref = weakref.ref(executor, _weakref_collected)
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

        TODO: if it becomes useful, expose scheduler algo at top-level.
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
                # We always want at least one job, regardless of cost of the work item.
                return self._pending_work.popleft()

        return False  # no work for you

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
                self._schedule_work(work)
                del work  # don't hold onto references

    @classmethod
    def _work_item_done(
        cls, scheduler: "_Scheduler", wi: _WorkItem[_T], is_error_callback: bool, result: _T | BaseException
    ) -> None:
        """Callback when async_apply is complete."""
        if is_error_callback:
            assert isinstance(result, BaseException)
            wi.future.set_exception(result)
        else:
            assert not isinstance(result, BaseException)
            wi.future.set_result(result)
        scheduler._release_resouces(wi)

    def _schedule_work(self, work: _WorkItem[Any]) -> None:
        """must hold lock"""
        executor = self.executor_ref()
        if executor is None:
            # can happen if the ResourcePoolExeuctor was collected
            return
        self._debug_msg(f"adding work to pool {id(work):#x} [resources={work.resources}]")
        self.resources_in_use += work.resources
        _work_item_done = partial(self._work_item_done, self, work, False)
        _work_item_error = partial(self._work_item_done, self, work, True)
        executor.process_pool.apply_async(
            work.fn, work.args, work.kwargs, callback=_work_item_done, error_callback=_work_item_error
        )

    def _release_resouces(self, wi: _WorkItem[Any]) -> None:
        with self._condition:
            self.resources_in_use -= wi.resources
            self._condition.notify()

    def _debug_msg(self, msg: str) -> None:
        logging.debug(
            f"ResourcePoolProcessExecutor: {msg} ["
            f"free={self.max_resources-self.resources_in_use} "
            f"in_use={self.resources_in_use} "
            f"unsched={len(self._pending_work)}"
            "]"
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
        _mp_config_checks()

        super().__init__()
        logging.debug(f"ResourcePoolProcessExecutor: starting process pool with args ({args} {kwargs})")

        max_workers = kwargs.pop("max_workers", None)
        initializer = kwargs.pop("initializer", None)
        initargs = kwargs.pop("initargs", None)
        max_tasks_per_child = kwargs.pop("max_tasks_per_child", None)
        self.process_pool: multiprocessing.pool.Pool = multiprocessing.Pool(
            processes=max_workers, initializer=initializer, initargs=initargs, maxtasksperchild=max_tasks_per_child
        )

        # create and start scheduler thread
        self.scheduler = _Scheduler(self, max_resources)
        self.scheduler.start()

    @property
    def _broken(self) -> bool:
        return self.process_pool._state not in ["RUN", "INIT", "CLOSE"]  # type: ignore[attr-defined]

    def submit(self, resources: int, fn: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> Future[_T]:
        f = Future[_T]()
        self.scheduler.submit(_WorkItem[_T](resources=resources, future=f, fn=fn, args=args, kwargs=kwargs))
        return f

    # TODO: implement map

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.scheduler.shutdown()
        self.process_pool.close()

    def __exit__(
        self, exc_type: Optional[type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        self.shutdown(wait=True)
        return None


def create_resource_pool_executor(
    args: CensusBuildArgs,
    max_resources: Optional[int] = None,
    max_workers: Optional[int] = None,
    max_tasks_per_child: Optional[int] = None,
) -> ResourcePoolProcessExecutor:
    assert _mp_config_checks()

    if max_resources is None:
        max_resources = args.config.memory_budget
    if max_workers is None:
        max_workers = cpu_count() + 2
    if max_tasks_per_child is None:
        # not strictly necessary, but helps avoid leaks by turning over sub-processes
        max_tasks_per_child = 10

    logging.debug(f"create_resource_pool_executor [max_workers={max_workers}, max_resources={max_resources}]")
    return ResourcePoolProcessExecutor(
        max_resources=max_resources,
        max_workers=max_workers,
        max_tasks_per_child=max_tasks_per_child,
        initializer=process_init,
        initargs=(args,),
    )
