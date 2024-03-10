import logging
import multiprocessing
import time

import dask
import dask.distributed

from ..build_state import CensusBuildArgs
from ..process_init import process_init
from ..util import cpu_count

logger = logging.getLogger(__name__)


def _mp_config_checks() -> bool:
    # We rely on the pool configuration being correct. Failure to do this will
    # lead to strange errors on some OS (eg., Linux defaults to fork). Rather
    # than chase those bugs, assert correct configuration.
    assert multiprocessing.get_start_method(True) == "spawn"

    return True


# def _hard_process_cap(args: CensusBuildArgs, n_proc: int) -> int:
#     """Enforce the configured worker process limit.

#     NOTE: logic below only enforces this limit in cases using the default worker count,
#     as there are special cases where we want higher limits, due to special knowledge that we
#     will not be subject to the default resource constraints (e.g., VM map usage by SOMA).
#     """
#     return min(int(args.config.max_worker_processes), n_proc)


# def _default_worker_process_count(args: CensusBuildArgs) -> int:
#     """Return the default worker process count, subject to configured limit."""
#     return _hard_process_cap(args, cpu_count())


# def create_process_pool_executor(
#     args: CensusBuildArgs,
#     max_workers: int | None = None,
#     max_tasks_per_child: int | None = None,
# ) -> ProcessPoolExecutor:
#     assert _mp_config_checks()
#     if max_workers is None:
#         max_workers = _default_worker_process_count(args)
#     max_workers = max(1, max_workers)
#     logger.debug(f"create_process_pool_executor [max_workers={max_workers}, max_tasks_per_child={max_tasks_per_child}]")
#     return ProcessPoolExecutor(
#         max_workers=max_workers, initializer=process_init, initargs=(args,), max_tasks_per_child=max_tasks_per_child
#     )


# def log_on_broken_process_pool(ppe: ProcessPoolExecutor) -> None:
#     """There are a number of conditions where the Process Pool can be broken,
#     such that it will hang in a shutdown. This will cause the context __exit__
#     to hang indefinitely, as it calls ProcessPoolExecutor.shutdown with
#     `wait=True`.

#     An example condition which can cause a deadlock is an OOM, where a the
#     repear kills a process.

#     This routine is used to detect the condition and log the error, so a
#     human has a chance of detecting/diagnosing.

#     Caution: uses ProcessPoolExecutor internal API, as this state is not
#     otherwise visible.
#     """
#     if ppe._broken:
#         logger.critical(f"Process pool broken and may fail or hang: {ppe._broken}")

#     return


class SetupDaskWorker(dask.distributed.WorkerPlugin):  # type: ignore[misc]
    """Pass config to all workers."""

    def __init__(self, args: CensusBuildArgs):
        self.args = args

    def setup(self, worker: dask.distributed.Worker) -> None:
        process_init(self.args)


def create_dask_client(
    args: CensusBuildArgs,
    *,
    n_workers: int | None = None,
    threads_per_worker: int | None = None,
    memory_limit: str | float | int | None = "auto",
) -> dask.distributed.Client:
    """Create and return a Dask client."""
    # create a new client
    assert _mp_config_checks()

    n_workers = max(1, n_workers or cpu_count())
    dask.config.set(
        {
            "distributed.scheduler.worker-ttl": "24 hours",  # some tasks are very long-lived, e.g., consolidation
        }
    )

    client = dask.distributed.Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        dashboard_address=":8787" if args.config.dashboard else None,
    )
    client.register_plugin(SetupDaskWorker(args))
    logger.info(f"Dask client created: {client}")
    logger.info(f"Dask client using cluster: {client.cluster}")
    if args.config.dashboard:
        logger.info(f"Dashboard link: {client.dashboard_link}")

    # Release GIL, allowing the scheduler thread to run. Without this, the scheduler and
    # worker startup will race, occasionally causing a heartbeat error to be logged on startup.
    # The only side-effect is to keep logs cleaner.
    time.sleep(0.1)

    return client


def shutdown_dask_cluster(client: dask.distributed.Client) -> None:
    """Clean-ish shutdown, designed to prevent hangs and error messages in log."""
    client.retire_workers()
    time.sleep(1)
    client.shutdown()
    logger.info("Dask cluster shut down")
