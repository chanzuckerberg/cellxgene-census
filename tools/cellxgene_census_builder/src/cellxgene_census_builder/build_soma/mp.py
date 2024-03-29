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
    try:
        client.retire_workers()
    except TimeoutError:
        # Quiet Tornado errors
        pass

    time.sleep(1)
    try:
        client.shutdown()
    except TimeoutError:
        # Quiet Tornado errors
        pass

    logger.info("Dask cluster shut down")
