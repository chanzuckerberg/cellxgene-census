import argparse
import concurrent.futures
import logging
import multiprocessing
import os
from typing import Optional, cast


def cpu_count() -> int:
    """Sign, os.cpu_count() returns None if "undetermined" number of CPUs"""
    cpu_count = os.cpu_count()
    if os.cpu_count() is None:
        return 1
    return cast(int, cpu_count)


def process_initializer(verbose: int = 0) -> None:
    level = logging.DEBUG if verbose > 1 else logging.INFO if verbose == 1 else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s %(process)-7s %(levelname)-8s %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.captureWarnings(True)


def create_process_pool_executor(
    args: argparse.Namespace, max_workers: Optional[int] = None
) -> concurrent.futures.ProcessPoolExecutor:
    # We rely on the pool configuration being correct. Failure to do this will
    # lead to strange errors on some OS (eg., Linux defaults to fork). Rather
    # than chase those bugs, assert correct configuration.
    assert multiprocessing.get_start_method(True) == "spawn"

    return concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers if max_workers is None else max_workers,
        initializer=process_initializer,
        initargs=(args.verbose,),
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
