import logging
import multiprocessing
import os

from .build_state import CensusBuildArgs
from .logging import logging_init
from .util import cpu_count

logger = logging.getLogger(__name__)


def env_var_init() -> None:
    """Set environment variables as needed by dependencies, etc.

    This controls thread allocation for worker (child) processes. It is executed too
    late to influence __init__ time thread pool allocations for the main process.
    """
    # Each of these control thread-pool allocation for commonly used packages that
    # may be pulled into our environment, and which have import-time pool allocation.
    # Most do import time thread pool allocation equal to host CPU count, which can
    # result in excessive unused thread pools on high CPU machines.
    #
    # Where we are confident we have no performance dependency related to their concurrency,
    # set their pool size to "1". Otherwise set to something useful.
    #
    # OMP_NUM_THREADS: OpenMp,
    # OPENBLAS_NUM_THREADS: OpenBLAS,
    # MKL_NUM_THREADS: Intel MKL,
    # VECLIB_MAXIMUM_THREADS: Accelerate,
    # NUMEXPR_NUM_THREADS: NumExpr

    if "NUMEXPR_MAX_THREADS" not in os.environ:
        # ref: https://numexpr.readthedocs.io/en/latest/user_guide.html#threadpool-configuration
        # In particular, the docs state that >8 threads is not helpful except in extreme circumstances.
        val = str(min(8, max(1, cpu_count() // 2)))
        os.environ["NUMEXPR_MAX_THREADS"] = val
        logger.debug(f'Setting NUMEXPR_MAX_THREADS environment variable to "{val}"')

    for env_name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        if env_name not in os.environ:
            logger.debug(f'Setting {env_name} environment variable to "1"')
            os.environ[env_name] = "1"


def process_init(args: CensusBuildArgs) -> None:
    """Called on every process start to configure global package/module behavior."""
    logging_init(args)

    if multiprocessing.get_start_method(True) != "spawn":
        multiprocessing.set_start_method("spawn", True)

    env_var_init()
