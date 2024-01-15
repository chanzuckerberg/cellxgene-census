import logging
import os
import sys
from typing import Union

import psutil

from .build_state import CensusBuildArgs, CensusBuildConfig
from .logging import hr_binary_unit, hr_decimal_unit

logger = logging.getLogger(__name__)


def _check(condition: bool, message: str) -> bool:
    """Like assert, but logs"""
    if not condition:
        logger.critical(message)
    return condition


def check_os() -> bool:
    """
    Check that we run on Posix (Linux, MacOS), as we rely on
    Posix semantics for a few things.
    """
    return _check(os.name == "posix" and psutil.POSIX, "Census builder requires Posix OS")


def check_physical_memory(min_physical_memory: int) -> bool:
    """
    Check for sufficient physical and virtual memory.
    """
    svmem = psutil.virtual_memory()
    logger.debug(f"Host: {hr_binary_unit(svmem.total)} memory found")
    return _check(
        svmem.total >= min_physical_memory,
        f"Insufficient memory (found {hr_binary_unit(svmem.total)}, " f"require {hr_binary_unit(min_physical_memory)})",
    )


def check_swap_memory(min_swap_memory: int) -> bool:
    """
    Check for sufficient physical and virtual memory.
    """
    svswap = psutil.swap_memory()
    logger.debug(f"Host: {hr_binary_unit(svswap.total)} swap found")
    return _check(
        svswap.total >= min_swap_memory,
        f"Insufficient swap space (found {hr_binary_unit(svswap.total)}, "
        f"require {hr_binary_unit(min_swap_memory)})",
    )


def check_free_disk(working_dir: Union[str, os.PathLike[str]], min_free_disk_space: int) -> bool:
    """
    Check for sufficient free disk space.
    """
    working_dir_fspath = working_dir.__fspath__() if isinstance(working_dir, os.PathLike) else working_dir
    skdiskusage = psutil.disk_usage(working_dir_fspath)
    logger.debug(f"Host: {hr_decimal_unit(skdiskusage.free)} free disk space found")
    return _check(
        skdiskusage.free >= min_free_disk_space,
        f"Insufficient free disk space (found {hr_decimal_unit(skdiskusage.free)}, "
        f"require {hr_decimal_unit(min_free_disk_space)})",
    )


def check_host(args: CensusBuildArgs) -> bool:
    """Verify all host requirments. Return True if OK, False if conditions not met"""
    if args.config.host_validation_disable:
        return True

    return (
        check_os()
        and check_physical_memory(args.config.host_validation_min_physical_memory)
        and check_swap_memory(args.config.host_validation_min_swap_memory)
        and check_free_disk(args.working_dir, args.config.host_validation_min_free_disk_space)
    )


# Return zero on success (all good) or non-zero on a
# host which does not validate.
if __name__ == "__main__":
    """For CLI testing"""

    config = CensusBuildConfig()

    def main() -> int:
        if not (
            check_os()
            and check_physical_memory(config.host_validation_min_physical_memory)
            and check_swap_memory(config.host_validation_min_swap_memory)
            and check_free_disk(os.getcwd(), config.host_validation_min_free_disk_space)
        ):  # assumed working directory is CWD
            print("Host validation FAILURE")
            return 1

        print("Host validation success")
        return 0

    sys.exit(main())
