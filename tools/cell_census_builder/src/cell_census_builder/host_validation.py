import logging
import os
import sys
from typing import Optional

import psutil

from cell_census_builder.logging import hr_multibyte_unit, setup_logging

"""Minimum physical RAM"""
MIN_RAM = 512 * 1024**3  # 512GiB

"""Minimum virtual memory/swap"""
MIN_SWAP = 2 * 1024**4  # 2TiB

"""Minimum free disk space"""
MIN_FREE_DISK_SPACE = 1 * 1024**4  # 1 TiB


def check_os() -> None:
    """
    Check that we run on Posix (Linux, MacOS), as we rely on
    Posix semantics for a few things.
    """
    assert psutil.POSIX


def check_memory() -> None:
    """
    Check for sufficient physical and virtual memory.
    """
    svmem = psutil.virtual_memory()
    logging.debug(f"Host: {hr_multibyte_unit(svmem.total)} memory found")
    assert svmem.total >= MIN_RAM, f"Insufficient memory (found {svmem.total}, require {MIN_RAM})"

    svswap = psutil.swap_memory()
    logging.debug(f"Host: {hr_multibyte_unit(svswap.total)} swap found")
    assert svswap.total >= MIN_SWAP, f"Insufficient swap space (found {svswap.total}, require {MIN_SWAP})"


def check_free_disk(working_dir: Optional[str] = ".") -> None:
    """
    Check for sufficient free disk space.
    """
    skdiskusage = psutil.disk_usage(working_dir)
    logging.debug(f"Host: {hr_multibyte_unit(skdiskusage.free)} free disk space found")
    assert (
        skdiskusage.free >= MIN_FREE_DISK_SPACE
    ), f"Insufficient free disk space (found {skdiskusage.free}, require {MIN_FREE_DISK_SPACE})"


def run_all_checks() -> int:
    """
    Run all host validation checks.  Returns zero or raises an exception.
    """
    check_os()
    check_memory()
    check_free_disk(os.getcwd())  # assumed working directory is CWD
    logging.info("Host validation success")
    return 0


# Process MUST return zero on success (all good) or non-zero on a
# host which does not validate.
if __name__ == "__main__":
    setup_logging(verbose=1)
    sys.exit(run_all_checks())
