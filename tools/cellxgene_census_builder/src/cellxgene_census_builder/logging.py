import logging
import math
import pathlib
import sys
from typing import List, Tuple

from .build_state import CensusBuildArgs


def logging_init(args: CensusBuildArgs) -> None:
    """
    Configure the logger.
    """
    level = logging.DEBUG if args.config.verbose > 1 else logging.INFO if args.config.verbose == 1 else logging.WARNING
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    # Create logging directory if configured appropriately
    if args.config.log_dir and args.config.log_file:
        logs_dir = pathlib.PosixPath(args.working_dir) / pathlib.PosixPath(args.config.log_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        logs_file = logs_dir / args.config.log_file
        handlers.insert(0, logging.FileHandler(logs_file))

    logging.basicConfig(
        format="%(asctime)s %(process)-7s %(levelname)-8s %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    logging.captureWarnings(True)


def _hr_multibyte_unit(n_bytes: int, unit_base: int, unit_size_names: Tuple[str, ...]) -> str:
    """Private. Convert number of bytes into a human-readable multi-byte unit string."""
    if n_bytes == 0:
        return "0B"

    unit = int(math.floor(math.log(n_bytes, unit_base)))
    n_units = round(n_bytes / math.pow(unit_base, unit))
    return f"{n_units}{unit_size_names[unit]}"


def hr_binary_unit(n_bytes: int) -> str:
    return _hr_multibyte_unit(n_bytes, 1024, ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"))


def hr_decimal_unit(n_bytes: int) -> str:
    """Convert number of bytes into a human-readable decimal (power of 1000) multi-byte unit string."""
    return _hr_multibyte_unit(n_bytes, 1000, ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"))
