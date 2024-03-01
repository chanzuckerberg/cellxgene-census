from __future__ import annotations

import functools
import logging
import math
import pathlib
import sys
import time
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from .build_state import CensusBuildArgs
from .util import clamp


def logging_init_params(verbose: int, handlers: list[logging.Handler] | None = None) -> None:
    """Configure the logger defaults with explicit config params."""

    def get_level(v: int) -> int:
        levels = [logging.WARNING, logging.INFO, logging.DEBUG]
        return levels[clamp(v, 0, len(levels) - 1)]

    logging.basicConfig(
        format="%(asctime)s %(process)-7s %(levelname)-8s %(message)s",
        level=get_level(verbose),
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    logging.captureWarnings(True)
    logging.getLogger(__package__).setLevel(get_level(verbose + 1))

    # these are super noisy! Enable only at an extra high level of verbosity
    if verbose < 3:
        for pkg in ["h5py", "fsspec"]:
            logging.getLogger(pkg).setLevel(get_level(verbose - 1))
    # and even higher for numba, which spews...
    if verbose < 4:
        for pkg in ["numba"]:
            logging.getLogger(pkg).setLevel(get_level(verbose - 2))


def logging_init(args: CensusBuildArgs) -> None:
    """Configure the logger from CensusBuildArgs, including extra handlers."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    # Create logging directory if configured appropriately
    if args.config.log_dir and args.config.log_file:
        logs_dir = pathlib.PosixPath(args.working_dir) / pathlib.PosixPath(args.config.log_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        logs_file = logs_dir / args.config.log_file
        handlers.insert(0, logging.FileHandler(logs_file))

    logging_init_params(args.config.verbose, handlers)


def _hr_multibyte_unit(n_bytes: int, unit_base: int, unit_size_names: tuple[str, ...]) -> str:
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


P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])


def logit(
    logger: logging.Logger, *, level: int = logging.INFO, msg: str | None = None, timeit: bool = True
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Log decorator factory -- logs entry and exit, with optional timing and user configurable message.

    Args:
        logger:
            A logger instance.
        level
            Log level.
        msg:
            User-specified message. May contain formatting parameters, which will be
            applied as ``msg.format(*args, **kwargs)``
        timeit:
            If true, log processing time.

    Example:
        @logit(logger, msg='{0}')
        def inc(a: int) -> int:
            return a + 1

    """
    exit_log_level = level
    enter_log_level = {logging.NOTSET: logging.NOTSET, logging.INFO: logging.DEBUG}.get(level, logging.DEBUG)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if msg is None:
                _msg = ""
            else:
                _msg = ": " + msg.format(*args, **kwargs)
            logger.log(enter_log_level, f"{func.__name__} [enter]{_msg}")
            t = time.perf_counter()
            result: R = func(*args, **kwargs)
            call_time = "" if not timeit else f", {(time.perf_counter()-t):.2f}s"
            logger.log(exit_log_level, f"{func.__name__} [exit{call_time}]{_msg}")
            return result

        return wrapper

    return decorator
