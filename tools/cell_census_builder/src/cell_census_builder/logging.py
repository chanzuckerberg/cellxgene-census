import logging
import math


def setup_logging(verbose: int = 0) -> None:
    """
    Configure the logger
    """
    level = logging.DEBUG if verbose > 1 else logging.INFO if verbose == 1 else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s %(process)-7s %(levelname)-8s %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.captureWarnings(True)


def hr_multibyte_unit(n_bytes: int) -> str:
    """Convert number of bytes into a human-readable binary (power of 1024) multi-byte unit string."""
    if n_bytes == 0:
        return "0B"

    unit_size_name = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")
    unit = int(math.floor(math.log(n_bytes, 1024)))
    n_units = round(n_bytes / math.pow(1024, unit))
    return f"{n_units}{unit_size_name[unit]}"
