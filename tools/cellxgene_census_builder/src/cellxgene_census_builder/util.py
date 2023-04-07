import logging
import multiprocessing
import platform
import re
import urllib.parse

from .build_state import CensusBuildArgs
from .logging import logging_init


def urljoin(base: str, url: str) -> str:
    """
    like urllib.parse.urljoin, but doesn't get confused by S3://
    """
    p_url = urllib.parse.urlparse(url)
    if p_url.netloc:
        return url

    p_base = urllib.parse.urlparse(base)
    path = urllib.parse.urljoin(p_base.path, p_url.path)
    parts = [p_base.scheme, p_base.netloc, path, p_url.params, p_url.query, p_url.fragment]
    return urllib.parse.urlunparse(parts)


def urlcat(base: str, *paths: str) -> str:
    """
    Concat one or more paths, separated with '/'. Similar to urllib.parse.urljoin,
    but doesn't get confused by S3:// and other "non-standard" protocols (treats
    them as if they are same as http: or file:)

    Similar to urllib.parse.urljoin except it takes an iterator, and
    assumes the container_uri is a 'directory'/container, ie, ends in '/'.
    """

    url = base
    for p in paths:
        url = url if url.endswith("/") else url + "/"
        url = urljoin(url, p)
    return url


def process_init(args: CensusBuildArgs) -> None:
    """
    Called on every process start to configure global package/module behavior.
    """
    if multiprocessing.get_start_method(True) != "spawn":
        multiprocessing.set_start_method("spawn", True)

    logging_init(args)


class ProcessResourceGetter:
    """
    Access to process resource state, primary for diagnostic/debugging purposes. Currently
    provides current and high water mark for:
    * thread count
    * mmaps

    Linux-only at the moment.
    """

    # historical maxima
    max_thread_count = -1
    max_map_count = -1

    @property
    def thread_count(self) -> int:
        """Return the thread count for the current process. Retain the historical maximum."""
        if platform.system() != "Linux":
            return -1

        with open("/proc/self/status") as f:
            status = f.read()
            thread_count = int(re.split(".*\nThreads:\t(\d+)\n.*", status)[1])
            self.max_thread_count = max(thread_count, self.max_thread_count)
        return thread_count

    @property
    def map_count(self) -> int:
        """Return the memory map count for the current process. Retain the historical maximum."""
        if platform.system() != "Linux":
            return -1

        with open("/proc/self/maps") as f:
            maps = f.read()
            map_count = maps.count("\n")
            self.max_map_count = max(map_count, self.max_map_count)
        return map_count


_resouce_getter = ProcessResourceGetter()


def log_process_resource_status(preface: str = "Resource use:") -> None:
    """Print current and historical max of thread and (memory) map counts"""
    if platform.system() == "Linux":
        logging.debug(
            f"{preface} threads: {_resouce_getter.thread_count} "
            f"[max: {_resouce_getter.max_thread_count}], "
            f"maps: {_resouce_getter.map_count} "
            f"[max: {_resouce_getter.max_map_count}]"
        )
