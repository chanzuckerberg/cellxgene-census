import multiprocessing
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
