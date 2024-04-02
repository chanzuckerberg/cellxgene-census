import urllib.parse

import tiledbsoma as soma


def _uri_join(base: str, url: str) -> str:
    """Like urllib.parse.urljoin, but doesn't get confused by s3://."""
    p_url = urllib.parse.urlparse(url)
    if p_url.netloc:
        return url

    p_base = urllib.parse.urlparse(base)
    path = urllib.parse.urljoin(p_base.path, p_url.path)
    parts = [
        p_base.scheme,
        p_base.netloc,
        path,
        p_url.params,
        p_url.query,
        p_url.fragment,
    ]
    return urllib.parse.urlunparse(parts)


def _extract_census_version(census: soma.Collection) -> str:
    """Extract the Census version from the given Census object."""
    try:
        version: str = urllib.parse.urlparse(census.uri).path.split("/")[2]
    except (KeyError, IndexError):
        raise ValueError("Unable to extract Census version.") from None

    return version
