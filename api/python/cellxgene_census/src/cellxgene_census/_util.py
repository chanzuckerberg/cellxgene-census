import urllib.parse

import tiledbsoma as soma

USER_AGENT_ENVVAR = "CELLXGENE_CENSUS_USERAGENT"
"""Environment variable used to add more information into the user-agent."""


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


def _user_agent() -> str:
    import os

    import cellxgene_census

    if env_specifier := os.environ.get(USER_AGENT_ENVVAR, None):
        return f"cellxgene-census-python/{cellxgene_census.__version__} {env_specifier}"
    else:
        return f"cellxgene-census-python/{cellxgene_census.__version__}"
