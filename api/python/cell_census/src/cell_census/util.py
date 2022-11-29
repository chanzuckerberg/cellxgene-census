import urllib.parse


def uri_join(base: str, url: str) -> str:
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
