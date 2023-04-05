from cellxgene_census._util import _uri_join


def test_uri_join() -> None:
    assert _uri_join("https://foo/", "bar") == "https://foo/bar"
    assert _uri_join("https://foo/a", "bar") == "https://foo/bar"
    assert _uri_join("https://foo/a/", "bar") == "https://foo/a/bar"
    assert _uri_join("https://foo/", "a/b") == "https://foo/a/b"
    assert _uri_join("https://foo/", "a/b/") == "https://foo/a/b/"

    assert _uri_join("https://foo/?a=b#99", "a?b=c#1") == "https://foo/a?b=c#1"

    assert _uri_join("http://foo/bar/", "a") == "http://foo/bar/a"
    assert _uri_join("https://foo/bar/", "a") == "https://foo/bar/a"
    assert _uri_join("s3://foo/bar/", "a") == "s3://foo/bar/a"
    assert _uri_join("foo/bar", "a") == "foo/a"
    assert _uri_join("/foo/bar", "a") == "/foo/a"
    assert _uri_join("file://foo/bar", "a") == "file://foo/a"
    assert _uri_join("file:///foo/bar", "a") == "file:///foo/a"

    assert _uri_join("https://foo/bar", "https://a/b") == "https://a/b"
