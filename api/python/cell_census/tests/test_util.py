from cell_census.util import uri_join


def test_uri_join() -> None:
    assert uri_join("https://foo/", "bar") == "https://foo/bar"
    assert uri_join("https://foo/a", "bar") == "https://foo/bar"
    assert uri_join("https://foo/a/", "bar") == "https://foo/a/bar"
    assert uri_join("https://foo/", "a/b") == "https://foo/a/b"
    assert uri_join("https://foo/", "a/b/") == "https://foo/a/b/"

    assert uri_join("https://foo/?a=b#99", "a?b=c#1") == "https://foo/a?b=c#1"

    assert uri_join("http://foo/bar/", "a") == "http://foo/bar/a"
    assert uri_join("https://foo/bar/", "a") == "https://foo/bar/a"
    assert uri_join("s3://foo/bar/", "a") == "s3://foo/bar/a"
    assert uri_join("foo/bar", "a") == "foo/a"
    assert uri_join("/foo/bar", "a") == "/foo/a"
    assert uri_join("file://foo/bar", "a") == "file://foo/a"
    assert uri_join("file:///foo/bar", "a") == "file:///foo/a"
