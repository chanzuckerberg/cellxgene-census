import numpy as np
from cellxgene_census_builder.build_soma.util import is_nonnegative_integral
from cellxgene_census_builder.util import urlcat, urljoin
from scipy.sparse import csr_matrix


def test_is_nonnegative_integral() -> None:
    X = np.array([1, 2, 3], dtype=np.float32)
    assert is_nonnegative_integral(X)

    X = np.array([-1, 2, 3], dtype=np.float32)
    assert not is_nonnegative_integral(X)

    X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    assert is_nonnegative_integral(X)

    X = np.array([[-1, 2, 3], [4, 5, 6]], dtype=np.float32)
    assert not is_nonnegative_integral(X)

    X = np.array([[1.2, 0, 3], [4, 5, 6]], dtype=np.float32)
    assert not is_nonnegative_integral(X)

    X = np.zeros((3, 4), dtype=np.float32)
    assert is_nonnegative_integral(X)

    X = csr_matrix([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    assert is_nonnegative_integral(X)

    X = csr_matrix([[-1, 2, 3], [4, 5, 6]], dtype=np.float32)
    assert not is_nonnegative_integral(X)

    X = csr_matrix([[1.2, 0, 3], [4, 5, 6]], dtype=np.float32)
    assert not is_nonnegative_integral(X)

    X = csr_matrix([0, 0, 0], dtype=np.float32)
    assert is_nonnegative_integral(X)

    X = np.empty(0, dtype=np.float32)  # Empty ndarray
    assert is_nonnegative_integral(X)

    X = csr_matrix((0, 0), dtype=np.float32)  # Empty sparse matrix
    assert is_nonnegative_integral(X)


def test_urljoin() -> None:
    assert urljoin("path", "to") == "to"
    assert urljoin("path/", "to") == "path/to"
    assert urljoin("path/", "to/") == "path/to/"
    assert urljoin("file:///path/to", "somewhere") == "file:///path/somewhere"
    assert urljoin("file:///path/to/", "somewhere") == "file:///path/to/somewhere"
    assert urljoin("file:///path/to", "somewhere") == "file:///path/somewhere"
    assert urljoin("file:///path/to/", "/absolute") == "file:///absolute"
    assert urljoin("file://path/to", "file://somewhere") == "file://somewhere"
    assert urljoin("file:///path/to", "file://somewhere") == "file://somewhere"
    assert urljoin("file:///path/to", "file:///somewhere") == "file:///somewhere"
    assert urljoin("s3://foo", "bar") == "s3://foo/bar"
    assert urljoin("s3://foo/", "bar") == "s3://foo/bar"
    assert urljoin("s3://foo", "bar/") == "s3://foo/bar/"


def test_urlcat() -> None:
    assert urlcat("path", "to", "somewhere") == "path/to/somewhere"
    assert urlcat("path/", "to/", "somewhere") == "path/to/somewhere"
    assert urlcat("path/", "to/", "somewhere/") == "path/to/somewhere/"
    assert urlcat("file:///path/to", "somewhere") == "file:///path/to/somewhere"
    assert urlcat("file:///path/to/", "somewhere") == "file:///path/to/somewhere"
    assert urlcat("file:///path/to", "somewhere") == "file:///path/to/somewhere"
    assert urlcat("file:///path/to/", "/absolute") == "file:///absolute"
    assert urlcat("file://path/to", "file://somewhere") == "file://somewhere"
    assert urlcat("file:///path/to", "file://somewhere") == "file://somewhere"
    assert urlcat("file:///path/to", "file:///somewhere") == "file:///somewhere"
    assert urlcat("s3://foo", "bar", "baz") == "s3://foo/bar/baz"
    assert urlcat("s3://foo", "bar/", "baz") == "s3://foo/bar/baz"
    assert urlcat("s3://foo", "bar/", "baz/") == "s3://foo/bar/baz/"
