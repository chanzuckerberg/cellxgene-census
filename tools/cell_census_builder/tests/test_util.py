import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix, triu

from tools.cell_census_builder.util import array_chunker, is_nonnegative_integral, uricat


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

    X = csr_matrix([[1, 2, 3], [4, 5, 6]])
    assert is_nonnegative_integral(X)

    X = csr_matrix([[-1, 2, 3], [4, 5, 6]])
    assert not is_nonnegative_integral(X)

    X = csr_matrix([[1.2, 0, 3], [4, 5, 6]])
    assert not is_nonnegative_integral(X)

    X = csr_matrix([0, 0, 0])
    assert is_nonnegative_integral(X)

    X = np.empty(0, dtype=np.float32)  # Empty ndarray
    assert is_nonnegative_integral(X)

    X = csr_matrix((0, 0))  # Empty sparse matrix
    assert is_nonnegative_integral(X)


def test_array_chunker() -> None:
    # Case 1: dense matrix (np.ndarray)
    X = np.ones(1200).reshape(30, 40)
    # If nnz_chunk_size is less than the number of cols, the number of cols is used (40 in this example)
    chunked = list(array_chunker(X, nnz_chunk_size=10))
    assert len(chunked) == 30
    for i, s in enumerate(chunked):
        assert isinstance(s, coo_matrix)
        assert s.nnz == 40
        assert s.shape == (30, 40)
        # The i-th row of the matrix should have 40 nonzeros (which implies the rest are zeros)
        csr = s.tocsr()
        assert csr.getrow(i).nnz == 40

    # If nnz_chunk_size is less than
    chunked = list(array_chunker(X, nnz_chunk_size=600))
    assert len(chunked) == 2
    for s in chunked:
        assert isinstance(s, coo_matrix)
        assert s.nnz == 600
        assert s.shape == (30, 40)

    chunked = list(array_chunker(X, nnz_chunk_size=2400))
    assert len(chunked) == 1

    # Case 2: compressed row sparse matrix (csr_matrix)
    # we'll use an upper triangular matrix with all ones (avg 5 nnz per row)
    # [
    #     [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    #     [0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    #     [0., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
    #     [0., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
    #     [0., 0., 0., 0., 1., 1., 1., 1., 1., 1.],
    #     [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
    #     [0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
    #     [0., 0., 0., 0., 0., 0., 0., 1., 1., 1.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    # ]

    X = triu(np.ones(100).reshape(10, 10)).tocsr()
    # In this case, chunks will be 2 rows x 10 column (since on average each row contains 5 nonzeros)
    chunked = list(array_chunker(X, nnz_chunk_size=10))
    assert len(chunked) == 5
    assert chunked[0].nnz == 19
    assert chunked[1].nnz == 15
    assert chunked[2].nnz == 11
    assert chunked[3].nnz == 7
    assert chunked[4].nnz == 3

    # Case 3: compressed column sparse matrix (csc_matrix)
    # We'll use the same example as for csr, but note that chunking is done by column and not by row.

    X = triu(np.ones(100).reshape(10, 10)).tocsc()
    # In this case, chunks will be 2 rows x 10 column (since on average each row contains 5 nonzeros)
    chunked = list(array_chunker(X, nnz_chunk_size=10))
    assert len(chunked) == 5
    assert chunked[0].nnz == 3
    assert chunked[1].nnz == 7
    assert chunked[2].nnz == 11
    assert chunked[3].nnz == 15
    assert chunked[4].nnz == 19

    # Other formats are rejected by the method
    X = triu(np.ones(100).reshape(10, 10)).tolil()
    with pytest.raises(NotImplementedError):
        list(array_chunker(X))


def test_uricat() -> None:
    assert uricat("path", "to", "somewhere") == "path/to/somewhere"
    assert uricat("path/", "to/", "somewhere") == "path/to/somewhere"
    assert uricat("path/", "to/", "somewhere/") == "path/to/somewhere/"
    assert uricat("file:///path/to", "somewhere") == "file:///path/to/somewhere"
    assert uricat("file:///path/to/", "somewhere") == "file:///path/to/somewhere"
    assert uricat("file:///path/to", "somewhere") == "file:///path/to/somewhere"
    assert uricat("file:///path/to/", "/absolute") == "file:///absolute"
    assert uricat("file://path/to", "file://somewhere") == "file://somewhere"
    assert uricat("file:///path/to", "file://somewhere") == "file://somewhere"
    assert uricat("file:///path/to", "file:///somewhere") == "file:///somewhere"
