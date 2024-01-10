import os
import time
from typing import Any, Iterator, Optional, Union

import numpy as np
import numpy.typing as npt
import requests
import urllib3
from scipy import sparse


def array_chunker(
    arr: Union[npt.NDArray[Any], sparse.spmatrix],
    nnz_chunk_size: Optional[int] = 256 * 1024**2,  # goal (~2.4GiB for a 32-bit COO)
) -> Iterator[sparse.coo_matrix]:
    """
    Return the array as multiple chunks, each a coo_matrix.
    The slicing is always done by row (for ndarray and csr_matrix) or by column (for csc_matrix),
    and will never split a row (or column) into two separate slices.

    Args:
        arr:
            The array to slice (either a numpy ndarray, a scipy.sparse csr_matrix or csc_matrix).
        nnz_chunk_size:
            Approximate number of elements in each chunk.

    Returns:
        An iterator containing the chunks.

    Raises:
        NotImplementedError: If the matrix type is not supported.
    """

    if isinstance(arr, sparse.csr_matrix) or isinstance(arr, sparse.csr_array):
        avg_nnz_per_row = arr.nnz // arr.shape[0]
        row_chunk_size = max(1, round(nnz_chunk_size / avg_nnz_per_row))
        for row_idx in range(0, arr.shape[0], row_chunk_size):
            slc = arr[row_idx : row_idx + row_chunk_size, :].tocoo()
            slc.resize(arr.shape)
            slc.row += row_idx
            yield slc
        return

    if isinstance(arr, sparse.csc_matrix) or isinstance(arr, sparse.csc_array):
        avg_nnz_per_col = arr.nnz // arr.shape[1]
        col_chunk_size = max(1, round(nnz_chunk_size / avg_nnz_per_col))
        for col_idx in range(0, arr.shape[1], col_chunk_size):
            slc = arr[:, col_idx : col_idx + col_chunk_size].tocoo()
            slc.resize(arr.shape)
            slc.col += col_idx
            yield slc
        return

    if isinstance(arr, np.ndarray):
        row_chunk_size = max(1, nnz_chunk_size // arr.shape[1])  # type: ignore
        for row_idx in range(0, arr.shape[0], row_chunk_size):
            slc = sparse.coo_matrix(arr[row_idx : row_idx + row_chunk_size, :])
            slc.resize(arr.shape)
            slc.row += row_idx
            yield slc
        return

    raise NotImplementedError("array_chunker: unsupported array type")


def fetch_json(url: str, delay_secs: float = 0.0) -> object:
    s = requests.Session()
    retries = urllib3.util.Retry(
        backoff_factor=1,  # i.e., sleep for [0s, 2s, 4s, 8s, ...]
        status_forcelist=[500, 502, 503, 504],  # 5XX can occur on CDN failures, so retry
    )
    s.mount("https://", requests.adapters.HTTPAdapter(max_retries=retries))
    response = s.get(url)
    response.raise_for_status()
    time.sleep(delay_secs)
    return response.json()


def is_nonnegative_integral(X: Union[npt.NDArray[np.floating[Any]], sparse.spmatrix]) -> bool:
    """
    Return true if the matrix/array contains only positive integral values,
    False otherwise.
    """
    data = X if isinstance(X, np.ndarray) else X.data

    if np.signbit(data).any():
        return False
    elif np.any(~np.equal(np.mod(data, 1), 0)):
        return False
    else:
        return True


def get_git_commit_sha() -> str:
    """
    Returns the git commit SHA for the current repo
    """
    # Try to get the git commit SHA from the COMMIT_SHA env variable
    commit_sha_var = os.getenv("COMMIT_SHA")
    if commit_sha_var is not None:
        return commit_sha_var

    import git  # Scoped import - this requires the git executable to exist on the machine

    # work around https://github.com/gitpython-developers/GitPython/issues/1349
    # by explicitly referencing git.repo.base.Repo instead of git.Repo
    repo = git.repo.base.Repo(search_parent_directories=True)
    hexsha: str = repo.head.object.hexsha
    return hexsha


def is_git_repo_dirty() -> bool:
    """
    Returns True if the git repo is dirty, i.e. there are uncommitted changes
    """
    import git  # Scoped import - this requires the git executable to exist on the machine

    # work around https://github.com/gitpython-developers/GitPython/issues/1349
    # by explicitly referencing git.repo.base.Repo instead of git.Repo
    repo = git.repo.base.Repo(search_parent_directories=True)
    is_dirty: bool = repo.is_dirty()
    return is_dirty
