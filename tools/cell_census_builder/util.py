import urllib.parse
from typing import Any, Union
from warnings import warn

import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
from scipy import sparse


def array_chunker(arr: Union[npt.NDArray[Any], sparse.spmatrix]) -> sparse.coo_matrix:
    """
    Return the array as multiple chunks, each a coo_matrix.
    """
    nnz_chunk_size = 256 * 1024**2  # goal (~2.4GiB for a 32-bit COO)

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
        row_chunk_size = max(1, nnz_chunk_size // arr.shape[1])
        for row_idx in range(0, arr.shape[0], row_chunk_size):
            slc = sparse.coo_matrix(arr[row_idx : row_idx + row_chunk_size, :])
            slc.resize(arr.shape)
            slc.row += row_idx
            yield slc
        return

    raise NotImplementedError("array_chunker: unsupported array type")


def uricat(container_uri: str, *paths: str) -> str:
    """
    Concat one or more paths, separated with '/'

    Similar to urllib.parse.urljoin except it takes an iterator, and
    assumes the container_uri is a 'directory'/container, ie, ends in '/'.
    """

    uri = container_uri
    for p in paths:
        uri = uri if uri.endswith("/") else uri + "/"
        uri = urllib.parse.urljoin(uri, p)
    return uri


def fetch_json(url: str) -> object:
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def is_positive_integral(X: Union[npt.NDArray[np.floating[Any]], sparse.spmatrix]) -> bool:
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


def anndata_ordered_bool_issue_853_workaround(df: pd.DataFrame) -> pd.DataFrame:
    # """
    # TileDB-SOMA does not support creating dataframe with categorical / dictionary
    # column types.
    # """
    # copied = False
    # for k in df.keys():
    #     if pd.api.types.is_categorical_dtype(df[k]):
    #         if not copied:
    #             df = df.copy()
    #             copied = True

    #         df[k] = df[k].astype(df[k].cat.categories.dtype)

    # AnnData has a bug (https://github.com/scverse/anndata/issues/853) which will
    # cause Pandas CategoricalDtype `ordered` to be a numpy.bool_, rather than a bool.
    # This causes Arrow to blow up.
    copied = False
    for k in df.keys():
        if pd.api.types.is_categorical_dtype(df[k]) and type(df[k].cat.ordered) == np.bool_:
            if not copied:
                df = df.copy()
                copied = True

            df[k] = df[k].cat.set_categories(df[k].cat.categories, ordered=bool(df[k].cat.ordered))

    return df
