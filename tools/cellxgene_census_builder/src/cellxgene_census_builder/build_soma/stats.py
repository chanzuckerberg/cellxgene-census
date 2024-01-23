from typing import Union

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sparse

from .globals import CENSUS_OBS_TABLE_SPEC, CENSUS_VAR_TABLE_SPEC


def get_obs_stats(
    raw_X: Union[sparse.csr_matrix, sparse.csc_matrix],
) -> pd.DataFrame:
    """Compute summary stats for obs axis, and return as a dataframe."""

    if not isinstance(raw_X, sparse.csr_matrix) and not isinstance(raw_X, sparse.csc_matrix):
        raise NotImplementedError(f"get_obs_stats: unsupported type {type(raw_X)}")

    raw_sum = raw_X.sum(axis=1, dtype=np.float64).A1
    nnz = raw_X.getnnz(axis=1)
    with np.errstate(divide="ignore"):
        raw_mean_nnz = raw_sum / nnz
    raw_mean_nnz[~np.isfinite(raw_mean_nnz)] = 0.0
    raw_variance_nnz = _var(raw_X, axis=1, ddof=1)

    return pd.DataFrame(
        data={
            "raw_sum": raw_sum.astype(CENSUS_OBS_TABLE_SPEC.field("raw_sum").to_pandas_dtype()),
            "nnz": nnz.astype(CENSUS_OBS_TABLE_SPEC.field("nnz").to_pandas_dtype()),
            "raw_mean_nnz": raw_mean_nnz.astype(CENSUS_OBS_TABLE_SPEC.field("raw_mean_nnz").to_pandas_dtype()),
            "raw_variance_nnz": raw_variance_nnz.astype(
                CENSUS_OBS_TABLE_SPEC.field("raw_variance_nnz").to_pandas_dtype()
            ),
            "n_measured_vars": -1,  # placeholder - actual stat calculated from presence matrix
        }
    )


def get_var_stats(
    raw_X: Union[sparse.csr_matrix, sparse.csc_matrix, npt.NDArray[np.float32]],
) -> pd.DataFrame:
    if isinstance(raw_X, sparse.csr_matrix) or isinstance(raw_X, sparse.csc_matrix):
        nnz = raw_X.getnnz(axis=0)

    elif isinstance(raw_X, np.ndarray):
        nnz = np.count_nonzero(raw_X, axis=0)

    else:
        raise NotImplementedError(f"get_var_stats: unsupported array type {type(raw_X)}")

    return pd.DataFrame(
        data={
            "nnz": nnz.astype(CENSUS_VAR_TABLE_SPEC.field("nnz").to_pandas_dtype()),
        }
    )


@numba.jit(
    numba.float64(numba.types.Array(numba.float32, 1, "C", readonly=True), numba.int64),
    nogil=True,
    nopython=True,
)  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _var_ndarray(data: npt.NDArray[np.float32], ddof: int) -> float:
    """
    Return variance of an ndarray. Computed as variance of shifted distribution,
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    n = len(data)
    if n < 2:
        return 0.0
    K = data[0]
    Ex = Ex2 = 0.0
    n = len(data)
    for i in range(n):
        x = data[i]
        Ex += x - K
        Ex2 += (x - K) ** 2

    variance = (Ex2 - Ex**2 / n) / (n - ddof)
    return variance


@numba.jit(
    [
        numba.void(
            numba.types.Array(numba.float32, 1, "C", readonly=True),
            numba.types.Array(numba.int32, 1, "C", readonly=True),
            numba.int64,
            numba.float64[:],
        ),
        numba.void(
            numba.types.Array(numba.float32, 1, "C", readonly=True),
            numba.types.Array(numba.int64, 1, "C", readonly=True),
            numba.int64,
            numba.float64[:],
        ),
    ],
    nopython=True,
    nogil=True,
)  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _var_matrix(
    data: npt.NDArray[np.float32],
    indptr: npt.NDArray[np.int32],
    ddof: int,
    out: npt.NDArray[np.float64],
) -> None:
    n_elem = len(indptr) - 1
    for i in range(n_elem):
        out[i] = _var_ndarray(data[indptr[i] : indptr[i + 1]], ddof)


def _var(
    matrix: Union[sparse.csr_matrix, sparse.csc_matrix],
    axis: int = 0,
    ddof: int = 1,
) -> npt.NDArray[np.float64]:
    if axis == 0:
        n_elem, axis_len = matrix.shape
        matrix = matrix.tocsc()
    else:
        axis_len, n_elem = matrix.shape
        matrix = matrix.tocsr()

    out = np.empty((axis_len,), dtype=np.float64)
    _var_matrix(matrix.data, matrix.indptr, ddof, out)
    return out
