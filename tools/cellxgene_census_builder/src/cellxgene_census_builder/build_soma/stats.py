from typing import Union

import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as sparse

from .globals import CENSUS_OBS_STATS_COLUMNS, CENSUS_VAR_STATS_COLUMNS


def get_obs_stats(
    raw_X: Union[sparse.csr_matrix, sparse.csc_matrix, npt.NDArray[np.float32]],
) -> pd.DataFrame:
    """Compute summary stats for obs axis, and return as a dataframe."""

    if isinstance(raw_X, sparse.csr_matrix) or isinstance(raw_X, sparse.csc_matrix):
        raw_sum = raw_X.sum(axis=1).A1
        nnz = raw_X.getnnz(axis=1)
        raw_mean, raw_variance = sparse_mean_var(raw_X, axis=1, ddof=1)

    elif isinstance(raw_X, np.ndarray):
        raw_sum = raw_X.sum(axis=1)
        nnz = np.count_nonzero(raw_X, axis=1)
        raw_mean = np.mean(raw_X, axis=1)
        raw_variance = np.var(raw_X, axis=1, ddof=1)

    else:
        raise NotImplementedError(f"get_obs_stats: unsupported type {type(raw_X)}")

    n_measured_vars = np.full((raw_X.shape[0],), (raw_X.sum(axis=0) > 0).sum(), dtype=np.int64)

    return pd.DataFrame(
        data={
            "raw_sum": raw_sum.astype(CENSUS_OBS_STATS_COLUMNS["raw_sum"].to_pandas_dtype()),
            "nnz": nnz.astype(CENSUS_OBS_STATS_COLUMNS["nnz"].to_pandas_dtype()),
            "raw_mean": raw_mean.astype(CENSUS_OBS_STATS_COLUMNS["raw_mean"].to_pandas_dtype()),
            "raw_variance": raw_variance.astype(CENSUS_OBS_STATS_COLUMNS["raw_variance"].to_pandas_dtype()),
            "n_measured_vars": n_measured_vars.astype(CENSUS_OBS_STATS_COLUMNS["n_measured_vars"].to_pandas_dtype()),
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

    n_measured_obs = raw_X.shape[0] * (raw_X.sum(axis=0) > 0).A1

    return pd.DataFrame(
        data={
            "nnz": nnz.astype(CENSUS_VAR_STATS_COLUMNS["nnz"].to_pandas_dtype()),
            "n_measured_obs": n_measured_obs.astype(CENSUS_VAR_STATS_COLUMNS["n_measured_obs"].to_pandas_dtype()),
        }
    )


def sparse_mean_var(
    matrix: Union[sparse.csr_matrix, sparse.csc_matrix],
    axis: int = 0,
    ddof: int = 1,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if axis == 0:
        n_elem, axis_len = matrix.shape
    else:
        axis_len, n_elem = matrix.shape

    n_a = np.zeros((axis_len,), dtype=np.int32)
    u_a = np.zeros((axis_len,), dtype=np.float64)
    M2_a = np.zeros((axis_len,), dtype=np.float64)

    # chunk to lower memory consumption. Slow if not on primary axis, but
    # should still work.
    n_elem_stride = 2**17
    for idx in range(0, n_elem, n_elem_stride):
        slc = matrix[idx : idx + n_elem_stride].tocoo()
        coords = slc.col if axis == 0 else slc.row
        _sparse_mean_var_accumulate(slc.data, coords, n_a, u_a, M2_a)

    u, M2 = _sparse_mean_var_finalize(n_elem, n_a, u_a, M2_a)

    var: npt.NDArray[np.float64] = M2 / max(1, (n_elem - ddof))

    return u, var


@numba.jit(nopython=True, nogil=True)  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _sparse_mean_var_accumulate(
    data_arr: npt.NDArray[np.float32],
    col_arr: npt.NDArray[np.int64],
    n: npt.NDArray[np.int32],
    u: npt.NDArray[np.float64],
    M2: npt.NDArray[np.float64],
) -> None:
    """
    Incrementally accumulate mean and sum of square of distance from mean using
    Welford's online method.
    """
    for col, data in zip(col_arr, data_arr):
        u_prev = u[col]
        M2_prev = M2[col]
        n[col] += 1
        u[col] = u_prev + (data - u_prev) / n[col]
        M2[col] = M2_prev + (data - u_prev) * (data - u[col])


@numba.jit(nopython=True, nogil=True)  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _sparse_mean_var_finalize(
    n_rows: int,
    n_a: npt.NDArray[np.int32],
    u_a: npt.NDArray[np.float64],
    M2_a: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Finalize incremental values, acconting for missing elements (due to sparse input).
    Non-sparse and sparse combined using Chan's parallel adaptation of Welford's.
    The code assumes the sparse elements are all zero and ignores those terms.
    """
    n_b = n_rows - n_a
    delta = -u_a  # assumes u_b == 0
    u = (n_a * u_a) / n_rows
    M2 = M2_a + delta**2 * n_a * n_b / n_rows  # assumes M2_b == 0
    return u, M2
