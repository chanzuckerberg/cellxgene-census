from typing import Tuple

import numba
import numpy as np
import numpy.typing as npt


class OnlineMatrixMeanVariance:
    n_samples: int
    n_variables: int

    def __init__(self, n_samples: int, n_variables: int):
        """
        Compute mean and variance for n_variables over n_samples, encoded
        in a COO format. Equivalent to:
            numpy.mean(data, axis=0)
            numpy.var(data, axix=0)
        where the input `data` is of shape (n_samples, n_variables)
        """
        self.n_samples = n_samples
        self.n_variables = n_variables

        self.n_a = np.zeros((n_variables,), dtype=np.int32)
        self.u_a = np.zeros((n_variables,), dtype=np.float64)
        self.M2_a = np.zeros((n_variables,), dtype=np.float64)

    def update(self, coord_vec: npt.NDArray[np.int64], value_vec: npt.NDArray[np.float32]) -> None:
        _mean_variance_update(coord_vec, value_vec, self.n_a, self.u_a, self.M2_a)

    def finalize(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Returns tuple containing mean and variance
        """
        u, M2 = _mean_variance_finalize(self.n_samples, self.n_a, self.u_a, self.M2_a)

        # compute sample variance
        var = M2 / max(1, (self.n_samples - 1))

        return u, var


# TODO: add type signatures to annotation, removing need to do dynamic generation


@numba.jit(nopython=True, nogil=True)  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _mean_variance_update(
    col_arr: npt.NDArray[np.int64],
    val_arr: npt.NDArray[np.float32],
    n: npt.NDArray[np.int32],
    u: npt.NDArray[np.float64],
    M2: npt.NDArray[np.float64],
) -> None:
    """
    Incrementally accumulate mean and sum of square of distance from mean using
    Welford's online method.
    """
    for col, val in zip(col_arr, val_arr):
        u_prev = u[col]
        M2_prev = M2[col]
        n[col] += 1
        u[col] = u_prev + (val - u_prev) / n[col]
        M2[col] = M2_prev + (val - u_prev) * (val - u[col])


@numba.jit(nopython=True, nogil=True)  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _mean_variance_finalize(
    n_samples: int, n_a: npt.NDArray[np.int32], u_a: npt.NDArray[np.float64], M2_a: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Finalize incremental values, acconting for missing elements (due to sparse input).
    Non-sparse and sparse combined using Chan's parallel adaptation of Welford's.
    The code assumes the sparse elements are all zero and ignores those terms.
    """
    n_b = n_samples - n_a
    delta = -u_a  # assumes u_b == 0
    u = (n_a * u_a) / n_samples
    M2 = M2_a + delta**2 * n_a * n_b / n_samples  # assumes M2_b == 0
    return u, M2
