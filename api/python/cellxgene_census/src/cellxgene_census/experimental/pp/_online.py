from typing import Tuple

import numba
import numpy as np
import numpy.typing as npt


class MeanVarianceAccumulator:
    """
    Online mean/variance for n_variables over n_samples, where the samples are
    divided into n_batches (n_batches << n_samples). Accumulates each batch separately.

    Batches implemented using Chan's parallel adaptation of Welford's online algorithm.

    References:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    and
        Knuth, Art of Computer Programming, volume II
    """

    def __init__(self, n_batches: int, n_samples: npt.NDArray[np.int64], n_variables: int):
        if n_samples.sum() <= 0:
            raise ValueError("No samples provided - can't calculate mean or variance.")

        self.n_batches = n_batches
        self.n_samples = n_samples
        self.n = np.zeros((n_batches, n_variables), dtype=np.int32)
        self.u = np.zeros((n_batches, n_variables), dtype=np.float64)
        self.M2 = np.zeros((n_batches, n_variables), dtype=np.float64)

    def update_by_batch(
        self, batch_vec: npt.NDArray[np.int64], var_vec: npt.NDArray[np.int64], val_vec: npt.NDArray[np.float32]
    ) -> None:
        _mbomv_update_by_batch(batch_vec, var_vec, val_vec, self.n, self.u, self.M2)

    def update_single_batch(self, var_vec: npt.NDArray[np.int64], val_vec: npt.NDArray[np.float32]) -> None:
        assert self.n_batches == 1
        _mbomv_update_single_batch(var_vec, val_vec, self.n, self.u, self.M2)

    def finalize(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # TODO: do we want to add ddof param?  Currently hard-wired to 1 (see var calcs)

        # correct each batch to account for sparsity
        _mbomv_sparse_correct_batches(self.n_batches, self.n_samples, self.n, self.u, self.M2)

        # compute u, var for each batch
        batches_u = self.u
        batches_var = (self.M2.T / (self.n_samples - 1)).T

        # accum all batches using Chan's
        all_u, all_M2 = _mbomv_combine_batches(self.n_batches, self.n_samples, self.u, self.M2)
        all_var = all_M2 / max(1, (self.n_samples.sum() - 1))

        return batches_u, batches_var, all_u, all_var


class CountsAccumulator:
    def __init__(self, n_batches: int, n_variables: int, clip_val: npt.NDArray[np.float64]):
        self.n_batches = n_batches
        self.n_variables = n_variables
        self.clip_val = clip_val
        self.counts_sum = np.zeros((n_batches, n_variables), dtype=np.float64)  # clipped
        self.squared_counts_sum = np.zeros((n_batches, n_variables), dtype=np.float64)  # clipped

    def update_by_batch(
        self, batch_vec: npt.NDArray[np.int64], var_vec: npt.NDArray[np.int64], val_vec: npt.NDArray[np.float32]
    ) -> None:
        _accum_clipped_counts_by_batch(
            self.counts_sum, self.squared_counts_sum, batch_vec, var_vec, val_vec, self.clip_val
        )

    def update_single_batch(self, var_vec: npt.NDArray[np.int64], val_vec: npt.NDArray[np.float32]) -> None:
        assert self.n_batches == 1
        _accum_clipped_counts(self.counts_sum[0], self.squared_counts_sum[0], var_vec, val_vec, self.clip_val[0])

    def finalize(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        return self.counts_sum, self.squared_counts_sum


"""
Private performance related top-level functions.

Down the road, it would be nice to hide via @jitclass, but it is still experimental and
there appear to be performance issues.
"""


@numba.jit(
    [
        numba.void(
            numba.int64[:],
            numba.types.Array(numba.int64, 1, "C", readonly=True),
            numba.types.Array(numba.float32, 1, "C", readonly=True),
            numba.int32[:, :],
            numba.float64[:, :],
            numba.float64[:, :],
        ),
        numba.void(
            numba.int64[:],
            numba.types.Array(numba.int32, 1, "C", readonly=True),
            numba.types.Array(numba.float32, 1, "C", readonly=True),
            numba.int32[:, :],
            numba.float64[:, :],
            numba.float64[:, :],
        ),
    ],
    nopython=True,
    nogil=True,
)  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _mbomv_update_by_batch(
    batch_vec: npt.NDArray[np.int64],
    var_vec: npt.NDArray[np.int64],
    val_vec: npt.NDArray[np.float32],
    n: npt.NDArray[np.int32],
    u: npt.NDArray[np.float64],
    M2: npt.NDArray[np.float64],
) -> None:
    """
    Incrementally accumulate mean and sum of square of distance from mean using
    Welford's online method.
    """
    for batch, col, val in zip(batch_vec, var_vec, val_vec):
        u_prev = u[batch, col]
        M2_prev = M2[batch, col]
        n[batch, col] += 1
        u[batch, col] = u_prev + (val - u_prev) / n[batch, col]
        M2[batch, col] = M2_prev + (val - u_prev) * (val - u[batch, col])


@numba.jit(
    numba.void(
        numba.types.Array(numba.int64, 1, "C", readonly=True),
        numba.types.Array(numba.float32, 1, "C", readonly=True),
        numba.int32[:, :],
        numba.float64[:, :],
        numba.float64[:, :],
    ),
    nopython=True,
    nogil=True,
)  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _mbomv_update_single_batch(
    var_vec: npt.NDArray[np.int64],
    val_vec: npt.NDArray[np.float32],
    n: npt.NDArray[np.int32],
    u: npt.NDArray[np.float64],
    M2: npt.NDArray[np.float64],
) -> None:
    """
    Incrementally accumulate mean and sum of square of distance from mean using
    Welford's online method.
    """
    for col, val in zip(var_vec, val_vec):
        u_prev = u[0, col]
        M2_prev = M2[0, col]
        n[0, col] += 1
        u[0, col] = u_prev + (val - u_prev) / n[0, col]
        M2[0, col] = M2_prev + (val - u_prev) * (val - u[0, col])


@numba.jit(
    numba.void(numba.int64, numba.int64[:], numba.int32[:, :], numba.float64[:, :], numba.float64[:, :]),
    nopython=True,
    nogil=True,
)  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _mbomv_sparse_correct_batches(
    n_batches: int,
    n_samples: npt.NDArray[np.int64],
    n: npt.NDArray[np.int32],
    u: npt.NDArray[np.float64],
    M2: npt.NDArray[np.float64],
) -> None:
    """
    Finalize incremental accumulators to account for missing elements (due to sparse
    input). Non-sparse and sparse combined using Chan's parallel adaptation of Welford's.
    The code assumes the sparse elements are all zero.
    """
    for batch in range(n_batches):
        n_b = n_samples[batch] - n[batch]
        delta = -u[batch]  # assumes u_b == 0
        _u = (n[batch] * u[batch]) / n_samples[batch]
        _M2 = M2[batch] + delta**2 * n[batch] * n_b / n_samples[batch]  # assumes M2_b == 0
        u[batch] = _u
        M2[batch] = _M2
        n[batch] = n_samples[batch]


@numba.jit(
    numba.types.Tuple((numba.float64[:], numba.float64[:]))(
        numba.int64, numba.int64[:], numba.float64[:, :], numba.float64[:, :]
    ),
    nopython=True,
    nogil=True,
)  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _mbomv_combine_batches(
    n_batches: int,
    n_samples: npt.NDArray[np.int64],
    u: npt.NDArray[np.float64],
    M2: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Combine all batches using Chan's parallel adaptation of Welford's.

    Returns tuple of (u, M2).
    """
    # initialize with first batch that contains samples
    for first_batch in range(0, n_batches):
        if n_samples[first_batch] > 0:
            acc_n = n_samples[first_batch]
            acc_u = u[first_batch].copy()
            acc_M2 = M2[first_batch].copy()
            break

    # TODO: does not handle case where there is no data, i.e., n_samples.sum() == 0

    for batch in range(first_batch + 1, n_batches):
        # ignore batches with no data
        if n_samples[batch] == 0:
            continue

        n = acc_n + n_samples[batch]
        delta = u[batch] - acc_u
        _u = (acc_n * acc_u + n_samples[batch] * u[batch]) / n
        _M2 = acc_M2 + M2[batch] + delta**2 * acc_n * n_samples[batch] / n
        # TODO: reduce memory allocs?
        acc_n = n
        acc_u = _u
        acc_M2 = _M2

    return acc_u, acc_M2


@numba.jit(
    numba.void(
        numba.float64[:],
        numba.float64[:],
        numba.types.Array(numba.int64, 1, "C", readonly=True),
        numba.types.Array(numba.float32, 1, "C", readonly=True),
        numba.float64[:],
    ),
    nopython=True,
    nogil=True,
)  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _accum_clipped_counts(
    counts_sum: npt.NDArray[np.float64],
    squared_counts_sum: npt.NDArray[np.float64],
    var_dim: npt.NDArray[np.int64],
    data: npt.NDArray[np.float64],
    clip_val: npt.NDArray[np.float64],
) -> None:
    for col, val in zip(var_dim, data):
        if val > clip_val[col]:
            val = clip_val[col]
        counts_sum[col] += val
        squared_counts_sum[col] += val**2


@numba.jit(
    numba.void(
        numba.float64[:, :],
        numba.float64[:, :],
        numba.int64[:],
        numba.types.Array(numba.int64, 1, "C", readonly=True),
        numba.types.Array(numba.float32, 1, "C", readonly=True),
        numba.float64[:, :],
    ),
    nopython=True,
    nogil=True,
)  # type: ignore[misc]  # See https://github.com/numba/numba/issues/7424
def _accum_clipped_counts_by_batch(
    counts_sum: npt.NDArray[np.float64],
    squared_counts_sum: npt.NDArray[np.float64],
    batch: npt.NDArray[np.int64],
    var_dim: npt.NDArray[np.int64],
    data: npt.NDArray[np.float32],
    clip_val: npt.NDArray[np.float64],
) -> None:
    for bid, col, val in zip(batch, var_dim, data):
        if val > clip_val[bid, col]:
            val = clip_val[bid, col]
        counts_sum[bid, col] += val
        squared_counts_sum[bid, col] += val**2
