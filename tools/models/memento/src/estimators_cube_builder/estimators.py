import logging
from typing import Tuple, cast

import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from numba import njit
from numpy import random
from scipy.sparse import csc_array

RELIABILITY_THRESHOLD = 0.05


def bin_size_factor(size_factor: npt.NDArray[np.float64], num_bins: int = 30) -> npt.NDArray[np.float64]:
    """Bin the size factors to speed up bootstrap."""

    binned_stat = stats.binned_statistic(size_factor, size_factor, bins=num_bins, statistic="mean")
    bin_idx = np.clip(binned_stat[2], a_min=1, a_max=binned_stat[0].shape[0])
    approx_sf = binned_stat[0][bin_idx - 1]
    max_sf = size_factor.max()
    approx_sf[size_factor == max_sf] = max_sf

    return cast(npt.NDArray[np.float64], approx_sf)


def fill_invalid(val: npt.NDArray[np.float64], group_name: Tuple[str, ...]) -> npt.NDArray[np.float64]:
    """Fill invalid entries by randomly selecting a valid entry."""

    # negatives and nan values are invalid values for our purposes
    invalid_mask = np.less_equal(val, 0.0, where=~np.isnan(val)) | np.isnan(val)
    num_invalid = invalid_mask.sum()

    if num_invalid == val.shape[0]:
        # if all values are invalid, there are no valid values to choose from, so return all nans
        logging.debug(f"all bootstrap variances are invalid for group {group_name}")
        return np.full(shape=val.shape, fill_value=np.nan)

    val[invalid_mask] = np.random.choice(val[~invalid_mask], num_invalid)

    return val


def unique_expr(
    X: csc_array, size_factor: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """
    Find (approximately) unique combinations of expression values and size factors.
    The random component is for mapping (expr, size_factor) to a single number.
    This can certainly be performed more efficiently using sparsity.
    """

    code = X.dot(np.random.random(X.shape[1]))
    approx_sf = size_factor

    code += np.random.random() * approx_sf

    _, index, count = np.unique(code, return_index=True, return_counts=True)

    expr_to_return = X[index].toarray()

    return 1 / approx_sf[index].reshape(-1, 1), expr_to_return, count


@njit  # type: ignore[misc]
def gen_multinomial(counts: npt.NDArray[np.int64], n_obs: int, num_boot: int) -> npt.NDArray[np.int64]:
    # reset numpy random generator
    # TODO: why is this necessary?
    np.random.seed(5)

    return random.multinomial(n_obs, counts / counts.sum(), size=num_boot).T
