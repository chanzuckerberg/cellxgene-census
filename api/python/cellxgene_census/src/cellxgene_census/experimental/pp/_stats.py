from __future__ import annotations

import os
from concurrent import futures
from typing import Any, Generator, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import tiledbsoma as soma

from ..util._eager_iter import _EagerIterator
from ._online import MeanAccumulator, MeanVarianceAccumulator


def mean_variance(
    query: soma.ExperimentAxisQuery,
    layer: str = "raw",
    calculate_mean: bool = False,
    calculate_variance: bool = False,
) -> pd.DataFrame:
    """
    Calculate  mean and/or variance along the `obs` axis from query results. Calculations
    are done in an accumulative chunked fashion.

    Args:
        query:
            A SOMA query, specifying the obs/var selection over which mean and variance are calculated.

        layer:
            X layer used, e.g., `raw`

        calculate_mean:
            If `True it calculates mean, otherwise skips calculation

        calculate_variance:
            If `True it calculates variance, otherwise skips calculation

    Returns:
        Pandas DataFrame indexed by the obs `soma_joinid` and with columns
       `mean` (if `calculate_mean = True`), and `variance` (if `calculate_variance = True`)

    Lifecycle:
        experimental
    """

    if calculate_mean is False and calculate_variance is False:
        raise ValueError("At least one of `calculate_mean` or `calculate_variance` must be True")

    n_batches = 1
    n_samples = np.array([query.n_vars], dtype=np.int64)

    def iterate() -> Generator[Tuple[npt.NDArray[np.int64], Any], None, None]:
        max_workers = (os.cpu_count() or 4) + 2
        with futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for arrow_tbl in _EagerIterator(query.X(layer).tables(), pool=pool):
                obs_indexer = query.indexer
                obs_dim = obs_indexer.by_obs(arrow_tbl["soma_dim_0"])
                data = arrow_tbl["soma_data"].to_numpy()
                yield obs_dim, data

    result = pd.DataFrame(
        index=pd.Index(data=query.obs_joinids(), name="soma_joinid"),
    )

    if calculate_variance:
        mvn = MeanVarianceAccumulator(n_batches, n_samples, query.n_obs)
        for obs_dim, data in iterate():
            mvn.update_single_batch(obs_dim, data)
            _, _, all_u, all_var = mvn.finalize()
            if calculate_mean:
                result["mean"] = all_u
            result["variance"] = all_var
    else:
        mn = MeanAccumulator(n_batches, n_samples, query.n_obs)
        for obs_dim, data in iterate():
            mn.update_single_batch(obs_dim, data)
            _, all_u = mn.finalize()
            result["mean"] = all_u

    return result
