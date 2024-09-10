from __future__ import annotations

from concurrent import futures
from typing import Any, Generator

import numpy as np
import numpy.typing as npt
import pandas as pd
import tiledbsoma as soma

from ..util._eager_iter import _EagerIterator
from ._online import MeanAccumulator, MeanVarianceAccumulator


def mean_variance(
    query: soma.ExperimentAxisQuery,
    layer: str = "raw",
    axis: int = 0,
    calculate_mean: bool = False,
    calculate_variance: bool = False,
    ddof: int = 1,
    nnz_only: bool = False,
) -> pd.DataFrame:
    """Calculate  mean and/or variance along the ``obs`` axis from query results. Calculations are done in an accumulative
    chunked fashion. For the mean and variance calculations, the total number of elements (N) is, by default, the
    corresponding dimension size: for column-wise calculations (``axis = 0``) N is number of rows, for row-wise
    calculations (``axis = 1``) N is number of columns. For metrics calculated only on nnz (explicitly stored) values of
    the sparse matrix, specify ``nnz_only=True``.

    Args:
        query:
            A :class:`tiledbsoma.ExperimentAxisQuery`, specifying the ``obs``/``var`` selection over which mean and
            variance are calculated.
        layer:
            X layer used, e.g., ``"raw"``.
        axis:
           Axis or axes along which the statistics are computed.
        calculate_mean:
            If ``True`` it calculates mean, otherwise skips calculation.
        calculate_variance:
            If ``True`` it calculates variance, otherwise skips calculation.
        ddof:
            "Delta Degrees of Freedom": the divisor used in the calculation for variance is ``N - ddof``, where ``N``
            represents the number of elements.
        nnz_only:
            If ``True`` mean and variance will only be calculated over explicitly stored values in the sparse matrix.
            Defaults to ``False``.

    Returns:
        :class:`pandas.DataFrame` indexed by the ``soma_joinid`` and with columns ``mean`` (if
        ``calculate_mean = True``), and ``variance`` (if ``calculate_variance = True``).

    Lifecycle:
        experimental
    """
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    if calculate_mean is False and calculate_variance is False:
        raise ValueError("At least one of `calculate_mean` or `calculate_variance` must be True")

    if query.n_obs == 0 or query.n_vars == 0:
        raise ValueError("The query cannot yield an empty result")

    n_dim_0 = query.n_obs if axis == 1 else query.n_vars
    n_dim_1 = query.n_vars if axis == 1 else query.n_obs

    n_batches = 1
    n_samples = np.array([n_dim_1], dtype=np.int64)

    idx = pd.Index(
        data=query.obs_joinids() if axis == 1 else query.var_joinids(),
        name="soma_joinid",
    )

    def iterate() -> Generator[tuple[npt.NDArray[np.int64], Any], None, None]:
        with futures.ThreadPoolExecutor(max_workers=1) as pool:  # Note: _EagerIterator only supports one thread
            for arrow_tbl in _EagerIterator(query.X(layer).tables(), pool=pool):
                dim = idx.get_indexer(arrow_tbl[f"soma_dim_{1-axis}"].to_numpy())
                data = arrow_tbl["soma_data"].to_numpy()
                yield dim, data

    result = pd.DataFrame(
        index=idx,
    )

    if calculate_variance:
        mvn = MeanVarianceAccumulator(n_batches, n_samples, n_dim_0, ddof=ddof, nnz_only=nnz_only)
        for dim, data in iterate():
            mvn.update(dim, data)
        _, _, all_u, all_var = mvn.finalize()
        if calculate_mean:
            result["mean"] = all_u
        result["variance"] = all_var
    else:
        mn = MeanAccumulator(n_dim_1, n_dim_0, nnz_only=nnz_only)
        for dim, data in iterate():
            mn.update(dim, data)
        all_u = mn.finalize()
        result["mean"] = all_u

    return result
